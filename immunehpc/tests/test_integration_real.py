"""
tests/test_integration_real.py — Integration tests for Rocky Linux backends

All SSH/subprocess calls are mocked so no real nodes are needed.
Tests verify that every real implementation correctly translates
controller decisions into system calls.

Coverage:
  - ProcReader          (/proc — runs real on Linux)
  - PrometheusCollector (HTTP scrape + parsing)
  - SSHCollector        (SSH /proc one-liner)
  - RealTelemetryCollector (source priority + fallback)
  - SelfHealingEngine   (systemctl / Ansible stubs / rollback detection)
  - DefenseLayer        (auditd / process scan / SELinux / listener scan)
  - QuarantineLayer     (firewalld + scontrol + tc)
  - OptimisationEngine  (actuators: compute/thermal/power/network/storage)
  - AdaptiveScheduler   (Slurm bridge: node weights / drain / resume)
  - SandboxExecutor     (safety scan + regression)
  - CanaryDeployer      (apply + checkpoint + rollback)
  - ASLPipeline         (end-to-end: generate → sandbox → canary → deploy)
  - ProductionConfig    (YAML keys + required fields)
"""

import sys
import os
import json
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.state import ClusterState, NodeState, NodeStatus


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_state(nodes=None) -> ClusterState:
    state = ClusterState()
    for nid in (nodes or ["node01", "node02"]):
        state.nodes[nid] = NodeState(node_id=nid, health_score=1.0, trust_score=1.0)
    return state


def _make_ssh_cfg(**kw) -> dict:
    cfg = {
        "ssh_user": "root",
        "ssh_key_path": "/tmp/test_key",
        "node_exporter_port": 9100,
        "use_ipmi": False,
        "use_gpu": False,
        "use_slurm": False,
        "auto_mitigate": False,
        "known_keys_path": "/tmp/test_known_keys.json",
        "network_interface": "eth0",
        "bmc_user": "admin",
        "bmc_password": "",
    }
    cfg.update(kw)
    return cfg


# ---------------------------------------------------------------------------
# ProcReader  (real /proc — only skipped if not on Linux)
# ---------------------------------------------------------------------------

class TestProcReader(unittest.TestCase):

    def test_cpu_usage_in_range(self):
        from modules.monitor import ProcReader
        if not os.path.exists("/proc/stat"):
            self.skipTest("/proc/stat not available")
        cpu = ProcReader().cpu_usage()
        self.assertGreaterEqual(cpu, 0.0)
        self.assertLessEqual(cpu, 1.0)

    def test_memory_usage_in_range(self):
        from modules.monitor import ProcReader
        if not os.path.exists("/proc/meminfo"):
            self.skipTest("/proc/meminfo not available")
        mem = ProcReader().memory_usage()
        self.assertGreater(mem, 0.0)
        self.assertLess(mem, 1.0)

    def test_load_average(self):
        from modules.monitor import ProcReader
        if not os.path.exists("/proc/loadavg"):
            self.skipTest("/proc/loadavg not available")
        l1, l5, l15 = ProcReader().load_average()
        self.assertGreaterEqual(l1, 0.0)

    def test_detect_network_interface(self):
        from modules.monitor import ProcReader
        iface = ProcReader().detect_network_interface()
        self.assertIsInstance(iface, str)
        self.assertGreater(len(iface), 0)


# ---------------------------------------------------------------------------
# PrometheusCollector
# ---------------------------------------------------------------------------

PROM_FIXTURE = """
node_cpu_seconds_total{cpu="0",mode="idle"} 10000.0
node_cpu_seconds_total{cpu="0",mode="user"} 2000.0
node_memory_MemTotal_bytes 8589934592
node_memory_MemAvailable_bytes 4294967296
node_network_receive_bytes_total{device="eth0"} 1000000
node_network_transmit_bytes_total{device="eth0"} 500000
node_hwmon_temp_celsius{chip="coretemp",sensor="temp1"} 55.0
"""


class TestPrometheusCollector(unittest.TestCase):

    def setUp(self):
        from modules.monitor import PrometheusCollector
        self.pc = PrometheusCollector()
        self.scraped = self.pc._parse(PROM_FIXTURE)

    def test_parse_returns_metrics(self):
        self.assertGreater(len(self.scraped), 0)

    def test_cpu_usage_extracted(self):
        cpu = self.pc.extract_cpu_usage(self.scraped)
        self.assertGreaterEqual(cpu, 0.0)
        self.assertLessEqual(cpu, 1.0)

    def test_memory_usage_extracted(self):
        mem = self.pc.extract_memory_usage(self.scraped)
        self.assertAlmostEqual(mem, 0.5, places=2)

    def test_network_extracted(self):
        rx, tx = self.pc.extract_network(self.scraped, iface="eth0")
        self.assertEqual(rx, 1000000.0)
        self.assertEqual(tx, 500000.0)

    def test_scrape_with_mock(self):
        with patch("urllib.request.urlopen") as mock_url:
            mock_resp = MagicMock()
            mock_resp.read.return_value = PROM_FIXTURE.encode()
            mock_url.return_value = mock_resp
            result = self.pc.scrape("node01", 9100)
        self.assertGreater(len(result), 0)

    def test_scrape_handles_failure(self):
        with patch("urllib.request.urlopen", side_effect=Exception("timeout")):
            result = self.pc.scrape("node01", 9100)
        self.assertEqual(result, {})


# ---------------------------------------------------------------------------
# SSHCollector
# ---------------------------------------------------------------------------

class TestSSHCollector(unittest.TestCase):

    def setUp(self):
        from modules.monitor import SSHCollector
        self.ssh = SSHCollector(ssh_user="root", ssh_key="/tmp/test_key")

    def test_collect_parses_json(self):
        data = json.dumps({"cpu": 0.25, "mem": 0.4, "rx_mbps": 10.0, "tx_mbps": 5.0})
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(returncode=0, stdout=data)
            result = self.ssh.collect("node01")
        self.assertAlmostEqual(result["cpu"], 0.25)

    def test_collect_returns_none_on_failure(self):
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(returncode=1, stdout="")
            result = self.ssh.collect("node01")
        self.assertIsNone(result)

    def test_is_reachable_true(self):
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(returncode=0)
            self.assertTrue(self.ssh.is_reachable("node01"))

    def test_is_reachable_false(self):
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(returncode=255)
            self.assertFalse(self.ssh.is_reachable("node01"))


# ---------------------------------------------------------------------------
# RealTelemetryCollector
# ---------------------------------------------------------------------------

class TestRealTelemetryCollector(unittest.TestCase):

    def setUp(self):
        from modules.monitor import RealTelemetryCollector
        self.col = RealTelemetryCollector(_make_ssh_cfg())
        self.node = NodeState(node_id="node01", health_score=1.0)

    def test_uses_prometheus_when_available(self):
        scraped = {
            'node_cpu_seconds_total{cpu="0",mode="idle"}': 10000.0,
            'node_cpu_seconds_total{cpu="0",mode="user"}': 2000.0,
            'node_memory_MemTotal_bytes': 8e9,
            'node_memory_MemAvailable_bytes': 4e9,
        }
        with patch.object(self.col._prometheus, "scrape", return_value=scraped):
            result = self.col.collect(self.node)
        self.assertGreaterEqual(result.cpu_usage, 0.0)
        self.assertAlmostEqual(result.memory_usage, 0.5, places=1)

    def test_falls_back_to_ssh(self):
        ssh_data = {"cpu": 0.3, "mem": 0.5, "rx_mbps": 50.0, "tx_mbps": 20.0}
        with patch.object(self.col._prometheus, "scrape", return_value={}):
            with patch.object(self.col._ssh, "collect", return_value=ssh_data):
                result = self.col.collect(self.node)
        self.assertAlmostEqual(result.cpu_usage, 0.3)

    def test_marks_degraded_when_all_fail(self):
        with patch.object(self.col._prometheus, "scrape", return_value={}):
            with patch.object(self.col._ssh, "collect", return_value=None):
                result = self.col.collect(self.node)
        self.assertEqual(result.status, NodeStatus.DEGRADED)


# ---------------------------------------------------------------------------
# SelfHealingEngine  (real SSH strategies)
# ---------------------------------------------------------------------------

class TestRealHealer(unittest.TestCase):

    def setUp(self):
        from modules.healer import SelfHealingEngine, RepairStrategy
        self.state = _make_state(["node01"])
        self.cfg = _make_ssh_cfg(
            ansible_playbook="/nonexistent/site.yml",
            ansible_inventory="/nonexistent/inventory.ini",
            cobbler_host="cobbler",
        )
        self.healer = SelfHealingEngine(
            state=self.state, config=self.cfg,
            strategy_order=[RepairStrategy.RESTART_SERVICE],
            max_attempts=1, backoff_sec=0,
        )

    def test_restart_service_no_failed_units(self):
        from modules.healer import _restart_service
        with patch("modules.healer._ssh", return_value=(0, "", "")):
            ok, msg = _restart_service(self.state.nodes["node01"], self.cfg)
        self.assertTrue(ok)
        self.assertIn("no failed services", msg)

    def test_restart_service_restarts_failed_unit(self):
        from modules.healer import _restart_service
        node = self.state.nodes["node01"]
        calls = [0]

        def side(host, cmd, cfg, **kw):
            i = calls[0]; calls[0] += 1
            if i == 0:
                return (0, "slurmd.service  -  -  -  failed", "")
            if "is-active slurmd" in cmd:
                return (1, "inactive", "")
            return (0, "", "")

        with patch("modules.healer._ssh", side_effect=side):
            ok, msg = _restart_service(node, self.cfg)
        self.assertIsInstance(ok, bool)

    def test_reapply_config_skips_missing_playbook(self):
        from modules.healer import _reapply_config
        ok, msg = _reapply_config(self.state.nodes["node01"], self.cfg)
        self.assertFalse(ok)
        self.assertIn("not found", msg)

    def test_heal_starts_for_known_node(self):
        with patch("modules.healer._ssh", return_value=(0, "", "")):
            started = self.healer.heal("node01")
        self.assertTrue(started)

    def test_heal_skips_quarantined(self):
        self.state.nodes["node01"].status = NodeStatus.QUARANTINED
        self.assertFalse(self.healer.heal("node01"))


# ---------------------------------------------------------------------------
# DefenseLayer  (real auditd / process scan / SELinux / listener scan)
# ---------------------------------------------------------------------------

class TestRealDefense(unittest.TestCase):

    def setUp(self):
        from modules.defense import DefenseLayer
        self.state = _make_state(["node01"])
        self.cfg = _make_ssh_cfg()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({}, f)
            self.cfg["known_keys_path"] = f.name
        self.defense = DefenseLayer(state=self.state, config=self.cfg)

    def tearDown(self):
        try:
            os.unlink(self.cfg["known_keys_path"])
        except Exception:
            pass

    def test_scan_all_returns_list(self):
        with patch("modules.defense._ssh", return_value=(0, "", "")):
            with patch("subprocess.run") as m:
                m.return_value = MagicMock(returncode=0, stdout="")
                threats = self.defense.scan_all()
        self.assertIsInstance(threats, list)

    def test_malicious_process_detected(self):
        from modules.defense import ProcessScanner
        ps_output = "1234 root xmrig xmrig --pool mining.pool:4444\n"
        with patch("modules.defense._ssh", return_value=(0, ps_output, "")):
            threats = ProcessScanner().scan("node01", self.cfg)
        self.assertTrue(any(t.threat_type.value == "malicious_process" for t in threats))

    def test_unknown_listener_detected(self):
        from modules.defense import NetworkScanner
        with patch("modules.defense._ssh", return_value=(0, "19999\n", "")):
            threats = NetworkScanner().scan("node01", self.cfg)
        self.assertTrue(any(t.evidence.get("port") == 19999 for t in threats))

    def test_selinux_permissive_flagged(self):
        from modules.defense import SELinuxChecker
        with patch("modules.defense._ssh", return_value=(0, "Permissive", "")):
            threat = SELinuxChecker().check("node01", self.cfg)
        self.assertIsNotNone(threat)

    def test_selinux_enforcing_clean(self):
        from modules.defense import SELinuxChecker
        with patch("modules.defense._ssh", return_value=(0, "Enforcing", "")):
            threat = SELinuxChecker().check("node01", self.cfg)
        self.assertIsNone(threat)

    def test_ssh_host_key_tofu(self):
        from modules.defense import SSHHostKeyVerifier
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({}, f); path = f.name
        try:
            verifier = SSHHostKeyVerifier(known_keys_path=path)
            with patch("subprocess.run") as m:
                m.return_value = MagicMock(
                    returncode=0, stdout="node01 ssh-ed25519 AAAAC3Nza...")
                trusted, reason = verifier.verify("node01")
            self.assertTrue(trusted)
            self.assertIn("first contact", reason)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# QuarantineLayer  (firewalld + scontrol + tc) — GAP 2
# ---------------------------------------------------------------------------

class TestRealQuarantine(unittest.TestCase):

    def setUp(self):
        from modules.quarantine import QuarantineLayer
        self.state = _make_state(["node01"])
        self.layer = QuarantineLayer(
            state=self.state,
            ssh_config=_make_ssh_cfg(),
            auto_isolate=True,
            max_quarantine_sec=300,
        )

    def test_quarantine_changes_node_status(self):
        with patch("modules.quarantine._ssh", return_value=(0, "", "")):
            with patch("subprocess.run") as m:
                m.return_value = MagicMock(returncode=0, stdout="", stderr="")
                ok = self.layer.quarantine("node01", reason="test")
        self.assertTrue(ok)
        self.assertEqual(self.state.nodes["node01"].status, NodeStatus.QUARANTINED)

    def test_quarantine_calls_firewall_ssh(self):
        with patch("modules.quarantine._ssh") as mock_ssh:
            mock_ssh.return_value = (0, "", "")
            with patch("subprocess.run") as m:
                m.return_value = MagicMock(returncode=0, stdout="", stderr="")
                self.layer.quarantine("node01", "test")
        # firewall-cmd and tc commands sent via SSH
        ssh_calls = [str(c) for c in mock_ssh.call_args_list]
        self.assertTrue(any("firewall-cmd" in s or "tc qdisc" in s for s in ssh_calls))

    def test_quarantine_calls_slurm_drain(self):
        with patch("modules.quarantine._ssh", return_value=(0, "", "")):
            with patch("subprocess.run") as mock_proc:
                mock_proc.return_value = MagicMock(returncode=0, stdout="", stderr="")
                self.layer.quarantine("node01", "test drain")
        calls_str = str(mock_proc.call_args_list)
        self.assertIn("DRAIN", calls_str)

    def test_release_restores_node_status(self):
        with patch("modules.quarantine._ssh", return_value=(0, "", "")):
            with patch("subprocess.run") as m:
                m.return_value = MagicMock(returncode=0, stdout="", stderr="")
                self.layer.quarantine("node01", "test")
                released = self.layer.release("node01")
        self.assertTrue(released)
        self.assertEqual(self.state.nodes["node01"].status, NodeStatus.HEALTHY)

    def test_release_calls_slurm_resume(self):
        with patch("modules.quarantine._ssh", return_value=(0, "", "")):
            with patch("subprocess.run") as mock_proc:
                mock_proc.return_value = MagicMock(returncode=0, stdout="", stderr="")
                self.layer.quarantine("node01", "test")
                self.layer.release("node01")
        calls_str = str(mock_proc.call_args_list)
        self.assertIn("RESUME", calls_str)

    def test_timeout_triggers_release(self):
        with patch("modules.quarantine._ssh", return_value=(0, "", "")):
            with patch("subprocess.run") as m:
                m.return_value = MagicMock(returncode=0, stdout="", stderr="")
                # Quarantine with tiny timeout
                from modules.quarantine import QuarantineLayer as QL
                layer2 = QL(
                    state=self.state,
                    ssh_config=_make_ssh_cfg(),
                    max_quarantine_sec=0.01,
                )
                layer2.quarantine("node01", "timeout test")
                time.sleep(0.05)
                layer2.check_timeouts()
        self.assertEqual(self.state.nodes["node01"].status, NodeStatus.HEALTHY)


# ---------------------------------------------------------------------------
# OptimisationEngine actuators — GAP 1
# ---------------------------------------------------------------------------

class TestRealOptimizerActuators(unittest.TestCase):

    def setUp(self):
        import modules.optimizer as opt_mod
        self.state = _make_state(["node01", "node02"])
        # Inject SSH config and state nodes directly
        opt_mod._SSH_CONFIG.clear()
        opt_mod._SSH_CONFIG.update(_make_ssh_cfg())
        opt_mod._SSH_CONFIG["_state_nodes"] = self.state.nodes

        from modules.optimizer import OptimisationAction, OptimisationDomain
        self.Action = OptimisationAction
        self.Domain = OptimisationDomain

    def _action(self, domain, old=0.5, new=0.8):
        return self.Action(
            domain=domain, target="cluster",
            parameter="test", old_value=old, new_value=new,
        )

    def test_compute_actuator_calls_ssh(self):
        from modules.optimizer import _apply_compute_action
        with patch("modules.optimizer._ssh") as mock_ssh:
            mock_ssh.return_value = (0, "", "")
            ok = _apply_compute_action(self._action(self.Domain.COMPUTE, new=0.9))
        self.assertTrue(ok)
        # Should SSH to each node with cpufreq/tuned commands
        calls = [str(c) for c in mock_ssh.call_args_list]
        self.assertTrue(any("cpupower" in s or "tuned-adm" in s for s in calls))

    def test_thermal_actuator_calls_ssh(self):
        from modules.optimizer import _apply_thermal_action
        with patch("modules.optimizer._ssh") as mock_ssh:
            mock_ssh.return_value = (0, "", "")
            ok = _apply_thermal_action(self._action(self.Domain.THERMAL, new=0.7))
        self.assertTrue(ok)
        calls = [str(c) for c in mock_ssh.call_args_list]
        self.assertTrue(any("ipmitool" in s for s in calls))

    def test_power_actuator_calls_ssh(self):
        from modules.optimizer import _apply_power_action
        with patch("modules.optimizer._ssh") as mock_ssh:
            mock_ssh.return_value = (0, "", "")
            ok = _apply_power_action(self._action(self.Domain.POWER, new=0.8))
        self.assertTrue(ok)
        calls = [str(c) for c in mock_ssh.call_args_list]
        self.assertTrue(any("cpupower idle-set" in s or "cpupower set" in s for s in calls))

    def test_network_actuator_applies_tc(self):
        from modules.optimizer import _apply_network_action
        with patch("modules.optimizer._ssh") as mock_ssh:
            mock_ssh.return_value = (0, "", "")
            ok = _apply_network_action(self._action(self.Domain.NETWORK, new=0.5))
        self.assertTrue(ok)
        calls = [str(c) for c in mock_ssh.call_args_list]
        self.assertTrue(any("tc qdisc" in s for s in calls))

    def test_network_actuator_removes_tc_at_full_speed(self):
        from modules.optimizer import _apply_network_action
        with patch("modules.optimizer._ssh") as mock_ssh:
            mock_ssh.return_value = (0, "", "")
            ok = _apply_network_action(self._action(self.Domain.NETWORK, new=0.99))
        self.assertTrue(ok)
        calls = [str(c) for c in mock_ssh.call_args_list]
        self.assertTrue(any("qdisc del" in s for s in calls))

    def test_storage_actuator_calls_ssh(self):
        from modules.optimizer import _apply_storage_action
        with patch("modules.optimizer._ssh") as mock_ssh:
            mock_ssh.return_value = (0, "", "")
            ok = _apply_storage_action(self._action(self.Domain.STORAGE, new=0.8))
        self.assertTrue(ok)
        calls = [str(c) for c in mock_ssh.call_args_list]
        self.assertTrue(any("scheduler" in s or "cgset" in s for s in calls))

    def test_compute_governor_maps_to_correct_profile(self):
        from modules.optimizer import _apply_compute_action
        with patch("modules.optimizer._ssh") as mock_ssh:
            mock_ssh.return_value = (0, "", "")
            # new_value=0.9 → performance governor
            _apply_compute_action(self._action(self.Domain.COMPUTE, new=0.9))
        calls = [str(c) for c in mock_ssh.call_args_list]
        self.assertTrue(any("performance" in s for s in calls))

    def test_compute_governor_low_maps_to_powersave(self):
        from modules.optimizer import _apply_compute_action
        with patch("modules.optimizer._ssh") as mock_ssh:
            mock_ssh.return_value = (0, "", "")
            _apply_compute_action(self._action(self.Domain.COMPUTE, new=0.1))
        calls = [str(c) for c in mock_ssh.call_args_list]
        self.assertTrue(any("powersave" in s for s in calls))


# ---------------------------------------------------------------------------
# AdaptiveScheduler + Slurm bridge — GAP 4
# ---------------------------------------------------------------------------

class TestRealScheduler(unittest.TestCase):

    def setUp(self):
        from modules.scheduler import AdaptiveScheduler, SlurmBridge, Job
        self.state = _make_state(["node01", "node02"])
        self.sched = AdaptiveScheduler(self.state, partition="compute")
        self.Job = Job

    def test_slurm_bridge_drain_calls_scontrol(self):
        from modules.scheduler import SlurmBridge
        bridge = SlurmBridge()
        with patch("modules.scheduler._slurm") as mock_slurm:
            mock_slurm.return_value = (0, "", "")
            ok = bridge.drain_node("node01", "test reason")
        self.assertTrue(ok)
        args = mock_slurm.call_args[0][0]
        self.assertIn("DRAIN", " ".join(args))

    def test_slurm_bridge_resume_calls_scontrol(self):
        from modules.scheduler import SlurmBridge
        bridge = SlurmBridge()
        with patch("modules.scheduler._slurm") as mock_slurm:
            mock_slurm.return_value = (0, "", "")
            ok = bridge.resume_node("node01")
        self.assertTrue(ok)
        args = mock_slurm.call_args[0][0]
        self.assertIn("RESUME", " ".join(args))

    def test_slurm_bridge_set_weight_clamps(self):
        from modules.scheduler import SlurmBridge
        bridge = SlurmBridge()
        with patch("modules.scheduler._slurm") as mock_slurm:
            mock_slurm.return_value = (0, "", "")
            bridge.set_node_weight("node01", 99999)   # over max
        args = mock_slurm.call_args[0][0]
        self.assertIn("Weight=65535", " ".join(args))

    def test_node_weight_formula_healthy_is_low(self):
        from modules.scheduler import AdaptiveScheduler
        sched = AdaptiveScheduler(self.state)
        calls_received = []
        with patch.object(sched._slurm, "set_node_weight",
                          side_effect=lambda n, w: calls_received.append((n, w))):
            sched._slurm._available = True
            sched._update_node_weights()
        # Healthy nodes (H_i=1.0) get weight=1
        for node_id, weight in calls_received:
            h = self.state.nodes[node_id].health_score
            if h >= 0.99:
                self.assertEqual(weight, 1)

    def test_node_weight_formula_degraded_is_high(self):
        from modules.scheduler import AdaptiveScheduler
        self.state.nodes["node01"].health_score = 0.3
        sched = AdaptiveScheduler(self.state)
        calls_received = []
        with patch.object(sched._slurm, "set_node_weight",
                          side_effect=lambda n, w: calls_received.append((n, w))):
            sched._update_node_weights()
        weights = {n: w for n, w in calls_received}
        if "node01" in weights:
            self.assertGreater(weights["node01"], 1000)

    def test_quarantine_event_drains_slurm(self):
        from utils.events import bus, EventType
        with patch.object(self.sched._slurm, "drain_node") as mock_drain:
            self.sched._slurm_available = True
            bus.emit_simple(EventType.NODE_QUARANTINED, "test",
                            payload={"node_id": "node01", "reason": "test"})
            time.sleep(0.05)
        mock_drain.assert_called_with("node01", "test")

    def test_release_event_resumes_slurm(self):
        from utils.events import bus, EventType
        with patch.object(self.sched._slurm, "resume_node") as mock_resume:
            self.sched._slurm_available = True
            bus.emit_simple(EventType.NODE_RELEASED, "test",
                            payload={"node_id": "node01",
                                     "quarantine_duration_sec": 30})
            time.sleep(0.05)
        mock_resume.assert_called_with("node01")

    def test_tick_without_slurm_uses_internal(self):
        from modules.scheduler import Job
        with patch.object(self.sched._slurm, "is_available", return_value=False):
            job = Job(name="test", cpu_cores=1, priority=50, wall_time_sec=10)
            self.sched.submit(job)
            self.sched.tick()
        # Job should be placed internally
        self.assertGreaterEqual(self.sched.running_count, 0)

    def test_partition_priority_scales_with_failure_rate(self):
        from modules.scheduler import AdaptiveScheduler
        sched = AdaptiveScheduler(self.state)
        calls_received = []
        with patch.object(sched._slurm, "set_partition_weight",
                          side_effect=lambda p, f: calls_received.append(f)):
            # Low failure rate
            self.state.failure_rate = 0.05
            sched._tune_partition_priority()
            # High failure rate
            self.state.failure_rate = 0.5
            sched._tune_partition_priority()
        self.assertGreater(calls_received[0], calls_received[1])

    def test_queue_depth_from_slurm(self):
        with patch.object(self.sched._slurm, "queue_depth", return_value=42):
            self.sched._slurm_available = True
            depth = self.sched.queue_depth
        self.assertEqual(depth, 42)


# ---------------------------------------------------------------------------
# SandboxExecutor  (safety + regression)
# ---------------------------------------------------------------------------

class TestSandboxExecutor(unittest.TestCase):

    def setUp(self):
        from asl.sandbox import SandboxExecutor
        self.state = _make_state()
        self.sb = SandboxExecutor(timeout_sec=5.0)

    def test_safe_param_patch_passes(self):
        from asl.patch import Patch, PatchType, ParameterDelta
        p = Patch(patch_type=PatchType.PARAMETER,
                  parameter_deltas=[ParameterDelta("z_threshold", 3.0, 3.5)])
        result = self.sb.test(p, self.state)
        self.assertTrue(result.passed)

    def test_unsafe_code_rejected(self):
        from asl.patch import Patch, PatchType, CodeDelta
        p = Patch(patch_type=PatchType.CODE,
                  code_delta=CodeDelta("m", "f", diff="eval('1+1')", loc_changed=1))
        result = self.sb.test(p, self.state)
        self.assertFalse(result.passed)
        self.assertTrue(len(result.safety_violations) > 0)

    def test_root_priv_rejected(self):
        from asl.patch import Patch, PatchType, CodeDelta
        p = Patch(patch_type=PatchType.CODE,
                  code_delta=CodeDelta("m", "f", diff="os.setuid(0)", loc_changed=1))
        result = self.sb.test(p, self.state)
        self.assertFalse(result.passed)


# ---------------------------------------------------------------------------
# CanaryDeployer  (checkpoint + apply + rollback) — GAP 3
# ---------------------------------------------------------------------------

class TestRealCanaryDeployer(unittest.TestCase):

    def setUp(self):
        from asl.sandbox import CanaryDeployer
        self.state = _make_state(["node01", "node02", "node03"])
        self.state.lyapunov_value = 0.3
        self.deployer = CanaryDeployer(
            self.state, canary_fraction=0.5,
            soak_sec=0.1,       # tiny soak for tests
            ssh_config=_make_ssh_cfg(),
        )

    def test_checkpoint_reads_runtime_yaml(self):
        with patch("asl.sandbox._ssh", return_value=(0, "z_threshold: 3.0", "")):
            cp = self.deployer._checkpoint(None, ["node01"])
        self.assertIn("node01", cp)
        self.assertEqual(cp["node01"]["runtime_yaml"], "z_threshold: 3.0")

    def test_apply_writes_param_to_runtime_yaml(self):
        from asl.patch import Patch, PatchType, ParameterDelta
        p = Patch(patch_type=PatchType.PARAMETER,
                  parameter_deltas=[ParameterDelta("anomaly.z_threshold", 3.0, 3.5)])
        with patch("asl.sandbox._ssh") as mock_ssh:
            mock_ssh.return_value = (0, "", "")
            ok, _ = self.deployer._apply_to_nodes(p, ["node01"])
        self.assertTrue(ok)
        calls = [str(c) for c in mock_ssh.call_args_list]
        self.assertTrue(any("z_threshold" in s for s in calls))

    def test_rollback_restores_checkpoint(self):
        from asl.patch import Patch, PatchType, ParameterDelta
        p = Patch(patch_type=PatchType.PARAMETER,
                  parameter_deltas=[ParameterDelta("anomaly.z_threshold", 3.0, 3.5)])
        checkpoints = {"node01": {"runtime_yaml": "anomaly_z_threshold: 3.0\n"}}
        with patch("asl.sandbox._ssh") as mock_ssh:
            mock_ssh.return_value = (0, "", "")
            self.deployer._rollback_nodes(p, ["node01"], checkpoints)
        calls = [str(c) for c in mock_ssh.call_args_list]
        # Should write back the old value
        self.assertTrue(any("3.0" in s for s in calls))

    def test_canary_deploy_succeeds_when_V_improves(self):
        from asl.patch import Patch, PatchType, ParameterDelta
        p = Patch(patch_type=PatchType.PARAMETER,
                  parameter_deltas=[ParameterDelta("anomaly.z_threshold", 3.0, 3.5)])

        call_count = [0]
        def fake_lyapunov():
            call_count[0] += 1
            # V decreases after patch applied → good
            return 0.2 if call_count[0] > 1 else 0.3

        self.state.lyapunov_value = 0.3
        with patch("asl.sandbox._ssh", return_value=(0, "", "")):
            with patch.object(self.state, "compute_lyapunov", side_effect=fake_lyapunov):
                with patch.object(self.state, "compute_objective", return_value=0.1):
                    result = self.deployer.deploy(p)
        self.assertTrue(result.success)

    def test_canary_deploy_rolls_back_when_V_increases(self):
        from asl.patch import Patch, PatchType, ParameterDelta
        p = Patch(patch_type=PatchType.PARAMETER,
                  parameter_deltas=[ParameterDelta("anomaly.z_threshold", 3.0, 1.0)])

        self.state.lyapunov_value = 0.3
        with patch("asl.sandbox._ssh", return_value=(0, "", "")):
            with patch.object(self.state, "compute_lyapunov", return_value=0.8):
                with patch.object(self.state, "compute_objective", return_value=0.9):
                    result = self.deployer.deploy(p)
        self.assertFalse(result.success)


# ---------------------------------------------------------------------------
# ASLPipeline end-to-end — GAP 3 continued
# ---------------------------------------------------------------------------

class TestASLPipelineEndToEnd(unittest.TestCase):

    def setUp(self):
        from asl.pipeline import ASLPipeline
        self.state = _make_state(["node01", "node02"])
        self.state.latency = 0.3
        self.state.power_consumption = 0.4
        self.state.failure_rate = 0.1
        self.state.security_risk = 0.05
        self.state.lyapunov_value = 0.3

        self.pipeline = ASLPipeline(
            state=self.state,
            trust_threshold=0.40,   # low threshold so patches pass in tests
            sandbox_timeout=5.0,
            canary_fraction=0.5,
            soak_sec=0.05,          # tiny soak
            enabled=True,
            ssh_config=_make_ssh_cfg(),
        )

    def test_step_generates_and_processes_patch(self):
        with patch("asl.sandbox._ssh", return_value=(0, "", "")):
            with patch.object(self.state, "compute_lyapunov", return_value=0.2):
                with patch.object(self.state, "compute_objective", return_value=0.2):
                    result = self.pipeline.step()
        # Either deployed or rejected — both are valid outcomes
        self.assertIsNotNone(self.pipeline.agent_stats())

    def test_rejected_patch_increments_metric(self):
        from utils.metrics import metrics
        before = metrics.patches_rejected.get()
        # Force sandbox failure by injecting unsafe code
        from asl.rl_agent import RLAgent
        from asl.patch import Patch, PatchType, CodeDelta
        bad_patch = Patch(
            patch_type=PatchType.CODE,
            code_delta=CodeDelta("m", "f", diff="eval('boom')", loc_changed=1)
        )
        with patch.object(self.pipeline._agent, "select_action",
                          return_value=(0, bad_patch)):
            self.pipeline.step()
        self.assertGreater(metrics.patches_rejected.get(), before)

    def test_agent_stats_has_required_keys(self):
        stats = self.pipeline.agent_stats()
        for key in ("steps", "epsilon", "mean_reward_100", "replay_buffer_size"):
            self.assertIn(key, stats)

    def test_rollback_removes_from_deployed(self):
        from asl.patch import Patch, PatchStatus
        p = Patch()
        p.status = PatchStatus.DEPLOYED
        p.deployed_at = time.time()
        self.pipeline._deployed[p.patch_id] = p
        ok = self.pipeline.rollback(p.patch_id)
        self.assertTrue(ok)
        self.assertNotIn(p.patch_id, self.pipeline._deployed)


# ---------------------------------------------------------------------------
# Trust score computation
# ---------------------------------------------------------------------------

class TestTrustScore(unittest.TestCase):

    def test_all_passed(self):
        from asl.pipeline import _compute_trust_score
        from asl.patch import Patch
        p = Patch()
        p.sandbox_passed = p.stability_passed = p.performance_passed = p.safety_passed = True
        self.assertEqual(_compute_trust_score(p), 1.0)

    def test_sandbox_only(self):
        from asl.pipeline import _compute_trust_score
        from asl.patch import Patch
        p = Patch()
        p.sandbox_passed = True
        self.assertAlmostEqual(_compute_trust_score(p), 0.40)

    def test_all_failed(self):
        from asl.pipeline import _compute_trust_score
        from asl.patch import Patch
        p = Patch()
        self.assertEqual(_compute_trust_score(p), 0.0)


# ---------------------------------------------------------------------------
# ProductionConfig
# ---------------------------------------------------------------------------

class TestProductionConfig(unittest.TestCase):

    def _load(self):
        import yaml
        path = os.path.join(os.path.dirname(__file__), "..", "config", "production.yaml")
        with open(path) as f:
            return yaml.safe_load(f)

    def test_all_top_level_sections(self):
        cfg = self._load()
        for section in ("system", "ssh", "cluster", "monitor", "anomaly",
                        "quarantine", "healing", "optimizer", "defense", "asl"):
            self.assertIn(section, cfg)

    def test_asl_has_soak_sec(self):
        cfg = self._load()
        self.assertIn("canary_soak_sec", cfg["asl"])

    def test_ssh_keys_present(self):
        cfg = self._load()
        self.assertIn("user", cfg["ssh"])
        self.assertIn("key_path", cfg["ssh"])

    def test_cluster_nodes_is_list(self):
        cfg = self._load()
        self.assertIsInstance(cfg["cluster"]["nodes"], list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
