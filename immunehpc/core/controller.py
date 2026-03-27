"""
core/controller.py — ImmuneHPC+ MAPE-K Controller (Rocky Linux production)

Differences from simulation version:
  - Uses RealTelemetryCollector (Prometheus / SSH / IPMI / nvidia-smi / Slurm)
  - Uses real SelfHealingEngine (SSH → systemctl / Ansible / LVM / Cobbler)
  - Uses real DefenseLayer (auditd / SELinux / process scanner / firewalld)
  - Node discovery from config file (hostnames / IPs) instead of auto-gen
  - Exports metrics to Prometheus Pushgateway if configured
  - Writes structured JSONL audit trail to disk
"""

from __future__ import annotations
import json
import os
import signal
import time
import threading
from typing import Dict, List, Optional

import yaml

from core.state import ClusterState, NodeState, NodeStatus
from core.supervisor import AutonomousSupervisor
from modules.monitor import ImmuneMonitor
from modules.anomaly import AnomalyDetector
from modules.quarantine import QuarantineLayer
from modules.healer import SelfHealingEngine
from modules.optimizer import OptimisationEngine
from modules.defense import DefenseLayer
from modules.scheduler import AdaptiveScheduler
from asl.pipeline import ASLPipeline
from utils.events import bus, EventType
from utils.logger import get_logger
from utils.metrics import metrics

log = get_logger("controller")


class ImmuneHPCController:
    """Top-level controller — Rocky Linux production build."""

    def __init__(self, config_path: str = "config/production.yaml") -> None:
        self.config = self._load_config(config_path)
        self._running = False
        self._loop_interval = self.config["system"]["loop_interval_sec"]
        self._audit_log_path = self.config["system"].get("audit_log", "logs/audit.jsonl")
        os.makedirs(os.path.dirname(self._audit_log_path), exist_ok=True)

        self.state = ClusterState()

        # Build sub-configs to pass into real backends
        collector_cfg = {
            "ssh_user":           self.config["ssh"]["user"],
            "ssh_key_path":       self.config["ssh"]["key_path"],
            "node_exporter_port": self.config["monitor"].get("node_exporter_port", 9100),
            "use_ipmi":           self.config["monitor"].get("use_ipmi", True),
            "use_gpu":            self.config["monitor"].get("use_gpu", False),
            "use_slurm":          self.config["monitor"].get("use_slurm", True),
        }
        healer_cfg = {
            **collector_cfg,
            "ansible_playbook":  self.config["healing"].get("ansible_playbook", "ansible/site.yml"),
            "ansible_inventory": self.config["healing"].get("ansible_inventory", "ansible/inventory.ini"),
            "cobbler_host":      self.config["healing"].get("cobbler_host", "cobbler"),
            "cobbler_user":      self.config["healing"].get("cobbler_user", "cobbler"),
            "cobbler_password":  self.config["healing"].get("cobbler_password", "cobbler"),
            "bmc_user":          self.config["healing"].get("bmc_user", "admin"),
            "bmc_password":      self.config["healing"].get("bmc_password", ""),
        }
        defense_cfg = {
            **collector_cfg,
            "auto_mitigate":   self.config["defense"].get("auto_mitigate", True),
            "known_keys_path": self.config["defense"].get(
                "known_keys_path", "config/known_host_keys.json"),
        }

        cfg_mon  = self.config["monitor"]
        cfg_anom = self.config["anomaly"]
        cfg_opt  = self.config["optimizer"]
        cfg_asl  = self.config["asl"]
        cfg_quar = self.config["quarantine"]
        cfg_heal = self.config["healing"]

        self.monitor = ImmuneMonitor(
            state=self.state,
            collector_config=collector_cfg,
            interval_sec=cfg_mon["telemetry_interval_sec"],
            health_threshold=cfg_mon["health_threshold"],
        )
        self.anomaly_detector = AnomalyDetector(
            z_threshold=cfg_anom["z_score_threshold"],
            method=cfg_anom["method"],
        )
        self.quarantine = QuarantineLayer(
            state=self.state,
            ssh_config=collector_cfg,
            auto_isolate=cfg_quar["auto_isolate"],
            max_quarantine_sec=cfg_quar["max_quarantine_sec"],
        )
        self.healer = SelfHealingEngine(
            state=self.state,
            config=healer_cfg,
            max_attempts=cfg_heal["max_attempts"],
            backoff_sec=cfg_heal["backoff_sec"],
        )
        self.optimizer = OptimisationEngine(
            state=self.state,
            weights=cfg_opt["objectives"],
            interval_sec=cfg_opt["interval_sec"],
            ssh_config=collector_cfg,
        )
        self.defense = DefenseLayer(state=self.state, config=defense_cfg)
        self.scheduler = AdaptiveScheduler(state=self.state)
        self.asl = ASLPipeline(
            state=self.state,
            trust_threshold=cfg_asl["patch_trust_threshold"],
            sandbox_timeout=cfg_asl["sandbox_timeout_sec"],
            canary_fraction=cfg_asl["canary_fraction"],
            soak_sec=cfg_asl.get("canary_soak_sec", 60.0),
            enabled=cfg_asl["enabled"],
            ssh_config=collector_cfg,
            llm_config=cfg_asl.get("llm"),           # None = ΔC disabled
            code_gen_interval=cfg_asl.get("code_gen_interval", 20),
        )
        self.supervisor = AutonomousSupervisor(state=self.state)

        bus.subscribe_all(self._audit_event)
        signal.signal(signal.SIGINT,  self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        log.info("ImmuneHPC+ Controller (production) initialised")

    # ------------------------------------------------------------------
    # Provisioning
    # ------------------------------------------------------------------

    def provision_nodes(self, node_list: Optional[List[str]] = None) -> None:
        nodes = node_list or self.config["cluster"]["nodes"]
        for host in nodes:
            node = NodeState(node_id=host, health_score=1.0)
            self.state.nodes[host] = node
            self.defense.register_node(host)
            log.info("Provisioned node: %s", host)
        self.state.aggregate()
        log.info("Cluster ready: %d nodes", len(self.state.nodes))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, block: bool = True) -> None:
        self._running = True
        self.monitor.start()
        log.info("=" * 60)
        log.info("ImmuneHPC+ PRODUCTION Controller STARTED")
        log.info("Nodes: %d | Loop: %.1fs", len(self.state.nodes), self._loop_interval)
        log.info("=" * 60)
        if block:
            self._loop()
        else:
            t = threading.Thread(target=self._loop, daemon=True, name="mape-loop")
            t.start()

    def stop(self) -> None:
        log.info("Stopping ImmuneHPC+...")
        self._running = False
        self.monitor.stop()
        bus.emit_simple(EventType.SHUTDOWN, "controller")
        self._push_metrics()
        log.info("Final metrics:\n%s", json.dumps(metrics.dump(), indent=2))

    # ------------------------------------------------------------------
    # MAPE-K loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        tick = 0
        while self._running:
            t0 = time.time()
            tick += 1
            try:
                self._mape_tick(tick)
            except Exception as exc:
                log.error("MAPE tick %d error: %s", tick, exc, exc_info=True)
            time.sleep(max(0.0, self._loop_interval - (time.time() - t0)))
            bus.emit_simple(EventType.LOOP_TICK, "controller", payload={"tick": tick})

    def _mape_tick(self, tick: int) -> None:
        # ANALYZE
        self.anomaly_detector.analyze(self.state)

        # EXECUTE — Defense scan (SSH-heavy; run every 6 ticks ≈ 1 min)
        if tick % 6 == 0:
            self.defense.scan_all()

        # EXECUTE — Optimise
        self.optimizer.step()

        # EXECUTE — Schedule
        self.scheduler.tick()

        # ARBITRATE
        self.supervisor.supervise(
            quarantine_layer=self.quarantine,
            healer=self.healer,
            asl_pipeline=self.asl,
        )

        if tick % 3 == 0:
            self._log_status(tick)

        # Push metrics every 30 ticks
        if tick % 30 == 0:
            self._push_metrics()

    # ------------------------------------------------------------------
    # Prometheus Pushgateway export
    # ------------------------------------------------------------------

    def _push_metrics(self) -> None:
        pgw = self.config.get("prometheus", {}).get("pushgateway_url", "")
        if not pgw:
            return
        import urllib.request
        lines = [f"immunehpc_{k} {v}" for k, v in metrics.dump().items()
                 if isinstance(v, (int, float))]
        body = "\n".join(lines) + "\n"
        try:
            req = urllib.request.Request(
                f"{pgw}/metrics/job/immunehpc",
                data=body.encode(), method="POST")
            urllib.request.urlopen(req, timeout=5.0)
        except Exception as exc:
            log.debug("Pushgateway push failed: %s", exc)

    # ------------------------------------------------------------------
    # Audit log
    # ------------------------------------------------------------------

    def _audit_event(self, event) -> None:
        try:
            record = {"ts": event.timestamp, "event": event.type.value,
                      "source": event.source, "id": event.event_id}
            with open(self._audit_log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass

    def _log_status(self, tick: int) -> None:
        snap = self.state.snapshot()
        rl   = self.asl.agent_stats()
        log.info(
            "[tick %d] nodes=%d/%d J=%.4f V=%.4f | anomalies=%d threats=%d | ε=%.3f R=%.4f",
            tick, snap["healthy"], snap["node_count"],
            snap["objective_J"], snap["lyapunov_V"],
            len(self.anomaly_detector.active_anomalies()),
            len(self.defense.active_threats),
            rl["epsilon"], rl["mean_reward_100"],
        )

    def status_report(self) -> Dict:
        return {
            "cluster":    self.state.snapshot(),
            "supervisor": self.supervisor.status(),
            "scheduler":  {"queue": self.scheduler.queue_depth,
                           "running": self.scheduler.running_count},
            "asl":        self.asl.agent_stats(),
            "metrics":    metrics.dump(),
        }

    def _handle_signal(self, sig, frame) -> None:
        log.info("Signal %d received", sig)
        self.stop()

    def _load_config(self, path: str) -> dict:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        log.info("Config loaded from %s", path)
        return cfg
