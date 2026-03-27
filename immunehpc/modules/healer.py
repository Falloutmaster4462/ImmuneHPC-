"""
modules/healer.py — Self-Healing Engine (Rocky Linux production)

Hierarchical repair pipeline using real Rocky Linux tooling:

  1. restart_service  — SSH → systemctl restart <services>
  2. reapply_config   — Ansible playbook push
  3. rollback         — LVM snapshot revert or btrfs rollback
  4. reimage_node     — Cobbler power-cycle + PXE reinstall

Each strategy runs over SSH (key-based, no password).
Requires:
  - ssh key access to all managed nodes
  - Ansible installed on the controller: dnf install -y ansible
  - LVM snapshots or btrfs subvolumes pre-configured on nodes
  - Cobbler or similar PXE server for reimaging
"""

from __future__ import annotations
import os
import time
import subprocess
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

from core.state import ClusterState, NodeState, NodeStatus
from utils.events import bus, EventType
from utils.logger import get_logger
from utils.metrics import metrics

log = get_logger("healer")


class RepairStrategy(Enum):
    RESTART_SERVICE = "restart_service"
    REAPPLY_CONFIG = "reapply_config"
    ROLLBACK = "rollback"
    REIMAGE_NODE = "reimage_node"


@dataclass
class RepairAttempt:
    node_id: str
    strategy: RepairStrategy
    started_at: float = field(default_factory=time.time)
    success: Optional[bool] = None
    duration_sec: float = 0.0
    message: str = ""


@dataclass
class HealingRecord:
    node_id: str
    started_at: float = field(default_factory=time.time)
    attempts: List[RepairAttempt] = field(default_factory=list)
    resolved: bool = False
    resolved_at: Optional[float] = None

    @property
    def mttr(self) -> Optional[float]:
        if self.resolved and self.resolved_at:
            return self.resolved_at - self.started_at
        return None


# ---------------------------------------------------------------------------
# SSH helper
# ---------------------------------------------------------------------------

def _ssh(host: str, command: str, config: Dict,
         timeout: float = 30.0) -> Tuple[int, str, str]:
    """
    Run a command on a remote host over SSH.
    Returns (returncode, stdout, stderr).
    """
    user = config.get("ssh_user", "root")
    key = config.get("ssh_key_path", "")
    ssh_cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        "-o", "BatchMode=yes",
    ]
    if key:
        ssh_cmd += ["-i", key]
    ssh_cmd.append(f"{user}@{host}")
    ssh_cmd.append(command)

    try:
        result = subprocess.run(
            ssh_cmd, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return 1, "", f"SSH command timed out after {timeout}s"
    except FileNotFoundError:
        return 1, "", "ssh not found in PATH"


# ---------------------------------------------------------------------------
# Strategy 1: restart_service — systemctl restart
# ---------------------------------------------------------------------------

# Services watched on each node; extend for your stack
WATCHED_SERVICES = [
    "slurmd",        # Slurm node daemon
    "munge",         # Slurm auth
    "prometheus-node-exporter",
    "sshd",
    "firewalld",
    "chronyd",       # NTP
]


def _restart_service(node: NodeState, config: Dict) -> Tuple[bool, str]:
    """
    SSH to node and restart any failed systemd services.

    1. Find all failed units
    2. Restart them in order
    3. Re-check status
    """
    host = node.node_id

    # Find failed services
    rc, out, _ = _ssh(host, "systemctl list-units --state=failed --no-legend --plain -o json 2>/dev/null || "
                            "systemctl list-units --state=failed --no-legend --plain", config)
    failed_services = []
    if rc == 0 and out:
        for line in out.splitlines():
            parts = line.split()
            if parts:
                svc = parts[0].strip()
                if svc.endswith(".service"):
                    failed_services.append(svc)

    # Also check our watched list
    for svc in WATCHED_SERVICES:
        rc2, _, _ = _ssh(host, f"systemctl is-active {svc}", config, timeout=5.0)
        if rc2 != 0:
            svc_full = svc if svc.endswith(".service") else f"{svc}.service"
            if svc_full not in failed_services:
                failed_services.append(svc_full)

    if not failed_services:
        log.info("No failed services found on %s", host)
        return True, "no failed services detected"

    log.info("Restarting failed services on %s: %s", host, failed_services)
    all_ok = True
    messages = []

    for svc in failed_services:
        # Reset failed state first, then restart
        _ssh(host, f"systemctl reset-failed {svc}", config, timeout=10.0)
        rc, out, err = _ssh(host, f"systemctl restart {svc}", config, timeout=30.0)
        if rc == 0:
            messages.append(f"{svc}:restarted")
        else:
            messages.append(f"{svc}:FAILED({err[:60]})")
            all_ok = False

    return all_ok, " | ".join(messages)


# ---------------------------------------------------------------------------
# Strategy 2: reapply_config — Ansible playbook
# ---------------------------------------------------------------------------

def _reapply_config(node: NodeState, config: Dict) -> Tuple[bool, str]:
    """
    Run an Ansible playbook targeting the specific node.

    Expects:
      config["ansible_playbook"]  — path to playbook YAML
      config["ansible_inventory"] — path to inventory file or directory
      config["ssh_key_path"]      — SSH key for Ansible

    Playbook should be idempotent and cover:
      - Package state (dnf)
      - Service state (systemd)
      - Config files (templates)
      - Firewall rules
    """
    playbook = config.get("ansible_playbook", "ansible/site.yml")
    inventory = config.get("ansible_inventory", "ansible/inventory.ini")
    ssh_key = config.get("ssh_key_path", "")
    host = node.node_id

    if not os.path.exists(playbook):
        log.warning("Ansible playbook not found: %s — skipping reapply_config", playbook)
        return False, f"playbook not found: {playbook}"

    cmd = [
        "ansible-playbook",
        "-i", inventory,
        "--limit", host,
        "--private-key", ssh_key,
        "-e", f"target_host={host}",
        playbook,
    ]

    log.info("Running Ansible playbook for %s: %s", host, " ".join(cmd))
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300.0
        )
        success = result.returncode == 0
        # Extract Ansible recap line
        recap = ""
        for line in result.stdout.splitlines():
            if "PLAY RECAP" in line or host in line:
                recap += line.strip() + " "
        msg = recap.strip() or result.stderr[:120]
        log.debug("Ansible stdout:\n%s", result.stdout[-1000:])
        return success, msg[:200]
    except subprocess.TimeoutExpired:
        return False, "ansible-playbook timed out (300s)"
    except FileNotFoundError:
        return False, "ansible-playbook not found — install ansible on controller"


# ---------------------------------------------------------------------------
# Strategy 3: rollback — LVM snapshot or btrfs subvolume
# ---------------------------------------------------------------------------

def _rollback(node: NodeState, config: Dict) -> Tuple[bool, str]:
    """
    Revert the node's root filesystem to the last known-good snapshot.

    Supports two backends (auto-detected):
      A. LVM snapshots:
         - Assumes a snapshot named <vg>/<lv>_snap exists
         - Merges snapshot back into origin LV
         - Node must reboot to complete merge

      B. btrfs snapshots:
         - Assumes /.snapshots/<N>/snapshot directory exists
         - Replaces default subvolume with last snapshot
         - Node must reboot to activate

    In both cases the node is rebooted after initiating the rollback.
    """
    host = node.node_id

    # Detect filesystem type
    rc, out, _ = _ssh(host, "findmnt -n -o FSTYPE /", config, timeout=10.0)
    fstype = out.strip().lower()

    if "btrfs" in fstype:
        return _rollback_btrfs(host, config)
    else:
        return _rollback_lvm(host, config)


def _rollback_lvm(host: str, config: Dict) -> Tuple[bool, str]:
    """LVM snapshot merge rollback."""
    # Find the most recent snapshot for root VG/LV
    rc, out, _ = _ssh(host, "lvs --noheadings -o lv_name,vg_name,lv_attr", config)
    if rc != 0:
        return False, "lvs failed — cannot detect LVM snapshots"

    snap_lv = snap_vg = ""
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and "s" in parts[2]:   # 's' in attr = snapshot
            snap_lv, snap_vg = parts[0], parts[1]
            break

    if not snap_lv:
        return False, "no LVM snapshots found"

    log.info("Merging LVM snapshot %s/%s on %s", snap_vg, snap_lv, host)

    rc, _, err = _ssh(host, f"lvconvert --merge {snap_vg}/{snap_lv}", config, timeout=30.0)
    if rc != 0:
        return False, f"lvconvert failed: {err[:100]}"

    # Reboot to complete merge
    _ssh(host, "shutdown -r +1 'ImmuneHPC+ rollback reboot'", config, timeout=10.0)
    node_state_ref = None  # will be picked up by monitor after reboot
    return True, f"LVM snapshot merge initiated on {host} — rebooting in 1 min"


def _rollback_btrfs(host: str, config: Dict) -> Tuple[bool, str]:
    """btrfs snapshot rollback using snapper."""
    # List snapper snapshots
    rc, out, _ = _ssh(host, "snapper list --columns number,type,description 2>/dev/null | tail -5", config)
    if rc != 0 or not out.strip():
        return False, "snapper not available or no snapshots"

    # Get the most recent pre-snapshot
    snap_num = None
    for line in reversed(out.strip().splitlines()):
        parts = line.split("|")
        if len(parts) >= 2:
            try:
                snap_num = int(parts[0].strip())
                break
            except ValueError:
                continue

    if snap_num is None:
        return False, "no usable btrfs snapshots found"

    log.info("Rolling back to snapper snapshot #%d on %s", snap_num, host)

    rc, _, err = _ssh(host, f"snapper rollback {snap_num}", config, timeout=30.0)
    if rc != 0:
        return False, f"snapper rollback failed: {err[:100]}"

    _ssh(host, "shutdown -r +1 'ImmuneHPC+ btrfs rollback'", config, timeout=10.0)
    return True, f"btrfs rollback to snapshot #{snap_num} initiated on {host}"


# ---------------------------------------------------------------------------
# Strategy 4: reimage — Cobbler / BMC PXE boot
# ---------------------------------------------------------------------------

def _reimage_node(node: NodeState, config: Dict) -> Tuple[bool, str]:
    """
    Full OS reinstall via PXE:
      1. Set Cobbler system to netboot
      2. IPMI power-cycle
      3. Kickstart installs Rocky Linux unattended
      4. Wait for node to come back (poll SSH)

    Requires:
      config["cobbler_host"]     — Cobbler server hostname
      config["cobbler_user"]     — Cobbler xmlrpc user
      config["cobbler_password"] — Cobbler xmlrpc password
      config["bmc_user"]         — IPMI user
      config["bmc_password"]     — IPMI password
    """
    import xmlrpc.client

    host = node.node_id
    cobbler_host = config.get("cobbler_host", "cobbler")
    bmc_user = config.get("bmc_user", "admin")
    bmc_pass = config.get("bmc_password", "")
    ssh_config = config

    log.warning("REIMAGING node %s — full OS reinstall", host)
    node.status = NodeStatus.REIMAGING

    # --- Step 1: Cobbler — enable netboot ---
    try:
        url = f"http://{cobbler_host}/cobbler_api"
        server = xmlrpc.client.ServerProxy(url)
        token = server.login(
            config.get("cobbler_user", "cobbler"),
            config.get("cobbler_password", "cobbler"),
        )
        system_handle = server.get_system_handle(host, token)
        server.modify_system(system_handle, "netboot-enabled", True, token)
        server.save_system(system_handle, token)
        server.sync(token)
        log.info("Cobbler netboot enabled for %s", host)
    except Exception as exc:
        log.warning("Cobbler unavailable: %s — falling back to ipmitool pxe boot", exc)

    # --- Step 2: IPMI — set boot device to PXE and power cycle ---
    bmc_ip = config.get(f"bmc_ip_{host}", host)  # per-node BMC IP in config
    ipmi_base = ["ipmitool", "-H", bmc_ip, "-U", bmc_user, "-P", bmc_pass, "-I", "lanplus"]

    for cmd_args in [
        ["chassis", "bootdev", "pxe", "options=persistent"],
        ["chassis", "power", "cycle"],
    ]:
        try:
            result = subprocess.run(
                ipmi_base + cmd_args,
                capture_output=True, text=True, timeout=15.0
            )
            if result.returncode != 0:
                log.warning("IPMI cmd %s failed: %s", cmd_args, result.stderr)
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            log.warning("IPMI not available: %s", exc)

    # --- Step 3: Wait for node to come back (up to 20 minutes) ---
    log.info("Waiting for %s to reimage (up to 20 min)...", host)
    deadline = time.time() + 1200   # 20 minutes

    while time.time() < deadline:
        time.sleep(30)
        rc, out, _ = _ssh(host, "uptime", ssh_config, timeout=10.0)
        if rc == 0:
            node.status = NodeStatus.HEALTHY
            node.health_score = 1.0
            node.anomaly_flags.clear()
            log.info("Node %s reimaged and back online", host)
            return True, f"node reimaged successfully (uptime: {out})"

    node.status = NodeStatus.OFFLINE
    return False, f"node {host} did not come back within 20 minutes after reimage"


# ---------------------------------------------------------------------------
# Strategy dispatch table
# ---------------------------------------------------------------------------

def _make_strategy(fn, config: Dict):
    """Wrap a strategy function with the config dict."""
    return lambda node: fn(node, config)


# ---------------------------------------------------------------------------
# Self-Healing Engine
# ---------------------------------------------------------------------------

class SelfHealingEngine:
    """
    MAPE-K Execute phase — Rocky Linux production implementation.

    Each repair strategy runs over SSH using real system tools.
    All strategies are non-blocking (run in daemon threads).
    """

    def __init__(
        self,
        state: ClusterState,
        config: Dict,
        strategy_order: Optional[List[RepairStrategy]] = None,
        max_attempts: int = 2,
        backoff_sec: float = 30.0,
    ) -> None:
        self.state = state
        self.config = config
        self.strategy_order = strategy_order or [
            RepairStrategy.RESTART_SERVICE,
            RepairStrategy.REAPPLY_CONFIG,
            RepairStrategy.ROLLBACK,
            RepairStrategy.REIMAGE_NODE,
        ]
        self.max_attempts = max_attempts
        self.backoff_sec = backoff_sec
        self._records: Dict[str, HealingRecord] = {}
        self._lock = threading.Lock()

        bus.subscribe(EventType.HEALTH_DEGRADED, self._on_health_degraded)
        bus.subscribe(EventType.ANOMALY_DETECTED, self._on_anomaly)

    def heal(self, node_id: str) -> bool:
        node = self.state.nodes.get(node_id)
        if not node:
            log.error("Cannot heal unknown node: %s", node_id)
            return False
        if node.status in (NodeStatus.QUARANTINED, NodeStatus.REIMAGING):
            return False

        with self._lock:
            record = self._records.get(node_id)
            if record and not record.resolved:
                log.debug("Healing already in progress for %s", node_id)
                return False
            record = HealingRecord(node_id=node_id)
            self._records[node_id] = record

        log.info("Initiating healing for %s", node_id)
        bus.emit_simple(EventType.HEAL_STARTED, "healer", payload={"node_id": node_id})

        t = threading.Thread(
            target=self._run_pipeline, args=(node, record),
            daemon=True, name=f"heal-{node_id}"
        )
        t.start()
        return True

    def active_healings(self) -> List[HealingRecord]:
        with self._lock:
            return [r for r in self._records.values() if not r.resolved]

    def _strategy_fn(self, strategy: RepairStrategy):
        fns = {
            RepairStrategy.RESTART_SERVICE: _restart_service,
            RepairStrategy.REAPPLY_CONFIG:  _reapply_config,
            RepairStrategy.ROLLBACK:        _rollback,
            RepairStrategy.REIMAGE_NODE:    _reimage_node,
        }
        raw = fns[strategy]
        return lambda node: raw(node, self.config)

    def _run_pipeline(self, node: NodeState, record: HealingRecord) -> None:
        for strategy in self.strategy_order:
            for attempt_num in range(self.max_attempts):
                attempt = RepairAttempt(node_id=node.node_id, strategy=strategy)
                record.attempts.append(attempt)

                log.info("Healing %s: strategy=%s attempt=%d/%d",
                         node.node_id, strategy.value, attempt_num + 1, self.max_attempts)

                fn = self._strategy_fn(strategy)
                t0 = time.time()
                try:
                    success, message = fn(node)
                except Exception as exc:
                    success, message = False, str(exc)

                attempt.duration_sec = time.time() - t0
                attempt.success = success
                attempt.message = message

                if success:
                    self._on_success(node, record, strategy, message)
                    return

                log.warning("Strategy %s failed for %s: %s", strategy.value, node.node_id, message)
                if attempt_num < self.max_attempts - 1:
                    time.sleep(self.backoff_sec)

        self._on_exhausted(node, record)

    def _on_success(self, node, record, strategy, message):
        with self._lock:
            record.resolved = True
            record.resolved_at = time.time()

        if node.status not in (NodeStatus.REIMAGING,):
            node.status = NodeStatus.HEALTHY
            node.health_score = max(node.health_score, 0.7)
            node.anomaly_flags.clear()

        mttr = record.mttr or 0.0
        metrics.recovery_events.inc()
        metrics.mttr_histogram.observe(mttr)
        log.info("Healing SUCCESS for %s via %s in %.1fs — %s",
                 node.node_id, strategy.value, mttr, message)
        bus.emit_simple(EventType.HEAL_SUCCESS, "healer", payload={
            "node_id": node.node_id, "strategy": strategy.value, "mttr_sec": mttr,
        })

    def _on_exhausted(self, node, record):
        with self._lock:
            record.resolved = False
        log.error("All strategies EXHAUSTED for %s — escalating to quarantine", node.node_id)
        bus.emit_simple(EventType.HEAL_FAILED, "healer", payload={
            "node_id": node.node_id, "attempts": len(record.attempts),
        })

    def _on_health_degraded(self, event):
        node: NodeState = event.payload
        if node:
            self.heal(node.node_id)

    def _on_anomaly(self, event):
        from modules.anomaly import Severity
        anomaly = event.payload
        if anomaly and hasattr(anomaly, "node_id"):
            if anomaly.severity in (Severity.HIGH, Severity.CRITICAL):
                self.heal(anomaly.node_id)
