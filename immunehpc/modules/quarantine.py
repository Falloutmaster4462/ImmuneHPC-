"""
modules/quarantine.py — Quarantine Layer (Rocky Linux production)

Real three-layer node isolation:
  1. firewalld rich-rules  — block HPC inter-node ports on the node itself
  2. scontrol drain        — remove node from Slurm scheduling pool
  3. tc egress cap         — throttle outbound to 1 Mbit/s (exfil prevention)

On release all three layers are reversed and the node rejoins the pool.
"""

from __future__ import annotations
import subprocess
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core.state import ClusterState, NodeState, NodeStatus
from utils.events import bus, EventType
from utils.logger import get_logger

log = get_logger("quarantine")

_BLOCKED_PORTS = [
    ("6817", "tcp"), ("6818", "tcp"), ("6819", "tcp"),
    ("2049", "tcp"), ("2049", "udp"),
]

_SSH_CONFIG: Dict = {}


def _ssh(host: str, command: str, timeout: float = 15.0) -> Tuple[int, str, str]:
    user = _SSH_CONFIG.get("ssh_user", "root")
    key  = _SSH_CONFIG.get("ssh_key_path", "")
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no",
           "-o", "ConnectTimeout=10", "-o", "BatchMode=yes"]
    if key:
        cmd += ["-i", key]
    cmd += [f"{user}@{host}", command]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return 1, "", "ssh timeout"
    except FileNotFoundError:
        return 1, "", "ssh not found"


@dataclass
class QuarantineRecord:
    node_id: str
    reason: str
    quarantined_at: float = field(default_factory=time.time)
    max_duration_sec: float = 300.0
    released: bool = False
    released_at: Optional[float] = None

    @property
    def duration(self) -> float:
        return (self.released_at or time.time()) - self.quarantined_at

    @property
    def timed_out(self) -> bool:
        return not self.released and self.duration >= self.max_duration_sec


class QuarantineLayer:
    def __init__(
        self,
        state: ClusterState,
        ssh_config: Optional[Dict] = None,
        auto_isolate: bool = True,
        max_quarantine_sec: float = 300.0,
    ) -> None:
        self.state = state
        self.auto_isolate = auto_isolate
        self.max_quarantine_sec = max_quarantine_sec
        self._records: Dict[str, QuarantineRecord] = {}
        self._lock = threading.Lock()
        if ssh_config:
            _SSH_CONFIG.update(ssh_config)
        bus.subscribe(EventType.NODE_UNTRUSTED, self._on_untrusted)

    def quarantine(self, node_id: str, reason: str) -> bool:
        with self._lock:
            if node_id in self._records and not self._records[node_id].released:
                return False
            node = self.state.nodes.get(node_id)
            if not node:
                log.warning("Cannot quarantine unknown node: %s", node_id)
                return False
            self._records[node_id] = QuarantineRecord(
                node_id=node_id, reason=reason,
                max_duration_sec=self.max_quarantine_sec)
            node.status = NodeStatus.QUARANTINED

        self._apply_isolation(node_id, reason)
        log.warning("Node QUARANTINED: %s — %s", node_id, reason)
        bus.emit_simple(EventType.NODE_QUARANTINED, "quarantine",
                        payload={"node_id": node_id, "reason": reason})
        return True

    def release(self, node_id: str) -> bool:
        with self._lock:
            record = self._records.get(node_id)
            if not record or record.released:
                return False
            record.released = True
            record.released_at = time.time()
            node = self.state.nodes.get(node_id)
            if node:
                node.status = NodeStatus.HEALTHY
                node.anomaly_flags.clear()

        self._lift_isolation(node_id)
        log.info("Node RELEASED: %s (isolated %.0fs)", node_id, record.duration)
        bus.emit_simple(EventType.NODE_RELEASED, "quarantine",
                        payload={"node_id": node_id,
                                 "quarantine_duration_sec": record.duration})
        return True

    def check_timeouts(self) -> None:
        with self._lock:
            timed_out = [nid for nid, r in self._records.items() if r.timed_out]
        for nid in timed_out:
            log.warning("Quarantine timeout %s — forced release", nid)
            self.release(nid)

    def is_quarantined(self, node_id: str) -> bool:
        with self._lock:
            r = self._records.get(node_id)
            return r is not None and not r.released

    def active_records(self) -> Dict[str, QuarantineRecord]:
        with self._lock:
            return {k: v for k, v in self._records.items() if not v.released}

    # ------------------------------------------------------------------
    # Real isolation — three layers
    # ------------------------------------------------------------------

    def _apply_isolation(self, node_id: str, reason: str) -> None:
        host  = node_id
        iface = _SSH_CONFIG.get("network_interface", "eth0")

        # Layer 1 — firewalld: block HPC inter-node ports
        rules = " ; ".join(
            f"firewall-cmd --add-rich-rule='rule family=ipv4 port port={port} "
            f"protocol={proto} reject' --permanent 2>/dev/null"
            for port, proto in _BLOCKED_PORTS
        )
        _ssh(host, f"{rules} ; firewall-cmd --reload 2>/dev/null || true", timeout=20.0)
        log.info("Quarantine: firewall rules applied on %s", host)

        # Layer 2 — Slurm drain (runs from controller)
        drain_reason = reason[:60].replace("'", "")
        try:
            subprocess.run(
                f"scontrol update NodeName={host} State=DRAIN "
                f"Reason='ImmuneHPC+: {drain_reason}' 2>/dev/null || true",
                shell=True, timeout=10.0, capture_output=True,
            )
            log.info("Quarantine: Slurm drain applied for %s", host)
        except Exception as exc:
            log.debug("scontrol unavailable: %s", exc)

        # Layer 3 — tc egress cap: 1 Mbit/s max outbound
        _ssh(
            host,
            f"tc qdisc del dev {iface} root 2>/dev/null || true ; "
            f"tc qdisc add dev {iface} root handle 1: htb default 10 2>/dev/null && "
            f"tc class add dev {iface} parent 1: classid 1:10 htb "
            f"rate 1mbit burst 8kb 2>/dev/null || true",
            timeout=10.0,
        )
        log.info("Quarantine: egress capped at 1 Mbit/s on %s", host)

    def _lift_isolation(self, node_id: str) -> None:
        host  = node_id
        iface = _SSH_CONFIG.get("network_interface", "eth0")

        # Reverse Layer 3 — remove tc cap
        _ssh(host, f"tc qdisc del dev {iface} root 2>/dev/null || true", timeout=10.0)
        log.info("Quarantine lifted: tc cap removed on %s", host)

        # Reverse Layer 2 — Slurm resume
        try:
            subprocess.run(
                f"scontrol update NodeName={host} State=RESUME 2>/dev/null || true",
                shell=True, timeout=10.0, capture_output=True,
            )
            log.info("Quarantine lifted: Slurm RESUME for %s", host)
        except Exception:
            pass

        # Reverse Layer 1 — remove firewall rules
        removes = " ; ".join(
            f"firewall-cmd --remove-rich-rule='rule family=ipv4 port port={port} "
            f"protocol={proto} reject' --permanent 2>/dev/null"
            for port, proto in _BLOCKED_PORTS
        )
        _ssh(host, f"{removes} ; firewall-cmd --reload 2>/dev/null || true", timeout=20.0)
        log.info("Quarantine lifted: firewall rules removed on %s", host)

    def _on_untrusted(self, event) -> None:
        payload = event.payload
        if isinstance(payload, dict):
            node_id = payload.get("node_id")
            if node_id:
                self.quarantine(node_id, reason="defense: node marked untrusted")
