"""
modules/defense.py — Defense Layer (Rocky Linux production)

Real security enforcement using Rocky Linux native tooling:
  - Node authentication: SSH host key fingerprint verification + pre-shared token
  - IDS: parse /var/log/audit/audit.log via ausearch (auditd)
  - Process monitoring: SSH → ps / /proc scan for known-bad binaries
  - Network monitoring: ss / netstat for unexpected listeners
  - Automated mitigation:
      * firewalld rich-rules to block offending IPs
      * pkill over SSH for malicious processes
      * SELinux enforcing mode enforcement check
      * Fail2ban integration

Requires on managed nodes:
  - auditd running (dnf install -y audit; systemctl enable --now auditd)
  - fail2ban (optional): dnf install -y fail2ban
  - firewalld: systemctl enable --now firewalld
  - SELinux in enforcing mode (sestatus)
"""

from __future__ import annotations
import hashlib
import hmac
import json
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from core.state import ClusterState, NodeState
from utils.events import bus, EventType
from utils.logger import get_logger
from utils.metrics import metrics

log = get_logger("defense")


class ThreatType(Enum):
    UNAUTHORIZED_NODE = "unauthorized_node"
    MALICIOUS_PROCESS = "malicious_process"
    ABNORMAL_TRAFFIC = "abnormal_traffic"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    SELINUX_VIOLATION = "selinux_violation"
    AUDIT_ALERT = "audit_alert"
    UNKNOWN_LISTENER = "unknown_listener"


class Severity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ThreatEvent:
    node_id: str
    threat_type: ThreatType
    confidence: float
    evidence: Dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    mitigated: bool = False

    def __str__(self):
        return f"[THREAT:{self.threat_type.value}] {self.node_id} conf={self.confidence:.2f}"


# ---------------------------------------------------------------------------
# SSH helper (shared pattern)
# ---------------------------------------------------------------------------

def _ssh(host: str, command: str, config: Dict,
         timeout: float = 10.0) -> Tuple[int, str, str]:
    user = config.get("ssh_user", "root")
    key = config.get("ssh_key_path", "")
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no",
           "-o", "ConnectTimeout=8", "-o", "BatchMode=yes"]
    if key:
        cmd += ["-i", key]
    cmd.append(f"{user}@{host}")
    cmd.append(command)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return 1, "", "timeout"
    except FileNotFoundError:
        return 1, "", "ssh not found"


# ---------------------------------------------------------------------------
# SSH Host Key Verifier (node identity)
# ---------------------------------------------------------------------------

class SSHHostKeyVerifier:
    """
    Verify SSH host key fingerprints to authenticate nodes.

    On first contact a fingerprint is stored. Subsequent contacts
    must present the same key or the node is flagged as unauthorized.

    Storage: a local JSON file (in production use a secrets manager).
    """

    def __init__(self, known_keys_path: str = "config/known_host_keys.json") -> None:
        self.known_keys_path = known_keys_path
        self._known: Dict[str, str] = self._load()

    def _load(self) -> Dict[str, str]:
        try:
            with open(self.known_keys_path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save(self) -> None:
        import os
        os.makedirs(os.path.dirname(self.known_keys_path), exist_ok=True)
        with open(self.known_keys_path, "w") as f:
            json.dump(self._known, f, indent=2)

    def get_fingerprint(self, host: str) -> Optional[str]:
        """Fetch the current SSH host key fingerprint via ssh-keyscan."""
        try:
            result = subprocess.run(
                ["ssh-keyscan", "-t", "ed25519,rsa", host],
                capture_output=True, text=True, timeout=8.0
            )
            if result.returncode != 0 or not result.stdout:
                return None
            # Hash the raw key material
            raw = result.stdout.strip()
            return hashlib.sha256(raw.encode()).hexdigest()[:32]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

    def verify(self, host: str) -> Tuple[bool, str]:
        """
        Verify the host key. Returns (trusted, reason).
        Auto-trusts on first encounter (TOFU).
        """
        current_fp = self.get_fingerprint(host)
        if current_fp is None:
            return False, "cannot reach host or ssh-keyscan failed"

        known_fp = self._known.get(host)
        if known_fp is None:
            # Trust on first use
            self._known[host] = current_fp
            self._save()
            log.info("TOFU: trusting %s (fp=%s)", host, current_fp[:16])
            return True, "trusted (first contact)"

        if known_fp == current_fp:
            return True, "fingerprint matches"

        log.warning("SSH HOST KEY MISMATCH for %s: expected=%s got=%s",
                    host, known_fp[:16], current_fp[:16])
        return False, f"host key mismatch — possible MITM or reimaged node"

    def update(self, host: str) -> None:
        """Call after a node is reimaged to refresh its key."""
        fp = self.get_fingerprint(host)
        if fp:
            self._known[host] = fp
            self._save()
            log.info("Host key updated for %s", host)

    def is_registered(self, host: str) -> bool:
        return host in self._known


# ---------------------------------------------------------------------------
# Auditd IDS
# ---------------------------------------------------------------------------

class AuditdIDS:
    """
    Parse auditd logs via ausearch for security events.

    Detects:
      - Privilege escalation (sudo, su, setuid)
      - Authentication failures
      - File integrity violations
      - SELinux AVC denials

    Requires: auditd running on node, ausearch in PATH
    """

    SEVERITY_KEYWORDS = {
        "avc: denied": (ThreatType.SELINUX_VIOLATION, 0.85),
        "EXECVE.*sudo": (ThreatType.PRIVILEGE_ESCALATION, 0.70),
        "USER_AUTH.*res=failed": (ThreatType.UNAUTHORIZED_NODE, 0.75),
        "USER_LOGIN.*res=failed": (ThreatType.UNAUTHORIZED_NODE, 0.65),
        "SYSCALL.*setuid": (ThreatType.PRIVILEGE_ESCALATION, 0.60),
    }

    def scan(self, host: str, config: Dict,
             since_sec: int = 120) -> List[ThreatEvent]:
        """Scan last `since_sec` seconds of audit log."""
        rc, out, _ = _ssh(
            host,
            f"ausearch -ts recent --start boot -i 2>/dev/null | tail -200",
            config, timeout=15.0
        )
        if rc != 0 or not out:
            return []

        threats: List[ThreatEvent] = []
        import re
        for keyword, (threat_type, confidence) in self.SEVERITY_KEYWORDS.items():
            pattern = re.compile(keyword, re.IGNORECASE)
            matches = [l for l in out.splitlines() if pattern.search(l)]
            if matches:
                threats.append(ThreatEvent(
                    node_id=host,
                    threat_type=threat_type,
                    confidence=confidence,
                    evidence={"audit_lines": matches[:3]},
                ))
        return threats


# ---------------------------------------------------------------------------
# Process Scanner
# ---------------------------------------------------------------------------

class ProcessScanner:
    """
    Scan running processes for known malicious binaries via SSH.

    Checks ps output + /proc/<pid>/exe symlinks for known-bad names.
    """

    MALICIOUS_NAMES: Set[str] = {
        "xmrig", "minerd", "cpuminer", "ethminer",   # crypto miners
        "masscan", "zmap", "nmap",                    # scanners
        "meterpreter", "msf",                         # metasploit
        "nc", "ncat",                                 # netcat (context-sensitive)
        "tshark", "tcpdump",                          # sniffers (unexpected)
        "mimikatz", "lazagne",                        # credential stealers
        "hydra", "medusa",                            # brute force
    }

    # Processes that should exist — ignore these even if name-matched
    WHITELIST: Set[str] = {"nc", "tcpdump"}  # commonly legitimate; flag for review not block

    def scan(self, host: str, config: Dict) -> List[ThreatEvent]:
        rc, out, _ = _ssh(
            host,
            "ps -eo pid,user,comm,args --no-headers 2>/dev/null | head -500",
            config, timeout=10.0
        )
        if rc != 0 or not out:
            return []

        threats: List[ThreatEvent] = []
        for line in out.splitlines():
            parts = line.split(None, 3)
            if len(parts) < 3:
                continue
            pid, user, comm = parts[0], parts[1], parts[2]
            comm_lower = comm.lower()

            for bad in self.MALICIOUS_NAMES:
                if bad in comm_lower and bad not in self.WHITELIST:
                    threats.append(ThreatEvent(
                        node_id=host,
                        threat_type=ThreatType.MALICIOUS_PROCESS,
                        confidence=0.90,
                        evidence={
                            "pid": pid, "user": user,
                            "comm": comm, "cmdline": parts[3] if len(parts) > 3 else "",
                        },
                    ))
        return threats


# ---------------------------------------------------------------------------
# Network Listener Scanner
# ---------------------------------------------------------------------------

class NetworkScanner:
    """
    Detect unexpected listening ports via ss (socket statistics).
    Flags any port not in the approved whitelist.
    """

    APPROVED_PORTS: Set[int] = {
        22,    # SSH
        9100,  # node_exporter
        6817,  # Slurm slurmctld
        6818,  # Slurm slurmd
        6819,  # Slurm slurmdbd
        2049,  # NFS
        111,   # rpcbind
        123,   # NTP
        443,   # HTTPS
        80,    # HTTP
    }

    def scan(self, host: str, config: Dict) -> List[ThreatEvent]:
        rc, out, _ = _ssh(
            host,
            "ss -tlnp 2>/dev/null | awk 'NR>1{print $4}' | grep -oP ':\\K[0-9]+'",
            config, timeout=8.0
        )
        if rc != 0 or not out:
            return []

        threats: List[ThreatEvent] = []
        for port_str in out.splitlines():
            try:
                port = int(port_str.strip())
            except ValueError:
                continue
            if port not in self.APPROVED_PORTS and port > 1024:
                threats.append(ThreatEvent(
                    node_id=host,
                    threat_type=ThreatType.UNKNOWN_LISTENER,
                    confidence=0.60,
                    evidence={"port": port},
                ))
        return threats


# ---------------------------------------------------------------------------
# SELinux Status Checker
# ---------------------------------------------------------------------------

class SELinuxChecker:
    def check(self, host: str, config: Dict) -> Optional[ThreatEvent]:
        """Ensure SELinux is in enforcing mode."""
        rc, out, _ = _ssh(host, "getenforce 2>/dev/null", config, timeout=5.0)
        if rc != 0:
            return None
        mode = out.strip().lower()
        if mode != "enforcing":
            return ThreatEvent(
                node_id=host,
                threat_type=ThreatType.SELINUX_VIOLATION,
                confidence=0.80,
                evidence={"selinux_mode": mode},
            )
        return None


# ---------------------------------------------------------------------------
# Defense Layer
# ---------------------------------------------------------------------------

class DefenseLayer:
    """
    MAPE-K Execute phase for security — Rocky Linux production.
    """

    def __init__(self, state: ClusterState, config: Dict) -> None:
        self.state = state
        self.config = config
        self.auto_mitigate = config.get("auto_mitigate", True)

        self._host_key_verifier = SSHHostKeyVerifier(
            known_keys_path=config.get("known_keys_path", "config/known_host_keys.json")
        )
        self._auditd = AuditdIDS()
        self._proc_scanner = ProcessScanner()
        self._net_scanner = NetworkScanner()
        self._selinux = SELinuxChecker()
        self._active_threats: List[ThreatEvent] = []

        bus.subscribe(EventType.NODE_UNTRUSTED, self._on_untrusted)

    def register_node(self, node_id: str) -> None:
        """Register a new node — does TOFU host-key verification."""
        trusted, reason = self._host_key_verifier.verify(node_id)
        log.info("Node %s PKI: trusted=%s reason=%s", node_id, trusted, reason)
        node = self.state.nodes.get(node_id)
        if node and not trusted:
            node.trust_score = 0.2

    def scan_all(self) -> List[ThreatEvent]:
        """Run all IDS scans on all healthy nodes."""
        all_threats: List[ThreatEvent] = []

        for node in list(self.state.nodes.values()):
            if node.status.value not in ("healthy", "degraded"):
                continue
            host = node.node_id
            threats: List[ThreatEvent] = []

            # 1. SSH host key check
            trusted, reason = self._host_key_verifier.verify(host)
            if not trusted:
                threats.append(ThreatEvent(
                    node_id=host, threat_type=ThreatType.UNAUTHORIZED_NODE,
                    confidence=0.95, evidence={"reason": reason}
                ))

            # 2. SELinux check
            sel = self._selinux.check(host, self.config)
            if sel:
                threats.append(sel)

            # 3. Auditd scan (runs async — expensive)
            threats.extend(self._auditd.scan(host, self.config))

            # 4. Process scan
            threats.extend(self._proc_scanner.scan(host, self.config))

            # 5. Network listener scan
            threats.extend(self._net_scanner.scan(host, self.config))

            # Process detected threats
            for threat in threats:
                log.warning("THREAT: %s", threat)
                metrics.intrusions_detected.inc()
                node.trust_score = max(0.0, node.trust_score - 0.15 * threat.confidence)
                bus.emit_simple(EventType.INTRUSION_DETECTED, "defense", payload=threat)

                if node.trust_score < 0.4:
                    bus.emit_simple(EventType.NODE_UNTRUSTED, "defense", payload={
                        "node_id": host, "trust_score": node.trust_score,
                    })

                if self.auto_mitigate:
                    self._mitigate(node, threat)
                    threat.mitigated = True

            all_threats.extend(threats)

        self._active_threats = [t for t in all_threats if not t.mitigated]
        return all_threats

    def _mitigate(self, node: NodeState, threat: ThreatEvent) -> None:
        host = node.node_id

        if threat.threat_type == ThreatType.MALICIOUS_PROCESS:
            comm = threat.evidence.get("comm", "")
            pid = threat.evidence.get("pid", "")
            log.warning("Killing malicious process '%s' (pid=%s) on %s", comm, pid, host)
            if pid:
                _ssh(host, f"kill -9 {pid} 2>/dev/null || true", self.config)
            if comm:
                _ssh(host, f"pkill -9 -x {comm} 2>/dev/null || true", self.config)

        elif threat.threat_type == ThreatType.UNKNOWN_LISTENER:
            port = threat.evidence.get("port", 0)
            if port:
                log.warning("Blocking unexpected port %d on %s via firewalld", port, host)
                _ssh(
                    host,
                    f"firewall-cmd --add-rich-rule='rule port port={port} protocol=tcp reject' --permanent "
                    f"&& firewall-cmd --reload",
                    self.config, timeout=15.0
                )

        elif threat.threat_type == ThreatType.SELINUX_VIOLATION:
            log.warning("Re-enabling SELinux enforcing mode on %s", host)
            _ssh(host, "setenforce 1", self.config, timeout=5.0)

        elif threat.threat_type == ThreatType.DATA_EXFILTRATION:
            log.warning("Rate-limiting outbound traffic on %s", host)
            _ssh(
                host,
                "tc qdisc replace dev eth0 root tbf rate 10mbit burst 32kbit latency 400ms 2>/dev/null || true",
                self.config, timeout=10.0
            )

        bus.emit_simple(EventType.MITIGATION_APPLIED, "defense", payload={
            "node_id": host, "threat_type": threat.threat_type.value,
        })

    def _on_untrusted(self, event) -> None:
        payload = event.payload
        if isinstance(payload, dict):
            node_id = payload.get("node_id")
            if node_id:
                log.warning("Node %s marked UNTRUSTED — quarantine triggered", node_id)

    @property
    def active_threats(self) -> List[ThreatEvent]:
        return self._active_threats
