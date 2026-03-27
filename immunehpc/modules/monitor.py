"""
modules/monitor.py — Immune Monitor (Rocky Linux production implementation)

Telemetry collection from real Rocky Linux nodes via:
  - /proc and /sys filesystem (local node, zero-dep)
  - SSH + remote /proc scraping (remote nodes)
  - Prometheus node_exporter HTTP API (preferred for remote)
  - IPMI via ipmitool (hardware temps, power draw)
  - Slurm REST API / squeue CLI (job state)
  - nvidia-smi (GPU metrics, optional)

Requires on each managed node:
  - python3, ssh access (key-based, no password)
  - prometheus-node-exporter running on port 9100 (or fallback to SSH)
  - ipmitool installed and BMC accessible (for thermal/power)
  - slurm-client installed (for job metrics)
"""

from __future__ import annotations
import re
import time
import json
import shlex
import subprocess
import threading
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Tuple

from core.state import ClusterState, NodeState, NodeStatus
from utils.events import bus, EventType
from utils.logger import get_logger
from utils.metrics import metrics

log = get_logger("monitor")


# ---------------------------------------------------------------------------
# Health scoring (unchanged from sim — pure maths)
# ---------------------------------------------------------------------------

def _compute_health_score(node: NodeState) -> float:
    """H_i ∈ [0, 1] — weighted composite of resource and security signals."""
    penalties = 0.0
    penalties += 0.20 * node.cpu_usage
    penalties += 0.15 * node.gpu_usage
    penalties += 0.15 * node.memory_usage

    if node.temperature_c > 90:
        penalties += 0.25
    elif node.temperature_c > 70:
        penalties += 0.10 * ((node.temperature_c - 70) / 20)

    high_rx = max(0.0, node.network_rx_mbps - 800) / 200
    high_tx = max(0.0, node.network_tx_mbps - 800) / 200
    penalties += 0.10 * min(high_rx + high_tx, 1.0)
    penalties += 0.15 * (1.0 - node.trust_score)
    penalties += 0.05 * min(len(node.anomaly_flags), 3)

    return max(0.0, min(1.0, 1.0 - penalties))


# ---------------------------------------------------------------------------
# Local /proc reader (runs on the controller node itself)
# ---------------------------------------------------------------------------

class ProcReader:
    """Read metrics from the local /proc and /sys filesystem."""

    def cpu_usage(self) -> float:
        """Sample CPU idle twice, return busy fraction."""
        def read_idle() -> Tuple[int, int]:
            with open("/proc/stat") as f:
                parts = f.readline().split()
            vals = [int(x) for x in parts[1:]]
            idle = vals[3] + vals[4]          # idle + iowait
            total = sum(vals)
            return idle, total

        idle1, total1 = read_idle()
        time.sleep(0.2)
        idle2, total2 = read_idle()
        delta_total = total2 - total1
        delta_idle = idle2 - idle1
        if delta_total == 0:
            return 0.0
        return 1.0 - (delta_idle / delta_total)

    def memory_usage(self) -> float:
        """Return fraction of memory in use (0–1)."""
        info: Dict[str, int] = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1])
        total = info.get("MemTotal", 1)
        available = info.get("MemAvailable", total)
        return 1.0 - (available / total)

    def load_average(self) -> Tuple[float, float, float]:
        with open("/proc/loadavg") as f:
            parts = f.read().split()
        return float(parts[0]), float(parts[1]), float(parts[2])

    def network_mbps(self, iface: str = "eth0") -> Tuple[float, float]:
        """Return (rx_mbps, tx_mbps) over a 1s sample."""
        def read_bytes(iface: str) -> Tuple[int, int]:
            rx_path = f"/sys/class/net/{iface}/statistics/rx_bytes"
            tx_path = f"/sys/class/net/{iface}/statistics/tx_bytes"
            try:
                with open(rx_path) as f:
                    rx = int(f.read())
                with open(tx_path) as f:
                    tx = int(f.read())
                return rx, tx
            except FileNotFoundError:
                return 0, 0

        rx1, tx1 = read_bytes(iface)
        time.sleep(1.0)
        rx2, tx2 = read_bytes(iface)
        rx_mbps = (rx2 - rx1) * 8 / 1_000_000
        tx_mbps = (tx2 - tx1) * 8 / 1_000_000
        return max(0.0, rx_mbps), max(0.0, tx_mbps)

    def detect_network_interface(self) -> str:
        """Find the first non-loopback interface."""
        try:
            with open("/proc/net/dev") as f:
                for line in f:
                    line = line.strip()
                    if ":" in line:
                        iface = line.split(":")[0].strip()
                        if iface != "lo":
                            return iface
        except Exception:
            pass
        return "eth0"


# ---------------------------------------------------------------------------
# Prometheus node_exporter scraper
# ---------------------------------------------------------------------------

class PrometheusCollector:
    """
    Scrape a Prometheus node_exporter endpoint.
    Default port: 9100.

    Install on Rocky: dnf install -y prometheus-node-exporter
                      systemctl enable --now node_exporter
    """

    def __init__(self, timeout_sec: float = 3.0) -> None:
        self.timeout = timeout_sec

    def scrape(self, host: str, port: int = 9100) -> Dict[str, float]:
        """Return a flat dict of metric_name → value."""
        url = f"http://{host}:{port}/metrics"
        try:
            req = urllib.request.urlopen(url, timeout=self.timeout)
            text = req.read().decode("utf-8")
            return self._parse(text)
        except Exception as exc:
            log.debug("Prometheus scrape failed for %s: %s", host, exc)
            return {}

    def _parse(self, text: str) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for line in text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            # e.g. node_cpu_seconds_total{cpu="0",mode="idle"} 12345.67
            match = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\s*({[^}]*})?\s+([\d.eE+\-nan]+)', line)
            if match:
                name = match.group(1)
                labels_str = match.group(2) or ""
                value_str = match.group(3)
                try:
                    value = float(value_str)
                    key = name + labels_str
                    metrics[key] = value
                except ValueError:
                    pass
        return metrics

    def extract_cpu_usage(self, scraped: Dict[str, float]) -> float:
        """Compute CPU busy fraction from node_cpu_seconds_total."""
        idle = sum(v for k, v in scraped.items()
                   if "node_cpu_seconds_total" in k and 'mode="idle"' in k)
        total = sum(v for k, v in scraped.items()
                    if "node_cpu_seconds_total" in k)
        if total == 0:
            return 0.0
        # This is cumulative — real usage needs delta; use as approximation
        return max(0.0, min(1.0, 1.0 - idle / total))

    def extract_memory_usage(self, scraped: Dict[str, float]) -> float:
        total = scraped.get("node_memory_MemTotal_bytes", 0)
        available = scraped.get("node_memory_MemAvailable_bytes", total)
        if total == 0:
            return 0.0
        return 1.0 - (available / total)

    def extract_network(self, scraped: Dict[str, float], iface: str = "") -> Tuple[float, float]:
        """Extract total rx/tx bytes (caller must delta over time)."""
        if iface:
            rx_key = f'node_network_receive_bytes_total{{device="{iface}"}}'
            tx_key = f'node_network_transmit_bytes_total{{device="{iface}"}}'
            rx = scraped.get(rx_key, 0.0)
            tx = scraped.get(tx_key, 0.0)
        else:
            rx = sum(v for k, v in scraped.items()
                     if "node_network_receive_bytes_total" in k and "lo" not in k)
            tx = sum(v for k, v in scraped.items()
                     if "node_network_transmit_bytes_total" in k and "lo" not in k)
        return rx, tx


# ---------------------------------------------------------------------------
# IPMI collector (hardware temps + power)
# ---------------------------------------------------------------------------

class IPMICollector:
    """
    Read hardware sensors via ipmitool.

    Requires:
      - dnf install -y ipmitool
      - BMC network access or local /dev/ipmi0
      - IPMI credentials in config

    Usage:
      local:  ipmitool sdr type Temperature
      remote: ipmitool -H <bmc_ip> -U admin -P <pass> sdr type Temperature
    """

    def __init__(
        self,
        bmc_host: Optional[str] = None,
        bmc_user: str = "admin",
        bmc_password: str = "",
        use_local: bool = True,
        timeout_sec: float = 5.0,
    ) -> None:
        self.bmc_host = bmc_host
        self.bmc_user = bmc_user
        self.bmc_password = bmc_password
        self.use_local = use_local
        self.timeout = timeout_sec

    def _base_cmd(self) -> List[str]:
        if self.use_local or not self.bmc_host:
            return ["ipmitool"]
        return [
            "ipmitool", "-H", self.bmc_host,
            "-U", self.bmc_user, "-P", self.bmc_password,
            "-I", "lanplus",
        ]

    def _run(self, *args: str) -> Optional[str]:
        cmd = self._base_cmd() + list(args)
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout
            )
            if result.returncode == 0:
                return result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            log.debug("ipmitool error: %s", exc)
        return None

    def cpu_temperature(self) -> Optional[float]:
        """Return highest CPU temperature in Celsius."""
        output = self._run("sdr", "type", "Temperature")
        if not output:
            return None
        temps = []
        for line in output.splitlines():
            # "CPU Temp         | 3Ah | ok  |  3.1 | 45 degrees C"
            match = re.search(r"(\d+)\s+degrees C", line, re.IGNORECASE)
            if match and ("cpu" in line.lower() or "inlet" in line.lower() or "temp" in line.lower()):
                temps.append(float(match.group(1)))
        return max(temps) if temps else None

    def power_watts(self) -> Optional[float]:
        """Return current power draw in Watts."""
        output = self._run("dcmi", "power", "reading")
        if not output:
            # Fallback: sdr type Current
            output = self._run("sdr", "type", "Current")
            if not output:
                return None
        match = re.search(r"Instantaneous power reading:\s*(\d+)\s*Watts", output)
        if match:
            return float(match.group(1))
        # Try generic Watts pattern
        match = re.search(r"(\d+(?:\.\d+)?)\s*Watts", output)
        return float(match.group(1)) if match else None

    def is_available(self) -> bool:
        return self._run("mc", "info") is not None


# ---------------------------------------------------------------------------
# GPU collector (nvidia-smi)
# ---------------------------------------------------------------------------

class GPUCollector:
    """
    Collect GPU metrics via nvidia-smi.

    Requires: nvidia-smi in PATH (nvidia-smi comes with the CUDA driver)

    Query format: nvidia-smi --query-gpu=utilization.gpu,temperature.gpu
                             --format=csv,noheader,nounits
    """

    def collect(self) -> List[Dict[str, float]]:
        """Return list of GPU dicts, one per GPU."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,temperature.gpu,power.draw,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=5.0,
            )
            if result.returncode != 0:
                return []
            gpus = []
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    try:
                        gpus.append({
                            "index": int(parts[0]),
                            "utilization": float(parts[1]) / 100.0,
                            "temperature_c": float(parts[2]),
                            "power_w": float(parts[3]),
                            "mem_used_mib": float(parts[4]),
                            "mem_total_mib": float(parts[5]),
                        })
                    except ValueError:
                        pass
            return gpus
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []

    def is_available(self) -> bool:
        try:
            subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=3.0)
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False


# ---------------------------------------------------------------------------
# Slurm collector
# ---------------------------------------------------------------------------

class SlurmCollector:
    """
    Collect job metrics from Slurm.

    Requires: slurm-client (squeue, sinfo) on the controller node.
    The controller must be able to run squeue as the slurm user.

    Install: dnf install -y slurm-client
    """

    def node_jobs(self, node_id: str) -> List[str]:
        """Return list of job IDs running on this node."""
        try:
            result = subprocess.run(
                ["squeue", "-h", "-o", "%i", "-w", node_id],
                capture_output=True, text=True, timeout=5.0,
            )
            if result.returncode == 0:
                return [j.strip() for j in result.stdout.strip().splitlines() if j.strip()]
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            log.debug("squeue error: %s", exc)
        return []

    def queue_depth(self) -> int:
        """Return total pending jobs cluster-wide."""
        try:
            result = subprocess.run(
                ["squeue", "-h", "-t", "PENDING", "-o", "%i"],
                capture_output=True, text=True, timeout=5.0,
            )
            if result.returncode == 0:
                return len([l for l in result.stdout.strip().splitlines() if l.strip()])
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return 0

    def is_available(self) -> bool:
        try:
            subprocess.run(["squeue", "--version"], capture_output=True, timeout=3.0)
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False


# ---------------------------------------------------------------------------
# SSH remote collector (fallback when no node_exporter)
# ---------------------------------------------------------------------------

class SSHCollector:
    """
    Collect /proc metrics from a remote node over SSH.

    Requirements:
      - ssh key-based auth (no password prompt)
      - ssh config or known_hosts set up
      - python3 on the remote node

    The remote script is a single-line python3 one-liner to avoid
    needing any extra packages on remote nodes.
    """

    REMOTE_SCRIPT = """
import json, time, os, re

def cpu():
    def r():
        with open('/proc/stat') as f: p = f.readline().split()
        v = [int(x) for x in p[1:]]
        return v[3]+v[4], sum(v)
    i1,t1=r(); time.sleep(0.3); i2,t2=r()
    dt=t2-t1; return 0.0 if dt==0 else 1.0-(i2-i1)/dt

def mem():
    d={}
    with open('/proc/meminfo') as f:
        for l in f:
            p=l.split();
            if len(p)>=2: d[p[0].rstrip(':')]=int(p[1])
    t=d.get('MemTotal',1); a=d.get('MemAvailable',t)
    return 1.0-(a/t)

def net():
    iface=''; rx1=tx1=rx2=tx2=0
    with open('/proc/net/dev') as f:
        for l in f:
            if ':' in l:
                i,_=l.split(':',1)
                i=i.strip()
                if i!='lo': iface=i; break
    if not iface: return 0.0,0.0
    def rb(i):
        p='/sys/class/net/'+i+'/statistics/'
        try:
            rx=int(open(p+'rx_bytes').read()); tx=int(open(p+'tx_bytes').read())
            return rx,tx
        except: return 0,0
    rx1,tx1=rb(iface); time.sleep(0.5); rx2,tx2=rb(iface)
    return (rx2-rx1)*8/1e6/0.5, (tx2-tx1)*8/1e6/0.5

c=cpu(); m=mem(); rx,tx=net()
print(json.dumps({'cpu':round(c,4),'mem':round(m,4),'rx_mbps':round(max(0,rx),2),'tx_mbps':round(max(0,tx),2)}))
""".strip()

    def __init__(self, ssh_user: str = "root", ssh_key: Optional[str] = None,
                 timeout_sec: float = 8.0) -> None:
        self.ssh_user = ssh_user
        self.ssh_key = ssh_key
        self.timeout = timeout_sec

    def _ssh_cmd(self, host: str) -> List[str]:
        cmd = ["ssh", "-o", "StrictHostKeyChecking=no",
               "-o", "ConnectTimeout=5",
               "-o", "BatchMode=yes"]
        if self.ssh_key:
            cmd += ["-i", self.ssh_key]
        cmd.append(f"{self.ssh_user}@{host}")
        return cmd

    def collect(self, host: str) -> Optional[Dict[str, float]]:
        cmd = self._ssh_cmd(host) + ["python3", "-c", self.REMOTE_SCRIPT]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout
            )
            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout.strip())
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as exc:
            log.debug("SSH collect failed for %s: %s", host, exc)
        return None

    def is_reachable(self, host: str) -> bool:
        cmd = self._ssh_cmd(host) + ["echo", "ok"]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=5.0)
            return result.returncode == 0
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Real TelemetryCollector — orchestrates all sources
# ---------------------------------------------------------------------------

class RealTelemetryCollector:
    """
    Production telemetry collector for Rocky Linux nodes.

    Source priority per metric:
      CPU/Mem/Net : Prometheus node_exporter → SSH /proc → local /proc (self)
      Temperature : IPMI → node_exporter (hwmon) → /sys/class/thermal
      GPU         : nvidia-smi (local) or SSH nvidia-smi
      Jobs        : Slurm squeue
    """

    def __init__(self, config: Dict) -> None:
        self.ssh_user = config.get("ssh_user", "root")
        self.ssh_key = config.get("ssh_key_path")
        self.node_exporter_port = config.get("node_exporter_port", 9100)
        self.use_ipmi = config.get("use_ipmi", True)
        self.use_gpu = config.get("use_gpu", False)
        self.use_slurm = config.get("use_slurm", True)

        self._prometheus = PrometheusCollector()
        self._ipmi_cache: Dict[str, IPMICollector] = {}
        self._ssh = SSHCollector(ssh_user=self.ssh_user, ssh_key=self.ssh_key)
        self._gpu = GPUCollector()
        self._slurm = SlurmCollector()
        self._local_proc = ProcReader()

        # Delta tracking for Prometheus cumulative counters
        self._net_prev: Dict[str, Tuple[float, float, float]] = {}  # host → (rx, tx, t)

        log.info("RealTelemetryCollector: slurm=%s gpu=%s ipmi=%s node_exporter_port=%d",
                 self.use_slurm, self.use_gpu, self.use_ipmi, self.node_exporter_port)

    def collect(self, node: NodeState) -> NodeState:
        """
        Collect all metrics for a node and update node state in-place.
        Returns the updated node.
        """
        host = node.node_id   # node_id must be the hostname or IP

        # --- Try Prometheus first (fastest, most complete) ---
        scraped = self._prometheus.scrape(host, self.node_exporter_port)

        if scraped:
            node.cpu_usage = self._prometheus.extract_cpu_usage(scraped)
            node.memory_usage = self._prometheus.extract_memory_usage(scraped)
            node.network_rx_mbps, node.network_tx_mbps = self._net_delta(
                host, *self._prometheus.extract_network(scraped)
            )
            # hwmon temperature from node_exporter
            temp = self._prom_temperature(scraped)
            if temp is not None:
                node.temperature_c = temp
        else:
            # --- Fallback: SSH /proc reader ---
            ssh_data = self._ssh.collect(host)
            if ssh_data:
                node.cpu_usage = ssh_data.get("cpu", node.cpu_usage)
                node.memory_usage = ssh_data.get("mem", node.memory_usage)
                node.network_rx_mbps = ssh_data.get("rx_mbps", node.network_rx_mbps)
                node.network_tx_mbps = ssh_data.get("tx_mbps", node.network_tx_mbps)
            else:
                log.warning("No telemetry available for %s (prometheus + SSH both failed)", host)
                node.status = NodeStatus.DEGRADED

        # --- IPMI temperature / power (best source for hardware sensors) ---
        if self.use_ipmi:
            ipmi = self._get_ipmi(host)
            temp = ipmi.cpu_temperature()
            if temp is not None:
                node.temperature_c = temp
            power = ipmi.power_watts()
            if power is not None:
                node.power_watts = power   # store raw; aggregated at cluster level

        # --- GPU ---
        if self.use_gpu:
            gpus = self._gpu.collect()    # assumes nvidia-smi over SSH; extend as needed
            if gpus:
                avg_gpu = sum(g["utilization"] for g in gpus) / len(gpus)
                node.gpu_usage = avg_gpu

        # --- Slurm jobs ---
        if self.use_slurm:
            node.running_jobs = self._slurm.node_jobs(host)
            node.job_queue_depth = self._slurm.queue_depth()

        node.timestamp = time.time()
        return node

    def _get_ipmi(self, host: str) -> IPMICollector:
        if host not in self._ipmi_cache:
            self._ipmi_cache[host] = IPMICollector(
                bmc_host=host,
                bmc_user="admin",
                bmc_password="",      # read from config/vault in production
                use_local=(host in ("localhost", "127.0.0.1")),
            )
        return self._ipmi_cache[host]

    def _net_delta(self, host: str, rx_bytes: float, tx_bytes: float) -> Tuple[float, float]:
        """Convert cumulative Prometheus byte counters to Mbps."""
        now = time.time()
        prev = self._net_prev.get(host)
        self._net_prev[host] = (rx_bytes, tx_bytes, now)
        if prev is None:
            return 0.0, 0.0
        prx, ptx, pt = prev
        dt = now - pt
        if dt <= 0:
            return 0.0, 0.0
        rx_mbps = max(0.0, (rx_bytes - prx) * 8 / 1_000_000 / dt)
        tx_mbps = max(0.0, (tx_bytes - ptx) * 8 / 1_000_000 / dt)
        return rx_mbps, tx_mbps

    def _prom_temperature(self, scraped: Dict[str, float]) -> Optional[float]:
        """Extract CPU temperature from node_exporter hwmon metrics."""
        temps = [
            v for k, v in scraped.items()
            if "node_hwmon_temp_celsius" in k and v > 0
        ]
        return max(temps) if temps else None


# ---------------------------------------------------------------------------
# ImmuneMonitor — unchanged interface, real collector underneath
# ---------------------------------------------------------------------------

class ImmuneMonitor:
    """
    MAPE-K Monitor phase — production Rocky Linux implementation.

    Runs on a background thread, collecting telemetry from all managed
    nodes every `interval` seconds using the RealTelemetryCollector.
    """

    def __init__(
        self,
        state: ClusterState,
        collector_config: Dict,
        interval_sec: float = 10.0,
        health_threshold: float = 0.6,
    ) -> None:
        self.state = state
        self.interval = interval_sec
        self.health_threshold = health_threshold
        self.collector = RealTelemetryCollector(collector_config)
        self._prev_health: Dict[str, float] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="monitor")
        self._thread.start()
        log.info("ImmuneMonitor started (real mode, interval=%.1fs)", self.interval)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        log.info("ImmuneMonitor stopped")

    def collect_once(self) -> None:
        self._collect_all()

    def _loop(self) -> None:
        while self._running:
            try:
                self._collect_all()
            except Exception as exc:
                log.error("Monitor loop error: %s", exc, exc_info=True)
            time.sleep(self.interval)

    def _collect_all(self) -> None:
        self.state.timestamp = time.time()

        for node_id, node in list(self.state.nodes.items()):
            if node.status in (NodeStatus.QUARANTINED, NodeStatus.REIMAGING):
                continue
            try:
                self.collector.collect(node)
            except Exception as exc:
                log.error("Telemetry failed for %s: %s", node_id, exc)
                continue

            prev_h = self._prev_health.get(node_id, node.health_score)
            node.health_score = _compute_health_score(node)
            self._prev_health[node_id] = node.health_score

            if prev_h >= self.health_threshold and node.health_score < self.health_threshold:
                log.warning("Node %s health degraded → %.2f", node_id, node.health_score)
                bus.emit_simple(EventType.HEALTH_DEGRADED, "monitor", payload=node)
                metrics.failure_events.inc()

            elif prev_h < self.health_threshold and node.health_score >= self.health_threshold:
                log.info("Node %s health recovered → %.2f", node_id, node.health_score)
                bus.emit_simple(EventType.HEALTH_RECOVERED, "monitor", payload=node)
                metrics.recovery_events.inc()

        self.state.aggregate()
        self.state.compute_lyapunov()
        metrics.lyapunov_V.set(self.state.lyapunov_value)
        metrics.objective_J.set(self.state.compute_objective())
        bus.emit_simple(EventType.TELEMETRY_COLLECTED, "monitor", payload=self.state.snapshot())
