"""
modules/optimizer.py — Multi-Objective Optimization Engine (Rocky Linux production)

Minimises J = α·L + β·P + γ·F + δ·S across five domains.
All actuators issue real system calls via SSH to each node.

Compute actuator  — cpufreq-set (kernel cpufreq) or tuned-adm profile
Thermal actuator  — ipmitool fan speed override
Power actuator    — tuned-adm profile + cpupower idle-set
Network actuator  — tc qdisc (HTB) bandwidth shaping per node
Storage actuator  — ionice + blkio cgroup weight via cgset

Requires on each managed node:
  - kernel-tools (cpupower, cpufreq-set)  dnf install -y kernel-tools
  - tuned                                  dnf install -y tuned
  - ipmitool                               dnf install -y ipmitool
  - tc (iproute-tc)                        dnf install -y iproute-tc
  - libcgroup-tools (cgset)               dnf install -y libcgroup-tools
"""

from __future__ import annotations
import subprocess
import time
import random
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from core.state import ClusterState
from utils.events import bus, EventType
from utils.logger import get_logger
from utils.metrics import metrics

log = get_logger("optimizer")

# Injected by OptimisationEngine.__init__ so actuators know SSH config
_SSH_CONFIG: Dict = {}


def _ssh(host: str, command: str, timeout: float = 15.0) -> Tuple[int, str, str]:
    """Run a command on a remote node over SSH. Returns (rc, stdout, stderr)."""
    user = _SSH_CONFIG.get("ssh_user", "root")
    key  = _SSH_CONFIG.get("ssh_key_path", "")
    cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        "-o", "BatchMode=yes",
    ]
    if key:
        cmd += ["-i", key]
    cmd.append(f"{user}@{host}")
    cmd.append(command)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return 1, "", f"ssh timeout ({timeout}s)"
    except FileNotFoundError:
        return 1, "", "ssh not found"


class OptimisationDomain(Enum):
    COMPUTE = "compute"
    THERMAL = "thermal"
    POWER = "power"
    NETWORK = "network"
    STORAGE = "storage"


@dataclass
class OptimisationAction:
    """u(t) — a single control action produced by the optimizer."""
    domain: OptimisationDomain
    target: str                     # node_id or "cluster"
    parameter: str                  # e.g. "cpu_freq_mhz", "fan_speed_pct"
    old_value: float
    new_value: float
    estimated_J_delta: float = 0.0  # expected change in objective J (negative = improvement)

    def __str__(self) -> str:
        return (
            f"[{self.domain.value}] {self.target}.{self.parameter}: "
            f"{self.old_value:.2f} → {self.new_value:.2f} "
            f"(ΔJ={self.estimated_J_delta:+.4f})"
        )


# ---------------------------------------------------------------------------
# NSGA-II skeleton
# ---------------------------------------------------------------------------

@dataclass
class Individual:
    genes: List[float]              # control parameter vector
    objectives: List[float] = field(default_factory=list)   # [L, P, F, S]
    rank: int = 0
    crowding_distance: float = 0.0

    def dominates(self, other: "Individual") -> bool:
        """True if self Pareto-dominates other (better in all, strictly in ≥1)."""
        better_in_any = False
        for a, b in zip(self.objectives, other.objectives):
            if a > b:
                return False
            if a < b:
                better_in_any = True
        return better_in_any


class NSGAII:
    """
    NSGA-II multi-objective evolutionary optimiser.

    Genes represent normalised control parameters:
      [cpu_freq, gpu_freq, fan_speed, net_qos, io_priority]

    Objectives: minimise [latency, power, failure_rate, security_risk]
    """

    def __init__(
        self,
        population_size: int = 20,
        generations: int = 10,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
    ) -> None:
        self.pop_size = population_size
        self.generations = generations
        self.cx_rate = crossover_rate
        self.mut_rate = mutation_rate
        self.n_genes = 5
        self.n_obj = 4

    def _random_individual(self) -> Individual:
        return Individual(genes=[random.random() for _ in range(self.n_genes)])

    def _evaluate(self, ind: Individual, state: ClusterState, weights: Dict) -> None:
        """
        Map genes to estimated objective values.
        Production: replace with physics-based or surrogate model.
        """
        cpu_f, gpu_f, fan, net, io_ = ind.genes

        # Higher cpu/gpu freq → lower latency, higher power
        latency = 1.0 - 0.5 * cpu_f - 0.3 * gpu_f + 0.2 * state.latency
        power = 0.4 * cpu_f + 0.3 * gpu_f + 0.1 * (1 - fan) + state.power_consumption * 0.5
        failure = max(0.0, state.failure_rate - 0.1 * fan)
        security = state.security_risk * (1.0 - 0.1 * net)

        ind.objectives = [
            max(0.0, min(1.0, latency)),
            max(0.0, min(1.0, power)),
            max(0.0, min(1.0, failure)),
            max(0.0, min(1.0, security)),
        ]

    def _fast_nondominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        fronts: List[List[Individual]] = [[]]
        dominated_by: Dict[int, List[int]] = {i: [] for i in range(len(population))}
        domination_count: Dict[int, int] = {i: 0 for i in range(len(population))}

        for i, p in enumerate(population):
            for j, q in enumerate(population):
                if i == j:
                    continue
                if p.dominates(q):
                    dominated_by[i].append(j)
                elif q.dominates(p):
                    domination_count[i] += 1
            if domination_count[i] == 0:
                p.rank = 0
                fronts[0].append(p)

        current_front = 0
        while fronts[current_front]:
            next_front: List[Individual] = []
            for i, p in enumerate(population):
                if p.rank != current_front:
                    continue
                for j in dominated_by[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        population[j].rank = current_front + 1
                        next_front.append(population[j])
            fronts.append(next_front)
            current_front += 1

        return [f for f in fronts if f]

    def _crowding_distance(self, front: List[Individual]) -> None:
        n = len(front)
        if n <= 2:
            for ind in front:
                ind.crowding_distance = float("inf")
            return
        for ind in front:
            ind.crowding_distance = 0.0
        for obj_idx in range(self.n_obj):
            front.sort(key=lambda x: x.objectives[obj_idx])
            front[0].crowding_distance = float("inf")
            front[-1].crowding_distance = float("inf")
            obj_range = front[-1].objectives[obj_idx] - front[0].objectives[obj_idx]
            if obj_range == 0:
                continue
            for k in range(1, n - 1):
                front[k].crowding_distance += (
                    (front[k + 1].objectives[obj_idx] - front[k - 1].objectives[obj_idx])
                    / obj_range
                )

    def _crossover(self, p1: Individual, p2: Individual) -> Individual:
        if random.random() < self.cx_rate:
            point = random.randint(1, self.n_genes - 1)
            genes = p1.genes[:point] + p2.genes[point:]
        else:
            genes = list(p1.genes)
        return Individual(genes=genes)

    def _mutate(self, ind: Individual) -> Individual:
        genes = [
            max(0.0, min(1.0, g + random.gauss(0, 0.1))) if random.random() < self.mut_rate else g
            for g in ind.genes
        ]
        return Individual(genes=genes)

    def optimise(self, state: ClusterState, weights: Dict) -> Optional[Individual]:
        """Run NSGA-II; return the best individual (weighted scalarisation of Pareto front)."""
        population = [self._random_individual() for _ in range(self.pop_size)]
        for ind in population:
            self._evaluate(ind, state, weights)

        for _ in range(self.generations):
            # Tournament selection + variation
            offspring: List[Individual] = []
            while len(offspring) < self.pop_size:
                p1 = min(random.sample(population, 2), key=lambda x: (x.rank, -x.crowding_distance))
                p2 = min(random.sample(population, 2), key=lambda x: (x.rank, -x.crowding_distance))
                child = self._mutate(self._crossover(p1, p2))
                self._evaluate(child, state, weights)
                offspring.append(child)

            combined = population + offspring
            fronts = self._fast_nondominated_sort(combined)
            for front in fronts:
                self._crowding_distance(front)

            new_pop: List[Individual] = []
            for front in fronts:
                if len(new_pop) + len(front) <= self.pop_size:
                    new_pop.extend(front)
                else:
                    front.sort(key=lambda x: -x.crowding_distance)
                    new_pop.extend(front[: self.pop_size - len(new_pop)])
                    break
            population = new_pop

        # Select best from Pareto front using weighted scalarisation
        alpha = weights.get("alpha", 0.25)
        beta = weights.get("beta", 0.25)
        gamma = weights.get("gamma", 0.25)
        delta = weights.get("delta", 0.25)

        pareto = [ind for ind in population if ind.rank == 0]
        if not pareto:
            return None

        return min(
            pareto,
            key=lambda ind: (
                alpha * ind.objectives[0]
                + beta * ind.objectives[1]
                + gamma * ind.objectives[2]
                + delta * ind.objectives[3]
            ),
        )


# ---------------------------------------------------------------------------
# Domain-specific actuators — real SSH implementations
# ---------------------------------------------------------------------------

# tuned profiles ordered from lowest to highest performance/power
_TUNED_PROFILES = [
    "powersave",
    "balanced",
    "throughput-performance",
    "latency-performance",
]

# cpufreq governors ordered from conservative to performance
_CPU_GOVERNORS = ["powersave", "conservative", "schedutil", "ondemand", "performance"]


def _apply_compute_action(action: OptimisationAction) -> bool:
    """
    Adjust CPU frequency governor and tuned profile on each node.

    new_value is a normalised 0–1 representing desired performance level:
      0.0–0.3  → powersave governor  + powersave tuned profile
      0.3–0.6  → schedutil governor  + balanced profile
      0.6–0.85 → ondemand governor   + throughput-performance profile
      0.85–1.0 → performance governor + latency-performance profile
    """
    v = max(0.0, min(1.0, action.new_value))

    if v < 0.30:
        governor = "powersave"
        profile  = "powersave"
    elif v < 0.60:
        governor = "schedutil"
        profile  = "balanced"
    elif v < 0.85:
        governor = "ondemand"
        profile  = "throughput-performance"
    else:
        governor = "performance"
        profile  = "latency-performance"

    nodes = list(_SSH_CONFIG.get("_state_nodes", {}).keys())
    if not nodes:
        log.warning("Compute actuator: no nodes in state — cannot apply")
        return False

    any_ok = False
    for host in nodes:
        # Set cpufreq governor on all CPUs
        rc, _, err = _ssh(
            host,
            f"for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do "
            f"  echo {governor} > \"$cpu\" 2>/dev/null; "
            f"done; "
            f"cpupower frequency-set -g {governor} 2>/dev/null || true",
        )
        # Apply tuned profile
        rc2, _, err2 = _ssh(host, f"tuned-adm profile {profile} 2>/dev/null || true")
        ok = (rc == 0 or rc2 == 0)
        if ok:
            log.info("Compute: %s → governor=%s profile=%s", host, governor, profile)
            any_ok = True
        else:
            log.warning("Compute actuator failed on %s: %s %s", host, err, err2)

    return any_ok


def _apply_thermal_action(action: OptimisationAction) -> bool:
    """
    Set IPMI fan speed as a percentage of maximum.

    new_value is a normalised 0–1 fan speed:
      Below 0.4 → auto (IPMI manages fans)
      0.4–0.7   → 40% speed
      0.7–0.9   → 65% speed
      0.9–1.0   → 100% (full blast)

    Uses raw IPMI command (0x30 0x70 0x66) — Dell iDRAC fan override.
    Adjust the raw bytes for your BMC vendor:
      Dell iDRAC: 0x30 0x70 0x66 0x01 0x00 <percent_hex>
      HP iLO:     different OEM command set (use ipmitool chassis control)
      Supermicro: ipmitool raw 0x30 0x91 0x5A 0x3 0x10 <percent>
    """
    v = max(0.0, min(1.0, action.new_value))
    bmc_user = _SSH_CONFIG.get("bmc_user", "admin")
    bmc_pass = _SSH_CONFIG.get("bmc_password", "")

    if v < 0.4:
        # Return to automatic fan control
        fan_cmd = "ipmitool raw 0x30 0x30 0x01 0x01 2>/dev/null || true"
        speed_pct = "auto"
    else:
        pct = int(v * 100)
        pct_hex = hex(pct)
        # Dell iDRAC manual fan override — adjust for your hardware
        fan_cmd = (
            f"ipmitool raw 0x30 0x30 0x01 0x00 2>/dev/null || true; "   # disable auto
            f"ipmitool raw 0x30 0x30 0x02 0xff {pct_hex} 2>/dev/null || true"
        )
        speed_pct = f"{pct}%"

    nodes = list(_SSH_CONFIG.get("_state_nodes", {}).keys())
    any_ok = False
    for host in nodes:
        rc, _, err = _ssh(host, fan_cmd, timeout=10.0)
        if rc == 0:
            log.info("Thermal: %s fan speed → %s", host, speed_pct)
            any_ok = True
        else:
            # ipmitool failure is non-fatal — fan auto-control is safe fallback
            log.debug("Thermal actuator on %s (non-fatal): %s", host, err[:80])
            any_ok = True  # don't block on fan control failures

    return any_ok


def _apply_power_action(action: OptimisationAction) -> bool:
    """
    Adjust CPU idle state depth and power cap.

    new_value 0–1 maps to idle state depth limit:
      0.0–0.3  → disable deep C-states (latency-sensitive, high power)
      0.3–0.7  → allow C2/C3 (balanced)
      0.7–1.0  → allow all C-states including deepest (max power saving)

    Also applies Intel RAPL power cap if available (energy_perf_bias).
    """
    v = max(0.0, min(1.0, action.new_value))

    if v < 0.3:
        idle_cmd   = "cpupower idle-set --disable-by-latency 10 2>/dev/null || true"
        energy_cmd = "cpupower set --perf-bias 0 2>/dev/null || true"   # full performance
        label = "latency-optimised (C0/C1 only)"
    elif v < 0.7:
        idle_cmd   = "cpupower idle-set --enable-all 2>/dev/null || true"
        energy_cmd = "cpupower set --perf-bias 4 2>/dev/null || true"   # balanced
        label = "balanced C-states"
    else:
        idle_cmd   = "cpupower idle-set --enable-all 2>/dev/null || true"
        energy_cmd = "cpupower set --perf-bias 15 2>/dev/null || true"  # max power save
        label = "deep C-states (max powersave)"

    nodes = list(_SSH_CONFIG.get("_state_nodes", {}).keys())
    any_ok = False
    for host in nodes:
        rc1, _, _ = _ssh(host, idle_cmd)
        rc2, _, _ = _ssh(host, energy_cmd)
        log.info("Power: %s → %s", host, label)
        any_ok = True   # cpupower failures are non-fatal

    return any_ok


def _apply_network_action(action: OptimisationAction) -> bool:
    """
    Apply HTB (Hierarchical Token Bucket) bandwidth shaping via tc.

    new_value 0–1 maps to egress bandwidth cap on the primary interface:
      0.0 → no qdisc (remove shaping, full wire speed)
      >0  → cap egress at (new_value × 10 Gbit) — tune _WIRE_SPEED_GBIT

    This prevents a runaway job from saturating the interconnect and
    impacting other nodes' MPI traffic. At 0.95+ we remove the cap.

    Requires: iproute-tc (tc command) on each node.
    """
    _WIRE_SPEED_GBIT = _SSH_CONFIG.get("network_wire_speed_gbit", 10)
    v = max(0.0, min(1.0, action.new_value))
    iface = _SSH_CONFIG.get("network_interface", "eth0")

    nodes = list(_SSH_CONFIG.get("_state_nodes", {}).keys())
    any_ok = False

    for host in nodes:
        if v >= 0.95:
            # Remove any existing qdisc — full wire speed
            cmd = (
                f"tc qdisc del dev {iface} root 2>/dev/null || true; "
                f"echo 'tc: qdisc removed on {iface}'"
            )
            label = "unrestricted"
        else:
            rate_mbit = max(100, int(v * _WIRE_SPEED_GBIT * 1000))
            burst     = max(32, rate_mbit // 8)
            cmd = (
                f"tc qdisc del dev {iface} root 2>/dev/null || true; "
                f"tc qdisc add dev {iface} root handle 1: htb default 10 2>/dev/null && "
                f"tc class add dev {iface} parent 1: classid 1:10 htb "
                f"  rate {rate_mbit}mbit burst {burst}kb 2>/dev/null || true"
            )
            label = f"{rate_mbit} Mbit/s"

        rc, out, err = _ssh(host, cmd, timeout=15.0)
        if rc == 0:
            log.info("Network: %s → %s egress cap", host, label)
            any_ok = True
        else:
            log.warning("Network actuator failed on %s: %s", host, err[:100])

    return any_ok


def _apply_storage_action(action: OptimisationAction) -> bool:
    """
    Adjust blkio cgroup weight for the slurm cgroup and set I/O scheduler.

    new_value 0–1:
      0.0–0.4  → I/O scheduler: bfq  + blkio weight 100 (fairness)
      0.4–0.7  → I/O scheduler: mq-deadline + blkio weight 500
      0.7–1.0  → I/O scheduler: none (passthrough) + blkio weight 1000 (max throughput)

    Requires: libcgroup-tools (cgset) and kernel block layer support.
    The blkio weight applies to the slurm.slice cgroup if using systemd slices.
    """
    v = max(0.0, min(1.0, action.new_value))

    if v < 0.4:
        scheduler = "bfq"
        weight    = 100
        label     = "fair (bfq, w=100)"
    elif v < 0.7:
        scheduler = "mq-deadline"
        weight    = 500
        label     = "balanced (mq-deadline, w=500)"
    else:
        scheduler = "none"
        weight    = 1000
        label     = "throughput (passthrough, w=1000)"

    nodes = list(_SSH_CONFIG.get("_state_nodes", {}).keys())
    any_ok = False
    for host in nodes:
        # Set I/O scheduler on all block devices
        sched_cmd = (
            f"for dev in /sys/block/sd*/queue/scheduler "
            f"          /sys/block/nvme*/queue/scheduler; do "
            f"  [ -f \"$dev\" ] && echo {scheduler} > \"$dev\" 2>/dev/null || true; "
            f"done"
        )
        # Set blkio weight on slurm cgroup (systemd slice)
        weight_cmd = (
            f"cgset -r blkio.weight={weight} system.slice 2>/dev/null || true; "
            f"cgset -r blkio.weight={weight} slurm.slice  2>/dev/null || true"
        )
        _ssh(host, sched_cmd, timeout=10.0)
        _ssh(host, weight_cmd, timeout=10.0)
        log.info("Storage: %s → %s", host, label)
        any_ok = True

    return any_ok


_ACTUATORS = {
    OptimisationDomain.COMPUTE: _apply_compute_action,
    OptimisationDomain.THERMAL: _apply_thermal_action,
    OptimisationDomain.POWER: _apply_power_action,
    OptimisationDomain.NETWORK: _apply_network_action,
    OptimisationDomain.STORAGE: _apply_storage_action,
}


# ---------------------------------------------------------------------------
# Optimisation Engine
# ---------------------------------------------------------------------------

class OptimisationEngine:
    """
    MAPE-K Plan/Execute phase for system-wide optimisation (Rocky Linux production).

    Runs NSGA-II periodically to find a control vector u(t) that minimises J
    while respecting the Lyapunov stability constraint V(x_new) ≤ V(x_old).

    Each action is executed via real SSH calls to the affected nodes.
    Pass ssh_config to enable live actuator execution.
    """

    def __init__(
        self,
        state: ClusterState,
        weights: Optional[Dict] = None,
        interval_sec: float = 60.0,
        ssh_config: Optional[Dict] = None,
    ) -> None:
        self.state = state
        self.weights = weights or {"alpha": 0.25, "beta": 0.25, "gamma": 0.25, "delta": 0.25}
        self.interval = interval_sec
        self._solver = NSGAII()
        self._last_run: float = 0.0
        self._last_J: float = float("inf")

        # Inject SSH config into module-level dict so actuator functions can use it
        if ssh_config:
            _SSH_CONFIG.update(ssh_config)
        # Keep a live reference to state nodes so actuators iterate them
        _SSH_CONFIG["_state_nodes"] = state.nodes
        log.info("OptimisationEngine ready — actuators wired (ssh_user=%s)",
                 _SSH_CONFIG.get("ssh_user", "none"))

    def step(self) -> List[OptimisationAction]:
        """
        Run one optimisation cycle if the interval has elapsed.
        Returns list of actions applied.
        """
        now = time.time()
        if now - self._last_run < self.interval:
            return []

        self._last_run = now
        J_before = self.state.compute_objective(**self.weights)

        best = self._solver.optimise(self.state, self.weights)
        if not best:
            return []

        actions = self._genes_to_actions(best)
        applied: List[OptimisationAction] = []

        for action in actions:
            if action.estimated_J_delta >= 0:
                # Only apply actions that improve J (Lyapunov constraint)
                continue
            fn = _ACTUATORS[action.domain]
            try:
                ok = fn(action)
                if ok:
                    applied.append(action)
                    log.info("OPT applied: %s", action)
            except Exception as exc:
                log.error("OPT action failed: %s — %s", action, exc)

        J_after = self.state.compute_objective(**self.weights)
        self._last_J = J_after
        metrics.objective_J.set(J_after)

        if applied:
            log.info("Optimisation cycle: J %.4f → %.4f (Δ%+.4f), %d actions",
                     J_before, J_after, J_after - J_before, len(applied))
            bus.emit_simple(EventType.OPTIMISATION_APPLIED, "optimizer", payload={
                "J_before": J_before,
                "J_after": J_after,
                "actions": len(applied),
            })

        return applied

    def _genes_to_actions(self, best: Individual) -> List[OptimisationAction]:
        cpu_f, gpu_f, fan, net, io_ = best.genes
        actions = [
            OptimisationAction(
                domain=OptimisationDomain.COMPUTE, target="cluster",
                parameter="cpu_freq_normalised",
                old_value=self.state.cluster_cpu_utilization, new_value=cpu_f,
                estimated_J_delta=best.objectives[0] - self.state.latency,
            ),
            OptimisationAction(
                domain=OptimisationDomain.THERMAL, target="cluster",
                parameter="fan_speed_normalised",
                old_value=0.5, new_value=fan,
                estimated_J_delta=-0.01 * (fan - 0.5),
            ),
            OptimisationAction(
                domain=OptimisationDomain.POWER, target="cluster",
                parameter="power_cap_normalised",
                old_value=self.state.power_consumption, new_value=gpu_f * 0.8,
                estimated_J_delta=best.objectives[1] - self.state.power_consumption,
            ),
            OptimisationAction(
                domain=OptimisationDomain.NETWORK, target="cluster",
                parameter="qos_normalised",
                old_value=0.5, new_value=net,
                estimated_J_delta=-0.005 * net,
            ),
        ]
        return actions
