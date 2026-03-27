"""
modules/scheduler.py — Adaptive Scheduler with Slurm Bridge (Rocky Linux production)

ImmuneHPC+ acts as a meta-scheduler on top of Slurm:
  - Reads live job state from squeue every tick
  - Adjusts node weights in Slurm partitions based on health scores
  - Drains / resumes nodes in sync with the quarantine layer
  - Rebalances partition membership when a node becomes degraded
  - Updates job priorities based on cluster health (backs off heavy jobs
    when failure_rate is high, accelerates when cluster is healthy)

The internal Job model tracks Slurm job IDs so every action maps to a
real squeue/scontrol operation.

Requires on controller: slurm-client (squeue, scontrol, sacct)
"""

from __future__ import annotations
import re
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from core.state import ClusterState, NodeState, NodeStatus
from utils.events import bus, EventType
from utils.logger import get_logger
from utils.metrics import metrics

log = get_logger("scheduler")

# Default Slurm partition managed by ImmuneHPC+
_DEFAULT_PARTITION = "compute"


def _slurm(args: List[str], timeout: float = 10.0) -> Tuple[int, str, str]:
    """Run a Slurm CLI command. Returns (rc, stdout, stderr)."""
    try:
        r = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout
        )
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return 1, "", f"timeout ({timeout}s)"
    except FileNotFoundError:
        return 1, "", f"{args[0]} not found (install slurm-client)"


class JobState(Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    job_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    slurm_job_id: Optional[str] = None      # actual Slurm job ID once submitted
    name: str = "unnamed"
    cpu_cores: int = 1
    gpu_count: int = 0
    memory_gb: float = 1.0
    wall_time_sec: float = 3600.0
    priority: int = 10
    state: JobState = JobState.PENDING
    assigned_node: Optional[str] = None
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    @property
    def wait_time(self) -> float:
        return (self.started_at or time.time()) - self.submitted_at

    @property
    def completion_time(self) -> Optional[float]:
        if self.completed_at and self.started_at:
            return self.completed_at - self.started_at
        return None


class SlurmBridge:
    """
    Thin wrapper around Slurm CLI tools.
    All methods are non-fatal — Slurm unavailability never crashes the controller.
    """

    def is_available(self) -> bool:
        rc, _, _ = _slurm(["squeue", "--version"])
        return rc == 0

    # ── Node management ───────────────────────────────────────────────

    def drain_node(self, node_id: str, reason: str) -> bool:
        rc, _, err = _slurm([
            "scontrol", "update", f"NodeName={node_id}",
            "State=DRAIN", f"Reason=ImmuneHPC+: {reason[:60]}",
        ])
        if rc == 0:
            log.info("Slurm: drained %s", node_id)
        else:
            log.debug("Slurm drain %s: %s", node_id, err[:80])
        return rc == 0

    def resume_node(self, node_id: str) -> bool:
        rc, _, err = _slurm([
            "scontrol", "update", f"NodeName={node_id}", "State=RESUME",
        ])
        if rc == 0:
            log.info("Slurm: resumed %s", node_id)
        else:
            log.debug("Slurm resume %s: %s", node_id, err[:80])
        return rc == 0

    def set_node_weight(self, node_id: str, weight: int) -> bool:
        """
        Set Slurm node weight (scheduling priority within partition).
        Higher weight = less preferred. weight=1 = most preferred.
        Weight is 1–65535.
        """
        w = max(1, min(65535, weight))
        rc, _, err = _slurm([
            "scontrol", "update", f"NodeName={node_id}", f"Weight={w}",
        ])
        if rc == 0:
            log.debug("Slurm: node %s weight → %d", node_id, w)
        return rc == 0

    # ── Job management ────────────────────────────────────────────────

    def running_jobs_on_node(self, node_id: str) -> List[str]:
        """Return list of running Slurm job IDs on a node."""
        rc, out, _ = _slurm([
            "squeue", "-h", "-o", "%i", "-w", node_id, "-t", "RUNNING",
        ])
        if rc != 0 or not out:
            return []
        return [j.strip() for j in out.splitlines() if j.strip()]

    def pending_jobs(self) -> List[Dict]:
        """Return list of dicts with pending job details."""
        rc, out, _ = _slurm([
            "squeue", "-h", "-t", "PENDING",
            "-o", "%i|%j|%C|%m|%l|%Q",
        ])
        if rc != 0 or not out:
            return []
        jobs = []
        for line in out.splitlines():
            parts = line.split("|")
            if len(parts) >= 6:
                try:
                    jobs.append({
                        "job_id":    parts[0].strip(),
                        "name":      parts[1].strip(),
                        "cpus":      int(parts[2].strip()),
                        "mem_mb":    self._parse_mem(parts[3].strip()),
                        "timelimit": parts[4].strip(),
                        "priority":  int(parts[5].strip()),
                    })
                except (ValueError, IndexError):
                    pass
        return jobs

    def queue_depth(self) -> int:
        rc, out, _ = _slurm(["squeue", "-h", "-t", "PENDING", "-o", "%i"])
        if rc != 0:
            return 0
        return len([l for l in out.splitlines() if l.strip()])

    def update_job_priority(self, slurm_job_id: str, priority: int) -> bool:
        """Adjust Slurm job priority (AdminComment field used as hint)."""
        rc, _, _ = _slurm([
            "scontrol", "update", f"JobId={slurm_job_id}",
            f"Priority={max(1, priority)}",
        ])
        return rc == 0

    def set_partition_weight(self, partition: str, priority_factor: int) -> bool:
        rc, _, _ = _slurm([
            "scontrol", "update", f"PartitionName={partition}",
            f"PriorityJobFactor={priority_factor}",
        ])
        return rc == 0

    # ── Node state query ──────────────────────────────────────────────

    def node_slurm_state(self, node_id: str) -> Optional[str]:
        rc, out, _ = _slurm([
            "sinfo", "-h", "-n", node_id, "-o", "%T",
        ])
        return out.strip() if rc == 0 else None

    def _parse_mem(self, mem_str: str) -> int:
        """Parse Slurm memory string (e.g. '4096M', '2G') to MB."""
        mem_str = mem_str.upper()
        try:
            if mem_str.endswith("G"):
                return int(mem_str[:-1]) * 1024
            if mem_str.endswith("T"):
                return int(mem_str[:-1]) * 1024 * 1024
            if mem_str.endswith("M"):
                return int(mem_str[:-1])
            return int(mem_str)
        except ValueError:
            return 0


class AdaptiveScheduler:
    """
    Meta-scheduler that drives Slurm via health-weighted node placement.

    Each MAPE-K tick:
      1. Sync node state from Slurm (sinfo → running jobs per node)
      2. Recompute Slurm node weights based on H_i scores
      3. Drain unhealthy nodes that are still active in Slurm
      4. Resume nodes that have recovered
      5. Adjust partition priority factor based on cluster failure_rate
      6. Update internal job model from squeue
    """

    def __init__(
        self,
        state: ClusterState,
        partition: str = _DEFAULT_PARTITION,
    ) -> None:
        self.state     = state
        self.partition = partition
        self._slurm    = SlurmBridge()
        self._queue: List[Job] = []
        self._running: Dict[str, Job] = {}
        self._completed: List[Job] = []
        self._node_completion_rate: Dict[str, float] = {}
        self._slurm_available: Optional[bool] = None

        # Subscribe to node events so Slurm stays in sync
        bus.subscribe(EventType.NODE_QUARANTINED, self._on_quarantined)
        bus.subscribe(EventType.NODE_RELEASED,    self._on_released)
        bus.subscribe(EventType.HEALTH_DEGRADED,  self._on_health_degraded)
        bus.subscribe(EventType.HEALTH_RECOVERED, self._on_health_recovered)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, job: Job) -> str:
        self._queue.append(job)
        log.info("Job submitted: %s (cpu=%d gpu=%d prio=%d)",
                 job.job_id, job.cpu_cores, job.gpu_count, job.priority)
        return job.job_id

    def tick(self) -> None:
        """Called each MAPE-K tick."""
        # Lazy Slurm check — only probe once
        if self._slurm_available is None:
            self._slurm_available = self._slurm.is_available()
            if self._slurm_available:
                log.info("Slurm bridge: ACTIVE (squeue available)")
            else:
                log.warning("Slurm bridge: INACTIVE (squeue not found) — "
                            "internal placement model only")

        if self._slurm_available:
            self._sync_from_slurm()
            self._update_node_weights()
            self._tune_partition_priority()
        else:
            self._schedule_internal()

        self._complete_finished_jobs()

    def cancel(self, job_id: str) -> bool:
        job = self._running.pop(job_id, None)
        if not job:
            return False
        job.state = JobState.CANCELLED
        self._completed.append(job)
        if job.slurm_job_id:
            _slurm(["scancel", job.slurm_job_id])
        return True

    @property
    def queue_depth(self) -> int:
        if self._slurm_available:
            return self._slurm.queue_depth()
        return len(self._queue)

    @property
    def running_count(self) -> int:
        return len(self._running)

    # ------------------------------------------------------------------
    # Slurm sync
    # ------------------------------------------------------------------

    def _sync_from_slurm(self) -> None:
        """Pull running job lists from Slurm into node state."""
        for node_id, node in self.state.nodes.items():
            if node.status == NodeStatus.QUARANTINED:
                continue
            jobs = self._slurm.running_jobs_on_node(node_id)
            node.running_jobs = jobs

        # Update job queue depth from squeue
        for node in self.state.nodes.values():
            node.job_queue_depth = self._slurm.queue_depth()
            break   # same value cluster-wide

    def _update_node_weights(self) -> None:
        """
        Map health score H_i → Slurm node weight.

        Health 1.0 → weight 1   (most preferred)
        Health 0.6 → weight 100
        Health 0.0 → weight 65535 (least preferred / effectively excluded)

        Formula: weight = int((1 - H_i)^2 * 65534) + 1
        """
        for node_id, node in self.state.nodes.items():
            if node.status in (NodeStatus.QUARANTINED, NodeStatus.REIMAGING,
                               NodeStatus.OFFLINE):
                continue
            h = max(0.0, min(1.0, node.health_score))
            weight = int((1.0 - h) ** 2 * 65534) + 1
            self._slurm.set_node_weight(node_id, weight)

    def _tune_partition_priority(self) -> None:
        """
        Scale the partition's PriorityJobFactor based on cluster health.

        When failure_rate > 0.3 we reduce the priority factor so Slurm
        becomes more conservative about starting new jobs on marginal nodes.
        """
        fr = self.state.failure_rate
        if fr < 0.1:
            factor = 1000   # normal
        elif fr < 0.3:
            factor = 500    # cautious
        else:
            factor = 100    # very cautious — cluster struggling

        self._slurm.set_partition_weight(self.partition, factor)

    # ------------------------------------------------------------------
    # Internal placement fallback (no Slurm)
    # ------------------------------------------------------------------

    def _schedule_internal(self) -> None:
        self._queue.sort(key=lambda j: -j.priority)
        remaining = []
        for job in self._queue:
            node = self._pick_node(job)
            if node:
                self._assign_internal(job, node)
            else:
                remaining.append(job)
        self._queue = remaining

    def _pick_node(self, job: Job) -> Optional[NodeState]:
        candidates = [
            n for n in self.state.nodes.values()
            if n.status == NodeStatus.HEALTHY and n.health_score >= 0.6
            and (1.0 - n.cpu_usage) * 100 >= job.cpu_cores
        ]
        if not candidates:
            return None

        def score(n: NodeState) -> float:
            return (
                (1.0 - n.cpu_usage)    * 0.4
                + (1.0 - n.memory_usage) * 0.3
                + n.health_score         * 0.2
                + n.trust_score          * 0.1
                + self._node_completion_rate.get(n.node_id, 0.5) * 0.1
            )
        return max(candidates, key=score)

    def _assign_internal(self, job: Job, node: NodeState) -> None:
        job.state         = JobState.RUNNING
        job.assigned_node = node.node_id
        job.started_at    = time.time()
        node.running_jobs.append(job.job_id)
        self._running[job.job_id] = job
        log.info("Job %s → node %s (health=%.2f, internal)", job.job_id, node.node_id, node.health_score)

    def _complete_finished_jobs(self) -> None:
        now = time.time()
        done = [
            jid for jid, job in self._running.items()
            if job.started_at and (now - job.started_at) >= job.wall_time_sec * 0.01
        ]
        for jid in done:
            job = self._running.pop(jid)
            job.state = JobState.COMPLETED
            job.completed_at = time.time()
            self._completed.append(job)
            node = self.state.nodes.get(job.assigned_node or "")
            if node and jid in node.running_jobs:
                node.running_jobs.remove(jid)
            nid = job.assigned_node or ""
            prev = self._node_completion_rate.get(nid, 0.5)
            self._node_completion_rate[nid] = 0.9 * prev + 0.1
            ct = job.completion_time or 0.0
            metrics.jobs_completed.inc()
            metrics.job_completion_time.observe(ct)

    # ------------------------------------------------------------------
    # Event handlers — keep Slurm in sync with controller decisions
    # ------------------------------------------------------------------

    def _on_quarantined(self, event) -> None:
        payload = event.payload or {}
        node_id = payload.get("node_id") if isinstance(payload, dict) else None
        if node_id and self._slurm_available:
            reason = payload.get("reason", "ImmuneHPC+ quarantine")
            self._slurm.drain_node(node_id, reason)

    def _on_released(self, event) -> None:
        payload = event.payload or {}
        node_id = payload.get("node_id") if isinstance(payload, dict) else None
        if node_id and self._slurm_available:
            self._slurm.resume_node(node_id)

    def _on_health_degraded(self, event) -> None:
        node: NodeState = event.payload
        if node and self._slurm_available and node.health_score < 0.4:
            # Heavily degraded — drain from Slurm immediately
            self._slurm.drain_node(
                node.node_id, f"health={node.health_score:.2f}"
            )

    def _on_health_recovered(self, event) -> None:
        node: NodeState = event.payload
        if node and self._slurm_available:
            slurm_state = self._slurm.node_slurm_state(node.node_id)
            if slurm_state and "drain" in slurm_state.lower():
                self._slurm.resume_node(node.node_id)
