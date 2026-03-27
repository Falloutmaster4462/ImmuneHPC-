"""
core/state.py — System State Vector x(t)

Models the full cluster state as a dataclass that snapshots every metric
used across the MAPE-K loop. This is the x(t) defined in the paper.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from enum import Enum


class NodeStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    QUARANTINED = "quarantined"
    OFFLINE = "offline"
    REIMAGING = "reimaging"


@dataclass
class NodeState:
    """Per-node state snapshot."""
    node_id: str
    status: NodeStatus = NodeStatus.HEALTHY
    health_score: float = 1.0          # H_i ∈ [0, 1]

    # Resource metrics
    cpu_usage: float = 0.0             # 0–1
    gpu_usage: float = 0.0             # 0–1
    memory_usage: float = 0.0          # 0–1
    temperature_c: float = 0.0
    network_rx_mbps: float = 0.0
    network_tx_mbps: float = 0.0

    # Job state
    running_jobs: List[str] = field(default_factory=list)
    job_queue_depth: int = 0

    # Security
    trust_score: float = 1.0           # PKI/IDS-derived
    anomaly_flags: List[str] = field(default_factory=list)

    timestamp: float = field(default_factory=time.time)

    def is_healthy(self, threshold: float = 0.6) -> bool:
        return self.health_score >= threshold and self.status == NodeStatus.HEALTHY

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class ClusterState:
    """
    Global system state vector x(t).

    Aggregates all node states and cluster-level metrics.
    Used as input to every module in the MAPE-K loop.
    """
    timestamp: float = field(default_factory=time.time)
    nodes: Dict[str, NodeState] = field(default_factory=dict)

    # Cluster-level aggregates
    total_jobs_running: int = 0
    total_jobs_queued: int = 0
    cluster_cpu_utilization: float = 0.0
    cluster_gpu_utilization: float = 0.0
    cluster_avg_temperature: float = 0.0
    cluster_power_watts: float = 0.0

    # Objective function components (J = α·L + β·P + γ·F + δ·S)
    latency: float = 0.0               # L — avg job wait time (s)
    power_consumption: float = 0.0     # P — normalised 0–1
    failure_rate: float = 0.0          # F — failures/min
    security_risk: float = 0.0         # S — composite risk score 0–1

    # Derived Lyapunov value V(x)
    lyapunov_value: float = 0.0

    # Active alerts
    active_alerts: List[str] = field(default_factory=list)

    def compute_objective(
        self,
        alpha: float = 0.25,
        beta: float = 0.25,
        gamma: float = 0.25,
        delta: float = 0.25,
    ) -> float:
        """J = α·L + β·P + γ·F + δ·S  (minimise)"""
        return (
            alpha * self.latency
            + beta * self.power_consumption
            + gamma * self.failure_rate
            + delta * self.security_risk
        )

    def compute_lyapunov(self, **weights) -> float:
        """V(x) — instability measure; must be non-increasing under valid control."""
        v = self.compute_objective(**weights)
        self.lyapunov_value = v
        return v

    @property
    def healthy_nodes(self) -> List[NodeState]:
        return [n for n in self.nodes.values() if n.status == NodeStatus.HEALTHY]

    @property
    def quarantined_nodes(self) -> List[NodeState]:
        return [n for n in self.nodes.values() if n.status == NodeStatus.QUARANTINED]

    @property
    def unhealthy_nodes(self) -> List[NodeState]:
        return [
            n for n in self.nodes.values()
            if n.status not in (NodeStatus.HEALTHY, NodeStatus.REIMAGING)
        ]

    def aggregate(self) -> None:
        """Recompute cluster-level aggregates from individual node states."""
        nodes = list(self.nodes.values())
        if not nodes:
            return

        self.cluster_cpu_utilization = sum(n.cpu_usage for n in nodes) / len(nodes)
        self.cluster_gpu_utilization = sum(n.gpu_usage for n in nodes) / len(nodes)
        self.cluster_avg_temperature = sum(n.temperature_c for n in nodes) / len(nodes)
        self.total_jobs_running = sum(len(n.running_jobs) for n in nodes)
        self.total_jobs_queued = sum(n.job_queue_depth for n in nodes)
        self.failure_rate = len(self.unhealthy_nodes) / max(len(nodes), 1)

    def snapshot(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "node_count": len(self.nodes),
            "healthy": len(self.healthy_nodes),
            "quarantined": len(self.quarantined_nodes),
            "latency": self.latency,
            "power": self.power_consumption,
            "failure_rate": self.failure_rate,
            "security_risk": self.security_risk,
            "objective_J": self.compute_objective(),
            "lyapunov_V": self.lyapunov_value,
        }
