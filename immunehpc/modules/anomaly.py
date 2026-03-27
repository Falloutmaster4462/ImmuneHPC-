"""
modules/anomaly.py — Anomaly Detection Engine

Hybrid detection pipeline (Section 4.2, Module 3):
  1. Statistical: Z-score on rolling windows per metric
  2. ML: Isolation Forest for multi-variate anomaly scoring
  3. Rule-based: hard thresholds for known failure signatures

Emits ANOMALY_DETECTED events with severity and anomaly type.
Analogous to innate immunity — fast, pattern-based threat recognition.
"""

from __future__ import annotations
import time
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

from core.state import ClusterState, NodeState
from utils.events import bus, EventType
from utils.logger import get_logger

log = get_logger("anomaly")


class AnomalyType(Enum):
    CPU_SPIKE = "cpu_spike"
    GPU_SPIKE = "gpu_spike"
    MEMORY_PRESSURE = "memory_pressure"
    THERMAL_RUNAWAY = "thermal_runaway"
    NETWORK_FLOOD = "network_flood"
    HEALTH_DEGRADED = "health_degraded"
    SECURITY_THREAT = "security_threat"
    MULTIVARIATE = "multivariate"         # ML-detected


class Severity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Anomaly:
    node_id: str
    anomaly_type: AnomalyType
    severity: Severity
    score: float                          # 0–1; higher = more anomalous
    description: str
    metrics: Dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        return (
            f"[{self.severity.name}] {self.anomaly_type.value} "
            f"on {self.node_id} (score={self.score:.2f}): {self.description}"
        )


# ---------------------------------------------------------------------------
# Statistical detector — per-metric Z-score on rolling window
# ---------------------------------------------------------------------------

class RollingStats:
    """Online mean and standard deviation using Welford's algorithm."""

    def __init__(self, window: int = 60) -> None:
        self._window: Deque[float] = deque(maxlen=window)

    def update(self, value: float) -> Tuple[float, float]:
        """Returns (mean, std) after adding value."""
        self._window.append(value)
        n = len(self._window)
        if n < 2:
            return value, 0.0
        mean = sum(self._window) / n
        variance = sum((x - mean) ** 2 for x in self._window) / (n - 1)
        return mean, math.sqrt(variance)

    def z_score(self, value: float) -> float:
        n = len(self._window)
        if n < 2:
            return 0.0
        mean = sum(self._window) / n
        variance = sum((x - mean) ** 2 for x in self._window) / (n - 1)
        if variance < 1e-12:
            return 0.0          # all values identical → no deviation
        std = math.sqrt(variance)
        return abs(value - mean) / std


class StatisticalDetector:
    def __init__(self, z_threshold: float = 3.0) -> None:
        self.z_threshold = z_threshold
        # node_id → metric_name → RollingStats
        self._stats: Dict[str, Dict[str, RollingStats]] = defaultdict(
            lambda: defaultdict(RollingStats)
        )

    def detect(self, node: NodeState) -> List[Anomaly]:
        anomalies: List[Anomaly] = []
        nid = node.node_id

        checks = {
            "cpu_usage": (node.cpu_usage, AnomalyType.CPU_SPIKE, 0.90),
            "gpu_usage": (node.gpu_usage, AnomalyType.GPU_SPIKE, 0.90),
            "memory_usage": (node.memory_usage, AnomalyType.MEMORY_PRESSURE, 0.92),
            "temperature_c": (node.temperature_c, AnomalyType.THERMAL_RUNAWAY, 88.0),
            "network_rx_mbps": (node.network_rx_mbps, AnomalyType.NETWORK_FLOOD, 900.0),
        }

        for metric, (value, atype, hard_threshold) in checks.items():
            self._stats[nid][metric].update(value)
            z = self._stats[nid][metric].z_score(value)

            # Hard threshold check
            hard_hit = value > hard_threshold
            # Statistical check
            stat_hit = z > self.z_threshold

            if hard_hit or stat_hit:
                score = min(1.0, max(z / (self.z_threshold * 2), 0.5 if hard_hit else 0.0))
                severity = Severity.CRITICAL if score > 0.85 else (
                    Severity.HIGH if score > 0.65 else Severity.MEDIUM
                )
                anomalies.append(Anomaly(
                    node_id=nid,
                    anomaly_type=atype,
                    severity=severity,
                    score=score,
                    description=f"{metric}={value:.2f} z={z:.1f}",
                    metrics={metric: value, "z_score": z},
                ))

        return anomalies


# ---------------------------------------------------------------------------
# ML detector — Isolation Forest (pure Python, no sklearn dependency)
# ---------------------------------------------------------------------------

class IsolationForest:
    """
    Minimal Isolation Forest implementation.

    For production: replace with sklearn.ensemble.IsolationForest
    or a pre-trained ONNX model.
    """

    def __init__(self, n_trees: int = 50, subsample: int = 256) -> None:
        self.n_trees = n_trees
        self.subsample = subsample
        self._fitted = False
        self._data_buffer: List[List[float]] = []

    def _extract_features(self, node: NodeState) -> List[float]:
        return [
            node.cpu_usage,
            node.gpu_usage,
            node.memory_usage,
            node.temperature_c / 100.0,
            node.network_rx_mbps / 1000.0,
            node.network_tx_mbps / 1000.0,
            1.0 - node.trust_score,
            node.health_score,
        ]

    def update(self, node: NodeState) -> None:
        self._data_buffer.append(self._extract_features(node))
        if len(self._data_buffer) >= self.subsample:
            self._fitted = True

    def anomaly_score(self, node: NodeState) -> float:
        """
        Returns score in [0, 1]; values near 1 indicate anomalies.

        Stub implementation: uses distance from running mean as proxy.
        Replace with proper tree traversal for production.
        """
        if not self._fitted:
            return 0.0

        features = self._extract_features(node)
        n = len(self._data_buffer)

        means = [sum(row[i] for row in self._data_buffer) / n for i in range(len(features))]
        variances = [
            sum((row[i] - means[i]) ** 2 for row in self._data_buffer) / n
            for i in range(len(features))
        ]

        # Mahalanobis-like distance (diagonal covariance)
        dist = sum(
            ((features[i] - means[i]) ** 2) / max(variances[i], 1e-9)
            for i in range(len(features))
        )
        # Normalise to [0, 1]
        return min(1.0, dist / (len(features) * 9))


class MLDetector:
    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold
        self._models: Dict[str, IsolationForest] = {}

    def _get_model(self, node_id: str) -> IsolationForest:
        if node_id not in self._models:
            self._models[node_id] = IsolationForest()
        return self._models[node_id]

    def detect(self, node: NodeState) -> List[Anomaly]:
        model = self._get_model(node.node_id)
        model.update(node)
        score = model.anomaly_score(node)

        if score >= self.threshold:
            severity = Severity.CRITICAL if score > 0.9 else (
                Severity.HIGH if score > 0.75 else Severity.MEDIUM
            )
            return [Anomaly(
                node_id=node.node_id,
                anomaly_type=AnomalyType.MULTIVARIATE,
                severity=severity,
                score=score,
                description=f"Isolation Forest anomaly score={score:.3f}",
            )]
        return []


# ---------------------------------------------------------------------------
# Composite anomaly detector
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """
    MAPE-K Analyze phase.

    Runs both detectors on every telemetry update and fires
    ANOMALY_DETECTED events for confirmed anomalies.
    """

    def __init__(
        self,
        z_threshold: float = 3.0,
        ml_threshold: float = 0.6,
        method: str = "hybrid",
    ) -> None:
        self.method = method
        self._statistical = StatisticalDetector(z_threshold)
        self._ml = MLDetector(ml_threshold)
        self._active: Dict[str, List[Anomaly]] = defaultdict(list)

        bus.subscribe(EventType.TELEMETRY_COLLECTED, self._on_telemetry)

    def analyze(self, state: ClusterState) -> List[Anomaly]:
        all_anomalies: List[Anomaly] = []

        for node in state.nodes.values():
            node_anomalies: List[Anomaly] = []

            if self.method in ("statistical", "hybrid"):
                node_anomalies.extend(self._statistical.detect(node))

            if self.method in ("ml", "hybrid"):
                node_anomalies.extend(self._ml.detect(node))

            # Update node's anomaly flags
            node.anomaly_flags = [a.anomaly_type.value for a in node_anomalies]

            # Emit events for new anomalies
            prev_types = {a.anomaly_type for a in self._active.get(node.node_id, [])}
            for anomaly in node_anomalies:
                if anomaly.anomaly_type not in prev_types:
                    log.warning("ANOMALY: %s", anomaly)
                    bus.emit_simple(EventType.ANOMALY_DETECTED, "anomaly", payload=anomaly)

            self._active[node.node_id] = node_anomalies
            all_anomalies.extend(node_anomalies)

        return all_anomalies

    def _on_telemetry(self, event) -> None:
        # Triggered by monitor — delegate full state analysis to controller
        pass

    def active_anomalies(self) -> Dict[str, List[Anomaly]]:
        return dict(self._active)
