"""
utils/metrics.py — Metrics collection, aggregation, and Prometheus-style export.

Tracks all evaluation metrics defined in Section 7 of the paper:
  - Reliability: MTTR, uptime
  - Performance: job completion time, throughput
  - Efficiency: performance/watt, temperature stability
  - Security: detection latency, false positives
  - Autonomy: human intervention count
  - Self-Improvement: patch success rate, regression rate, learning speed
"""

from __future__ import annotations
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional


@dataclass
class Sample:
    value: float
    timestamp: float = field(default_factory=time.time)


class Gauge:
    """Single current value."""
    def __init__(self) -> None:
        self._value: float = 0.0

    def set(self, v: float) -> None:
        self._value = v

    def get(self) -> float:
        return self._value


class Counter:
    """Monotonically increasing count."""
    def __init__(self) -> None:
        self._count: int = 0

    def inc(self, n: int = 1) -> None:
        self._count += n

    def get(self) -> int:
        return self._count


class Histogram:
    """Rolling window of samples for mean/min/max."""
    def __init__(self, window: int = 100) -> None:
        self._samples: Deque[float] = deque(maxlen=window)

    def observe(self, v: float) -> None:
        self._samples.append(v)

    def mean(self) -> float:
        return sum(self._samples) / len(self._samples) if self._samples else 0.0

    def min(self) -> float:
        return min(self._samples) if self._samples else 0.0

    def max(self) -> float:
        return max(self._samples) if self._samples else 0.0


class MetricsRegistry:
    """
    Central registry for all ImmuneHPC+ metrics.
    Can be scraped by Prometheus or dumped as a dict.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # --- Reliability ---
        self.uptime_start = time.time()
        self.failure_events = Counter()
        self.recovery_events = Counter()
        self.mttr_histogram = Histogram()          # seconds

        # --- Performance ---
        self.job_completion_time = Histogram()     # seconds
        self.job_throughput = Gauge()              # jobs/min
        self.jobs_completed = Counter()

        # --- Efficiency ---
        self.performance_per_watt = Gauge()
        self.temperature_variance = Gauge()

        # --- Security ---
        self.intrusions_detected = Counter()
        self.false_positives = Counter()
        self.detection_latency = Histogram()       # seconds

        # --- Autonomy ---
        self.human_interventions = Counter()

        # --- Self-Improvement ---
        self.patches_generated = Counter()
        self.patches_deployed = Counter()
        self.patches_rejected = Counter()
        self.patches_rolled_back = Counter()
        self.patch_latency = Histogram()           # seconds from gen → deploy

        # --- Objective ---
        self.objective_J = Gauge()
        self.lyapunov_V = Gauge()

    def patch_success_rate(self) -> float:
        total = self.patches_generated.get()
        return self.patches_deployed.get() / total if total else 0.0

    def patch_regression_rate(self) -> float:
        deployed = self.patches_deployed.get()
        return self.patches_rolled_back.get() / deployed if deployed else 0.0

    def uptime_seconds(self) -> float:
        return time.time() - self.uptime_start

    def dump(self) -> Dict:
        return {
            "uptime_sec": round(self.uptime_seconds(), 1),
            "failure_events": self.failure_events.get(),
            "recovery_events": self.recovery_events.get(),
            "mttr_mean_sec": round(self.mttr_histogram.mean(), 2),
            "jobs_completed": self.jobs_completed.get(),
            "job_throughput_per_min": round(self.job_throughput.get(), 2),
            "avg_job_completion_sec": round(self.job_completion_time.mean(), 2),
            "performance_per_watt": round(self.performance_per_watt.get(), 4),
            "temperature_variance": round(self.temperature_variance.get(), 2),
            "intrusions_detected": self.intrusions_detected.get(),
            "false_positives": self.false_positives.get(),
            "detection_latency_mean_sec": round(self.detection_latency.mean(), 3),
            "human_interventions": self.human_interventions.get(),
            "patches_generated": self.patches_generated.get(),
            "patches_deployed": self.patches_deployed.get(),
            "patches_rejected": self.patches_rejected.get(),
            "patches_rolled_back": self.patches_rolled_back.get(),
            "patch_success_rate": round(self.patch_success_rate(), 3),
            "patch_regression_rate": round(self.patch_regression_rate(), 3),
            "objective_J": round(self.objective_J.get(), 4),
            "lyapunov_V": round(self.lyapunov_V.get(), 4),
        }


# Module-level singleton
metrics = MetricsRegistry()
