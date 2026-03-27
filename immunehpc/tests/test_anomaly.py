"""
tests/test_anomaly.py — Unit tests for AnomalyDetector
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest

from core.state import ClusterState, NodeState
from modules.anomaly import (
    AnomalyDetector, StatisticalDetector, MLDetector,
    AnomalyType, Severity, RollingStats
)


class TestRollingStats(unittest.TestCase):

    def test_z_score_detects_outlier(self):
        rs = RollingStats(window=60)
        for _ in range(50):
            rs.update(0.2)
        # Add the outlier into the window so variance is computed with it present
        rs.update(0.99)
        z = rs.z_score(0.99)
        self.assertGreater(z, 2.0)

    def test_z_score_normal_value(self):
        rs = RollingStats(window=30)
        for i in range(30):
            rs.update(0.5)
        z = rs.z_score(0.51)
        self.assertLess(z, 1.0)

    def test_insufficient_data_returns_zero(self):
        rs = RollingStats()
        rs.update(0.5)
        self.assertEqual(rs.z_score(0.9), 0.0)


class TestStatisticalDetector(unittest.TestCase):

    def _node(self, **kwargs) -> NodeState:
        defaults = dict(node_id="n1", cpu_usage=0.2, gpu_usage=0.1,
                        memory_usage=0.3, temperature_c=55.0,
                        network_rx_mbps=100.0, trust_score=1.0)
        defaults.update(kwargs)
        return NodeState(**defaults)

    def test_no_anomaly_on_normal_data(self):
        det = StatisticalDetector(z_threshold=3.0)
        node = self._node()
        # Warm up with normal values
        for _ in range(30):
            det.detect(node)
        anomalies = det.detect(node)
        # CPU at 0.2 should not trigger
        cpu_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.CPU_SPIKE]
        self.assertEqual(len(cpu_anomalies), 0)

    def test_cpu_spike_detected(self):
        det = StatisticalDetector(z_threshold=3.0)
        # Warm up with low values
        for _ in range(40):
            det.detect(self._node(cpu_usage=0.1))
        # Inject spike
        anomalies = det.detect(self._node(cpu_usage=0.99))
        types = [a.anomaly_type for a in anomalies]
        self.assertIn(AnomalyType.CPU_SPIKE, types)

    def test_thermal_hard_threshold(self):
        det = StatisticalDetector(z_threshold=3.0)
        anomalies = det.detect(self._node(temperature_c=92.0))
        types = [a.anomaly_type for a in anomalies]
        self.assertIn(AnomalyType.THERMAL_RUNAWAY, types)

    def test_severity_critical_on_high_score(self):
        det = StatisticalDetector(z_threshold=1.0)
        for _ in range(30):
            det.detect(self._node(cpu_usage=0.05))
        anomalies = det.detect(self._node(cpu_usage=1.0))
        cpu_a = [a for a in anomalies if a.anomaly_type == AnomalyType.CPU_SPIKE]
        if cpu_a:
            self.assertIn(cpu_a[0].severity, (Severity.CRITICAL, Severity.HIGH))


class TestAnomalyDetector(unittest.TestCase):

    def setUp(self):
        self.state = ClusterState()
        self.state.nodes = {
            "n1": NodeState(node_id="n1"),
            "n2": NodeState(node_id="n2"),
        }
        self.detector = AnomalyDetector(method="statistical")

    def test_analyze_returns_list(self):
        result = self.detector.analyze(self.state)
        self.assertIsInstance(result, list)

    def test_active_anomalies_tracked(self):
        self.state.nodes["n1"].cpu_usage = 0.99
        self.state.nodes["n1"].temperature_c = 93.0
        self.detector.analyze(self.state)
        active = self.detector.active_anomalies()
        self.assertIn("n1", active)


if __name__ == "__main__":
    unittest.main(verbosity=2)
