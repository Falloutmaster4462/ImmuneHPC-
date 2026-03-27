"""
tests/test_asl.py — Unit tests for ASL Pipeline, Patch model, Sandbox, RL Agent
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest

from core.state import ClusterState, NodeState
from asl.patch import (
    Patch, PatchType, PatchStatus, CodeDelta, ParameterDelta, PolicyDelta
)
from asl.sandbox import SandboxExecutor, RegressionSuite
from asl.rl_agent import RLAgent, ReplayBuffer, Transition
from asl.pipeline import ASLPipeline, _compute_trust_score


class TestPatchModel(unittest.TestCase):

    def test_patch_fingerprint_deterministic(self):
        p1 = Patch(patch_type=PatchType.PARAMETER,
                   parameter_deltas=[ParameterDelta("x", 1.0, 2.0)])
        p2 = Patch(patch_type=PatchType.PARAMETER,
                   parameter_deltas=[ParameterDelta("x", 1.0, 2.0)])
        # Fingerprints should be identical for same content
        self.assertEqual(p1.fingerprint, p2.fingerprint)

    def test_patch_is_valid_requires_all_gates(self):
        p = Patch()
        p.sandbox_passed = True
        p.stability_passed = True
        p.performance_passed = True
        p.safety_passed = True
        p.trust_score = 0.80
        self.assertTrue(p.is_valid)

    def test_patch_invalid_if_trust_too_low(self):
        p = Patch()
        p.sandbox_passed = True
        p.stability_passed = True
        p.performance_passed = True
        p.safety_passed = True
        p.trust_score = 0.50   # below 0.75
        self.assertFalse(p.is_valid)

    def test_lyapunov_monotonic(self):
        p = Patch()
        p.V_before = 0.5
        p.V_after = 0.4    # decreased — good
        self.assertTrue(p.lyapunov_monotonic)

        p.V_after = 0.6    # increased — bad
        self.assertFalse(p.lyapunov_monotonic)

    def test_patch_summary_returns_dict(self):
        p = Patch(patch_type=PatchType.CODE)
        s = p.summary()
        self.assertIn("patch_id", s)
        self.assertIn("trust_score", s)
        self.assertIn("is_valid", s)


class TestRegressionSuite(unittest.TestCase):

    def test_clean_patch_passes_all(self):
        reg = RegressionSuite()
        p = Patch(
            patch_type=PatchType.PARAMETER,
            parameter_deltas=[ParameterDelta("z_threshold", 3.0, 3.5)],
        )
        ok, failures = reg.run(p)
        self.assertTrue(ok)
        self.assertEqual(failures, [])

    def test_negative_param_fails_range_check(self):
        reg = RegressionSuite()
        p = Patch(
            patch_type=PatchType.PARAMETER,
            parameter_deltas=[ParameterDelta("threshold", 1.0, -0.5)],
        )
        ok, failures = reg.run(p)
        self.assertFalse(ok)
        self.assertIn("parameter_range_valid", failures)

    def test_oversized_code_patch_fails(self):
        reg = RegressionSuite()
        p = Patch(
            patch_type=PatchType.CODE,
            code_delta=CodeDelta(
                target_module="healer", target_function="heal",
                diff="# large patch", loc_changed=500,
            ),
        )
        ok, failures = reg.run(p)
        self.assertFalse(ok)
        self.assertIn("patch_size_limit", failures)

    def test_dangerous_code_fails_safety(self):
        reg = RegressionSuite()
        p = Patch(
            patch_type=PatchType.CODE,
            code_delta=CodeDelta(
                target_module="m", target_function="f",
                diff="os.setuid(0)",
            ),
        )
        ok, failures = reg.run(p)
        self.assertFalse(ok)
        self.assertIn("no_root_privilege", failures)


class TestSandboxExecutor(unittest.TestCase):

    def setUp(self):
        self.state = ClusterState()
        self.state.nodes = {"n1": NodeState(node_id="n1")}
        self.sandbox = SandboxExecutor(timeout_sec=5.0)

    def test_safe_parameter_patch_passes(self):
        p = Patch(
            patch_type=PatchType.PARAMETER,
            parameter_deltas=[ParameterDelta("threshold", 3.0, 3.5)],
        )
        result = self.sandbox.test(p, self.state)
        self.assertTrue(result.passed)

    def test_unsafe_code_patch_fails(self):
        p = Patch(
            patch_type=PatchType.CODE,
            code_delta=CodeDelta("m", "f", diff="eval('1+1')", loc_changed=1),
        )
        result = self.sandbox.test(p, self.state)
        self.assertFalse(result.passed)
        self.assertTrue(len(result.safety_violations) > 0)


class TestTrustScore(unittest.TestCase):

    def test_all_passed_gives_full_score(self):
        p = Patch()
        p.sandbox_passed = True
        p.stability_passed = True
        p.performance_passed = True
        p.safety_passed = True
        self.assertEqual(_compute_trust_score(p), 1.0)

    def test_partial_pass_gives_partial_score(self):
        p = Patch()
        p.sandbox_passed = True   # +0.40
        p.stability_passed = False
        p.performance_passed = False
        p.safety_passed = False
        score = _compute_trust_score(p)
        self.assertAlmostEqual(score, 0.40)

    def test_all_failed_gives_zero(self):
        p = Patch()
        p.sandbox_passed = False
        p.stability_passed = False
        p.performance_passed = False
        p.safety_passed = False
        self.assertEqual(_compute_trust_score(p), 0.0)


class TestRLAgent(unittest.TestCase):

    def setUp(self):
        self.state = ClusterState()
        self.state.nodes = {"n1": NodeState(node_id="n1")}
        self.state.latency = 0.3
        self.state.power_consumption = 0.4
        self.state.failure_rate = 0.1
        self.state.security_risk = 0.05
        self.agent = RLAgent(learning_rate=0.01, epsilon=0.0)

    def test_select_action_returns_patch(self):
        idx, patch = self.agent.select_action(self.state)
        self.assertIsInstance(patch, Patch)
        self.assertIsInstance(idx, int)

    def test_observe_records_transition(self):
        idx, patch = self.agent.select_action(self.state)
        self.agent.observe(0.5, idx, patch, 0.45)
        self.assertEqual(len(self.agent._replay), 1)

    def test_epsilon_decays(self):
        self.agent.epsilon = 0.5
        idx, patch = self.agent.select_action(self.state)
        for _ in range(100):
            self.agent.observe(0.5, idx, patch, 0.5)
        self.assertLess(self.agent.epsilon, 0.5)

    def test_stats_returns_dict(self):
        stats = self.agent.stats()
        self.assertIn("steps", stats)
        self.assertIn("epsilon", stats)
        self.assertIn("mean_reward_100", stats)

    def test_q_table_updated_after_observe(self):
        idx, patch = self.agent.select_action(self.state)
        self.agent.observe(0.5, idx, patch, 0.4)
        self.assertGreater(len(self.agent._q_table), 0)


class TestReplayBuffer(unittest.TestCase):

    def test_push_and_sample(self):
        buf = ReplayBuffer(capacity=10)
        for i in range(5):
            buf.push(Transition(float(i), f"pid-{i}", float(-i), float(i - 0.1)))
        samples = buf.sample(3)
        self.assertEqual(len(samples), 3)

    def test_capacity_respected(self):
        buf = ReplayBuffer(capacity=5)
        for i in range(10):
            buf.push(Transition(float(i), f"p{i}", 0.0, 0.0))
        self.assertEqual(len(buf), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
