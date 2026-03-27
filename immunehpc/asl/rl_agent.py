"""
asl/rl_agent.py — Reinforcement Learning Agent for Patch Generation

Implements the RL formulation from Section 5.4:
  State:  s_t = x(t)  — cluster state vector
  Action: a_t = P     — a proposed patch
  Reward: R = -J      — negative objective (we minimise J)

Uses a tabular Q-learning backbone (suitable for the discrete patch action space).
Production: replace with a deep RL agent (PPO/SAC) with a neural network policy.
"""

from __future__ import annotations
import random
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from core.state import ClusterState
from asl.patch import (
    Patch, PatchType, CodeDelta, ParameterDelta, PolicyDelta, PatchStatus
)
from modules.healer import RepairStrategy
from utils.logger import get_logger

log = get_logger("rl_agent")


# ---------------------------------------------------------------------------
# Experience replay buffer
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    state_J: float           # J value before action
    patch_id: str
    reward: float
    next_state_J: float      # J value after action
    terminal: bool = False


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000) -> None:
        self._buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, t: Transition) -> None:
        self._buffer.append(t)

    def sample(self, n: int) -> List[Transition]:
        return random.sample(list(self._buffer), min(n, len(self._buffer)))

    def __len__(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
# Patch action space
# ---------------------------------------------------------------------------

# Discrete action catalogue — each entry is a patch template
PARAMETER_ACTIONS = [
    # (namespace, delta_fn) — delta_fn(current_val) → new_val
    ("anomaly.z_threshold",     lambda v: max(1.0, v - 0.5)),   # make more sensitive
    ("anomaly.z_threshold",     lambda v: min(5.0, v + 0.5)),   # make less sensitive
    ("monitor.health_threshold", lambda v: max(0.4, v - 0.05)),
    ("monitor.health_threshold", lambda v: min(0.8, v + 0.05)),
    ("optimizer.alpha",          lambda v: max(0.1, v - 0.05)),  # reduce latency weight
    ("optimizer.beta",           lambda v: max(0.1, v - 0.05)),  # reduce power weight
    ("asl.patch_trust_threshold",lambda v: max(0.5, v - 0.05)),
]

POLICY_ACTIONS = [
    # Healing strategy reorderings
    {
        "policy_name": "healing.strategy_order",
        "new_policy": [
            RepairStrategy.REAPPLY_CONFIG,
            RepairStrategy.RESTART_SERVICE,
            RepairStrategy.ROLLBACK,
            RepairStrategy.REIMAGE_NODE,
        ],
        "description": "Try config reapply before service restart",
    },
    {
        "policy_name": "healing.strategy_order",
        "new_policy": [
            RepairStrategy.RESTART_SERVICE,
            RepairStrategy.ROLLBACK,
            RepairStrategy.REAPPLY_CONFIG,
            RepairStrategy.REIMAGE_NODE,
        ],
        "description": "Skip config reapply, go to rollback faster",
    },
]


class PatchActionSpace:
    """Enumerates the discrete set of patches the agent can generate."""

    def __init__(self) -> None:
        self._param_config: Dict[str, float] = {
            ns: 3.0 if "threshold" in ns else 0.25
            for ns, _ in PARAMETER_ACTIONS
        }

    def sample_random(self) -> Patch:
        if random.random() < 0.7:
            return self._random_parameter_patch()
        return self._random_policy_patch()

    def _random_parameter_patch(self) -> Patch:
        ns, delta_fn = random.choice(PARAMETER_ACTIONS)
        current = self._param_config.get(ns, 1.0)
        new_val = delta_fn(current)
        pd = ParameterDelta(namespace=ns, old_value=current, new_value=new_val)
        return Patch(
            patch_type=PatchType.PARAMETER,
            parameter_deltas=[pd],
            generated_by="rl_agent",
            generation_reason=f"RL: adjust {ns}",
        )

    def _random_policy_patch(self) -> Patch:
        pa = random.choice(POLICY_ACTIONS)
        pd = PolicyDelta(
            policy_name=pa["policy_name"],
            old_policy=None,
            new_policy=pa["new_policy"],
            description=pa["description"],
        )
        return Patch(
            patch_type=PatchType.POLICY,
            policy_delta=pd,
            generated_by="rl_agent",
            generation_reason=f"RL: policy change — {pa['description']}",
        )

    def apply_param_delta(self, delta: ParameterDelta) -> None:
        """Update internal config model after successful patch deployment."""
        self._param_config[delta.namespace] = delta.new_value


# ---------------------------------------------------------------------------
# Q-Learning agent
# ---------------------------------------------------------------------------

class RLAgent:
    """
    Tabular Q-learning agent operating over a discretised state space.

    State is discretised by binning J into 10 buckets.
    Actions are indices into the patch action catalogue.

    For production: replace with a neural-network policy (PPO) that
    takes the full x(t) vector as input and outputs a structured patch.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        epsilon: float = 0.1,
        replay_capacity: int = 10_000,
        batch_size: int = 32,
    ) -> None:
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.batch_size = batch_size

        self._action_space = PatchActionSpace()
        self._replay = ReplayBuffer(replay_capacity)

        # Q-table: state_bin → action_idx → Q-value
        self._n_actions = len(PARAMETER_ACTIONS) + len(POLICY_ACTIONS)
        self._q_table: Dict[int, List[float]] = {}

        self._steps = 0
        self._episode_rewards: List[float] = []
        log.info("RLAgent initialised (lr=%.4f γ=%.2f ε=%.2f)", self.lr, discount_factor, epsilon)

    def _discretise_state(self, J: float) -> int:
        """Map objective value J to a discrete state bucket."""
        return min(9, max(0, int(J * 10)))

    def _get_q(self, state_bin: int) -> List[float]:
        if state_bin not in self._q_table:
            self._q_table[state_bin] = [0.0] * self._n_actions
        return self._q_table[state_bin]

    def select_action(self, state: ClusterState) -> Tuple[int, Patch]:
        """ε-greedy action selection."""
        J = state.compute_objective()
        s = self._discretise_state(J)

        if random.random() < self.epsilon:
            # Explore
            idx = random.randrange(self._n_actions)
        else:
            # Exploit
            q = self._get_q(s)
            idx = q.index(max(q))

        patch = self._idx_to_patch(idx)
        return idx, patch

    def _idx_to_patch(self, idx: int) -> Patch:
        if idx < len(PARAMETER_ACTIONS):
            ns, delta_fn = PARAMETER_ACTIONS[idx]
            current = self._action_space._param_config.get(ns, 1.0)
            pd = ParameterDelta(namespace=ns, old_value=current, new_value=delta_fn(current))
            return Patch(
                patch_type=PatchType.PARAMETER,
                parameter_deltas=[pd],
                generated_by="rl_agent",
            )
        else:
            pa = POLICY_ACTIONS[idx - len(PARAMETER_ACTIONS)]
            pol = PolicyDelta(
                policy_name=pa["policy_name"],
                old_policy=None,
                new_policy=pa["new_policy"],
                description=pa["description"],
            )
            return Patch(patch_type=PatchType.POLICY, policy_delta=pol, generated_by="rl_agent")

    def observe(self, J_before: float, action_idx: int, patch: Patch, J_after: float) -> float:
        """Record transition and compute reward R = -J."""
        reward = -J_after    # R = -J
        self._episode_rewards.append(reward)
        self._steps += 1

        t = Transition(
            state_J=J_before,
            patch_id=patch.patch_id,
            reward=reward,
            next_state_J=J_after,
        )
        self._replay.push(t)
        self._update_q(t, action_idx)

        # Decay epsilon over time
        self.epsilon = max(0.01, self.epsilon * 0.999)
        return reward

    def _update_q(self, t: Transition, action_idx: int) -> None:
        s = self._discretise_state(t.state_J)
        s_next = self._discretise_state(t.next_state_J)

        q = self._get_q(s)
        q_next = self._get_q(s_next)

        # Bellman update
        target = t.reward + self.gamma * max(q_next)
        q[action_idx] += self.lr * (target - q[action_idx])

    def mean_reward(self, last_n: int = 100) -> float:
        recent = self._episode_rewards[-last_n:]
        return sum(recent) / len(recent) if recent else 0.0

    def stats(self) -> Dict:
        return {
            "steps": self._steps,
            "epsilon": round(self.epsilon, 4),
            "mean_reward_100": round(self.mean_reward(100), 4),
            "replay_buffer_size": len(self._replay),
            "q_states_visited": len(self._q_table),
        }
