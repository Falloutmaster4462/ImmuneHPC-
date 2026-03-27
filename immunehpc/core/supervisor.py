"""
core/supervisor.py — Autonomous Supervisor

Central arbiter that enforces the Lyapunov stability constraint
dV/dt ≤ 0 across all control actions (Section 1 of Extended doc).

Responsibilities:
  - Gate all control actions through the stability constraint
  - Arbitrate between competing modules (healer vs optimizer)
  - Escalate quarantine when healing is exhausted
  - Trigger ASL when repeated failures signal a systemic issue
  - Maintain the MAPE-K feedback loop
"""

from __future__ import annotations
import time
from collections import defaultdict
from typing import Dict, List, Optional

from core.state import ClusterState, NodeStatus
from utils.events import bus, EventType
from utils.logger import get_logger
from utils.metrics import metrics

log = get_logger("supervisor")


class AutonomousSupervisor:
    """
    The top-level control authority in ImmuneHPC+.

    All modules report to the supervisor. The supervisor decides:
      - Whether to allow or veto a proposed action (V(x) gate)
      - When to escalate from healing → quarantine
      - When repeated failures justify an ASL patch cycle
    """

    def __init__(
        self,
        state: ClusterState,
        lyapunov_tolerance: float = 0.05,   # allow V to increase by at most 5% transiently
        asl_failure_threshold: int = 3,      # repeated failures before triggering ASL
    ) -> None:
        self.state = state
        self.lyapunov_tolerance = lyapunov_tolerance
        self.asl_failure_threshold = asl_failure_threshold

        self._failure_counts: Dict[str, int] = defaultdict(int)
        self._V_history: List[float] = []
        self._last_asl_trigger: float = 0.0
        self._asl_cooldown_sec = 60.0

        # Subscribe to failure events
        bus.subscribe(EventType.HEAL_FAILED, self._on_heal_failed)
        bus.subscribe(EventType.PATCH_ROLLED_BACK, self._on_patch_rolled_back)

    # ------------------------------------------------------------------
    # Stability gating
    # ------------------------------------------------------------------

    def approve_action(self, proposed_V: float, label: str = "") -> bool:
        """
        Gate a proposed control action against the Lyapunov condition.

        dV/dt ≤ 0  (with tolerance for transient instability)

        Returns True if the action is approved.
        """
        current_V = self.state.lyapunov_value
        self._V_history.append(current_V)
        if len(self._V_history) > 200:
            self._V_history.pop(0)

        # Allow the action if V doesn't increase beyond tolerance
        max_allowed_V = current_V * (1.0 + self.lyapunov_tolerance)
        approved = proposed_V <= max_allowed_V

        if not approved:
            log.warning("SUPERVISOR VETO: %s proposed V=%.4f > max=%.4f",
                        label or "action", proposed_V, max_allowed_V)

        return approved

    # ------------------------------------------------------------------
    # Supervision tick (called each MAPE-K loop)
    # ------------------------------------------------------------------

    def supervise(
        self,
        quarantine_layer,
        healer,
        asl_pipeline,
    ) -> None:
        """
        One supervision pass:
          1. Check quarantine timeouts
          2. Check if any nodes need escalation
          3. Decide if ASL should be triggered
        """
        quarantine_layer.check_timeouts()
        self._check_escalations(quarantine_layer, healer)
        self._consider_asl(asl_pipeline)

    def _check_escalations(self, quarantine_layer, healer) -> None:
        """Escalate persistently-failing nodes to quarantine."""
        for node in self.state.nodes.values():
            if node.status == NodeStatus.HEALTHY and node.health_score < 0.3:
                # Very low health + still not quarantined
                if self._failure_counts.get(node.node_id, 0) >= 2:
                    log.warning("Supervisor escalating %s to quarantine (health=%.2f)",
                                node.node_id, node.health_score)
                    quarantine_layer.quarantine(
                        node.node_id,
                        reason="supervisor: persistently degraded health"
                    )
                    metrics.human_interventions.inc()   # counts as escalation

    def _consider_asl(self, asl_pipeline) -> None:
        """
        Trigger an ASL patch cycle when:
          - Multiple nodes have repeatedly failed
          - J is trending upward (V non-decreasing)
          - ASL cooldown has elapsed
        """
        if not asl_pipeline.enabled:
            return

        now = time.time()
        if now - self._last_asl_trigger < self._asl_cooldown_sec:
            return

        # Check if J is trending up
        if len(self._V_history) >= 10:
            recent = self._V_history[-10:]
            trend = recent[-1] - recent[0]
            if trend > 0.05:   # V increased by > 5% over last 10 ticks
                log.info("Supervisor triggering ASL (V trend=+%.4f)", trend)
                self._last_asl_trigger = now
                asl_pipeline.step()

        # Check repeated failures
        repeated_failures = sum(
            1 for count in self._failure_counts.values()
            if count >= self.asl_failure_threshold
        )
        if repeated_failures >= 2:
            log.info("Supervisor triggering ASL (%d nodes with repeated failures)",
                     repeated_failures)
            self._last_asl_trigger = now
            self._failure_counts.clear()
            asl_pipeline.step()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_heal_failed(self, event) -> None:
        payload = event.payload or {}
        node_id = payload.get("node_id", "")
        if node_id:
            self._failure_counts[node_id] += 1
            log.debug("Supervisor: failure count for %s = %d",
                      node_id, self._failure_counts[node_id])

    def _on_patch_rolled_back(self, event) -> None:
        log.warning("Supervisor: patch rolled back — monitoring for instability")

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def status(self) -> Dict:
        return {
            "lyapunov_V": round(self.state.lyapunov_value, 4),
            "objective_J": round(self.state.compute_objective(), 4),
            "failure_counts": dict(self._failure_counts),
            "nodes_total": len(self.state.nodes),
            "nodes_healthy": len(self.state.healthy_nodes),
            "nodes_quarantined": len(self.state.quarantined_nodes),
        }
