"""
asl/patch.py — Patch Model P = (ΔC, Δθ, Δπ)

Formalises the self-improvement patch as defined in Section 5.2:
  ΔC  — code changes (new or modified logic)
  Δθ  — parameter updates (thresholds, weights, hyperparameters)
  Δπ  — policy updates (scheduling rules, repair strategies)

Each patch carries a trust score T(P) ∈ [0, 1] that gates deployment.
"""

from __future__ import annotations
import time
import uuid
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class PatchType(Enum):
    CODE = "code"           # ΔC — new/modified logic
    PARAMETER = "parameter" # Δθ — hyperparameter/threshold update
    POLICY = "policy"       # Δπ — scheduling/repair policy change
    COMPOSITE = "composite" # All three combined


class PatchStatus(Enum):
    DRAFT = "draft"
    SANDBOX_PENDING = "sandbox_pending"
    SANDBOX_PASSED = "sandbox_passed"
    SANDBOX_FAILED = "sandbox_failed"
    CANARY = "canary"
    DEPLOYED = "deployed"
    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"


@dataclass
class CodeDelta:
    """ΔC — a code-level change."""
    target_module: str          # e.g. "modules.healer"
    target_function: str        # e.g. "_restart_service"
    diff: str                   # unified diff or full replacement
    language: str = "python"
    loc_changed: int = 0


@dataclass
class ParameterDelta:
    """Δθ — parameter/threshold update."""
    namespace: str              # e.g. "anomaly.z_threshold"
    old_value: Any
    new_value: Any
    rationale: str = ""


@dataclass
class PolicyDelta:
    """Δπ — policy/strategy change."""
    policy_name: str            # e.g. "healing.strategy_order"
    old_policy: Any
    new_policy: Any
    description: str = ""


@dataclass
class Patch:
    """
    P = (ΔC, Δθ, Δπ) — the atomic unit of self-improvement.

    A patch is accepted iff:
      1. Stability:   V(x_new) ≤ V(x_old)
      2. Performance: J_new ≤ J_old
      3. Safety:      no privilege escalation, no data corruption
      4. Trust:       T(P) ≥ trust_threshold
    """
    patch_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    patch_type: PatchType = PatchType.COMPOSITE

    # The three deltas
    code_delta: Optional[CodeDelta] = None
    parameter_deltas: List[ParameterDelta] = field(default_factory=list)
    policy_delta: Optional[PolicyDelta] = None

    # Metadata
    generated_by: str = "rl_agent"
    generation_reason: str = ""
    status: PatchStatus = PatchStatus.DRAFT
    trust_score: float = 0.0         # T(P) ∈ [0, 1]

    # Validation results
    sandbox_passed: bool = False
    stability_passed: bool = False
    performance_passed: bool = False
    safety_passed: bool = False

    # Timestamps
    created_at: float = field(default_factory=time.time)
    deployed_at: Optional[float] = None
    rolled_back_at: Optional[float] = None

    # Lyapunov comparison
    V_before: Optional[float] = None
    V_after: Optional[float] = None
    J_before: Optional[float] = None
    J_after: Optional[float] = None

    def __post_init__(self) -> None:
        self._compute_fingerprint()

    def _compute_fingerprint(self) -> str:
        """Content-addressable fingerprint for deduplication."""
        content = f"{self.patch_type}{self.code_delta}{self.parameter_deltas}{self.policy_delta}"
        self.fingerprint = hashlib.sha256(content.encode()).hexdigest()[:16]
        return self.fingerprint

    @property
    def is_valid(self) -> bool:
        return (
            self.sandbox_passed
            and self.stability_passed
            and self.performance_passed
            and self.safety_passed
            and self.trust_score >= 0.75
        )

    @property
    def lyapunov_monotonic(self) -> bool:
        """V(x_new) ≤ V(x_old) — the core stability condition."""
        if self.V_before is None or self.V_after is None:
            return False
        return self.V_after <= self.V_before

    def summary(self) -> Dict:
        return {
            "patch_id": self.patch_id,
            "type": self.patch_type.value,
            "status": self.status.value,
            "trust_score": round(self.trust_score, 3),
            "sandbox_passed": self.sandbox_passed,
            "stability_passed": self.stability_passed,
            "performance_passed": self.performance_passed,
            "safety_passed": self.safety_passed,
            "is_valid": self.is_valid,
            "V_before": self.V_before,
            "V_after": self.V_after,
            "J_before": self.J_before,
            "J_after": self.J_after,
        }

    def __str__(self) -> str:
        return (
            f"Patch({self.patch_id}, {self.patch_type.value}, "
            f"status={self.status.value}, T={self.trust_score:.2f})"
        )
