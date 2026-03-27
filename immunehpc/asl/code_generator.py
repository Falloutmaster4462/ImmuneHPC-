"""
asl/code_generator.py — ΔC Autonomous Code Generation

Uses the LLMRouter to generate real Python patches for the ImmuneHPC+
codebase in response to observed failure patterns.

The generator:
  1. Diagnoses a failure from the ClusterState and anomaly history
  2. Identifies which module function caused / could fix the problem
  3. Constructs a tightly-scoped prompt with the current function source
  4. Asks the LLM for a corrected/improved version
  5. Returns a CodeDelta patch for the sandbox to validate

Safety constraints baked into the prompt:
  - No privilege escalation
  - No eval/exec
  - Must pass existing regression suite
  - Changes must be < 200 LOC
  - Must include a docstring explaining the change

Supported patch targets (functions the LLM is allowed to modify):
  - modules/monitor.py   :: _compute_health_score
  - modules/anomaly.py   :: StatisticalDetector.detect
  - modules/healer.py    :: _restart_service
  - modules/scheduler.py :: AdaptiveScheduler._pick_node
  - asl/rl_agent.py      :: RLAgent._update_q
"""

from __future__ import annotations
import ast
import inspect
import os
import re
import textwrap
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from asl.patch import Patch, PatchType, CodeDelta
from asl.llm_backend import LLMRouter, LLMResponse
from core.state import ClusterState
from utils.logger import get_logger

log = get_logger("asl.codegen")


# ---------------------------------------------------------------------------
# Allowed patch targets — functions the LLM may propose changes to.
# Expanding this list is a deliberate research decision, not a config option.
# ---------------------------------------------------------------------------

PATCH_TARGETS: Dict[str, Dict] = {
    "health_scoring": {
        "module_path": "modules/monitor.py",
        "function_name": "_compute_health_score",
        "description": "per-node health score computation H_i ∈ [0,1]",
        "trigger": "health scores not correlating with actual node failures",
    },
    "anomaly_detection": {
        "module_path": "modules/anomaly.py",
        "function_name": "StatisticalDetector.detect",
        "description": "statistical anomaly detection on telemetry metrics",
        "trigger": "high false-positive or false-negative anomaly rate",
    },
    "service_restart": {
        "module_path": "modules/healer.py",
        "function_name": "_restart_service",
        "description": "SSH-based service restart repair strategy",
        "trigger": "service restart strategy consistently failing",
    },
    "job_placement": {
        "module_path": "modules/scheduler.py",
        "function_name": "AdaptiveScheduler._pick_node",
        "description": "health-weighted job placement scoring",
        "trigger": "jobs repeatedly assigned to degraded nodes",
    },
    "q_learning": {
        "module_path": "asl/rl_agent.py",
        "function_name": "RLAgent._update_q",
        "description": "Q-value Bellman update for RL agent",
        "trigger": "RL agent reward not improving over time",
    },
}


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert Python engineer working on ImmuneHPC+,
an autonomous HPC cluster management system. You generate safe, minimal,
well-tested patches to improve the system's self-healing and optimization
capabilities.

STRICT RULES — violations cause automatic rejection:
1. Never use eval(), exec(), __import__(), or os.system()
2. Never escalate privileges (no os.setuid, no sudo)
3. Changes must be < 200 lines
4. Keep the same function signature
5. Include a docstring explaining what changed and why
6. The function must be syntactically valid Python 3.8+
7. Do not import new external packages (only stdlib + existing imports)

OUTPUT FORMAT:
Return ONLY the complete replacement function, nothing else.
No markdown fences, no explanation outside the function.
Start directly with 'def ' or 'async def '.
"""

PATCH_PROMPT_TEMPLATE = """
The function `{function_name}` in `{module_path}` is underperforming.

OBSERVED PROBLEM:
{problem_description}

CURRENT METRICS:
{metrics_summary}

CURRENT FUNCTION SOURCE:
{current_source}

TASK:
Rewrite `{function_name}` to fix the observed problem.
Improve it based on the metrics. Keep changes minimal and targeted.
Return ONLY the complete new function definition.
""".strip()


# ---------------------------------------------------------------------------
# Source reader
# ---------------------------------------------------------------------------

def _read_function_source(module_path: str, function_name: str) -> Optional[str]:
    """Read the current source of a function from disk."""
    # Resolve relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(project_root, module_path)

    try:
        with open(full_path) as f:
            source = f.read()
    except FileNotFoundError:
        log.warning("Source file not found: %s", full_path)
        return None

    # Handle Class.method notation
    if "." in function_name:
        class_name, method_name = function_name.split(".", 1)
        pattern = rf"(    def {re.escape(method_name)}\s*\([^)]*\).*?)(?=\n    def |\nclass |\Z)"
    else:
        pattern = rf"(def {re.escape(function_name)}\s*\([^)]*\).*?)(?=\ndef |\nclass |\Z)"

    match = re.search(pattern, source, re.DOTALL)
    if not match:
        log.warning("Function not found in source: %s in %s", function_name, module_path)
        return None

    return textwrap.dedent(match.group(1)).strip()


def _validate_python(code: str) -> Tuple[bool, str]:
    """Check that generated code is syntactically valid Python."""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as exc:
        return False, str(exc)


def _count_loc(code: str) -> int:
    return len([l for l in code.splitlines() if l.strip() and not l.strip().startswith("#")])


# ---------------------------------------------------------------------------
# Problem diagnoser — maps cluster state to a patch target
# ---------------------------------------------------------------------------

@dataclass
class DiagnosisResult:
    target_key: str
    problem_description: str
    metrics_summary: str
    confidence: float   # 0–1


def _diagnose(state: ClusterState, failure_history: List[dict]) -> Optional[DiagnosisResult]:
    """
    Analyse cluster state and recent failures to identify the best
    patch target.

    Returns None if no clear diagnosis can be made (don't generate
    a patch if we don't know what's wrong).
    """
    if not failure_history:
        return None

    recent = failure_history[-10:]

    # --- Pattern: health scores not tracking reality ---
    # Many nodes with low health but high availability suggests H_i is miscalibrated
    healthy_frac = len(state.healthy_nodes) / max(len(state.nodes), 1)
    low_health = [n for n in state.nodes.values() if n.health_score < 0.5]

    if len(low_health) > len(state.nodes) * 0.4 and healthy_frac > 0.8:
        return DiagnosisResult(
            target_key="health_scoring",
            problem_description=(
                f"{len(low_health)} nodes have H_i < 0.5 but the cluster is "
                f"{healthy_frac:.0%} available. The health scoring weights may be "
                f"miscalibrated for this hardware, causing unnecessary alerts."
            ),
            metrics_summary=(
                f"healthy_nodes={len(state.healthy_nodes)}/{len(state.nodes)}, "
                f"avg_health={sum(n.health_score for n in state.nodes.values())/max(len(state.nodes),1):.2f}, "
                f"failure_rate={state.failure_rate:.3f}"
            ),
            confidence=0.75,
        )

    # --- Pattern: repeated heal_failed events ---
    heal_failures = [e for e in recent if e.get("event") == "heal.failed"]
    if len(heal_failures) >= 3:
        return DiagnosisResult(
            target_key="service_restart",
            problem_description=(
                f"The restart_service strategy has failed {len(heal_failures)} times "
                f"recently. The service detection or restart logic may need adjustment "
                f"for the specific services running on this cluster."
            ),
            metrics_summary=(
                f"heal_failures_last_10={len(heal_failures)}, "
                f"failure_rate={state.failure_rate:.3f}, "
                f"quarantined={len(state.quarantined_nodes)}"
            ),
            confidence=0.80,
        )

    # --- Pattern: jobs landing on bad nodes (low completion rate) ---
    completion_events = [e for e in recent if e.get("event") == "job.completed"]
    failure_events    = [e for e in recent if e.get("event") == "job.failed"]
    if failure_events and completion_events:
        fail_rate = len(failure_events) / (len(completion_events) + len(failure_events))
        if fail_rate > 0.3:
            return DiagnosisResult(
                target_key="job_placement",
                problem_description=(
                    f"Job failure rate is {fail_rate:.0%} in recent history. "
                    f"The node scoring function may be sending jobs to nodes "
                    f"that appear healthy but have underlying issues."
                ),
                metrics_summary=(
                    f"recent_job_failures={len(failure_events)}, "
                    f"recent_completions={len(completion_events)}, "
                    f"cluster_health={healthy_frac:.0%}"
                ),
                confidence=0.70,
            )

    # --- Pattern: RL reward stagnating ---
    reward_events = [e for e in recent if e.get("event") == "asl.patch_deployed"]
    if len(reward_events) >= 3:
        rewards = [e.get("reward", 0) for e in reward_events if "reward" in e]
        if rewards and max(rewards) - min(rewards) < 0.001:
            return DiagnosisResult(
                target_key="q_learning",
                problem_description=(
                    f"RL agent rewards have been flat ({rewards[-1]:.4f}) for "
                    f"{len(rewards)} consecutive patches. The Bellman update "
                    f"or exploration strategy may need adjustment."
                ),
                metrics_summary=(
                    f"recent_rewards={[round(r,4) for r in rewards[-3:]]}, "
                    f"objective_J={state.compute_objective():.4f}"
                ),
                confidence=0.65,
            )

    return None


# ---------------------------------------------------------------------------
# Code generator
# ---------------------------------------------------------------------------

class CodeGenerator:
    """
    Generates ΔC patches using the LLMRouter.

    Workflow:
      1. diagnose(state, history) → DiagnosisResult
      2. Read current source of target function
      3. Build prompt with problem + metrics + source
      4. Call LLM
      5. Validate syntax + safety
      6. Return CodeDelta
    """

    def __init__(self, llm_router: LLMRouter) -> None:
        self._llm     = llm_router
        self._calls   = 0
        self._patches = 0

    def generate_patch(
        self,
        state: ClusterState,
        failure_history: List[dict],
    ) -> Optional[Patch]:
        """
        Diagnose the most pressing issue and generate a code patch.
        Returns None if no good diagnosis or LLM call fails.
        """
        # Step 1: Diagnose
        diagnosis = _diagnose(state, failure_history)
        if diagnosis is None:
            log.debug("CodeGen: no clear diagnosis — skipping ΔC generation")
            return None

        if diagnosis.confidence < 0.65:
            log.debug("CodeGen: diagnosis confidence %.2f too low — skipping",
                      diagnosis.confidence)
            return None

        target = PATCH_TARGETS.get(diagnosis.target_key)
        if not target:
            return None

        # Step 2: Read current source
        current_source = _read_function_source(
            target["module_path"], target["function_name"]
        )
        if not current_source:
            return None

        # Step 3: Build prompt
        prompt = PATCH_PROMPT_TEMPLATE.format(
            function_name=target["function_name"],
            module_path=target["module_path"],
            problem_description=diagnosis.problem_description,
            metrics_summary=diagnosis.metrics_summary,
            current_source=current_source,
        )

        log.info("CodeGen: requesting ΔC patch for %s (target=%s confidence=%.2f)",
                 target["function_name"], diagnosis.target_key, diagnosis.confidence)

        # Step 4: Call LLM
        self._calls += 1
        response = self._llm.generate(
            prompt,
            system=SYSTEM_PROMPT,
            max_tokens=1200,
            temperature=0.15,   # low temperature for deterministic code
        )

        if not response.ok:
            log.warning("CodeGen: LLM call failed: %s", response.error)
            return None

        # Step 5: Validate
        generated_code = response.text.strip()

        # Strip any accidental markdown fences
        generated_code = re.sub(r"^```python\n?", "", generated_code)
        generated_code = re.sub(r"^```\n?",       "", generated_code)
        generated_code = re.sub(r"\n?```$",        "", generated_code)
        generated_code = generated_code.strip()

        syntax_ok, syntax_err = _validate_python(generated_code)
        if not syntax_ok:
            log.warning("CodeGen: generated code has syntax error: %s", syntax_err)
            return None

        loc = _count_loc(generated_code)
        if loc > 200:
            log.warning("CodeGen: patch too large (%d LOC > 200 limit)", loc)
            return None

        # Must still contain the function definition
        fn_bare = target["function_name"].split(".")[-1]
        if f"def {fn_bare}" not in generated_code:
            log.warning("CodeGen: generated code missing expected function definition")
            return None

        log.info("CodeGen: ΔC patch generated (%d LOC) via %s/%s",
                 loc, response.provider, response.model)

        # Step 6: Build CodeDelta patch
        self._patches += 1
        code_delta = CodeDelta(
            target_module=target["module_path"].replace("/", ".").rstrip(".py"),
            target_function=target["function_name"],
            diff=generated_code,
            language="python",
            loc_changed=loc,
        )

        return Patch(
            patch_type=PatchType.CODE,
            code_delta=code_delta,
            generated_by=f"llm:{response.provider}/{response.model}",
            generation_reason=(
                f"ΔC: {diagnosis.problem_description[:80]}"
            ),
        )

    def stats(self) -> dict:
        return {
            "llm_calls":      self._calls,
            "patches_generated": self._patches,
            "llm_backend_stats": self._llm.stats(),
        }
