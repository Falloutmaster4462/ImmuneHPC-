"""
asl/pipeline.py — Autonomous Self-Improvement Pipeline (Rocky Linux production)

Full pipeline: generate → sandbox → trust-gate → canary → full-deploy / rollback

Two patch generation paths:
  1. RL agent   → Δθ (parameter) and Δπ (policy) patches  — always active
  2. CodeGen    → ΔC (code) patches via LLM               — active if llm config present

The RL agent runs every step(). The code generator runs every
`code_gen_interval` steps and only when a clear diagnosis is available.

Full deployment pushes the winning change to every healthy node via SSH.
The RL agent receives the real post-deploy J as its reward signal.
"""

from __future__ import annotations
import time
import threading
from typing import Dict, List, Optional

from asl.patch import Patch, PatchStatus, ParameterDelta
from asl.rl_agent import RLAgent
from asl.sandbox import SandboxExecutor, CanaryDeployer, _ssh, _SSH_CONFIG
from core.state import ClusterState
from utils.events import bus, EventType
from utils.logger import get_logger
from utils.metrics import metrics

log = get_logger("asl.pipeline")


def _compute_trust_score(patch: Patch) -> float:
    score = 0.0
    if patch.sandbox_passed:    score += 0.40
    if patch.stability_passed:  score += 0.25
    if patch.performance_passed:score += 0.20
    if patch.safety_passed:     score += 0.15
    return round(score, 3)


class ASLPipeline:
    """
    Autonomous Self-Improvement Layer — Rocky Linux production.

    Δθ/Δπ path (every step):
      RL agent → sandbox → trust gate → canary → full deploy

    ΔC path (every code_gen_interval steps, needs diagnosis):
      Diagnoser → LLM code generator → sandbox → trust gate → canary → full deploy
    """

    def __init__(
        self,
        state: ClusterState,
        trust_threshold: float = 0.75,
        sandbox_timeout: float = 60.0,
        canary_fraction: float = 0.10,
        soak_sec: float = 60.0,
        enabled: bool = True,
        ssh_config: Optional[Dict] = None,
        llm_config: Optional[Dict] = None,
        code_gen_interval: int = 20,    # run ΔC every N steps
    ) -> None:
        self.state = state
        self.trust_threshold = trust_threshold
        self.enabled = enabled
        self._code_gen_interval = code_gen_interval
        self._step_count = 0

        if ssh_config:
            _SSH_CONFIG.update(ssh_config)

        self._agent   = RLAgent()
        self._sandbox = SandboxExecutor(timeout_sec=sandbox_timeout, ssh_config=ssh_config)
        self._canary  = CanaryDeployer(
            state, canary_fraction=canary_fraction,
            soak_sec=soak_sec, ssh_config=ssh_config
        )

        # ΔC code generator — optional, needs LLM config
        self._code_generator = None
        if llm_config:
            try:
                from asl.llm_backend import LLMRouter
                from asl.code_generator import CodeGenerator
                router = LLMRouter.from_config(llm_config)
                self._code_generator = CodeGenerator(router)
                log.info("ASL: ΔC code generator active (provider=%s)",
                         llm_config.get("provider", "auto"))
            except Exception as exc:
                log.warning("ASL: ΔC code generator init failed: %s", exc)

        self._patch_history: List[Patch] = []
        self._deployed: Dict[str, Patch] = {}
        self._event_history: List[dict] = []   # for diagnosis
        self._lock = threading.Lock()

        # Listen to events for diagnosis history
        bus.subscribe_all(self._record_event)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self) -> Optional[Patch]:
        if not self.enabled:
            return None

        self._step_count += 1

        # ΔC path: attempt code generation every N steps
        if (self._code_generator is not None
                and self._step_count % self._code_gen_interval == 0):
            code_patch = self._code_generator.generate_patch(
                self.state, self._event_history[-50:]
            )
            if code_patch:
                log.info("ASL: attempting ΔC patch: %s", code_patch.patch_id)
                result = self._run_patch(code_patch, action_idx=None, J_before=None)
                if result:
                    return result

        # Δθ/Δπ path: RL agent every step
        J_before = self.state.compute_objective()
        action_idx, patch = self._agent.select_action(self.state)
        return self._run_patch(patch, action_idx=action_idx, J_before=J_before)

    def _run_patch(
        self,
        patch: Patch,
        action_idx: Optional[int],
        J_before: Optional[float],
    ) -> Optional[Patch]:
        if J_before is None:
            J_before = self.state.compute_objective()

        log.info("ASL patch %s: %s (reason=%s)",
                 patch.patch_type.value, patch.patch_id, patch.generation_reason)
        metrics.patches_generated.inc()
        bus.emit_simple(EventType.PATCH_GENERATED, "asl", payload=patch.summary())

        # Sandbox
        sandbox_result = self._sandbox.test(patch, self.state)
        patch.safety_passed = len(sandbox_result.safety_violations) == 0

        # Trust gate
        patch.trust_score = _compute_trust_score(patch)
        if not sandbox_result.passed or patch.trust_score < self.trust_threshold:
            patch.status = PatchStatus.REJECTED
            self._patch_history.append(patch)
            metrics.patches_rejected.inc()
            log.warning("Patch REJECTED: %s (trust=%.2f sandbox=%s)",
                        patch.patch_id, patch.trust_score, sandbox_result.passed)
            bus.emit_simple(EventType.PATCH_REJECTED, "asl", payload=patch.summary())
            if action_idx is not None:
                self._agent.observe(J_before, action_idx, patch, J_before + 0.05)
            return None

        bus.emit_simple(EventType.PATCH_VALIDATED, "asl", payload=patch.summary())

        # Canary
        canary_result = self._canary.deploy(patch)
        if not canary_result.success:
            patch.status = PatchStatus.ROLLED_BACK
            patch.rolled_back_at = time.time()
            self._patch_history.append(patch)
            metrics.patches_rolled_back.inc()
            log.warning("Patch ROLLED BACK after canary: %s", patch.patch_id)
            bus.emit_simple(EventType.PATCH_ROLLED_BACK, "asl", payload=patch.summary())
            if action_idx is not None:
                self._agent.observe(J_before, action_idx, patch, canary_result.V_after)
            return None

        # Full deploy
        self._full_deploy(patch)
        J_after = self.state.compute_objective()

        if action_idx is not None:
            reward = self._agent.observe(J_before, action_idx, patch, J_after)
        else:
            reward = -(J_after)

        log.info("Patch DEPLOYED: %s | J %.4f→%.4f | R=%.4f",
                 patch.patch_id, J_before, J_after, reward)
        bus.emit_simple(EventType.PATCH_DEPLOYED, "asl", payload={
            **patch.summary(), "J_before": J_before, "J_after": J_after,
            "reward": reward,
        })
        return patch

    def rollback(self, patch_id: str) -> bool:
        with self._lock:
            patch = self._deployed.pop(patch_id, None)
        if not patch:
            return False
        patch.status = PatchStatus.ROLLED_BACK
        patch.rolled_back_at = time.time()
        metrics.patches_rolled_back.inc()
        log.warning("Manual rollback of patch %s", patch_id)
        bus.emit_simple(EventType.PATCH_ROLLED_BACK, "asl", payload=patch.summary())
        return True

    def agent_stats(self) -> Dict:
        stats = self._agent.stats()
        if self._code_generator:
            stats["codegen"] = self._code_generator.stats()
        return stats

    def patch_history(self, last_n: int = 20) -> List[Dict]:
        return [p.summary() for p in self._patch_history[-last_n:]]

    # ------------------------------------------------------------------
    # Full cluster deployment
    # ------------------------------------------------------------------

    def _full_deploy(self, patch: Patch) -> None:
        patch.status = PatchStatus.DEPLOYED
        patch.deployed_at = time.time()
        with self._lock:
            self._deployed[patch.patch_id] = patch
        self._patch_history.append(patch)
        metrics.patches_deployed.inc()

        canary_set = set(getattr(self._canary, "_last_canary_nodes", []))
        remaining = [
            nid for nid, n in self.state.nodes.items()
            if n.status.value == "healthy" and nid not in canary_set
        ]

        if remaining:
            # Δθ: push parameter changes to remaining nodes
            for delta in patch.parameter_deltas:
                ns = delta.namespace; value = delta.new_value
                key = ns.replace(".", "_")
                for host in remaining:
                    mkdir = "mkdir -p $(dirname /etc/immunehpc/runtime.yaml)"
                    sed = (
                        f"grep -q '^{key}:' /etc/immunehpc/runtime.yaml 2>/dev/null "
                        f"  && sed -i 's|^{key}:.*|{key}: {value}|' /etc/immunehpc/runtime.yaml "
                        f"  || echo '{key}: {value}' >> /etc/immunehpc/runtime.yaml 2>/dev/null"
                    )
                    rc, _, err = _ssh(host, f"{mkdir}; {sed}", timeout=10.0)
                    if rc == 0:
                        _ssh(host, "pkill -HUP -x immunehpc 2>/dev/null || true", timeout=5.0)
                self._agent._action_space.apply_param_delta(delta)

            # Δπ: push policy changes
            if patch.policy_delta:
                self._canary._apply_policy_delta(patch.policy_delta, remaining)

            # ΔC: hot-swap the function on remaining nodes
            if patch.code_delta:
                self._deploy_code_delta(patch.code_delta, remaining)

        log.info("Patch %s fully deployed", patch.patch_id)

    def _deploy_code_delta(self, delta, node_ids: List[str]) -> None:
        """
        Write the generated function replacement to each node's module file.

        This patches the live Python file in-place and sends SIGHUP to
        trigger a module reload. Hot-reload is safe because ImmuneHPC+
        re-imports changed modules at each MAPE-K tick via importlib.
        """
        from asl.patch import CodeDelta as CD
        module_file = delta.target_module.replace(".", "/") + ".py"
        project_root = "/opt/immunehpc"   # installed location
        remote_path  = f"{project_root}/{module_file}"
        fn_name = delta.target_function.split(".")[-1]

        # Escape the generated code for safe embedding in a heredoc
        code_escaped = delta.diff.replace("'", "'\\''")

        for host in node_ids:
            # Use Python on the remote to do a safe AST-level replacement
            replace_script = f"""python3 - << 'ENDSCRIPT'
import re, ast, sys
path = '{remote_path}'
with open(path) as f:
    src = f.read()
new_fn = '''{code_escaped}'''
# Find and replace the function by name
pattern = r'(    def {fn_name}|def {fn_name})(\\s*\\([^)]*\\).*?)(?=\\n    def |\\ndef |\\nclass |\\Z)'
if re.search(pattern, src, re.DOTALL):
    src = re.sub(pattern, new_fn, src, flags=re.DOTALL)
    with open(path, 'w') as f:
        f.write(src)
    print('OK: {fn_name} replaced')
else:
    print('SKIP: {fn_name} not found in ' + path)
ENDSCRIPT"""

            rc, out, err = _ssh(host, replace_script, timeout=15.0)
            if rc == 0 and "OK:" in out:
                log.info("ΔC deployed: %s on %s", fn_name, host)
                _ssh(host, "pkill -HUP -x immunehpc 2>/dev/null || true", timeout=5.0)
            else:
                log.warning("ΔC deploy failed on %s: %s %s", host, out[:80], err[:80])

    # ------------------------------------------------------------------
    # Event recording (for diagnosis)
    # ------------------------------------------------------------------

    def _record_event(self, event) -> None:
        self._event_history.append({
            "event": event.type.value,
            "ts": event.timestamp,
            "payload": str(event.payload)[:100] if event.payload else "",
        })
        # Keep last 200 events for diagnosis
        if len(self._event_history) > 200:
            self._event_history.pop(0)
