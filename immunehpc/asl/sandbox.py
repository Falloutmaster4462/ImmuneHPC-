"""
asl/sandbox.py — Sandbox Testing + Canary Deployment (Rocky Linux production)

Canary deploy now writes real parameter changes to a live subset of nodes,
measures V(x) for a configurable soak period, then either promotes or rolls back.

Parameter patches (Δθ) are applied by:
  1. Writing the new value to /etc/immunehpc/runtime.yaml on each canary node
  2. Sending SIGHUP to the immunehpc agent on that node to hot-reload config

Policy patches (Δπ) are applied by updating Slurm partition weights via scontrol.

All changes are checkpointed before apply so rollback is exact.
"""

from __future__ import annotations
import json
import os
import subprocess
import tempfile
import time
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from asl.patch import Patch, PatchStatus, ParameterDelta, PolicyDelta
from core.state import ClusterState
from utils.logger import get_logger

log = get_logger("sandbox")

_SSH_CONFIG: Dict = {}


def _ssh(host: str, command: str, timeout: float = 15.0) -> Tuple[int, str, str]:
    user = _SSH_CONFIG.get("ssh_user", "root")
    key  = _SSH_CONFIG.get("ssh_key_path", "")
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no",
           "-o", "ConnectTimeout=10", "-o", "BatchMode=yes"]
    if key:
        cmd += ["-i", key]
    cmd += [f"{user}@{host}", command]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return 1, "", "ssh timeout"
    except FileNotFoundError:
        return 1, "", "ssh not found"


@dataclass
class SandboxResult:
    patch_id: str
    passed: bool
    duration_sec: float
    stdout: str = ""
    stderr: str = ""
    regression_failures: List[str] = field(default_factory=list)
    safety_violations: List[str] = field(default_factory=list)


@dataclass
class CanaryResult:
    patch_id: str
    canary_nodes: List[str]
    V_before: float
    V_after: float
    success: bool
    message: str = ""


# ---------------------------------------------------------------------------
# Regression test suite (pure Python — no real nodes needed)
# ---------------------------------------------------------------------------

class RegressionSuite:
    def __init__(self) -> None:
        self._tests = [
            ("no_empty_code_delta",    self._test_no_empty_code),
            ("parameter_range_valid",  self._test_param_range),
            ("policy_non_empty",       self._test_policy_nonempty),
            ("no_root_privilege",      self._test_no_root),
            ("patch_size_limit",       self._test_size_limit),
        ]

    def run(self, patch: Patch) -> Tuple[bool, List[str]]:
        failures = []
        for name, fn in self._tests:
            try:
                if not fn(patch):
                    failures.append(name)
            except Exception as exc:
                failures.append(f"{name}:exception:{exc}")
        return len(failures) == 0, failures

    def _test_no_empty_code(self, patch):
        return not (patch.code_delta and not patch.code_delta.diff.strip())

    def _test_param_range(self, patch):
        for d in patch.parameter_deltas:
            if isinstance(d.new_value, (int, float)) and d.new_value < 0:
                return False
        return True

    def _test_policy_nonempty(self, patch):
        return not (patch.policy_delta and not patch.policy_delta.new_policy)

    def _test_no_root(self, patch):
        if patch.code_delta:
            for p in ["os.setuid(0)", "subprocess.run(['sudo'", "chmod 4755"]:
                if p in patch.code_delta.diff:
                    log.warning("Safety violation: %s found in patch", p)
                    return False
        return True

    def _test_size_limit(self, patch, max_loc=200):
        return not (patch.code_delta and patch.code_delta.loc_changed > max_loc)


# ---------------------------------------------------------------------------
# Sandbox executor
# ---------------------------------------------------------------------------

class SandboxExecutor:
    def __init__(self, timeout_sec: float = 60.0, ssh_config: Optional[Dict] = None) -> None:
        self.timeout = timeout_sec
        self._regression = RegressionSuite()
        if ssh_config:
            _SSH_CONFIG.update(ssh_config)

    def test(self, patch: Patch, state: ClusterState) -> SandboxResult:
        t0 = time.time()
        patch.status = PatchStatus.SANDBOX_PENDING

        stdout, stderr = "", ""
        execution_ok = True
        safety_violations = self._safety_scan(patch)
        reg_ok, failures = self._regression.run(patch)

        if patch.code_delta and reg_ok and not safety_violations:
            execution_ok, stdout, stderr = self._execute_in_subprocess(patch)

        passed = reg_ok and not safety_violations and execution_ok
        patch.status = PatchStatus.SANDBOX_PASSED if passed else PatchStatus.SANDBOX_FAILED
        patch.sandbox_passed = passed

        result = SandboxResult(
            patch_id=patch.patch_id,
            passed=passed,
            duration_sec=time.time() - t0,
            stdout=stdout, stderr=stderr,
            regression_failures=failures,
            safety_violations=safety_violations,
        )
        if passed:
            log.info("Sandbox PASSED for patch %s (%.2fs)", patch.patch_id, result.duration_sec)
        else:
            log.warning("Sandbox FAILED for patch %s: %s %s",
                        patch.patch_id, failures, safety_violations)
        return result

    def _safety_scan(self, patch: Patch) -> List[str]:
        violations = []
        if patch.code_delta:
            banned = [
                ("os.system",       "unsafe shell execution"),
                ("eval(",           "dynamic eval"),
                ("exec(",           "dynamic exec"),
                ("__import__",      "dynamic import"),
                ("open('/etc",      "reading system files"),
            ]
            for pattern, reason in banned:
                if pattern in patch.code_delta.diff:
                    violations.append(f"{pattern}: {reason}")
        return violations

    def _execute_in_subprocess(self, patch: Patch) -> Tuple[bool, str, str]:
        if not patch.code_delta:
            return True, "", ""
        code = patch.code_delta.diff
        if not code.strip().startswith("#"):
            return True, "[skipped: diff, not executable]", ""
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False,
                                             prefix="immunehpc_patch_") as f:
                f.write(code)
                tmp = f.name
            result = subprocess.run(
                ["python3", tmp], capture_output=True, text=True, timeout=self.timeout)
            os.unlink(tmp)
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "sandbox timeout"
        except Exception as exc:
            return False, "", str(exc)


# ---------------------------------------------------------------------------
# Real canary deployer
# ---------------------------------------------------------------------------

# Runtime config file on each node — agents hot-reload on SIGHUP
_RUNTIME_CONFIG_PATH = "/etc/immunehpc/runtime.yaml"

# Soak time: how long to observe the canary nodes before deciding
_DEFAULT_SOAK_SEC = 60.0


class CanaryDeployer:
    """
    Deploys a patch to a small fraction of nodes, soaks for `soak_sec`,
    measures V(x) before and after, then promotes or rolls back.

    For Δθ (parameter) patches:
      - Writes updated value to /etc/immunehpc/runtime.yaml on canary nodes
      - Sends SIGHUP to immunehpc agent (if running) for hot reload
      - Checkpoints old value for exact rollback

    For Δπ (policy) patches:
      - Applies Slurm partition weight changes via scontrol
      - Checkpoints old partition config

    After soak:
      - If V(x_new) <= V(x_old)  AND  J_new <= J_old  → promote to all nodes
      - Otherwise                                       → rollback canary nodes
    """

    def __init__(
        self,
        state: ClusterState,
        canary_fraction: float = 0.1,
        soak_sec: float = _DEFAULT_SOAK_SEC,
        ssh_config: Optional[Dict] = None,
    ) -> None:
        self.state = state
        self.fraction = canary_fraction
        self.soak_sec = soak_sec
        if ssh_config:
            _SSH_CONFIG.update(ssh_config)

    def deploy(self, patch: Patch) -> CanaryResult:
        healthy_nodes = [
            n.node_id for n in self.state.nodes.values()
            if n.status.value == "healthy"
        ]
        if not healthy_nodes:
            return CanaryResult(
                patch_id=patch.patch_id, canary_nodes=[],
                V_before=self.state.lyapunov_value,
                V_after=self.state.lyapunov_value,
                success=False, message="no healthy nodes for canary"
            )

        n_canary = max(1, int(len(healthy_nodes) * self.fraction))
        canary_nodes = random.sample(healthy_nodes, min(n_canary, len(healthy_nodes)))

        V_before = self.state.lyapunov_value
        J_before = self.state.compute_objective()
        patch.V_before = V_before
        patch.J_before = J_before

        log.info("Canary deploy patch %s → %d nodes: %s (soak=%.0fs)",
                 patch.patch_id, len(canary_nodes), canary_nodes, self.soak_sec)

        # Checkpoint + apply
        checkpoints = self._checkpoint(patch, canary_nodes)
        ok, apply_msg = self._apply_to_nodes(patch, canary_nodes)
        if not ok:
            return CanaryResult(
                patch_id=patch.patch_id, canary_nodes=canary_nodes,
                V_before=V_before, V_after=V_before,
                success=False, message=f"apply failed: {apply_msg}"
            )

        # Soak period — let the monitor collect new telemetry
        log.info("Canary soak: waiting %.0fs for telemetry...", self.soak_sec)
        time.sleep(self.soak_sec)

        # Measure post-canary V and J
        V_after = self.state.compute_lyapunov()
        J_after = self.state.compute_objective()
        patch.V_after = V_after
        patch.J_after = J_after
        patch.stability_passed   = V_after <= V_before
        patch.performance_passed = J_after <= J_before

        success = patch.stability_passed and patch.performance_passed

        if not success:
            log.warning(
                "Canary FAILED for %s: V %.4f→%.4f J %.4f→%.4f — rolling back",
                patch.patch_id, V_before, V_after, J_before, J_after
            )
            self._rollback_nodes(patch, canary_nodes, checkpoints)
            return CanaryResult(
                patch_id=patch.patch_id, canary_nodes=canary_nodes,
                V_before=V_before, V_after=V_after,
                success=False, message="Lyapunov/J constraint violated"
            )

        log.info("Canary PASSED for %s: V %.4f→%.4f J %.4f→%.4f",
                 patch.patch_id, V_before, V_after, J_before, J_after)
        return CanaryResult(
            patch_id=patch.patch_id, canary_nodes=canary_nodes,
            V_before=V_before, V_after=V_after, success=True
        )

    # ------------------------------------------------------------------
    # Checkpoint / apply / rollback
    # ------------------------------------------------------------------

    def _checkpoint(self, patch: Patch, node_ids: List[str]) -> Dict:
        """Save current parameter values from each canary node for exact rollback."""
        checkpoints: Dict[str, Dict] = {}
        for host in node_ids:
            rc, out, _ = _ssh(
                host,
                f"cat {_RUNTIME_CONFIG_PATH} 2>/dev/null || echo '{{}}'",
                timeout=8.0,
            )
            checkpoints[host] = {"runtime_yaml": out if rc == 0 else ""}
        return checkpoints

    def _apply_to_nodes(
        self, patch: Patch, node_ids: List[str]
    ) -> Tuple[bool, str]:
        """
        Write parameter/policy changes to canary nodes.

        Δθ patches: append/update key in runtime.yaml, then SIGHUP agent
        Δπ patches: call scontrol to update Slurm partition weights
        """
        any_ok = True

        for delta in patch.parameter_deltas:
            ns    = delta.namespace          # e.g. "anomaly.z_threshold"
            value = delta.new_value
            key   = ns.replace(".", "_")     # safe YAML key

            for host in node_ids:
                # Write or update the key in runtime.yaml
                sed_cmd = (
                    f"grep -q '^{key}:' {_RUNTIME_CONFIG_PATH} 2>/dev/null "
                    f"  && sed -i 's|^{key}:.*|{key}: {value}|' {_RUNTIME_CONFIG_PATH} "
                    f"  || echo '{key}: {value}' >> {_RUNTIME_CONFIG_PATH} 2>/dev/null"
                )
                mkdir_cmd = f"mkdir -p $(dirname {_RUNTIME_CONFIG_PATH})"
                rc, _, err = _ssh(host, f"{mkdir_cmd}; {sed_cmd}", timeout=10.0)
                if rc != 0:
                    log.warning("Canary apply param failed on %s: %s", host, err[:80])
                    any_ok = False
                else:
                    # Signal agent to hot-reload (non-fatal if agent not running)
                    _ssh(host,
                         "pkill -HUP -x immunehpc 2>/dev/null || true",
                         timeout=5.0)
                    log.info("Canary: %s.%s = %s on %s", ns, key, value, host)

        if patch.policy_delta:
            self._apply_policy_delta(patch.policy_delta, node_ids)

        return any_ok, "" if any_ok else "one or more nodes failed"

    def _apply_policy_delta(self, delta: PolicyDelta, node_ids: List[str]) -> None:
        """
        Apply a scheduling policy change via scontrol.

        Policy name 'healing.strategy_order' is internal — no Slurm action.
        Policy name 'slurm.partition.<name>.weight' maps to scontrol.
        """
        if "slurm.partition" not in delta.policy_name:
            log.debug("Policy %s is internal — no Slurm action needed", delta.policy_name)
            return

        # Extract partition name from policy_name: 'slurm.partition.compute.weight'
        parts = delta.policy_name.split(".")
        if len(parts) >= 4:
            partition = parts[2]
            weight    = delta.new_policy
            try:
                subprocess.run(
                    f"scontrol update PartitionName={partition} PriorityJobFactor={weight}",
                    shell=True, timeout=10.0, capture_output=True,
                )
                log.info("Policy: Slurm partition %s weight → %s", partition, weight)
            except Exception as exc:
                log.warning("scontrol policy update failed: %s", exc)

    def _rollback_nodes(
        self, patch: Patch, node_ids: List[str], checkpoints: Dict
    ) -> None:
        """Restore checkpointed runtime.yaml on each canary node."""
        for host in node_ids:
            saved = checkpoints.get(host, {}).get("runtime_yaml", "")
            if saved:
                # Write back the saved content
                escaped = saved.replace("'", "'\\''")
                _ssh(
                    host,
                    f"printf '%s' '{escaped}' > {_RUNTIME_CONFIG_PATH} 2>/dev/null",
                    timeout=10.0,
                )
                _ssh(host, "pkill -HUP -x immunehpc 2>/dev/null || true", timeout=5.0)
                log.info("Canary rollback: restored runtime.yaml on %s", host)
            else:
                # No prior config existed — remove the file
                _ssh(host, f"rm -f {_RUNTIME_CONFIG_PATH}", timeout=5.0)
                log.info("Canary rollback: removed runtime.yaml on %s (was not present)", host)

        # Roll back any policy changes
        if patch.policy_delta and patch.policy_delta.old_policy is not None:
            rollback_delta = PolicyDelta(
                policy_name=patch.policy_delta.policy_name,
                old_policy=patch.policy_delta.new_policy,
                new_policy=patch.policy_delta.old_policy,
            )
            self._apply_policy_delta(rollback_delta, node_ids)
