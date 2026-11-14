"""Run every job in a fresh, locked-down container."""
from __future__ import annotations

import dataclasses
import functools
import shutil
import subprocess
from typing import Sequence

from agi.domain.exceptions import SandboxUnavailable
from agi.domain.value_objects.sandbox import SandboxPolicy, SandboxResult
from agi.infrastructure.adapters.sandbox.process import Sandbox


class DockerSandbox(Sandbox):
    """Runs every job in a fresh, locked-down container.

    The container has no network, a read-only root filesystem, no Linux
    capabilities, an unprivileged user, and hard memory, CPU and process limits.
    Works with any Docker-compatible CLI (``binary="podman"``).
    """

    name = "docker"

    def __init__(self, policy: SandboxPolicy | None = None, image: str = "python:3.12-alpine",
                 binary: str = "docker", cpus: float = 1.0, pids: int = 64, extra_args: Sequence[str] = (),
                 check: bool = True):
        policy = policy or SandboxPolicy()
        self.image, self.binary, self.cpus, self.pids = image, binary, cpus, pids
        self.extra_args = tuple(extra_args)
        if check and not self.available(binary):
            raise SandboxUnavailable(f"'{binary}' is not installed or its daemon is not reachable")
        # The container provides the network boundary, so the host-side namespace wrapper is not used.
        super().__init__(dataclasses.replace(policy, launcher=tuple(self.launcher(policy)),
                                             interpreter="python3", network_isolation=False))

    def launcher(self, policy: SandboxPolicy) -> list[str]:
        return [self.binary, "run", "--rm", "-i",
                "--network", "none",
                "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m", "--workdir", "/tmp",
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--user", "65534:65534",
                "--memory", f"{policy.memory_mb}m", "--memory-swap", f"{policy.memory_mb}m",
                "--cpus", f"{self.cpus:g}", "--pids-limit", str(self.pids),
                "--ulimit", f"cpu={policy.cpu_time_s}", "--ulimit", "nofile=64",
                *self.extra_args, self.image]

    @staticmethod
    @functools.lru_cache(maxsize=4)
    def available(binary: str = "docker") -> bool:
        exe = shutil.which(binary)
        if not exe:
            return False
        try:
            return subprocess.run([exe, "info"], capture_output=True, timeout=10).returncode == 0
        except Exception:
            return False

    @property
    def isolation(self) -> list[str]:
        return ["static-ast", "container", "no-network", "read-only-rootfs", "cap-drop-all",
                "cgroup-limits", "audit-hook"]

    def _environment(self) -> dict:
        import os

        # The docker CLI needs to find its daemon; nothing here reaches the container.
        return {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "DOCKER_HOST", "XDG_RUNTIME_DIR")}

    def run_shell(self, script: str, shell: str = "sh") -> SandboxResult:
        """Run a shell script inside the container (same limits, no network)."""
        import time
        argv = [*self.policy.launcher, shell, "-s"]
        started = time.perf_counter()
        try:
            done = subprocess.run(argv, input=script, capture_output=True, text=True,
                                  timeout=self.policy.wall_time_s, env=self._environment())
        except subprocess.TimeoutExpired:
            return SandboxResult(ok=False, timed_out=True, error_type="Timeout",
                                 error=f"exceeded {self.policy.wall_time_s}s", duration_s=time.perf_counter() - started)
        except OSError as exc:
            return SandboxResult(ok=False, error_type="SandboxCrash", error=str(exc))
        limit = self.policy.max_output_bytes
        return SandboxResult(ok=done.returncode == 0, stdout=done.stdout[:limit], value=done.returncode,
                             error_type=None if done.returncode == 0 else "ShellError",
                             error=None if done.returncode == 0 else (done.stderr.strip()[-500:]
                                                                      or f"exit status {done.returncode}"),
                             duration_s=time.perf_counter() - started)
