"""Stronger sandbox backends: containers and WebAssembly.

All three backends run the *same* job protocol and return the same
:class:`~agi.core.sandbox.SandboxResult`, so they are interchangeable:

==========  =====================================================  ==========================
backend     isolation boundary                                     needs
==========  =====================================================  ==========================
process     subprocess + rlimits + network namespace               nothing (default)
docker      container: no network, read-only root, dropped         a Docker/Podman daemon
            capabilities, memory/CPU/pid limits, unprivileged user
wasm        WebAssembly: no host files, no network, no processes;  ``pip install wasi-python``
            fuel, memory and deadline limits
==========  =====================================================  ==========================

:func:`best_sandbox` picks the strongest backend that works on this host.
"""
from __future__ import annotations

import dataclasses
import functools
import shutil
import subprocess
import sys
from typing import Sequence

from .sandbox import _RUNNER, Sandbox, SandboxPolicy, SandboxResult, check_source


class SandboxUnavailable(RuntimeError):
    """The requested isolation backend cannot run on this host."""


# --------------------------------------------------------------------------- #
# containers
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# WebAssembly
# --------------------------------------------------------------------------- #
class WasmSandbox(Sandbox):
    """Runs every job in CPython compiled to WebAssembly (the ``wasi-python`` package).

    The guest has its own linear memory and can reach the host only through
    WASI calls, of which none for files, sockets or processes are granted. Time
    is bounded by a wall-clock deadline and, optionally, by ``fuel`` (a count of
    executed instructions, which makes runs deterministic to cut off).

    The first run on a machine compiles the interpreter (about a minute); later
    runs start in tens of milliseconds from the cache.
    """

    name = "wasm"

    def __init__(self, policy: SandboxPolicy | None = None, fuel: int | None = None, check: bool = True):
        if check and not self.available():
            raise SandboxUnavailable("the 'wasi-python' package is not installed (pip install wasi-python)")
        super().__init__(policy)
        self.fuel = fuel
        self.trapped: str | None = None                    # why the last run was cut off, if it was

    @staticmethod
    def available() -> bool:
        try:
            import wasi_python                              # noqa: F401
            return True
        except Exception:
            return False

    @property
    def isolation(self) -> list[str]:
        return ["static-ast", "wasm-memory", "no-filesystem", "no-network", "no-processes",
                "deadline", "memory-limit"] + (["fuel"] if self.fuel else []) + ["audit-hook"]

    def _transport(self, payload: str) -> tuple[str, str, int, bool]:
        from wasi_python import run
        policy = self.policy
        try:
            done = run(["-I", "-B", "-c", _RUNNER], stdin=payload.encode(), fuel=self.fuel,
                       deadline_s=policy.wall_time_s, memory_bytes=policy.memory_mb * 1024 * 1024,
                       output_bytes=max(policy.max_output_bytes * 4, 1 << 20))
        except Exception as exc:                            # the runtime itself failed
            return "", f"wasm runtime error: {type(exc).__name__}: {exc}", 126, False
        self.trapped = done.trapped
        timed_out = done.trapped in ("deadline", "out of fuel")
        status = done.exit_code if done.exit_code is not None else (-9 if done.trapped else 0)
        if done.trapped and not timed_out:
            status = -9                                     # a limit other than time: report as ResourceLimit
        return done.stdout.decode("utf-8", "replace"), done.stderr.decode("utf-8", "replace"), status, timed_out


# --------------------------------------------------------------------------- #
# choosing
# --------------------------------------------------------------------------- #
def available_backends() -> dict[str, bool]:
    return {"docker": DockerSandbox.available(), "wasm": WasmSandbox.available(), "process": True}


def best_sandbox(policy: SandboxPolicy | None = None, prefer: Sequence[str] = ("docker", "wasm", "process"),
                 require: str | None = None) -> Sandbox:
    """The strongest sandbox that works here.

    ``require="docker"`` (or ``"wasm"``) refuses to fall back: use it when
    running hostile code without a hard boundary would be worse than not running it.
    """
    makers = {"docker": lambda: DockerSandbox(policy), "wasm": lambda: WasmSandbox(policy),
              "process": lambda: Sandbox(policy)}
    order = [require] if require else list(prefer)
    for name in order:
        if name not in makers:
            raise ValueError(f"unknown sandbox backend {name!r} (use 'docker', 'wasm' or 'process')")
        try:
            return makers[name]()
        except SandboxUnavailable:
            if require:
                raise
    return Sandbox(policy)


__all__ = ["DockerSandbox", "WasmSandbox", "SandboxUnavailable", "best_sandbox", "available_backends",
           "check_source"]
