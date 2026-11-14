"""Run every job in CPython compiled to WebAssembly."""
from __future__ import annotations

import time

from agi.domain.exceptions import SandboxUnavailable
from agi.domain.value_objects.sandbox import SandboxPolicy
from agi.infrastructure.adapters.sandbox.process import _RUNNER, Sandbox


class WasmSandbox(Sandbox):
    """Runs every job in CPython compiled to WebAssembly (the ``wasi-python`` package).

    The guest has its own linear memory and can reach the host only through
    WASI calls, of which none for files, sockets or processes are granted. Time
    is bounded by a wall-clock deadline and, optionally, by ``fuel`` (a count of
    executed instructions, which makes runs deterministic to cut off).

    The first run on a machine compiles the interpreter (about a minute); later
    runs start in tens of milliseconds from the cache. Metering fuel needs a
    second, separately compiled build, so the first run *with* ``fuel`` pays
    that minute again. Call :meth:`warm_up` at start-up to pay both up front
    rather than inside a user's request.

    Starting the interpreter alone burns about 36 million units of fuel, so
    ``fuel`` below :data:`MIN_FUEL` is refused: every run would be cut off
    before the first line of the job.
    """

    name = "wasm"
    MIN_FUEL = 100_000_000

    def __init__(self, policy: SandboxPolicy | None = None, fuel: int | None = None, check: bool = True):
        if check and not self.available():
            raise SandboxUnavailable("the 'wasi-python' package is not installed (pip install wasi-python)")
        if fuel is not None and fuel < self.MIN_FUEL:
            raise ValueError(f"fuel={fuel:,} is too little: starting the interpreter uses about 36,000,000 units. "
                             f"Use at least {self.MIN_FUEL:,}, or fuel=None to rely on the time limit.")
        super().__init__(policy)
        self.fuel = fuel
        self.trapped: str | None = None                    # why the last run was cut off, if it was

    @staticmethod
    def available() -> bool:
        try:
            import wasi_python  # noqa: F401
            return True
        except Exception:
            return False

    def warm_up(self) -> float:
        """Compile the interpreter build(s) this sandbox uses, if not cached yet. Returns the seconds it took.

        About a minute per build on first use (two builds when ``fuel`` is set), near zero afterwards.
        """
        from wasi_python import run
        started = time.perf_counter()
        run(["-I", "-B", "-c", "pass"], deadline_s=600)
        if self.fuel:
            run(["-I", "-B", "-c", "pass"], deadline_s=600, fuel=max(self.fuel, self.MIN_FUEL))
        return time.perf_counter() - started

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
