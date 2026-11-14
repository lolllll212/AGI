"""What sandboxed code may do (:class:`SandboxPolicy`) and what came back (:class:`SandboxResult`)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

DEFAULT_IMPORTS = frozenset({
    "math", "cmath", "itertools", "functools", "collections", "heapq", "bisect",
    "re", "string", "json", "fractions", "decimal", "statistics", "random",
    "dataclasses", "typing", "enum", "copy", "datetime", "textwrap", "array",
    "numbers", "abc",
})

FORBIDDEN_NAMES = frozenset({
    "eval", "exec", "compile", "open", "input", "breakpoint", "exit", "quit",
    "help", "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "memoryview", "__import__", "__builtins__", "__loader__", "__spec__",
})

# Attribute names that are never needed by honest code but are the usual first
# hop of an escape (``re.enum.sys.modules['os']``).
FORBIDDEN_ATTRS = frozenset({
    "sys", "os", "modules", "builtins", "subprocess", "socket", "importlib",
    "ctypes", "shutil", "pathlib", "system", "popen", "gi_frame", "cr_frame",
    "f_back", "f_globals", "f_locals", "f_builtins", "tb_frame", "func_globals",
})


#: Underscore attributes that honest code needs and that lead nowhere dangerous.
SAFE_DUNDERS = frozenset({"__name__", "__qualname__", "__doc__", "__init__", "__post_init__"})


@dataclass(frozen=True)
class SandboxPolicy:
    """Immutable description of what sandboxed code may do."""

    allowed_imports: frozenset = DEFAULT_IMPORTS
    wall_time_s: float = 5.0
    cpu_time_s: int = 5
    memory_mb: int = 512
    max_output_bytes: int = 32_000
    max_source_bytes: int = 50_000
    network_isolation: bool = True
    #: Extra argv prefix for a stronger jail, e.g. ``("bwrap", "--unshare-all", ...)``.
    launcher: tuple = ()
    interpreter: str = ""              # python executable inside the launcher (default: this one)

    def fingerprint(self) -> str:
        data = asdict(self)
        data["allowed_imports"] = sorted(self.allowed_imports)
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class SandboxResult:
    ok: bool
    value: Any = None
    stdout: str = ""
    error_type: str | None = None
    error: str | None = None
    traceback: str | None = None
    timed_out: bool = False
    violations: tuple = ()
    duration_s: float = 0.0
    #: ``{test_name: None | "ErrType: message"}`` for :meth:`Sandbox.run_tests`.
    tests: dict = field(default_factory=dict)
    #: Per-call outcomes for :meth:`Sandbox.call_many`.
    calls: list = field(default_factory=list)

    @property
    def failure(self) -> str | None:
        """One-line reason for failure, or ``None`` on success."""
        if self.ok:
            return None
        if self.violations:
            return "PolicyViolation: " + "; ".join(self.violations)
        if self.timed_out:
            return f"Timeout: {self.error or 'time limit exceeded'}"
        failed = {k: v for k, v in self.tests.items() if v}
        if failed and not self.error_type:
            name, why = next(iter(failed.items()))
            return f"TestFailed: {name}: {why}"
        return f"{self.error_type or 'Error'}: {self.error or ''}".strip()
