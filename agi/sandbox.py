"""Sandboxed execution of untrusted (usually model-written) Python.

Every other subsystem leans on this module: the reasoner uses it to *test*
hypotheses instead of trusting them, the world bridge uses it as an environment
whose consequences can be observed, and the tool forge uses it so agents can
write code for themselves without that code touching the host.

Defence is layered, cheapest first:

1. **Static gate** (:func:`check_source`) - AST allowlist of imports, no
   ``eval``/``exec``/``open``/``getattr``..., no underscore-attribute access.
2. **Process isolation** - a fresh ``python -I -S -B`` interpreter with an empty
   environment, a throwaway working directory and its own process group.
3. **Kernel limits** - CPU seconds, address space, zero file size, no ``fork``.
4. **Network namespace** - ``unshare --user --net`` when the host allows it.
5. **Runtime gate** - restricted builtins, an import allowlist and a PEP 578
   audit hook that vetoes file, process, socket and ctypes events.

.. warning::
   Layers 1 and 5 live inside CPython and are *not* a security boundary against
   a determined adversary; layers 2-4 are the real boundary and they are only
   as strong as the host kernel. For hostile code add a container, gVisor or a
   VM via :attr:`SandboxPolicy.launcher`.
"""
from __future__ import annotations

import ast
import functools
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

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


# --------------------------------------------------------------------------- #
# Layer 1: static gate
# --------------------------------------------------------------------------- #
def check_source(code: str, policy: SandboxPolicy | None = None) -> list[str]:
    """Return policy violations found by static analysis (empty list == clean)."""
    policy = policy or SandboxPolicy()
    if len(code.encode("utf-8", "replace")) > policy.max_source_bytes:
        return [f"source larger than {policy.max_source_bytes} bytes"]
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax error: {exc.msg} (line {exc.lineno})"]
    except (ValueError, RecursionError, MemoryError) as exc:
        return [f"unparseable source: {type(exc).__name__}"]

    found: list[str] = []

    def flag(node: ast.AST, message: str) -> None:
        text = f"line {getattr(node, 'lineno', '?')}: {message}"
        if text not in found:
            found.append(text)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in policy.allowed_imports:
                    flag(node, f"import of '{alias.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level or root not in policy.allowed_imports:
                flag(node, f"import from '{node.module or '.'}' is not allowed")
            if any(a.name == "*" for a in node.names):
                flag(node, "star imports are not allowed")
            for alias in node.names:
                if alias.name.startswith("_") or alias.name in FORBIDDEN_ATTRS:
                    flag(node, f"import of name '{alias.name}' is not allowed")
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES:
                flag(node, f"use of '{node.id}' is not allowed")
            elif node.id.startswith("__") and node.id not in {"__name__", "__doc__"}:
                flag(node, f"use of dunder name '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr in SAFE_DUNDERS:
                continue
            if node.attr.startswith("_"):
                flag(node, f"access to private attribute '{node.attr}' is not allowed")
            elif node.attr in FORBIDDEN_ATTRS:
                flag(node, f"access to attribute '{node.attr}' is not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                if name.startswith("__"):
                    flag(node, f"rebinding '{name}' is not allowed")
    return found


# --------------------------------------------------------------------------- #
# Layers 3 + 5: the program that runs *inside* the child interpreter
# --------------------------------------------------------------------------- #
_RUNNER = r'''
import builtins, io, json, sys, traceback
real_out = sys.stdout
job = json.loads(sys.stdin.read())
try:
    import resource
    def cap(what, soft, hard=None):
        try: resource.setrlimit(what, (soft, soft if hard is None else hard))
        except Exception: pass
    cap(resource.RLIMIT_CPU, job["cpu"], job["cpu"] + 1)
    cap(resource.RLIMIT_AS, job["mem"])
    cap(resource.RLIMIT_FSIZE, 0)
    cap(resource.RLIMIT_CORE, 0)
    cap(resource.RLIMIT_NPROC, 0)
except ImportError:
    pass

ALLOWED = frozenset(job["imports"])
LIMIT = job["out"]
STDLIB = tuple(p for p in sys.path if p)
real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split(".")[0] not in ALLOWED:
        raise ImportError("import of %r is blocked by the sandbox policy" % name)
    return real_import(name, globals, locals, fromlist, level)

class Capture(io.StringIO):
    def write(self, text):
        room = LIMIT - self.tell()
        return super().write(text[:room]) if room > 0 else 0

DENIED_BUILTINS = ("eval exec compile open input breakpoint exit quit help globals locals "
                   "vars getattr setattr delattr memoryview copyright credits license").split()
safe = {k: v for k, v in vars(builtins).items() if k not in DENIED_BUILTINS}
safe["__import__"] = guarded_import

DENIED_EVENTS = ("os.", "subprocess.", "socket.", "ctypes.", "shutil.", "urllib.", "http.",
                 "ftplib.", "smtplib.", "sqlite3.", "mmap.", "fcntl.", "pty.", "tempfile.",
                 "glob.", "pathlib.", "resource.", "syslog.", "webbrowser.", "signal.",
                 "cpython.run_", "sys.addaudithook", "sys.setprofile", "sys.settrace",
                 "code.__new__", "gc.", "pickle.")
DENIED_MODULES = frozenset("ctypes _ctypes socket _socket subprocess _posixsubprocess "
                           "multiprocessing mmap pty ssl _ssl asyncio select selectors".split())

def audit(event, args):
    if event == "open":
        path, mode = args[0], args[1]
        if isinstance(path, str) and path.startswith(STDLIB) and (mode is None or mode in ("r", "rb")):
            return
        raise PermissionError("sandbox: file access is blocked (%r)" % (path,))
    if event == "import":
        if args[0].split(".")[0] in DENIED_MODULES:
            raise ImportError("sandbox: module %r is blocked" % (args[0],))
        return
    if event in ("os.listdir", "os.scandir"):
        if isinstance(args[0], str) and args[0].startswith(STDLIB):
            return
    if event.startswith(DENIED_EVENTS):
        raise PermissionError("sandbox: %s is blocked" % event)

def describe(exc):
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename in ("<code>", "<tests>")]
    where = ["%s line %d: %s" % (f.filename, f.lineno, (f.line or "").strip()) for f in frames[-4:]]
    return {"error_type": type(exc).__name__, "error": str(exc)[:2000], "traceback": "\n".join(where)}

def source_lines(name, text):
    import linecache
    linecache.cache[name] = (len(text), None, text.splitlines(True), name)

out = {"ok": False, "value": None, "tests": {}, "calls": []}
buf = Capture()
sys.stdout = sys.stderr = buf
ns = {"__builtins__": safe, "__name__": "__sandbox__"}
source_lines("<code>", job["code"])
program = compile(job["code"], "<code>", "exec")
checks = None
if job["mode"] == "tests":
    source_lines("<tests>", job["tests"])
    checks = compile(job["tests"], "<tests>", "exec")
sys.addaudithook(audit)
try:
    exec(program, ns)
    mode = job["mode"]
    if mode == "exec":
        out["value"] = ns.get("result")
        out["ok"] = True
    elif mode == "call":
        out["value"] = ns[job["func"]](*job["args"], **job["kwargs"])
        out["ok"] = True
    elif mode == "calls":
        def attempt(fn, args):
            try:
                return {"ok": True, "value": fn(*args)}
            except BaseException as exc:
                return dict(describe(exc), ok=False, value=None)
        if isinstance(job["func"], list):
            out["calls"] = [[attempt(ns[name], args) for args in job["args"]] for name in job["func"]]
        else:
            out["calls"] = [attempt(ns[job["func"]], args) for args in job["args"]]
        out["ok"] = True
    elif mode == "tests":
        exec(checks, ns)
        names = [n for n, v in list(ns.items()) if n.startswith("test_") and callable(v)]
        for name in names:
            try:
                ns[name]()
                out["tests"][name] = None
            except BaseException as exc:
                info = describe(exc)
                out["tests"][name] = (info["error_type"] + ": " + info["error"]).strip(": ")
        if not names:
            out.update(error_type="NoTests", error="no test_* functions were defined")
        out["ok"] = bool(names) and not any(out["tests"].values())
except BaseException as exc:
    out.update(describe(exc))
out["stdout"] = buf.getvalue()
try:
    text = json.dumps(out, default=repr)
except Exception as exc:
    text = json.dumps({"ok": False, "error_type": "Unserializable", "error": str(exc)[:500]})
real_out.write(text)
real_out.flush()
'''


@functools.lru_cache(maxsize=1)
def _netns_prefix() -> tuple:
    """``unshare`` argv prefix if unprivileged network namespaces work here."""
    exe = shutil.which("unshare")
    if not exe:
        return ()
    prefix = (exe, "--user", "--map-root-user", "--net", "--")
    try:
        done = subprocess.run([*prefix, sys.executable, "-I", "-S", "-c", "pass"],
                              capture_output=True, timeout=5)
        return prefix if done.returncode == 0 else ()
    except Exception:
        return ()


class Sandbox:
    """Run code under a :class:`SandboxPolicy` and report what actually happened."""

    def __init__(self, policy: SandboxPolicy | None = None):
        self.policy = policy or SandboxPolicy()
        self.runs = 0

    @property
    def isolation(self) -> list[str]:
        """Names of the isolation layers active on this host."""
        layers = ["static-ast", "subprocess", "audit-hook"]
        if os.name == "posix":
            layers.append("rlimits")
        if self.policy.network_isolation and _netns_prefix():
            layers.append("netns")
        if self.policy.launcher:
            layers.append("launcher")
        return layers

    # -- public API --------------------------------------------------------- #
    def run(self, code: str) -> SandboxResult:
        """Execute ``code``; ``value`` is the module-level name ``result`` if set."""
        return self._launch(code, {"mode": "exec"})

    def call(self, code: str, func: str, args: Sequence = (), kwargs: dict | None = None) -> SandboxResult:
        """Execute ``code`` then call ``func(*args, **kwargs)`` inside the sandbox."""
        return self._launch(code, {"mode": "call", "func": func, "args": list(args),
                                   "kwargs": kwargs or {}})

    def call_many(self, code: str, func: str, arg_lists: Iterable[Sequence]) -> SandboxResult:
        """Call ``func`` once per argument list in a single sandbox process."""
        return self._launch(code, {"mode": "calls", "func": func,
                                   "args": [list(a) for a in arg_lists]})

    def call_each(self, code: str, funcs: Sequence[str], arg_lists: Iterable[Sequence]) -> SandboxResult:
        """Call every function in ``funcs`` on every argument list, in one process.

        ``result.calls[i][j]`` is the outcome of ``funcs[i](*arg_lists[j])``.
        """
        return self._launch(code, {"mode": "calls", "func": list(funcs),
                                   "args": [list(a) for a in arg_lists]})

    def run_tests(self, code: str, tests: str) -> SandboxResult:
        """Execute ``code`` then every ``test_*`` function defined by ``tests``."""
        return self._launch(code, {"mode": "tests", "tests": tests}, also_check=tests)

    # -- internals ---------------------------------------------------------- #
    def _launch(self, code: str, job: dict, also_check: str | None = None) -> SandboxResult:
        policy = self.policy
        violations = check_source(code, policy)
        if also_check is not None:
            violations += [f"tests: {v}" for v in check_source(also_check, policy)]
        if violations:
            return SandboxResult(ok=False, violations=tuple(violations),
                                 error_type="PolicyViolation", error="; ".join(violations))
        try:
            payload = json.dumps(dict(
                job, code=code, cpu=policy.cpu_time_s, mem=policy.memory_mb * 1024 * 1024,
                out=policy.max_output_bytes, imports=sorted(policy.allowed_imports)))
        except (TypeError, ValueError) as exc:
            return SandboxResult(ok=False, error_type="BadArguments", error=str(exc))

        argv = [*policy.launcher]
        if policy.network_isolation:
            argv += _netns_prefix()
        argv += [sys.executable, "-I", "-S", "-B", "-c", _RUNNER]
        self.runs += 1
        too_slow = f"exceeded the time limit ({policy.cpu_time_s}s CPU / {policy.wall_time_s}s wall-clock)"
        started = time.perf_counter()
        workdir = tempfile.mkdtemp(prefix="agi-sbx-")
        try:
            proc = subprocess.Popen(
                argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=workdir, env={}, text=True, start_new_session=True)
            try:
                out, err = proc.communicate(payload, timeout=policy.wall_time_s)
            except subprocess.TimeoutExpired:
                _kill_group(proc)
                proc.communicate()
                return SandboxResult(ok=False, timed_out=True, error_type="Timeout", error=too_slow,
                                     duration_s=time.perf_counter() - started)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        elapsed = time.perf_counter() - started

        try:
            data = json.loads(out)
            if not isinstance(data, dict) or not isinstance(data.get("ok"), bool):
                raise ValueError("malformed result")
        except ValueError:
            # Killed by a kernel limit, crashed, or wrote to the result channel itself.
            code_ = proc.returncode or 0
            if code_ == -getattr(signal, "SIGXCPU", 24):
                return SandboxResult(ok=False, timed_out=True, error_type="Timeout", error=too_slow,
                                     duration_s=elapsed)
            reason = "ResourceLimit" if code_ < 0 else "SandboxCrash"
            return SandboxResult(ok=False, error_type=reason, duration_s=elapsed,
                                 error=f"no valid result (exit status {code_}) "
                                       f"{err.strip()[-300:]}".strip())
        return SandboxResult(
            ok=data["ok"], value=data.get("value"), stdout=data.get("stdout", ""),
            error_type=data.get("error_type"), error=data.get("error"),
            traceback=data.get("traceback"), duration_s=elapsed,
            tests=data.get("tests") or {}, calls=data.get("calls") or [])


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, AttributeError):
        proc.kill()
