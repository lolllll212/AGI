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

import functools
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Iterable, Sequence

from agi.domain.services.source_check import check_source
from agi.domain.value_objects.sandbox import SandboxPolicy, SandboxResult

# --------------------------------------------------------------------------- #
# the program that runs *inside* the child interpreter
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

    # -- transport --------------------------------------------------------- #
    def command(self) -> list[str]:
        """The argv that starts the isolated interpreter (the job arrives on its stdin)."""
        policy = self.policy
        argv = [*policy.launcher]
        if policy.network_isolation:
            argv += _netns_prefix()
        return argv + [policy.interpreter or sys.executable, "-I", "-S", "-B", "-c", _RUNNER]

    def _transport(self, payload: str) -> tuple[str, str, int, bool]:
        """Run one job. Returns ``(stdout, stderr, exit status, timed out)``.

        Subclasses replace this to run the same job somewhere else (a container, a Wasm runtime).
        """
        workdir = tempfile.mkdtemp(prefix="agi-sbx-")
        try:
            proc = subprocess.Popen(
                self.command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=workdir, env=self._environment(), text=True, start_new_session=True)
            try:
                out, err = proc.communicate(payload, timeout=self.policy.wall_time_s)
            except subprocess.TimeoutExpired:
                _kill_group(proc)
                proc.communicate()
                return "", "", -9, True
            return out, err, proc.returncode or 0, False
        except OSError as exc:                              # the launcher itself could not start
            return "", f"could not start the sandbox: {exc}", 127, False
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _environment(self) -> dict:
        return {}

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

        self.runs += 1
        too_slow = f"exceeded the time limit ({policy.cpu_time_s}s CPU / {policy.wall_time_s}s wall-clock)"
        started = time.perf_counter()
        out, err, status, timed_out = self._transport(payload)
        elapsed = time.perf_counter() - started
        if timed_out:
            return SandboxResult(ok=False, timed_out=True, error_type="Timeout", error=too_slow,
                                 duration_s=elapsed)

        try:
            data = json.loads(out)
            if not isinstance(data, dict) or not isinstance(data.get("ok"), bool):
                raise ValueError("malformed result")
        except ValueError:
            # Killed by a kernel limit, crashed, or wrote to the result channel itself.
            code_ = status or 0
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
