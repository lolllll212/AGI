"""Self-healing execution: when code fails, find out why, patch it, run it again.

The loop is: execute -> on failure capture the error and traceback -> *diagnose*
a root cause -> produce a patch -> re-execute, up to ``max_retries`` times.

Patches come from two places, cheapest first:

* **rule fixers** for failures with an unambiguous cure (a missing ``import``,
  a misspelt name, tabs mixed with spaces, a method from another language);
* **a model**, if one is given, shown the code, the error, the diagnosis and
  every patch already tried.

Two things keep the loop honest. A patch only counts if the code then *passes*
(runs clean, and passes the tests if there are any). And the loop stops early
when a patch repeats an earlier attempt or the error stops changing - retrying
the same thing is not healing.
"""
from __future__ import annotations

import ast
import builtins
import difflib
import re
from typing import Any, Callable, Mapping

from pydantic import Field

from ..core.llm import extract_code
from ..core.types import Model
from .base import Param, Tool, ToolError

_METHOD_FIXES = {  # habits from other languages -> Python
    ("list", "push"): "append", ("list", "add"): "append", ("list", "length"): None,
    ("str", "contains"): None, ("list", "size"): None, ("dict", "has_key"): None,
    ("str", "trim"): "strip", ("str", "toUpperCase"): "upper", ("str", "toLowerCase"): "lower",
    ("str", "startsWith"): "startswith", ("str", "endsWith"): "endswith", ("list", "indexOf"): "index",
    ("set", "append"): "add", ("dict", "iteritems"): "items", ("dict", "iterkeys"): "keys",
    ("dict", "itervalues"): "values",
}

ROOT_CAUSES = {
    "missing_import": "a module is used but never imported",
    "disallowed_import": "the code imports a module the sandbox does not allow",
    "name_typo": "a name is misspelt",
    "undefined_name": "a name is used before anything defines it",
    "wrong_method": "a method that does not exist on this type is called",
    "syntax": "the code is not valid Python",
    "indentation": "the indentation is inconsistent",
    "type_mismatch": "a value has a different type from what the operation needs",
    "bad_index": "an index or key is outside what the container holds",
    "empty_input": "the code assumes a non-empty sequence",
    "division_by_zero": "a divisor can be zero",
    "timeout": "the code does not finish in time (an infinite loop, or too slow an algorithm)",
    "recursion": "the recursion never reaches its base case, or is too deep",
    "memory": "the code needs more memory than is allowed",
    "wrong_result": "the code runs but gives the wrong answer",
    "policy": "the code does something the sandbox forbids",
    "unknown": "the cause is not one of the known patterns",
}


class Diagnosis(Model):
    error_type: str
    message: str
    root_cause: str
    explanation: str
    line: int | None = None
    symbol: str | None = None            # the name, module or attribute at fault
    suggestion: str = ""


class HealAttempt(Model):
    attempt: int
    code: str
    ok: bool
    error_type: str | None = None
    error: str | None = None
    diagnosis: Diagnosis | None = None
    fixer: str | None = None             # what produced the *next* version: "rule:<name>" or "model"


class HealResult(Model):
    ok: bool
    code: str                            # the last version tried (the working one if ok)
    value: Any = None
    stdout: str = ""
    attempts: list[HealAttempt] = Field(default_factory=list)
    stopped: str = ""                    # why the loop ended

    @property
    def repaired(self) -> bool:
        return self.ok and len(self.attempts) > 1

    @property
    def fixes(self) -> list[str]:
        return [a.fixer for a in self.attempts if a.fixer]

    def summary(self) -> str:
        if not self.attempts:
            return "not run"
        if self.ok and not self.repaired:
            return "ran cleanly"
        chain = "; ".join(f"{a.diagnosis.root_cause} ({a.error_type}) -> {a.fixer}"
                          for a in self.attempts if a.diagnosis and a.fixer)
        if self.ok:
            return f"self-healed after {len(self.attempts) - 1} repair(s): {chain}"
        last = self.attempts[-1]
        return (f"still failing after {len(self.attempts)} attempt(s) [{self.stopped}]: "
                f"{last.error_type}: {last.error}" + (f" (repairs tried: {chain})" if chain else ""))


# --------------------------------------------------------------------------- #
# diagnosis
# --------------------------------------------------------------------------- #
def _line_of(traceback: str | None, message: str) -> int | None:
    found = re.findall(r'File "<(?:sandbox|string|tests)>", line (\d+)', traceback or "")
    if found:
        return int(found[-1])
    match = re.search(r"\bline (\d+)", message or "")
    return int(match.group(1)) if match else None


def _defined_names(code: str) -> set[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {name for pair in re.findall(r"\b(?:def|class)\s+(\w+)|^\s*(\w+)\s*=", code, re.M)
                for name in pair if name}
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            if not isinstance(node, ast.ClassDef):
                names |= {a.arg for a in node.args.args + node.args.kwonlyargs}
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(alias.asname or alias.name).split(".")[0] for alias in node.names}
    return names


def diagnose(code: str, error_type: str | None, message: str | None, traceback: str | None = None,
             allowed_imports: frozenset | set = frozenset()) -> Diagnosis:
    """Classify a failure into a root cause, naming the symbol and line where possible."""
    error_type, message = error_type or "Error", message or ""
    if error_type == "PolicyViolation" and message.startswith("syntax error:"):
        # The sandbox's static gate reports unparsable code as a violation; name the real problem.
        indent = re.search(r"\bindent|tabs and spaces", message)
        error_type = "IndentationError" if indent else "SyntaxError"
    line = _line_of(traceback, message)

    def make(cause: str, symbol: str | None = None, suggestion: str = "") -> Diagnosis:
        return Diagnosis(error_type=error_type, message=message[:500], root_cause=cause,
                         explanation=ROOT_CAUSES[cause], line=line, symbol=symbol, suggestion=suggestion)

    if error_type == "NameError":
        match = re.search(r"name '(\w+)' is not defined", message)
        symbol = match.group(1) if match else None
        if symbol and symbol in allowed_imports:
            return make("missing_import", symbol, f"add `import {symbol}` at the top")
        if symbol:
            candidates = (_defined_names(code) | set(dir(builtins))) - {symbol}
            close = difflib.get_close_matches(symbol, sorted(candidates), 1, 0.75)
            if close:
                return make("name_typo", symbol, f"`{symbol}` is probably `{close[0]}`")
        return make("undefined_name", symbol, f"define `{symbol}` before using it" if symbol else "")
    if error_type in ("ModuleNotFoundError", "ImportError") or (
            error_type == "PolicyViolation" and "import of" in message):
        match = re.search(r"import of '([\w.]+)'|No module named '([\w.]+)'|module '([\w.]+)'", message)
        symbol = next((g for g in (match.groups() if match else ()) if g), None)
        return make("disallowed_import", symbol,
                    "use only the allowed modules: " + ", ".join(sorted(allowed_imports)) if allowed_imports else "")
    if error_type == "PolicyViolation":
        return make("policy", None, "remove the forbidden construct; files, network and eval are not available")
    if error_type in ("IndentationError", "TabError"):
        return make("indentation", None, "indent with four spaces consistently")
    if error_type == "SyntaxError":
        return make("syntax", None, "fix the syntax on the reported line")
    if error_type == "AttributeError":
        match = re.search(r"'(\w+)' object has no attribute '(\w+)'", message)
        if match:
            fix = _METHOD_FIXES.get((match.group(1), match.group(2)))
            return make("wrong_method", match.group(2),
                        f"use `.{fix}` instead of `.{match.group(2)}`" if fix else
                        f"`{match.group(1)}` has no `.{match.group(2)}`; check the type and the method name")
        return make("wrong_method")
    if error_type == "TypeError":
        return make("type_mismatch", None, "check the types of the operands and arguments on the reported line")
    if error_type in ("IndexError", "KeyError"):
        empty = "empty" in message or "pop from" in message
        return make("empty_input" if empty else "bad_index", None,
                    "handle the empty case first" if empty else "check the index or key exists before using it")
    if error_type == "ValueError" and ("empty" in message or "arg is an empty sequence" in message):
        return make("empty_input", None, "handle the empty case first")
    if error_type == "ZeroDivisionError":
        return make("division_by_zero", None, "guard the division against a zero divisor")
    if error_type == "Timeout":
        return make("timeout", None, "make sure every loop terminates; use a faster algorithm if the input is large")
    if error_type == "RecursionError":
        return make("recursion", None, "check the base case, or rewrite the recursion as a loop")
    if error_type in ("MemoryError", "ResourceLimit"):
        return make("memory", None, "avoid building huge structures; process the data in a stream")
    if error_type in ("AssertionError", "TestFailed", "WrongAnswer"):
        return make("wrong_result", None, "re-read what the failing test expects and trace the code by hand")
    return make("unknown")


# --------------------------------------------------------------------------- #
# rule fixers: return patched code, or None if the rule does not apply
# --------------------------------------------------------------------------- #
def _fix_missing_import(code: str, d: Diagnosis) -> str | None:
    if d.root_cause != "missing_import" or not d.symbol:
        return None
    lines = code.splitlines()
    at = 0
    while at < len(lines) and (lines[at].startswith(("#", "from __future__")) or not lines[at].strip()):
        at += 1
    return "\n".join(lines[:at] + [f"import {d.symbol}"] + lines[at:]) + ("\n" if code.endswith("\n") else "")


def _fix_name_typo(code: str, d: Diagnosis) -> str | None:
    if d.root_cause != "name_typo" or not d.symbol:
        return None
    match = re.search(r"is probably `(\w+)`", d.suggestion)
    if not match:
        return None
    return re.sub(rf"(?<![\w.]){re.escape(d.symbol)}(?!\w)", match.group(1), code)


def _fix_indentation(code: str, d: Diagnosis) -> str | None:
    if d.root_cause != "indentation" or "\t" not in code:
        return None
    return code.expandtabs(4)


def _fix_wrong_method(code: str, d: Diagnosis) -> str | None:
    if d.root_cause != "wrong_method" or not d.symbol:
        return None
    match = re.search(r"use `\.(\w+)` instead", d.suggestion)
    if not match:
        return None
    return re.sub(rf"\.{re.escape(d.symbol)}(?=\s*\()", "." + match.group(1), code)


RULE_FIXERS: list[tuple[str, Callable[[str, Diagnosis], str | None]]] = [
    ("missing_import", _fix_missing_import), ("name_typo", _fix_name_typo),
    ("indentation", _fix_indentation), ("wrong_method", _fix_wrong_method),
]

PATCH_PROMPT = """This Python code failed. Fix it.

CODE
```python
{code}
```

ERROR
{error_type}: {error}
{traceback}
ROOT CAUSE (automatic analysis)
{cause}{suggestion}
{tests}{history}
Reply with the complete corrected code in one ```python block. Change only what the failure requires.
Allowed imports: {imports}. No file or network access."""


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #
class SelfHealer:
    def __init__(self, sandbox, llm: Callable[[str], str] | None = None, max_retries: int = 3, tracer=None):
        self.sandbox, self.llm, self.max_retries, self.tracer = sandbox, llm, max_retries, tracer
        self.runs = self.repairs = 0

    def _execute(self, code: str, tests: str | None, call: tuple | None):
        if tests:
            return self.sandbox.run_tests(code, tests)
        if call:
            return self.sandbox.call(code, call[0], call[1] if len(call) > 1 else (), call[2] if len(call) > 2 else None)
        return self.sandbox.run(code)

    def _patch(self, code: str, diagnosis: Diagnosis, outcome, tests: str | None,
               tried: list[str]) -> tuple[str | None, str | None]:
        for name, fixer in RULE_FIXERS:
            patched = fixer(code, diagnosis)
            if patched and patched != code and patched not in tried:
                return patched, f"rule:{name}"
        if self.llm is None:
            return None, None
        history = ""
        if len(tried) > 1:
            history = "\nALREADY TRIED (these did not work - do something different)\n" + "\n---\n".join(
                t[-600:] for t in tried[:-1][-2:]) + "\n"
        prompt = PATCH_PROMPT.format(
            code=code, error_type=diagnosis.error_type, error=diagnosis.message,
            traceback=(outcome.traceback or "")[-800:], cause=diagnosis.explanation,
            suggestion=f" - {diagnosis.suggestion}" if diagnosis.suggestion else "",
            tests=f"\nTESTS THE CODE MUST PASS\n```python\n{tests}\n```\n" if tests else "", history=history,
            imports=", ".join(sorted(self.sandbox.policy.allowed_imports)))
        try:
            patched = extract_code(self.llm(prompt))
        except Exception:                                   # the model is down: nothing more we can do
            return None, None
        if not patched or not patched.strip():
            return None, None
        return patched, "model"

    def run(self, code: str, tests: str | None = None, call: tuple | None = None) -> HealResult:
        """Run ``code`` (or its ``tests``, or ``call=(func, args, kwargs)``), repairing it on failure."""
        self.runs += 1
        attempts: list[HealAttempt] = []
        tried: list[str] = []
        stopped = ""
        outcome = None
        for number in range(1, self.max_retries + 2):       # the first run plus max_retries repairs
            tried.append(code)
            outcome = self._execute(code, tests, call)
            if outcome.ok:
                attempts.append(HealAttempt(attempt=number, code=code, ok=True))
                stopped = "passed"
                break
            error_type = outcome.error_type or ("TestFailed" if outcome.tests else "Error")
            message = outcome.error or (outcome.failure or "")
            diagnosis = diagnose(code, error_type, message, outcome.traceback, self.sandbox.policy.allowed_imports)
            attempt = HealAttempt(attempt=number, code=code, ok=False, error_type=error_type,
                                  error=message[:500], diagnosis=diagnosis)
            attempts.append(attempt)
            if self.tracer:
                self.tracer.event("heal", "diagnosis", status="error", attempt=number, error_type=error_type,
                                  root_cause=diagnosis.root_cause, symbol=diagnosis.symbol, line=diagnosis.line)
            if number > self.max_retries:
                stopped = "retries exhausted"
                break
            patched, fixer = self._patch(code, diagnosis, outcome, tests, tried)
            if patched is None:
                stopped = "no patch available"
                break
            if patched in tried:
                stopped = "the patch repeats an earlier attempt"
                break
            attempt.fixer = fixer
            code = patched
        ok = bool(outcome and outcome.ok)
        if ok and len(attempts) > 1:
            self.repairs += 1
        result = HealResult(ok=ok, code=code, value=outcome.value if outcome else None,
                            stdout=outcome.stdout if outcome else "", attempts=attempts, stopped=stopped)
        if self.tracer:
            self.tracer.event("heal", "result", status="ok" if ok else "error", attempts=len(attempts),
                              repaired=result.repaired, fixes=result.fixes, stopped=stopped)
        return result


def healing_python_tool(healer: SelfHealer) -> Tool:
    """A ``python`` tool that repairs failing code before reporting back.

    The observation always says what was changed, so the model is never left
    believing its original code ran.
    """
    allowed = ", ".join(sorted(healer.sandbox.policy.allowed_imports))

    def python(code: str) -> str:
        outcome = healer.run(code)
        parts = []
        if outcome.stdout.strip():
            parts.append(outcome.stdout.rstrip())
        if outcome.ok:
            if outcome.value is not None:
                parts.append(f"result = {outcome.value!r}")
            body = "\n".join(parts) or "(ran without output; print something or assign to `result`)"
            if outcome.repaired:
                body += f"\n[{outcome.summary()}]\n[the code that ran]\n{outcome.code}"
            return body
        last = outcome.attempts[-1]
        note = f" [{outcome.summary()}]" if len(outcome.attempts) > 1 else ""
        hint = f" Hint: {last.diagnosis.suggestion}." if last.diagnosis and last.diagnosis.suggestion else ""
        raise ToolError(f"{last.error_type}: {last.error}{note}{hint}"
                        + ("\n[printed before the error]\n" + parts[0] if parts else ""))

    return Tool("python", "Run Python code in an isolated sandbox and return what it prints, plus the value "
                "of a variable named `result` if you set one. Simple mistakes are repaired automatically and "
                f"reported. No file or network access; imports limited to: {allowed}. State does not persist "
                "between calls.", python, {"code": Param("string", "complete Python source")},
                timeout_s=(healer.sandbox.policy.wall_time_s + 2) * (healer.max_retries + 1) + 5)


def repair_arguments(tool: Tool, args: Mapping) -> tuple[dict, str] | None:
    """Fix argument *names* a model got slightly wrong. Returns ``(new args, what changed)`` or ``None``.

    Only unambiguous repairs are made: a unique close match to a parameter name,
    or a single unknown argument where a single required parameter is missing.
    Values are never invented.
    """
    if not isinstance(args, Mapping):
        return None
    unknown = [k for k in args if k not in tool.params]
    if not unknown:
        return None
    free = [p for p in tool.params if p not in args]
    fixed, notes = dict(args), []
    for key in unknown:
        close = difflib.get_close_matches(key.lower(), [p.lower() for p in free], 1, 0.6)
        target = next((p for p in free if close and p.lower() == close[0]), None)
        if target is None:
            missing = [p for p in free if tool.params[p].required]
            if len(unknown) == 1 and len(missing) == 1:
                target = missing[0]
        if target is None:
            return None
        fixed[target] = fixed.pop(key)
        free.remove(target)
        notes.append(f"'{key}' -> '{target}'")
    return fixed, "renamed argument " + ", ".join(notes)
