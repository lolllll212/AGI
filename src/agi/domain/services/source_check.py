"""The static gate: reject code that breaks a :class:`SandboxPolicy` before anything runs it."""
from __future__ import annotations

import ast

from agi.domain.value_objects.sandbox import FORBIDDEN_ATTRS, FORBIDDEN_NAMES, SAFE_DUNDERS, SandboxPolicy


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
