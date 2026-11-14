"""Arithmetic without ``eval``: a small, safe expression evaluator."""
from __future__ import annotations

import ast
import math
import operator
from fractions import Fraction
from typing import Any, Callable

from agi.domain.entities.tool import ToolError

_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
_COMPARE = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
            ast.Eq: operator.eq, ast.NotEq: operator.ne}
_FUNCTIONS: dict[str, Callable] = {
    name: getattr(math, name) for name in (
        "sqrt", "exp", "log", "log10", "log2", "sin", "cos", "tan", "asin", "acos", "atan",
        "atan2", "sinh", "cosh", "tanh", "floor", "ceil", "factorial", "comb", "perm", "gcd",
        "hypot", "degrees", "radians", "isqrt", "fsum")}
_FUNCTIONS.update(abs=abs, round=round, min=min, max=max, sum=lambda *v: math.fsum(
    v[0] if len(v) == 1 and isinstance(v[0], (list, tuple)) else v))
_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf}


def evaluate(expression: str) -> int | float | bool:
    """Evaluate an arithmetic expression without ``eval``.

    Integers and ratios of integers are computed exactly, so ``0.1 + 0.2`` style
    surprises only appear when a float is genuinely involved.
    """
    if len(expression) > 1000:
        raise ToolError("expression is too long")
    if "^" in expression:
        raise ToolError("'^' is not a power operator here; write ** instead")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"cannot parse {expression!r}: {exc.msg}. Use plain arithmetic such as "
                        f"(1200 * 1.07) / 12; write numbers without thousands separators") from None

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ToolError(f"only numbers are allowed, not {node.value!r}")
            return Fraction(node.value) if isinstance(node.value, int) else node.value
        if isinstance(node, ast.Name):
            if node.id not in _CONSTANTS:
                raise ToolError(f"unknown name {node.id!r}; available constants: {', '.join(_CONSTANTS)}")
            return _CONSTANTS[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = walk(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
                raise ToolError("division by zero")
            return _BINARY[type(node.op)](left, right)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            base, power = walk(node.left), walk(node.right)
            if isinstance(base, Fraction) and isinstance(power, Fraction) and power.denominator == 1:
                bits = max(base.numerator.bit_length(), base.denominator.bit_length())
                if abs(power) * max(bits, 1) > 20_000:
                    raise ToolError("that power is too large to compute exactly")
                if base == 0 and power < 0:
                    raise ToolError("division by zero")
                return base ** int(power)
            try:
                result = float(base) ** float(power)
            except (OverflowError, ZeroDivisionError) as exc:
                raise ToolError(f"cannot compute that power: {exc}") from None
            if isinstance(result, complex):
                raise ToolError("the result is not a real number")
            return result
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _COMPARE:
            return _COMPARE[type(node.ops[0])](walk(node.left), walk(node.comparators[0]))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            fn = _FUNCTIONS.get(node.func.id)
            if fn is None:
                raise ToolError(f"unknown function {node.func.id!r}; available: "
                                f"{', '.join(sorted(_FUNCTIONS))}")
            args = [walk(a) for a in node.args]
            args = [int(a) if isinstance(a, Fraction) and a.denominator == 1
                    else float(a) if isinstance(a, Fraction) else a for a in args]
            try:
                value = fn(*args)
            except (ValueError, TypeError, OverflowError) as exc:
                raise ToolError(f"{node.func.id}: {exc}") from None
            return Fraction(value) if isinstance(value, int) and not isinstance(value, bool) else value
        if isinstance(node, (ast.List, ast.Tuple)):
            return [walk(e) for e in node.elts]
        raise ToolError(f"unsupported syntax: {type(node).__name__}")

    value = walk(tree)
    if isinstance(value, list):                    # "1,200 + 5" parses as the pair (1, 205)
        raise ToolError("that is a list of values, not one expression; write numbers without "
                        "thousands separators (1200, not 1,200)")
    if isinstance(value, Fraction):
        return value.numerator if value.denominator == 1 else float(value)
    return value


def format_number(value: Any) -> str:
    if isinstance(value, float):
        if value != value or value in (math.inf, -math.inf):
            return str(value)
        return f"{value:.12g}"
    return str(value)
