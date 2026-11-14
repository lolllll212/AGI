"""Turning code into tools: docstring analysis and schema generation.

:func:`tool_from_function` reads a live function's signature, type hints and
docstring (Google, NumPy or reST style). :func:`tool_from_source` does the same
for source code the agent wrote *without importing it*: the schema comes from
the AST, and every call runs in a sandbox.
"""
from __future__ import annotations

import ast
import inspect
import re
import types
import typing
from typing import Any, Callable, Sequence

from .base import Param, Tool, ToolError

_JSON = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object",
         tuple: "array", set: "array"}
_NAMES = {"str": "string", "int": "integer", "float": "number", "bool": "boolean", "list": "array",
          "dict": "object", "tuple": "array", "set": "array", "List": "array", "Dict": "object",
          "Sequence": "array", "Iterable": "array", "Mapping": "object", "Any": "string"}

_SECTION = re.compile(r"^\s*(Args|Arguments|Parameters|Params|Returns?|Yields?|Raises|Examples?|Notes?)\s*:?\s*$", re.I)
_GOOGLE = re.compile(r"^\s*\*{0,2}(\w+)\s*(?:\(([^)]*)\))?\s*:\s*(.*)$")
_NUMPY = re.compile(r"^(\w+)\s*:\s*([^:]+)$")
_REST = re.compile(r"^\s*:param\s+(?:[\w\[\], .|]+\s+)?(\w+)\s*:\s*(.*)$")
_REST_RETURN = re.compile(r"^\s*:returns?\s*:\s*(.*)$")


class Doc(typing.NamedTuple):
    summary: str
    params: dict
    returns: str


def parse_docstring(doc: str | None) -> Doc:
    """Summary, per-parameter descriptions and the return description of a docstring."""
    doc = inspect.cleandoc(doc or "")
    if not doc:
        return Doc("", {}, "")
    lines = doc.splitlines()
    summary_lines = []
    for line in lines:
        if not line.strip() or _SECTION.match(line) or _REST.match(line) or _REST_RETURN.match(line):
            break
        summary_lines.append(line.strip())
    params: dict[str, str] = {}
    returns = ""
    section, current = "", None
    index = 0
    while index < len(lines):
        line = lines[index]
        header = _SECTION.match(line)
        if header:
            section, current = header.group(1).lower(), None
            if index + 1 < len(lines) and set(lines[index + 1].strip()) <= {"-", "="} and lines[index + 1].strip():
                index += 1                                  # NumPy underline
            index += 1
            continue
        rest, rest_return = _REST.match(line), _REST_RETURN.match(line)
        if rest:
            current = rest.group(1)
            params[current] = rest.group(2).strip()
        elif rest_return:
            returns, current = rest_return.group(1).strip(), None
        elif section in ("args", "arguments", "parameters", "params"):
            google = _GOOGLE.match(line)
            numpy = _NUMPY.match(line) if not line.startswith((" ", "\t")) else None
            if numpy and (not google or not google.group(3) or numpy.group(2).strip() in _NAMES
                          or re.match(r"^[\w\[\], .|]+$", numpy.group(2)) and index + 1 < len(lines)
                          and lines[index + 1].startswith((" ", "\t"))):
                current = numpy.group(1)
                params[current] = ""
            elif google and (current is None or len(line) - len(line.lstrip()) <= _indent(lines, params, current)):
                current = google.group(1)
                params[current] = google.group(3).strip()
            elif current and line.strip():
                params[current] = (params[current] + " " + line.strip()).strip()
        elif section in ("return", "returns") and line.strip():
            bare_type = (not line.startswith((" ", "\t")) and index + 1 < len(lines)
                         and lines[index + 1].startswith((" ", "\t")) and lines[index + 1].strip())
            if not bare_type:                               # NumPy style puts the type on its own line
                returns = (returns + " " + line.strip()).strip()
        elif not section:
            head, sep, tail = line.strip().partition(":")   # the simple "name: text" form
            if sep and head.isidentifier() and tail.strip():
                params[head] = tail.strip()
        index += 1
    return Doc(" ".join(summary_lines), params, returns)


def _indent(lines: list[str], params: dict, current: str) -> int:
    for line in lines:
        if re.match(rf"^\s*\*{{0,2}}{re.escape(current)}\b", line):
            return len(line) - len(line.lstrip())
    return 0


def _schema_of(annotation: Any) -> dict:
    """``{"type": ..., "enum": ..., "items": ..., "optional": bool}`` for a type hint."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "string"}
    if isinstance(annotation, str):
        optional = "None" in annotation or annotation.startswith("Optional")
        head = re.sub(r"^Optional\[|\s*\|\s*None|None\s*\|\s*", "", annotation).split("[")[0].strip()
        return {"type": _NAMES.get(head.split(".")[-1], "string"), "optional": optional}
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin is typing.Literal:
        kinds = {type(a) for a in args}
        return {"type": _JSON.get(kinds.pop(), "string") if len(kinds) == 1 else "string", "enum": list(args)}
    if origin in (typing.Union, types.UnionType):
        rest = [a for a in args if a is not type(None)]
        schema = _schema_of(rest[0]) if len(rest) == 1 else {"type": "string"}
        return {**schema, "optional": len(rest) < len(args)}
    if origin is not None:
        schema = {"type": _JSON.get(origin, "array" if isinstance(origin, type) and issubclass(
            origin, typing.Iterable) and origin is not dict else "object" if origin is dict else "string")}
        if schema["type"] == "array" and args and args[0] is not Ellipsis:
            schema["items"] = _schema_of(args[0])["type"]
        return schema
    if isinstance(annotation, type) and issubclass(annotation, __import__("enum").Enum):
        return {"type": "string", "enum": [member.value for member in annotation]}
    return {"type": _JSON.get(annotation, "string")}


def tool_from_function(fn: Callable, name: str | None = None, description: str | None = None,
                       risk: str = "safe", timeout_s: float = 30.0) -> Tool:
    """Build a :class:`Tool` from a function's signature, type hints and docstring."""
    doc = parse_docstring(inspect.getdoc(fn))
    try:
        hints = typing.get_type_hints(fn)
    except Exception:                                       # unresolved forward references
        hints = {}
    params: dict[str, Param] = {}
    for pname, p in inspect.signature(fn).parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            raise TypeError(f"tool functions cannot take *args/**kwargs ({getattr(fn, '__name__', fn)})")
        schema = _schema_of(hints.get(pname, p.annotation))
        required = p.default is p.empty
        params[pname] = Param(schema["type"], doc.params.get(pname, ""), required, None if required else p.default,
                              enum=schema.get("enum"), items=schema.get("items"))
    summary = description or doc.summary or getattr(fn, "__name__", "tool")
    if doc.returns and not description:
        summary += f" Returns: {doc.returns}"
    return Tool(name or fn.__name__, summary, fn, params, risk, timeout_s)


def _literal(node: ast.AST | None) -> Any:
    try:
        return ast.literal_eval(node) if node is not None else None
    except Exception:
        return None


def signature_from_source(code: str, func: str | None = None) -> tuple[str, str, dict[str, Param]]:
    """``(function name, description, params)`` read from source code without executing it."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ToolError(f"SyntaxError: line {exc.lineno}: {exc.msg}") from None
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")]
    if func:
        functions = [n for n in functions if n.name == func]
    if not functions:
        raise ToolError(f"BadTool: the code defines no public function" + (f" named {func!r}" if func else ""))
    node = functions[-1]
    if node.args.vararg or node.args.kwarg or node.args.posonlyargs:
        raise ToolError("BadTool: tool functions cannot take *args, **kwargs or positional-only parameters")
    doc = parse_docstring(ast.get_docstring(node))
    arguments = node.args.args + node.args.kwonlyargs
    defaults: list[ast.AST | None] = ([None] * (len(node.args.args) - len(node.args.defaults))
                                      + list(node.args.defaults) + list(node.args.kw_defaults))
    params = {}
    for argument, default in zip(arguments, defaults):
        schema = _schema_of(ast.unparse(argument.annotation)) if argument.annotation else {"type": "string"}
        params[argument.arg] = Param(schema["type"], doc.params.get(argument.arg, ""), default is None,
                                     _literal(default))
    description = doc.summary + (f" Returns: {doc.returns}" if doc.returns else "")
    return node.name, description, params


def tool_from_source(code: str, sandbox, name: str | None = None, func: str | None = None,
                     description: str | None = None, tests: Sequence[str] = ()) -> Tool:
    """A tool backed by source code that only ever runs inside ``sandbox``.

    ``tests`` are assert statements; the tool is refused unless all of them pass.
    """
    func_name, doc, params = signature_from_source(code, func)
    summary = description or doc
    if not summary:
        raise ToolError("BadTool: the function needs a docstring (or a description) saying what it does")
    probe = sandbox.run(code + ("\n\n" + "\n".join(tests) + "\n" if tests else "\n"))
    if not probe.ok:
        raise ToolError(f"BadTool: the code failed verification - {probe.failure}")

    def call(**kwargs: Any) -> Any:
        outcome = sandbox.call(code, func_name, kwargs=kwargs)
        if not outcome.ok:
            raise ToolError(outcome.failure or "the tool failed")
        return outcome.value

    tool = Tool(name or func_name, summary, call, params, "safe", sandbox.policy.wall_time_s + 5)
    tool.source = code                                      # type: ignore[attr-defined]
    return tool
