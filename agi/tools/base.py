"""Tools: the typed, logged, failure-tolerant interface between an agent and the world.

A :class:`Tool` is a described Python callable. A :class:`ToolRegistry` is the
only way an agent invokes one: arguments are validated against the tool's
schema, the call is timed and time-limited, output is capped, and *nothing a
tool does can raise into the agent loop* - a failure comes back as an
observation the agent can read and react to.
"""
from __future__ import annotations

import contextvars
import inspect
import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean",
               list: "array", dict: "object"}
_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


_KIND = re.compile(r"^([A-Z][A-Za-z]+): (.*)$", re.S)


class ToolError(Exception):
    """Raise inside a tool to report a clean, user-facing failure."""


@dataclass
class Param:
    type: str = "string"
    description: str = ""
    required: bool = True
    default: Any = None
    enum: list | None = None            # the only values allowed, if restricted
    items: str | None = None            # element type of an array, if known


def _param_schema(self) -> dict:
    schema: dict = {"type": self.type, "description": self.description}
    if self.enum:
        schema["enum"] = list(self.enum)
    if self.items:
        schema["items"] = {"type": self.items}
    if not self.required and self.default is not None:
        schema["default"] = self.default
    return schema


Param.schema = _param_schema            # type: ignore[attr-defined]


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]
    params: dict[str, Param] = field(default_factory=dict)
    risk: str = "safe"                  # "safe" | "write" | "network": what the tool can touch
    timeout_s: float = 30.0

    def signature(self) -> str:
        parts = [f"{name}: {p.type}" + ("" if p.required else f" = {json.dumps(p.default, default=str)}")
                 for name, p in self.params.items()]
        return f"{self.name}({', '.join(parts)})"

    def schema(self) -> dict:
        """OpenAI-style function schema, for models with native tool calling."""
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": {"type": "object",
                           "properties": {n: p.schema() for n, p in self.params.items()},
                           "required": [n for n, p in self.params.items() if p.required]}}}


@dataclass
class ToolResult:
    tool: str
    args: dict
    ok: bool
    output: str                         # what the agent sees
    value: Any = None                   # the raw return value, for programmatic use
    error_type: str | None = None
    seconds: float = 0.0
    truncated: bool = False

    @property
    def error(self) -> str | None:
        return None if self.ok else self.output


def _render(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, indent=None, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(value)


def from_function(fn: Callable, name: str | None = None, description: str | None = None,
                  risk: str = "safe", timeout_s: float = 30.0) -> Tool:
    """Build a :class:`Tool` from a plain function's signature, type hints and docstring.

    Google, NumPy and reST docstrings are understood, as are plain ``name: text`` lines.
    """
    from .introspect import tool_from_function
    return tool_from_function(fn, name, description, risk, timeout_s)


class ToolRegistry:
    """A set of tools plus the rules for calling them."""

    def __init__(self, tools: Iterable[Tool | Callable] = (), max_output_chars: int = 4000,
                 max_calls: int | None = None, allow: Iterable[str] = ("safe", "write", "network")):
        self._tools: dict[str, Tool] = {}
        self.max_output_chars = max_output_chars
        self.max_calls = max_calls
        self.allow = frozenset(allow)
        self.log: list[ToolResult] = []
        for tool in tools:
            self.register(tool)

    # -- registration ------------------------------------------------------ #
    def register(self, tool: Tool | Callable, replace: bool = False, **kwargs) -> Tool:
        if not isinstance(tool, Tool):
            tool = from_function(tool, **kwargs)
        if not tool.name.isidentifier():
            raise ValueError(f"tool name {tool.name!r} must be an identifier")
        if tool.name in self._tools and not replace:
            raise ValueError(f"a tool named {tool.name!r} is already registered")
        self._tools[tool.name] = tool
        for listener in getattr(self, "_listeners", ()):
            listener(tool)
        return tool

    def on_register(self, listener: Callable[[Tool], None]) -> None:
        """Call ``listener(tool)`` whenever a tool is added (routers use this to stay current)."""
        if not hasattr(self, "_listeners"):
            self._listeners = []
        self._listeners.append(listener)

    def tool(self, fn: Callable | None = None, **kwargs):
        """Decorator form of :meth:`register`."""
        def wrap(f):
            self.register(f, **kwargs)
            return f
        return wrap(fn) if fn is not None else wrap

    def extend(self, tools: Iterable[Tool | Callable]) -> "ToolRegistry":
        for tool in tools:
            self.register(tool)
        return self

    # -- inspection -------------------------------------------------------- #
    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def describe(self, names: Iterable[str] | None = None) -> str:
        """The tool list as it is shown to a model (optionally only the tools in ``names``)."""
        wanted = None if names is None else set(names)
        lines = []
        for tool in self._tools.values():
            if wanted is not None and tool.name not in wanted:
                continue
            lines.append(f"- {tool.signature()}: {tool.description}")
            for name, p in tool.params.items():
                note = p.description + (f" (one of: {', '.join(map(str, p.enum))})" if p.enum else "")
                if note:
                    lines.append(f"    {name}: {note}")
        return "\n".join(lines)

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def schemas(self) -> list[dict]:
        return [tool.schema() for tool in self._tools.values()]

    # -- calling ----------------------------------------------------------- #
    def _fail(self, name: str, args: Any, kind: str, message: str, seconds: float = 0.0) -> ToolResult:
        result = ToolResult(name, args if isinstance(args, dict) else {}, False,
                            f"{kind}: {message}", None, kind, seconds)
        self.log.append(result)
        return result

    def _bind(self, tool: Tool, args: Mapping) -> dict | str:
        """Validated keyword arguments, or a message saying what is wrong."""
        unknown = [k for k in args if k not in tool.params]
        if unknown:
            return (f"unexpected argument(s) {', '.join(map(repr, unknown))}; "
                    f"the signature is {tool.signature()}")
        bound = {}
        for name, param in tool.params.items():
            if name not in args or args[name] is None:
                if param.required:
                    return f"missing required argument '{name}'; the signature is {tool.signature()}"
                continue
            value = args[name]
            if param.type in ("integer", "number") and isinstance(value, str):
                try:                                     # models often quote numbers
                    value = int(value) if param.type == "integer" else float(value)
                except ValueError:
                    pass
            if param.type == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
                value = str(value)
            if not _CHECKS[param.type](value):
                return (f"argument '{name}' must be of type {param.type}, "
                        f"got {type(value).__name__}")
            if param.enum and value not in param.enum:
                return f"argument '{name}' must be one of {', '.join(map(repr, param.enum))}; got {value!r}"
            bound[name] = value
        return bound

    def call(self, name: str, args: Mapping | None = None) -> ToolResult:
        """Invoke a tool. Never raises: every failure is returned as a result."""
        args = {} if args is None else args
        tool = self._tools.get(name)
        if tool is None:
            return self._fail(name, args, "UnknownTool",
                              f"no tool named {name!r}. Available tools: {', '.join(self._tools) or '(none)'}")
        if not isinstance(args, Mapping):
            return self._fail(name, args, "BadArguments",
                              f"arguments must be a JSON object; the signature is {tool.signature()}")
        if tool.risk not in self.allow:
            return self._fail(name, args, "NotPermitted",
                              f"tools with '{tool.risk}' access are disabled in this session")
        if self.max_calls is not None and len(self.log) >= self.max_calls:
            return self._fail(name, args, "BudgetExceeded",
                              f"the limit of {self.max_calls} tool calls has been reached")
        bound = self._bind(tool, args)
        if isinstance(bound, str):
            return self._fail(name, args, "BadArguments", bound)

        box: dict = {}

        def work():
            try:
                box["value"] = tool.fn(**bound)
            except BaseException as exc:                 # noqa: BLE001 - tools must not crash the loop
                box["error"] = exc

        started = time.perf_counter()
        # Run in a copy of the caller's context so tracing spans and depth counters carry over.
        thread = threading.Thread(target=contextvars.copy_context().run, args=(work,), daemon=True,
                                  name=f"tool-{name}")
        thread.start()
        thread.join(tool.timeout_s)
        seconds = time.perf_counter() - started
        if thread.is_alive():
            return self._fail(name, dict(args), "Timeout",
                              f"{name} did not finish within {tool.timeout_s:g}s", seconds)
        if "error" in box:
            exc = box["error"]
            kind, message = type(exc).__name__, str(exc)
            if isinstance(exc, ToolError):               # "Kind: message" names its own error type
                tagged = _KIND.match(message)
                kind, message = (tagged.group(1), tagged.group(2)) if tagged else ("ToolError", message)
            return self._fail(name, dict(args), kind, message or kind, seconds)

        text = _render(box.get("value"))
        truncated = len(text) > self.max_output_chars
        if truncated:
            text = (text[: self.max_output_chars]
                    + f"\n[output truncated: {len(text) - self.max_output_chars} more characters]")
        result = ToolResult(name, dict(args), True, text, box.get("value"), None, seconds, truncated)
        self.log.append(result)
        return result
