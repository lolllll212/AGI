"""Semantic tool routing and runtime tool management.

With five tools, a model can read the whole list. With fifty, the list crowds
out the task and the model picks worse. :class:`SemanticToolRouter` ranks tools
against the task and shows the model only the relevant ones (every tool stays
callable). :func:`toolsmith_tools` gives the agent tools *about* tools: list,
describe, search, and - behind a sandbox and tests - create new ones.
"""
from __future__ import annotations

import json
from typing import Callable, Iterable, Sequence

from pydantic import Field

from ..core.types import Model
from ..memory.embed import ConceptEmbedder, cosine, tokens
from .base import Param, Tool, ToolError, ToolRegistry
from .introspect import tool_from_source


class ToolMatch(Model):
    name: str
    score: float
    why: list[str] = Field(default_factory=list)          # words the task and the tool share


# What people ask for when they need each built-in tool, in their words rather than the tool's.
DEFAULT_HINTS: dict[str, str] = {
    "calculator": "arithmetic how much is add subtract multiply divide percent percentage interest exact sum of numbers",
    "python": "write code script simulate loop algorithm statistics complex computation run program",
    "current_time": "today now what day date is it how many days until since deadline weekday this week month year",
    "list_files": "which files exist what is in the folder directory workspace",
    "read_file": "open show the contents of a text file document log",
    "search_files": "grep find lines containing a word pattern occurrences errors in logs mention",
    "write_file": "save create export a report output file",
    "query_data": "csv table spreadsheet rows columns total average count highest lowest per group filter join "
                  "orders sales customers data",
    "http_get": "download url website web page online internet api link",
    "remember": "store save note a fact for later future sessions",
    "recall": "what do we know from earlier previous sessions memory lessons",
}


class SemanticToolRouter:
    """Ranks the tools in a registry by relevance to a task."""

    def __init__(self, registry: ToolRegistry, embed: Callable[[str], dict] | None = None,
                 always: Iterable[str] = (), hints: dict[str, str] | None = None):
        """``embed`` is a function, or an object with an ``embed`` method, that turns text into a sparse
        vector (default: :class:`~agi.memory.embed.ConceptEmbedder`; pass
        :class:`~agi.core.llm.OpenAICompatEmbedder` for real semantic matching).
        ``always`` tools are shown whatever the task; ``hints`` add search keywords to a tool."""
        self.registry = registry
        self._embed = getattr(embed, "embed", embed) or ConceptEmbedder().embed
        self.always = set(always)
        self.hints = dict(DEFAULT_HINTS)
        for name, words in (hints or {}).items():           # your hints add to the built-in ones
            self.hints[name] = f"{self.hints.get(name, '')} {words}".strip()
        self._vectors: dict[str, dict] = {}
        self._words: dict[str, set[str]] = {}
        for tool in registry:
            self.index(tool)
        registry.on_register(self.index)                    # tools added later are indexed too

    def _text(self, tool: Tool) -> str:
        parts = [tool.name.replace("_", " "), tool.description, self.hints.get(tool.name, "")]
        parts += [f"{name.replace('_', ' ')} {p.description}" for name, p in tool.params.items()]
        return " ".join(parts)

    def index(self, tool: Tool) -> None:
        text = self._text(tool)
        self._vectors[tool.name] = self._embed(text)
        self._words[tool.name] = set(tokens(text))

    def rank(self, task: str) -> list[ToolMatch]:
        target, words = self._embed(task), set(tokens(task))
        matches = []
        for tool in self.registry:
            if tool.name not in self._vectors:
                self.index(tool)
            shared = sorted(words & self._words[tool.name])
            matches.append(ToolMatch(name=tool.name, score=cosine(target, self._vectors[tool.name]), why=shared))
        matches.sort(key=lambda m: (-m.score, m.name))
        return matches

    def select(self, task: str, k: int = 5, min_score: float = 0.05) -> list[str]:
        """Names of the tools to show for ``task``: the ``always`` set plus the ``k`` most relevant."""
        ranked = self.rank(task)
        chosen = [m.name for m in ranked if m.score >= min_score][:k]
        if not chosen and ranked:                           # nothing matched: do not hide everything
            chosen = [m.name for m in ranked[:k]]
        return [n for n in self.registry.names if n in self.always or n in chosen]

    def describe(self, task: str, k: int = 5) -> str:
        return self.registry.describe(self.select(task, k))

    def best(self, task: str) -> ToolMatch | None:
        ranked = self.rank(task)
        return ranked[0] if ranked and ranked[0].score > 0 else None


def toolsmith_tools(registry: ToolRegistry, sandbox=None, router: SemanticToolRouter | None = None,
                    allow_create: bool = True, max_created: int = 20) -> list[Tool]:
    """Tools that let an agent inspect and extend its own toolbox at runtime."""
    router = router or SemanticToolRouter(registry)
    created: list[str] = []

    def list_tools() -> str:
        return "\n".join(f"{t.signature()} - {t.description.split('. ')[0]}" for t in registry)

    def describe_tool(name: str) -> str:
        tool = registry.get(name)
        if tool is None:
            raise ToolError(f"UnknownTool: no tool named {name!r}. Use list_tools or find_tools.")
        return json.dumps({"schema": tool.schema()["function"], "risk": tool.risk,
                           "source": getattr(tool, "source", None)}, indent=1, default=str)

    def find_tools(query: str, limit: int = 5) -> str:
        matches = [m for m in router.rank(query) if m.score > 0.02][:limit]
        if not matches:
            return "No tool looks relevant. You can write one with create_tool."
        return "\n".join(f"{m.name} (relevance {m.score:.2f}"
                         + (f"; matches: {', '.join(m.why[:5])}" if m.why else "") + ") - "
                         + registry.get(m.name).description.split(". ")[0] for m in matches)

    tools = [
        Tool("list_tools", "List every tool that is available right now, with its signature.", list_tools, {}),
        Tool("describe_tool", "Show one tool's full schema (parameters, types, defaults) and, for tools "
             "created at runtime, its source code.", describe_tool, {"name": Param("string", "the tool's name")}),
        Tool("find_tools", "Search the toolbox for tools relevant to a need, described in plain words.",
             find_tools, {"query": Param("string", "what you need to do"),
                          "limit": Param("integer", "how many to return", False, 5)}),
    ]
    if not allow_create or sandbox is None:
        return tools

    def create_tool(name: str, code: str, tests: list, description: str = "") -> str:
        if len(created) >= max_created:
            raise ToolError(f"LimitReached: at most {max_created} tools can be created in one session")
        if not name.isidentifier():
            raise ToolError("BadTool: the name must be a valid identifier")
        if name in registry and name not in created:
            raise ToolError(f"BadTool: {name!r} is a built-in tool and cannot be replaced")
        if not tests or not all(isinstance(t, str) and t.strip().startswith("assert") for t in tests):
            raise ToolError("BadTool: give at least one test, each an `assert ...` statement that calls the function")
        tool = tool_from_source(code, sandbox, name=name, description=description or None, tests=tests)
        registry.register(tool, replace=True)
        if name not in created:
            created.append(name)
        return (f"Created tool {tool.signature()}: {tool.description} "
                f"It passed {len(tests)} test(s) and runs in the sandbox. You can call it now.")

    tools.append(Tool(
        "create_tool", "Define a new tool from Python code when no existing tool does the job. The code "
        "must define one public function with type hints and a docstring; it is checked against your "
        "tests in the sandbox before it is registered, and always runs sandboxed (no files, no network).",
        create_tool, {"name": Param("string", "identifier for the new tool"),
                      "code": Param("string", "Python source defining the function"),
                      "tests": Param("array", "assert statements that call the function", items="string"),
                      "description": Param("string", "what the tool does (default: the docstring)", False, "")},
        risk="write"))
    return tools
