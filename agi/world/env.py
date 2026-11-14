"""Environments: places where an action has a consequence the agent can observe.

The common shape is deliberately tiny (``reset`` / ``step``) so that a physics
engine, a code sandbox and a real API all look the same to an agent, and so
every interaction is logged as a :class:`Transition` it can learn from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..core.sandbox import Sandbox


@dataclass
class Transition:
    action: Any
    observation: Any
    reward: float = 0.0
    done: bool = False
    info: dict = field(default_factory=dict)


class Environment:
    """Base class. Implement :meth:`act`; :meth:`step` adds the experience log."""

    def __init__(self):
        self.history: list[Transition] = []

    def reset(self) -> Any:
        self.history.clear()
        return None

    def act(self, action: Any) -> Transition:
        raise NotImplementedError

    def step(self, action: Any) -> Transition:
        transition = self.act(action)
        self.history.append(transition)
        return transition


class CodeWorld(Environment):
    """The sandbox as an environment: act = run a program, observe = what it did."""

    def __init__(self, sandbox: Sandbox | None = None):
        super().__init__()
        self.sandbox = sandbox or Sandbox()

    def act(self, action: str) -> Transition:
        outcome = self.sandbox.run(action)
        observation = {"ok": outcome.ok, "value": outcome.value, "stdout": outcome.stdout,
                       "error": outcome.failure}
        return Transition(action, observation, reward=1.0 if outcome.ok else 0.0)


class FunctionWorld(Environment):
    """Bridge to real tools and APIs.

    Wrap any callables (an HTTP client method, a database query, a robot
    command). The agent can only invoke names on the allowlist, each call is
    budgeted, and failures come back as observations rather than exceptions -
    a failed call is information, not a crash.

        world = FunctionWorld({"get_weather": client.weather}, budget=20)
        world.step({"tool": "get_weather", "args": {"city": "Patna"}}).observation
    """

    def __init__(self, tools: Mapping[str, Callable], budget: int | None = None):
        super().__init__()
        self.tools = dict(tools)
        self.budget = budget

    def act(self, action: Mapping) -> Transition:
        name, args = action.get("tool"), action.get("args", {})
        if name not in self.tools:
            observation = {"ok": False, "error": f"UnknownTool: '{name}' is not on the allowlist"}
        elif self.budget is not None and len(self.history) >= self.budget:
            observation = {"ok": False, "error": "BudgetExceeded: call budget exhausted"}
        else:
            try:
                observation = {"ok": True, "value": self.tools[name](**args)}
            except Exception as exc:
                observation = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return Transition(dict(action), observation, reward=1.0 if observation["ok"] else 0.0)
