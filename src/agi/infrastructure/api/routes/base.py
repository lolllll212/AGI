"""How a request finds its handler."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:                                           # pragma: no cover
    from agi.application.interfaces.service import AgentService
    from agi.infrastructure.api.server import AgentServer


@dataclass
class Context:
    """What a handler may use: the service, who is calling, and the parsed JSON body (for POST)."""

    server: "AgentServer"
    caller: str = "default"
    body: Any = None

    @property
    def service(self) -> "AgentService":
        return self.server.service

    def scoped(self, session: str | None) -> str | None:
        """One caller cannot name another caller's session."""
        return f"{self.caller}/{session}" if session and self.server.named else session

    def wait(self, coroutine) -> Any:
        """Run a coroutine on the server's event loop and block this request thread until it finishes."""
        return self.server.wait(coroutine)


@dataclass(frozen=True)
class Route:
    """``pattern`` is a path whose ``{name}`` segments match any one segment; ``{name*}`` matches the rest."""

    method: str
    pattern: str
    handler: Callable[..., Any]
    public: bool = False                                    # reachable without a token

    def match(self, parts: list[str]) -> list[str] | None:
        wanted = [p for p in self.pattern.strip("/").split("/") if p]
        captured: list[str] = []
        for index, segment in enumerate(wanted):
            if segment.startswith("{") and segment.endswith("*}"):
                if index >= len(parts):
                    return None
                return captured + ["/".join(parts[index:])]
            if index >= len(parts):
                return None
            if segment.startswith("{") and segment.endswith("}"):
                captured.append(parts[index])
            elif segment != parts[index]:
                return None
        return captured if len(parts) == len(wanted) else None
