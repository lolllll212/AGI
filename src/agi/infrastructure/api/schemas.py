"""The shapes that cross the HTTP boundary."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import Field

from agi.domain.value_objects.records import EngineResult, Model

MAX_BODY = 1_000_000
DRAIN_LIMIT = 8 * MAX_BODY


class RunRequest(Model):
    task: str = Field(min_length=1, max_length=20_000)
    mode: Literal["reflex", "react", "deliberate"] | None = None
    session: str | None = Field(default=None, min_length=1, max_length=128)
    steps: bool = False                  # include the full step list in the response


class BatchRequest(Model):
    tasks: list[str] = Field(min_length=1, max_length=32)
    mode: Literal["reflex", "react", "deliberate"] | None = None


class SolveRequest(Model):
    task: str = Field(min_length=1, max_length=20_000)
    subtasks: list[str] | None = Field(default=None, max_length=16)


class HTTPError(Exception):
    def __init__(self, status: int, kind: str, message: str):
        super().__init__(message)
        self.status, self.kind, self.message = status, kind, message


@dataclass
class Response:
    """A reply that is not JSON: a metrics page, a file of the web UI."""

    status: int
    body: bytes
    content_type: str
    headers: dict[str, str] = field(default_factory=dict)


def result_json(result: EngineResult, steps: bool = False) -> dict:
    data = result.model_dump(exclude=None if steps else {"steps", "route"})
    if not steps:
        data["steps"] = len(result.steps)
        data["route_reason"] = result.route.reason if result.route else None
    return data
