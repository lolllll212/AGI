"""Asking the agent to do something."""
from __future__ import annotations

from agi.infrastructure.api.schemas import BatchRequest, RunRequest, SolveRequest, result_json

from .base import Context, Route


def run(ctx: Context):
    request = RunRequest.model_validate(ctx.body)
    result = ctx.wait(ctx.service.ask(request.task, request.mode, ctx.scoped(request.session)))
    ctx.server.history.add(ctx.caller, result, request.session)
    return 200, result_json(result, request.steps)


def batch(ctx: Context):
    request = BatchRequest.model_validate(ctx.body)
    results = ctx.wait(ctx.service.ask_many(request.tasks, request.mode))
    for result in results:
        ctx.server.history.add(ctx.caller, result, None)
    return 200, {"results": [result_json(r) for r in results]}


def solve(ctx: Context):
    request = SolveRequest.model_validate(ctx.body)
    return 200, ctx.wait(ctx.service.solve_parallel(request.task, request.subtasks)).model_dump()


def recent(ctx: Context):
    """The caller's own latest runs (newest first); other callers' runs are not listed."""
    return 200, {"runs": ctx.server.history.of(ctx.caller)}


ROUTES = [
    Route("POST", "/v1/run", run),
    Route("POST", "/v1/batch", batch),
    Route("POST", "/v1/solve", solve),
    Route("GET", "/v1/runs", recent),
]
