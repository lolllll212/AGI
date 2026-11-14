"""The endpoints, one module per resource.

    GET    /health                 liveness (no token)
    GET    /v1/info                version, model label, example tasks
    GET    /v1/tools               the tools the agent has
    GET    /v1/stats               counters, memory sizes, drift status
    GET    /metrics                the same and more, for Prometheus
    POST   /v1/run                 {"task": "...", "mode"?, "session"?, "steps"?}
    POST   /v1/batch               {"tasks": ["...", ...]}
    POST   /v1/solve               {"task": "...", "subtasks"?: [...]}   split, run in parallel, combine
    GET    /v1/runs                the caller's latest runs
    GET    /v1/traces/{trace_id}   every recorded step of one run
    DELETE /v1/sessions/{id}       forget a conversation
    GET    /  /assets/...          the web console (no token; switch off with ui=False)
"""
from . import runs, sessions, system, traces, ui
from .base import Context, Route

API_ROUTES: list[Route] = system.ROUTES + runs.ROUTES + sessions.ROUTES + traces.ROUTES
UI_ROUTES: list[Route] = ui.ROUTES

__all__ = ["API_ROUTES", "UI_ROUTES", "Context", "Route"]
