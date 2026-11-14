"""Liveness, what the agent can do, how it is doing."""
from __future__ import annotations

from agi.infrastructure.adapters.observability.metrics import CONTENT_TYPE
from agi.infrastructure.api.schemas import HTTPError, Response

from .base import Context, Route


def health(ctx: Context):
    return 200, {**ctx.service.health(), "auth_required": ctx.server.authenticated}


def info(ctx: Context):
    from agi import __version__
    service = ctx.service
    return 200, {"version": __version__, "model": getattr(service, "label", None),
                 "demo": bool(getattr(service, "demo", False)),
                 "examples": list(getattr(service, "examples", []) or []),
                 "caller": ctx.caller, "sandbox": type(getattr(service.engine, "sandbox", None)).__name__,
                 "metrics": ctx.server.metrics is not None}


def tools(ctx: Context):
    return 200, {"tools": ctx.service.tools()}


def stats(ctx: Context):
    return 200, {**ctx.service.stats(), "http": ctx.server.http_stats()}


def metrics(ctx: Context):
    if ctx.server.metrics is None:
        raise HTTPError(404, "not_found", "metrics are switched off on this server")
    return Response(200, ctx.server.metrics.render().encode(), CONTENT_TYPE)


ROUTES = [
    Route("GET", "/health", health, public=True),
    Route("GET", "/v1/info", info),
    Route("GET", "/v1/tools", tools),
    Route("GET", "/v1/stats", stats),
    Route("GET", "/metrics", metrics),
]
