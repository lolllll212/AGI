"""Every recorded step of one run."""
from __future__ import annotations

from agi.infrastructure.api.schemas import HTTPError

from .base import Context, Route


def trace(ctx: Context, trace_id: str):
    events = ctx.service.trace(trace_id)
    if not events:
        raise HTTPError(404, "not_found", f"no trace with id {trace_id!r}")
    return 200, {"trace_id": trace_id, "events": events}


ROUTES = [Route("GET", "/v1/traces/{trace_id}", trace)]
