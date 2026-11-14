"""Conversations."""
from __future__ import annotations

from agi.infrastructure.api.schemas import HTTPError

from .base import Context, Route


def end(ctx: Context, session: str):
    if not ctx.service.end_session(ctx.scoped(session)):
        raise HTTPError(404, "not_found", f"no session {session!r}")
    return 200, {"ended": session}


ROUTES = [Route("DELETE", "/v1/sessions/{id}", end)]
