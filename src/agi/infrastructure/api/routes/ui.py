"""The web console: static files built from ``web/`` into ``static/`` next to this package.

These routes need no token, because the files hold no secrets: the console asks
for the token itself and sends it with every API call.
"""
from __future__ import annotations

import mimetypes
from pathlib import Path

from agi.infrastructure.api.schemas import HTTPError, Response

from .base import Context, Route

STATIC = Path(__file__).resolve().parent.parent / "static"
_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
          ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".json": "application/json",
          ".map": "application/json", ".woff2": "font/woff2", ".png": "image/png", ".ico": "image/x-icon",
          ".txt": "text/plain; charset=utf-8", ".webmanifest": "application/manifest+json"}


def _headers(server, immutable: bool) -> dict[str, str]:
    policy = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
              "connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'")
    headers = {"Cache-Control": "public, max-age=31536000, immutable" if immutable else "no-cache",
               "Referrer-Policy": "no-referrer"}
    if not server.allow_framing:                            # another site must not be able to frame the console
        policy += "; frame-ancestors 'none'"
        headers["X-Frame-Options"] = "DENY"
    headers["Content-Security-Policy"] = policy
    return headers


def _file(ctx: Context, relative: str, immutable: bool = False) -> Response:
    root = Path(ctx.server.ui_dir or STATIC).resolve()
    target = (root / relative).resolve()
    if root not in target.parents or not target.is_file():  # also refuses "../" and absolute paths
        raise HTTPError(404, "not_found", f"no such file: /{relative}")
    kind = _TYPES.get(target.suffix.lower()) or mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    return Response(200, target.read_bytes(), kind, _headers(ctx.server, immutable))


def index(ctx: Context):
    root = Path(ctx.server.ui_dir or STATIC)
    if not (root / "index.html").is_file():
        raise HTTPError(404, "not_found", "the web console is not built: run 'npm install && npm run build' in web/")
    return _file(ctx, "index.html")


def asset(ctx: Context, path: str):
    return _file(ctx, f"assets/{path}", immutable=True)


def favicon(ctx: Context):
    return _file(ctx, "favicon.svg")


ROUTES = [
    Route("GET", "/", index, public=True),
    Route("GET", "/index.html", index, public=True),
    Route("GET", "/favicon.svg", favicon, public=True),
    Route("GET", "/assets/{path*}", asset, public=True),
]
