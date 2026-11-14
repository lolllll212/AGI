"""A small JSON-over-HTTP front end for :class:`AgentService` (standard library only).

The endpoints are listed in :mod:`agi.infrastructure.api.routes`.

It listens on 127.0.0.1 unless told otherwise, and refuses to listen on any
other address without a token: an agent that can run code and write files must
not be reachable anonymously.

For more than one caller, give each a key (``keys={"alice": "...", "ci": "..."}``):
every key has its own rate limit and its own sessions, so one caller can neither
starve nor read another. ``certfile``/``keyfile`` turn on TLS (1.2 or later).
Connections beyond ``max_connections`` are refused with 503 at once rather than
queued, and ``stop()`` lets requests in progress finish.

``/metrics`` serves Prometheus metrics (behind the same token), ``access_log``
writes one JSON line per request, and ``/`` serves the web console if it has
been built.
"""
from __future__ import annotations

import asyncio
import json
import re
import secrets
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote, urlsplit

from pydantic import ValidationError

from agi.application.cortex.resilience import RateLimiter
from agi.application.interfaces.service import AgentService
from agi.infrastructure.adapters.observability.logs import JsonLog
from agi.infrastructure.adapters.security.auth import MAX_AUTH_FAILURES, MIN_KEY_LENGTH, ApiKeys
from agi.infrastructure.adapters.security.tls import server_context
from agi.infrastructure.api.routes import API_ROUTES, UI_ROUTES, Context, Route
from agi.infrastructure.api.schemas import (DRAIN_LIMIT, MAX_BODY, BatchRequest, HTTPError, Response, RunRequest,
                                            SolveRequest, result_json)
from agi.infrastructure.api.telemetry import AgentMetrics, RunHistory

LOOPBACK = ("127.0.0.1", "::1", "localhost")
_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")
_BUSY_BODY = b'{"error": {"type": "overloaded", "message": "too many connections; retry shortly"}}'
_BUSY = (b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nRetry-After: 1\r\n"
         b"Connection: close\r\nContent-Length: " + str(len(_BUSY_BODY)).encode() + b"\r\n\r\n" + _BUSY_BODY)

__all__ = ["AgentServer", "BatchRequest", "DRAIN_LIMIT", "HTTPError", "LOOPBACK", "MAX_AUTH_FAILURES", "MAX_BODY",
           "MIN_KEY_LENGTH", "RunRequest", "SolveRequest", "result_json", "serve"]


class _Server(ThreadingHTTPServer):
    """Refuses connections beyond a cap instead of starting a thread for each."""

    daemon_threads = True
    request_queue_size = 128

    def __init__(self, address, handler, max_connections: int, tls: ssl.SSLContext | None):
        super().__init__(address, handler)
        self._slots = threading.BoundedSemaphore(max_connections)
        self._tls = tls
        self.refused = 0
        self.active = 0
        self._count = threading.Lock()

    def get_request(self):
        connection, address = super().get_request()
        if self._tls is not None:
            # Handshake later, in the request's own thread: a client that stalls its handshake
            # must not stop the server accepting everyone else.
            connection = self._tls.wrap_socket(connection, server_side=True, do_handshake_on_connect=False)
        return connection, address

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            self.refused += 1
            try:
                request.settimeout(1)
                request.sendall(_BUSY)
            except OSError:
                pass
            self.shutdown_request(request)
            return
        with self._count:
            self.active += 1
        super().process_request(request, client_address)

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self._count:
                self.active -= 1
            self._slots.release()

    def handle_error(self, request, client_address):       # a dropped or failed connection is not news
        pass


class AgentServer:
    """Serve an :class:`AgentService` over HTTP(S). ``port=0`` picks a free port."""

    def __init__(self, service: AgentService, host: str = "127.0.0.1", port: int = 8000,
                 token: str | None = None, rate_limit: float | None = None, burst: int = 10,
                 keys: dict[str, str] | None = None, certfile: str | None = None, keyfile: str | None = None,
                 max_connections: int = 64, metrics: bool = True, ui: bool = True, ui_dir: str | None = None,
                 allow_framing: bool = False, access_log: JsonLog | None = None):
        """``token`` is one shared secret; ``keys`` maps a caller's name to that caller's own secret.
        ``rate_limit`` (requests per second, with ``burst``) applies per caller.

        ``metrics=False`` removes ``/metrics``; ``ui=False`` removes the web console. ``allow_framing``
        lets another site show the console in a frame (off by default, to prevent clickjacking).
        ``access_log`` receives one event per request."""
        self.auth = ApiKeys(token, keys)
        if host not in LOOPBACK and not self.auth.enabled:
            raise ValueError(f"refusing to listen on {host} without a token: pass token=... or keys=... "
                             "(or bind to 127.0.0.1 for local use only)")
        tls = server_context(certfile, keyfile)
        self.service, self.token = service, token
        self.named = self.auth.named                        # named callers get their own sessions
        self.authenticated = self.auth.enabled
        self._rate, self._burst = rate_limit, burst
        self._limiters: dict[str, RateLimiter] = {}
        self.requests: dict[str, int] = {}
        self.rejected: dict[str, int] = {}
        self._count = threading.Lock()
        self.routes: list[Route] = API_ROUTES + (UI_ROUTES if ui else [])
        self.ui_dir, self.allow_framing = ui_dir, allow_framing
        self.access_log = access_log
        self.history = RunHistory(redact=lambda text: service.engine.guardrails.redact(text)[0])
        self.metrics: AgentMetrics | None = AgentMetrics(service, self) if metrics else None
        if self.metrics is not None:
            service.observers.append(self.metrics.observe_result)
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, name="agi-api-loop", daemon=True)
        self.tls = tls is not None
        self._httpd = _Server((host, port), self._handler(), max_connections, tls)
        self._thread: threading.Thread | None = None
        self.host, self.port = self._httpd.server_address[:2]

    @property
    def url(self) -> str:
        return f"{'https' if self.tls else 'http'}://{self.host}:{self.port}"

    @property
    def limiter(self) -> RateLimiter | None:                # kept for callers of the 0.3 attribute
        return self._limiter("default")

    def _limiter(self, caller: str) -> RateLimiter | None:
        if not self._rate:
            return None
        with self._count:
            if caller not in self._limiters:
                self._limiters[caller] = RateLimiter(self._rate, self._burst)
            return self._limiters[caller]

    def _tally(self, table: dict[str, int], key: str) -> None:
        with self._count:
            table[key] = table.get(key, 0) + 1

    def http_stats(self) -> dict:
        with self._count:
            return {"tls": self.tls, "requests": dict(self.requests), "rejected": dict(self.rejected),
                    "active_connections": self._httpd.active, "refused_connections": self._httpd.refused}

    # -- lifecycle ----------------------------------------------------------- #
    def start(self) -> "AgentServer":
        """Serve in background threads and return at once."""
        self._loop_thread.start()
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="agi-api-http", daemon=True)
        self._thread.start()
        return self

    def serve_forever(self) -> None:
        self._loop_thread.start()
        try:
            self._httpd.serve_forever()
        finally:
            self.stop()

    def stop(self, drain: float = 10.0) -> None:
        """Stop accepting connections, then give requests in progress up to ``drain`` seconds to finish."""
        if self._thread is not None:
            self._httpd.shutdown()
            self._thread.join(timeout=5)
            self._thread = None
        deadline = time.monotonic() + max(0.0, drain)
        while self.service.in_flight > 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        self._httpd.server_close()
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=5)
        if self.metrics is not None and self.metrics.observe_result in self.service.observers:
            self.service.observers.remove(self.metrics.observe_result)

    def __enter__(self) -> "AgentServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- routing ------------------------------------------------------------- #
    def wait(self, coroutine) -> Any:
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result()

    def find(self, method: str, path: str) -> tuple[Route | None, list[str], list[Route]]:
        """``(route, captured segments, every route for this path)``. ``route`` is ``None`` when no route
        takes this method; the last item says which methods the path does take."""
        parts = [unquote(p) for p in path.strip("/").split("/") if p]
        chosen, captured, same_path = None, [], []
        for route in self.routes:
            found = route.match(parts)
            if found is None:
                continue
            same_path.append(route)
            if route.method == method and chosen is None:
                chosen, captured = route, found
        return chosen, captured, same_path

    def handle(self, method: str, path: str, body: Any, caller: str = "default") -> tuple[int, Any] | Response:
        route, captured, same_path = self.find(method, path)
        if route is None:
            if same_path:
                allowed = ", ".join(sorted({r.method for r in same_path}))
                raise HTTPError(405, "method_not_allowed", f"use {allowed} for /{path.strip('/')}")
            raise HTTPError(404, "not_found", f"no such endpoint: {method} {path}")
        return route.handler(Context(self, caller, body), *captured)

    def dispatch(self, method: str, path: str, body: Any, caller: str = "default") -> tuple[int, Any]:
        """Answer a JSON endpoint without going through a socket. Raises :class:`HTTPError`."""
        outcome = self.handle(method, path, body, caller)
        if isinstance(outcome, Response):
            raise HTTPError(406, "not_json", f"{path} does not return JSON")
        return outcome

    def _handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            server_version = "agi"
            timeout = 60                                   # a client that stops sending cannot hold a thread for ever

            def log_message(self, *args) -> None:           # quiet: traces and the access log are the record
                pass

            def _head(self, status: int, content_type: str, length: int, headers: dict | None) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(length))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("X-Request-Id", self.request_id)
                if server.tls:
                    self.send_header("Strict-Transport-Security", "max-age=31536000")
                merged = {"Cache-Control": "no-store", **(headers or {})}
                for key, value in merged.items():
                    self.send_header(key, value)
                self.end_headers()

            def _send(self, status: int, payload: Any, headers: dict | None = None) -> None:
                data = json.dumps(payload, default=str).encode()
                self._head(status, "application/json", len(data), headers)
                self.wfile.write(data)

            def _fail(self, status: int, kind: str, message: str, headers: dict | None = None) -> None:
                self._send(status, {"error": {"type": kind, "message": message}}, headers)

            def _body(self) -> Any:
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    raise HTTPError(400, "bad_request", "Content-Length is not a number") from None
                if length > MAX_BODY:
                    # Read and discard a moderately oversized body so the client receives the 413 instead
                    # of a connection reset; give up on anything huge. Either way the connection is closed.
                    self.close_connection = True
                    remaining = length if length <= DRAIN_LIMIT else 0
                    while remaining > 0:
                        chunk = self.rfile.read(min(65536, remaining))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                    raise HTTPError(413, "too_large", f"request body exceeds {MAX_BODY} bytes")
                raw = self.rfile.read(length) if length else b""
                if not raw:
                    raise HTTPError(400, "bad_request", "a JSON body is required")
                try:
                    return json.loads(raw)
                except (ValueError, UnicodeDecodeError):
                    raise HTTPError(400, "bad_request", "the body is not valid JSON") from None

            def _admit(self, same_path: list[Route]) -> str:
                """Check the token and the rate limit; return the caller's name."""
                if any(route.public for route in same_path):
                    return "default"
                caller = "default"
                if server.authenticated:
                    caller = server.auth.who(self.headers.get("Authorization", ""))
                    if caller is None:
                        self.caller = "(unauthenticated)"
                        if not server.auth.may_guess(self.client_address[0]):
                            raise HTTPError(429, "rate_limited", "too many failed attempts; slow down")
                        raise HTTPError(401, "unauthorized", "missing or wrong bearer token")
                self.caller = caller
                limiter = server._limiter(caller)
                if limiter is not None and not limiter.allow():
                    raise HTTPError(429, "rate_limited", "too many requests; slow down")
                server._tally(server.requests, caller)
                return caller

            def _handle(self, method: str) -> None:
                started = time.monotonic()
                path = urlsplit(self.path).path
                given = self.headers.get("X-Request-Id", "")
                self.request_id = given if _REQUEST_ID.fullmatch(given) else secrets.token_hex(8)
                self.caller = "default"
                status, label = 500, "other"
                try:
                    route, captured, same_path = server.find(method, path)
                    if same_path:
                        label = same_path[0].pattern
                    caller = self._admit(same_path)
                    if route is None:
                        server.handle(method, path, None, caller)        # raises the right 404 or 405
                    body = self._body() if method == "POST" else None
                    outcome = route.handler(Context(server, caller, body), *captured)
                    if isinstance(outcome, Response):
                        status = outcome.status
                        self._head(status, outcome.content_type, len(outcome.body), outcome.headers)
                        self.wfile.write(outcome.body)
                    else:
                        status, payload = outcome
                        self._send(status, payload)
                except HTTPError as exc:
                    status = exc.status
                    if status in (401, 429):
                        server._tally(server.rejected, self.caller)
                    extra = {"WWW-Authenticate": "Bearer"} if status == 401 else \
                            {"Retry-After": "1"} if status == 429 else None
                    self._fail(status, exc.kind, exc.message, extra)
                except ValidationError as exc:
                    status = 422
                    problems = "; ".join(f"{'.'.join(str(p) for p in e['loc']) or 'body'}: {e['msg']}"
                                         for e in exc.errors()[:5])
                    self._fail(422, "invalid_request", problems)
                except (BrokenPipeError, ConnectionResetError):
                    status = 499                            # the client went away before the answer
                    self.close_connection = True
                except Exception as exc:                    # never leak a stack trace to a client
                    status = 500
                    self._fail(500, "internal_error", type(exc).__name__)
                finally:
                    seconds = time.monotonic() - started
                    if server.metrics is not None:
                        server.metrics.observe_http(method, label, status, seconds)
                    if server.access_log is not None:
                        server.access_log("request", "warning" if status >= 500 else "info",
                                          request_id=self.request_id, caller=self.caller, method=method,
                                          route=label, path=path[:200], status=status,
                                          duration_ms=round(seconds * 1000, 1), remote=self.client_address[0])

            def do_GET(self) -> None:
                self._handle("GET")

            def do_POST(self) -> None:
                self._handle("POST")

            def do_DELETE(self) -> None:
                self._handle("DELETE")

        return Handler


def serve(service: AgentService, host: str = "127.0.0.1", port: int = 8000, token: str | None = None,
          rate_limit: float | None = None, **options) -> None:
    """Run the server until interrupted. ``options`` are passed to :class:`AgentServer`."""
    server = AgentServer(service, host, port, token, rate_limit, **options)
    print(f"agent API listening on {server.url}" + ("" if server.authenticated else "  (no token: local use only)"),
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        service.close()
