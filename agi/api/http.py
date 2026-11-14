"""A small JSON-over-HTTP front end for :class:`AgentService` (standard library only).

    GET    /health                 liveness (no authentication)
    GET    /v1/tools               the tools the agent has
    GET    /v1/stats               counters, memory sizes, drift status
    GET    /v1/traces/{trace_id}   every recorded step of one run
    POST   /v1/run                 {"task": "...", "mode"?, "session"?, "steps"?}
    POST   /v1/batch               {"tasks": ["...", ...]}
    POST   /v1/solve               {"task": "...", "subtasks"?: [...]}   split, run in parallel, combine
    DELETE /v1/sessions/{id}       forget a conversation

It listens on 127.0.0.1 unless told otherwise, and refuses to listen on any
other address without a token: an agent that can run code and write files must
not be reachable anonymously. It speaks plain HTTP; put a TLS-terminating proxy
in front of it for anything beyond a trusted network.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Literal
from urllib.parse import unquote, urlsplit

from pydantic import Field, ValidationError

from ..core.resilience import RateLimiter
from ..core.types import EngineResult, Model
from .service import AgentService

MAX_BODY = 1_000_000
DRAIN_LIMIT = 8 * MAX_BODY
LOOPBACK = ("127.0.0.1", "::1", "localhost")


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


def result_json(result: EngineResult, steps: bool = False) -> dict:
    data = result.model_dump(exclude=None if steps else {"steps", "route"})
    if not steps:
        data["steps"] = len(result.steps)
        data["route_reason"] = result.route.reason if result.route else None
    return data


class AgentServer:
    """Serve an :class:`AgentService` over HTTP. ``port=0`` picks a free port."""

    def __init__(self, service: AgentService, host: str = "127.0.0.1", port: int = 8000,
                 token: str | None = None, rate_limit: float | None = None, burst: int = 10):
        if host not in LOOPBACK and not token:
            raise ValueError(f"refusing to listen on {host} without a token: pass token=... "
                             "(or bind to 127.0.0.1 for local use only)")
        self.service, self.token = service, token
        self.limiter = RateLimiter(rate_limit, burst) if rate_limit else None
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, name="agi-api-loop", daemon=True)
        self._httpd = ThreadingHTTPServer((host, port), self._handler())
        self._httpd.daemon_threads = True
        self._thread: threading.Thread | None = None
        self.host, self.port = self._httpd.server_address[:2]

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

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

    def stop(self) -> None:
        if self._thread is not None:
            self._httpd.shutdown()
            self._thread.join(timeout=5)
            self._thread = None
        self._httpd.server_close()
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=5)

    def __enter__(self) -> "AgentServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- routing ------------------------------------------------------------- #
    def _await(self, coroutine) -> Any:
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result()

    def dispatch(self, method: str, path: str, body: Any) -> tuple[int, Any]:
        service = self.service
        parts = [unquote(p) for p in path.strip("/").split("/") if p]
        if method == "GET" and parts == ["health"]:
            return 200, service.health()
        if method == "GET" and parts == ["v1", "tools"]:
            return 200, {"tools": service.tools()}
        if method == "GET" and parts == ["v1", "stats"]:
            return 200, service.stats()
        if method == "GET" and len(parts) == 3 and parts[:2] == ["v1", "traces"]:
            events = service.trace(parts[2])
            if not events:
                raise HTTPError(404, "not_found", f"no trace with id {parts[2]!r}")
            return 200, {"trace_id": parts[2], "events": events}
        if method == "DELETE" and len(parts) == 3 and parts[:2] == ["v1", "sessions"]:
            if not service.end_session(parts[2]):
                raise HTTPError(404, "not_found", f"no session {parts[2]!r}")
            return 200, {"ended": parts[2]}
        if method == "POST" and parts == ["v1", "run"]:
            request = RunRequest.model_validate(body)
            result = self._await(service.ask(request.task, request.mode, request.session))
            return 200, result_json(result, request.steps)
        if method == "POST" and parts == ["v1", "batch"]:
            request = BatchRequest.model_validate(body)
            results = self._await(service.ask_many(request.tasks, request.mode))
            return 200, {"results": [result_json(r) for r in results]}
        if method == "POST" and parts == ["v1", "solve"]:
            request = SolveRequest.model_validate(body)
            return 200, self._await(service.solve_parallel(request.task, request.subtasks)).model_dump()
        known = {("health",): "GET", ("v1", "tools"): "GET", ("v1", "stats"): "GET", ("v1", "run"): "POST",
                 ("v1", "batch"): "POST", ("v1", "solve"): "POST"}
        if tuple(parts) in known:
            raise HTTPError(405, "method_not_allowed", f"use {known[tuple(parts)]} for /{'/'.join(parts)}")
        raise HTTPError(404, "not_found", f"no such endpoint: {method} {path}")

    def _handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            server_version = "agi"
            timeout = 60                                   # a client that stops sending cannot hold a thread for ever

            def log_message(self, *args) -> None:           # quiet: traces are the record
                pass

            def _send(self, status: int, payload: Any, headers: dict | None = None) -> None:
                data = json.dumps(payload, default=str).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.end_headers()
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

            def _handle(self, method: str) -> None:
                path = urlsplit(self.path).path
                try:
                    if path.rstrip("/") != "/health":
                        if server.token:
                            given = self.headers.get("Authorization", "")
                            if not hmac.compare_digest(given.encode(), f"Bearer {server.token}".encode()):
                                raise HTTPError(401, "unauthorized", "missing or wrong bearer token")
                        if server.limiter is not None and server.limiter._take() > 0:
                            raise HTTPError(429, "rate_limited", "too many requests; slow down")
                    body = self._body() if method == "POST" else None
                    status, payload = server.dispatch(method, path, body)
                    self._send(status, payload)
                except HTTPError as exc:
                    extra = {"WWW-Authenticate": "Bearer"} if exc.status == 401 else \
                            {"Retry-After": "1"} if exc.status == 429 else None
                    self._fail(exc.status, exc.kind, exc.message, extra)
                except ValidationError as exc:
                    problems = "; ".join(f"{'.'.join(str(p) for p in e['loc']) or 'body'}: {e['msg']}"
                                         for e in exc.errors()[:5])
                    self._fail(422, "invalid_request", problems)
                except Exception as exc:                    # never leak a stack trace to a client
                    self._fail(500, "internal_error", type(exc).__name__)

            def do_GET(self) -> None:
                self._handle("GET")

            def do_POST(self) -> None:
                self._handle("POST")

            def do_DELETE(self) -> None:
                self._handle("DELETE")

        return Handler


def serve(service: AgentService, host: str = "127.0.0.1", port: int = 8000, token: str | None = None,
          rate_limit: float | None = None) -> None:
    """Run the server until interrupted."""
    server = AgentServer(service, host, port, token, rate_limit)
    print(f"agent API listening on {server.url}" + ("" if token else "  (no token: local use only)"))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        service.close()
