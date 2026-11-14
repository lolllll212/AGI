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
not be reachable anonymously.

For more than one caller, give each a key (``keys={"alice": "...", "ci": "..."}``):
every key has its own rate limit and its own sessions, so one caller can neither
starve nor read another. ``certfile``/``keyfile`` turn on TLS (1.2 or later).
Connections beyond ``max_connections`` are refused with 503 at once rather than
queued, and ``stop()`` lets requests in progress finish.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import re
import secrets
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Literal
from urllib.parse import unquote, urlsplit

from pydantic import Field, ValidationError

from agi.application.cortex.resilience import RateLimiter
from agi.application.interfaces.service import AgentService
from agi.domain.value_objects.records import EngineResult, Model

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


MIN_KEY_LENGTH = 16
_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")
_BUSY_BODY = b'{"error": {"type": "overloaded", "message": "too many connections; retry shortly"}}'
_BUSY = (b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nRetry-After: 1\r\n"
         b"Connection: close\r\nContent-Length: " + str(len(_BUSY_BODY)).encode() + b"\r\n\r\n" + _BUSY_BODY)
MAX_AUTH_FAILURES = 10                # wrong tokens allowed per address in a burst, then one per second


def _digest(secret: str) -> bytes:
    return hashlib.sha256(secret.encode()).digest()


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
                 max_connections: int = 64):
        """``token`` is one shared secret; ``keys`` maps a caller's name to that caller's own secret.
        ``rate_limit`` (requests per second, with ``burst``) applies per caller."""
        keys = dict(keys or {})
        for name, secret in keys.items():
            if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", name):
                raise ValueError(f"key name {name!r} must be 1-64 letters, digits, '.', '_' or '-'")
            if len(secret) < MIN_KEY_LENGTH:
                raise ValueError(f"the key for {name!r} is too short: use at least {MIN_KEY_LENGTH} characters "
                                 "(for example secrets.token_urlsafe(32))")
        if len(set(keys.values())) != len(keys) or (token and token in keys.values()):
            raise ValueError("every caller needs a different key")
        if host not in LOOPBACK and not (token or keys):
            raise ValueError(f"refusing to listen on {host} without a token: pass token=... or keys=... "
                             "(or bind to 127.0.0.1 for local use only)")
        if bool(certfile) != bool(keyfile) and not certfile:
            raise ValueError("keyfile was given without certfile")
        self.service, self.token = service, token
        self.named = bool(keys)                             # named callers get their own sessions
        self._keys: dict[str, bytes] = {name: _digest(secret) for name, secret in keys.items()}
        if token:
            self._keys.setdefault("default", _digest(token))
        self.authenticated = bool(self._keys)
        self._rate, self._burst = rate_limit, burst
        self._limiters: dict[str, RateLimiter] = {}
        self._failures: dict[str, RateLimiter] = {}
        self.requests: dict[str, int] = {}
        self.rejected: dict[str, int] = {}
        self._count = threading.Lock()
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, name="agi-api-loop", daemon=True)
        tls = None
        if certfile:
            tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            tls.minimum_version = ssl.TLSVersion.TLSv1_2
            tls.load_cert_chain(certfile, keyfile)
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

    def _may_guess(self, address: str) -> bool:
        """Whether this address may make another wrong guess at a token (slows brute force)."""
        with self._count:
            if len(self._failures) > 10_000:
                self._failures.clear()
            if address not in self._failures:
                self._failures[address] = RateLimiter(1.0, MAX_AUTH_FAILURES)
            return self._failures[address].allow()

    def _who(self, authorization: str) -> str | None:
        """The caller a bearer token belongs to. Compares against every key, in constant time each."""
        if not authorization.startswith("Bearer "):
            return None
        given, found = _digest(authorization[7:]), None
        for name, expected in self._keys.items():
            if hmac.compare_digest(given, expected):
                found = name
        return found

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

    def __enter__(self) -> "AgentServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- routing ------------------------------------------------------------- #
    def _await(self, coroutine) -> Any:
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result()

    def dispatch(self, method: str, path: str, body: Any, caller: str = "default") -> tuple[int, Any]:
        service = self.service
        parts = [unquote(p) for p in path.strip("/").split("/") if p]

        def scoped(session: str | None) -> str | None:      # one caller cannot name another caller's session
            return f"{caller}/{session}" if session and self.named else session

        if method == "GET" and parts == ["health"]:
            return 200, service.health()
        if method == "GET" and parts == ["v1", "tools"]:
            return 200, {"tools": service.tools()}
        if method == "GET" and parts == ["v1", "stats"]:
            return 200, {**service.stats(), "http": self.http_stats()}
        if method == "GET" and len(parts) == 3 and parts[:2] == ["v1", "traces"]:
            events = service.trace(parts[2])
            if not events:
                raise HTTPError(404, "not_found", f"no trace with id {parts[2]!r}")
            return 200, {"trace_id": parts[2], "events": events}
        if method == "DELETE" and len(parts) == 3 and parts[:2] == ["v1", "sessions"]:
            if not service.end_session(scoped(parts[2])):
                raise HTTPError(404, "not_found", f"no session {parts[2]!r}")
            return 200, {"ended": parts[2]}
        if method == "POST" and parts == ["v1", "run"]:
            request = RunRequest.model_validate(body)
            result = self._await(service.ask(request.task, request.mode, scoped(request.session)))
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
                self.send_header("X-Request-Id", self.request_id)
                if server.tls:
                    self.send_header("Strict-Transport-Security", "max-age=31536000")
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
                given = self.headers.get("X-Request-Id", "")
                self.request_id = given if _REQUEST_ID.fullmatch(given) else secrets.token_hex(8)
                caller = "default"
                try:
                    if path.rstrip("/") != "/health":
                        if server.authenticated:
                            caller = server._who(self.headers.get("Authorization", ""))
                            if caller is None:
                                caller = "(unauthenticated)"
                                if not server._may_guess(self.client_address[0]):
                                    raise HTTPError(429, "rate_limited", "too many failed attempts; slow down")
                                raise HTTPError(401, "unauthorized", "missing or wrong bearer token")
                        limiter = server._limiter(caller)
                        if limiter is not None and not limiter.allow():
                            raise HTTPError(429, "rate_limited", "too many requests; slow down")
                        server._tally(server.requests, caller)
                    body = self._body() if method == "POST" else None
                    status, payload = server.dispatch(method, path, body, caller)
                    self._send(status, payload)
                except HTTPError as exc:
                    if exc.status in (401, 429):
                        server._tally(server.rejected, caller)
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
          rate_limit: float | None = None, **options) -> None:
    """Run the server until interrupted. ``options`` are passed to :class:`AgentServer`."""
    server = AgentServer(service, host, port, token, rate_limit, **options)
    print(f"agent API listening on {server.url}" + ("" if server.authenticated else "  (no token: local use only)"))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        service.close()
