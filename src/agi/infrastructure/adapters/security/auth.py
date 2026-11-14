"""Bearer-token authentication for the HTTP API.

One shared ``token``, or one key per named caller. Secrets are compared as
SHA-256 digests in constant time, and against every key, so the time taken
says nothing about which key (if any) came close.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import threading

from agi.application.cortex.resilience import RateLimiter

MIN_KEY_LENGTH = 16
MAX_AUTH_FAILURES = 10                # wrong tokens allowed per address in a burst, then one per second
_NAME = re.compile(r"[A-Za-z0-9._-]{1,64}")


def _digest(secret: str) -> bytes:
    return hashlib.sha256(secret.encode()).digest()


class ApiKeys:
    def __init__(self, token: str | None = None, keys: dict[str, str] | None = None):
        keys = dict(keys or {})
        for name, secret in keys.items():
            if not _NAME.fullmatch(name):
                raise ValueError(f"key name {name!r} must be 1-64 letters, digits, '.', '_' or '-'")
            if len(secret) < MIN_KEY_LENGTH:
                raise ValueError(f"the key for {name!r} is too short: use at least {MIN_KEY_LENGTH} characters "
                                 "(for example secrets.token_urlsafe(32))")
        if len(set(keys.values())) != len(keys) or (token and token in keys.values()):
            raise ValueError("every caller needs a different key")
        self.named = bool(keys)                             # named callers get their own sessions
        self._keys: dict[str, bytes] = {name: _digest(secret) for name, secret in keys.items()}
        if token:
            self._keys.setdefault("default", _digest(token))
        self._failures: dict[str, RateLimiter] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self._keys)

    def callers(self) -> list[str]:
        return sorted(self._keys)

    def who(self, authorization: str) -> str | None:
        """The caller a ``Bearer`` header belongs to, or ``None``."""
        if not authorization.startswith("Bearer "):
            return None
        given, found = _digest(authorization[7:]), None
        for name, expected in self._keys.items():
            if hmac.compare_digest(given, expected):
                found = name
        return found

    def may_guess(self, address: str) -> bool:
        """Whether this address may make another wrong guess at a token (slows brute force)."""
        with self._lock:
            if len(self._failures) > 10_000:
                self._failures.clear()
            if address not in self._failures:
                self._failures[address] = RateLimiter(1.0, MAX_AUTH_FAILURES)
            return self._failures[address].allow()
