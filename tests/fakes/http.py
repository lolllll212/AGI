"""A minimal HTTP client for talking to a test server."""
from __future__ import annotations

import json
import urllib.error
import urllib.request


def fetch(server, method: str, path: str, body: dict | None = None, token: str | None = None,
          headers: dict | None = None) -> tuple[int, bytes, dict]:
    """``(status, body, headers)``; an error status is returned, not raised."""
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(server.url + path, data=data, method=method, headers=headers or {})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, error.read(), dict(error.headers)


def get_json(server, path: str, token: str | None = None) -> tuple[int, dict]:
    status, body, _ = fetch(server, "GET", path, token=token)
    return status, json.loads(body)


def send_json(server, path: str, body: dict, token: str | None = None, method: str = "POST") -> tuple[int, dict]:
    status, reply, _ = fetch(server, method, path, body, token)
    return status, json.loads(reply)
