#!/usr/bin/env python3
"""Check a running server from the outside: health, sign-in, one task, metrics, the console.

    python scripts/smoke.py http://127.0.0.1:8000              # token from AGI_API_TOKEN
    python scripts/smoke.py https://agent.example.com --token-file secrets/agi_api_token.txt

Exits 0 when every check passes, 1 otherwise. Uses only the standard library.
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import urllib.error
import urllib.request


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url", nargs="?", default="http://127.0.0.1:8000")
    parser.add_argument("--token-env", default="AGI_API_TOKEN")
    parser.add_argument("--token-file")
    parser.add_argument("--task", default="What is 17 * 23?")
    parser.add_argument("--expect", default="391", help="text the answer must contain ('' = any answer)")
    parser.add_argument("--insecure", action="store_true", help="accept a self-signed certificate")
    options = parser.parse_args(argv)
    base = options.url.rstrip("/")
    token = os.environ.get(options.token_env, "")
    if options.token_file:
        with open(options.token_file, encoding="utf-8") as handle:
            token = handle.read().strip()
    context = ssl._create_unverified_context() if options.insecure else None

    def call(method: str, path: str, body: dict | None = None, auth: bool = True) -> tuple[int, str, dict]:
        request = urllib.request.Request(base + path, method=method,
                                         data=None if body is None else json.dumps(body).encode())
        if body is not None:
            request.add_header("Content-Type", "application/json")
        if auth and token:
            request.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(request, timeout=330, context=context) as reply:
                return reply.status, reply.read().decode("utf-8", "replace"), dict(reply.headers)
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode("utf-8", "replace"), dict(error.headers)

    failures = 0

    def check(name: str, passed: bool, detail: str = "") -> None:
        nonlocal failures
        failures += not passed
        print(f"{'ok  ' if passed else 'FAIL'}  {name}{'  ' + detail if detail else ''}")

    try:
        status, text, _ = call("GET", "/health", auth=False)
    except (urllib.error.URLError, OSError) as error:
        print(f"FAIL  cannot reach {base}: {error}")
        return 1
    health = json.loads(text) if status == 200 else {}
    check("health", status == 200, f"status={health.get('status')} model={health.get('model')}")
    if health.get("auth_required"):
        status, _, _ = call("GET", "/v1/info", auth=False)
        check("requests without a token are refused", status == 401, f"got {status}")
        if not token:
            print(f"FAIL  the server needs a token: set {options.token_env} or pass --token-file")
            return 1

    status, text, _ = call("GET", "/v1/info")
    info = json.loads(text) if status == 200 else {}
    check("info", status == 200, f"version={info.get('version')} model={info.get('model')} demo={info.get('demo')}")

    status, text, _ = call("POST", "/v1/run", {"task": options.task})
    result = json.loads(text) if status == 200 else {}
    answer = str(result.get("answer") or "")
    check("run a task", status == 200 and bool(answer) and options.expect in answer,
          f"mode={result.get('mode')} answer={answer[:80]!r}" if status == 200 else f"got {status}: {text[:200]}")

    trace_id = result.get("trace_id")
    if trace_id:
        status, text, _ = call("GET", f"/v1/traces/{trace_id}")
        check("trace of that run", status == 200 and bool(json.loads(text).get("events")), f"got {status}")
    status, text, _ = call("GET", "/v1/runs")
    check("recent runs", status == 200 and bool(json.loads(text).get("runs")), f"got {status}")

    if info.get("metrics"):
        status, text, _ = call("GET", "/metrics")
        check("metrics", status == 200 and "agi_runs_total" in text, f"got {status}")
    else:
        print("skip  metrics (switched off, or hidden by the proxy)")

    status, text, headers = call("GET", "/", auth=False)
    if status == 200:
        check("web console", '<div id="root">' in text and "Content-Security-Policy" in headers)
    else:
        print(f"skip  web console (got {status}: not built or switched off)")

    print("all checks passed" if not failures else f"{failures} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
