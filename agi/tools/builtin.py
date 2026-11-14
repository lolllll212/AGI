"""Built-in tools: arithmetic, sandboxed Python, files, tabular data, HTTP, time, memory.

Each factory returns one or more :class:`~agi.tools.base.Tool` objects bound to
the resources they are allowed to touch (a workspace directory, a sandbox, a
memory engine). :func:`standard_tools` assembles the usual set.
"""
from __future__ import annotations

import ast
import csv
import datetime as _dt
import html
import io
import ipaddress
import json
import math
import operator
import re
import socket
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable

from ..sandbox import Sandbox
from .base import Param, Tool, ToolError, ToolRegistry

# --------------------------------------------------------------------------- #
# calculator
# --------------------------------------------------------------------------- #
_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
_COMPARE = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
            ast.Eq: operator.eq, ast.NotEq: operator.ne}
_FUNCTIONS: dict[str, Callable] = {
    name: getattr(math, name) for name in (
        "sqrt", "exp", "log", "log10", "log2", "sin", "cos", "tan", "asin", "acos", "atan",
        "atan2", "sinh", "cosh", "tanh", "floor", "ceil", "factorial", "comb", "perm", "gcd",
        "hypot", "degrees", "radians", "isqrt", "fsum")}
_FUNCTIONS.update(abs=abs, round=round, min=min, max=max, sum=lambda *v: math.fsum(
    v[0] if len(v) == 1 and isinstance(v[0], (list, tuple)) else v))
_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf}


def evaluate(expression: str) -> int | float | bool:
    """Evaluate an arithmetic expression without ``eval``.

    Integers and ratios of integers are computed exactly, so ``0.1 + 0.2`` style
    surprises only appear when a float is genuinely involved.
    """
    if len(expression) > 1000:
        raise ToolError("expression is too long")
    if "^" in expression:
        raise ToolError("'^' is not a power operator here; write ** instead")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"cannot parse {expression!r}: {exc.msg}. Use plain arithmetic such as "
                        f"(1200 * 1.07) / 12; write numbers without thousands separators") from None

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ToolError(f"only numbers are allowed, not {node.value!r}")
            return Fraction(node.value) if isinstance(node.value, int) else node.value
        if isinstance(node, ast.Name):
            if node.id not in _CONSTANTS:
                raise ToolError(f"unknown name {node.id!r}; available constants: {', '.join(_CONSTANTS)}")
            return _CONSTANTS[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = walk(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
                raise ToolError("division by zero")
            return _BINARY[type(node.op)](left, right)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            base, power = walk(node.left), walk(node.right)
            if isinstance(base, Fraction) and isinstance(power, Fraction) and power.denominator == 1:
                bits = max(base.numerator.bit_length(), base.denominator.bit_length())
                if abs(power) * max(bits, 1) > 20_000:
                    raise ToolError("that power is too large to compute exactly")
                if base == 0 and power < 0:
                    raise ToolError("division by zero")
                return base ** int(power)
            try:
                result = float(base) ** float(power)
            except (OverflowError, ZeroDivisionError) as exc:
                raise ToolError(f"cannot compute that power: {exc}") from None
            if isinstance(result, complex):
                raise ToolError("the result is not a real number")
            return result
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _COMPARE:
            return _COMPARE[type(node.ops[0])](walk(node.left), walk(node.comparators[0]))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            fn = _FUNCTIONS.get(node.func.id)
            if fn is None:
                raise ToolError(f"unknown function {node.func.id!r}; available: "
                                f"{', '.join(sorted(_FUNCTIONS))}")
            args = [walk(a) for a in node.args]
            args = [int(a) if isinstance(a, Fraction) and a.denominator == 1
                    else float(a) if isinstance(a, Fraction) else a for a in args]
            try:
                value = fn(*args)
            except (ValueError, TypeError, OverflowError) as exc:
                raise ToolError(f"{node.func.id}: {exc}") from None
            return Fraction(value) if isinstance(value, int) and not isinstance(value, bool) else value
        if isinstance(node, (ast.List, ast.Tuple)):
            return [walk(e) for e in node.elts]
        raise ToolError(f"unsupported syntax: {type(node).__name__}")

    value = walk(tree)
    if isinstance(value, list):                    # "1,200 + 5" parses as the pair (1, 205)
        raise ToolError("that is a list of values, not one expression; write numbers without "
                        "thousands separators (1200, not 1,200)")
    if isinstance(value, Fraction):
        return value.numerator if value.denominator == 1 else float(value)
    return value


def format_number(value: Any) -> str:
    if isinstance(value, float):
        if value != value or value in (math.inf, -math.inf):
            return str(value)
        return f"{value:.12g}"
    return str(value)


def calculator_tool() -> Tool:
    def calculator(expression: str) -> str:
        return format_number(evaluate(expression))
    return Tool("calculator", "Evaluate an arithmetic expression exactly. Supports + - * / // % **, "
                "parentheses, comparisons, constants pi and e, and functions such as sqrt, log, exp, "
                "round, floor, ceil, min, max, factorial, comb.", calculator,
                {"expression": Param("string", "e.g. \"2340 * 17 / 100\" or \"round(1000 * 1.05 ** 10, 2)\"")})


# --------------------------------------------------------------------------- #
# sandboxed python
# --------------------------------------------------------------------------- #
def python_tool(sandbox: Sandbox | None = None) -> Tool:
    box = sandbox or Sandbox()
    allowed = ", ".join(sorted(box.policy.allowed_imports))

    def python(code: str) -> str:
        outcome = box.run(code)
        parts = []
        if outcome.stdout.strip():
            parts.append(outcome.stdout.rstrip())
        if outcome.ok:
            if outcome.value is not None:
                parts.append(f"result = {outcome.value!r}")
            return "\n".join(parts) or "(ran without output; print something or assign to `result`)"
        if outcome.violations:
            raise ToolError("PolicyViolation: " + "; ".join(outcome.violations[:3])
                            + f". Allowed imports: {allowed}. Files and network are not available here.")
        failure = outcome.failure or "execution failed"
        raise ToolError(failure + ("\n[printed before the error]\n" + parts[0] if parts else ""))

    return Tool("python", "Run Python code in an isolated sandbox and return what it prints, plus "
                "the value of a variable named `result` if you set one. No file or network access; "
                f"imports limited to: {allowed}. State does not persist between calls.", python,
                {"code": Param("string", "complete Python source")},
                timeout_s=box.policy.wall_time_s + 5)


# --------------------------------------------------------------------------- #
# workspace files
# --------------------------------------------------------------------------- #
class Workspace:
    """A directory the agent may read and write; nothing outside it is reachable."""

    def __init__(self, root: str | Path, read_only: bool = False, max_file_bytes: int = 2_000_000):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(f"workspace {self.root} does not exist")
        self.read_only = read_only
        self.max_file_bytes = max_file_bytes

    def resolve(self, path: str) -> Path:
        if "\x00" in path:
            raise ToolError("invalid path")
        target = (self.root / path).resolve()             # follows symlinks before checking
        if target != self.root and self.root not in target.parents:
            raise ToolError(f"PermissionDenied: {path!r} is outside the workspace")
        return target

    def relative(self, path: Path) -> str:
        return str(path.relative_to(self.root)) or "."

    def existing(self, path: str) -> Path:
        target = self.resolve(path)
        if not target.exists():
            nearby = sorted(self.relative(p) for p in self.root.rglob("*") if p.is_file()
                            and not any(part.startswith(".") for part in p.relative_to(self.root).parts))[:20]
            raise ToolError(f"FileNotFound: no file {path!r} in the workspace. "
                            f"Files present: {', '.join(nearby) or '(none)'}")
        return target

    def read_text(self, path: str) -> str:
        target = self.existing(path)
        if target.is_dir():
            raise ToolError(f"IsDirectory: {path!r} is a directory; use list_files")
        if target.stat().st_size > self.max_file_bytes:
            raise ToolError(f"FileTooLarge: {path!r} is {target.stat().st_size} bytes; "
                            f"use search_files or query_data instead of reading it whole")
        try:
            return target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ToolError(f"NotText: {path!r} is not a UTF-8 text file") from None


def file_tools(workspace: Workspace) -> list[Tool]:
    def list_files(path: str = ".") -> str:
        target = workspace.existing(path)
        if target.is_file():
            return f"{workspace.relative(target)}  ({target.stat().st_size} bytes)"
        rows = []
        for item in sorted(target.rglob("*")):
            if any(part.startswith(".") for part in item.relative_to(target).parts):
                continue
            if item.is_file() and workspace.root in item.resolve().parents:
                rows.append(f"{workspace.relative(item)}  ({item.stat().st_size} bytes)")
            if len(rows) >= 200:
                rows.append("[more files not shown]")
                break
        return "\n".join(rows) or "(the directory is empty)"

    def read_file(path: str, start_line: int = 1, max_lines: int = 200) -> str:
        lines = workspace.read_text(path).splitlines()
        first = max(1, start_line)
        chunk = lines[first - 1: first - 1 + max(1, max_lines)]
        if not chunk:
            return f"(no lines from line {first}; the file has {len(lines)} lines)"
        body = "\n".join(chunk)
        last = first + len(chunk) - 1
        if first > 1 or last < len(lines):
            body += f"\n[lines {first}-{last} of {len(lines)}]"
        return body

    def search_files(pattern: str, path: str = ".") -> str:
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            raise ToolError(f"BadPattern: {exc}") from None
        target = workspace.existing(path)
        files = [target] if target.is_file() else sorted(p for p in target.rglob("*") if p.is_file())
        hits, total = [], 0
        for item in files:
            if workspace.root not in item.resolve().parents or item.stat().st_size > workspace.max_file_bytes:
                continue
            try:
                text = item.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for number, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    total += 1
                    if len(hits) < 50:
                        hits.append(f"{workspace.relative(item)}:{number}: {line.strip()[:200]}")
        if not hits:
            return f"no lines match {pattern!r}"
        return "\n".join(hits) + f"\n[{total} matching line(s) in total]"

    def write_file(path: str, content: str) -> str:
        if workspace.read_only:
            raise ToolError("PermissionDenied: the workspace is read-only")
        if len(content.encode()) > workspace.max_file_bytes:
            raise ToolError("FileTooLarge: content exceeds the workspace file size limit")
        target = workspace.resolve(path)
        if target == workspace.root or target.is_dir():
            raise ToolError(f"IsDirectory: {path!r} is a directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} characters to {workspace.relative(target)}"

    return [
        Tool("list_files", "List the files in the workspace (recursively) with their sizes.",
             list_files, {"path": Param("string", "directory inside the workspace", False, ".")}),
        Tool("read_file", "Read a text file from the workspace.", read_file,
             {"path": Param("string", "file path relative to the workspace"),
              "start_line": Param("integer", "first line to return (1-based)", False, 1),
              "max_lines": Param("integer", "how many lines to return", False, 200)}),
        Tool("search_files", "Search workspace files for a regular expression (case-insensitive); "
             "returns matching lines with file and line number, and the total match count.",
             search_files, {"pattern": Param("string", "regular expression"),
                            "path": Param("string", "file or directory to search", False, ".")}),
        Tool("write_file", "Create or overwrite a text file in the workspace.", write_file,
             {"path": Param("string", "file path relative to the workspace"),
              "content": Param("string", "the full file content")}, risk="write"),
    ]


# --------------------------------------------------------------------------- #
# tabular data
# --------------------------------------------------------------------------- #
_NUMBER = re.compile(r"^[+-]?(\d{1,3}(,\d{3})+|\d+)(\.\d+)?$")


def _cell(text: Any) -> Any:
    """CSV cells arrive as text; numbers (including '1,200.50' and '$40') become numbers."""
    if not isinstance(text, str):
        return text
    stripped = text.strip()
    if stripped == "":
        return None
    candidate = stripped.lstrip("$€£₹").rstrip("%").strip()
    if _NUMBER.match(candidate):
        plain = candidate.replace(",", "")
        return float(plain) if "." in plain else int(plain)
    return stripped


def _load_rows(workspace: Workspace, path: str) -> tuple[list[str], list[list]]:
    text = workspace.read_text(path)
    if path.lower().endswith((".json", ".jsonl")):
        try:
            records = ([json.loads(l) for l in text.splitlines() if l.strip()]
                       if path.lower().endswith(".jsonl") else json.loads(text))
        except ValueError as exc:
            raise ToolError(f"BadData: {path!r} is not valid JSON ({exc})") from None
        if isinstance(records, dict):
            records = next((v for v in records.values() if isinstance(v, list)), None)
        if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
            raise ToolError(f"BadData: {path!r} must contain a list of JSON objects")
        columns = list(dict.fromkeys(k for r in records for k in r))
        return columns, [[r.get(c) if not isinstance(r.get(c), (list, dict)) else json.dumps(r.get(c))
                          for c in columns] for r in records]
    try:
        dialect = csv.Sniffer().sniff(text[:4000], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    rows = [r for r in csv.reader(io.StringIO(text), dialect) if any(c.strip() for c in r)]
    if not rows:
        raise ToolError(f"BadData: {path!r} is empty")
    header = [re.sub(r"\W+", "_", h.strip()).strip("_").lower() or f"col{i}"
              for i, h in enumerate(rows[0])]
    header = [h if header.index(h) == i else f"{h}_{i}" for i, h in enumerate(header)]
    width = len(header)
    return header, [[_cell(c) for c in (r + [""] * width)[:width]] for r in rows[1:]]


def _authorize(action, *_):
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
    if hasattr(sqlite3, "SQLITE_RECURSIVE"):
        allowed.add(sqlite3.SQLITE_RECURSIVE)
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY


def data_tool(workspace: Workspace, max_rows: int = 50) -> Tool:
    def query_data(file: str, sql: str = "") -> str:
        tables = {}
        connection = sqlite3.connect(":memory:")
        try:
            for index, name in enumerate(part.strip() for part in file.split(",") if part.strip()):
                columns, rows = _load_rows(workspace, name)
                table = "data" if index == 0 else re.sub(r"\W+", "_", Path(name).stem).lower()
                quoted = ", ".join(f'"{c}"' for c in columns)
                connection.execute(f'CREATE TABLE "{table}" ({quoted})')
                connection.executemany(
                    f'INSERT INTO "{table}" VALUES ({", ".join("?" * len(columns))})', rows)
                tables[table] = (columns, len(rows))
            if not tables:
                raise ToolError("BadArguments: give at least one file")
            if not sql.strip():
                out = []
                for table, (columns, count) in tables.items():
                    sample = connection.execute(f'SELECT * FROM "{table}" LIMIT 3').fetchall()
                    out.append(f"table {table}: {count} rows; columns: {', '.join(columns)}\n"
                               + "\n".join("  " + " | ".join(map(str, r)) for r in sample))
                return "\n".join(out) + "\n(pass sql=... to query; the first file is table `data`)"
            if not re.match(r"^\s*(select|with)\b", sql, re.IGNORECASE):
                raise ToolError("OnlySelect: only SELECT queries are allowed")
            connection.set_authorizer(_authorize)
            try:
                cursor = connection.execute(sql)
                rows = cursor.fetchmany(max_rows + 1)
            except sqlite3.Error as exc:
                schema = "; ".join(f"{t}({', '.join(c)})" for t, (c, _) in tables.items())
                raise ToolError(f"SQLError: {exc}. Tables: {schema}") from None
            names = [d[0] for d in cursor.description or []]
            body = [" | ".join(names)] + [" | ".join("" if v is None else format_number(v)
                                                      for v in r) for r in rows[:max_rows]]
            if len(rows) > max_rows:
                body.append(f"[only the first {max_rows} rows are shown; add LIMIT or aggregate]")
            if len(rows) == 0:
                body.append("(no rows)")
            return "\n".join(body)
        finally:
            connection.close()

    return Tool("query_data", "Query a CSV or JSON file from the workspace with SQL (SQLite dialect). "
                "The file is loaded as a table named `data`; numbers like \"1,200.50\" or \"$40\" are "
                "parsed as numbers. Call it without sql first to see the columns. Several files can "
                "be given comma-separated; the others become tables named after their file stem.",
                query_data, {"file": Param("string", "e.g. \"sales.csv\" or \"orders.csv, customers.csv\""),
                             "sql": Param("string", "a SELECT statement", False, "")})


# --------------------------------------------------------------------------- #
# http
# --------------------------------------------------------------------------- #
_TAGS = re.compile(r"<(script|style|noscript|svg|head)\b.*?</\1>", re.S | re.I)


def html_to_text(page: str) -> str:
    page = _TAGS.sub(" ", page)
    page = re.sub(r"<(br|/p|/div|/li|/tr|/h[1-6])\b[^>]*>", "\n", page, flags=re.I)
    page = html.unescape(re.sub(r"<[^>]+>", " ", page))
    return re.sub(r"\n\s*\n+", "\n", re.sub(r"[ \t\r\f\v]+", " ", page)).strip()


def check_url(url: str, allow_hosts: Iterable[str] | None = None, allow_private: bool = False) -> str:
    """Validate a URL before fetching it; returns the host. Raises :class:`ToolError`.

    Blocks non-HTTP schemes and hosts that resolve to loopback, private or
    link-local addresses (the usual server-side request forgery targets, such
    as cloud metadata endpoints). The host is resolved again when the request is
    made, so this does not defend against a DNS server that changes its answer
    between the two lookups; use ``allow_hosts`` where that matters.
    """
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise ToolError(f"UrlBlocked: only http and https URLs can be fetched, not {parts.scheme or 'none'}")
    host = (parts.hostname or "").lower()
    if not host:
        raise ToolError("UrlBlocked: the URL has no host")
    if allow_hosts is not None and not any(host == h or host.endswith("." + h) for h in allow_hosts):
        raise ToolError(f"UrlBlocked: host {host!r} is not on the allowlist")
    if not allow_private:
        try:
            addresses = {info[4][0] for info in socket.getaddrinfo(host, parts.port or 80)}
        except socket.gaierror:
            raise ToolError(f"NetworkError: cannot resolve host {host!r}") from None
        for address in addresses:
            ip = ipaddress.ip_address(address.split("%")[0])
            if ip.version == 6 and ip.ipv4_mapped is not None:
                ip = ip.ipv4_mapped
            if not ip.is_global:
                raise ToolError(f"UrlBlocked: {host!r} resolves to a non-public address")
    return host


def http_tool(allow_hosts: Iterable[str] | None = None, allow_private: bool = False,
              max_bytes: int = 300_000, timeout_s: float = 15.0) -> Tool:
    hosts = None if allow_hosts is None else tuple(h.lower() for h in allow_hosts)

    class Redirects(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            check_url(newurl, hosts, allow_private)       # a redirect must pass the same checks
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    opener = urllib.request.build_opener(Redirects)

    def http_get(url: str) -> str:
        check_url(url, hosts, allow_private)
        request = urllib.request.Request(url, headers={"User-Agent": "agi-toolkit/0.2",
                                                       "Accept": "text/*, application/json"})
        try:
            with opener.open(request, timeout=timeout_s) as response:
                kind = response.headers.get_content_type()
                charset = response.headers.get_content_charset() or "utf-8"
                raw = response.read(max_bytes + 1)
        except urllib.error.HTTPError as exc:
            raise ToolError(f"HTTPError: {exc.code} {exc.reason}") from None
        except (urllib.error.URLError, OSError) as exc:
            raise ToolError(f"NetworkError: {getattr(exc, 'reason', exc)}") from None
        if not (kind.startswith("text/") or kind.endswith(("json", "xml"))):
            raise ToolError(f"UnsupportedContent: {kind} cannot be read as text")
        text = raw[:max_bytes].decode(charset, errors="replace")
        if kind == "text/html":
            text = html_to_text(text)
        if len(raw) > max_bytes:
            text += "\n[response truncated]"
        return text

    return Tool("http_get", "Fetch a web page or JSON API over HTTP(S) and return it as text "
                "(HTML is reduced to its readable text).", http_get,
                {"url": Param("string", "an http:// or https:// URL")},
                risk="network", timeout_s=timeout_s + 5)


# --------------------------------------------------------------------------- #
# time, memory, forged tools
# --------------------------------------------------------------------------- #
def time_tool(clock: Callable[[], _dt.datetime] | None = None) -> Tool:
    now = clock or (lambda: _dt.datetime.now(_dt.timezone.utc))

    def current_time() -> str:
        moment = now()
        return f"{moment.strftime('%Y-%m-%d %H:%M:%S')} UTC ({moment.strftime('%A')})"
    return Tool("current_time", "The current date, time (UTC) and weekday. Use it instead of "
                "assuming what today's date is.", current_time)


def memory_tools(memory) -> list[Tool]:
    from ..memory import Lesson

    def remember(fact: str, topic: str = "") -> str:
        lesson = memory.learn(Lesson("fact", topic or fact[:80], fact))
        return f"stored (id {lesson.id})"

    def recall(query: str) -> str:
        hits = memory.recall(query, k=5)
        if not hits:
            return "nothing relevant in memory"
        return "\n".join(f"{i}. {hit.lesson.render()}" for i, hit in enumerate(hits, 1))

    return [
        Tool("remember", "Save a fact to long-term memory so it is available in future sessions.",
             remember, {"fact": Param("string", "the fact, stated completely"),
                        "topic": Param("string", "when this fact is relevant", False, "")}, risk="write"),
        Tool("recall", "Search long-term memory for facts and lessons from earlier sessions.",
             recall, {"query": Param("string", "what you want to know")}),
    ]


def forge_tools(forge, agent: str | None = None) -> list[Tool]:
    """Tools the agent collective wrote, reviewed and approved - still run sandboxed."""
    tools = []
    for name, record in sorted(forge.tools(at=agent).items()):
        def run(args: list, _name=name):
            outcome = forge.call(_name, *args, at=agent)
            if not outcome.ok:
                raise ToolError(outcome.failure or "tool failed")
            return outcome.value
        tools.append(Tool(name, f"{record['description']} (collectively approved, v{record['version']})",
                          run, {"args": Param("array", "positional arguments, as a JSON list")}))
    return tools


def standard_tools(workspace: str | Path | Workspace | None = None, sandbox: Sandbox | None = None,
                   memory=None, network: bool = False, allow_hosts: Iterable[str] | None = None,
                   read_only: bool = False, forge=None, **registry_options) -> ToolRegistry:
    """The usual toolset. Files and data need a ``workspace``; HTTP is off unless ``network``."""
    registry = ToolRegistry([calculator_tool(), python_tool(sandbox), time_tool()], **registry_options)
    if workspace is not None:
        space = workspace if isinstance(workspace, Workspace) else Workspace(workspace, read_only)
        registry.extend(t for t in file_tools(space) if not (space.read_only and t.risk == "write"))
        registry.register(data_tool(space))
    if network:
        registry.register(http_tool(allow_hosts))
    if memory is not None:
        registry.extend(memory_tools(memory))
    if forge is not None:
        registry.extend(forge_tools(forge))
    return registry
