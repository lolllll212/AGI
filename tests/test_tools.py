import datetime
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agi.collective import Builder, Collective, SafetyReviewer, SpecReviewer, ToolNeed
from agi.memory import MemoryEngine
from agi.tools import (Param, Tool, ToolError, ToolRegistry, Workspace, calculator_tool, check_url,
                       data_tool, evaluate, file_tools, forge_tools, from_function, html_to_text,
                       http_tool, memory_tools, python_tool, standard_tools, time_tool)


# --------------------------------------------------------------- registry --- #
def add(a: int, b: int = 1) -> int:
    """Add two integers.

    a: the first number
    b: the second number
    """
    return a + b


def test_from_function_reads_signature_and_docstring():
    tool = from_function(add)
    assert tool.name == "add" and tool.description == "Add two integers."
    assert tool.params["a"] == Param("integer", "the first number", True, None)
    assert tool.params["b"] == Param("integer", "the second number", False, 1)
    assert tool.signature() == "add(a: integer, b: integer = 1)"
    schema = tool.schema()["function"]
    assert schema["parameters"]["required"] == ["a"] and schema["name"] == "add"
    with pytest.raises(TypeError):
        from_function(lambda *args: 0)


def test_registry_calls_validates_and_logs():
    registry = ToolRegistry([add])
    assert registry.call("add", {"a": 2, "b": 3}).value == 5
    assert registry.call("add", {"a": "4"}).value == 5            # quoted numbers are coerced
    assert "add" in registry and len(registry) == 1 and registry.names == ["add"]
    assert "add(a: integer, b: integer = 1): Add two integers." in registry.describe()
    assert [r.ok for r in registry.log] == [True, True]


@pytest.mark.parametrize("name, args, kind, fragment", [
    ("nope", {}, "UnknownTool", "Available tools: add"),
    ("add", {}, "BadArguments", "missing required argument 'a'"),
    ("add", {"a": 1, "c": 2}, "BadArguments", "unexpected argument(s) 'c'"),
    ("add", {"a": "x"}, "BadArguments", "must be of type integer"),
    ("add", {"a": True}, "BadArguments", "must be of type integer"),
    ("add", [1, 2], "BadArguments", "must be a JSON object"),
])
def test_bad_calls_come_back_as_readable_failures(name, args, kind, fragment):
    result = ToolRegistry([add]).call(name, args)
    assert not result.ok and result.error_type == kind and fragment in result.output
    assert result.output.startswith(kind + ": ") and result.error == result.output


def test_a_crashing_tool_never_raises_into_the_caller():
    registry = ToolRegistry()

    @registry.tool
    def explode(x: int) -> int:
        """Divide by zero."""
        return 1 // (x - x)

    @registry.tool(name="clean_failure")
    def fail() -> str:
        """Fail politely."""
        raise ToolError("QuotaExceeded: come back tomorrow")

    crash = registry.call("explode", {"x": 1})
    assert not crash.ok and crash.error_type == "ZeroDivisionError"
    polite = registry.call("clean_failure")
    assert polite.error_type == "QuotaExceeded" and polite.output == "QuotaExceeded: come back tomorrow"


def test_timeouts_budgets_truncation_and_permissions():
    slow = Tool("slow", "sleeps", lambda: time.sleep(5), timeout_s=0.1)
    big = Tool("big", "long output", lambda: "x" * 500)
    post = Tool("post", "sends", lambda: "sent", risk="network")
    registry = ToolRegistry([slow, big, post], max_output_chars=100, max_calls=3, allow=("safe",))
    started = time.perf_counter()
    assert registry.call("slow").error_type == "Timeout" and time.perf_counter() - started < 2
    out = registry.call("big")
    assert out.truncated and "[output truncated: 400 more characters]" in out.output
    assert registry.call("post").error_type == "NotPermitted"
    assert registry.call("big").error_type == "BudgetExceeded"
    with pytest.raises(ValueError):
        registry.register(big)
    with pytest.raises(ValueError):
        registry.register(Tool("not a name", "x", lambda: 0))


def test_non_string_results_are_rendered_as_json():
    registry = ToolRegistry([Tool("data", "x", lambda: {"a": [1, 2], "b": None})])
    assert json.loads(registry.call("data").output) == {"a": [1, 2], "b": None}


# ------------------------------------------------------------- calculator --- #
@pytest.mark.parametrize("expression, expected", [
    ("2340 * 17 / 100", 397.8), ("0.1 + 0.2", 0.30000000000000004), ("1/3 + 1/3 + 1/3", 1),
    ("2 ** 10", 1024), ("7 // 2", 3), ("-7 % 3", 2), ("(1 + 2) * (3 - 4)", -3),
    ("sqrt(16) + abs(-2)", 6.0), ("round(1000 * 1.05 ** 10, 2)", 1628.89), ("comb(5, 2)", 10),
    ("max(3, 9, 4)", 9), ("sum([1, 2, 3.5])", 6.5), ("2 ** 0.5", 2 ** 0.5), ("3 > 2", True),
    ("factorial(20)", 2432902008176640000), ("10 ** 20 + 1", 10 ** 20 + 1), ("floor(pi)", 3),
])
def test_calculator_is_correct_and_exact(expression, expected):
    assert evaluate(expression) == pytest.approx(expected)
    assert type(evaluate(expression)) is type(expected)


def test_integer_ratios_are_exact():
    assert calculator_tool().fn(expression="1/10 + 2/10") == "0.3"
    assert calculator_tool().fn(expression="12345678901234567890 * 98765432109876543210") == str(
        12345678901234567890 * 98765432109876543210)


@pytest.mark.parametrize("expression, fragment", [
    ("2 ^ 3", "write ** instead"), ("1 / 0", "division by zero"), ("0 ** -1", "division by zero"),
    ("__import__('os').system('id')", "unsupported syntax"), ("open('x')", "unknown function"),
    ("x + 1", "unknown name"), ("'a' * 3", "only numbers"), ("9 ** 9 ** 9", "too large"),
    ("(1).__class__", "unsupported syntax"), ("1 +", "cannot parse"), ("1,200 + 5", "thousands separators"),
    ("sqrt(-1)", "sqrt"), ("lambda: 1", "unsupported syntax"), ("(-8) ** 0.5", "not a real number"),
    ("[x for x in (1,2)]", "unsupported syntax"), ("1" + "+1" * 600, "too long"),
])
def test_calculator_rejects_what_it_should(expression, fragment):
    with pytest.raises(ToolError, match=fragment.replace("*", r"\*")):
        evaluate(expression)


# ----------------------------------------------------------------- python --- #
def test_python_tool_runs_in_the_sandbox():
    registry = ToolRegistry([python_tool()])
    ok = registry.call("python", {"code": "import statistics\nprint('hi')\nresult = statistics.mean([1, 2, 6])"})
    assert ok.ok and ok.output == "hi\nresult = 3"
    assert "ran without output" in registry.call("python", {"code": "x = 1"}).output
    blocked = registry.call("python", {"code": "import os\nos.listdir('/')"})
    assert blocked.error_type == "PolicyViolation" and "Allowed imports" in blocked.output
    crash = registry.call("python", {"code": "print('before')\n[][0]"})
    assert not crash.ok and "IndexError" in crash.output and "before" in crash.output


# ------------------------------------------------------------------ files --- #
@pytest.fixture
def space(tmp_path):
    root = tmp_path / "ws"
    (root / "sub").mkdir(parents=True)
    (root / "notes.txt").write_text("alpha\nbeta\nGamma ray\ndelta\n")
    (root / "sub" / "deep.md").write_text("# Title\ngamma again\n")
    (root / ".secret").write_text("hidden")
    (tmp_path / "outside.txt").write_text("TOP SECRET")
    return Workspace(root)


def files(space, **options):
    return ToolRegistry(file_tools(space), **options)


def test_file_tools_read_list_search_write(space):
    registry = files(space)
    listing = registry.call("list_files").output
    assert "notes.txt  (27 bytes)" in listing and "sub/deep.md" in listing and ".secret" not in listing
    assert registry.call("read_file", {"path": "notes.txt"}).output == "alpha\nbeta\nGamma ray\ndelta"
    part = registry.call("read_file", {"path": "notes.txt", "start_line": 2, "max_lines": 2}).output
    assert part == "beta\nGamma ray\n[lines 2-3 of 4]"
    found = registry.call("search_files", {"pattern": "gamma"}).output
    assert "notes.txt:3: Gamma ray" in found and "sub/deep.md:2" in found and "[2 matching" in found
    assert registry.call("search_files", {"pattern": "zzz"}).output == "no lines match 'zzz'"
    assert registry.call("search_files", {"pattern": "("}).error_type == "BadPattern"
    written = registry.call("write_file", {"path": "out/new.txt", "content": "hello"})
    assert written.ok and (space.root / "out" / "new.txt").read_text() == "hello"


def test_missing_files_get_a_helpful_error(space):
    result = files(space).call("read_file", {"path": "note.txt"})
    assert result.error_type == "FileNotFound" and "notes.txt" in result.output
    assert ".secret" not in result.output
    assert files(space).call("read_file", {"path": "sub"}).error_type == "IsDirectory"


@pytest.mark.parametrize("path", ["../outside.txt", "sub/../../outside.txt", "/etc/passwd",
                                  "../../../../etc/passwd", "bad\x00name"])
def test_nothing_outside_the_workspace_is_reachable(space, path):
    registry = files(space)
    for name, args in [("read_file", {"path": path}), ("write_file", {"path": path, "content": "x"}),
                       ("list_files", {"path": path}), ("search_files", {"pattern": "SECRET", "path": path})]:
        result = registry.call(name, args)
        assert not result.ok and "TOP SECRET" not in result.output, (name, path)
    assert (space.root.parent / "outside.txt").read_text() == "TOP SECRET"


def test_symlinks_cannot_be_used_to_escape(space):
    os.symlink(space.root.parent / "outside.txt", space.root / "link.txt")
    os.symlink(space.root.parent, space.root / "up")
    registry = files(space)
    assert registry.call("read_file", {"path": "link.txt"}).error_type == "PermissionDenied"
    assert registry.call("read_file", {"path": "up/outside.txt"}).error_type == "PermissionDenied"
    assert registry.call("write_file", {"path": "link.txt", "content": "pwned"}).error_type == "PermissionDenied"
    assert "TOP SECRET" not in registry.call("search_files", {"pattern": "SECRET"}).output
    assert "outside" not in registry.call("list_files").output
    assert (space.root.parent / "outside.txt").read_text() == "TOP SECRET"


def test_read_only_workspaces_and_size_limits(tmp_path):
    (tmp_path / "big.txt").write_text("x" * 5000)
    locked = Workspace(tmp_path, read_only=True, max_file_bytes=1000)
    registry = files(locked)
    assert registry.call("write_file", {"path": "a.txt", "content": "x"}).error_type == "PermissionDenied"
    assert registry.call("read_file", {"path": "big.txt"}).error_type == "FileTooLarge"
    assert "write_file" not in standard_tools(locked)
    with pytest.raises(NotADirectoryError):
        Workspace(tmp_path / "missing")


# ------------------------------------------------------------------- data --- #
@pytest.fixture
def data(tmp_path):
    (tmp_path / "sales.csv").write_text(
        'Region,Unit Price,Qty\nNorth,"1,200.50",3\nSouth,$800,2\nNorth,300,\nEast,12%,1\n')
    (tmp_path / "people.json").write_text(json.dumps(
        [{"name": "Asha", "region": "North", "tags": ["a"]}, {"name": "Bilal", "region": "South"}]))
    (tmp_path / "semi.csv").write_text("a;b\n1;2\n3;4\n")
    return ToolRegistry([data_tool(Workspace(tmp_path))])


def test_query_data_describes_and_queries(data):
    described = data.call("query_data", {"file": "sales.csv"}).output
    assert "table data: 4 rows; columns: region, unit_price, qty" in described
    total = data.call("query_data", {"file": "sales.csv", "sql":
                                     "select region, sum(unit_price) t from data group by region order by t desc"})
    assert total.output.splitlines()[:3] == ["region | t", "North | 1500.5", "South | 800"]
    nulls = data.call("query_data", {"file": "sales.csv", "sql": "select count(*) from data where qty is null"})
    assert nulls.output.splitlines()[-1] == "1"
    assert data.call("query_data", {"file": "semi.csv", "sql": "select sum(a + b) from data"}).output.endswith("10")


def test_query_data_joins_files_and_reads_json(data):
    joined = data.call("query_data", {"file": "sales.csv, people.json", "sql":
                                      "select p.name, sum(d.unit_price) from data d join people p "
                                      "on p.region = d.region group by p.name order by p.name"})
    assert joined.output.splitlines()[1:] == ["Asha | 1500.5", "Bilal | 800"]
    assert "(no rows)" in data.call("query_data", {"file": "people.json",
                                                    "sql": "select * from data where name = 'x'"}).output


@pytest.mark.parametrize("sql, kind", [
    ("drop table data", "OnlySelect"), ("delete from data", "OnlySelect"),
    ("select 1; drop table data", "SQLError"), ("with x as (select 1) delete from data", "SQLError"),
    ("select load_extension('x')", "SQLError"),
    ("select bogus from data", "SQLError"), ("pragma database_list", "OnlySelect"),
    ("attach database '/tmp/x.db' as x", "OnlySelect"),
])
def test_query_data_is_read_only(data, sql, kind):
    result = data.call("query_data", {"file": "sales.csv", "sql": sql})
    assert not result.ok and result.error_type == kind
    assert data.call("query_data", {"file": "sales.csv", "sql": "select count(*) from data"}).output.endswith("4")


def test_query_data_error_messages_show_the_schema(data):
    result = data.call("query_data", {"file": "sales.csv", "sql": "select price from data"})
    assert "Tables: data(region, unit_price, qty)" in result.output
    assert data.call("query_data", {"file": "missing.csv"}).error_type == "FileNotFound"


def test_large_results_are_capped(tmp_path):
    (tmp_path / "n.csv").write_text("n\n" + "\n".join(map(str, range(200))) + "\n")
    registry = ToolRegistry([data_tool(Workspace(tmp_path), max_rows=10)])
    out = registry.call("query_data", {"file": "n.csv", "sql": "select n from data"}).output
    assert len(out.splitlines()) == 12 and "only the first 10 rows" in out


# ------------------------------------------------------------------- http --- #
@pytest.fixture
def server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            routes = {
                "/page": (200, "text/html", b"<html><head><title>T</title><script>evil()</script></head>"
                                             b"<body><h1>Hello</h1><p>Price: &pound;5 &amp; up</p></body></html>"),
                "/api": (200, "application/json", b'{"temp_c": 31}'),
                "/image": (200, "image/png", b"\x89PNG"),
                "/big": (200, "text/plain", b"y" * 5000),
            }
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", f"http://localhost:{self.server.server_port}/api")
                self.end_headers()
                return
            status, kind, body = routes.get(self.path, (404, "text/plain", b"nope"))
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_http_get_reads_pages_and_apis(server):
    registry = ToolRegistry([http_tool(allow_private=True, max_bytes=1000)])
    page = registry.call("http_get", {"url": server + "/page"}).output
    assert "Hello" in page and "Price: £5 & up" in page and "evil" not in page and "<" not in page
    assert registry.call("http_get", {"url": server + "/api"}).output == '{"temp_c": 31}'
    assert registry.call("http_get", {"url": server + "/missing"}).output == "HTTPError: 404 Not Found"
    assert registry.call("http_get", {"url": server + "/image"}).error_type == "UnsupportedContent"
    big = registry.call("http_get", {"url": server + "/big"}).output
    assert big.endswith("[response truncated]") and len(big) < 1100


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/", "http://localhost:8080/admin", "http://169.254.169.254/latest/meta-data/",
    "http://10.0.0.5/", "http://192.168.1.1/", "http://[::1]/", "http://2130706433/", "http://0.0.0.0/",
    "file:///etc/passwd", "ftp://example.com/x", "gopher://x", "http:///nohost", "javascript:alert(1)",
])
def test_requests_to_internal_addresses_are_refused(url, server):
    result = ToolRegistry([http_tool()]).call("http_get", {"url": url})
    assert not result.ok and result.error_type in ("UrlBlocked", "NetworkError")
    with pytest.raises(ToolError):
        check_url(url)


def test_default_http_tool_refuses_the_local_test_server(server):
    assert ToolRegistry([http_tool()]).call("http_get", {"url": server + "/api"}).error_type == "UrlBlocked"


def test_allowlist_applies_to_redirects_too(server):
    registry = ToolRegistry([http_tool(allow_hosts=["127.0.0.1"], allow_private=True)])
    assert registry.call("http_get", {"url": server + "/api"}).ok
    hop = registry.call("http_get", {"url": server + "/redirect"})          # -> localhost, not listed
    assert hop.error_type == "UrlBlocked" and "allowlist" in hop.output
    assert check_url("https://api.example.com/x", ["example.com"], allow_private=True) == "api.example.com"
    with pytest.raises(ToolError):
        check_url("https://evilexample.com/x", ["example.com"], allow_private=True)


def test_network_tools_are_off_by_default(tmp_path):
    assert "http_get" not in standard_tools(tmp_path)
    assert "http_get" in standard_tools(tmp_path, network=True)
    assert standard_tools().names == ["calculator", "python", "current_time"]


def test_html_to_text():
    assert html_to_text("<p>a</p><style>x{}</style><div>b &lt; c</div>") == "a\n b < c"


# --------------------------------------------------- time, memory, forge --- #
def test_time_tool_uses_the_clock():
    clock = lambda: datetime.datetime(2026, 11, 14, 9, 30, tzinfo=datetime.timezone.utc)
    assert time_tool(clock).fn() == "2026-11-14 09:30:00 UTC (Saturday)"
    assert str(datetime.datetime.now(datetime.timezone.utc).year) in time_tool().fn()


def test_memory_tools_store_and_find_facts(tmp_path):
    path = str(tmp_path / "m.db")
    with MemoryEngine(path) as memory:
        registry = ToolRegistry(memory_tools(memory))
        assert registry.call("remember", {"fact": "The staging database is reset every Sunday night",
                                          "topic": "staging database maintenance"}).ok
        assert registry.call("recall", {"query": "poetry"}).output == "nothing relevant in memory"
    with MemoryEngine(path) as memory:                      # a later session
        found = ToolRegistry(memory_tools(memory)).call("recall", {"query": "when is the staging database reset?"})
        assert "every Sunday night" in found.output


def test_tools_approved_by_the_collective_become_agent_tools():
    need = ToolNeed("mean", "arithmetic mean of a list", ((([1, 2, 3],), 2.0), (([2, 3],), 2.5)))
    group = Collective([Builder("bob", {"mean": {"code": "def mean(xs):\n    return sum(xs) / len(xs)\n"}}),
                        SpecReviewer("carol"), SafetyReviewer("sam")])
    assert group.request(need)[-1].accepted
    registry = ToolRegistry(forge_tools(group.forge))
    assert "collectively approved, v1" in registry.get("mean").description
    assert registry.call("mean", {"args": [[1, 2, 3, 4]]}).value == 2.5
    failed = registry.call("mean", {"args": [[]]})           # still sandboxed, still fails cleanly
    assert not failed.ok and "ZeroDivisionError" in failed.output
