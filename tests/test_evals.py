import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agi import evals
from agi.__main__ import main
from agi.evals import SUITE, has_number, reference_llm, run_suite, write_fixtures
from agi.llm import LLMError, OpenAICompatEmbedder, OpenAICompatLLM
from agi.memory import Lesson, MemoryEngine
from agi.react import ReActAgent
from agi.tools import standard_tools


def test_truths_are_what_a_person_would_compute_by_hand():
    assert (evals.TOP_REGION, round(evals.TOP_TOTAL, 2)) == ("East", 3739.10)
    assert round(evals.PUNE_TOTAL, 2) == 1470.00 and evals.DAYS == 385 and evals.WORST_WAIT == 225
    assert (evals.ERROR_COUNT, evals.TOP_ERROR, evals.TOP_ERROR_COUNT) == (6, "database timeout after 30s", 3)
    assert evals.COMPOUND == 20687.45 and evals.AVERAGE_UNITS == 7.9


def test_every_task_is_solvable_with_the_builtin_tools():
    report = run_suite(lambda ws, task: ReActAgent(reference_llm(task), standard_tools(ws)))
    assert report.passed == len(SUITE) == 8 and report.success_rate == 1.0
    assert all(r.run.grounded and r.run.tool_errors == 0 for r in report.results)
    assert "8/8 tasks passed" in str(report)


def test_guessing_fails_every_task():
    guess = lambda prompt: "Thought: I know.\nFinal Answer: The total is 4,217.36."
    report = run_suite(lambda ws, task: ReActAgent(guess, standard_tools(ws)))
    assert report.passed == 0 and all(r.run.grounded is False for r in report.results)


@pytest.mark.parametrize("answer, ok", [
    ("It cannot be determined: sales.csv has no West region and no dates.", True),
    ("There is no West region in the data.", True),
    ("The West region sold 4,217.36 in March 2019.", False),
    ("The data does not include dates, but West sold 1,990.00.", False),
    ("West sold nothing.", False),
])
def test_the_unanswerable_task_rewards_honesty(answer, ok, tmp_path):
    task = next(t for t in SUITE if t.name == "unanswerable")
    assert task.check(answer, tmp_path) is ok


def test_checks_reject_near_misses(tmp_path):
    by_name = {t.name: t for t in SUITE}
    assert by_name["top_region"].check("East, with 3,739.10", tmp_path)
    assert not by_name["top_region"].check("North, with 3,739.10", tmp_path)
    assert not by_name["top_region"].check("East, with 3,090.75", tmp_path)
    assert not by_name["log_errors"].check("6 errors, mostly upstream returned 502", tmp_path)
    assert not by_name["write_report"].check("done", tmp_path)                 # the file must exist
    (tmp_path / "report.txt").write_text("average_units=7.9\n")
    assert not by_name["write_report"].check("done", tmp_path)                 # two decimals required
    (tmp_path / "report.txt").write_text("average_units=7.90\n")
    assert by_name["write_report"].check("done", tmp_path)
    assert has_number("about 1,470.00 rupees", 1470) and not has_number("1,471", 1470)


def test_each_task_runs_in_a_fresh_workspace():
    seen = []

    def make(workspace, task):
        seen.append((workspace, sorted(p.name for p in workspace.iterdir())))
        return ReActAgent(reference_llm(task), standard_tools(workspace))

    run_suite(make, SUITE[-2:])                              # write_report, then unanswerable
    assert seen[0][0] != seen[1][0] and "report.txt" not in seen[1][1]
    assert not seen[0][0].exists()                           # cleaned up afterwards


# ------------------------------------------------------------ command line --- #
def test_cli_eval_reference_and_export(tmp_path, capsys):
    out = tmp_path / "runs.jsonl"
    assert main(["eval", "--reference", "--export", str(out)]) == 0
    assert "8/8 tasks passed" in capsys.readouterr().out
    assert len(out.read_text().splitlines()) == 8


def test_cli_train_saves_a_policy(tmp_path, capsys):
    out = tmp_path / "policy.json"
    assert main(["train", "--tasks", "300", "--out", str(out)]) == 0
    assert "held-out 4-step tasks: solved" in capsys.readouterr().out
    assert set(json.loads(out.read_text())) == {"ops", "trained_on", "weights"}


@pytest.mark.parametrize("argv", [["run", "what is 2+2"], ["chat"], ["eval"]])
def test_cli_needs_a_model(argv, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2 and "--model" in capsys.readouterr().err


class _Endpoint:
    """A local stand-in for an OpenAI-compatible server."""

    def __init__(self, respond):
        outer = self
        self.requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append((self.path, self.headers.get("Authorization"), body))
                status, payload = respond(len(outer.requests), self.path, body)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


def chat(content):
    return 200, {"choices": [{"message": {"content": content}}],
                 "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def test_cli_run_drives_a_model_over_http(tmp_path, capsys, monkeypatch):
    write_fixtures(tmp_path)
    replies = ['Thought: sum it\nAction: query_data\nAction Input: {"file": "sales.csv", '
               '"sql": "select sum(units) from data"}',
               "Thought: done\nFinal Answer: 79 units were sold in total."]
    endpoint = _Endpoint(lambda n, path, body: chat(replies[n - 1]))
    monkeypatch.setenv("TEST_KEY", "sk-test")
    try:
        code = main(["run", "How many units were sold in total in sales.csv?", "--model", "stub",
                     "--base-url", endpoint.url, "--api-key-env", "TEST_KEY",
                     "--workspace", str(tmp_path), "--memory", str(tmp_path / "m.db")])
    finally:
        endpoint.close()
    out = capsys.readouterr().out
    assert code == 0 and "79 units were sold in total." in out and "action:  query_data" in out
    path, auth, body = endpoint.requests[0]
    assert path == "/v1/chat/completions" and auth == "Bearer sk-test"
    assert body["stop"] == ["\nObservation:"] and "query_data(file: string" in body["messages"][0]["content"]
    with MemoryEngine(str(tmp_path / "m.db")) as memory:     # the run was written to long-term memory
        assert memory.stats()["episodes"] == 1


def test_cli_chat_reads_turns_from_stdin(tmp_path, capsys, monkeypatch):
    replies = ['Thought: x\nAction: calculator\nAction Input: {"expression": "6 * 7"}',
               "Thought: done\nFinal Answer: 42", "Thought: made up\nFinal Answer: It is 5,000,000.",
               "Thought: insist\nFinal Answer: It is 5,000,000."]
    endpoint = _Endpoint(lambda n, path, body: chat(replies[n - 1]))
    lines = iter(["what is 6 times 7?", "", "how many stars are there?", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
    try:
        assert main(["chat", "--model", "stub", "--base-url", endpoint.url, "--quiet"]) == 0
    finally:
        endpoint.close()
    out = capsys.readouterr().out
    assert "42" in out and "It is 5,000,000." in out and "no tool result supports" in out
    assert "You answered: 42" in endpoint.requests[2][2]["messages"][0]["content"]


# --------------------------------------------------------- model clients --- #
def test_client_retries_transient_failures():
    def respond(n, path, body):
        return (429, {"error": "slow down"}) if n < 3 else chat("ok")
    endpoint = _Endpoint(respond)
    waits = []
    try:
        llm = OpenAICompatLLM("m", endpoint.url, "k", retries=3)
        llm._sleep = waits.append
        assert llm("hi") == "ok"
    finally:
        endpoint.close()
    assert len(endpoint.requests) == 3 and waits == [1.0, 2.0]
    assert llm.calls == 1 and llm.usage == {"prompt_tokens": 10, "completion_tokens": 5}


def test_client_gives_up_cleanly():
    always_down = _Endpoint(lambda n, path, body: (503, {"error": "down"}))
    bad_request = _Endpoint(lambda n, path, body: (400, {"error": "no such model"}))
    odd_shape = _Endpoint(lambda n, path, body: (200, {"unexpected": True}))
    try:
        llm = OpenAICompatLLM("m", always_down.url, retries=2)
        llm._sleep = lambda seconds: None
        with pytest.raises(LLMError, match="HTTP 503"):
            llm("hi")
        assert len(always_down.requests) == 3
        with pytest.raises(LLMError, match="no such model"):
            OpenAICompatLLM("m", bad_request.url)("hi")
        assert len(bad_request.requests) == 1                # client errors are not retried
        with pytest.raises(LLMError, match="unexpected response shape"):
            OpenAICompatLLM("m", odd_shape.url)("hi")
    finally:
        for endpoint in (always_down, bad_request, odd_shape):
            endpoint.close()
    unreachable = OpenAICompatLLM("m", "http://127.0.0.1:9/v1", retries=1, timeout=2)
    unreachable._sleep = lambda seconds: None
    with pytest.raises(LLMError, match="cannot reach"):
        unreachable("hi")


def test_embedding_endpoint_makes_recall_semantic():
    # A toy embedding space: one axis per topic, whatever the wording.
    topics = {"money": ("price", "cost", "invoice", "billing", "stock", "ticker"),
              "weather": ("rain", "forecast", "temperature", "weather")}

    def respond(n, path, body):
        text = body["input"].lower()
        vector = [float(sum(word in text for word in words)) for words in topics.values()]
        return 200, {"data": [{"embedding": vector + [0.01]}]}

    endpoint = _Endpoint(respond)
    try:
        memory = MemoryEngine(embedder=OpenAICompatEmbedder("embed-model", endpoint.url, "k"))
        memory.learn(Lesson("fact", "paying an invoice", "Billing closes on the 25th"))
        memory.learn(Lesson("fact", "planning around rain", "The forecast feed updates hourly"))
        hits = memory.recall("what does the stock ticker cost?")      # no word in common with the lesson
        assert [h.lesson.advice for h in hits][:1] == ["Billing closes on the 25th"]
        before = len(endpoint.requests)
        memory.recall("what does the stock ticker cost?")
        assert len(endpoint.requests) == before                        # every text is embedded once
        assert endpoint.requests[0][0] == "/v1/embeddings" and endpoint.requests[0][2]["model"] == "embed-model"
    finally:
        endpoint.close()
