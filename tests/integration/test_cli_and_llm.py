import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from agi.domain.services.text import extract_code, extract_json
from agi.infrastructure.adapters.llm.openai_compat import OpenAICompatLLM
from agi.infrastructure.adapters.llm.scripted import ScriptedLLM
from agi.infrastructure.cli.main import DEMOS, main

README = Path(__file__).resolve().parents[2] / "README.md"


@pytest.mark.parametrize("name", sorted(DEMOS))
def test_every_demo_runs(name, capsys):
    assert main([name]) == 0
    out = capsys.readouterr().out
    assert len(out) > 300 and "Traceback" not in out


def test_demo_claims_hold(capsys):
    main(["collective", "world"])
    out = capsys.readouterr().out
    assert out.count("ACCEPTED") == 2 and out.count("REJECTED") == 4
    assert "byte-identical: True" in out and "hash chain intact: True" in out
    assert "NOT causal" in out


def test_cli_rejects_unknown_demo(capsys):
    assert main(["nonsense"]) == 2
    assert main(["--help"]) == 0
    assert "usage:" in capsys.readouterr().out


def test_readme_examples_run():
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(), flags=re.S)
    assert len(blocks) >= 5
    for block in blocks:
        exec(compile(block, "<README>", "exec"), {"__name__": "__readme__"})


def test_extract_json_and_code():
    assert extract_json('Sure: {"a": [1, 2]} hope that helps') == {"a": [1, 2]}
    assert extract_json("noise [1, 2, 3]") == [1, 2, 3]
    assert extract_json("{broken} then {\"ok\": true}") == {"ok": True}
    with pytest.raises(ValueError):
        extract_json("nothing here")
    assert extract_code("text\n```python\nx = 1\n```\nmore") == "x = 1"
    assert extract_code("```\ny = 2\n```") == "y = 2"
    assert extract_code("z = 3") == "z = 3"


def test_scripted_llm():
    llm = ScriptedLLM(["one"])
    assert llm("p") == "one" and llm.prompts == ["p"]
    with pytest.raises(RuntimeError):
        llm("again")
    assert ScriptedLLM(lambda p: p.upper())("hi") == "HI"


def test_openai_compatible_client_speaks_the_protocol():
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["auth"] = self.headers.get("Authorization")
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            reply = json.dumps({"choices": [{"message": {"content": "pong"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        llm = OpenAICompatLLM("test-model", base_url=f"http://127.0.0.1:{server.server_port}/v1/",
                              api_key="sk-test", system="be brief")
        assert llm("ping") == "pong"
    finally:
        server.shutdown()
    assert seen["path"] == "/v1/chat/completions" and seen["auth"] == "Bearer sk-test"
    assert seen["body"]["model"] == "test-model"
    assert seen["body"]["messages"] == [{"role": "system", "content": "be brief"},
                                        {"role": "user", "content": "ping"}]
