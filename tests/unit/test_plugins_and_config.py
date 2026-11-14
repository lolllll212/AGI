"""Tools loaded from a plugins directory, and the settings file."""
import json
import sys
from pathlib import Path

import pytest

from agi.domain.entities.tool import ToolRegistry
from agi.infrastructure.adapters.plugins import PluginError, discover, load_plugins
from agi.infrastructure.config import ConfigError, load_settings

REPO = Path(__file__).resolve().parents[2]

GOOD = '''
from agi.domain.entities.tool import Param, Tool

def shout(text):
    return text.upper() + "!"

def tools():
    return [Tool("shout", "Upper-case a text.", shout, {"text": Param("string", "the text")})]
'''


def write(directory, name, source):
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def test_plugins_are_found_as_folders_or_single_files_in_name_order(tmp_path):
    write(tmp_path, "zeta/plugin.py", GOOD)
    write(tmp_path, "alpha.py", GOOD.replace("shout", "yell"))
    write(tmp_path, "_private.py", "raise RuntimeError('never imported')")
    write(tmp_path, ".hidden/plugin.py", "raise RuntimeError('never imported')")
    write(tmp_path, "not_a_plugin/readme.txt", "no plugin.py here")
    write(tmp_path, "notes.txt", "ignored")
    assert [name for name, _ in discover(tmp_path)] == ["alpha", "zeta"]
    tools = load_plugins(tmp_path)
    assert [tool.name for tool in tools] == ["yell", "shout"]
    result = ToolRegistry(tools).call("shout", {"text": "hello"})
    assert result.ok and result.output == "HELLO!"
    assert "agi_plugin_zeta" in sys.modules


@pytest.mark.parametrize("source, message", [
    ("x = 1\n", "must define a function tools()"),
    ("import nonexistent_module_xyz\n", "failed while importing: ModuleNotFoundError"),
    ("def tools(:\n", "failed while importing: SyntaxError"),
    ("def tools():\n    raise KeyError('nope')\n", "tools() failed: KeyError"),
    ("def tools():\n    return [lambda: 1]\n", "returned function, not a Tool"),
])
def test_a_broken_plugin_is_reported_by_name(tmp_path, source, message):
    write(tmp_path, "broken.py", source)
    with pytest.raises(PluginError) as caught:
        load_plugins(tmp_path)
    assert "'broken'" in str(caught.value) and message in str(caught.value)
    assert "agi_plugin_broken" not in sys.modules or "tools" in message or "returned" in message


def test_two_plugins_may_not_define_the_same_tool_and_the_directory_must_exist(tmp_path):
    write(tmp_path, "one.py", GOOD)
    write(tmp_path, "two.py", GOOD)
    with pytest.raises(PluginError, match="'one' and 'two' both define a tool named 'shout'"):
        load_plugins(tmp_path)
    with pytest.raises(PluginError, match="does not exist"):
        load_plugins(tmp_path / "missing")


def test_the_sample_plugins_in_the_repository_load_and_work():
    tools = ToolRegistry(load_plugins(REPO / "plugins"))
    assert {"describe_numbers", "prime_factors", "text_stats", "find_pattern"} <= {tool.name for tool in tools}
    assert json.loads(tools.call("prime_factors", {"n": 84}).output) == [2, 2, 3, 7]
    assert json.loads(tools.call("prime_factors", {"n": 999_999_999_989}).output) == [999_999_999_989]   # a prime
    assert not tools.call("prime_factors", {"n": 1}).ok and not tools.call("prime_factors", {"n": 10 ** 13}).ok
    summary = json.loads(tools.call("describe_numbers", {"numbers": [2, 4, 4, 4, 5, 5, 7, 9]}).output)
    assert (summary["count"], summary["mean"], summary["median"], summary["max"]) == (8, 5.0, 4.5, 9.0)
    assert not tools.call("describe_numbers", {"numbers": []}).ok
    stats = json.loads(tools.call("text_stats", {"text": "One two. Three!\nFour?"}).output)
    assert stats == {"characters": 21, "words": 4, "lines": 2, "sentences": 3}
    found = tools.call("find_pattern", {"text": "Ada 1815, Alan 1912", "pattern": r"\d{4}"})
    assert json.loads(found.output) == ["1815", "1912"]
    bad = tools.call("find_pattern", {"text": "x", "pattern": "("})
    assert not bad.ok and "not a valid regular expression" in bad.output


# ------------------------------------------------------------------ settings --- #
def test_settings_are_read_from_toml_sections_and_only_given_keys_are_returned(tmp_path):
    file = write(tmp_path, "agi.toml", '''
        [model]
        model = "local-model"
        base_url = "http://localhost:11434/v1"
        [server]
        port = 9001
        rate_limit = 5
        ui = false
        [agent]
        sandbox = "wasm"
    ''')
    assert load_settings(file) == {"model": "local-model", "base_url": "http://localhost:11434/v1", "port": 9001,
                                   "rate_limit": 5.0, "ui": False, "sandbox": "wasm"}


def test_settings_can_be_json_and_flat(tmp_path):
    file = write(tmp_path, "agi.json", json.dumps({"port": 8100, "observability": {"access_log": "-"}}))
    assert load_settings(file) == {"port": 8100, "access_log": "-"}


@pytest.mark.parametrize("name, text, message", [
    ("a.toml", "port = 'eighty'", "port"),
    ("a.toml", "prot = 8000", "prot"),                       # a misspelt key is an error, not silently ignored
    ("a.toml", "[agent]\nsandbox = 'vm'", "sandbox"),
    ("a.toml", "port = = 1", "not valid TOML"),
    ("a.json", "{not json", "not valid JSON"),
    ("a.json", "[1, 2]", "must contain a table"),
])
def test_a_wrong_settings_file_says_what_is_wrong(tmp_path, name, text, message):
    file = write(tmp_path, name, text)
    with pytest.raises(ConfigError) as caught:
        load_settings(file)
    assert message in str(caught.value) and name in str(caught.value)


def test_a_missing_settings_file_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="cannot read"):
        load_settings(tmp_path / "nope.toml")


def test_the_settings_files_in_the_repository_are_valid():
    local = load_settings(REPO / "config" / "agi.toml")
    assert local["host"] == "127.0.0.1" and local["port"] == 8000 and local["sandbox"] == "process"
    docker = load_settings(REPO / "config" / "docker.toml")
    assert docker["host"] == "0.0.0.0" and docker["token_file"] == "/run/secrets/agi_api_token"
