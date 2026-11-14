import os

import pytest

import agi.sandbox as sandbox_module
from agi.sandbox import Sandbox, SandboxPolicy, check_source

FAST = SandboxPolicy(wall_time_s=3.0, cpu_time_s=1, memory_mb=256)


@pytest.fixture
def sandbox():
    return Sandbox(FAST)


@pytest.fixture
def unguarded(monkeypatch):
    """A sandbox with the static gate switched off, to test the runtime layers alone."""
    monkeypatch.setattr(sandbox_module, "check_source", lambda code, policy=None: [])
    return Sandbox(FAST)


def test_run_returns_result_and_stdout(sandbox):
    outcome = sandbox.run("import math\nprint('hello')\nresult = math.sqrt(16)")
    assert outcome.ok and outcome.value == 4.0 and outcome.stdout == "hello\n"
    assert outcome.failure is None


def test_call_and_call_many_and_call_each(sandbox):
    code = "def half(a):\n    return 10 // a\ndef twice(a):\n    return a * 2\n"
    assert sandbox.call(code, "twice", [21]).value == 42
    many = sandbox.call_many(code, "half", [[2], [0], [5]])
    assert [c["ok"] for c in many.calls] == [True, False, True]
    assert many.calls[1]["error_type"] == "ZeroDivisionError"
    grid = sandbox.call_each(code, ["half", "twice"], [[1], [5]]).calls
    assert [[c["value"] for c in row] for row in grid] == [[10, 2], [2, 10]]


def test_run_tests_reports_each_test(sandbox):
    code = "def double(x):\n    return x * 2\n"
    good = sandbox.run_tests(code, "def test_a():\n    assert double(2) == 4\n")
    assert good.ok and good.tests == {"test_a": None}
    bad = sandbox.run_tests(code, "def test_a():\n    assert double(2) == 5, 'nope'\n"
                                  "def test_b():\n    assert double(0) == 0\n")
    assert not bad.ok and bad.tests["test_b"] is None
    assert bad.failure == "TestFailed: test_a: AssertionError: nope"
    assert not sandbox.run_tests(code, "x = 1").ok          # no tests is not a pass


def test_runtime_error_has_type_message_and_user_traceback(sandbox):
    outcome = sandbox.run("def boom():\n    return 1 / 0\nboom()")
    assert not outcome.ok and outcome.error_type == "ZeroDivisionError"
    assert "<code> line 2: return 1 / 0" in outcome.traceback


def test_ordinary_stdlib_code_is_not_obstructed(sandbox):
    outcome = sandbox.run(
        "import collections, dataclasses, enum, fractions, re, statistics\n"
        "@dataclasses.dataclass\nclass P:\n    x: int\n    def __post_init__(self):\n        self.x += 1\n"
        "class Base:\n    def __init__(self):\n        self.tag = 'b'\n"
        "class Child(Base):\n    def __init__(self):\n        super().__init__()\n"
        "T = collections.namedtuple('T', 'a b')\n"
        "class Colour(enum.Enum):\n    RED = 1\n"
        "try:\n    {}['k']\nexcept KeyError as exc:\n    name = type(exc).__name__\n"
        "result = [P(1).x, Child().tag, T(1, 2).b, Colour.RED.value, name,\n"
        "          str(fractions.Fraction(1, 3)), statistics.median([3, 1, 2]), bool(re.match('a+', 'aa'))]")
    assert outcome.ok, outcome.failure
    assert outcome.value == [2, "b", 2, 1, "KeyError", "1/3", 2, True]


@pytest.mark.parametrize("code", [
    "import os",
    "import subprocess",
    "from os import path",
    "import socket",
    "__import__('os')",
    "open('/etc/passwd')",
    "eval('1+1')",
    "exec('x = 1')",
    "getattr(int, 'real')",
    "().__class__.__base__.__subclasses__()",
    "import re\nre.enum.sys.modules",
    "x = __builtins__",
    "def f():\n    pass\nf.__globals__",
    "from math import *",
    "import math\nmath.__loader__",
    "try:\n    1/0\nexcept Exception as e:\n    e.__traceback__.tb_frame",
    "def (:",
])
def test_static_gate_rejects(code, sandbox):
    assert check_source(code), code
    outcome = sandbox.run(code)
    assert not outcome.ok and outcome.violations and outcome.error_type == "PolicyViolation"


def test_static_gate_rejects_oversized_source():
    assert check_source("x = 1\n" * 100, SandboxPolicy(max_source_bytes=50))


@pytest.mark.parametrize("code, expected", [
    ("import os", "ImportError"),
    ("print(open('/etc/passwd').read())", "NameError"),
    ("import re\nre.enum.sys.modules['os'].system('echo pwned > /tmp/agi_pwned')", "PermissionError"),
    ("import re\nresult = re.enum.sys.modules['os'].listdir('/')", "PermissionError"),
    ("import re\nresult = re.enum.sys.modules['os'].fork()", "PermissionError"),
    ("import re\nre.enum.sys.modules['io'].open('x.txt', 'w').write('a')", "PermissionError"),
    ("import re\nre.enum.sys.modules['io'].open('/etc/passwd').read()", "PermissionError"),
    ("import re\nimp = re.enum.sys.modules['importlib._bootstrap']\nimp._gcd_import('socket')", None),
])
def test_runtime_layers_hold_without_the_static_gate(code, expected, unguarded):
    outcome = unguarded.run(code)
    assert not outcome.ok
    if expected:
        assert outcome.error_type == expected, outcome.failure
    assert not os.path.exists("/tmp/agi_pwned")


def test_forging_the_result_channel_is_detected(unguarded):
    outcome = unguarded.run(
        "import re\nout = re.enum.sys.__stdout__\n"
        "out.write('{\"ok\": true, \"value\": 42}')\nout.flush()\nraise SystemExit")
    assert not outcome.ok and outcome.value != 42


def test_infinite_loop_is_killed(sandbox):
    outcome = sandbox.run("while True:\n    pass")
    assert not outcome.ok and outcome.timed_out and outcome.failure.startswith("Timeout")


def test_wall_clock_limit_applies_to_idle_code():
    outcome = Sandbox(SandboxPolicy(wall_time_s=0.5, cpu_time_s=5)).run(
        "import datetime\nend = datetime.datetime.now() + datetime.timedelta(seconds=30)\n"
        "while datetime.datetime.now() < end:\n    pass")
    assert outcome.timed_out and outcome.duration_s < 5


def test_memory_limit(sandbox):
    outcome = sandbox.run("x = [0] * (10 ** 9)")
    assert not outcome.ok and outcome.error_type in ("MemoryError", "ResourceLimit")


def test_output_is_capped():
    outcome = Sandbox(SandboxPolicy(max_output_bytes=100)).run("print('x' * 10 ** 6)\nresult = 1")
    assert outcome.ok and len(outcome.stdout) == 100


def test_unserialisable_values_degrade_to_repr(sandbox):
    assert "object" in sandbox.run("result = object()").value


def test_no_environment_or_working_files_leak(sandbox, monkeypatch):
    monkeypatch.setenv("AGI_SECRET", "hunter2")
    assert "hunter2" not in repr(sandbox.run("result = 1").__dict__)


def test_policy_is_immutable_and_fingerprinted():
    policy = SandboxPolicy()
    with pytest.raises(Exception):
        policy.memory_mb = 10 ** 6
    assert policy.fingerprint() == SandboxPolicy().fingerprint()
    assert policy.fingerprint() != SandboxPolicy(memory_mb=1).fingerprint()
    assert {"static-ast", "subprocess", "audit-hook"} <= set(Sandbox().isolation)


def test_custom_import_allowlist():
    strict = Sandbox(SandboxPolicy(allowed_imports=frozenset({"math"})))
    assert strict.run("import math\nresult = math.pi > 3").value is True
    assert not strict.run("import json").ok
