"""Sandbox backends: Docker (plumbing only here - no daemon) and WebAssembly (real, when installed)."""
import os
import stat
import sys
import time

import pytest

from agi.core.isolation import (DockerSandbox, SandboxUnavailable, WasmSandbox, available_backends,
                                best_sandbox)
from agi.core.sandbox import Sandbox, SandboxPolicy

needs_wasm = pytest.mark.skipif(not WasmSandbox.available(), reason="wasi-python is not installed")
needs_docker = pytest.mark.skipif(not DockerSandbox.available(), reason="no Docker daemon on this host")


# ------------------------------------------------------------------ docker --- #
def test_docker_command_locks_the_container_down():
    box = DockerSandbox(SandboxPolicy(memory_mb=128, cpu_time_s=3), image="python:3.12-slim", check=False)
    argv = box.command()
    joined = " ".join(argv[: argv.index("python:3.12-slim") + 1])
    for flag in ("run --rm -i", "--network none", "--read-only", "--cap-drop ALL",
                 "--security-opt no-new-privileges", "--user 65534:65534", "--memory 128m",
                 "--memory-swap 128m", "--pids-limit 64", "--ulimit cpu=3", "--cpus 1"):
        assert flag in joined, flag
    after_image = argv[argv.index("python:3.12-slim") + 1:]
    assert after_image[:5] == ["python3", "-I", "-S", "-B", "-c"]     # the container's own interpreter
    assert "unshare" not in " ".join(argv)                           # the container is the network boundary
    assert "container" in box.isolation and "no-network" in box.isolation
    podman = DockerSandbox(binary="podman", extra_args=["--runtime", "runsc"], check=False).command()
    assert podman[0] == "podman" and "runsc" in podman


def test_docker_is_refused_when_there_is_no_daemon():
    if DockerSandbox.available():
        pytest.skip("Docker is available here")
    with pytest.raises(SandboxUnavailable):
        DockerSandbox()
    with pytest.raises(SandboxUnavailable):
        best_sandbox(require="docker")                               # no silent downgrade when required
    missing = DockerSandbox(binary="definitely-not-installed", check=False).run("result = 1")
    assert not missing.ok and missing.error_type == "SandboxCrash" and "could not start" in missing.error


def test_jobs_travel_through_a_container_cli(tmp_path, monkeypatch):
    """A stand-in ``docker`` that drops the container flags and runs the interpreter locally.

    This proves the launcher plumbing (argv, stdin job, result parsing), not containment.
    """
    shim = tmp_path / "docker"
    shim.write_text(f"""#!{sys.executable}
import os, sys
args = sys.argv[1:]
if args[:1] == ["info"]:
    sys.exit(0)
open({str(tmp_path / 'argv.txt')!r}, "w").write("\\n".join(args[: args.index("fake-image")]))
rest = args[args.index("fake-image") + 1:]
if rest[0] == "python3":
    os.execv(sys.executable, [sys.executable] + rest[1:])
os.execvp(rest[0], rest)
""")
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    DockerSandbox.available.cache_clear()
    try:
        box = DockerSandbox(image="fake-image")
        assert box.run("import math\nresult = math.factorial(5)").value == 120
        assert box.call("def f(a, b):\n    return a - b", "f", [9, 4]).value == 5
        assert box.run("x = 1 / 0").error_type == "ZeroDivisionError"
        assert "--network\nnone" in (tmp_path / "argv.txt").read_text()
        shell = box.run_shell("echo hello; exit 3")
        assert not shell.ok and shell.stdout == "hello\n" and shell.value == 3 and shell.error_type == "ShellError"
        assert box.run_shell("echo fine").ok
        assert isinstance(best_sandbox(), DockerSandbox)             # the strongest available wins
    finally:
        DockerSandbox.available.cache_clear()


@needs_docker
def test_real_container_has_no_network_and_a_read_only_root():        # pragma: no cover - needs a daemon
    box = DockerSandbox(SandboxPolicy(wall_time_s=60))
    assert box.run("result = 6 * 7").value == 42
    shell = box.run_shell("touch /etc/x 2>&1; wget -T 2 -q -O- http://example.com 2>&1; echo done")
    assert "Read-only" in shell.stdout and "done" in shell.stdout


# -------------------------------------------------------------------- wasm --- #
@pytest.fixture(scope="module")
def wasm():
    return WasmSandbox(SandboxPolicy(wall_time_s=5, memory_mb=256))


@needs_wasm
def test_wasm_runs_the_same_job_protocol_as_the_process_sandbox(wasm):
    process = Sandbox()
    code = "import math\nprint('hi')\nresult = math.factorial(10)"
    for box in (wasm, process):
        outcome = box.run(code)
        assert outcome.ok and outcome.value == 3628800 and outcome.stdout.strip() == "hi"
        assert box.call("def f(a, b):\n    return a * b", "f", [6, 7]).value == 42
        each = box.call_each("def a(x):\n    return x + 1\ndef b(x):\n    return x * 2", ["a", "b"], [[1], [2]])
        assert [[c["value"] for c in row] for row in each.calls] == [[2, 3], [2, 4]]
        tests = box.run_tests("def inc(x):\n    return x + 1",
                              "def test_a():\n    assert inc(1) == 2\ndef test_b():\n    assert inc(1) == 3")
        assert not tests.ok and tests.tests["test_a"] is None and tests.tests["test_b"].startswith("AssertionError")
        failed = box.run("x = 1 / 0")
        assert failed.error_type == "ZeroDivisionError" and "line 1" in failed.traceback
    assert "wasm-memory" in wasm.isolation and "no-network" in wasm.isolation


@needs_wasm
def test_wasm_enforces_time_and_memory_limits():
    tight = WasmSandbox(SandboxPolicy(wall_time_s=1.5, memory_mb=128))
    spin = tight.run("while True:\n    pass")
    assert spin.timed_out and spin.error_type == "Timeout" and tight.trapped == "deadline"
    hog = tight.run("x = bytearray(600 * 1024 * 1024)")
    assert not hog.ok and hog.error_type in ("MemoryError", "ResourceLimit")
    assert tight.run("result = 1").ok                                 # the sandbox is reusable afterwards


@needs_wasm
def test_wasm_fuel_stops_a_runaway_loop_long_before_the_clock_does():
    box = WasmSandbox(SandboxPolicy(wall_time_s=30, memory_mb=256), fuel=2_000_000_000)
    assert "fuel" in box.isolation
    box.warm_up()                                  # compiles the metered build once per machine (about a minute)
    assert box.warm_up() < 5                       # ... and is nearly free afterwards
    started = time.perf_counter()
    spin = box.run("while True:\n    pass")
    assert spin.timed_out and spin.error_type == "Timeout" and box.trapped == "out of fuel"
    assert time.perf_counter() - started < 10      # the wall-clock limit is 30 s: fuel cut it off first
    fine = box.run("result = sum(i * i for i in range(20000))")
    assert fine.ok and fine.value == 2666466670000 and box.trapped is None


def test_wasm_refuses_fuel_too_small_to_start_the_interpreter():
    with pytest.raises(ValueError, match="starting the interpreter uses about 36,000,000"):
        WasmSandbox(fuel=1_000_000, check=False)
    assert WasmSandbox(fuel=WasmSandbox.MIN_FUEL, check=False).fuel == 100_000_000
    assert "fuel" not in WasmSandbox(check=False).isolation


@needs_wasm
def test_wasm_guest_cannot_reach_host_files_network_or_processes(wasm):
    # Bypass the static gate on purpose: the point is that the *runtime* boundary holds without it.
    from agi.core import sandbox as module
    probe = '''
import json, sys
out = {}
for name, attempt in {
    "read_passwd": lambda: open("/etc/passwd").read()[:20],
    "list_root": lambda: __import__("os").listdir("/home"),
    "socket": lambda: __import__("socket").create_connection(("93.184.216.34", 80), timeout=2),
    "subprocess": lambda: __import__("subprocess").run(["id"], capture_output=True).stdout,
    "write": lambda: open("/tmp/agi-escape-test", "w").write("x"),
}.items():
    try:
        out[name] = "REACHED: " + repr(attempt())[:40]
    except BaseException as exc:
        out[name] = type(exc).__name__
sys.stdout.write(json.dumps({"ok": True, "value": out, "stdout": ""}))
'''
    from wasi_python import run
    done = run(["-c", probe], deadline_s=20)
    import json
    result = json.loads(done.stdout)["value"]
    assert not any(str(v).startswith("REACHED") for v in result.values()), result
    assert not os.path.exists("/tmp/agi-escape-test")
    assert module.check_source("import socket")                       # and the static gate would have refused it


def test_best_sandbox_picks_what_works_here():
    found = available_backends()
    assert found["process"] is True and set(found) == {"docker", "wasm", "process"}
    box = best_sandbox(prefer=("process",))
    assert type(box) is Sandbox and box.run("result = 2 + 2").value == 4
    assert type(best_sandbox(prefer=("docker", "process"))) in (DockerSandbox, Sandbox)
    with pytest.raises(ValueError, match="unknown sandbox backend"):
        best_sandbox(prefer=("firecracker",))
    if not WasmSandbox.available():
        with pytest.raises(SandboxUnavailable):
            best_sandbox(require="wasm")
