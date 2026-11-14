"""Choose the strongest isolation backend this host can run."""
from __future__ import annotations

from typing import Sequence

from agi.domain.exceptions import SandboxUnavailable
from agi.domain.services.source_check import check_source
from agi.domain.value_objects.sandbox import SandboxPolicy
from agi.infrastructure.adapters.sandbox.docker import DockerSandbox
from agi.infrastructure.adapters.sandbox.process import Sandbox
from agi.infrastructure.adapters.sandbox.wasm import WasmSandbox


def available_backends() -> dict[str, bool]:
    return {"docker": DockerSandbox.available(), "wasm": WasmSandbox.available(), "process": True}


def best_sandbox(policy: SandboxPolicy | None = None, prefer: Sequence[str] = ("docker", "wasm", "process"),
                 require: str | None = None) -> Sandbox:
    """The strongest sandbox that works here.

    ``require="docker"`` (or ``"wasm"``) refuses to fall back: use it when
    running hostile code without a hard boundary would be worse than not running it.
    """
    makers = {"docker": lambda: DockerSandbox(policy), "wasm": lambda: WasmSandbox(policy),
              "process": lambda: Sandbox(policy)}
    order = [require] if require else list(prefer)
    for name in order:
        if name not in makers:
            raise ValueError(f"unknown sandbox backend {name!r} (use 'docker', 'wasm' or 'process')")
        try:
            return makers[name]()
        except SandboxUnavailable:
            if require:
                raise
    return Sandbox(policy)


__all__ = ["DockerSandbox", "WasmSandbox", "SandboxUnavailable", "best_sandbox", "available_backends",
           "check_source"]
