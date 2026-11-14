"""What the application needs from a place that runs untrusted code."""
from __future__ import annotations

from typing import Iterable, Protocol, Sequence, runtime_checkable

from ..value_objects.sandbox import SandboxPolicy, SandboxResult


@runtime_checkable
class SandboxPort(Protocol):
    """Implemented by the process, WebAssembly and container sandboxes."""

    policy: SandboxPolicy

    def run(self, code: str) -> SandboxResult: ...

    def call(self, code: str, func: str, args: Sequence = (), kwargs: dict | None = None) -> SandboxResult: ...

    def call_many(self, code: str, func: str, arg_lists: Iterable[Sequence]) -> SandboxResult: ...

    def run_tests(self, code: str, tests: str) -> SandboxResult: ...
