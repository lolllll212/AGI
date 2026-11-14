"""Every error the toolkit raises on purpose, in one place."""
from __future__ import annotations

from .entities.tool import ToolError


class LLMError(RuntimeError):
    """The model endpoint could not be used (after retries)."""


class SandboxUnavailable(RuntimeError):
    """The requested isolation backend cannot run on this host."""


class CircuitOpen(RuntimeError):
    """The breaker is open: the protected call was not attempted."""


class BudgetExceeded(RuntimeError):
    """A step, depth or time budget ran out."""


class LoopDetected(RuntimeError):
    """The same state keeps recurring."""


class AdapterMissing(LookupError):
    """The application asked for an adapter that nothing has registered."""


__all__ = ["AdapterMissing", "BudgetExceeded", "CircuitOpen", "LLMError", "LoopDetected", "SandboxUnavailable",
           "ToolError"]
