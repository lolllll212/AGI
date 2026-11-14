"""Default adapters, by name.

Application code never imports an adapter. When a caller does not pass one in
(``CognitiveEngine()`` with no tracer, ``CodeRepair(...)`` with no sandbox) the
application asks this registry for the default instead:

    sandbox = sandbox or make("sandbox")

:mod:`agi.infrastructure.di` fills the registry when ``agi`` is imported. A
program can replace any entry (``register("sandbox", MySandbox)``) to change
the default everywhere at once, and a test can do the same to run the
application layer with no infrastructure at all.
"""
from __future__ import annotations

import threading
from typing import Any, Callable

from ..exceptions import AdapterMissing

_factories: dict[str, Callable[..., Any]] = {}
_lock = threading.Lock()


def register(name: str, factory: Callable[..., Any]) -> Callable[..., Any] | None:
    """Make ``factory`` the default for ``name``. Returns the factory it replaced, if any."""
    with _lock:
        previous = _factories.get(name)
        _factories[name] = factory
    return previous


def unregister(name: str) -> None:
    with _lock:
        _factories.pop(name, None)


def registered() -> list[str]:
    with _lock:
        return sorted(_factories)


def make(name: str, *args: Any, **kwargs: Any) -> Any:
    """Build the default adapter registered under ``name``."""
    with _lock:
        factory = _factories.get(name)
    if factory is None:
        raise AdapterMissing(f"no adapter is registered for {name!r}. Importing 'agi' registers the defaults; "
                             f"otherwise call agi.domain.ports.register({name!r}, factory).")
    return factory(*args, **kwargs)
