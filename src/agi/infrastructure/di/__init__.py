"""The composition root: the one place that decides which adapter fills which port."""
from .defaults import DEFAULTS, wire_defaults

__all__ = ["DEFAULTS", "build_engine", "wire_defaults"]


def __getattr__(name: str):
    if name == "build_engine":                              # imported on demand: it pulls in every adapter
        from .container import build_engine
        return build_engine
    raise AttributeError(name)
