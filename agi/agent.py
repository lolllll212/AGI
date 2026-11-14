"""Compatibility alias: this module now lives at :mod:`agi.core.agent`."""
import sys

from .core import agent as _module

sys.modules[__name__] = _module
