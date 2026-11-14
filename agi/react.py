"""Compatibility alias: this module now lives at :mod:`agi.core.react`."""
import sys

from .core import react as _module

sys.modules[__name__] = _module
