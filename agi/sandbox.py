"""Compatibility alias: this module now lives at :mod:`agi.core.sandbox`."""
import sys

from .core import sandbox as _module

sys.modules[__name__] = _module
