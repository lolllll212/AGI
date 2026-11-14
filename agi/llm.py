"""Compatibility alias: this module now lives at :mod:`agi.core.llm`."""
import sys

from .core import llm as _module

sys.modules[__name__] = _module
