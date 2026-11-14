"""Compatibility alias: this module now lives at :mod:`agi.evaluation.suite`."""
import sys

from .evaluation import suite as _module

sys.modules[__name__] = _module
