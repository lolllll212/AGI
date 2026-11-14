"""An example plugin in a single file: two text tools built from plain functions.

``from_function`` reads the name, the parameters and their descriptions from the signature and
the docstring, so a plugin can be this short.
"""
from __future__ import annotations

import re

from agi.domain.entities.tool import Tool, from_function


def text_stats(text: str) -> dict:
    """Count the characters, words, lines and sentences in a piece of text.

    Args:
        text: the text to measure
    """
    words = re.findall(r"[^\W_]+(?:['’-][^\W_]+)*", text)
    return {"characters": len(text), "words": len(words), "lines": len(text.splitlines()),
            "sentences": len(re.findall(r"[^.!?]+[.!?]+", text)) or (1 if text.strip() else 0)}


def find_pattern(text: str, pattern: str, ignore_case: bool = False) -> list[str]:
    """List every match of a regular expression in a piece of text, in order (at most 200).

    Args:
        text: the text to search
        pattern: a Python regular expression
        ignore_case: match upper and lower case alike
    """
    if len(pattern) > 500:
        raise ValueError("pattern is too long (limit: 500 characters)")
    try:
        compiled = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as exc:
        raise ValueError(f"not a valid regular expression: {exc}") from None
    return [match.group(0) for match, _ in zip(compiled.finditer(text[:200_000]), range(200))]


def tools() -> list[Tool]:
    return [from_function(text_stats), from_function(find_pattern)]
