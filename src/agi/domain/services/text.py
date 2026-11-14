"""Pure text helpers: pulling JSON, code and numbers out of free-form model output."""
from __future__ import annotations

import json
import re
from typing import Any


def extract_json(text: str) -> Any:
    """Pull the first JSON object or array out of free-form model output.

    Raises ``ValueError`` if none is found. Tolerates code fences and chatter.
    """
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char in "[{":
            try:
                value, _ = decoder.raw_decode(text[start:])
                return value
            except ValueError:
                continue
    raise ValueError("no JSON value found in model output")


def extract_code(text: str) -> str:
    """Return the first fenced code block of ``text`` (or ``text`` itself)."""
    if "```" not in text:
        return text.strip()
    block = text.split("```", 2)[1]
    first, _, rest = block.partition("\n")
    return (rest if first.strip().isalpha() or not first.strip() else block).strip()


_NUM = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w]|\.\d)")


def numbers_in(text: str) -> list[tuple[float, int]]:
    """``(value, decimals)`` for every number in ``text`` (thousands separators understood)."""
    found = []
    for match in _NUM.finditer(text):
        token = match.group().replace(",", "")
        try:
            found.append((float(token), len(token.partition(".")[2])))
        except ValueError:
            continue
    return found


def render_answer(template: str, output: str, expected_figures: int | None = None) -> str | None:
    """Fill an answer template from a fresh tool output; ``None`` if the output has a different shape."""
    output = output.strip()
    figures = [value for value, _ in numbers_in(output)]
    if expected_figures is not None and len(figures) != expected_figures:
        return None

    def fill(match: re.Match) -> str:
        index, spec = int(match.group(1)), match.group(2)
        if index >= len(figures):
            raise IndexError(index)
        return format(figures[index], spec)

    try:
        text = re.sub(r"\{#(\d+):([^}]*)\}", fill, template)
    except (IndexError, ValueError):
        return None
    return text.replace("{result}", output).replace("{{", "{").replace("}}", "}")
