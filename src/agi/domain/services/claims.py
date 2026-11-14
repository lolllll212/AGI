"""Find the names an answer relies on, and check them against what was actually seen.

A *name* here is something an answer cannot have worked out, only looked up or
been told: a file name, an identifier, a quoted value, a proper noun. Numbers
are handled separately (:func:`agi.domain.services.text.numbers_in`).

The extraction is deliberately narrow. A missed name costs nothing new; a word
wrongly taken for a name makes the agent defend a perfectly good answer.
"""
from __future__ import annotations

import re

_FILE = re.compile(r"(?<![\w/.-])[\w./-]*\w\.(?:csv|tsv|json|jsonl|txt|md|py|log|ya?ml|toml|ini|db|sqlite|html?|xml|"
                   r"xlsx?|pdf|png|jpe?g|sh|sql|parquet)\b", re.I)
_IDENTIFIER = re.compile(r"(?<![\w@./-])(?:[A-Za-z]+[_-])+[A-Za-z]*\d\w*\b|(?<![\w@./-])[A-Za-z]{2,}\d{2,}\w*\b|"
                         r"(?<![\w@./-])[a-z]+(?:_[a-z0-9]+)+\b")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_QUOTED = re.compile(r"\"([^\"\n]{2,60})\"|(?<![\w])'([^'\n]{2,60})'(?![\w])|`([^`\n]{2,60})`|“([^”\n]{2,60})”")
_SENTENCE_START = re.compile(r"(?:^\s*(?:[-*•]|\d+[.)])?\s*|[.!?]\s+|\n\s*(?:[-*•]|\d+[.)])?\s*|[(\[\"“]\s*)$")
_PROPER = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)*(?:[ -][A-Z][a-z]+(?:[A-Z][a-z]+)*)*\b")
_WORDS = re.compile(r"[a-z0-9]+")

# Capitalised words that are ordinary vocabulary, not something looked up.
_ORDINARY = frozenset("""
monday tuesday wednesday thursday friday saturday sunday january february march april may june july august
september october november december today tomorrow yesterday
python java javascript typescript rust sql sqlite json csv markdown excel linux windows unicode ascii regex
english hindi french german spanish chinese
final answer note summary total result results yes no none true false ok okay sorry thanks
north south east west northern southern eastern western
the this that these those there here and but for with from into over all any each some most both either
however therefore because since while when where which what who how why also then thus hence overall
based according given using after before during between among per via not only table figure step steps
first second third next last finally sum mean median average count minimum maximum
""".split())


def names_in(text: str) -> list[str]:
    """The checkable names in ``text``, in order of appearance, without duplicates."""
    found: list[tuple[int, str]] = []
    taken: list[tuple[int, int]] = []

    def claim(start: int, end: int, value: str) -> None:
        if any(start < e and end > s for s, e in taken):
            return
        taken.append((start, end))
        found.append((start, value))

    for pattern in (_EMAIL, _FILE, _IDENTIFIER):
        for match in pattern.finditer(text):
            claim(match.start(), match.end(), match.group())
    for match in _QUOTED.finditer(text):
        value = next(group for group in match.groups() if group is not None).strip()
        if re.search(r"[A-Za-z]", value) and len(value.split()) <= 6:
            claim(match.start(), match.end(), value)
    for match in _PROPER.finditer(text):
        words = match.group().replace("-", " ").split()
        start = match.start()
        if _SENTENCE_START.search(text[:start]):            # a capital that only marks the start of a sentence
            start += len(words[0]) + 1
            words = words[1:]
        while words and words[0].lower() in _ORDINARY:      # "The North region" -> nothing; "In Pune" -> Pune
            start += len(words[0]) + 1
            words = words[1:]
        while words and words[-1].lower() in _ORDINARY:
            words = words[:-1]
        if words and not all(word.lower() in _ORDINARY for word in words):
            claim(start, start + len(" ".join(words)), " ".join(words))

    names: list[str] = []
    for _, value in sorted(found):
        if value.lower() not in (name.lower() for name in names):
            names.append(value)
    return names


class Evidence:
    """Text an answer may draw on: the task, tool results, the calls that succeeded, earlier turns."""

    def __init__(self, *texts: str):
        self.text = "\n".join(text for text in texts if text).lower()
        self.words = set(_WORDS.findall(self.text))

    def supports(self, name: str) -> bool:
        lowered = name.lower()
        if lowered in self.text:
            return True
        if re.search(r"[\d_@./\\]", lowered):              # a file name or an identifier must match exactly
            return False
        parts = _WORDS.findall(re.sub(r"['’]s\b", "", lowered))
        if not parts:
            return True

        def known(word: str) -> bool:                       # allow a plural or possessive ending
            return (word in self.words or word.rstrip("s") in self.words or word + "s" in self.words
                    or (word.endswith("es") and word[:-2] in self.words))
        return all(known(part) for part in parts)


def unsupported_names(answer: str, evidence: Evidence, limit: int = 6) -> list[str]:
    """Names in ``answer`` that appear nowhere in ``evidence``."""
    return [name for name in names_in(answer) if not evidence.supports(name)][:limit]
