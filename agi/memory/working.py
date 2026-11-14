"""Working memory: a token-budgeted context window with compaction.

What a model sees is a list of messages that must fit a budget. This module
keeps it there with three mechanisms, applied in this order when over budget:

1. **Shrinking** - long tool outputs in older messages are cut down to their head and tail.
2. **Compaction** - the oldest messages outside the recent window are replaced
   by a summary (written by a model if one is given, extractive otherwise).
3. **Sliding window** - if it still does not fit, the oldest unpinned messages are dropped.

Pinned messages (the system prompt, the task) are never touched.

On "KV-cache management": a hosted model's key-value cache cannot be controlled
from the client. What a client *can* do is keep the start of the prompt
byte-identical from call to call, which is what makes provider-side prompt
caching hit. So compaction happens in rare, large steps (down to ``low_water``)
instead of a little on every turn, and :attr:`WorkingMemory.cache_hit_ratio`
reports how much of each prompt repeated the previous one.
"""
from __future__ import annotations

import re
from typing import Callable, Literal

from pydantic import Field

from ..core.types import Model

Role = Literal["system", "user", "assistant", "tool", "summary"]


def estimate_tokens(text: str) -> int:
    """A deterministic approximation of a BPE token count (no tokenizer dependency).

    Roughly one token per short word or punctuation mark, and one per four
    characters of a long word. Within about 15% of real tokenizers on English
    prose and code; pass a real counter to :class:`WorkingMemory` when it matters.
    """
    count = 0
    for piece in re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", text):
        count += 1 if len(piece) <= 4 else (len(piece) + 3) // 4
    return count


class Message(Model):
    role: Role
    content: str
    pinned: bool = False
    tokens: int = 0
    meta: dict = Field(default_factory=dict)


class ContextStats(Model):
    tokens: int
    budget: int
    messages: int
    compactions: int
    dropped: int
    shrunk: int
    summarised_messages: int
    cache_hit_ratio: float


def extractive_summary(messages: list[Message], max_tokens: int = 200) -> str:
    """Summarise without a model: keep what was asked, what was done, and what came back."""
    lines: list[str] = []
    for message in messages:
        text = " ".join(message.content.split())
        if not text:
            continue
        if message.role == "summary":
            lines.append(text)
        elif message.role == "user":
            lines.append(f"User asked: {text[:160]}")
        elif message.role == "tool":
            name = message.meta.get("tool", "tool")
            status = "" if message.meta.get("ok", True) else " (failed)"
            lines.append(f"{name}{status} returned: {text[:120]}")
        elif message.role == "assistant":
            lines.append(f"Agent: {text[:140]}")
    kept: list[str] = []
    used = 0
    # Keep the first line (the original ask) and then as many of the latest as fit.
    for line in lines[:1] + lines[1:][::-1]:
        cost = estimate_tokens(line)
        if used + cost > max_tokens and kept:
            break
        kept.append(line)
        used += cost
    return "\n".join(kept[:1] + kept[1:][::-1])


class WorkingMemory:
    def __init__(self, budget: int = 8000, keep_recent: int = 6, low_water: float = 0.6,
                 summarizer: Callable[[str], str] | None = None,
                 count: Callable[[str], int] = estimate_tokens, summary_tokens: int = 300,
                 shrink_over: int = 400):
        """``summarizer(prompt) -> str`` is any model callable; without one, summaries are extractive."""
        if not 0 < low_water <= 1:
            raise ValueError("low_water must be in (0, 1]")
        self.budget, self.keep_recent, self.low_water = budget, keep_recent, low_water
        self.summarizer, self.count = summarizer, count
        self.summary_tokens, self.shrink_over = summary_tokens, shrink_over
        self.messages: list[Message] = []
        self.compactions = self.dropped = self.shrunk = self.summarised = 0
        self._last_render: str | None = None
        self._cache_hits = self._cache_total = 0

    # -- adding ------------------------------------------------------------ #
    def add(self, role: Role, content: str, pinned: bool = False, **meta) -> Message:
        message = Message(role=role, content=content, pinned=pinned, tokens=self.count(content), meta=meta)
        self.messages.append(message)
        if self.tokens > self.budget:
            self.compact()
        return message

    @property
    def tokens(self) -> int:
        return sum(m.tokens for m in self.messages)

    # -- compaction -------------------------------------------------------- #
    def _shrink(self, message: Message) -> None:
        words = message.content
        keep = max(80, self.shrink_over)                   # characters kept at each end
        if message.tokens <= self.shrink_over or len(words) <= 2 * keep + 40:
            return
        cut = len(words) - 2 * keep
        message.content = f"{words[:keep]}\n[... {cut} characters omitted ...]\n{words[-keep:]}"
        message.tokens = self.count(message.content)
        self.shrunk += 1

    def _summarise(self, messages: list[Message]) -> str:
        if self.summarizer is not None:
            transcript = "\n".join(f"{m.role.upper()}: {m.content[:1500]}" for m in messages)
            prompt = ("Summarise this part of an agent's work so that it can continue without the "
                      "original. Keep: the goal, decisions made, facts and figures found (exactly), "
                      f"errors hit and what fixed them, and anything still to do. At most "
                      f"{self.summary_tokens} tokens.\n\n{transcript}\n\nSummary:")
            try:
                text = self.summarizer(prompt).strip()
                if text:
                    return text
            except Exception:                              # a failed summary must not lose the run
                pass
        return extractive_summary(messages, self.summary_tokens)

    def compact(self, target: int | None = None) -> int:
        """Bring the context down to ``target`` tokens (default ``low_water * budget``).

        Returns the number of tokens freed.
        """
        target = int(self.budget * self.low_water) if target is None else target
        before = self.tokens
        if before <= target:
            return 0
        movable = [i for i, m in enumerate(self.messages) if not m.pinned]
        old = movable[: max(0, len(movable) - self.keep_recent)]

        for index in old:                                  # 1. shrink bulky old messages
            if self.tokens <= target:
                break
            self._shrink(self.messages[index])

        if self.tokens > target and len(old) >= 2:         # 2. summarise the old stretch
            chunk = [self.messages[i] for i in old]
            summary = self._summarise(chunk)
            note = Message(role="summary", content=f"[Summary of {len(chunk)} earlier messages]\n{summary}",
                           tokens=0, meta={"replaces": len(chunk)})
            note.tokens = self.count(note.content)
            if note.tokens < sum(m.tokens for m in chunk):
                first = old[0]
                gone = set(old)
                self.messages = [m for i, m in enumerate(self.messages) if i not in gone]
                self.messages.insert(first, note)
                self.summarised += len(chunk)
                self.compactions += 1

        while self.tokens > self.budget:                   # 3. sliding window: drop the oldest
            victims = [i for i, m in enumerate(self.messages) if not m.pinned]
            if len(victims) <= 1:
                break
            del self.messages[victims[0]]
            self.dropped += 1
        if self.tokens > self.budget:                      # one message bigger than the whole budget
            for message in self.messages:
                if not message.pinned:
                    self._shrink(message)
        return before - self.tokens

    # -- rendering --------------------------------------------------------- #
    def render(self) -> str:
        text = "\n\n".join(f"{m.role.upper()}: {m.content}" for m in self.messages)
        if self._last_render is not None:
            shared = 0
            for a, b in zip(self._last_render, text):
                if a != b:
                    break
                shared += 1
            self._cache_hits += shared
            self._cache_total += len(text)
        self._last_render = text
        return text

    def as_chat(self) -> list[dict]:
        """Messages in the OpenAI chat format (summaries become system notes)."""
        role = {"summary": "system", "tool": "user"}
        return [{"role": role.get(m.role, m.role), "content": m.content} for m in self.messages]

    @property
    def cache_hit_ratio(self) -> float:
        """Share of rendered prompt characters that repeated the previous prompt's prefix."""
        return self._cache_hits / self._cache_total if self._cache_total else 0.0

    def stats(self) -> ContextStats:
        return ContextStats(tokens=self.tokens, budget=self.budget, messages=len(self.messages),
                            compactions=self.compactions, dropped=self.dropped, shrunk=self.shrunk,
                            summarised_messages=self.summarised, cache_hit_ratio=round(self.cache_hit_ratio, 4))

    def clear(self, keep_pinned: bool = True) -> None:
        self.messages = [m for m in self.messages if m.pinned] if keep_pinned else []
        self._last_render = None
