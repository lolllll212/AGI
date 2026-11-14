"""Asynchronous multi-agent work: a message bus, agents as actors, and fan-out/fan-in.

* :class:`MessageBus` - agents talk by message, never by calling each other:
  direct mail, publish/subscribe topics, and request/reply with a timeout.
  Every message carries a hop count, so two agents cannot ping-pong forever.
* :class:`Actor` - an agent with a mailbox, processed one message at a time.
* :class:`Orchestrator` - split a task into independent parts, run them in
  parallel (bounded concurrency, per-part timeout, failures isolated), then
  combine the results.
"""
from __future__ import annotations

import asyncio
import itertools
import time
from typing import Any, Awaitable, Callable, Sequence

from pydantic import Field

from .llm import extract_json
from .resilience import LoopDetected
from .types import EngineResult, Model

_ids = itertools.count(1)


class Message(Model):
    id: int = Field(default_factory=lambda: next(_ids))
    sender: str
    recipient: str | None = None         # direct mail ...
    topic: str | None = None             # ... or a broadcast
    kind: str = "message"
    content: Any = None
    reply_to: int | None = None
    hops: int = 0                        # how many messages led to this one
    sent: float = Field(default_factory=time.time)


class BusStats(Model):
    sent: int = 0
    delivered: int = 0
    dropped: int = 0                     # no such recipient / no subscriber
    hop_limited: int = 0


class MessageBus:
    def __init__(self, max_hops: int = 8, history: int = 1000):
        self.max_hops = max_hops
        self._mailboxes: dict[str, asyncio.Queue] = {}
        self._topics: dict[str, set[str]] = {}
        self._waiting: dict[int, asyncio.Future] = {}
        self.stats = BusStats()
        self.log: list[Message] = []
        self._history = history

    def register(self, name: str) -> asyncio.Queue:
        if name in self._mailboxes:
            raise ValueError(f"an agent named {name!r} is already on the bus")
        self._mailboxes[name] = asyncio.Queue()
        return self._mailboxes[name]

    def unregister(self, name: str) -> None:
        self._mailboxes.pop(name, None)
        for members in self._topics.values():
            members.discard(name)

    def subscribe(self, name: str, topic: str) -> None:
        if name not in self._mailboxes:
            raise KeyError(f"{name!r} is not on the bus")
        self._topics.setdefault(topic, set()).add(name)

    @property
    def agents(self) -> list[str]:
        return list(self._mailboxes)

    async def send(self, message: Message) -> int:
        """Deliver a message. Returns how many mailboxes received it."""
        self.stats.sent += 1
        self.log.append(message)
        del self.log[: max(0, len(self.log) - self._history)]
        if message.hops > self.max_hops:
            self.stats.hop_limited += 1
            raise LoopDetected(f"message chain exceeded {self.max_hops} hops "
                               f"({message.sender} -> {message.recipient or message.topic})")
        if message.reply_to is not None and message.reply_to in self._waiting:
            future = self._waiting.pop(message.reply_to)
            if not future.done():
                future.set_result(message)
            self.stats.delivered += 1
            return 1
        if message.recipient is not None:
            targets = [message.recipient] if message.recipient in self._mailboxes else []
        else:
            targets = sorted(self._topics.get(message.topic or "", set()) - {message.sender})
        for target in targets:
            await self._mailboxes[target].put(message)
        self.stats.delivered += len(targets)
        self.stats.dropped += 0 if targets else 1
        return len(targets)

    async def request(self, sender: str, recipient: str, content: Any, kind: str = "request",
                      timeout: float = 30.0, hops: int = 0) -> Message:
        """Send a message and wait for the reply to it."""
        if recipient not in self._mailboxes:
            raise KeyError(f"no agent named {recipient!r} on the bus")
        message = Message(sender=sender, recipient=recipient, kind=kind, content=content, hops=hops)
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._waiting[message.id] = future
        try:
            await self.send(message)
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"{recipient!r} did not reply to {sender!r} within {timeout:g}s") from None
        finally:
            self._waiting.pop(message.id, None)


Handler = Callable[[Message, "Actor"], Awaitable[Any]]


class Actor:
    """An agent on the bus. ``handler(message, actor)`` may return a value, which is sent back as the reply."""

    def __init__(self, name: str, bus: MessageBus, handler: Handler, topics: Sequence[str] = ()):
        self.name, self.bus, self.handler = name, bus, handler
        self.mailbox = bus.register(name)
        for topic in topics:
            bus.subscribe(name, topic)
        self.handled = 0
        self.errors: list[str] = []
        self._task: asyncio.Task | None = None

    async def send(self, recipient: str, content: Any, kind: str = "message", cause: Message | None = None) -> int:
        return await self.bus.send(Message(sender=self.name, recipient=recipient, kind=kind, content=content,
                                           hops=cause.hops + 1 if cause else 0))

    async def publish(self, topic: str, content: Any, kind: str = "event", cause: Message | None = None) -> int:
        return await self.bus.send(Message(sender=self.name, topic=topic, kind=kind, content=content,
                                           hops=cause.hops + 1 if cause else 0))

    async def ask(self, recipient: str, content: Any, timeout: float = 30.0, cause: Message | None = None) -> Any:
        reply = await self.bus.request(self.name, recipient, content, timeout=timeout,
                                       hops=cause.hops + 1 if cause else 0)
        if reply.kind == "error":
            raise RuntimeError(f"{recipient} failed: {reply.content}")
        return reply.content

    async def _loop(self) -> None:
        while True:
            message = await self.mailbox.get()
            try:
                result = await self.handler(message, self)
                kind = "reply"
            except asyncio.CancelledError:
                raise
            except Exception as exc:                        # one bad message must not kill the agent
                self.errors.append(f"{type(exc).__name__}: {exc}")
                result, kind = f"{type(exc).__name__}: {exc}", "error"
            self.handled += 1
            if message.kind == "request" and (result is not None or kind == "error"):
                try:
                    await self.bus.send(Message(sender=self.name, recipient=message.sender, kind=kind,
                                                content=result, reply_to=message.id, hops=message.hops + 1))
                except LoopDetected as exc:
                    self.errors.append(str(exc))

    def start(self) -> "Actor":
        self._task = asyncio.get_running_loop().create_task(self._loop(), name=f"actor-{self.name}")
        return self

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.bus.unregister(self.name)


# --------------------------------------------------------------------------- #
# fan-out / fan-in
# --------------------------------------------------------------------------- #
class SubResult(Model):
    task: str
    ok: bool
    answer: str | None = None
    error: str | None = None
    seconds: float = 0.0


class OrchestrationResult(Model):
    task: str
    answer: str | None
    success: bool
    parts: list[SubResult] = Field(default_factory=list)
    decomposed: bool = False
    seconds: float = 0.0


DECOMPOSE = """Split this task into independent sub-tasks that can be worked on at the same time.
Each sub-task must be complete in itself (name the files and values it needs). If the task cannot be
split, return it as the single sub-task. At most {limit} sub-tasks.

TASK: {task}

Reply with JSON only: {{"subtasks": ["...", "..."]}}"""

SYNTHESISE = """Combine the results of these sub-tasks into one answer to the original task.
Use only what the results say. If a sub-task failed, say what is missing instead of guessing.

TASK: {task}

RESULTS
{results}

Answer:"""


class Orchestrator:
    """Split, run in parallel, combine."""

    def __init__(self, worker: Callable[[str], Awaitable[Any]], llm: Callable[[str], str] | None = None,
                 concurrency: int = 4, timeout: float | None = 120.0, max_subtasks: int = 6):
        """``worker(task)`` is awaited once per sub-task (e.g. ``engine.arun``)."""
        self.worker, self.llm = worker, llm
        self.concurrency, self.timeout, self.max_subtasks = concurrency, timeout, max_subtasks

    async def map(self, tasks: Sequence[str]) -> list[SubResult]:
        """Run every task through the worker, at most ``concurrency`` at a time, in the order given."""
        gate = asyncio.Semaphore(max(1, self.concurrency))

        async def one(task: str) -> SubResult:
            async with gate:
                started = time.perf_counter()
                try:
                    value = await asyncio.wait_for(self.worker(task), self.timeout)
                except asyncio.TimeoutError:
                    return SubResult(task=task, ok=False, error=f"timed out after {self.timeout:g}s",
                                     seconds=time.perf_counter() - started)
                except Exception as exc:
                    return SubResult(task=task, ok=False, error=f"{type(exc).__name__}: {exc}",
                                     seconds=time.perf_counter() - started)
                seconds = time.perf_counter() - started
                if isinstance(value, EngineResult):
                    return SubResult(task=task, ok=bool(value.success), answer=value.answer,
                                     error=None if value.success else (value.error or value.stop_reason),
                                     seconds=seconds)
                return SubResult(task=task, ok=True, answer=None if value is None else str(value), seconds=seconds)

        return list(await asyncio.gather(*(one(task) for task in tasks)))

    async def decompose(self, task: str) -> list[str]:
        if self.llm is None:
            return [task]
        try:
            reply = await asyncio.to_thread(self.llm, DECOMPOSE.format(task=task, limit=self.max_subtasks))
            data = extract_json(reply)
            items = data.get("subtasks", []) if isinstance(data, dict) else data
            parts = [str(item).strip() for item in items if str(item).strip()] if isinstance(items, list) else []
        except Exception:
            parts = []
        return list(dict.fromkeys(parts))[: self.max_subtasks] or [task]

    async def solve(self, task: str, subtasks: Sequence[str] | None = None) -> OrchestrationResult:
        started = time.perf_counter()
        parts = list(subtasks) if subtasks else await self.decompose(task)
        results = await self.map(parts)
        if len(results) == 1:
            answer, success = results[0].answer, results[0].ok
        elif self.llm is None:
            answer = "\n".join(f"- {r.task}: {r.answer if r.ok else 'FAILED (' + str(r.error) + ')'}" for r in results)
            success = all(r.ok for r in results)
        else:
            listing = "\n".join(f"[{i}] {r.task}\n    " + (f"result: {r.answer}" if r.ok else f"FAILED: {r.error}")
                                for i, r in enumerate(results, 1))
            try:
                answer = (await asyncio.to_thread(self.llm, SYNTHESISE.format(task=task, results=listing))).strip()
                success = all(r.ok for r in results)
            except Exception as exc:
                answer, success = f"The sub-tasks ran but could not be combined ({type(exc).__name__}: {exc}).\n{listing}", False
        return OrchestrationResult(task=task, answer=answer, success=success, parts=results,
                                   decomposed=len(parts) > 1, seconds=time.perf_counter() - started)
