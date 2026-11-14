"""ReAct: reason, act with a tool, read what actually happened, repeat.

The loop is built around one rule: **the model never gets to imagine an
observation.** It proposes one action; the runtime executes it and reports the
real result (including real errors); only then may the model continue. The
safeguards around that rule are what make it usable in practice:

* replies are parsed leniently, but anything written after an action - an
  invented ``Observation:`` or a premature ``Final Answer:`` - is discarded;
* tool failures come back as observations, so the model can correct course;
* repeating an identical call is detected and called out;
* a final answer whose numbers appear in no observation is *challenged* once,
  and the run reports whether the answer ended up grounded;
* lessons from earlier runs are recalled into the prompt, and every run is
  written back to memory as an episode.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from .llm import LLM, extract_json
from .tools import ToolRegistry

SYSTEM = """You are a careful problem-solving agent. You solve the task step by step, using tools
to get facts and to compute, instead of guessing.

TOOLS
{tools}

FORMAT
To use a tool, reply with exactly:

Thought: <your reasoning about what to do next>
Action: <one tool name>
Action Input: <a JSON object with the tool's arguments>

Then stop. You will receive an "Observation:" with the real result.
When you have enough information, reply with:

Thought: <why the task is now solved>
Final Answer: <the answer, stated completely>

RULES
- One action per reply. Never write "Observation:" yourself.
- Every number or fact in your final answer must come from the task or from an observation.
  Do arithmetic with a tool, not in your head.
- If a tool reports an error, read it and try something different; do not repeat the same call.
- If the task cannot be completed, say so in the Final Answer and explain what is missing."""

_ACTION = re.compile(r"^[ \t>*#-]*\**Action\**[ \t]*:[ \t]*(.+?)\s*$", re.M | re.I)
_INPUT = re.compile(r"^[ \t>*#-]*\**Action[ _]Input\**[ \t]*:[ \t]*", re.M | re.I)
_FINAL = re.compile(r"^[ \t>*#-]*\**Final[ _]Answer\**[ \t]*:[ \t]*", re.M | re.I)
_OBSERVATION = re.compile(r"^[ \t>*#-]*\**Observation\**[ \t]*:", re.M | re.I)
_THOUGHT = re.compile(r"^[ \t>*#-]*\**Thought\**[ \t]*:[ \t]*", re.M | re.I)


@dataclass
class Parsed:
    thought: str = ""
    action: str | None = None
    args: Any = None
    answer: str | None = None
    problem: str | None = None           # why the reply could not be used


def _thought(text: str) -> str:
    return _THOUGHT.sub("", text, count=1).strip()


def parse_reply(text: str, tools: ToolRegistry | None = None) -> Parsed:
    """Turn one model reply into an action or a final answer."""
    text = text.strip()
    cut = _OBSERVATION.search(text)
    if cut:                                             # the model started inventing results
        text = text[: cut.start()].rstrip()
    action, final = _ACTION.search(text), _FINAL.search(text)
    # `Action Input:` also matches the Action pattern's prefix; skip to a real one.
    while action and re.match(r"(?i)[ \t>*#-]*\**Action[ _]Input", action.group(0)):
        action = _ACTION.search(text, action.end())

    if action and (not final or action.start() < final.start()):
        name = re.sub(r"\(.*$", "", action.group(1)).strip(" `*'\".")
        rest = text[action.end():]
        marker = _INPUT.search(rest)
        raw = rest[marker.end():] if marker else ""
        stop = _FINAL.search(raw) or _THOUGHT.search(raw) or _ACTION.search(raw)
        if stop:                                        # anything after the input is premature
            raw = raw[: stop.start()]
        raw = raw.strip()
        parsed = Parsed(_thought(text[: action.start()]), name)
        tool = tools.get(name) if tools is not None else None
        if not raw or raw.lower() in ("none", "{}", "n/a"):
            parsed.args = {}
            return parsed
        try:
            value = extract_json(raw)
        except ValueError:
            value = re.sub(r"^```\w*\n?|\n?```$", "", raw).strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
        if not isinstance(value, dict):                 # bare value: fine if it is unambiguous
            required = [n for n, p in tool.params.items() if p.required] if tool else []
            names = required if len(required) == 1 else (list(tool.params) if tool else [])
            if len(names) == 1:
                value = {names[0]: value}
            else:
                parsed.problem = ("Action Input must be a JSON object of arguments"
                                  + (f"; the signature is {tool.signature()}" if tool else ""))
        parsed.args = value
        return parsed

    if final:
        answer = text[final.end():].strip()
        if not answer:
            return Parsed(_thought(text[: final.start()]), problem="the Final Answer is empty")
        return Parsed(_thought(text[: final.start()]), answer=answer)

    try:                                                # a model that prefers pure JSON
        value = extract_json(text)
        if isinstance(value, dict):
            lowered = {str(k).lower().replace(" ", "_"): v for k, v in value.items()}
            if lowered.get("final_answer") not in (None, ""):
                return Parsed(str(lowered.get("thought", "")), answer=str(lowered["final_answer"]))
            if isinstance(lowered.get("action"), str):
                args = lowered.get("action_input", lowered.get("input", lowered.get("arguments", {})))
                return Parsed(str(lowered.get("thought", "")), lowered["action"],
                              args if isinstance(args, dict) else {})
    except ValueError:
        pass
    return Parsed(_thought(text), problem="the reply contains neither an Action nor a Final Answer")


# --------------------------------------------------------------------------- #
# answer verification
# --------------------------------------------------------------------------- #
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


class GroundingVerifier:
    """Challenges a final answer that contains numbers no observation supports.

    A number counts as supported if it appears in the task or in any
    observation, allowing for rounding and for a ratio written as a percentage.
    Small whole numbers are ignored (counts, list positions, "2 files").
    This is a cheap check for the commonest hallucination - a figure that was
    never computed or looked up - not a proof that the answer is right.
    """

    def __init__(self, ignore_below: int = 13):
        self.ignore_below = ignore_below

    def unsupported(self, task: str, answer: str, steps: Sequence["ReActStep"]) -> list[str]:
        evidence = [v for v, _ in numbers_in(task)]
        for step in steps:
            if step.ok and step.action:
                evidence += [v for v, _ in numbers_in(step.observation)]
        missing = []
        for value, decimals in numbers_in(answer):
            if decimals == 0 and abs(value) < self.ignore_below:
                continue
            tolerance = 0.5 * 10 ** -decimals + 1e-9
            if any(abs(e - value) <= tolerance or abs(e * 100 - value) <= tolerance
                   for e in evidence):
                continue
            shown = f"{value:.{decimals}f}"
            if shown not in missing:
                missing.append(shown)
        return missing

    def __call__(self, task: str, answer: str, steps: Sequence["ReActStep"]) -> str | None:
        missing = self.unsupported(task, answer, steps)
        if not missing:
            return None
        return (f"The figure(s) {', '.join(missing)} in your answer do not appear in the task or in "
                f"any observation. Compute or look them up with a tool, then answer again. If they "
                f"are already justified, restate the answer and say where each figure comes from.")


class LLMVerifier:
    """A second opinion from a model: does the transcript support the answer?"""

    PROMPT = """You are checking another agent's work. Decide whether the ANSWER is fully supported
by the OBSERVATIONS (real tool results). Reply with JSON only:
{{"supported": true|false, "problem": "<what is unsupported or wrong, if anything>"}}

TASK: {task}

OBSERVATIONS:
{observations}

ANSWER: {answer}
"""

    def __init__(self, llm: LLM):
        self.llm = llm

    def __call__(self, task: str, answer: str, steps: Sequence["ReActStep"]) -> str | None:
        observations = "\n".join(f"[{s.action}] {s.observation[:800]}" for s in steps if s.action) or "(none)"
        try:
            reply = extract_json(self.llm(self.PROMPT.format(task=task, observations=observations,
                                                             answer=answer)))
            if reply.get("supported") is False:
                return f"A reviewer objects: {reply.get('problem') or 'the answer is not supported'}"
        except Exception:                               # advisory: an unusable review is no objection
            return None
        return None


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #
@dataclass
class ReActStep:
    thought: str
    action: str | None                  # None for a format error or a verifier challenge
    args: dict
    observation: str
    ok: bool
    reply: str = ""                     # the raw model output for this step
    seconds: float = 0.0


@dataclass
class Run:
    task: str
    answer: str | None = None
    steps: list[ReActStep] = field(default_factory=list)
    stop_reason: str = ""               # "answer" | "max_steps" | "format" | "loop" | "llm_error"
    grounded: bool | None = None        # None when no verifier ran
    success: bool | None = None         # set when a check was supplied, else a best guess
    final_reply: str = ""
    challenges: int = 0
    llm_calls: int = 0
    recalled: list = field(default_factory=list)
    learned: list = field(default_factory=list)
    seconds: float = 0.0
    error: str | None = None

    @property
    def finished(self) -> bool:
        return self.stop_reason == "answer"

    @property
    def tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.action)

    @property
    def tool_errors(self) -> int:
        return sum(1 for s in self.steps if s.action and not s.ok)

    def transcript(self, width: int = 600) -> str:
        lines = [f"Task: {self.task}"]
        for step in self.steps:
            if step.thought:
                lines.append(f"Thought: {step.thought}")
            if step.action:
                lines.append(f"Action: {step.action}")
                lines.append(f"Action Input: {json.dumps(step.args, ensure_ascii=False)}")
            text = step.observation if len(step.observation) <= width else step.observation[:width] + " [...]"
            lines.append(f"Observation: {text}")
        if self.answer is not None:
            lines.append(f"Final Answer: {self.answer}")
        return "\n".join(lines)

    def __str__(self) -> str:
        status = {True: "grounded", False: "NOT grounded", None: "unverified"}[self.grounded]
        return (f"{'answered' if self.finished else 'stopped (' + self.stop_reason + ')'}: "
                f"{self.answer!r}  [{self.tool_calls} tool calls, {self.tool_errors} errors, "
                f"{self.llm_calls} model calls, {status}]")


_FAILURES = {"answer": "WrongAnswer", "max_steps": "MaxSteps", "format": "FormatError",
             "loop": "RepeatedCall", "llm_error": "ModelError"}


class ReActAgent:
    """Runs the reason-act-observe loop for one task at a time.

    ``llm`` is any ``callable(prompt) -> str``. ``memory`` (a
    :class:`~agi.memory.MemoryEngine`) is optional: with it, relevant lessons are
    put in front of the model and each run is recorded.
    """

    def __init__(self, llm: LLM, tools: ToolRegistry, memory=None, max_steps: int = 12,
                 verifier: Callable | None | str = "grounding", max_challenges: int = 1,
                 max_context_chars: int = 24_000, tags: Iterable[str] = (),
                 system: str | None = None, on_step: Callable[[ReActStep], None] | None = None):
        self.llm, self.tools, self.memory = llm, tools, memory
        self.max_steps, self.max_challenges = max_steps, max_challenges
        self.verifier = GroundingVerifier() if verifier == "grounding" else verifier
        self.max_context_chars = max_context_chars
        self.tags = tuple(tags)
        self.system = system or SYSTEM
        self.on_step = on_step

    # -- prompt ------------------------------------------------------------ #
    def system_prompt(self) -> str:
        return self.system.format(tools=self.tools.describe()) if "{tools}" in self.system else self.system

    def _scratchpad(self, steps: Sequence[ReActStep], budget: int) -> str:
        blocks = []
        for step in steps:
            head = ""
            if step.thought:
                head += f"Thought: {step.thought}\n"
            if step.action:
                head += f"Action: {step.action}\nAction Input: {json.dumps(step.args, ensure_ascii=False)}\n"
            blocks.append([head, step.observation])
        # Working memory is finite: shorten the oldest observations first, keep the last three whole.
        index = 0
        while sum(len(h) + len(o) + 14 for h, o in blocks) > budget and index < len(blocks) - 3:
            if len(blocks[index][1]) > 240:
                blocks[index][1] = blocks[index][1][:200] + " [...older output shortened]"
            index += 1
        return "".join(f"{h}Observation: {o}\n" for h, o in blocks)

    def prompt(self, task: str, steps: Sequence[ReActStep], lessons: str = "", context: str = "",
               closing: str = "") -> str:
        parts = [self.system_prompt()]
        if lessons:
            parts.append(lessons)
        if context:
            parts.append(f"EARLIER IN THIS CONVERSATION\n{context}")
        head = "\n\n".join(parts) + f"\n\nTask: {task}\n"
        pad = self._scratchpad(steps, max(2000, self.max_context_chars - len(head)))
        return head + pad + closing + "Thought:"

    # -- run --------------------------------------------------------------- #
    def run(self, task: str, check: Callable[[str], bool] | None = None, context: str = "") -> Run:
        """Solve ``task``. ``check(answer)``, if given, decides success for memory and reports."""
        started = time.perf_counter()
        run = Run(task)
        lessons = ""
        if self.memory is not None:
            hits = self.memory.recall(task, self.tags, k=4)
            run.recalled = [hit.lesson for hit in hits]
            if hits:
                lessons = ("LESSONS FROM EARLIER RUNS (most relevant first)\n"
                           + "\n".join(f"{i}. {hit.lesson.render()}" for i, hit in enumerate(hits, 1)))

        seen: dict[str, str] = {}
        bad_format = repeats = 0
        if self.verifier is not None:
            run.grounded = False

        def note(step: ReActStep) -> None:
            run.steps.append(step)
            if self.on_step:
                self.on_step(step)

        while run.llm_calls < self.max_steps:
            try:
                reply = self.llm(self.prompt(task, run.steps, lessons, context))
            except Exception as exc:                    # the model endpoint failed
                run.stop_reason, run.error = "llm_error", f"{type(exc).__name__}: {exc}"
                break
            run.llm_calls += 1
            began = time.perf_counter()
            parsed = parse_reply(reply, self.tools)

            if parsed.problem:
                bad_format += 1
                note(ReActStep(parsed.thought, None, {}, f"FormatError: {parsed.problem}. Reply with "
                               f"either Thought/Action/Action Input or Thought/Final Answer.", False, reply))
                if bad_format >= 3:
                    run.stop_reason, run.error = "format", "three unusable replies in a row"
                    break
                continue
            bad_format = 0

            if parsed.answer is not None:
                objection = None
                if self.verifier is not None:
                    objection = self.verifier(task, parsed.answer, run.steps)
                    run.grounded = objection is None
                if objection and run.challenges < self.max_challenges:
                    run.challenges += 1
                    note(ReActStep(parsed.thought, None, {}, f"[verifier] {objection}", False, reply))
                    continue
                run.answer, run.final_reply, run.stop_reason = parsed.answer, reply, "answer"
                break

            key = f"{parsed.action}:{json.dumps(parsed.args, sort_keys=True, default=str)}"
            if key in seen:
                repeats += 1
                note(ReActStep(parsed.thought, parsed.action, parsed.args,
                               f"RepeatedCall: you already made this exact call; its result was: "
                               f"{seen[key][:300]}\nDo something different, or give the Final Answer.",
                               False, reply))
                if repeats >= 3:
                    run.stop_reason, run.error = "loop", "the same call was repeated"
                    break
                continue
            result = self.tools.call(parsed.action, parsed.args)
            seen[key] = result.output
            note(ReActStep(parsed.thought, parsed.action, result.args or parsed.args, result.output,
                           result.ok, reply, time.perf_counter() - began))
        else:
            run.stop_reason = "max_steps"

        if run.answer is None and run.stop_reason in ("max_steps", "loop"):
            closing = ("You have no steps left. Do not call a tool. Give your Final Answer now from "
                       "the observations above, and say plainly what you could not establish.\n")
            try:
                reply = self.llm(self.prompt(task, run.steps, lessons, context, closing))
                run.llm_calls += 1
                parsed = parse_reply(reply, self.tools)
                if parsed.answer is not None:
                    run.answer, run.final_reply = parsed.answer, reply
                    if self.verifier is not None:
                        run.grounded = self.verifier(task, parsed.answer, run.steps) is None
            except Exception as exc:
                run.error = f"{type(exc).__name__}: {exc}"

        if check is not None:
            try:
                run.success = bool(run.answer is not None and check(run.answer))
            except Exception:
                run.success = False
        else:
            run.success = run.finished and run.grounded is not False
        run.seconds = time.perf_counter() - started
        if self.memory is not None:
            self._remember(run)
        return run

    def _remember(self, run: Run) -> None:
        from .memory import FAILURE, SUCCESS, Episode, Step
        steps = []
        for step in run.steps:
            if step.action is None:
                continue
            call = f"{step.action}({json.dumps(step.args, ensure_ascii=False, default=str)[:160]})"
            steps.append(Step(call, step.ok, None if not step.ok else step.observation[:160],
                              step.observation[:300] if not step.ok else None))
        used = [s.action for s in steps if s.ok]
        episode = Episode(
            task=run.task, outcome=SUCCESS if run.success else FAILURE, steps=steps, tags=self.tags,
            solution=used if run.success and used else None,
            error=None if run.success else f"{_FAILURES.get(run.stop_reason, 'RunFailed')}: "
                                           f"{run.error or 'the answer was wrong or unsupported'}",
            meta={"answer": run.answer, "stop_reason": run.stop_reason, "grounded": run.grounded})
        run.learned = self.memory.record(episode)
        for lesson in run.recalled:                     # crude credit: was it present when we succeeded?
            self.memory.feedback(lesson.id, helped=bool(run.success))


class Conversation:
    """Several tasks in a row, each seeing a short summary of the ones before."""

    def __init__(self, agent: ReActAgent, keep: int = 6):
        self.agent, self.keep = agent, keep
        self.turns: list[tuple[str, str]] = []

    def ask(self, message: str, check: Callable[[str], bool] | None = None) -> Run:
        context = "\n".join(f"User: {q}\nYou answered: {a}" for q, a in self.turns[-self.keep:])
        run = self.agent.run(message, check, context)
        self.turns.append((message, (run.answer or f"(no answer: {run.stop_reason})")[:1500]))
        return run
