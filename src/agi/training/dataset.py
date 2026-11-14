"""Turn agent runs into fine-tuning data.

A ReAct run that ended in a verified answer is a worked example of good tool
use. :func:`export_jsonl` writes such runs in the chat-format JSONL accepted by
OpenAI-style fine-tuning APIs (and by most open-source trainers), so a model
can be trained on the agent's own successful behaviour.
"""
from __future__ import annotations

import json
from typing import Iterable

from agi.application.cortex.react import SYSTEM, Run


def to_chat_example(run: Run, system: str = "", include_errors: bool = True) -> dict:
    """One training example: the task, then alternating model turns and observations.

    Steps where the model's reply was malformed, repeated or challenged by the
    verifier are dropped - they are not behaviour worth imitating. Tool errors
    followed by a recovery are kept by default: recovering is worth learning.
    """
    messages = [{"role": "system", "content": system or SYSTEM.split("\n\nTOOLS")[0]},
                {"role": "user", "content": f"Task: {run.task}"}]
    for step in run.steps:
        if step.action is None or step.observation.startswith("RepeatedCall"):
            continue
        if not step.ok and not include_errors:
            continue
        messages.append({"role": "assistant", "content": (
            f"Thought: {step.thought}\nAction: {step.action}\n"
            f"Action Input: {json.dumps(step.args, ensure_ascii=False)}")})
        messages.append({"role": "user", "content": f"Observation: {step.observation}"})
    thought = ""
    if run.final_reply:
        head = run.final_reply.split("Final Answer", 1)[0].strip()
        thought = head[len("Thought:"):].strip() if head.startswith("Thought:") else head
    messages.append({"role": "assistant", "content": (
        (f"Thought: {thought}\n" if thought else "") + f"Final Answer: {run.answer}")})
    return {"messages": messages}


def export_jsonl(runs: Iterable[Run], path: str, system: str = "", only_successful: bool = True,
                 require_grounded: bool = True) -> int:
    """Write runs as chat fine-tuning JSONL; returns how many examples were written.

    By default only runs that succeeded *and* whose answer was grounded in
    observations are exported: training on unverified answers would teach the
    model to hallucinate with confidence.
    """
    written = 0
    with open(path, "w", encoding="utf-8") as handle:
        for run in runs:
            if run.answer is None:
                continue
            if only_successful and not run.success:
                continue
            if require_grounded and run.grounded is False:
                continue
            handle.write(json.dumps(to_chat_example(run, system), ensure_ascii=False) + "\n")
            written += 1
    return written
