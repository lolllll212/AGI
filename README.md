# AGI

Building blocks for agents that are more capable and easier to trust, in one
Python package with no heavy dependencies.

| Part | What it does | Module |
| --- | --- | --- |
| Tools | Typed, logged tools for files, tabular data, HTTP, arithmetic, sandboxed Python and memory | `agi.tools` |
| ReAct loop | Reason, act with a tool, read the real result, repeat; challenges unsupported answers | `agi.react` |
| Memory | Turns past failures into lessons in a knowledge graph and recalls them for new tasks | `agi.memory` |
| Reasoning | MCTS and Tree-of-Thoughts over hypotheses that are executed and verified | `agi.reasoning` |
| Training | A search policy trained by gradient descent; fine-tuning data exported from agent runs | `agi.training` |
| Evaluation | Eight real-world tasks checked against independently computed answers | `agi.evals` |
| World models | Learns from the consequences of actions; causal discovery by intervention | `agi.world` |
| Self-improvement | Agents write tools for each other under consensus, a sandbox and fixed safety rules | `agi.collective` |

This is not an AGI and it does not contain a language model. It is the machinery
around one: tools, a control loop, memory, search, verification, training and
governance. Everything runs offline with deterministic stand-ins; a real model
plugs in through one seam (`agi.llm`).

Where the field stands on skill acquisition, breadth × depth, long-horizon
autonomy and grounded memory, what this package already covers, and the six
proposals that follow from the gaps — with a reproduction of the memory-revision
defect — is in [`docs/research/agi-agenda.md`](docs/research/agi-agenda.md), and a Rust
implementation of the brain-grounded autonomy design is in [`agi-rs/`](agi-rs/).

## Quick start

Requires Python 3.10+ on a POSIX system; developed and tested on Python 3.11 on
Linux only. The only runtime dependency is
[`conflux-state`](https://github.com/lolllll212/Conflux).

```bash
pip install -e ".[test]"
python -m agi                 # all seven demos, about 35 seconds, no model needed
python -m agi react training  # or pick: memory reasoning world collective agent react training
pytest                        # 337 tests, about a minute
```

With a model, from any OpenAI-compatible endpoint (OpenAI, Ollama, vLLM, LM
Studio, OpenRouter and others). The API key is read from the environment:

```bash
export OPENAI_API_KEY=...
python -m agi run "Which region sold the most in sales.csv?" --model gpt-4o-mini --workspace ./data
python -m agi chat --model llama3.1 --base-url http://localhost:11434/v1 --workspace ./data --memory agent.db
python -m agi eval --model gpt-4o-mini       # score the model on the eight benchmark tasks
python -m agi train --out policy.json        # train the search policy
```

## Tools and the ReAct loop

`ReActAgent` drives a model through reason, act, observe. The model proposes one
tool call; the runtime executes it and reports what really happened; only then
does the model continue.

```python
import json, tempfile
from agi.evals import write_fixtures
from agi.llm import ScriptedLLM
from agi.memory import MemoryEngine
from agi.react import ReActAgent
from agi.tools import standard_tools

workspace = write_fixtures(tempfile.mkdtemp())      # sample files: sales.csv, app.log, ...
tools = standard_tools(workspace)                   # calculator, python, files, query_data, ...
sql = "select sum(amount) from data where region = 'North'"
model = ScriptedLLM([                               # a scripted stand-in for a real model
    f'Thought: query it\nAction: query_data\nAction Input: {json.dumps({"file": "sale.csv", "sql": sql})}',
    f'Thought: wrong name\nAction: query_data\nAction Input: {json.dumps({"file": "sales.csv", "sql": sql})}',
    "Thought: I have it\nFinal Answer: North sold about 3,100.",
    "Thought: use the observed figure\nFinal Answer: North sold 3,090.75.",
])
agent = ReActAgent(model, tools, memory=MemoryEngine())
run = agent.run("What is the total sales amount for the North region in sales.csv?")

print(run)                        # answered: 'North sold 3,090.75.'  [2 tool calls, 1 errors, 4 model calls, grounded]
print(run.steps[0].observation)   # FileNotFound: no file 'sale.csv' in the workspace. Files present: ...
print(run.steps[2].observation)   # [verifier] The figure(s) 3100 in your answer do not appear in ... any observation
```

Replace `ScriptedLLM([...])` with `OpenAICompatLLM("model-name")` to use a real model.

What the loop enforces:

- **No imagined results.** Anything a model writes after its action, such as an
  invented `Observation:` or a premature answer, is discarded.
- **Errors are information.** A failed call comes back as an observation with a
  specific message (a missing file lists the files that exist; a bad SQL query
  shows the table's columns).
- **Answers must be grounded.** A final answer containing a number that appears
  in no tool result is challenged once. `run.grounded` reports the outcome.
- **Bounded.** Step limit, repeated-call detection, a cap on unusable replies,
  and shortening of old observations when the context fills up.
- **Memory in the loop.** Relevant lessons are placed in the prompt; every run is
  recorded as an episode; a failed call followed by a working one becomes a
  lesson for next time.

Built-in tools (`standard_tools(workspace, memory=..., network=..., forge=...)`):

| Tool | Notes |
| --- | --- |
| `calculator` | Exact integer and rational arithmetic; no `eval` |
| `python` | Runs in the sandbox: no files, no network, limited imports |
| `list_files`, `read_file`, `search_files`, `write_file` | Confined to the workspace directory, symlinks included; `read_only=True` removes writing |
| `query_data` | SQL (SELECT only) over CSV and JSON files; parses `"1,200.50"` and `$40` as numbers; joins several files |
| `http_get` | Off unless `network=True`; refuses loopback, private and link-local addresses; optional host allowlist |
| `current_time` | So the model does not assume the date |
| `remember`, `recall` | Long-term facts, when a `MemoryEngine` is given |
| forged tools | Tools approved by the agent collective (see below), still sandboxed |

Adding your own tool takes a function with type hints and a docstring:

```python
from agi.tools import ToolRegistry

registry = ToolRegistry()

@registry.tool
def word_count(text: str) -> int:
    """Count the words in a piece of text.

    text: the text to count
    """
    return len(text.split())

assert registry.call("word_count", {"text": "to be or not to be"}).value == 6
assert registry.call("word_count", {}).output.startswith("BadArguments: missing required argument 'text'")
```

## Training

Two kinds, both in `agi.training`.

**A trained search policy.** `train_policy` generates random programs, runs them
to get exact input/output examples, and fits a softmax policy by stochastic
gradient descent to predict the next operation. Program search then uses the
policy to decide where to look first. Every candidate is still executed and
verified, so a poor policy costs time, not correctness.

```python
from agi.training import train_policy, make_tasks, compare

policy, report = train_policy(tasks=4000, seed=0)      # about 2 seconds, pure Python
print(report)
# 4000 dreamed tasks, 10000 samples; loss 1.685 -> 1.284;
# held-out next-step accuracy 11% -> 58% (top-3 80%)

unseen = make_tasks(30, seed=1003, lengths=(4,), minimal=True)    # 4-step tasks it never saw
print(compare(policy, unseen, budget=1500))
# solved 18/30 -> 30/30 within 1500 verifications; mean programs executed 905.5 -> 214.1
```

`LearningAgent(memory, policy=policy)` uses the policy and takes further gradient
steps on each task it solves. `policy.save(path)` and `SearchPolicy.load(path)`
persist it.

**Fine-tuning data for your model.** `export_jsonl(runs, path)` writes ReAct runs
as chat-format JSONL for OpenAI-style fine-tuning. By default only runs that
succeeded with a grounded answer are exported. This package does not fine-tune a
language model itself.

## Evaluation

`agi.evals` holds eight tasks over sample files: compound interest, a group-by,
a two-file join, log analysis, date arithmetic, read-then-compute, writing a
report file, and one question the data cannot answer (the correct response is to
say so). Each is checked against a value computed independently in plain Python.

```python
from agi.evals import run_suite, reference_llm
from agi.react import ReActAgent
from agi.tools import standard_tools

report = run_suite(lambda workspace, task: ReActAgent(reference_llm(task), standard_tools(workspace)))
assert report.passed == 8          # the reference solutions: proves the suite is solvable with the tools
```

Pass your own model in place of `reference_llm(task)`, or run
`python -m agi eval --model MODEL`. A stand-in that answers from imagination
scores 0 of 8, with all eight answers flagged as not grounded.

## Memory

`MemoryEngine` extracts lessons from episodes, stores them in a knowledge graph
(SQLite) and retrieves the relevant ones for a new task. No weights change, so
learning something new cannot overwrite something old.

```python
from agi.memory import MemoryEngine, Episode, Step, SUCCESS

memory = MemoryEngine()            # or MemoryEngine("agent.db") to persist across runs
memory.record(Episode(
    "Parse the customer CSV export and compute the average order value", SUCCESS, tags=("csv",),
    steps=[Step("float(row['amount'])", ok=False,
                error="ValueError: could not convert string to float: '1,200.50'"),
           Step("float(row['amount'].replace(',', ''))", ok=True)]))

print(memory.context("Load a CSV of payments and sum the amount column", tags=("csv",)))
# Lessons from past experience (most relevant first):
# 1. [fix] When: Parse the customer CSV export ... (ValueError). Then: 'float(row['amount'])'
#    failed with ValueError: ...; 'float(row['amount'].replace(',', ''))' worked instead. ...

assert memory.context("write a haiku about autumn") == ""    # unrelated tasks recall nothing
```

- **Extraction.** A failed step followed by one that worked becomes a `fix`
  lesson; an unrecovered failure becomes `avoid`; a verified solution becomes a
  `procedure`. `LLMExtractor` lets a model write the lessons instead.
- **Compression.** The same error in related tasks merges into one lesson with a
  support count. `consolidate()` merges duplicates, drops old traces and retires
  lessons that keep failing to help.
- **Retrieval.** IDF-weighted text similarity plus spreading activation over the
  graph. Retrieval is word-based by default; pass
  `MemoryEngine(embedder=OpenAICompatEmbedder("model"))` to use a real embedding model.

## Reasoning that checks itself

A `Problem` supplies hypotheses and a `verify` function. `MCTS` and
`TreeOfThoughts` explore, run each hypothesis through the verifier, and abandon a
branch as soon as it is refuted. A result is reported as solved only when
verification passed.

```python
from agi.reasoning import MCTS, PipelineSynthesis, check_argument
from agi.sandbox import Sandbox

examples = [([3, 1, 2], 6), ([5, 4, 9, 1], 18), ([-2, -7], -4)]
problem = PipelineSynthesis(examples, Sandbox())      # candidates run in the sandbox
result = MCTS(problem, seed=0, max_verifications=1500).search(6000)
print(result.summary())       # SOLVED after 77 verifications, 12 backtracks, 78 nodes explored
print(problem.source(result.best_state))
# def solve(x):
#     x = max(x)
#     x = x * 2
#     return x

proof = check_argument(["rain -> wet", "wet"], ["rain"])      # affirming the consequent
assert not proof.valid and proof.counterexample == {"rain": False, "wet": True}
```

Included: `PipelineSynthesis` (program synthesis from examples), `Game24`,
`CodeRepair` (a model writes code, the sandbox runs the tests, the real failure
goes into the next prompt) and a propositional logic verifier that returns a
counterexample for every invalid step.

## Learning from consequences

```python
from agi.world import ProjectileWorld, WorldModel, GroundedAgent, ideal_range

env = ProjectileWorld(drag=0.01, wind=-2.0)       # the real world has drag and wind
model = WorldModel(env.bounds, prior=lambda a: ideal_range(a["angle"], a["speed"]))  # the textbook does not
agent = GroundedAgent(env, model, observe="distance", seed=0)

for trial in agent.practice(target=30.0, trials=6):
    print(f"predicted {trial.predicted:5.1f} m, landed {trial.observed:5.1f} m, missed by {trial.miss:4.2f} m")
# predicted  30.0 m, landed  24.5 m, missed by 5.49 m
# ...
# predicted  30.0 m, landed  29.7 m, missed by 0.34 m
```

`agi.world` also has `CodeWorld` (the sandbox as an environment), `FunctionWorld`
(any callables or API clients behind an allowlist and call budget), and
`CausalLearner`, which intervenes on variables to tell causes from correlations.

## Self-improvement with hard limits

Agents write Python tools for each other. A tool enters the shared registry only
after five gates, and agents cannot change the rules of those gates.

```python
from agi.collective import Collective, Builder, SpecReviewer, SafetyReviewer, ToolNeed

need = ToolNeed("mean", "arithmetic mean of a list", ((([1, 2, 3],), 2.0), (([2, 3],), 2.5)))
group = Collective([
    Builder("ada", {"mean": {"code": "def mean(xs):\n    return sum(xs) / (len(xs) - 1)\n"}}),  # wrong
    Builder("bob", {"mean": {"code": "def mean(xs):\n    return sum(xs) / len(xs)\n"}}),
    SpecReviewer("carol"), SafetyReviewer("sam")])

for verdict in group.request(need):
    print(verdict)
# REJECTED mean by ada at gate 'author-tests': ... mean([1, 2, 3]) should be 2.0 but is 3.0
# ACCEPTED mean v1 by bob (approved by ada, carol, sam)

assert group.call("mean", [1, 2, 3, 4]).value == 2.5     # runs sandboxed, from any agent's replica
assert group.ledger.converged() and group.forge.audit.verify()
```

The gates, in order: **policy** (static safety check, naming, dependency and
capacity rules), **author-tests**, **regression** (a new version must pass every
test the old one was approved with), **safety-veto** (one safety concern rejects
the tool whatever the vote), and **consensus** (two thirds of independent
reviewers; the forge re-runs each reviewer's tests, and an approval whose tests
fail counts as a rejection).

Shared state lives in `conflux-state` CRDT replicas, one per agent. Every write
is a signed action restricted to the writer's namespace, so replicas converge in
any delivery order and an agent cannot write the registry, forge another agent's
review, or edit a message after signing it. The full list of rules is
`agi.collective.HARD_BOUNDARIES`. Approved tools can be handed to the ReAct loop
with `standard_tools(forge=group.forge)`.

## The sandbox, and what it does not do

`agi.sandbox.Sandbox` runs untrusted code in layers: a static AST gate, a fresh
interpreter process with an empty environment, kernel limits on CPU, memory,
file size and process count, a network namespace where the host allows it, and a
runtime audit hook. `Sandbox().isolation` reports which layers are active.

The static gate and audit hook run inside CPython and are not a security
boundary on their own. The process, kernel-limit and namespace layers are the
real boundary, and they are as strong as the host kernel. For code you consider
hostile, wrap the interpreter in a container or VM through
`SandboxPolicy(launcher=[...])`.

## Known limits

- **No model is bundled, and none was available while building this.** The ReAct
  loop, the CLI and the model clients are tested against scripted replies and a
  local stub server, never against a live model. Expect to tune the prompt for
  the model you use.
- **The grounding check is a heuristic.** It looks at numbers only. It catches a
  figure that no tool returned; it does not catch a wrong name, a misread
  result, or a number copied from the wrong row.
- **The trained policy is small and specific.** It is a linear softmax model
  over hand-written features, for one DSL of 22 list operations. The held-out
  tasks come from the same random generator as the training tasks. Rank and
  position features were added after the hand-written "second largest" task
  failed, so that task is not an independent test.
- **Memory credit is coarse.** A recalled lesson is marked as having helped
  whenever the run succeeded, whether or not the model used it.
- **Tool timeouts abandon, they do not kill.** A tool that overruns is reported
  as timed out but its thread keeps running until it returns. The `python` tool
  is different: the sandbox does kill it.
- **`http_get` resolves the host twice** (once to check, once to fetch), so a
  DNS server that changes its answer in between could get past the address
  check. Use the host allowlist where that matters.
- **The collective runs in one process.** Replication, signatures and namespace
  checks are real, but there is no network transport and the forge is a single
  trusted kernel.
- **Platforms.** Windows is not supported. macOS is untested and would run
  without the network-namespace layer.

## Layout

```
agi/
  tools/          tool registry and built-in tools
  react.py        the ReAct loop, answer verifiers, Conversation
  memory/         episodic memory and knowledge graph
  reasoning/      MCTS, Tree-of-Thoughts, logic verifier, problems
  training/       search policy training, fine-tuning data export
  evals.py        real-world benchmark tasks
  world/          environments, world model, causal discovery
  collective/     ledger, safety policy, tool forge, agents
  sandbox.py      layered sandbox
  llm.py          model and embedding clients (the only model-facing module)
  agent.py        LearningAgent: memory, verified search, trained policy
  __main__.py     demos and the run/chat/eval/train commands
tests/            one file per module
```
