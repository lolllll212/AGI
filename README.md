# AGI

Four building blocks for agents that are more capable and easier to trust, in one
small Python package. Each one targets a known weakness of today's models:

| Pillar | Weakness it targets | Module |
| --- | --- | --- |
| 1. Continual learning and episodic memory | Forgetting; repeating the same mistakes | `agi.memory` |
| 2. System 2 reasoning and self-verification | Hallucinated answers nobody checked | `agi.reasoning` |
| 3. Grounded world models and causal reasoning | Beliefs that never meet reality | `agi.world` |
| 4. Safe self-improvement and multi-agent consensus | Agents changing themselves unchecked | `agi.collective` |

This is not an AGI, and nothing here trains or contains a model. It is the
machinery around a model: memory, search, verification, grounding and
governance. Everything runs offline and deterministically; a language model plugs
in through one seam (`agi.llm`).

## Quick start

Requires Python 3.10+ on a POSIX system; developed and tested on Python 3.11 on
Linux only. The only runtime dependency is
[`conflux-state`](https://github.com/lolllll212/Conflux).

```bash
pip install -e ".[test]"
python -m agi          # runs all five demos, about 30 seconds
python -m agi agent    # or one of: memory, reasoning, world, collective, agent
pytest                 # 180 tests, under a minute
```

## 1. Memory that learns from failure: `agi.memory`

`MemoryEngine` turns an agent's past episodes into lessons, stores them in a
knowledge graph (SQLite), and retrieves the relevant ones for a new task. No
weights are retrained, so learning something new cannot overwrite something old.

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

- **Extraction.** A failed step followed by a step that worked becomes a `fix`
  lesson; an unrecovered failure becomes an `avoid` lesson; a verified solution
  becomes a `procedure`. `LLMExtractor` lets a model write the lessons and falls
  back to the heuristics when its output is unusable.
- **Compression.** The same error seen in related tasks merges into one lesson
  with a support count. `consolidate()` merges duplicates, drops old raw traces
  and retires lessons that keep failing to help.
- **Retrieval.** IDF-weighted text similarity plus spreading activation over the
  graph (lesson, concept, error type, episode). `feedback(lesson_id, helped)`
  moves a lesson up or down in future rankings.

## 2. Reasoning that checks itself: `agi.reasoning`

A `Problem` supplies hypotheses and a `verify` function. `MCTS` and
`TreeOfThoughts` explore the hypotheses, run each through the verifier, and
abandon a whole branch as soon as its assumption is refuted. A result is only
reported as solved when verification passed.

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

Included problems: `PipelineSynthesis` (program synthesis from examples over a
22-operation DSL), `Game24` (exact rational arithmetic), and `CodeRepair` (a
model writes code, the sandbox runs the tests, the real failure goes back into
the next prompt). `agi.reasoning.logic` is a propositional verifier (Tseitin
encoding plus DPLL) that returns a counterexample for every invalid step.

## 3. Learning from consequences: `agi.world`

An `Environment` is anything the agent can act on and observe. `WorldModel`
predicts the outcome of an action, measures its surprise when reality disagrees,
and corrects itself from that observation.

```python
from agi.world import ProjectileWorld, WorldModel, GroundedAgent, ideal_range

env = ProjectileWorld(drag=0.01, wind=-2.0)       # the real world has drag and wind
model = WorldModel(env.bounds, prior=lambda a: ideal_range(a["angle"], a["speed"]))  # the textbook does not
agent = GroundedAgent(env, model, observe="distance", seed=0)

for trial in agent.practice(target=30.0, trials=6):
    print(f"predicted {trial.predicted:5.1f} m, landed {trial.observed:5.1f} m, missed by {trial.miss:4.2f} m")
# predicted  30.0 m, landed  24.5 m, missed by 5.49 m
# predicted  30.0 m, landed  28.2 m, missed by 1.85 m
# ...
# predicted  30.0 m, landed  29.7 m, missed by 0.34 m
```

- **Bridges.** `ProjectileWorld` and `PhysicsWorld` (a small physics engine),
  `CodeWorld` (the sandbox as an environment), and `FunctionWorld` (any Python
  callables or API clients, behind an allowlist and a call budget).
- **Causal discovery.** `CausalLearner` intervenes on variables (`do(x)`) instead
  of only watching them. On the bundled example it recovers rain → wet grass →
  slippery, and reports that ice cream and sunburn are correlated but not causal
  (a hidden common cause).

## 4. Self-improvement with hard limits: `agi.collective`

Agents write new Python tools for each other. A tool enters the shared registry
only after passing five gates, and the rules of those gates are not something an
agent can change.

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

The gates, in order:

1. **policy**: static safety check of code and tests, naming, dependency and capacity rules.
2. **author-tests**: the author's tests must pass in the sandbox.
3. **regression**: a new version must pass every test the previous version was approved with.
4. **safety-veto**: one safety concern from any reviewer rejects the tool, whatever the vote.
5. **consensus**: two thirds of independent reviewers must approve. The forge
   re-runs each reviewer's tests itself; an approval whose tests fail counts as a rejection.

Shared state lives in `conflux-state` CRDT replicas, one per agent. Every write is
a signed action restricted to the writer's namespace, so replicas converge to
identical state in any delivery order, and an agent cannot write the registry,
forge another agent's review, or edit a message after signing it.

The boundaries are listed in code as `agi.collective.HARD_BOUNDARIES`:

- Tool code only ever runs inside the sandbox, during review and after approval.
- The policy is fixed at construction; agents cannot propose, vote on or edit it.
- Only the forge can write the tool registry.
- Authors cannot approve their own tools; approvals must come from distinct, registered agents.
- Evidence beats votes: a reviewer whose own tests fail is counted as a rejection.
- Any safety concern from any reviewer is a veto.
- A new version must pass every test the previous version was approved with.
- Tools compose only through declared dependencies, to a bounded depth.
- Every decision is recorded in a hash-chained audit log.

## Putting it together: `LearningAgent`

`agi.agent.LearningAgent` joins pillars 1 and 2: recall lessons, re-verify any
remembered solution instead of trusting it, otherwise search with priors taken
from memory, then record what happened.

```python
from agi import LearningAgent, Task
from agi.memory import MemoryEngine

task = Task("sort a list in descending order", [([3, 1, 2], [3, 2, 1]), ([5, 4, 9, 1], [9, 5, 4, 1])])
agent = LearningAgent(MemoryEngine())
print(agent.solve(task))    # solved via search: sort | reverse  [115 programs executed, ...]
print(agent.solve(task))    # solved via recalled: sort | reverse  [1 programs executed, ...]
```

In the demo (`python -m agi agent`), a task related to one solved earlier needs
194 program executions with memory against 2,341 without it.

## Using a language model

Everything above runs without a model. To add one, pass any `Callable[[str], str]`:

```python
from agi.llm import OpenAICompatLLM, ScriptedLLM

llm = ScriptedLLM(["def double(x):\n    return 2 * x\n"])     # canned replies, for tests
# llm = OpenAICompatLLM("gpt-4o-mini")       # any OpenAI-compatible endpoint; reads OPENAI_API_KEY
```

It is accepted by `agi.memory.LLMExtractor(llm)`, `agi.reasoning.CodeRepair(task, tests, llm)`
and `agi.collective.LLMAgent(agent_id, llm)`. All three fail closed: unusable
model output becomes a heuristic lesson, a refuted hypothesis, or a rejecting review.

## The sandbox, and what it does not do

`agi.sandbox.Sandbox` runs untrusted code in layers: a static AST gate (import
allowlist, no `eval`/`exec`/`open`, no underscore attributes), a fresh
interpreter process with an empty environment, kernel limits on CPU, memory,
file size and process count, a network namespace where the host allows it, and
a runtime audit hook. `Sandbox().isolation` reports which layers are active.

The static gate and the audit hook run inside CPython and are not a security
boundary on their own. The process, kernel-limit and namespace layers are the
real boundary, and they are as strong as the host kernel. For code you consider
hostile, wrap the interpreter in a container or VM through
`SandboxPolicy(launcher=[...])`.

## Known limits

- **Search without priors is weak on deep programs.** With no memory to guide it,
  `MCTS` on `PipelineSynthesis` solved the four-step "second largest" task on
  only 3 of 6 seeds within 3,000 verifications.
- **Examples can under-specify a task.** Synthesis returns the first program
  that fits every example, so too few examples can fit a coincidental program.
- **Memory retrieval is lexical.** It uses hashed word features, not a learned
  embedding, so a lesson phrased very differently from the query can be missed.
  Pass a different `embedder` to change this.
- **`OpenAICompatLLM` is tested against a local stub server only**, not against a
  live model endpoint.
- **The collective runs in one process.** Replication, signatures and namespace
  checks are real, but there is no network transport; the forge is a single
  trusted kernel.
- **Platforms.** Windows is not supported (the sandbox relies on POSIX resource
  limits). macOS is untested and would run without the network-namespace layer.

## Layout

```
agi/
  sandbox.py      layered sandbox used by pillars 2, 3 and 4
  llm.py          the only model-facing module
  agent.py        LearningAgent: memory plus verified search
  memory/         pillar 1
  reasoning/      pillar 2
  world/          pillar 3
  collective/     pillar 4
  __main__.py     python -m agi demos
tests/            one file per module
```
