# Research agenda: what would have to be true

**Status:** draft v0.1, 2026-11-16 · branch `arena/e57d7504-agi` · nothing built yet, this is the map

**Purpose.** Turn the four axes in the brief into falsifiable questions we can actually
run in this repository, and say plainly which parts are already covered, which are
measurably broken, and which need machinery that does not exist yet.

**How to read claims.** Each factual claim is tagged:

- `[local]` measured by running code in this checkout — command in Appendix A, reproducible
- `[primary]` from the benchmark's own publication or results page
- `[secondary]` from a third-party tracker — treat numbers as approximate, revisions differ
- `[inference]` my reasoning from the above, not a published result

---

## 0. What this repository is, measured

Not a summary of the README — what the code does when run.

| Property | Value | Tag |
| --- | --- | --- |
| Tests passing | 337 | `[local]` |
| Runtime dependencies | 1 (`conflux-state`); everything else stdlib | `[local]` |
| Modules | 9 (`tools`, `react`, `memory`, `reasoning`, `training`, `evals`, `world`, `collective`, `sandbox`) | `[local]` |
| Benchmark tasks | 8, in 8 distinct skills, longest reference solution 2 tool calls | `[local]` |
| Model needed to run | none — `ScriptedLLM` and reference solutions | `[local]` |
| Persistent state | SQLite lesson/episode graph (`agi.memory`), a policy JSON (`agi.training`) | `[local]` |
| Cross-process task state | none — `ReActAgent.run` is one call, `max_steps=12` | `[local]` |
| Environment seam | `agi.world.Environment` (3 implementations), no grid/visual/embodied env | `[local]` |
| Interactive benchmark support | none | `[local]` |

The important thing about this shape: **the toolkit is a verification engine with no
perception**. It is very good at "propose, execute, check the real result, keep the
lesson". It has no eyes, no long clock, and its memory has no notion of time. Those three
absences are exactly the three axes in the brief that are furthest from solved in the
field, which is either a lucky fit or the whole opportunity.

---

## 1. Where the frontier actually is (November 2026)

Numbers below are the ones I could source today; where estimates disagree between
revisions I give the disagreement rather than picking a winner.

### 1.1 ARC-AGI — the sample-efficiency axis

- ARC-AGI-2 went from **3.0%** at publication (May 2025) to **~95%** on the semi-private
  set by late September 2026. Cost per task **fell from ~$15.72 to ~$1.12–1.25** over the
  same period `[secondary]` ([leaderboard tracker](https://creuto.com/arc-agi-2-benchmark-general-reasoning-gap)).
- The ARC Prize 2025 write-up (December 2025) had the best verified commercial model at
  **37.6% / ~$2.20 per task** (Opus 4.5 thinking, 64k) and the best refinement solution at
  **54% / ~$30 per task** (Poetiq, on Gemini 3 Pro, up from that model's 31% baseline at
  $0.81) `[primary]` ([arcprize.org](https://arcprize.org/blog/arc-prize-2025-results-analysis)).
  Read that as: **+23 points of accuracy for ~38× the cost, bought by looping a model over
  its own attempts, not by a better model.**
- Human baseline as ARC accounts it: panel **100%** at **~$17 per task**; individual
  attempts around **60–75%** `[primary/secondary]`.
- ARC-AGI-3 launched **2026-03-25**: interactive, turn-based, no instructions, no stated
  goal, ~135 environments (110 private), scored by **RHAE** — human action count as the
  baseline, with the inefficiency penalty squared `[primary/secondary]`
  ([arcprize](https://arcprize.org/blog/arc-prize-2025-results-analysis),
  [coverage](https://winbuzzer.com/2026/03/30/arc-agi-3-offers-2m-ai-matching-human-reasoning-benchmark-xcxwbn/)).
- Headline: **humans 100%, every frontier model under 1%**; best reported ~0.37%
  (Gemini 3.1 Pro), and a month-long developer preview with custom scaffolding reached
  **12.58%** `[secondary]`. Humans clear environments in a median 7.4 minutes.

Two things worth sitting with. First, the model that scored 77.1% on ARC-AGI-2 scored
0.37% on ARC-AGI-3 — the drop is not a capability drop, it is the removal of a task
format the models had been shaped for. Second, ARC's own list of observed failure
modes on v3 is short and diagnostic: **thinking it is playing a different game, holding
an early hypothesis too long, failing to forecast forward** `[primary, quoted]`. Those are
not knowledge failures. They are state and model-maintenance failures.

### 1.2 Breadth × depth — the DeepMind levels axis

The Levels of AGI framing (Morris et al., 2023) is a matrix: performance (Emerging →
Competent 50th percentile → Expert 90th → Virtuoso 99th) × generality (narrow vs general,
where "general" explicitly includes metacognitive tasks like *learning new skills*)
`[primary]` ([arXiv 2311.02462](https://arxiv.org/html/2311.02462v2)).

Its own conclusion in 2023 was that frontier LLMs sat in the **Emerging AGI** cell, with
Competent/Expert performance on narrow tasks. The awkward part for benchmarks: the matrix
is a claim about *humans at percentiles*, and almost nothing we run is percentile-calibrated
against humans doing the same task. ARC is the exception, which is why it carries the weight
it does.

`[inference]` The matrix's real use is as a **coverage requirement**, not a score: a system
claiming Competent-general needs a wide domain set *and* per-domain depth, and our current
evaluation has neither — 8 tasks, 8 skills, 1–2 steps deep `[local]`.

### 1.3 Time-horizon autonomy — the METR axis

- Metric: 50% (and 80%) task-completion time horizon — the human-expert task duration at
  which the agent succeeds half (or 80%) of the time `[primary, Kwa et al. 2025]`.
- Trend: doubling roughly every **7 months** from 2019–2025; METR's Time Horizon 1.1
  update (Jan 2026) fits a **post-2023 doubling time of ~131 days**, with 228 tasks,
  31 of them 8+ hours `[secondary]` ([aiwiki](https://aiwiki.ai/wiki/metr_time_horizon),
  [tracker](https://ai2027-tracker.com/predictions/metr-doubling/)).
- Individual points, with visible estimator disagreement: GPT-5 at **2h17m** on TH1.0 vs
  **3h34m** on TH1.1; Claude Opus 4.6 at **~12–14.5h** on the 50% horizon and **~1h10m** at
  80% `[secondary]`. Confidence intervals are wide (±30–50%); treat single-model points as
  noisy and the *slope* as the finding.
- Sobering counterweight, same organisation: roughly **half of SWE-bench-passing PRs
  would not be merged by maintainers** `[secondary, METR Mar 2026 study]`. Benchmark
  success systematically overstates work that a human would accept.

**The analytic point that matters for us.** If a task is a chain of `n` independent steps
each succeeding with probability `p`, success is `p^n`, so the *ratio* of the 80% horizon to
the 50% horizon is fixed by arithmetic:

```
H80 / H50 = ln(0.8) / ln(0.5) = 0.322
```

Measured systems are far below that: GPT-5 ~0.23 (~33min / 2h18), Opus 4.6 ~0.10
(1h10 / 12h) `[inference from secondary]`. **Lower than 0.32 means failures are correlated,
not independent** — the agent is not making a fresh coin flip per step, it is getting stuck
in a bad state and staying there: context rot, an unmoved hypothesis, an unrecovered error.
That ratio is cheap to compute and it isolates the thing we can actually fix (state recovery)
from the thing we cannot (raw per-step competence). I want it as a first-class number in
every harness we build — call it the **horizon ratio** `[inference, proposal]`.

### 1.4 Grounded world model and memory — the state axis

- **Recall is not the bottleneck any more; revision is.** Memora + the FAMA metric
  (Forgetting-Aware Memory Accuracy), April 2026: across 4 LLMs and 6 memory agents,
  agents frequently **reuse invalidated memories**, performance degrades consistently from
  the weekly to the quarterly setting, and "the challenge stems less from memory capacity
  than from failures to maintain coherent, up-to-date memory under frequent mutation"
  `[primary]` ([arXiv 2604.20006](https://arxiv.org/html/2604.20006v1)). In their error
  analysis of the best agent's recommendation errors, **64% came from outdated memory not
  being forgotten**.
- **Long context is not a substitute.** Context-rot studies (18 models, ~194k calls)
  report accuracy declining as input grows, and a focused ~300-token prompt beating a
  ~113k-token full history on conversational memory `[secondary, Chroma]`. Another report:
  splitting one instruction across turns cost **39%** on average across 15 models, because
  early partial answers stayed in context and got built on `[secondary, Microsoft/Salesforce]`
  ([summary](https://aerospike.com/blog/context-engineering-ai-agents/)).
- **The field is converging on a four-operation memory contract:** ingestion, **revision**,
  **forgetting**, retrieval — with the named failure mode being *forgetting by capacity
  instead of by relevance* `[secondary, "Is Agent Memory a Database?" 2026]`.
- Benchmarks have fragmented along exactly those seams: Mem0 (single-hop/temporal/multi-hop),
  BEAM (10M-token conversations), LongMemEval (knowledge updates, temporal reasoning,
  multi-session), MemoryBench (learning from feedback), LongMemEval-V2 (workspace
  affordances and gotchas) `[secondary]` ([survey](https://www.graphlit.com/blog/survey-of-ai-agent-memory-frameworks)).

This axis is the one where a small repo can do real work, because the hard part is a data
model, not compute.

---

## 2. The four axes, against what we have

### 2.1 Skill acquisition — *we have the engine, not the measurement*

What exists `[local]`: `PipelineSynthesis` (program search over a 22-op DSL with executed
verification), `SearchPolicy` (a trained softmax policy over hand-written features,
58% held-out next-step accuracy), `MemoryEngine.recall` (lessons injected into the prompt),
`LearningAgent` (recall → verified search → write back).

What is missing: **a protocol**. There is no k-shot regime, no held-out task family, no
transfer measurement, and no measurement of interference — and interference is the question
that separates "learns" from "learns without forgetting". Every claim we can currently make
is about one task at a time.

The three numbers that would make this axis real:

1. **Trials-to-mastery** on a *new* skill at k ∈ {1, 3, 10} examples, with the exact-rule
   ceiling known (we can always compute it: search the DSL exhaustively).
2. **Transfer**: same skill, unseen task family, no additional examples.
3. **Interference**: train skill A, then skill B, re-measure A. Retention loss is the price
   of the update. In a weights-free design that price should be near zero — that is a
   *testable prediction about our own architecture*, and if it is not zero, embedding drift
   in retrieval is the cause, which is fixable.

### 2.2 Breadth × depth — *our evaluation is 1.5 steps deep*

`[local]` 8 tasks / 8 skills, longest reference chain 2 tool calls. As a coverage claim that
is nothing like "general"; as a *depth* claim it is barely above single-hop. The
depth axis is cheap to build and immediately honest: the same domain at 1, 2, 5 and 20
verified steps will separate models that reason from models that pattern-match a format —
and it is the axis where scaffolds fail first.

Also `[local]`: no environment seam for grids, pixels, or anything interactive. `agi.world`
has `ProjectileWorld`, `CodeWorld`, `FunctionWorld` — all single-process, single-object,
synchronous. That is a fine interface (and the right place to attach), but there is nothing
behind it that resembles ARC-AGI-3's shape.

### 2.3 Time-horizon autonomy — *structurally impossible today*

`[local]` `ReActAgent.run()` is one synchronous call bounded at `max_steps=12`; `Run` is a
dataclass returned in memory; `Conversation` keeps 6 turns in a list. There is no journal, no
checkpoint, no resume, no external clock, no budget accounting in money, and no way to say
"this task is 40% done, come back tomorrow". Context is shortened only when a character
budget fills, by dropping whole observations from the oldest end — no summarisation, no
promotion of state to persistent storage.

That means the toolkit cannot even *represent* a four-hour task, let alone run one. This is
a bigger gap than it looks, and it is also the most tractable: none of it needs a model.

`[local]` Related: the `python` tool's state "does not persist between calls" (by design, and
correctly so for safety) — but that also means an agent cannot build up an interactive
environment model through code. Whatever long-horizon runtime we build has to carry that
state in a durable store, not in the interpreter.

### 2.4 Grounded world model and memory — *a defect, reproduced*

`[local]` Two episodes, same task, conflicting fixes. Result:

```
episode 1: http_get vendor/upload  → HTTP 429 ; fix: sleep 2 then retry      (worked)
episode 2: concurrency 8           → HTTP 429 ; fix: bulk endpoint, one req  (worked)
```

What `MemoryEngine.context()` returns afterwards:

> 1. [fix] When: retry uploads to the vendor API (Error). Then: 'http_get vendor/upload'
> failed with Error: HTTP 429 rate limited; **'sleep 2 then http_get vendor/upload'** worked
> instead. (seen 2x, utility 0.50)

The second fix is **gone**, the lesson was merged into the first, and the support count went
up — so confidence in the superseded advice *increased* because a contradicting observation
arrived. `Lesson` has `created`, `last_used`, `support`, `helped`, `hurt` and no field for
validity, supersession, or a validity interval.

The mechanism is four lines: `MemoryEngine._absorb` keeps the keeper's advice and adds
support, and only replaces advice in the one case `keeper.kind == "avoid" and other.kind ==
"fix"`. Both of these are `fix`, so the richer advice is dropped silently. That rule is
right for *duplicate* lessons and wrong for *contradicting* ones — and the engine cannot
currently tell them apart, because equivalence is decided by text overlap, not by the
conditions under which each fix is valid.

This is the same failure the Memora authors measured at scale (`64%` of recommendation
errors from stale memory), reproduced in nine lines on our own engine `[local]` +
`[primary]`.

Good news: everything needed to fix it is already here — SQLite, a typed graph with
weighted edges, and a retriever that spreads activation. `supersedes` is an edge type, not
a rewrite.

---

## 3. Five findings that cut across the axes

**F1 — Verification is the only lever that moves all four axes at once.** ARC scores are
exact-grid-match; METR tasks are pass/fail on tests; memory is graded on whether the answer
matches the *current* world state; breadth×depth is per-cell verified. Nothing in the
literature rewards an agent for sounding right. This repo's core instinct — execute, read
the real result, refuse ungrounded answers — is the asset, and each axis is a different
place to apply it. `[inference]`

**F2 — Every saturated benchmark was saturated by scaffolding, and the scaffolding did not
transfer.** ARC-AGI-2: 54% → 95% while cost per task fell 14×; the *same* models on
ARC-AGI-3 score <1%. Sample-efficiency gains that come from looping a model over its own
attempts are real but format-bound `[inference from §1.1]`. Any harness we build must report
**score and cost together**, and must include a **format-shift twin** (the same capability
asked in an unfamiliar shape), or it will measure our scaffolding rather than the agent.

**F3 — The long-horizon problem is mostly a state problem.** The horizon ratio (H80/H50
≈ 0.1–0.23 vs 0.32 for independent steps) says the failures are correlated: the agent does
not recover. Combined with F4, that points the work at *recovery*, not at step quality.
`[inference]`

**F4 — Memory's unsolved half is revision, not recall.** Both the measurement (FAMA) and
our own reproduction say the same thing. Recall benchmarks flatter a system that hoards.

**F5 — Cost per verified outcome is the last discriminating axis.** At 95% for ~$1.12 a
task, accuracy no longer separates systems; ARC-AGI-3's RHAE squares the efficiency penalty
for exactly this reason, and the human panel's ~$17/task anchor only means something next to
a dollar figure. We should report `$/verified-success` and `human-actions / agent-actions`,
not pass rates. `[inference]`

---

## 4. Proposals

Ordered by (value ÷ effort). Each has a falsifiable acceptance test — if the test fails, the
proposal is wrong and we should say so in the README rather than tune it until it passes.

### R1 — A skill-acquisition harness on top of what exists · ~2 days · highest value/effort

Build `agi/skills.py` + `tests/test_skills.py`:

- A `Skill` = (DSL or tool API, task generator, verifier, exact-rule ceiling from exhaustive
  search). Reuse `OPS`, `PipelineSynthesis`, `make_tasks`.
- Protocol: for each skill, k-shot at k ∈ {1,3,10}; measure trials-to-mastery, transfer to a
  held-out generator seed family, and a **retention matrix** (learn A, learn B, re-test A
  without hints).
- Report: `mastery@k`, `transfer`, `interference`, and search cost per solve.

*Acceptance:* (a) retention loss < 10% relative after learning a second skill — if not, the
retrieval/embedding design is at fault and we fix it or document it; (b) transfer above
chance on held-out families for at least the 3 shortest skills; (c) each cell has a
known-solvable reference solution so a failure means the agent, not the task.

### R2 — Temporal, revisable memory · ~3 days · closes a reproduced defect

Extend `agi.memory` with the four-operation contract:

- `supersedes` / `contradicts` edges on lesson merges; `_absorb` only merges when the advice
  is equivalent, otherwise records a revision and marks the older lesson `superseded_by`.
- Validity fields (`valid_from`, `superseded_at`, `reason`) and a resolver that prefers the
  currently valid lesson; `consolidate()` retires by irrelevance, never by capacity.
- Retrieval that returns a **current-state view** and can say "this used to be true" —
  episodes yes, advice no.
- Falsification on recall: a recalled `procedure` is re-executed when cheap before being
  offered as advice (we already do this in `LearningAgent`; make it the default).

*Acceptance:* the R2 trap suite — (a) the vendor-upload case returns the second fix as
current and the first as superseded; (b) a Memora-style weekly→quarterly staleness suite
where the score must not degrade more than a few points as the horizon grows; (c) no
lesson is ever retired because the table got big.

### R3 — Long-horizon runtime · ~5 days · new subsystem, no model needed

New `agi/longrun/`:

- **Task DAG** with durable state: nodes, dependencies, per-node verifier, status, budget
  spent (tokens, dollars, wall-clock). SQLite, same pattern as `memory`.
- **Journal** — every step appended before it is acted on, so a crash loses one step.
- **Compaction** at a configurable fraction of budget (start at 60%, per the context-rot
  findings, not at the limit), producing a *state summary + open questions*, not a transcript.
- **Recovery policy**: on repeated failure, do not retry — re-open the parent, re-read the
  journal's last verified fact, and escalate to a smaller scoped subtask.
- **Resume**: `Runner(store).run(task)` is a loop over DAG nodes; state survives process
  death, which is the whole point.

*Acceptance:* (a) kill -9 mid-task, resume, finish with no duplicated side effects;
(b) a synthetic 20-step task with injected per-step failure rate 0.3 completes at ≥ 80%
reliability (that is the `p^n` arithmetic beaten by recovery); (c) horizon ratio reported
and above 0.32 at n=20 — i.e. failures become *independent*, which is what recovery means.

### R4 — Interactive environment seam (`agi/envs/`) · ~5 days · highest risk, highest ceiling

- A tiny protocol: `reset() -> observation`, `act(a) -> (observation, reward, done)`,
  `render()`, `replay()`. Point `agi.world.Environment` at it so existing world models and
  `CausalLearner` attach unchanged.
- Two environments we own: a **procedural grid game** with a hidden rule set (level
  transitions, keys/doors, goal inference) and a **dialogue-free tool environment** where the
  affordances are unknown at start.
- Score with a RHAE-style metric: `(human_actions / agent_actions)²`, and log a replay per
  episode.
- Tag replays with the three ARC-reported failure modes — *wrong-game*, *stuck-hypothesis*,
  *no-forecast* — as an automatic classifier over the trace (cheap, mechanical: e.g. no
  action of a new kind for N steps = stuck).
- **Format-shift twin** (F2): every grid task also appears as a text-symbol grid and as a
  Python-list grid; a system that "learns the rule" should score within a few points across
  forms, a system that learned the format should not.

*Acceptance:* the reference agent solves ≥ 80% of our environments and the human-scale
action baseline is consistent; the format-shift gap is reported as a first-class number.
(The field will not care about our scores. The *gap* between forms is the finding.)

### R5 — Coverage matrix, not a task list · ~2 days · makes breadth honest

Replace the 8-task list with `domains × depth`, where depth = verified step count ∈
{1, 2, 5, 20}: 6 domains (data, code, files/OS, arithmetic-proof, research/web, planning)
× 4 depths = 24 cells, each with an independently computed answer (as today) **and** a
reference solution proving the cell is reachable with our tools. Report per-cell, plus
`$/verified-success`. Keep the unanswerable cell — "knowing what is not known" is a
capability, and it is one of the few things a scaffolded system gets *worse* at.

*Acceptance:* every cell has a passing reference solution; the 20-step cells are runnable
end-to-end through R3; failures are attributable to the agent, not to missing tools
(any missing tool is listed as a dependency of the cell).

### R6 — Close the self-improvement loop · ~1 week · the leverage play

The forge (5 gates, consensus, sandbox) already accepts or rejects tools written by agents.
What it never does is *supply the demand signal*. Wire it: R1/R5 failures that recur in the
same shape generate a `ToolNeed`; a forged tool that passes the gates is registered and the
failing cell is re-run. Then measure the only thing that matters: **does the forged library
compound?** Time-to-solve for cell family N as a function of tools forged so far, with a
control arm that re-derives tools from scratch each time.

*Acceptance:* monotone decrease in time-to-solve across 5 families with a flat control arm.
If they track each other, the forge is decoration and we say so.

---

## 5. What would make this agenda wrong

Stated up front so it can be checked later, not defended later.

1. **Scaffolding is transient.** If the accuracy gains in §1.1 came from test-time looping
   against *the same distribution*, then building harnesses that reward looping optimises
   for a moving target, and the ARC-AGI-3 result is the preview. Mitigation: format-shift
   twins everywhere (F2).
2. **We are measuring our own agents.** Every acceptance test here is runnable with zero
   models, which is a strength for reproducibility and a weakness for relevance. Until a real
   model is behind `agi.llm`, R1/R3/R4 measure the machinery, not the frontier.
3. **The four axes are not independent.** If memory revision (R2) turns out to *cause* most
   of the long-horizon gain (R3), then this agenda is one problem wearing four hats, and the
   right move is to merge R2 and R3 and drop the rest.
4. **Percentile framing may be unfalsifiable in practice.** "50th percentile of skilled
   adults" needs humans doing the same tasks under the same tooling. We can do it for ARC
   and METR-style tasks because the data exists; for our own R5 cells we would be grading
   ourselves.
5. **Cost is under-counted.** `$/verified-success` from a scripted agent is not the real
   number; a model in the loop changes it by an order of magnitude. Report both, label which.

---

## 6. Decisions needed before building

1. **Which proposal first** — R1 (measurement), R2 (fix the reproduced defect), R3 (the
   structural gap), or R4 (the speculative ceiling).
2. **Model policy** — offline-only (scripted stand-ins, reproducible, not representative) or
   an OpenAI-compatible endpoint (representative, needs a key, results not reproducible
   offline). This determines whether any number we produce is comparable to §1.
3. **Whether we publish** — a research note per proposal with the acceptance test result
   included whether it passes or fails, versus internal notes only.

---

## Appendix A — reproduce the local findings

```bash
python -m venv .venv && .venv/bin/pip install -q pytest conflux-state
.venv/bin/python -m pytest -q                      # 337 passed (clean checkout, before any edits)

# the memory revision defect (§2.4)
.venv/bin/python - <<'PY'
from agi.memory import MemoryEngine, Episode, Step, SUCCESS
m = MemoryEngine()
m.record(Episode("retry uploads to the vendor API", SUCCESS, tags=("http",),
  steps=[Step("http_get vendor/upload", ok=False, error="HTTP 429 rate limited"),
         Step("sleep 2 then http_get vendor/upload", ok=True)]))
m.record(Episode("retry uploads to the vendor API", SUCCESS, tags=("http",),
  steps=[Step("http_get vendor/upload with concurrency 8", ok=False, error="HTTP 429 rate limited"),
         Step("bulk endpoint, single request", ok=True)]))
print(m.context("upload to vendor API with retries", tags=("http",)))
print(sorted(m.lessons()[0].__dict__))   # no validity / supersedes field
PY

# evaluation shape and the long-horizon bounds (§2.2, §2.3)
.venv/bin/python - <<'PY'
from agi.evals import SUITE
print(len(SUITE), "tasks;", "max reference chain:", max(len(t.reference) for t in SUITE))
PY
```

## Appendix B — reading list, in the order that matters for this repo

1. Kwa et al., *Measuring AI Ability to Complete Long Tasks* (2025) — the horizon metric, and
   the 50%/80% pair we reuse as a diagnostic.
2. *From Recall to Forgetting* / Memora + FAMA (2026) — the memory protocol we are behind on.
3. Chollet et al., ARC-AGI-2 and the ARC-AGI-3 technical report — the interactive format and
   RHAE scoring; the failure-mode taxonomy in the launch post is worth reading twice.
4. Morris et al., *Levels of AGI* (2023) — the coverage requirement, not the score.
5. *Is Agent Memory a Database?* (2026) and the context-rot studies — ingestion, revision,
   forgetting, retrieval, and why long context is not memory.
