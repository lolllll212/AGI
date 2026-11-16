# Brain-grounded autonomy: a portable design brief

**Version** 1.0 · 2026-11-16 · self-contained — written to be pasted into another AI
system as context. No dependency on the conversation that produced it.

**How to use this document.** Sections 1–2 are the argument. Sections 3–8 are the
specification. Section 9 is the test plan. Section 10 lists what would falsify the
thesis. Section 12 is a compressed prompt-block version for low-context use.
Every factual claim about the codebase in §2 was verified by reading or running it;
every claim about biology cites a specific result, and §10 says where the analogy
is weakest.

**Notation.** `[verified]` = checked against the repository at the commit named.
`[literature]` = published result, cite given. `[design claim]` = my position, with a
falsifier in §10.

---

## 1. The position, in one page

Most systems that claim "autonomous goal formation" implement a **generator**: a model
proposes objectives, a filter approves them, a loop pursues them. This brief argues that
framing is wrong in four specific ways, and that the biological account of motivated
behaviour gives a better architecture.

### 1.1 Goals are selected, not invented

`[design claim]` In animals, goals are not generated from nothing; they are **selected
from a small space of drives, given the affordances present**. A hungry mouse does not
invent "explore the left corridor" — hunger raises the incentive salience of
food-predictive cues, and a selection circuit picks among available actions
`[literature: Berridge & Robinson, incentive salience; Mink 1996, basal ganglia
selection]`.

The engineering consequence is large. If generation is the module, you must then judge
invented goals, and you have no ground truth for the judgement — which is why evaluators
in these systems end up asserting a progress float no measurement supports. If
**selection** is the module, the candidate set is bounded by (drive × affordance), and
every candidate carries a forecast that can be scored against what actually happens.

**Autonomy is arbitration under scarcity, not invention.**

### 1.2 A drive is a deficiency with a setpoint, never a reward to maximise

`[design claim]` Homeostatic reinforcement learning formalises this: reward *is* the
reduction of a drive's deviation from setpoint; the drive itself is not a reward signal
`[literature: Keramati & Gutkin 2014]`. This distinction is the difference between a
system that stops pursuing competence when it is competent, and one that optimises
"competence" forever and Goodharts itself into a metric-shaped hole.

Corollary: every drive needs a **satiation mechanism**, or attention never rotates.

### 1.3 Prediction error is the progress signal; there is no other honest one

`[design claim]` "Progress" as a scalar asserted by a model has no referent. The
measurable quantity is **reward prediction error**: the difference between the outcome the
world model forecast at commitment time and the outcome verified afterwards
`[literature: Schultz, Hollerman & Tremblay 1997 — dopamine as RPE]`.

This single substitution fixes three things at once:

- **Stall detection** becomes arithmetic: |RPE| ≈ 0 with no change in verified state
  means the goal is inert, however fluent the narration.
- **Pivot** becomes a decision rule: consistent negative RPE over k windows means the
  model of this goal is refuted.
- **Learning** is the same loop: RPE also updates the world model, so evaluation and
  model repair are one mechanism rather than two.

And it produces a non-obvious selection criterion: **a goal whose outcome the world model
predicts perfectly is not worth pursuing.** Value accrues where prediction is uncertain
but improvable. Expected information gain emerges from the arithmetic instead of being
bolted on.

### 1.4 Goals die by habituation, not by deadline

`[design claim]` The biologically correct termination is salience decay under repeated
non-reinforcement `[literature: Rankin et al. 2009, habituation]`, not a TTL and not a
`COMPLETED` status. Salience is a function of evidence, so abandonment becomes justified
by the record. A TTL kills goals that were about to succeed; habituation kills goals that
have demonstrably stopped moving.

### 1.5 The three sub-problems must not be conflated

The phrase "autonomous goal formation" hides three different problems with different
mechanisms and very different difficulty:

| Sub-problem | Question | Mechanism | Difficulty |
| --- | --- | --- | --- |
| **Generation** | what objectives are candidates? | drives × affordances | low, if drives are few |
| **Specification** | how is it checkable and maintained? | executable check + working-memory representation | **high — this is where systems die** |
| **Commitment / revision** | persist, switch, or abandon? | RPE + conflict monitoring + habituation | medium, given the above |

`[design claim]` Generation is the easiest and gets the most attention. A system that
generates goals it cannot check has built a machine for looking busy. **Specification is
the bottleneck.**

### 1.6 You cannot get autonomy all the way down

`[design claim]` Goal derivation is a chain: goals come from drives, drives come from
something. If the bottom is learned, it is circular; if it is given, that is a design
choice. The honest target is a **small explicit kernel of drives plus a revisable layer
above it** — the kernel is config, not an emergent property. This is not a compromise; it
is the only version that is auditable, and it converts an open research question into
ordinary engineering.

### 1.7 Autonomy must be spent from a budget, not granted as a mode

`[design claim]` Self-proposed work must draw from an explicit budget share (start at
~20% of compute). Without it, autonomy consumes the mission. The biologically motivated
form of this is not a hard cap but a **competing drive**: cost and uncertainty are drives
in the same arbitration as competence, so the limit is enforced endogenously rather than
by an accountant.

---

## 2. Verified ground truth about the target codebase

Repository `lolllll212/nexus`, `master` at `0421b50`. Read before designing against it.

### 2.1 What exists `[verified]`

- `src/nexus/domain/entities/goal.py` — `Goal`, `GoalStep`, `GoalEvent`, `GoalStatus`
  (`PROPOSED/ACTIVE/BLOCKED/COMPLETED/FAILED/CANCELLED`), `GoalPriority`. Fields include
  `budget_units`, `budget_spent`, `max_cost_per_step`, `requires_approval`, `deadline`,
  `plan`, `history`. Invariants present: `budget_remaining`, `has_budget`, `spend()`
  refusing over-spend, `cancel()` refusing to cancel a terminal goal.
- `src/nexus/domain/ports/autonomy.py` — `AutonomyPolicy` ABC: `authorize_action`,
  `authorize_step`, `spend`, `require_approval`, `grant_approval`, `audit`,
  `pending_approvals`, `audit_log`.
- `src/nexus/application/autonomy/goals.py` — `CreateGoalUseCase`, `ApproveGoalUseCase`,
  `CancelGoalUseCase`, `ListGoalsUseCase`, `GetGoalUseCase`, `AutonomyLoopUseCase`,
  `StepExecutor` protocol, `StepOutcome`.
- `src/nexus/application/subcortex/` — `amygdala.py`, `basal_ganglia.py`, `thalamus.py`,
  `pattern_detection.py` (`PatternInsight{pattern, evidence_count, recommendation}`),
  `synthesis.py` (`EntitySynthesisUseCase`), `dreaming/dream_session.py`,
  `dreaming/simulate.py` (`SimulationResult{problems_identified, solutions_tested,
  solutions_verified, deployments_ready}`).
- `src/nexus/application/interfaces/subconscious_coordinator.py` — background scheduler:
  pattern detection every `pattern_interval_seconds=1800`, dreaming at `dream_hour=3`,
  both as `asyncio` tasks ("cortex never waits for subcortex").
- `src/nexus/domain/value_objects/valence.py` — affective tagging; no predictive model.
- `src/nexus/infrastructure/api/routes/goals.py` — `POST /v1/goals`, `GET /v1/goals`,
  `GET /v1/goals/audit`, `GET /v1/goals/{id}`, `POST /v1/goals/{id}/approve`,
  `POST /v1/goals/{id}/cancel`, `POST /v1/goals/{id}/run`.

### 2.2 What does not exist `[verified]`

A proposed "autonomous goal layer" (`GoalOrigin`, `GoalProposal`, `GoalEvaluation`,
`GoalGeneratorPort`, `GoalEvaluatorPort`, `GoalGeneratorUseCase`,
`GoalEvaluatorUseCase`, `AutonomousGoalCycleUseCase`, four generation sources,
`NEXUS_AUTONOMOUS_GOALS_*` config, `/v1/goals/generate/*` and `/v1/goals/cycle`
endpoints) returns **zero matches across all 15 branches including `master` and `main`**,
with no open pull requests. If a design references those symbols, the design is
aspirational.

### 2.3 Three defects in the existing skeleton `[verified]`

1. **The autonomy loop cannot represent failure.** In `AutonomyLoopUseCase.run_goal`, the
   `else` branch also sets `step.status = StepStatus.DONE`. A failed step is recorded as
   success; the loop terminates only when budget exhausts or policy denies. Consequence:
   *no downstream component can detect a stalled goal, because nothing is recorded as
   stalled.* Any evaluator built on top of this is decorative. **Fix this first.**
2. **`deadline` is declared and never enforced.** No check in the loop, no scheduler, no
   expiry. A goal with a past deadline is pursued normally.
3. **`max_active` is bypassed when approval is required.** In `CreateGoalUseCase`, the
   ceiling is only checked `if not requires_approval`, so the default path
   (`requires_approval=True`) is unbounded — the quota does not apply where it matters.

Also worth noting: `PatternInsight` carries an LLM-produced `pattern` string with no
executable check attached, and no minimum-support threshold; the 30-minute cadence over
short windows will manufacture spurious patterns. Any goal sourced from it needs a
confirmation run before it can consume horizon.

---

## 3. Domain model

### 3.1 The drive kernel

Four drives, each a deficiency with a setpoint, a value function over world state, and a
satiation gain. Chosen because each has an **externally verifiable proxy**:

| Drive | Proxy | Setpoint semantics |
| --- | --- | --- |
| `competence` | verified success rate over active task families | target rate; deficiency below it |
| `coverage` | count of task families with ≥1 verified success | saturating; diminishing returns |
| `cost` | spend per verified success | ceiling; deficiency above it |
| `uncertainty` | reducible model error (see §3.3 learnability filter) | target band, not zero |

```python
@dataclass(frozen=True)
class Drive:
    name: str
    setpoint: float
    value: Callable[[WorldState], float]        # measured, never self-reported
    deficiency: Callable[[float, float], float] # signed, floors at 0 beyond setpoint
    satiation: float                            # gain in [0,1], decays when satisfied
    learnability: Callable[[WorldState], float] # in [0,1]; filters unfixable uncertainty
```

Rules: (a) `deficiency` floors at zero when the drive is satisfied — a satisfied drive
contributes no urgency; (b) `satiation` decays with time-since-last-contribution and
recovers on contribution, rotating attention; (c) proxies are recomputed by the runtime
from the audit ledger, never supplied by the model.

### 3.2 Candidate generation as (drive × affordance × forecast)

A **goal proposal is a triple**, not a sentence:

```python
@dataclass(frozen=True)
class GoalProposal:
    drive: str                       # which deficiency this addresses
    affordance: Affordance           # an actionable option, discovered or given
    forecast: Forecast               # expected outcome + uncertainty
    parent_id: str | None = None     # derivation, orthogonal to origin
    origin: GoalOrigin               # provenance of the evidence, not of the goal
    evidence: tuple[EvidenceRef, ...]  # append-only; runtime-written
    check: GoalCheck                 # must exist before ACTIVE, see §3.5
```

`GoalOrigin ∈ {HUMAN, PATTERN_DETECTION, DREAM_SIMULATION, SELF_GENERATED, SUBGOAL}` is
retained as **provenance of evidence**, with two corrections:

- **`origin` and `parent_id` are orthogonal.** A subgoal of a dream-origin goal is
  `origin=DREAM_SIMULATION, parent_id=<dream>`. `SUBGOAL` as an origin is a category
  error and makes the common case unrepresentable.
- The four "generation sources" are not four generators. They are **four evidence
  channels feeding one selection function** (pattern detection raises the salience of a
  recurring problem; dreaming supplies simulated forecasts; synthesis supplies concepts
  that make an affordance legible; decomposition supplies a parent to expand). One gate,
  one ledger, one metric set — which is what makes the system auditable.

### 3.3 Lifecycle

```
PROPOSED → SCREENED → SHADOWED → (APPROVED | REJECTED) → SEEKING │ CRAVING │ PAUSED
                                      SEEKING → ACHIEVED │ STALLED → PIVOTED │ HABITUATED
```

- **`CRAVING`** — high drive urgency, no affordance available. Stored, re-evaluated on
  each affordance scan, **never executed**. This one state removes most goal
  proliferation, because an agent with nothing actionable stops inventing work.
- **`SEEKING`** — urgency and affordance both present; this is the only executing state.
  Replaces the overloaded `ACTIVE`.
- **`PAUSED`** — low urgency or externally blocked. Subject to salience decay.
- **`STALLED`** — no change in verified state across k evaluations (§4.2).
- **`HABITUATED`** — salience below ε; dropped without ceremony, retained for audit.

Only the **runtime** may write transitions. If the model can write its own status,
self-proposed goals get relabelled as instructions and provenance becomes fiction.

### 3.4 Screening gates, in order

| Gate | Rejects | Output on fail |
| --- | --- | --- |
| Verifiability | no executable check exists | demote to a bounded probe (budget ≤ 1 step) |
| Learnability | uncertainty not reducible (noise, not ignorance) | drop |
| Value of information | low expected gain per unit cost | drop |
| Reversibility / blast radius | irreversible or unbounded side effects | require approval; cap blast radius |
| Redundancy | similarity vs `SEEKING` goals above τ | merge into the existing goal |
| Interference | conflicts with standing commitments | enqueue behind the priority ledger |
| Provenance completeness | evidence refs missing or model-written | drop |

### 3.5 The check contract — the load-bearing object

**Invariant: no goal may enter `SEEKING` without a `GoalCheck` that has executed at least
once.** A model may propose checks; a check only counts once it has run in the sandbox.

```python
@dataclass(frozen=True)
class GoalCheck:
    kind: Literal["verifier", "metric", "gate"]
    spec: str                      # executable: test id, SQL, predicate, tool call
    baseline: float                # measured at screening
    target: float
    forecast_curve: tuple[float, ...]   # expected trajectory; enables RPE before completion
    ttl: datetime                  # expires the check, not the goal
```

Four non-human signals that actually work as inputs: terminal verifier outcome; slope of a
stable metric over a window (**slope, not level** — level confounds with baseline); artifact
novelty/entropy; and no-change-in-verified-state across k evaluations (which needs no
judgement at all).

---

## 4. The evaluation contract

### 4.1 Prediction error, not progress

```python
@dataclass(frozen=True)
class PredictionError:
    goal_id: str
    predicted: float            # forecast at commitment time
    observed: float             # verified outcome, post-hoc
    horizon: int                # steps from commitment to observation
    rpe: float                  # observed - predicted
    uncertainty: float          # model's own, calibrated
    evidence_refs: tuple[str, ...]

@dataclass(frozen=True)
class GoalEvaluation:
    goal_id: str
    errors: tuple[PredictionError, ...]     # MUST be non-empty; else evaluation is invalid
    residual_trend: float                   # slope of |rpe| over the window
    state_change: bool                      # did any verified fact change?
    recommendation: Literal["continue", "replan", "pivot", "abandon"]
    rationale: str
```

`RPE_t = r_t + γ·V̂(s_{t+1}) − V̂(s_t)` for the temporal-difference form, or the simpler
terminal form above when the goal has one outcome. The terminal form is enough to start.

Validity rule: **an evaluation whose `errors` are empty is discarded, not logged as zero
progress.** Silence must not read as success — this is the same defect class as §2.3
defect 1.

### 4.2 Decision rule (deterministic, with hysteresis)

Evaluate **on verified state change**, not on a step counter; include a minimum wall-clock
or budget interval so a goal cannot be re-evaluated dozens of times per unit of work.

```
continue  if  |rpe| within tolerance  and  state_change
replan    if  residual_trend < 0  and  an alternative decomposition exists
pivot     if  residual_trend < 0  for k consecutive windows        (k = 2, start)
abandon   if  pivot_budget exhausted  or  salience < ε  or  check ttl expired
```

`pivot_budget` (start at 2 per goal) is what prevents a stuck goal from generating
children forever. A pivot **supersedes** the parent (inherits budget and evidence, closes
the parent's ledger) — it does not create a sibling. "Pivot" that adds goals is
proliferation in different clothing.

### 4.3 Habituation

```
salience ← salience · decay^(failed_attempts_since_last_progress) · (1 + η·rpe⁺)
prune when salience < ε
```

`decay ≈ 0.85`, `ε ≈ 0.05`. Two properties matter: it is monotone in evidence, and it is
reversible — a goal whose drive urgency spikes again rises in salience without being
re-proposed from scratch.

---

## 5. Arbitration (the selection circuit)

Mapping to basal ganglia function: candidate goals compete for **one** action channel;
selection is biased by drive urgency × expected gain × salience, gated by cost, and
subject to **lateral inhibition** so the winner suppresses the rest (the mechanism behind
"one goal at a time" behaviour).

```python
def select(goals, drives, world_model, budget) -> Goal | None:
    scored = []
    for g in goals:
        if g.state != "SEEKING":          continue
        u = urgency(drives[g.drive]) * g.salience
        ev = expected_gain(g, world_model)          # predicted Δ deficiency
        cost = cost_estimate(g, world_model)
        if cost > budget.remaining_share: continue
        scored.append((u * ev / max(cost, ε), g))
    if not scored: return None
    top = max(scored, key=lambda t: t[0])
    # lateral inhibition: suppress close competitors this cycle
    for score, g in scored:
        if g is not top and score > 0.9 * top[0]:
            g.salience *= 0.7
    return top[1]
```

Two design points worth defending:

- **Selection returns `None` legitimately.** An agent with no `SEEKING` goal must be able
  to idle and scan affordances. Systems that cannot idle invent work.
- **The softmax-with-temperature variant is worse here than argmax + inhibition.** With
  softmax you keep sampling low-value goals *because they are possible*, which is the
  exact behaviour lateral inhibition exists to prevent. Use stochasticity for exploration
  in *which affordances to scan*, not in which goal to pursue.

### 5.1 Frontier selection as a derived policy

A cheap and high-yield instance of the above: maintain a map of task family → verified
success rate and prefer families in the **50–70% band**. This is self-calibrating
curriculum with no invention at all, and it is the second source to add after failure
mining.

### 5.2 Conflict monitoring

`[literature: Botvinick et al. 2001, ACC]` Maintain a running conflict estimate (competing
goals with similar scores; contradictory checks; goals whose evidence was invalidated).
Rising conflict should (a) raise evaluation frequency and (b) bias toward *replanning*
rather than *abandoning*, because conflict signals ambiguity rather than failure.

---

## 6. Memory, replay, consolidation

### 6.1 Complementary learning systems

`[literature: McClelland, McNaughton & O'Reilly 1995]` Fast episodic store (one-shot, high
interference) + slow semantic store (interleaved, low interference). Consolidation
transfers between them. **Interleaving is the anti-forgetting mechanism** — replay a
fraction of old items alongside each new consolidation, rather than consolidating in
blocks. This is cheap (one line of sampling) and it is the difference between gaining a
skill and trading one away.

### 6.2 Replay is reverse and surprise-prioritised

`[literature: Foster & Wilson 2006; Diba & Buzsáki 2007 — reverse replay]` Replay
trajectories **backwards from the outcome**, prioritised by |RPE|. Reverse order is what
assigns credit to the early steps that made the outcome happen; prioritisation is what
makes replay worth the compute. Both are cheap and both address long-horizon credit
assignment without a model.

Concretely: on goal completion or abandonment, sample the episode's steps in reverse,
weighted by |RPE|, and update the world model on (state, action, next state) triples.

### 6.3 Dreaming = offline simulation against the world model

The existing `dreaming/simulate.py` returns `solutions_verified` — that is already an
observation to compare against a forecast. Make the target explicit: dreaming should
**generate forecasts for unresolved trajectories**, and its products enter the system as
`Forecast` objects with uncertainty, not as findings with implied truth. Anything sourced
from a dream requires a cheap real-world probe before it can reach `SEEKING`.

### 6.4 Revision, not just recall

`[literature: Memora/FAMA 2026; "Is Agent Memory a Database?" 2026]` The dominant
long-horizon memory failure is **reusing invalidated knowledge**, not failing to recall
(in one study, 64% of the best agent's recommendation errors came from outdated memory not
being forgotten). Required data-model properties:

- `valid_from` / `superseded_by` / `superseded_at` / `reason` on every lesson-like record.
- Merging is only legal when advice is **equivalent**; *contradicting* fixes become a
  revision edge (`supersedes` / `contradicts`), and support counts must not increase for
  superseded advice.
- Retrieval returns a **current-state view** and can say "this used to be true."
- Forgetting is by relevance, never by capacity.
- A goal whose supporting evidence was later invalidated must be **re-screened**.

Reproduction of the failure, for anyone implementing this: record two episodes with
conflicting fixes for the same problem where both lessons have kind `fix`; a merge rule
that keeps the keeper's advice and adds support will silently drop the newer fix and
*raise* confidence in the superseded one. `[verified]` in a reference implementation.

---

## 7. The world model (the missing prerequisite)

`[design claim]` Without a predictive model, autonomy is **unmeasurable**: RPE has no
`predicted` term, arbitration has no `expected_gain`, and there is nothing to be wrong
about. This component is a prerequisite for §4, not a nice-to-have.

Minimum viable:

```
V̂: (state, goal) → (expected outcome, calibrated uncertainty)
M̂: (state, action) → (next state, outcome)
```

- **Calibrated uncertainty is mandatory.** Use it in three places: arbitration (divide by
  cost with a risk term), the learnability filter (§3.1), and RPE normalisation (a large
  error under large uncertainty is not evidence of failure).
- **Interventions, not correlations.** Attribute an outcome to a goal only if acting
  differed from the counterfactual. Alternating goal/no-goal periods, or a held-out
  control family, is enough at this scale. Without it you will attribute every
  environment improvement to whatever goal was active.
- **Update from every RPE**, including abandoned goals. Abandonment is a refutation and is
  the most informative kind of update available.

---

## 8. Interfaces, for implementers

```python
# domain/ports/drives.py
class DriveKernel(ABC):
    def read(self, state: WorldState) -> Mapping[str, float]: ...      # deficiencies
    def satiation(self) -> Mapping[str, float]: ...

# domain/ports/goal_selection.py
class GoalCandidateSource(ABC):                      # one per evidence channel
    async def candidates(self, *, limit: int) -> Sequence[GoalProposal]: ...

class GoalSelector(ABC):
    def select(self, goals, drives, world_model, budget) -> Goal | None: ...

# domain/ports/goal_check.py
class GoalCheckRunner(ABC):
    async def run(self, check: GoalCheck) -> CheckResult: ...          # sandboxed

# domain/ports/goal_evaluator.py
class GoalEvaluator(ABC):
    async def evaluate(self, goal: Goal) -> GoalEvaluation: ...        # MUST cite checks

# domain/ports/world_model.py
class PredictiveModel(ABC):
    def forecast(self, state, goal) -> Forecast: ...
    def uncertainty(self, state, goal) -> float: ...
    def update(self, transition, rpe: float) -> None: ...

# domain/ports/replay.py
class ReplayBuffer(ABC):
    def sample(self, *, n: int, reverse: bool, priority: Literal["rpe"] = "rpe") -> ...: ...
```

Port placement rules learned from the target codebase: **ports describe what the domain
needs from outside**; driving use cases (`PatternDetectionUseCase`, `DreamSessionUseCase`,
`EntitySynthesisUseCase`) are application services, not ports. Putting them in
`domain/ports` inverts the dependency direction and makes the domain depend on scheduling
concerns.

### 8.1 Proposer / evaluator separation

Never let one model both invent a goal and score its progress — that is a self-grading
loop with no adversarial pressure. The evaluator runs checks in a sandbox and is
structurally unable to assert progress no check supports. Ideal: the evaluator does not
see the proposer's rationale, to avoid anchoring.

### 8.2 Operational requirements (all cheap, all commonly missing)

- **Durable queue, single consumer, idempotency key** per cycle. A crash mid-dream must not
  duplicate goals on restart.
- **Missed-cycle reconciliation.** A 3 AM wall-clock scheduler silently skips nights when
  the process is down; persist `last_completed(source, period)` and catch up on boot.
- **Backpressure with priority.** Over queue depth: drop dreaming first, synthesis second,
  pattern detection last.
- **Shadow mode.** `SHADOW=true` logs proposals, acts on nothing. Two weeks gives proposal
  precision *before* anything moves.
- **Per-source flags**, not one global switch. Dreaming is the source most likely to
  produce fictions and is the first you will want off.
- **Kill switch** that stops new `SEEKING` transitions without stopping in-flight work, so
  demonstration of the stop is possible at any moment.
- **API**: cycle endpoint is admin-only + idempotent; pivot supports `dry_run=true`
  returning the mutation plan; proposals return `evidence_refs` so a rejection is
  reviewable.

---

## 9. Test plan

Every item is falsifiable in a harness with no language model, which is the point: the
machinery should be testable before any model is in the loop.

| # | Test | Pass condition |
| --- | --- | --- |
| T1 | Failure is representable | a step that fails is recorded `FAILED`; goal reaches `STALLED` not `COMPLETED` |
| T2 | No unbacked evaluation | an evaluator returning progress with zero checks raises / is discarded |
| T3 | Empty affordance idles | with no actionable option, zero goals enter `SEEKING`; zero work executed |
| T4 | Proliferation invariant | over a fixed workload, total goals grow **sublinearly**; active goals ≤ N |
| T5 | Budget isolation | self-proposed work never exceeds its share, under adversarial drive urgency |
| T6 | Supersession | a pivot closes the parent, inherits budget + evidence, creates no sibling |
| T7 | Habituation | a goal with k non-reinforced attempts drops below ε and is pruned |
| T8 | Revision | contradicting evidence supersedes and **lowers** confidence in the old advice |
| T9 | Retention | learn A, learn B, re-test A: retention loss < 10% relative (interleaving) |
| T10 | Recovery beats arithmetic | synthetic 20-step task, per-step failure p=0.3, completes ≥ 80% of the time |
| T11 | Canary | a deliberately cheat-shaped goal is available and never selected |
| T12 | Calibration | predicted vs observed RPE calibration curve is monotone; report ECE |
| T13 | Crash safety | kill -9 mid-task, resume, complete with no duplicated side effects |
| T14 | Format shift | same capability in two task representations scores within a few points |

T10 deserves a note: for a chain of `n` independent steps each succeeding with `p`, success
is `p^n`, so the ratio of the 80%-reliability horizon to the 50%-reliability horizon is
fixed at `ln(0.8)/ln(0.5) ≈ 0.32`. Measured agent systems sit **well below** that
(roughly 0.10–0.23 for published frontier measurements), which means their failures are
**correlated** — the agent gets stuck in a bad state rather than making a fresh attempt.
Recovery machinery should move that ratio **up toward 0.32**, and that is a better target
for autonomy work than any accuracy number.

### 9.1 Metrics to ship with the feature, not after

- **Novel-but-accepted rate** — proposals a human would not have made that turned out
  useful. If the system only proposes what a human would, it is a parrot.
- **Realized value in a human-free holdout** — verified improvement per unit cost in a
  domain where nobody specifies tasks for N days. This is the only metric that answers the
  question.
- **Proposal precision** (accepted / proposed) and **abandonment calibration** (fraction of
  compute spent on goals whose expected value had already collapsed).
- **Drift audit on the goal distribution**, not per goal — review the distribution.
- **Interference** — does pursuing self-proposed goals degrade given tasks?
- **Salience/priority sanity** — is anything ever `CRITICAL`? If everything is, the field
  is decoration.

---

## 10. Where this is wrong, and what would falsify it

1. **The biological mapping is partly decoration.** Regional labels port well when a
   measurable quantity sits behind them (prediction error, salience decay, setpoint error,
   interleaving, lateral inhibition) and badly when they do not ("amygdala = threat
   detector" becomes a one-line wrapper). Also: the dopamine-as-RPE account is itself
   contested — recent work argues dopamine encodes state uncertainty/likelihood in some
   tasks. The *design* only needs a signed prediction error, which is defensible
   independently of the neuroscience.
2. **Recovery might not buy horizon.** If correlated failure is caused by model capability
   rather than state management, T10 will fail and the priority should move back to
   per-step quality. Test T10 before investing in the runtime.
3. **Memory revision might be the whole story.** If RPE-driven revision (T8) produces most
   of the horizon gain, then this brief is one mechanism described in many parts and the
   remainder should be cut rather than integrated.
4. **Selection-over-generation may be too conservative.** An agent restricted to
   (drive × affordance) cannot discover affordances the environment does not present. The
   counter-mechanism is deliberate exploration of the affordance space — but if that turns
   out to be where all the value is, the "selection not invention" thesis is weakened and
   the generator deserves a budget of its own.
5. **The evaluation may be self-grading anyway.** If the check author and the goal proposer
   are the same model in practice (different prompts, same weights), structural separation
   is thinner than it looks. Mitigation: checks must be written before the forecast is
   made, and the evaluator never sees the rationale.
6. **Cost is under-counted everywhere.** All metrics above need a dollar figure attached;
   `cost per verified success` from a scripted agent is not the real number.
7. **Percentile framing is unfalsifiable for our own tasks.** "Competent = 50th percentile
   of skilled adults" needs humans on the same tasks with the same tools. Use it only where
   human baselines exist.

---

## 11. Ten anti-patterns, with the mechanism that prevents each

| Anti-pattern | Looks like | Prevention |
| --- | --- | --- |
| Self-graded progress | a model asserts "70% complete" | evaluation invalid without executed checks |
| Pivot-as-proliferation | stalled goal spawns children | pivot supersedes; `pivot_budget` ≤ 2 |
| Idle invention | nothing actionable, so it makes work | `CRAVING` state + `select()` may return `None` |
| Metric wireheading | optimizing the proxy past setpoint | drives are deficiencies with floors, not maxima |
| Noisy-TV | chasing irreducible uncertainty | learnability filter in the drive kernel |
| Confident fictions | dream findings treated as facts | dream products are forecasts + require a probe |
| Mandate creep | distribution drifts, each goal fine | budget share + distribution-level drift audit |
| Silent success | failed steps marked done | failure is a first-class status, tested (T1) |
| Learned timidity | only trivially safe proposals | credit *attempted* frontier goals, not just successes |
| Unfalsifiable perpetuity | "in progress" forever | check `ttl` + habituation, both evidence-based |

---

## 12. Compressed prompt-block (for pasting into a low-context system)

> Design an autonomous goal system for an agent, grounded in how motivated behaviour
> actually works. Follow these positions.
>
> **1. Goals are selected, not invented.** Candidates are bounded by (drive × affordance).
> A goal proposal is a triple — which drive deficiency, which actionable option, expected
> outcome — not a sentence. Autonomy is arbitration under scarcity.
>
> **2. Drives are deficiencies with setpoints, never rewards to maximise.** Four drives,
> each with an externally verified proxy: competence (verified success rate), coverage
> (families solved), cost (spend per verified success), reducible uncertainty. Each has a
> satiation gain that decays, rotating attention. Anything above a setpoint contributes
> zero urgency.
>
> **3. The progress signal is reward prediction error, not an asserted float.** Predicted
> outcome at commitment vs verified outcome afterwards. An evaluation that cites no
> executed check is discarded, never logged as zero progress. Consequence: a goal the world
> model predicts perfectly is not worth pursuing — value accrues where prediction is
> uncertain but improvable.
>
> **4. Goals die by habituation.** Salience decays with repeated non-reinforcement and is
> pruned below a threshold. Not a TTL, not a `COMPLETED` flag.
>
> **5. Specification is the bottleneck, not generation.** No goal may execute without an
> executable check that has already run once. Four signals work without human labels:
> terminal verifier outcome, metric slope (not level), artifact novelty, and no verified
> state change across k evaluations.
>
> **6. Three states, not one.** `SEEKING` (urgent + actionable, the only executing state),
> `CRAVING` (urgent + not actionable: store, re-scan, never execute), `PAUSED`. An agent
> that cannot idle invents work. Selection may legitimately return nothing.
>
> **7. Selection is winner-take-all with lateral inhibition**, biased by drive urgency ×
> salience × expected gain ÷ cost. Prefer argmax + inhibition over temperature sampling:
> sampling pursues low-value goals precisely because they are possible.
>
> **8. Pivot supersedes.** It closes the parent, inherits its budget and evidence, and
> creates no sibling. Cap pivots per goal. Evaluate on verified state change, not step
> count, with hysteresis so it cannot thrash.
>
> **9. A world model is a prerequisite, not a nice-to-have** — it supplies the predicted
> term. Calibrated uncertainty, updated by every prediction error including abandonments.
> Attribute outcomes by intervention, not correlation.
>
> **10. Replay is reverse and surprise-prioritised**; consolidation interleaves old and new
> or it trades one skill for another. Memory must support **revision** (supersede, valid-
> from, relevance-based forgetting), because reusing invalidated knowledge — not failing to
> recall — is the dominant long-horizon memory failure.
>
> **11. Autonomy draws from a budget share** (~20% of compute) and runs in shadow mode
> first. Only the runtime writes status transitions; the proposer never grades itself.
> Ship with these metrics: novel-but-accepted rate, realized value in a human-free holdout,
> proposal precision, abandonment calibration, distribution-level drift, and a cheat-shaped
> canary goal that must never be taken.
>
> **12. Do not get autonomy all the way down.** A small explicit drive kernel is config, not
> an emergent property. That is what makes the system auditable.

---

## Appendix A — biology → computation → artifact

| Claim from biology | Computation | Artifact |
| --- | --- | --- |
| Homeostatic regulation, setpoints | drive deficiency with floor + satiation | `Drive`, `DriveKernel` |
| Incentive salience (wanting ≠ liking) | `CRAVING` / `SEEKING` split | goal state machine |
| Basal ganglia action selection | winner-take-all + lateral inhibition | `GoalSelector` |
| Dopamine RPE | signed predicted-vs-observed | `PredictionError`, `GoalEvaluation` |
| ACC conflict monitoring | conflict estimate → evaluation frequency | arbitration bookkeeping |
| Prefrontal maintenance | keep goal active against distractors; stable representation | `GoalCheck` + plan |
| Habituation | salience decay in evidence | pruning rule |
| CLS consolidation | interleaved replay | `ReplayBuffer` |
| Reverse replay | backwards credit assignment | replay sampler |
| Predictive coding | forecast + calibrated uncertainty | `PredictiveModel` |

## Appendix B — readings that carry the load

Keramati & Gutkin 2014 (homeostatic RL — drives as setpoints, the single most useful
formal anchor here); Schultz, Hollerman & Tremblay 1997 (dopamine RPE); Berridge &
Robinson (incentive salience); Mink 1996 (basal ganglia selection); Botvinick et al. 2001
(conflict monitoring); McClelland, McNaughton & O'Reilly 1995 (CLS); Foster & Wilson 2006
and Diba & Buzsáki 2007 (reverse replay); Rankin et al. 2009 (habituation); Memora/FAMA
2026 and "Is Agent Memory a Database?" 2026 (memory revision, forgetting by relevance).
