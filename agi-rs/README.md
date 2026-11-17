# agi-core (Rust)

An autonomy kernel in Rust, implementing the design in
[`../docs/research/brain-grounded-autonomy.md`](../docs/research/brain-grounded-autonomy.md).

**Status: unverified — never compiled.** The sandbox this was written in has no Rust
toolchain and cannot reach `static.rust-lang.org`, so `cargo` was unavailable. The code was
written to be conservative (std only, no macros beyond `format!`/`vec!`/`write!`, no unsafe,
no async, no closures over borrowed mutables) and was audited by script for balanced
delimiters, dangling imports, undefined symbols and failing test logic — six real bugs were
found and fixed that way. **Run `cargo test` before trusting it.** If something fails, the
failure will be a compile error of the ordinary kind (type mismatch, borrow), not a design
error.

## What it is

Not "AGI in Rust". It is the part of an agent that decides *what to work on* and *whether it
is working* — the part that current scaffolds get wrong in ways that are cheap to fix and
easy to measure.

| Module | What it does |
| --- | --- |
| `types` | ids, deterministic xorshift RNG, Jaccard, least-squares slope, Wilson interval, ECE |
| `drives` | four homeostatic drives with setpoints, satiation decay, learnability filter |
| `goals` | goal lifecycle as a validated state machine, checks, habituation, budget |
| `selection` | arbitration: winner-take-all with lateral inhibition; may return `None` |
| `eval` | reward prediction error, the decision rule, calibration error |
| `world` | small calibrated predictive model, updated by every outcome |
| `memory` | revision-aware lessons: absorb duplicates, supersede contradictions |
| `journal` | append-only audit, reverse surprise-prioritised replay, JSONL export |
| `runner` | the tick loop, pivots, metrics |

## The five invariants this crate is built around

1. **A goal cannot execute without a check that has run.** `Goal::begin_seeking` returns
   `Err(NoExecutedCheck)`. Verification is the bottleneck; this makes it structural.
2. **An evaluation that cites no prediction error is rejected.**
   `Evaluation::score(vec![])` returns `Err(UnbackedEvaluation)`. Silence never reads as
   success.
3. **Selection may return nothing.** `TickOutcome::Idle` is a recorded outcome. An agent
   that cannot idle invents work.
4. **A pivot supersedes.** The parent becomes terminal, the child inherits the *remaining*
   budget and no sibling is created. Pivot count is capped.
5. **Contradicting advice supersedes and lowers confidence.** `Memory::absorb` refuses
   non-equivalent advice; `Memory::revise` creates a revision edge. This is the defect that
   the Python reference reproduced: merging two `fix` lessons silently dropped the newer one
   and *raised* support for the superseded advice.

## Build and test

```bash
cargo test              # 81 tests: unit tests per module + tests/invariants.rs (T1-T9)
cargo run --example demo
cargo clippy -- -D warnings    # not available in the authoring sandbox; expected clean
```

Zero dependencies, so `cargo test` works offline. `edition = "2021"`, `rust-version = "1.70"`.

## The one number to look at

`runner::INDEPENDENT_HORIZON_RATIO = ln(0.8)/ln(0.5) ≈ 0.322`.

For a chain of `n` independent steps, the ratio of the 80%-reliability task horizon to the
50% horizon is fixed at this constant. Published agent measurements sit well below it
(roughly 0.10–0.23), which means their failures are **correlated** — the agent gets stuck in
a bad state rather than making a fresh attempt. The point of the recovery machinery here
(pivot, habituation, re-screening) is to move a measured ratio *up* toward 0.322. That is a
better target than any accuracy number, and `runner::horizon_ratio` computes it from two
measurements.

## Deliberate simplifications

- **`Runner::simulate_step` is synthetic.** There is no tool layer; the step outcome is a
  deterministic function of the goal's statement with a 15% failure chance. That is enough
  to exercise recovery, and it keeps the tests model-free. A real deployment replaces it
  with a tool call and feeds the observed value to `record_outcome`.
- **`Store` is in memory.** The API is shaped so a SQLite-backed store is a mechanical swap;
  the journal should be persisted *before* acting on each step, so a crash loses one step.
- **Checks are data, not execution.** `Check.spec` is a string and the caller runs it. This
  keeps the kernel pure and testable; a real system runs it in a sandbox.
- **The memory index is Jaccard over tokens.** No embeddings. Adequate for the invariants;
  not adequate for a large corpus, where the retrieval quality claim would need measuring.
- **One process, one goal per tick.** No concurrency. Long-horizon reliability depends on
  durable state, which is the omitted half (see the brief's §8.2).

## What would make this wrong

- If the horizon ratio does not move when recovery is added, the thesis that correlated
  failure is a *state* problem is wrong and the effort belongs in per-step quality.
- If `Idle` turns out to be the dominant outcome in real use, the drive setpoints are wrong,
  not the design — but that is a real possibility and the idle fraction is reported first for
  that reason.
- `argmax` + lateral inhibition could be worse than sampling in genuinely novel
  environments, where the highest-scoring goal is the one the model understands best. The
  `AffordanceScan` exploration budget is the hedge, and it is deliberately separate so the
  two can be measured against each other.
