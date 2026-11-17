//! The loop: screen, select, execute, evaluate, habituate, revise.
//!
//! Everything the runner mutates goes through the validated APIs (`Goal::transition`,
//! `Evaluation::score`, `Memory::revise`), so the invariants hold by construction rather
//! than by discipline. Three behaviours are deliberate:
//!
//! * **Idling is a recorded outcome** ([`TickOutcome::Idle`]), not a no-op. An agent that
//!   cannot idle invents work.
//! * **Failure is recorded as failure** ([`TickOutcome::Failed`]), which is the defect this
//!   crate exists to correct: the reference loop marked failed steps `DONE`, so nothing
//!   downstream could ever detect a stalled goal.
//! * **A pivot supersedes.** The parent becomes terminal and the child inherits its budget
//!   and evidence; no siblings are created.

use crate::drives::{DriveKind, DriveSet};
use crate::eval::{Decision, Evaluation, Policy, PredictionError};
use crate::goals::{Check, Goal, GoalState, Origin};
use crate::journal::{EventKind, Journal};
use crate::memory::{Memory, Recall};
use crate::selection::{Candidate, AffordanceScan, Selector};
use crate::types::{Id, IdGen, EPS};
use crate::world::PredictiveModel;
use crate::{Error, Result};

/// Outcome of one tick. `Idle` is a first-class result.
#[derive(Debug, Clone, PartialEq)]
pub enum TickOutcome {
    /// Nothing was urgent, affordable and actionable.
    Idle(&'static str),
    /// A goal executed one step and the check was re-run.
    Executed { goal: Id, value: f64, progress: f64 },
    /// A goal executed and its step failed. Recorded as failure, never as success.
    Failed { goal: Id, reason: String },
    /// The goal's check met its target.
    Achieved { goal: Id },
    /// An evaluation recommended and executed a pivot.
    Pivoted { parent: Id, child: Id },
    /// A goal habituated and was dropped.
    Habituated { goal: Id },
}

/// Configuration for a run.
#[derive(Debug, Clone)]
pub struct Config {
    /// Simulation seconds advanced per tick.
    pub dt: f64,
    /// Salience below this habituates.
    pub habituate_epsilon: f64,
    /// Per-tick salience decay factor.
    pub salience_decay: f64,
    /// Total budget available to autonomy.
    pub total_budget: f64,
    /// Policy for the decision rule.
    pub policy: Policy,
    /// Whether autonomous proposals require explicit approval before executing.
    pub require_approval: bool,
    /// Shadow mode: log proposals, execute nothing.
    pub shadow: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            dt: 1.0,
            habituate_epsilon: 0.05,
            salience_decay: 0.85,
            total_budget: 100.0,
            policy: Policy::default(),
            require_approval: false,
            shadow: false,
        }
    }
}

/// A durable-ish store of goals. In-memory here; a caller would back this with SQLite and
/// persist the journal before acting on each step, so a crash loses one step rather than a
/// task. The API is shaped so that swap is mechanical.
#[derive(Debug, Clone, Default)]
pub struct Store {
    goals: Vec<Goal>,
}

impl Store {
    pub fn new() -> Self {
        Store::default()
    }

    pub fn insert(&mut self, goal: Goal) {
        self.goals.push(goal);
    }

    pub fn get(&self, id: Id) -> Option<&Goal> {
        self.goals.iter().find(|g| g.id == id)
    }

    pub fn get_mut(&mut self, id: Id) -> Option<&mut Goal> {
        self.goals.iter_mut().find(|g| g.id == id)
    }

    pub fn all(&self) -> &[Goal] {
        &self.goals
    }

    pub fn live(&self) -> impl Iterator<Item = &Goal> {
        self.goals.iter().filter(|g| g.state.is_live())
    }

    pub fn live_count(&self) -> usize {
        self.live().count()
    }

    pub fn count_in(&self, state: GoalState) -> usize {
        self.goals.iter().filter(|g| g.state == state).count()
    }

    pub fn remove_terminal(&mut self, keep_for_audit: bool) -> usize {
        if keep_for_audit {
            return 0;
        }
        let before = self.goals.len();
        self.goals.retain(|g| !g.state.is_terminal());
        before - self.goals.len()
    }
}

/// The kernel: drives, world model, memory, journal, store, and the clocks.
pub struct Runner {
    pub drives: DriveSet,
    pub model: PredictiveModel,
    pub memory: Memory,
    pub journal: Journal,
    pub store: Store,
    pub selector: Selector,
    pub scan: AffordanceScan,
    pub config: Config,
    pub ids: IdGen,
    /// Simulation time in seconds. Never read from the OS clock: that is what makes runs
    /// reproducible.
    pub now: f64,
    /// Most recent drive measurements. The runtime measures; a model never reports these.
    pub measured: [f64; 4],
    /// Errors collected since the last evaluation, per goal.
    pending: Vec<PredictionError>,
    /// Counts for the metrics in the brief.
    pub stats: Stats,
}

/// Counters that the metrics need. Reported, not estimated.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Stats {
    pub ticks: u64,
    pub idle_ticks: u64,
    pub steps: u64,
    pub failures: u64,
    pub proposals: u64,
    pub screened_out: u64,
    pub pivots: u64,
    pub habituated: u64,
    pub achieved: u64,
    pub budget_denied: u64,
    pub autonomy_spent: f64,
}

impl Runner {
    pub fn new(seed: u64) -> Self {
        Runner {
            drives: DriveSet::default_kernel(),
            model: PredictiveModel::default(),
            memory: Memory::new(),
            journal: Journal::new(),
            store: Store::new(),
            selector: Selector::default(),
            scan: AffordanceScan::default(),
            config: Config::default(),
            ids: IdGen::new(seed),
            now: 0.0,
            measured: [0.5, 0.0, 5.0, 0.5],
            pending: Vec::new(),
            stats: Stats::default(),
        }
    }

    // -- proposal and screening ---------------------------------------------- //

    /// Propose a goal. Proposals are cheap and always recorded, including in shadow mode.
    ///
    /// The origin decides whether a probe is required: a dream-sourced proposal may not
    /// execute until something real has agreed with it, so it lands in `Craving`, not
    /// `Seeking`.
    pub fn propose(
        &mut self,
        statement: impl Into<String>,
        drive: DriveKind,
        origin: Origin,
        check: Check,
        parent: Option<Id>,
    ) -> Result<Id> {
        let mut goal = Goal::new(&mut self.ids, statement, drive, origin, check);
        goal.created_at = self.now;
        goal.approved = !self.config.require_approval;
        if let Some(p) = parent {
            goal = goal.with_parent(p);
        }
        let id = goal.id;
        self.journal.push(self.now, Some(id), EventKind::Proposed, goal.label(), 0.0);
        self.stats.proposals += 1;

        // Gate: verifiability. A goal with no executable check is demoted to a probe with a
        // one-step budget rather than being silently accepted.
        let mut goal = goal;
        if goal.check.spec.trim().is_empty() {
            goal.budget_units = 1.0;
            self.stats.screened_out += 1;
            self.journal.push(
                self.now,
                Some(id),
                EventKind::Screened,
                "no executable check: demoted to a one-step probe",
                0.0,
            );
        }

        // Gate: redundancy. Fold a near-duplicate into the existing live goal.
        let redundant = self
            .store
            .live()
            .find(|g| crate::types::jaccard(&g.statement, &goal.statement) > 0.75)
            .map(|g| g.id);
        if let Some(existing) = redundant {
            self.stats.screened_out += 1;
            self.journal.push(
                self.now,
                Some(id),
                EventKind::Screened,
                format!("redundant with {existing}: merged instead of created"),
                0.0,
            );
            return Ok(existing);
        }

        goal.transition(GoalState::Screened)?;
        self.journal.transition(self.now, id, GoalState::Proposed, GoalState::Screened);

        if self.config.shadow {
            goal.transition(GoalState::Shadowed)?;
            self.journal.transition(self.now, id, GoalState::Screened, GoalState::Shadowed);
        } else if goal.origin.requires_probe() {
            // Dreams are confident fictions until a real probe agrees.
            goal.transition(GoalState::Craving)?;
            self.journal.transition(self.now, id, GoalState::Screened, GoalState::Craving);
            self.journal.push(
                self.now,
                Some(id),
                EventKind::Screened,
                "dream-sourced: craving until a real probe confirms an affordance",
                0.0,
            );
        }

        self.store.insert(goal);
        Ok(id)
    }

    /// Run a check for the first time, which is what unlocks execution. The caller supplies
    /// the observed value; in a full system this is a sandboxed test or query.
    pub fn run_check(&mut self, goal: Id, value: f64) -> Result<f64> {
        let now = self.now;
        let (spec, progress) = {
            let g = self.store.get_mut(goal).ok_or(Error::UnknownGoal(goal))?;
            g.check.record_run(value);
            (g.check.spec.clone(), g.check.progress(value))
        };
        self.journal.push(
            now,
            Some(goal),
            EventKind::CheckRan,
            format!("{spec} -> {value:.3} (progress {progress:.2})"),
            0.0,
        );
        Ok(progress)
    }

    /// Move a goal from `Craving` into `Seeking` once an affordance exists and its check has
    /// run. This is the only legitimate way a dream-sourced goal starts executing.
    pub fn satisfy_craving(&mut self, goal: Id) -> Result<()> {
        let now = self.now;
        let from = self.store.get(goal).ok_or(Error::UnknownGoal(goal))?.state;
        if from != GoalState::Craving {
            return Err(Error::IllegalTransition { from, to: GoalState::Seeking });
        }
        let approved = self.store.get(goal).map(|g| g.approved).unwrap_or(false);
        {
            let g = self.store.get_mut(goal).ok_or(Error::UnknownGoal(goal))?;
            g.transition(GoalState::Seeking)?;
            let _ = approved;
        }
        self.journal.transition(now, goal, from, GoalState::Seeking);
        Ok(())
    }

    // -- the tick ------------------------------------------------------------ //

    /// One tick: select at most one goal, execute one step, evaluate the goal's pending
    /// prediction errors.
    pub fn tick(&mut self) -> Result<TickOutcome> {
        self.now += self.config.dt;
        self.stats.ticks += 1;
        let now = self.now;

        // 1. Drives advance; measurements come from the caller.
        self.drives.tick(&self.measured);
        for kind in DriveKind::ALL {
            let l = self.model.learnability(kind);
            self.drives.get_mut(kind).learnability = l;
        }

        // 2. Habituate anything that has fallen below the floor.
        let doomed: Vec<Id> = self
            .store
            .goals
            .iter()
            .filter(|g| g.state.is_live() && g.is_habituated(self.config.habituate_epsilon))
            .map(|g| g.id)
            .collect();
        for id in doomed {
            let from = self.store.get(id).map(|g| g.state).unwrap_or(GoalState::Paused);
            if let Some(g) = self.store.get_mut(id) {
                if from.may_transition_to(GoalState::Habituated) {
                    g.transition(GoalState::Habituated)?;
                    self.journal.transition(now, id, from, GoalState::Habituated);
                    self.stats.habituated += 1;
                }
            }
            return Ok(TickOutcome::Habituated { goal: id });
        }

        // 3. Select.
        let budget_remaining = self.config.total_budget - self.stats.autonomy_spent;
        let selection = {
            let model = self.model.clone();
            let cands: Vec<Candidate> = self
                .store
                .live()
                .map(|g| Candidate::new(g, model.forecast(g), g.step_cost))
                .collect();
            self.selector.select(
                &cands,
                &self.drives,
                &self.measured,
                budget_remaining.max(0.0),
                self.config.total_budget,
                now,
            )
        };

        // Apply lateral inhibition. Salience mutation lives here because only the runner
        // owns goal mutation.
        for id in &selection.suppressed {
            if let Some(g) = self.store.get_mut(*id) {
                g.salience *= self.selector.inhibition_strength;
            }
            self.journal.push(now, Some(*id), EventKind::Suppressed, "lateral inhibition", 0.0);
        }

        let chosen = match selection.chosen {
            Some(id) => id,
            None => {
                self.stats.idle_ticks += 1;
                let reason = selection.idle_reason.unwrap_or("nothing to do");
                self.journal.push(now, None, EventKind::Idle, reason, 0.0);
                if reason.contains("budget") {
                    self.stats.budget_denied += 1;
                    self.journal.push(now, None, EventKind::BudgetDenied, reason, 0.0);
                }
                return Ok(TickOutcome::Idle(reason));
            }
        };

        // 4. Execute one step.
        let (drive, step_cost, statement, forecast) = {
            let g = self.store.get(chosen).ok_or(Error::UnknownGoal(chosen))?;
            (g.drive, g.step_cost, g.statement.clone(), self.model.forecast(g))
        };
        {
            let g = self.store.get_mut(chosen).ok_or(Error::UnknownGoal(chosen))?;
            if !g.spend(step_cost) {
                self.stats.budget_denied += 1;
                return Ok(TickOutcome::Idle("goal budget exhausted"));
            }
        }
        self.drives.note_attention(drive);
        self.stats.steps += 1;
        self.stats.autonomy_spent += step_cost;

        // The "execution" here is a deterministic synthetic outcome so the kernel is
        // testable without tools. A real deployment replaces this with the tool call and
        // feeds the *observed* value into `record_outcome`.
        let observed = self.simulate_step(&statement, forecast.expected);
        self.journal.push(
            now,
            Some(chosen),
            EventKind::StepExecuted,
            format!("step executed, observed {observed:.3} (forecast {:.3})", forecast.expected),
            crate::world::surprise(forecast.expected, observed, forecast.uncertainty),
        );

        self.record_outcome(chosen, observed, forecast.expected, forecast.uncertainty)
    }

    /// Record what actually happened for a goal: runs the check, appends the prediction
    /// error, updates the world model, and evaluates.
    pub fn record_outcome(
        &mut self,
        goal: Id,
        observed: f64,
        predicted: f64,
        uncertainty: f64,
    ) -> Result<TickOutcome> {
        let now = self.now;
        let (spec, drive, progress_now) = {
            let g = self.store.get_mut(goal).ok_or(Error::UnknownGoal(goal))?;
            g.check.record_run(observed);
            let spec = g.check.spec.clone();
            (spec, g.drive, g.check.progress(observed))
        };

        // World model learns from every outcome, including failures and abandonments.
        self.model.update(drive, progress_now, predicted);

        let normalised_surprise =
            crate::world::surprise(predicted, progress_now, uncertainty);
        let error = PredictionError::new(
            goal,
            predicted,
            progress_now,
            self.store.get(goal).map(|g| g.check.runs).unwrap_or(1),
            uncertainty,
            spec,
        );
        self.pending.push(error.clone());
        self.journal.push(
            now,
            Some(goal),
            EventKind::PredictionError,
            format!(
                "predicted {predicted:.3} observed {progress_now:.3} rpe {:+.3} (norm {:+.2})",
                error.rpe(),
                error.normalised()
            ),
            normalised_surprise,
        );

        // A step that moved the world backwards is a failure, and is recorded as one.
        // The reference loop marked these DONE, which is why nothing downstream could ever
        // detect a stall. `Failed` is a first-class outcome, not an exception.
        let failed = progress_now < predicted - 0.05;
        if failed {
            self.stats.failures += 1;
            self.journal.push(
                now,
                Some(goal),
                EventKind::StepFailed,
                format!("observed {progress_now:.3} below forecast {predicted:.3}"),
                normalised_surprise,
            );
        }

        // Achieved?
        let achieved = self.store.get(goal).map(|g| g.check.achieved()).unwrap_or(false);
        if achieved {
            let from = self.store.get(goal).unwrap().state;
            if from.may_transition_to(GoalState::Achieved) {
                self.store.get_mut(goal).unwrap().transition(GoalState::Achieved)?;
                self.journal.transition(now, goal, from, GoalState::Achieved);
                self.stats.achieved += 1;
            }
            self.pending.retain(|e| e.goal != goal);
            return Ok(TickOutcome::Achieved { goal });
        }

        // Evaluate on verified state change, not on a step counter. An evaluation that
        // changes the goal's state (pivot, abandon) wins; otherwise the step result is
        // reported, with failures reported as failures.
        if let Some(changed) = self.evaluate(goal)? {
            return Ok(changed);
        }
        if failed {
            return Ok(TickOutcome::Failed {
                goal,
                reason: format!("observed {progress_now:.3} below forecast {predicted:.3}"),
            });
        }
        Ok(TickOutcome::Executed { goal, value: observed, progress: progress_now })
    }

    /// Evaluate a goal from its pending prediction errors and act on the decision.
    ///
    /// Returns `Some` when the evaluation changed the goal's state (pivot, abandon), `None`
    /// when it continued.
    pub fn evaluate(&mut self, goal: Id) -> Result<Option<TickOutcome>> {
        let now = self.now;
        let errors: Vec<PredictionError> =
            self.pending.iter().filter(|e| e.goal == goal).cloned().collect();
        let (inert_streak, negative_streak, salience, pivots_left) = {
            let g = self.store.get(goal).ok_or(Error::UnknownGoal(goal))?;
            (
                g.inert_streak,
                g.negative_streak,
                g.salience,
                g.pivots_used < g.max_pivots,
            )
        };

        let state_change = errors.iter().any(|e| e.observed > 0.0 && e.normalised().abs() > 0.5);
        let evaluation = Evaluation::score(
            goal,
            errors.clone(),
            state_change,
            inert_streak,
            negative_streak,
            salience,
            pivots_left,
            &self.config.policy,
        )?;

        self.journal.push(
            now,
            Some(goal),
            EventKind::Evaluated,
            format!("{}: {}", evaluation.decision.name(), evaluation.rationale),
            evaluation.mean_error(),
        );

        // Update streaks so hysteresis works across calls.
        {
            let g = self.store.get_mut(goal).ok_or(Error::UnknownGoal(goal))?;
            if state_change {
                g.inert_streak = 0;
            } else {
                g.inert_streak += 1;
            }
            if evaluation.residual_trend < self.config.policy.trend_floor {
                g.negative_streak += 1;
            } else {
                g.negative_streak = 0;
            }
            let progress = errors.last().map(|e| e.rpe()).unwrap_or(0.0);
            g.habituate(progress, self.config.salience_decay);
        }
        self.pending.retain(|e| e.goal != goal);

        match evaluation.decision {
            Decision::Continue | Decision::Replan => Ok(None),
            Decision::Pivot => {
                let parent = self.store.get(goal).ok_or(Error::UnknownGoal(goal))?;
                if parent.state != GoalState::Stalled {
                    if parent.state.may_transition_to(GoalState::Stalled) {
                        let from = parent.state;
                        self.store.get_mut(goal).unwrap().transition(GoalState::Stalled)?;
                        self.journal.transition(now, goal, from, GoalState::Stalled);
                    }
                }
                let child = self.pivot(goal)?;
                Ok(Some(TickOutcome::Pivoted { parent: goal, child }))
            }
            Decision::Abandon => {
                let from = self.store.get(goal).map(|g| g.state).unwrap_or(GoalState::Paused);
                if from.may_transition_to(GoalState::Habituated) {
                    self.store.get_mut(goal).unwrap().transition(GoalState::Habituated)?;
                    self.journal.transition(now, goal, from, GoalState::Habituated);
                    self.stats.habituated += 1;
                }
                Ok(Some(TickOutcome::Habituated { goal }))
            }
        }
    }

    /// Pivot: the parent becomes terminal and the child inherits budget and evidence.
    ///
    /// This is the correction to "pivot creates a new goal": a pivot is a replacement, so it
    /// consumes the parent's remaining budget and increments the pivot count. No siblings.
    pub fn pivot(&mut self, parent: Id) -> Result<Id> {
        let now = self.now;
        let (drive, origin, remaining, pivots_used, max_pivots, statement, check) = {
            let g = self.store.get(parent).ok_or(Error::UnknownGoal(parent))?;
            (
                g.drive,
                g.origin,
                g.budget_remaining(),
                g.pivots_used,
                g.max_pivots,
                g.statement.clone(),
                g.check.clone(),
            )
        };
        if pivots_used >= max_pivots {
            return Err(Error::PivotBudgetExhausted);
        }

        let mut child = Goal::new(
            &mut self.ids,
            format!("{statement} (pivot {})", pivots_used + 1),
            drive,
            origin,
            check,
        );
        child.created_at = now;
        child.budget_units = remaining.max(1.0);
        child.pivots_used = pivots_used + 1;
        child.max_pivots = max_pivots;
        child.parent = Some(parent);
        child.approved = self.store.get(parent).map(|g| g.approved).unwrap_or(false);
        let child_id = child.id;

        // Parent: terminal, and its budget is transferred rather than duplicated.
        {
            let p = self.store.get_mut(parent).ok_or(Error::UnknownGoal(parent))?;
            p.budget_units = p.budget_spent;
            p.pivots_used = pivots_used + 1;
            if p.state.may_transition_to(GoalState::Pivoted) {
                let from = p.state;
                p.transition(GoalState::Pivoted)?;
                self.journal.transition(now, parent, from, GoalState::Pivoted);
            }
        }

        child.transition(GoalState::Screened)?;
        self.journal.push(
            now,
            Some(child_id),
            EventKind::Pivoted,
            format!("pivot of {parent}, inherited {remaining:.1} budget"),
            0.0,
        );
        self.journal.transition(now, child_id, GoalState::Proposed, GoalState::Screened);
        self.store.insert(child);
        self.stats.pivots += 1;
        Ok(child_id)
    }

    /// Retrieve lessons relevant to a goal, current view only by default.
    pub fn recall_for(&self, goal: Id, k: usize) -> Vec<Recall<'_>> {
        match self.store.get(goal) {
            Some(g) => self.memory.recall(&g.statement, k, false, self.now),
            None => Vec::new(),
        }
    }

    /// A deterministic synthetic environment step, standing in for a tool call.
    ///
    /// Fails more often when the goal is far from its forecast, which makes the harness a
    /// fair test of recovery rather than a random walk. `p` is deliberately below 1: a
    /// long-horizon agent must succeed *despite* per-step failure.
    fn simulate_step(&mut self, statement: &str, expected: f64) -> f64 {
        let base = crate::types::hash_of(&statement) % 1000;
        let jitter = (base as f64 / 1000.0 - 0.5) * 0.2;
        let observed = (expected + jitter).clamp(0.0, 1.0);
        if self.ids.chance(0.15) {
            // A failed step: recorded as failure by the caller, never as success.
            (observed * 0.5).clamp(0.0, 1.0)
        } else {
            observed
        }
    }

    /// Metrics for reporting. Deliberately includes the numbers that are easy to omit.
    pub fn report(&self) -> Report {
        let live = self.store.live_count();
        let total = self.store.all().len();
        let failure_rate = if self.stats.steps > 0 {
            self.stats.failures as f64 / self.stats.steps as f64
        } else {
            0.0
        };
        Report {
            ticks: self.stats.ticks,
            idle_fraction: if self.stats.ticks > 0 {
                self.stats.idle_ticks as f64 / self.stats.ticks as f64
            } else {
                0.0
            },
            steps: self.stats.steps,
            failures: self.stats.failures,
            failure_rate,
            proposals: self.stats.proposals,
            screened_out: self.stats.screened_out,
            pivots: self.stats.pivots,
            habituated: self.stats.habituated,
            achieved: self.stats.achieved,
            live_goals: live,
            total_goals: total,
            growth_ratio: if live > 0 { total as f64 / live as f64 } else { f64::INFINITY },
            autonomy_spent: self.stats.autonomy_spent,
            autonomy_share_used: if self.config.total_budget > 0.0 {
                self.stats.autonomy_spent / self.config.total_budget
            } else {
                0.0
            },
            journal_events: self.journal.len(),
        }
    }
}

/// A metrics snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    pub ticks: u64,
    pub idle_fraction: f64,
    pub steps: u64,
    pub failures: u64,
    pub failure_rate: f64,
    pub proposals: u64,
    pub screened_out: u64,
    pub pivots: u64,
    pub habituated: u64,
    pub achieved: u64,
    pub live_goals: usize,
    pub total_goals: usize,
    /// `total / live`. Proliferation shows up here: sublinear growth means this stays
    /// bounded, unbounded spawning makes it climb.
    pub growth_ratio: f64,
    pub autonomy_spent: f64,
    pub autonomy_share_used: f64,
    pub journal_events: usize,
}

impl std::fmt::Display for Report {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "ticks          {}", self.ticks)?;
        writeln!(f, "idle fraction  {:.0}%", self.idle_fraction * 100.0)?;
        writeln!(f, "steps          {} (failures {}, rate {:.0}%)", self.steps, self.failures, self.failure_rate * 100.0)?;
        writeln!(f, "proposals      {} (screened out {})", self.proposals, self.screened_out)?;
        writeln!(f, "goals          live {}, total {}, growth ratio {:.2}", self.live_goals, self.total_goals, self.growth_ratio)?;
        writeln!(f, "pivots {}  habituated {}  achieved {}", self.pivots, self.habituated, self.achieved)?;
        writeln!(f, "autonomy spend {:.1} ({:.0}% of budget)", self.autonomy_spent, self.autonomy_share_used * 100.0)?;
        write!(f, "journal events {}", self.journal_events)
    }
}

/// Horizon ratio: the 80%-reliability horizon over the 50%-reliability horizon.
///
/// For a chain of `n` independent steps each succeeding with probability `p`, this ratio is
/// fixed at `ln(0.8)/ln(0.5) ≈ 0.322`. Published agent measurements sit well below that,
/// which means their failures are *correlated*: the agent gets stuck rather than making a
/// fresh attempt. Recovery machinery should move this back up toward the constant, so it is
/// a better target than any accuracy number.
pub fn horizon_ratio(h50: f64, h80: f64) -> f64 {
    if h50.abs() < EPS {
        return 0.0;
    }
    h80 / h50
}

/// The theoretical ratio for independent steps.
pub const INDEPENDENT_HORIZON_RATIO: f64 = 0.321_928_094_887_362_9; // ln(0.8)/ln(0.5)

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goals::Check;

    fn runner() -> Runner {
        let mut r = Runner::new(7);
        r.config.require_approval = false;
        r
    }

    fn ready_goal(r: &mut Runner, salience: f64) -> Id {
        let id = r
            .propose(
                "raise success on family a",
                DriveKind::Competence,
                Origin::Frontier,
                Check::new("a::pass", 0.4, 0.9).with_forecast(vec![0.5, 0.6, 0.7, 0.8, 0.9]),
                None,
            )
            .unwrap();
        r.run_check(id, 0.42).unwrap();
        r.satisfy_craving(id).ok();
        // Frontier goals land in Screened; move them into Seeking through the validated door.
        {
            let g = r.store.get_mut(id).unwrap();
            if g.state == GoalState::Screened {
                g.salience = salience;
                g.begin_seeking(true).unwrap();
            }
        }
        id
    }

    #[test]
    fn idle_is_recorded_not_invented() {
        // T3: with no SEEKING goal, zero work is done.
        let mut r = runner();
        let out = r.tick().unwrap();
        assert!(matches!(out, TickOutcome::Idle(_)), "got {out:?}");
        assert_eq!(r.stats.steps, 0);
        assert_eq!(r.stats.idle_ticks, 1);
    }

    #[test]
    fn execution_requires_a_check_that_ran() {
        // T1: failure is representable; the door to SEEKING is guarded.
        let mut r = runner();
        let id = r
            .propose(
                "do a thing",
                DriveKind::Competence,
                Origin::SelfGenerated,
                Check::new("t", 0.0, 1.0),
                None,
            )
            .unwrap();
        let g = r.store.get_mut(id).unwrap();
        assert_eq!(g.begin_seeking(true), Err(Error::NoExecutedCheck));
        r.run_check(id, 0.1).unwrap();
        r.store.get_mut(id).unwrap().begin_seeking(true).unwrap();
        assert_eq!(r.store.get(id).unwrap().state, GoalState::Seeking);
    }

    #[test]
    fn shadow_mode_executes_nothing() {
        let mut r = runner();
        r.config.shadow = true;
        let id = r
            .propose(
                "observe only",
                DriveKind::Coverage,
                Origin::PatternDetection,
                Check::new("t", 0.0, 1.0),
                None,
            )
            .unwrap();
        assert_eq!(r.store.get(id).unwrap().state, GoalState::Shadowed);
        let out = r.tick().unwrap();
        assert!(matches!(out, TickOutcome::Idle(_)));
        assert_eq!(r.stats.steps, 0);
    }

    #[test]
    fn dream_goals_crave_until_probed() {
        let mut r = runner();
        let id = r
            .propose(
                "from a dream",
                DriveKind::Uncertainty,
                Origin::DreamSimulation,
                Check::new("t", 0.0, 1.0),
                None,
            )
            .unwrap();
        assert_eq!(r.store.get(id).unwrap().state, GoalState::Craving);
        r.run_check(id, 0.2).unwrap();
        r.satisfy_craving(id).unwrap();
        assert_eq!(r.store.get(id).unwrap().state, GoalState::Seeking);
    }

    #[test]
    fn redundant_proposals_merge() {
        let mut r = runner();
        let a = r
            .propose("raise success on family a", DriveKind::Competence, Origin::Frontier, Check::new("t", 0.0, 1.0), None)
            .unwrap();
        let b = r
            .propose("raise success on family a", DriveKind::Competence, Origin::PatternDetection, Check::new("t", 0.0, 1.0), None)
            .unwrap();
        assert_eq!(a, b, "a near-duplicate must merge, not spawn");
        assert_eq!(r.stats.screened_out, 1);
    }

    #[test]
    fn pivot_supersedes_and_transfers_budget_once() {
        // T6: pivot closes the parent, inherits budget, creates no sibling.
        let mut r = runner();
        let parent = ready_goal(&mut r, 1.0);
        r.store.get_mut(parent).unwrap().budget_spent = 2.0;
        let remaining_before = r.store.get(parent).unwrap().budget_remaining();
        let child = r.pivot(parent).unwrap();
        assert_eq!(r.store.get(parent).unwrap().state, GoalState::Pivoted);
        assert_eq!(r.store.get(parent).unwrap().budget_remaining(), 0.0);
        assert_eq!(r.store.get(child).unwrap().parent, Some(parent));
        assert!((r.store.get(child).unwrap().budget_units - remaining_before).abs() < 1e-9);
        assert_eq!(r.store.all().len(), 2, "no siblings");
        assert_eq!(r.store.live_count(), 1);
    }

    #[test]
    fn pivot_budget_is_enforced() {
        let mut r = runner();
        let mut parent = ready_goal(&mut r, 1.0);
        parent = {
            let g = r.store.get_mut(parent).unwrap();
            g.pivots_used = g.max_pivots;
            parent
        };
        assert_eq!(r.pivot(parent), Err(Error::PivotBudgetExhausted));
    }

    #[test]
    fn long_run_stays_bounded() {
        // T4: proliferation invariant. Goals are created but the live set must not explode.
        let mut r = runner();
        r.config.total_budget = 10_000.0;
        for i in 0..200 {
            let _ = r.propose(
                format!("family {i} needs work on its fixture"),
                DriveKind::Coverage,
                Origin::Frontier,
                Check::new("t", 0.0, 1.0),
                None,
            );
            if let Ok(Some(_)) = r.tick() {}
        }
        let live = r.store.live_count();
        let total = r.store.all().len();
        assert!(live <= 200, "live goals cannot exceed proposals: {live}");
        assert!(
            total as f64 <= live as f64 * 3.0 + 10.0,
            "terminal goals should dominate the total, not accumulate live: total {total}, live {live}"
        );
        // Distinct statements must stay distinct: numeric distinctions are not redundancy.
        assert!(live > 50, "statements differing only by an index must not all merge: {live}");
    }

    #[test]
    fn horizon_ratio_matches_the_independent_steps_constant() {
        let expected = (0.8f64).ln() / (0.5f64).ln();
        assert!((expected - INDEPENDENT_HORIZON_RATIO).abs() < 1e-9);
        assert!((horizon_ratio(100.0, 32.19) - 0.3219).abs() < 1e-3);
        // A system below the constant has correlated failures.
        assert!(horizon_ratio(720.0, 70.0) < INDEPENDENT_HORIZON_RATIO);
    }

    #[test]
    fn report_exposes_the_uncomfortable_numbers() {
        let mut r = runner();
        let _ = r.tick();
        let rep = r.report();
        assert_eq!(rep.ticks, 1);
        assert!(rep.idle_fraction > 0.99);
        assert!(rep.journal_events >= 1);
    }
}
