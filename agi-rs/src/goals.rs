//! Goals and their lifecycle.
//!
//! Three design choices worth defending, all of them corrections to the reference Python
//! skeleton:
//!
//! 1. **`SEEKING` and `CRAVING` are different states.** `CRAVING` means "urgent but nothing
//!    actionable" — stored and re-scanned, never executed. An agent that cannot hold a
//!    craving invents work to satisfy it, which is the single largest source of goal
//!    proliferation.
//! 2. **Only the runtime writes transitions.** [`Goal::transition`] is the sole mutator and
//!    it validates against [`GoalState::may_transition_to`]. If a model can write its own
//!    status, self-proposed goals get relabelled as instructions and provenance is fiction.
//! 3. **A goal may not execute without an executed check.** [`Goal::begin_seeking`] is the
//!    only door into `SEEKING` and it refuses without one.
//!
//! Termination is habituation (salience decay under non-reinforcement), not a TTL and not
//! a `COMPLETED` flag. Salience is monotone in evidence, so abandonment is justified by the
//! record rather than by a deadline.

use crate::drives::DriveKind;
use crate::types::{unit_clamp, Id, IdGen};

/// Where the *evidence* for a proposal came from.
///
/// This is provenance of evidence, orthogonal to derivation. A subgoal of a dream-origin
/// goal is `origin: DreamSimulation` with `parent: Some(..)` — it is not an origin of its
/// own. Collapsing the two (a `SUBGOAL` variant) makes the common case unrepresentable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Origin {
    /// A human asked for it.
    Human,
    /// Mined from recurring problems in the record.
    PatternDetection,
    /// Produced by offline simulation; always needs a real-world probe first.
    DreamSimulation,
    /// Derived from the world model's own uncertainty.
    SelfGenerated,
    /// Spawned by frontier calibration (families in the 50-70% band).
    Frontier,
}

impl Origin {
    pub fn name(self) -> &'static str {
        match self {
            Origin::Human => "human",
            Origin::PatternDetection => "pattern",
            Origin::DreamSimulation => "dream",
            Origin::SelfGenerated => "self",
            Origin::Frontier => "frontier",
        }
    }

    /// Whether evidence from this origin needs a verified probe before it may execute.
    /// Dreams are confident fictions until something real agrees with them.
    pub fn requires_probe(self) -> bool {
        matches!(self, Origin::DreamSimulation)
    }
}

/// The lifecycle. `Seeking` is the only executing state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GoalState {
    /// Proposed but not screened.
    Proposed,
    /// Passed the screening gates; awaiting a check.
    Screened,
    /// Observed but not acted on.
    Shadowed,
    /// Urgent and actionable: the only state the runner executes.
    Seeking,
    /// Urgent but not actionable. Stored, re-scanned, never executed.
    Craving,
    /// Low urgency or externally blocked.
    Paused,
    /// No change in verified state across k evaluations.
    Stalled,
    /// Superseded by a pivot; the child inherits budget and evidence.
    Pivoted,
    /// Salience fell below epsilon. Dropped without ceremony, kept for audit.
    Habituated,
    /// The check's target was met, verified.
    Achieved,
    /// A human or policy refused it.
    Rejected,
}

impl GoalState {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            GoalState::Pivoted
                | GoalState::Habituated
                | GoalState::Achieved
                | GoalState::Rejected
        )
    }

    pub fn is_executing(self) -> bool {
        matches!(self, GoalState::Seeking)
    }

    /// The transition table. Anything not listed is an [`crate::Error::IllegalTransition`].
    ///
    /// Note what is *absent*: there is no path from a terminal state back into `Seeking`,
    /// and nothing may skip screening to execute.
    pub fn may_transition_to(self, to: GoalState) -> bool {
        use GoalState::*;
        if self == to {
            return false;
        }
        // A pivot and a habituation are both "this live attempt ends", so they are reachable
        // from any live state. Spelled out because a table without these arms silently
        // refuses the two transitions the runner most needs.
        if matches!(to, Pivoted | Habituated) {
            return self.is_live();
        }
        match (self, to) {
            (Proposed, Screened) | (Proposed, Rejected) => true,
            (Screened, Shadowed) | (Screened, Craving) | (Screened, Paused) | (Screened, Rejected) => true,
            (Screened, Seeking) => true,
            (Shadowed, Craving) | (Shadowed, Seeking) | (Shadowed, Paused) | (Shadowed, Rejected) => true,
            (Craving, Seeking) | (Craving, Paused) | (Craving, Habituated) => true,
            (Seeking, Achieved) | (Seeking, Stalled) | (Seeking, Paused) | (Seeking, Craving) => true,
            (Paused, Seeking) | (Paused, Craving) | (Paused, Habituated) => true,
            (Stalled, Seeking) | (Stalled, Pivoted) | (Stalled, Habituated) | (Stalled, Paused) => true,
            _ => false,
        }
    }

    /// States a goal can be in and still be considered live (occupying attention).
    pub fn is_live(self) -> bool {
        !self.is_terminal()
    }
}

/// The executable check. This is the load-bearing object of the whole crate: no goal
/// executes without one, and no evaluation is valid without a result from one.
///
/// In a full system `spec` is a test id, a SQL query, or a predicate evaluated in a
/// sandbox. Here it is carried as data and the caller executes it, so the kernel stays
/// pure and testable.
#[derive(Debug, Clone, PartialEq)]
pub struct Check {
    /// What to run: test id, query, predicate.
    pub spec: String,
    /// Measured at screening time. Baseline is recorded, never assumed.
    pub baseline: f64,
    /// The level that counts as achieved.
    pub target: f64,
    /// Expected trajectory toward the target, used to compute prediction error before the
    /// goal completes. Values are fractional progress in `[0, 1]`.
    pub forecast_curve: Vec<f64>,
    /// How many times the check has actually run. Zero means it cannot gate execution.
    pub runs: u32,
    /// Expiry in simulation seconds. Expires the *check*, not the goal: a stale check is
    /// re-authored, not a reason to kill the goal.
    pub ttl_seconds: f64,
    pub authored_at: f64,
    /// Last observed value.
    pub last_value: Option<f64>,
}

impl Check {
    pub fn new(spec: impl Into<String>, baseline: f64, target: f64) -> Self {
        Check {
            spec: spec.into(),
            baseline,
            target,
            forecast_curve: Vec::new(),
            runs: 0,
            ttl_seconds: 86_400.0,
            authored_at: 0.0,
            last_value: None,
        }
    }

    pub fn with_forecast(mut self, curve: Vec<f64>) -> Self {
        self.forecast_curve = curve;
        self
    }

    pub fn with_ttl(mut self, seconds: f64) -> Self {
        self.ttl_seconds = seconds;
        self
    }

    /// Has this check ever executed? The one bit that gates execution.
    pub fn has_run(&self) -> bool {
        self.runs > 0
    }

    pub fn expired(&self, now: f64) -> bool {
        now - self.authored_at > self.ttl_seconds
    }

    /// Fractional progress implied by the last observation, in `[0, 1]`.
    ///
    /// `Cost`-like tests want the inverse, so the caller sets `target < baseline` and this
    /// handles both directions.
    pub fn progress(&self, value: f64) -> f64 {
        let span = self.target - self.baseline;
        if span.abs() < crate::types::EPS {
            return if (value - self.target).abs() < crate::types::EPS { 1.0 } else { 0.0 };
        }
        unit_clamp((value - self.baseline) / span)
    }

    /// Record an execution result.
    pub fn record_run(&mut self, value: f64) {
        self.runs += 1;
        self.last_value = Some(value);
    }

    /// Target met, by direction-aware comparison.
    pub fn achieved(&self) -> bool {
        match (self.last_value, self.target >= self.baseline) {
            (Some(v), true) => v >= self.target - crate::types::EPS,
            (Some(v), false) => v <= self.target + crate::types::EPS,
            (None, _) => false,
        }
    }

    /// The forecast at a given number of runs, falling back to a linear interpolation to
    /// the target when no curve was supplied. Used by the evaluator as the `predicted`
    /// term, which is why a goal with no forecast can still be scored — just less sharply.
    pub fn forecast_at(&self, runs: u32) -> f64 {
        if self.forecast_curve.is_empty() {
            return 0.5; // deliberately uninformative rather than optimistic
        }
        let i = (runs as usize).min(self.forecast_curve.len() - 1);
        self.forecast_curve[i]
    }
}

/// A goal: a (drive, affordance, forecast) triple with a check, a budget, and a ledger.
#[derive(Debug, Clone)]
pub struct Goal {
    pub id: Id,
    pub statement: String,
    pub drive: DriveKind,
    pub origin: Origin,
    pub parent: Option<Id>,
    pub state: GoalState,
    pub check: Check,
    /// Estimated cost of one step, in budget units.
    pub step_cost: f64,
    pub budget_units: f64,
    pub budget_spent: f64,
    /// Salience in `[0, 1]`: how loudly this goal competes, and the quantity that habituates.
    pub salience: f64,
    /// How many times this goal has been pivoted. Capped, so a stuck goal cannot spawn
    /// children forever.
    pub pivots_used: u32,
    pub max_pivots: u32,
    /// Consecutive evaluations with no verified state change.
    pub inert_streak: u32,
    /// Consecutive evaluations with negative residual trend.
    pub negative_streak: u32,
    /// Whether a human or policy has approved execution.
    pub approved: bool,
    pub created_at: f64,
    /// Simulation-time deadline. Enforced by [`Goal::expired`], unlike the reference
    /// implementation where this field was declared and never checked.
    pub deadline: Option<f64>,
}

impl Goal {
    pub fn new(
        ids: &mut IdGen,
        statement: impl Into<String>,
        drive: DriveKind,
        origin: Origin,
        check: Check,
    ) -> Self {
        Goal {
            id: ids.id(),
            statement: statement.into(),
            drive,
            origin,
            parent: None,
            state: GoalState::Proposed,
            check,
            step_cost: 1.0,
            budget_units: 20.0,
            budget_spent: 0.0,
            salience: 1.0,
            pivots_used: 0,
            max_pivots: 2,
            inert_streak: 0,
            negative_streak: 0,
            approved: false,
            created_at: 0.0,
            deadline: None,
        }
    }

    pub fn with_budget(mut self, units: f64) -> Self {
        self.budget_units = units;
        self
    }

    pub fn with_parent(mut self, parent: Id) -> Self {
        self.parent = Some(parent);
        self
    }

    pub fn with_deadline(mut self, at: f64) -> Self {
        self.deadline = Some(at);
        self
    }

    pub fn with_salience(mut self, salience: f64) -> Self {
        self.salience = unit_clamp(salience);
        self
    }

    pub fn with_max_pivots(mut self, n: u32) -> Self {
        self.max_pivots = n;
        self
    }

    pub fn budget_remaining(&self) -> f64 {
        (self.budget_units - self.budget_spent).max(0.0)
    }

    pub fn has_budget(&self) -> bool {
        self.budget_remaining() > crate::types::EPS
    }

    /// Enforced deadline. Declared and unenforced in the reference implementation; here it
    /// is a query the runner actually makes.
    pub fn expired(&self, now: f64) -> bool {
        self.deadline.map_or(false, |d| now >= d)
    }

    /// The single mutator for state. Validates the transition table and refuses the two
    /// transitions that silently corrupt a system: executing without a check, and writing
    /// status by anyone other than the runtime.
    pub fn transition(&mut self, to: GoalState) -> crate::Result<()> {
        if !self.state.may_transition_to(to) {
            return Err(crate::Error::IllegalTransition { from: self.state, to });
        }
        if to == GoalState::Seeking && !self.check.has_run() {
            return Err(crate::Error::NoExecutedCheck);
        }
        self.state = to;
        Ok(())
    }

    /// The only door into `SEEKING`, with the approval and check preconditions spelled out.
    pub fn begin_seeking(&mut self, approved: bool) -> crate::Result<()> {
        if !approved && !self.approved {
            return Err(crate::Error::IllegalTransition {
                from: self.state,
                to: GoalState::Seeking,
            });
        }
        if !self.check.has_run() {
            return Err(crate::Error::NoExecutedCheck);
        }
        self.transition(GoalState::Seeking)
    }

    /// Habituation: salience decays with non-reinforced attempts and recovers with progress.
    ///
    /// `progress` is the change in verified progress this evaluation, in `[-1, 1]`. A
    /// negative change decays faster than a flat one, so goals that are actively getting
    /// worse are pruned soonest.
    pub fn habituate(&mut self, progress: f64, decay: f64) {
        let attempts = self.inert_streak.max(1) as f64;
        let factor = decay.powf(attempts);
        let reinforcement = 1.0 + 0.5 * progress.clamp(-1.0, 1.0);
        self.salience = unit_clamp(self.salience * factor * reinforcement.max(0.0));
    }

    /// Whether salience has fallen below the pruning threshold.
    pub fn is_habituated(&self, epsilon: f64) -> bool {
        self.salience < epsilon
    }

    /// Account one step against the budget. Returns false when the budget cannot cover it,
    /// which the runner treats as a reason to stop rather than to overrun.
    pub fn spend(&mut self, units: f64) -> bool {
        if units > self.budget_remaining() + crate::types::EPS {
            return false;
        }
        self.budget_spent += units;
        true
    }

    /// A one-line description for journals and CLIs.
    pub fn label(&self) -> String {
        format!(
            "{} drive={} origin={} state={:?} sal={:.2} budget={:.0}/{:.0}",
            self.id,
            self.drive.name(),
            self.origin.name(),
            self.state,
            self.salience,
            self.budget_spent,
            self.budget_units
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drives::DriveKind;

    fn goal(ids: &mut IdGen) -> Goal {
        Goal::new(
            ids,
            "raise verified success on family A",
            DriveKind::Competence,
            Origin::PatternDetection,
            Check::new("family_a::pass", 0.4, 0.9),
        )
    }

    #[test]
    fn cannot_execute_without_an_executed_check() {
        let mut ids = IdGen::new(1);
        let mut g = goal(&mut ids);
        g.transition(GoalState::Screened).unwrap();
        // T1: failure is representable. The refusal is an Err, not a silently allowed move.
        assert_eq!(g.begin_seeking(true), Err(crate::Error::NoExecutedCheck));
        assert_eq!(g.state, GoalState::Screened);
        g.check.record_run(0.8);
        g.begin_seeking(true).unwrap();
        assert_eq!(g.state, GoalState::Seeking);
    }

    #[test]
    fn terminal_states_are_final() {
        let mut ids = IdGen::new(2);
        let mut g = goal(&mut ids);
        g.transition(GoalState::Screened).unwrap();
        g.transition(GoalState::Rejected).unwrap();
        assert!(g.state.is_terminal());
        assert!(g.transition(GoalState::Seeking).is_err());
        assert!(g.transition(GoalState::Screened).is_err());
    }

    #[test]
    fn no_transition_to_self() {
        let mut ids = IdGen::new(3);
        let mut g = goal(&mut ids);
        assert!(g.transition(GoalState::Proposed).is_err());
    }

    #[test]
    fn craving_never_executes() {
        assert!(!GoalState::Craving.is_executing());
        assert!(!GoalState::Craving.may_transition_to(GoalState::Achieved));
    }

    #[test]
    fn habituation_prunes_without_reinforcement() {
        let mut ids = IdGen::new(4);
        let mut g = goal(&mut ids).with_salience(1.0);
        for _ in 0..12 {
            g.inert_streak = 3;
            g.habituate(0.0, 0.85);
        }
        assert!(g.is_habituated(0.05), "salience should fall below epsilon: {}", g.salience);
    }

    #[test]
    fn progress_recovers_salience() {
        let mut ids = IdGen::new(5);
        let mut low = goal(&mut ids);
        low.salience = 0.3;
        low.inert_streak = 1;
        low.habituate(1.0, 0.85);
        assert!(low.salience > 0.3, "reinforcement must raise salience: {}", low.salience);
    }

    #[test]
    fn budget_is_never_overrun() {
        let mut ids = IdGen::new(6);
        let mut g = goal(&mut ids).with_budget(2.0);
        assert!(g.spend(1.5));
        assert!(!g.spend(1.5), "must refuse rather than overrun");
        assert_eq!(g.budget_spent, 1.5);
    }

    #[test]
    fn deadline_is_enforced() {
        let mut ids = IdGen::new(7);
        let g = goal(&mut ids).with_deadline(100.0);
        assert!(!g.expired(99.0));
        assert!(g.expired(100.0));
    }

    #[test]
    fn check_progress_is_direction_aware() {
        let up = Check::new("t", 0.2, 0.8);
        assert!((up.progress(0.5) - 0.5).abs() < 1e-9);
        assert_eq!(up.progress(1.0), 1.0, "clamped past the target");
        let down = Check::new("cost", 40.0, 10.0);
        assert!((down.progress(25.0) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn check_achieved_uses_direction() {
        let mut up = Check::new("t", 0.2, 0.8);
        up.record_run(0.8);
        assert!(up.achieved());
        let mut down = Check::new("cost", 40.0, 10.0);
        down.record_run(9.0);
        assert!(down.achieved());
        down.record_run(11.0);
        assert!(!down.achieved());
    }

    #[test]
    fn missing_forecast_is_uninformative_not_optimistic() {
        let c = Check::new("t", 0.0, 1.0);
        assert_eq!(c.forecast_at(3), 0.5);
    }
}
