//! # agi-core
//!
//! An autonomy kernel in the sense of `docs/research/brain-grounded-autonomy.md`:
//!
//! * **Drives** are homeostatic deficiencies with setpoints, never rewards to maximise.
//! * **Goals are selected**, not generated: candidates come from (drive x affordance x
//!   forecast), and selection may legitimately return `None`.
//! * **Reward prediction error** is the only progress signal. An evaluation that cites no
//!   executed check is invalid and is discarded, never logged as zero progress.
//! * **Goals die by habituation**, not by TTL or a `COMPLETED` flag.
//! * **Pivots supersede**, they do not spawn siblings.
//! * **Memory revises**: contradicting advice supersedes and *lowers* confidence in the
//!   old advice instead of merging with it.
//!
//! The crate is std-only and deterministic. `SystemTime` never enters the logic — time is
//! passed in as seconds by the caller, which is what makes the tests reproducible.
//!
//! ## Layout
//!
//! | module | contents |
//! |---|---|
//! | [`types`]   | `Id`, `IdGen`, deterministic RNG, small math helpers |
//! | [`drives`]  | `Drive`, `DriveSet`, urgency with satiation |
//! | [`goals`]   | `Goal`, `GoalState`, `Origin`, `Check`, transitions, habituation |
//! | [`selection`] | arbitration: winner-take-all with lateral inhibition |
//! | [`eval`]    | `PredictionError`, `Evaluation`, validity rule, decision rule |
//! | [`memory`]  | lessons, supersession (revision), relevance-based forgetting |
//! | [`world`]   | calibrated predictive model, updated by every prediction error |
//! | [`journal`] | append-only audit + replay (reverse, surprise-prioritised) |
//! | [`runner`]  | the loop that ties it together |
//!
//! Run `cargo test` for the invariant suite (T1–T9 of the brief) and
//! `cargo run --example demo` for an end-to-end trace.

pub mod drives;
pub mod eval;
pub mod goals;
pub mod journal;
pub mod memory;
pub mod runner;
pub mod selection;
pub mod types;
pub mod world;

pub use drives::{Drive, DriveSet};
pub use eval::{Decision, Evaluation, PredictionError};
pub use goals::{Check, Goal, GoalState, Origin};
pub use journal::{Event, Journal};
pub use memory::{Lesson, Memory, Recall};
pub use runner::{Runner, TickOutcome};
pub use selection::{Candidate, Selector};
pub use types::{Id, IdGen};
pub use world::{Forecast, PredictiveModel};

/// Errors that the kernel reports to its caller rather than swallowing.
///
/// The reference Python skeleton this crate replaces marked failed steps as `DONE`, which
/// made failure unrepresentable and every downstream metric dishonest. Every fallible
/// operation here returns `Err` with a reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A transition the state machine does not allow, e.g. `SEEKING -> SEEKING`.
    IllegalTransition { from: GoalState, to: GoalState },
    /// A goal was pushed to `SEEKING` without a check that has executed.
    NoExecutedCheck,
    /// An evaluation cited no prediction errors; it cannot be scored.
    UnbackedEvaluation,
    /// A pivot was requested after the goal's pivot budget was exhausted.
    PivotBudgetExhausted,
    /// The caller asked for a goal id that does not exist.
    UnknownGoal(Id),
    /// Budget for autonomous work is exhausted.
    BudgetExhausted,
    /// A check's TTL expired; the check must be re-authored, not the goal killed.
    CheckExpired,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::IllegalTransition { from, to } => {
                write!(f, "illegal transition {from:?} -> {to:?}")
            }
            Error::NoExecutedCheck => write!(
                f,
                "goal cannot execute: no check has run (verification is the bottleneck)"
            ),
            Error::UnbackedEvaluation => write!(
                f,
                "evaluation cites no prediction error and is discarded, not logged as zero progress"
            ),
            Error::PivotBudgetExhausted => write!(f, "pivot budget exhausted: supersede or habituate"),
            Error::UnknownGoal(id) => write!(f, "unknown goal {id}"),
            Error::BudgetExhausted => write!(f, "autonomy budget exhausted"),
            Error::CheckExpired => write!(f, "check ttl expired"),
        }
    }
}

impl std::error::Error for Error {}

/// Every fallible operation in the kernel.
pub type Result<T> = std::result::Result<T, Error>;
