//! Append-only audit and replay.
//!
//! Two jobs. First, **audit**: every transition, check, evaluation and memory revision is
//! recorded with a monotonic sequence number, so provenance is written by the runtime and
//! cannot be back-dated by a model. Second, **replay**: the credit-assignment mechanism.
//!
//! Replay here is reverse and surprise-prioritised — replay backwards from the outcome,
//! ordered by how wrong the model was. Reverse order is what assigns credit to the early
//! steps that made the outcome happen, which is the cheap answer to long-horizon credit
//! assignment that needs no model at all.

use crate::goals::GoalState;
use crate::types::{hash_of, Id};

/// What happened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    Proposed,
    Screened,
    Shadowed,
    Transition,
    StepExecuted,
    StepFailed,
    CheckRan,
    Evaluated,
    PredictionError,
    Pivoted,
    MemoryRevised,
    Suppressed,
    BudgetDenied,
    Idle,
}

impl EventKind {
    pub fn name(self) -> &'static str {
        match self {
            EventKind::Proposed => "proposed",
            EventKind::Screened => "screened",
            EventKind::Shadowed => "shadowed",
            EventKind::Transition => "transition",
            EventKind::StepExecuted => "step",
            EventKind::StepFailed => "step_failed",
            EventKind::CheckRan => "check",
            EventKind::Evaluated => "evaluated",
            EventKind::PredictionError => "rpe",
            EventKind::Pivoted => "pivoted",
            EventKind::MemoryRevised => "revised",
            EventKind::Suppressed => "suppressed",
            EventKind::BudgetDenied => "budget_denied",
            EventKind::Idle => "idle",
        }
    }
}

/// One journal entry.
#[derive(Debug, Clone)]
pub struct Event {
    pub seq: u64,
    pub at: f64,
    pub goal: Option<Id>,
    pub kind: EventKind,
    pub detail: String,
    /// How surprising this event was, in `[0, 10]`. Drives replay priority.
    pub surprise: f64,
}

/// The journal.
#[derive(Debug, Clone, Default)]
pub struct Journal {
    events: Vec<Event>,
}

impl Journal {
    pub fn new() -> Self {
        Journal::default()
    }

    pub fn push(
        &mut self,
        at: f64,
        goal: Option<Id>,
        kind: EventKind,
        detail: impl Into<String>,
        surprise: f64,
    ) -> u64 {
        let seq = self.events.len() as u64;
        self.events.push(Event {
            seq,
            at,
            goal,
            kind,
            detail: detail.into(),
            surprise: surprise.max(0.0),
        });
        seq
    }

    /// Convenience for state changes, which are the bulk of the traffic.
    pub fn transition(&mut self, at: f64, goal: Id, from: GoalState, to: GoalState) -> u64 {
        self.push(
            at,
            Some(goal),
            EventKind::Transition,
            format!("{:?} -> {:?}", from, to),
            0.0,
        )
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Event> {
        self.events.iter()
    }

    pub fn for_goal(&self, goal: Id) -> impl Iterator<Item = &Event> {
        self.events.iter().filter(move |e| e.goal == Some(goal))
    }

    pub fn of_kind(&self, kind: EventKind) -> impl Iterator<Item = &Event> {
        self.events.iter().filter(move |e| e.kind == kind)
    }

    /// Replay for credit assignment: most surprising first, ties broken by later events
    /// first (reverse time).
    ///
    /// This is deliberate: what you replay is what surprised you, and you replay it
    /// backwards from the outcome.
    pub fn replay(&self, goal: Id, limit: usize) -> Vec<&Event> {
        let mut picked: Vec<&Event> = self.for_goal(goal).collect();
        picked.sort_by(|a, b| {
            b.surprise
                .partial_cmp(&a.surprise)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.seq.cmp(&a.seq))
        });
        picked.truncate(limit);
        picked
    }

    /// The state path a goal took, in order. Useful for asserting "this goal never executed".
    pub fn state_path(&self, goal: Id) -> Vec<GoalState> {
        let mut path = Vec::new();
        for e in self.for_goal(goal) {
            if e.kind == EventKind::Transition {
                if let Some((_, to)) = e.detail.split_once("-> ") {
                    path.push(match to.trim() {
                        "Proposed" => GoalState::Proposed,
                        "Screened" => GoalState::Screened,
                        "Shadowed" => GoalState::Shadowed,
                        "Seeking" => GoalState::Seeking,
                        "Craving" => GoalState::Craving,
                        "Paused" => GoalState::Paused,
                        "Stalled" => GoalState::Stalled,
                        "Pivoted" => GoalState::Pivoted,
                        "Habituated" => GoalState::Habituated,
                        "Achieved" => GoalState::Achieved,
                        "Rejected" => GoalState::Rejected,
                        _ => continue,
                    });
                }
            }
        }
        path
    }

    /// Whether a goal ever executed. The T3 assertion in one call.
    pub fn ever_executed(&self, goal: Id) -> bool {
        self.state_path(goal).contains(&GoalState::Seeking)
    }

    /// Stable digest of the whole record: provenance you can compare across runs.
    pub fn digest(&self) -> u64 {
        let tuples: Vec<(u64, u64, &str)> = self
            .events
            .iter()
            .map(|e| (e.seq, hash_of(&e.kind.name()), e.detail.as_str()))
            .collect();
        hash_of(&tuples)
    }

    /// JSONL export. Hand-rolled because the crate has no dependencies.
    pub fn to_jsonl(&self) -> String {
        let mut out = String::new();
        for e in &self.events {
            out.push_str(&format!(
                "{{\"seq\":{},\"at\":{:.3},\"goal\":{},\"kind\":\"{}\",\"surprise\":{:.3},\"detail\":\"{}\"}}\n",
                e.seq,
                e.at,
                e.goal.map(|g| format!("\"{g}\"")).unwrap_or_else(|| "null".to_string()),
                e.kind.name(),
                e.surprise,
                escape(&e.detail)
            ));
        }
        out
    }
}

fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_is_monotonic() {
        let mut j = Journal::new();
        let a = j.push(0.0, None, EventKind::Idle, "first", 0.0);
        let b = j.push(1.0, None, EventKind::Idle, "second", 0.0);
        assert!(b > a);
        assert_eq!(j.len(), 2);
    }

    #[test]
    fn replay_is_reverse_and_surprise_ordered() {
        let mut j = Journal::new();
        let g = Id(1);
        j.push(0.0, Some(g), EventKind::StepExecuted, "boring", 0.1);
        j.push(1.0, Some(g), EventKind::StepExecuted, "surprising", 5.0);
        j.push(2.0, Some(g), EventKind::StepExecuted, "also surprising", 5.0);
        let order = j.replay(g, 3);
        assert_eq!(order[0].detail, "also surprising", "ties break toward the later event");
        assert_eq!(order[1].detail, "surprising");
        assert_eq!(order[2].detail, "boring");
    }

    #[test]
    fn state_path_reads_transitions() {
        let mut j = Journal::new();
        let g = Id(2);
        j.transition(0.0, g, GoalState::Proposed, GoalState::Screened);
        j.transition(1.0, g, GoalState::Screened, GoalState::Craving);
        assert_eq!(
            j.state_path(g),
            vec![GoalState::Screened, GoalState::Craving]
        );
        assert!(!j.ever_executed(g));
    }

    #[test]
    fn jsonl_escapes_quotes_and_newlines() {
        let mut j = Journal::new();
        j.push(0.0, None, EventKind::Idle, "a \"quoted\"\nline", 0.0);
        let text = j.to_jsonl();
        assert!(text.contains("\\\"quoted\\\""));
        assert!(text.contains("\\n"));
        assert!(!text.trim_end().contains('\n'));
    }

    #[test]
    fn digest_is_stable_and_sensitive() {
        let mut a = Journal::new();
        let mut b = Journal::new();
        a.push(0.0, None, EventKind::Idle, "same", 0.0);
        b.push(0.0, None, EventKind::Idle, "same", 0.0);
        assert_eq!(a.digest(), b.digest());
        b.push(1.0, None, EventKind::Idle, "different", 0.0);
        assert_ne!(a.digest(), b.digest());
    }
}
