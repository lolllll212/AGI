//! Evaluation: reward prediction error, and the decision rule that follows from it.
//!
//! "Progress" as a scalar asserted by a model has no referent. The measurable quantity is
//! **prediction error**: the outcome the world model forecast at commitment time, minus the
//! outcome a check actually verified. That has two measured terms, so it can be wrong.
//!
//! Three consequences fall out:
//!
//! * **Stall detection** is arithmetic: near-zero error with no verified state change means
//!   the goal is inert, however fluent the narration.
//! * **Pivot** is a decision rule, not a judgement call: a persistent negative residual
//!   trend means the model of this goal is refuted.
//! * **Learning** is the same loop: the error also updates the world model.
//!
//! And one non-obvious selection criterion: a goal whose outcome is predicted perfectly is
//! not worth pursuing. Value accrues where prediction is uncertain but improvable.
//!
//! The validity rule is the important one. **An evaluation that cites no prediction errors
//! is discarded, not logged as zero progress.** Silence must never read as success.

use crate::types::{slope, unit_clamp, Id, EPS};

/// Scalar objective discussed in the module docs: one forecast, one observation.
#[derive(Debug, Clone, PartialEq)]
pub struct PredictionError {
    pub goal: Id,
    /// Forecast at commitment time, in `[0, 1]` progress units.
    pub predicted: f64,
    /// Verified outcome afterwards, same units.
    pub observed: f64,
    /// Steps between commitment and observation.
    pub horizon: u32,
    /// The model's own uncertainty at forecast time. A large error under large uncertainty
    /// is not evidence of failure.
    pub uncertainty: f64,
    /// The check that produced `observed`. Never empty in a valid error.
    pub check_spec: String,
}

impl PredictionError {
    pub fn new(
        goal: Id,
        predicted: f64,
        observed: f64,
        horizon: u32,
        uncertainty: f64,
        check_spec: impl Into<String>,
    ) -> Self {
        PredictionError {
            goal,
            predicted: unit_clamp(predicted),
            observed: unit_clamp(observed),
            horizon,
            uncertainty: unit_clamp(uncertainty),
            check_spec: check_spec.into(),
        }
    }

    /// Signed error. Positive means better than expected.
    pub fn rpe(&self) -> f64 {
        self.observed - self.predicted
    }

    /// Error normalised by uncertainty: a surprise only counts as a surprise if the model
    /// claimed to be confident.
    pub fn normalised(&self) -> f64 {
        self.rpe() / (self.uncertainty.max(0.05))
    }

    /// Whether this error is informative enough to drive a decision.
    pub fn is_signal(&self, threshold: f64) -> bool {
        self.normalised().abs() > threshold
    }
}

/// What the evaluator recommends. Deterministic, derived from the numbers above.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// Error within tolerance and state changed.
    Continue,
    /// Model of this goal looks wrong; a different decomposition might work.
    Replan,
    /// Model of this goal is refuted; supersede it.
    Pivot,
    /// Give up; habituate.
    Abandon,
}

impl Decision {
    pub fn name(self) -> &'static str {
        match self {
            Decision::Continue => "continue",
            Decision::Replan => "replan",
            Decision::Pivot => "pivot",
            Decision::Abandon => "abandon",
        }
    }
}

/// Thresholds for the decision rule. Exposed as configuration, not constants, so the
/// hysteresis is auditable and tunable per deployment.
#[derive(Debug, Clone, Copy)]
pub struct Policy {
    /// |normalised error| below this is "no surprise".
    pub tolerance: f64,
    /// Consecutive negative-trend evaluations before a pivot.
    pub pivot_after: u32,
    /// Residual trend below this counts as negative.
    pub trend_floor: f64,
    /// Salience below this habituates.
    pub salience_floor: f64,
}

impl Default for Policy {
    fn default() -> Self {
        Policy { tolerance: 0.5, pivot_after: 2, trend_floor: -0.02, salience_floor: 0.05 }
    }
}

/// A complete evaluation. Construction enforces the validity rule.
#[derive(Debug, Clone)]
pub struct Evaluation {
    pub goal: Id,
    pub errors: Vec<PredictionError>,
    /// Slope of |error| over the window.
    pub residual_trend: f64,
    /// Did any verified fact change since the last evaluation?
    pub state_change: bool,
    pub decision: Decision,
    pub rationale: String,
}

impl Evaluation {
    /// Score a window of prediction errors.
    ///
    /// Returns [`crate::Error::UnbackedEvaluation`] if the window is empty — the single
    /// most important line in this file, because it is what stops an unpriced guess from
    /// being recorded as measured progress.
    pub fn score(
        goal: Id,
        errors: Vec<PredictionError>,
        state_change: bool,
        prior_inert_streak: u32,
        prior_negative_streak: u32,
        salience: f64,
        pivot_budget_left: bool,
        policy: &Policy,
    ) -> crate::Result<Self> {
        if errors.is_empty() {
            return Err(crate::Error::UnbackedEvaluation);
        }

        let magnitudes: Vec<f64> = errors.iter().map(|e| e.normalised().abs()).collect();
        let residual_trend = slope(&magnitudes);
        let mean_error = magnitudes.iter().sum::<f64>() / magnitudes.len() as f64;

        let (decision, rationale) = if !state_change && mean_error < policy.tolerance {
            // Nothing moved and nothing surprised us: inert.
            let streak = prior_inert_streak + 1;
            if streak >= policy.pivot_after {
                if pivot_budget_left {
                    (
                        Decision::Pivot,
                        format!(
                            "no verified change for {streak} evaluations and no surprise (mean |e| {mean_error:.2}): the goal is inert"
                        ),
                    )
                } else {
                    (
                        Decision::Abandon,
                        format!(
                            "inert for {streak} evaluations and no pivot budget left: habituate"
                        ),
                    )
                }
            } else {
                (
                    Decision::Continue,
                    format!("no change yet (streak {streak}); within pivot hysteresis"),
                )
            }
        } else if residual_trend < policy.trend_floor {
            // Errors getting worse: the model of this goal is refuted.
            let streak = prior_negative_streak + 1;
            if streak >= policy.pivot_after {
                if pivot_budget_left {
                    (
                        Decision::Pivot,
                        format!(
                            "residual trend {residual_trend:+.2} for {streak} evaluations: the forecast is refuted"
                        ),
                    )
                } else {
                    (Decision::Abandon, "refuted with no pivot budget left".to_string())
                }
            } else {
                (
                    Decision::Replan,
                    format!("residual trend {residual_trend:+.2}: try a different decomposition"),
                )
            }
        } else if salience < policy.salience_floor {
            (Decision::Abandon, format!("salience {salience:.2} below floor"))
        } else {
            (
                Decision::Continue,
                format!("mean |error| {mean_error:.2}, trend {residual_trend:+.2}: on track"),
            )
        };

        Ok(Evaluation { goal, errors, residual_trend, state_change, decision, rationale })
    }

    /// Mean absolute normalised error, for calibration reporting.
    pub fn mean_error(&self) -> f64 {
        if self.errors.is_empty() {
            return 0.0;
        }
        self.errors.iter().map(|e| e.normalised().abs()).sum::<f64>() / self.errors.len() as f64
    }

    /// Signed mean error: positive means the system systematically under-forecasts.
    pub fn bias(&self) -> f64 {
        if self.errors.is_empty() {
            return 0.0;
        }
        self.errors.iter().map(|e| e.rpe()).sum::<f64>() / self.errors.len() as f64
    }

    /// Whether the window showed a genuine change in verified state.
    pub fn is_progress(&self) -> bool {
        self.state_change && self.bias() > EPS
    }
}

/// Expected calibration error over (predicted, observed) pairs, binned.
///
/// Reported alongside any accuracy figure: an agent that says "70%" should be right 70% of
/// the time, and nothing else in this crate measures that.
pub fn calibration_error(pairs: &[(f64, f64)], bins: usize) -> f64 {
    if pairs.is_empty() || bins == 0 {
        return 0.0;
    }
    let mut sums = vec![(0.0f64, 0.0f64, 0usize); bins];
    for (predicted, observed) in pairs {
        let idx = ((unit_clamp(*predicted) * bins as f64) as usize).min(bins - 1);
        let s = &mut sums[idx];
        s.0 += unit_clamp(*predicted);
        s.1 += unit_clamp(*observed);
        s.2 += 1;
    }
    let total = pairs.len() as f64;
    sums.iter()
        .filter(|(_, _, n)| *n > 0)
        .map(|(p, o, n)| {
            let gap = (p / *n as f64 - o / *n as f64).abs();
            gap * (*n as f64 / total)
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err(predicted: f64, observed: f64, uncertainty: f64) -> PredictionError {
        PredictionError::new(Id(0), predicted, observed, 3, uncertainty, "t::check")
    }

    fn score_continue(errors: Vec<PredictionError>, state_change: bool) -> Evaluation {
        Evaluation::score(
            Id(0),
            errors,
            state_change,
            0,
            0,
            1.0,
            true,
            &Policy::default(),
        )
        .unwrap()
    }

    #[test]
    fn empty_window_is_rejected() {
        // T2: no unbacked evaluation. This must be an Err, never a zero.
        let r = Evaluation::score(Id(0), vec![], true, 0, 0, 1.0, true, &Policy::default());
        assert_eq!(r.err(), Some(crate::Error::UnbackedEvaluation));
    }

    #[test]
    fn rpe_is_signed_and_normalised() {
        let e = err(0.5, 0.8, 0.2);
        assert!((e.rpe() - 0.3).abs() < 1e-9);
        assert!((e.normalised() - 1.5).abs() < 1e-9);
        let conf = err(0.5, 0.55, 0.05);
        assert!(conf.normalised() > 9.0, "surprise scales as confidence rises");
    }

    #[test]
    fn on_track_when_error_small_and_state_moves() {
        let ev = score_continue(vec![err(0.5, 0.52, 0.1), err(0.6, 0.62, 0.1)], true);
        assert_eq!(ev.decision, Decision::Continue);
        assert!(ev.is_progress());
    }

    #[test]
    fn inert_goal_pivots_after_hysteresis() {
        // No state change, no surprise, second consecutive streak -> pivot.
        let r = Evaluation::score(
            Id(0),
            vec![err(0.5, 0.5, 0.4), err(0.5, 0.5, 0.4)],
            false,
            1, // prior inert streak
            0,
            1.0,
            true,
            &Policy::default(),
        )
        .unwrap();
        assert_eq!(r.decision, Decision::Pivot, "{}", r.rationale);
    }

    #[test]
    fn inert_without_pivot_budget_abandons() {
        let r = Evaluation::score(
            Id(0),
            vec![err(0.5, 0.5, 0.4)],
            false,
            5,
            0,
            1.0,
            false, // no pivots left
            &Policy::default(),
        )
        .unwrap();
        assert_eq!(r.decision, Decision::Abandon);
    }

    #[test]
    fn worsening_trend_replans_then_pivots() {
        let worsening = vec![err(0.9, 0.8, 0.05), err(0.9, 0.6, 0.05), err(0.9, 0.3, 0.05)];
        let first = Evaluation::score(
            Id(0), worsening.clone(), true, 0, 0, 1.0, true, &Policy::default(),
        )
        .unwrap();
        assert_eq!(first.decision, Decision::Replan, "{}", first.rationale);
        let second = Evaluation::score(
            Id(0), worsening, true, 0, 1, 1.0, true, &Policy::default(),
        )
        .unwrap();
        assert_eq!(second.decision, Decision::Pivot, "{}", second.rationale);
    }

    #[test]
    fn low_salience_abandons() {
        let ev = Evaluation::score(
            Id(0),
            vec![err(0.5, 0.55, 0.1)],
            true,
            0,
            0,
            0.01,
            true,
            &Policy::default(),
        )
        .unwrap();
        assert_eq!(ev.decision, Decision::Abandon);
    }

    #[test]
    fn calibration_error_is_zero_when_perfect() {
        let pairs = vec![(0.2, 0.2), (0.8, 0.8), (0.5, 0.5)];
        assert!(calibration_error(&pairs, 5) < 1e-9);
        let overconfident = vec![(0.9, 0.1), (0.9, 0.2)];
        assert!(calibration_error(&overconfident, 5) > 0.5);
    }
}
