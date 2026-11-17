//! A small calibrated predictive model.
//!
//! Without this, autonomy is unmeasurable: prediction error has no `predicted` term,
//! arbitration has no expected gain, and there is nothing to be wrong about. So it is a
//! prerequisite, not a nice-to-have.
//!
//! The model is deliberately tiny — per-drive running statistics with a running
//! uncertainty — because the property that matters is not accuracy but **calibration**: the
//! model must know how much it does not know. Uncertainty is consumed in three places:
//!
//! * arbitration discounts expected gain by it,
//! * the learnability filter uses it to suppress irreducible uncertainty (noisy-TV),
//! * prediction-error normalisation divides by it, so an error under acknowledged
//!   uncertainty is not treated as failure.

use crate::drives::DriveKind;
use crate::goals::Goal;
use crate::types::{unit_clamp, EPS};

/// A forecast for one goal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Forecast {
    /// Expected fractional progress in `[0, 1]`.
    pub expected: f64,
    /// Calibrated uncertainty in `[0, 1]`.
    pub uncertainty: f64,
    /// How many observations this estimate rests on.
    pub support: u32,
}

impl Forecast {
    /// Uninformative forecast: no support, maximum uncertainty. Used for goals with no
    /// check history, so their cost of being attempted is charged up front.
    pub fn uninformed() -> Self {
        Forecast { expected: 0.5, uncertainty: 1.0, support: 0 }
    }

    /// Expected gain for arbitration.
    ///
    /// Two terms. `expected * confidence` is plain expected progress discounted by how
    /// little the model knows. The second term is a bounded exploration bonus for reducible
    /// uncertainty: `exploration_weight * uncertainty * (1 - uncertainty)`.
    ///
    /// The bonus is largest at `uncertainty = 0.5` and vanishes at both extremes, which is
    /// the behaviour we want — a goal the model predicts perfectly gets no bonus (nothing
    /// left to learn), and a goal the model knows nothing about gets no bonus either,
    /// because that is a probe, not a pursuit. Deliberate probing is a separate mechanism
    /// (`selection::AffordanceScan`) with its own budget.
    pub fn expected_gain(&self) -> f64 {
        let confidence = 1.0 - self.uncertainty;
        let progress = self.expected * confidence;
        let exploration = self.uncertainty * (1.0 - self.uncertainty);
        (progress + 0.3 * exploration).max(0.0)
    }

    /// Discounted value used when the forecaster is asked what a goal is worth.
    pub fn value(&self) -> f64 {
        self.expected * (1.0 - 0.5 * self.uncertainty)
    }
}

/// Per-drive running statistics, updated by every verified outcome.
#[derive(Debug, Clone, Copy)]
struct Stats {
    n: u32,
    mean: f64,
    /// Exponentially weighted mean absolute deviation: cheap, robust, and interpretable as
    /// "how wrong have I been lately".
    mad: f64,
}

impl Stats {
    fn new() -> Self {
        Stats { n: 0, mean: 0.5, mad: 0.5 }
    }

    fn update(&mut self, observed: f64, alpha: f64) {
        let observed = unit_clamp(observed);
        if self.n == 0 {
            self.mean = observed;
            self.mad = 0.5;
        } else {
            let deviation = (observed - self.mean).abs();
            self.mean += alpha * (observed - self.mean);
            self.mad += alpha * (deviation - self.mad);
        }
        self.n = self.n.saturating_add(1);
    }

    /// Uncertainty shrinks with support and tracks recent error.
    fn uncertainty(&self) -> f64 {
        if self.n == 0 {
            return 1.0;
        }
        let support_discount = 1.0 / (1.0 + self.n as f64 / 3.0);
        unit_clamp((self.mad * 1.5).max(support_discount))
    }
}

/// The predictive model: one `Stats` per drive, plus a bias correction learned from signed
/// prediction error (so a systematically optimistic model gets corrected).
#[derive(Debug, Clone)]
pub struct PredictiveModel {
    stats: [Stats; 4],
    /// Running signed bias per drive: positive means outcomes beat forecasts.
    bias: [f64; 4],
    /// Learning rate for stats and bias.
    alpha: f64,
    /// Whether a goal's own check forecast curve should override the pooled estimate.
    prefer_check_forecast: bool,
}

impl Default for PredictiveModel {
    fn default() -> Self {
        PredictiveModel {
            stats: [Stats::new(); 4],
            bias: [0.0; 4],
            alpha: 0.25,
            prefer_check_forecast: true,
        }
    }
}

impl PredictiveModel {
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = unit_clamp(alpha);
        self
    }

    pub fn with_check_forecast_preference(mut self, prefer: bool) -> Self {
        self.prefer_check_forecast = prefer;
        self
    }

    fn index(kind: DriveKind) -> usize {
        match kind {
            DriveKind::Competence => 0,
            DriveKind::Coverage => 1,
            DriveKind::Cost => 2,
            DriveKind::Uncertainty => 3,
        }
    }

    /// Forecast a goal's next outcome.
    ///
    /// When the goal's own check carries a forecast curve, that is sharper than the pooled
    /// estimate; the pooled uncertainty still applies, because a curve is an assertion and
    /// all assertions have unknown reliability.
    pub fn forecast(&self, goal: &Goal) -> Forecast {
        let i = Self::index(goal.drive);
        let stats = self.stats[i];
        let pooled = Forecast {
            expected: unit_clamp(stats.mean + self.bias[i]),
            uncertainty: stats.uncertainty(),
            support: stats.n,
        };
        if !self.prefer_check_forecast || goal.check.forecast_curve.is_empty() {
            return pooled;
        }
        let curve_value = goal.check.forecast_at(goal.check.runs);
        Forecast {
            expected: unit_clamp(curve_value + self.bias[i]),
            uncertainty: pooled.uncertainty,
            support: stats.n,
        }
    }

    /// Update from a verified outcome. Called for abandoned goals too: abandonment is a
    /// refutation, and refutations are the most informative updates available.
    pub fn update(&mut self, kind: DriveKind, observed: f64, predicted: f64) {
        let i = Self::index(kind);
        let observed = unit_clamp(observed);
        self.stats[i].update(observed, self.alpha);
        let error = observed - unit_clamp(predicted);
        self.bias[i] += self.alpha * error;
        // Keep the correction bounded so a run of outliers cannot invert the model.
        self.bias[i] = self.bias[i].clamp(-0.5, 0.5);
    }

    /// Learnability in `[0, 1]`: how much of the remaining uncertainty is reducible.
    ///
    /// Operationally: a drive whose outcomes are highly variable *despite* many attempts is
    /// irreducible (noise). High variance that is falling with support is reducible
    /// (ignorance). The kernel feeds this straight into the drive's urgency so the
    /// noisy-TV failure cannot compete for attention.
    pub fn learnability(&self, kind: DriveKind) -> f64 {
        let s = self.stats[Self::index(kind)];
        if s.n == 0 {
            return 1.0;
        }
        // Variance that survives a lot of evidence is noise; variance that shrinks is
        // ignorance. `mad` is our proxy for variance.
        let noise_floor = 0.5;
        let evidence = s.n as f64;
        let reducible = (noise_floor - s.mad).max(0.0) / noise_floor;
        let optimism = (evidence / (evidence + 5.0)).min(1.0);
        unit_clamp(0.25 + 0.75 * reducible * optimism)
    }

    /// Calibration report for a set of realised forecasts, for the evaluator's ledger.
    pub fn calibration(&self) -> [(f64, f64); 4] {
        let mut out = [(0.0, 0.0); 4];
        for (i, s) in self.stats.iter().enumerate() {
            out[i] = (s.mean, s.uncertainty());
        }
        out
    }

    pub fn support(&self, kind: DriveKind) -> u32 {
        self.stats[Self::index(kind)].n
    }
}

/// Estimate how much a single outcome teaches us, used to order replay.
///
/// Reverse, surprise-prioritised replay is the cheap credit-assignment mechanism: replay
/// what surprised you, and replay it backwards from the outcome.
pub fn surprise(predicted: f64, observed: f64, uncertainty: f64) -> f64 {
    ((observed - predicted).abs() / uncertainty.max(EPS)).min(10.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goals::Check;
    use crate::types::IdGen;

    fn goal() -> Goal {
        let mut ids = IdGen::new(1);
        Goal::new(
            &mut ids,
            "t",
            DriveKind::Competence,
            crate::goals::Origin::SelfGenerated,
            Check::new("c", 0.0, 1.0),
        )
    }

    #[test]
    fn uninformed_forecast_gets_no_large_gain() {
        let gain = Forecast::uninformed().expected_gain();
        assert!(gain <= 0.1, "an unsupported forecast must not look attractive: {gain}");
    }

    #[test]
    fn exploration_bonus_is_zero_at_both_extremes() {
        // A goal predicted perfectly has nothing left to learn; a goal the model knows
        // nothing about is a probe, not a pursuit. Both must lose to a mid-uncertainty goal.
        let sure = Forecast { expected: 0.9, uncertainty: 0.0, support: 50 };
        let mid = Forecast { expected: 0.9, uncertainty: 0.5, support: 50 };
        let blank = Forecast { expected: 0.9, uncertainty: 1.0, support: 0 };
        assert!(mid.expected_gain() > sure.expected_gain(), "mid uncertainty should win");
        assert!(mid.expected_gain() > blank.expected_gain(), "total ignorance should not win");
        assert!(sure.expected_gain() > 0.85, "a confident good forecast still has value");
    }

    #[test]
    fn support_reduces_uncertainty() {
        let mut model = PredictiveModel::default();
        let g = goal();
        let before = model.forecast(&g).uncertainty;
        for _ in 0..40 {
            model.update(DriveKind::Competence, 0.6, 0.6);
        }
        let after = model.forecast(&g).uncertainty;
        assert!(after < before, "uncertainty should fall with support: {before} -> {after}");
        assert!(model.support(DriveKind::Competence) == 40);
    }

    #[test]
    fn bias_correction_follows_systematic_error() {
        let mut model = PredictiveModel::default().with_check_forecast_preference(false);
        for _ in 0..30 {
            model.update(DriveKind::Competence, 0.8, 0.5); // always better than forecast
        }
        let f = model.forecast(&goal());
        assert!(f.expected > 0.6, "optimism should be learned: {}", f.expected);
    }

    #[test]
    fn noise_is_less_learnable_than_ignorance() {
        // Ignorance: consistent values, so the model is converging.
        let mut ignorance = PredictiveModel::default();
        for _ in 0..40 {
            ignorance.update(DriveKind::Competence, 0.7, 0.5);
        }
        // Noise: maximally inconsistent, so nothing is converging.
        let mut noise = PredictiveModel::default();
        for i in 0..40 {
            let v = if i % 2 == 0 { 0.0 } else { 1.0 };
            noise.update(DriveKind::Competence, v, 0.5);
        }
        assert!(
            ignorance.learnability(DriveKind::Competence) > noise.learnability(DriveKind::Competence),
            "noisy tv must be filtered: {} vs {}",
            ignorance.learnability(DriveKind::Competence),
            noise.learnability(DriveKind::Competence)
        );
    }

    #[test]
    fn surprise_is_normalised_by_uncertainty() {
        let confident = surprise(0.5, 0.9, 0.05);
        let unsure = surprise(0.5, 0.9, 0.9);
        assert!(confident > unsure * 5.0);
        assert!(surprise(0.5, 0.5, 0.1) < 0.01);
    }
}
