//! Arbitration: which goal gets the single action channel this tick.
//!
//! Biologically this is basal ganglia selection: candidates compete, the winner suppresses
//! close competitors (lateral inhibition), and — importantly — **selection may return
//! nothing**. An agent with no `SEEKING` goal must be able to idle and scan affordances.
//! Systems that cannot idle invent work, which is the failure this whole crate is arranged
//! to prevent.
//!
//! Why `argmax` + inhibition rather than softmax sampling: with sampling you keep choosing
//! low-value goals *because they are possible*. Stochasticity belongs in which affordances
//! get scanned, not in which goal gets pursued. Exploration of the option space is a
//! separate mechanism; see [`AffordanceScan`].

use crate::drives::{DriveKind, DriveSet};
use crate::goals::{Goal, GoalState};
use crate::world::{Forecast, PredictiveModel};
use crate::types::{Id, EPS};

/// A candidate for selection: a live goal plus the freshness of its forecast.
#[derive(Debug, Clone)]
pub struct Candidate<'a> {
    pub goal: &'a Goal,
    /// Forecast for this goal from the world model.
    pub forecast: Forecast,
    /// What one attempt is expected to cost, in budget units.
    pub cost: f64,
}

impl<'a> Candidate<'a> {
    pub fn new(goal: &'a Goal, forecast: Forecast, cost: f64) -> Self {
        Candidate { goal, forecast, cost: cost.max(EPS) }
    }

    /// Urgency x expected gain / cost.
    ///
    /// Expected gain uses the *predicted reduction in deficiency*, not predicted reward.
    /// A goal whose outcome the model predicts perfectly has no uncertainty to spend, and
    /// contributes little — which is the "predicted perfectly is not worth pursuing" rule
    /// showing up as arithmetic rather than as a special case.
    pub fn score(&self, drives: &DriveSet, measured: &[f64; 4]) -> f64 {
        let urgency = drives.get(self.goal.drive).urgency(measured[drive_index(self.goal.drive)]);
        let gain = self.forecast.expected_gain();
        let salience = self.goal.salience;
        (urgency * gain * salience) / self.cost
    }
}

pub fn drive_index(kind: DriveKind) -> usize {
    match kind {
        DriveKind::Competence => 0,
        DriveKind::Coverage => 1,
        DriveKind::Cost => 2,
        DriveKind::Uncertainty => 3,
    }
}

/// Selection policy knobs.
#[derive(Debug, Clone, Copy)]
pub struct Selector {
    /// Competitors scoring above this fraction of the winner are suppressed.
    pub inhibition_ratio: f64,
    /// How much salience a suppressed competitor loses.
    pub inhibition_strength: f64,
    /// Goals scoring below this are not worth an action this tick.
    pub min_score: f64,
    /// What fraction of total budget autonomous work may consume.
    pub autonomy_share: f64,
}

impl Default for Selector {
    fn default() -> Self {
        Selector {
            inhibition_ratio: 0.9,
            inhibition_strength: 0.7,
            min_score: 1e-6,
            autonomy_share: 0.2,
        }
    }
}

/// The outcome of arbitration.
#[derive(Debug, Clone)]
pub struct Selection {
    /// The winner, if any. `None` is a legitimate, expected result.
    pub chosen: Option<Id>,
    /// Scores for every candidate, in input order: (goal id, score).
    pub scores: Vec<(Id, f64)>,
    /// Competitors suppressed this tick.
    pub suppressed: Vec<Id>,
    /// Why nothing was chosen, when nothing was.
    pub idle_reason: Option<&'static str>,
}

impl Selector {
    /// Choose at most one goal to execute.
    ///
    /// `budget_remaining` and `total_budget` implement the autonomy share: self-proposed
    /// work draws from a fraction of the total, so autonomy cannot consume the mission.
    pub fn select(
        &self,
        candidates: &[Candidate<'_>],
        drives: &DriveSet,
        measured: &[f64; 4],
        budget_remaining: f64,
        total_budget: f64,
        now: f64,
    ) -> Selection {
        let mut scores: Vec<(Id, f64)> = Vec::with_capacity(candidates.len());

        // Anything not in SEEKING is ineligible by definition; CRAVING especially, because
        // there is no affordance to act on.
        let mut eligible: Vec<(&Candidate<'_>, f64)> = Vec::new();
        for c in candidates {
            if c.goal.state != GoalState::Seeking {
                continue;
            }
            if c.goal.expired(now) {
                continue;
            }
            if !c.goal.has_budget() {
                continue;
            }
            if c.cost > budget_remaining + EPS {
                continue;
            }
            let score = c.score(drives, measured);
            scores.push((c.goal.id, score));
            if score > self.min_score {
                eligible.push((c, score));
            }
        }

        if eligible.is_empty() {
            return Selection {
                chosen: None,
                scores,
                suppressed: Vec::new(),
                idle_reason: Some("no goal is urgent, affordable and actionable"),
            };
        }

        // Autonomy share: if self-proposed work has already eaten its fraction, only
        // human-origin goals may run.
        if total_budget > 0.0 {
            let autonomy_cap = total_budget * self.autonomy_share;
            let spent_on_autonomy: f64 =
                candidates.iter().filter(|c| c.goal.origin != crate::goals::Origin::Human)
                    .map(|c| c.goal.budget_spent).sum();
            if spent_on_autonomy >= autonomy_cap {
                eligible.retain(|(c, _)| c.goal.origin == crate::goals::Origin::Human);
                if eligible.is_empty() {
                    return Selection {
                        chosen: None,
                        scores,
                        suppressed: Vec::new(),
                        idle_reason: Some("autonomy budget share exhausted"),
                    };
                }
            }
        }

        eligible.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let (winner, top_score) = eligible[0];

        // Lateral inhibition: suppress near-competitors so they do not immediately
        // reshuffle the ordering next tick.
        let mut suppressed = Vec::new();
        for (c, score) in eligible.iter().skip(1) {
            if *score > self.inhibition_ratio * top_score {
                suppressed.push(c.goal.id);
            }
        }

        Selection { chosen: Some(winner.goal.id), scores, suppressed, idle_reason: None }
    }

    /// Salience reduction to apply to suppressed competitors. Returned rather than applied,
    /// because only the runner owns goal mutation.
    pub fn inhibition_for(&self, suppressed: bool) -> f64 {
        if suppressed {
            self.inhibition_strength
        } else {
            1.0
        }
    }
}

/// Deliberate exploration of the affordance space.
///
/// This is deliberately *not* part of goal selection. The brief's fourth falsifier is that
/// restricting candidates to (drive x affordance) forbids discovering affordances the
/// environment does not present; the counter-mechanism is a separate budget for probing
/// unknown options. Keeping the two mechanisms separate is what lets you measure which one
/// produced the value.
#[derive(Debug, Clone, Copy)]
pub struct AffordanceScan {
    /// Fraction of ticks spent probing rather than pursuing.
    pub exploration_share: f64,
}

impl Default for AffordanceScan {
    fn default() -> Self {
        AffordanceScan { exploration_share: 0.05 }
    }
}

impl AffordanceScan {
    /// Whether this tick is an exploration tick, given a seeded unit draw.
    pub fn should_probe(&self, unit_draw: f64) -> bool {
        unit_draw < self.exploration_share
    }
}

/// Convenience: build candidates by forecasting every live goal.
pub fn candidates<'a>(
    goals: &'a [Goal],
    model: &PredictiveModel,
) -> Vec<Candidate<'a>> {
    goals
        .iter()
        .filter(|g| g.state.is_live())
        .map(|g| {
            let forecast = model.forecast(g);
            Candidate::new(g, forecast, g.step_cost)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drives::{Drive, DriveKind};
    use crate::goals::{Check, Goal, Origin};
    use crate::types::IdGen;
    use crate::world::PredictiveModel;

    fn seeking_goal(ids: &mut IdGen, drive: DriveKind, salience: f64) -> Goal {
        let mut g = Goal::new(
            ids,
            "do the thing",
            drive,
            Origin::SelfGenerated,
            Check::new("t", 0.0, 1.0),
        )
        .with_salience(salience);
        g.state = GoalState::Screened;
        g.check.record_run(0.1);
        g.begin_seeking(true).unwrap();
        g
    }

    fn starved_drives() -> (DriveSet, [f64; 4]) {
        let set = DriveSet::with(vec![
            Drive::new(DriveKind::Competence, 0.85, 1.0),
            Drive::new(DriveKind::Coverage, 8.0, 0.5),
            Drive::new(DriveKind::Cost, 20.0, 0.6),
            Drive::new(DriveKind::Uncertainty, 0.25, 0.7),
        ]);
        (set, [0.0, 0.0, 1.0, 0.9])
    }

    #[test]
    fn selection_can_legitimately_return_none() {
        // T3: empty affordance idles. No SEEKING goals -> no work, no invention.
        let ids = IdGen::new(1);
        let (drives, measured) = starved_drives();
        let mut ids = ids;
        let mut craving = seeking_goal(&mut ids, DriveKind::Competence, 1.0);
        craving.state = GoalState::Craving;
        let model = PredictiveModel::default();
        let cands = vec![Candidate::new(
            &craving,
            model.forecast(&craving),
            1.0,
        )];
        let sel = Selector::default().select(&cands, &drives, &measured, 100.0, 100.0, 0.0);
        assert!(sel.chosen.is_none(), "a craving must never execute");
        assert!(sel.idle_reason.is_some());
    }

    #[test]
    fn winner_is_the_highest_score_and_near_competitors_are_suppressed() {
        let mut ids = IdGen::new(2);
        let a = seeking_goal(&mut ids, DriveKind::Competence, 1.0);
        let b = seeking_goal(&mut ids, DriveKind::Competence, 0.98);
        let c = seeking_goal(&mut ids, DriveKind::Competence, 0.10);
        let (drives, measured) = starved_drives();
        let model = PredictiveModel::default();
        let cands: Vec<Candidate> = [&a, &b, &c]
            .iter()
            .map(|g| Candidate::new(g, model.forecast(g), 1.0))
            .collect();
        let sel = Selector::default().select(&cands, &drives, &measured, 100.0, 100.0, 0.0);
        assert_eq!(sel.chosen, Some(a.id));
        assert!(sel.suppressed.contains(&b.id), "near-competitor should be inhibited");
        assert!(!sel.suppressed.contains(&c.id), "distant competitor is not a threat");
    }

    #[test]
    fn unaffordable_goals_are_skipped_not_overspent() {
        let mut ids = IdGen::new(3);
        let mut g = seeking_goal(&mut ids, DriveKind::Competence, 1.0);
        g.step_cost = 50.0;
        let (drives, measured) = starved_drives();
        let model = PredictiveModel::default();
        let cands = vec![Candidate::new(&g, model.forecast(&g), 50.0)];
        let sel = Selector::default().select(&cands, &drives, &measured, 10.0, 100.0, 0.0);
        assert!(sel.chosen.is_none());
    }

    #[test]
    fn autonomy_share_caps_self_proposed_work() {
        // T5: budget isolation, under maximal drive urgency.
        let mut ids = IdGen::new(4);
        let mut g = seeking_goal(&mut ids, DriveKind::Competence, 1.0);
        g.budget_spent = 60.0; // already used its share of a 100-unit budget
        let (drives, measured) = starved_drives();
        let model = PredictiveModel::default();
        let cands = vec![Candidate::new(&g, model.forecast(&g), 1.0)];
        let sel = Selector::default().select(&cands, &drives, &measured, 100.0, 100.0, 0.0);
        assert!(sel.chosen.is_none(), "self-proposed work must not exceed its share");
        assert_eq!(sel.idle_reason, Some("autonomy budget share exhausted"));
    }

    #[test]
    fn human_goals_run_after_autonomy_share_is_spent() {
        let mut ids = IdGen::new(5);
        let mut human = Goal::new(
            &mut ids,
            "human asks for this",
            DriveKind::Competence,
            Origin::Human,
            Check::new("t", 0.0, 1.0),
        );
        human.state = GoalState::Screened;
        human.check.record_run(0.0);
        human.begin_seeking(true).unwrap();
        let (drives, measured) = starved_drives();
        let model = PredictiveModel::default();
        let cands = vec![Candidate::new(&human, model.forecast(&human), 1.0)];
        let sel = Selector::default().select(&cands, &drives, &measured, 100.0, 0.0, 0.0);
        assert_eq!(sel.chosen, Some(human.id));
    }

    #[test]
    fn expired_goals_are_not_selected() {
        let mut ids = IdGen::new(6);
        let g = seeking_goal(&mut ids, DriveKind::Competence, 1.0).with_deadline(10.0);
        let (drives, measured) = starved_drives();
        let model = PredictiveModel::default();
        let cands = vec![Candidate::new(&g, model.forecast(&g), 1.0)];
        let sel = Selector::default().select(&cands, &drives, &measured, 100.0, 100.0, 99.0);
        assert!(sel.chosen.is_none(), "deadline must be enforced");
    }
}
