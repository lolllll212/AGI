//! The brief's invariant suite (T1-T9), as integration tests against the public API.
//!
//! Each test names the failure it prevents. If you are reviewing this crate, these are the
//! tests to read first: they are the specification, not the implementation.

use agi_core::drives::DriveKind;
use agi_core::eval::{calibration_error, Decision, Evaluation, Policy, PredictionError};
use agi_core::goals::{Check, Goal, GoalState, Origin};
use agi_core::memory::Memory;
use agi_core::runner::{horizon_ratio, Runner, TickOutcome, INDEPENDENT_HORIZON_RATIO};
use agi_core::selection::Selector;
use agi_core::types::IdGen;
use agi_core::Error;

fn runner(seed: u64) -> Runner {
    let mut r = Runner::new(seed);
    r.config.require_approval = false;
    r.config.total_budget = 10_000.0;
    r
}

fn ready(r: &mut Runner, statement: &str, salience: f64) -> agi_core::Id {
    let id = r
        .propose(
            statement,
            DriveKind::Competence,
            Origin::Frontier,
            Check::new("t::check", 0.3, 0.95).with_forecast(vec![0.4, 0.5, 0.6, 0.7, 0.8]),
            None,
        )
        .unwrap();
    r.run_check(id, 0.35).unwrap();
    let g = r.store.get_mut(id).unwrap();
    if g.state == GoalState::Screened {
        g.salience = salience;
        g.begin_seeking(true).unwrap();
    }
    id
}

#[test]
fn t1_failure_is_representable() {
    // The reference loop marked failed steps DONE, so nothing could detect a stall.
    let mut ids = IdGen::new(1);
    let mut g = Goal::new(
        &mut ids,
        "work",
        DriveKind::Competence,
        Origin::SelfGenerated,
        Check::new("t", 0.0, 1.0),
    );
    g.transition(GoalState::Screened).unwrap();
    // Executing before any check has run is an error, and the state does not move.
    assert_eq!(g.begin_seeking(true), Err(Error::NoExecutedCheck));
    assert_eq!(g.state, GoalState::Screened);
    // A stalled goal is reachable and distinct from completion.
    g.check.record_run(0.1);
    g.begin_seeking(true).unwrap();
    g.transition(GoalState::Stalled).unwrap();
    assert_eq!(g.state, GoalState::Stalled);
    assert!(!g.state.is_terminal());
}

#[test]
fn t2_no_unbacked_evaluation() {
    let r = Evaluation::score(
        agi_core::Id(0),
        Vec::new(),
        true,
        0,
        0,
        1.0,
        true,
        &Policy::default(),
    );
    assert_eq!(r.err(), Some(Error::UnbackedEvaluation));
}

#[test]
fn t3_no_affordance_means_no_work() {
    let mut r = runner(3);
    for _ in 0..25 {
        assert!(matches!(r.tick().unwrap(), TickOutcome::Idle(_)));
    }
    assert_eq!(r.stats.steps, 0, "an untouched system must not invent work");
    assert_eq!(r.stats.ticks, 25);
}

#[test]
fn t4_goal_growth_is_bounded() {
    let mut r = runner(4);
    for i in 0..150 {
        let _ = r.propose(
            format!("family {i} needs fixture repair"),
            DriveKind::Coverage,
            Origin::Frontier,
            Check::new("t", 0.0, 1.0),
            None,
        );
        let _ = r.tick();
    }
    let rep = r.report();
    // Live goals cannot exceed what was proposed, and the total is dominated by terminal
    // records rather than accumulating live work.
    assert!(rep.live_goals <= 150);
    assert!(
        rep.total_goals <= rep.live_goals * 3 + 10,
        "proliferation: total {} vs live {}",
        rep.total_goals,
        rep.live_goals
    );
}

#[test]
fn t5_autonomy_share_is_respected() {
    let mut r = runner(5);
    r.config.total_budget = 10.0;
    // Self-proposed, not human-origin: the share only binds autonomous work.
    let id = r
        .propose(
            "self-proposed work on family c",
            DriveKind::Competence,
            Origin::SelfGenerated,
            Check::new("t::check", 0.3, 0.95).with_forecast(vec![0.4, 0.5, 0.6]),
            None,
        )
        .unwrap();
    r.run_check(id, 0.35).unwrap();
    {
        let g = r.store.get_mut(id).unwrap();
        if g.state == GoalState::Screened {
            g.salience = 1.0;
            g.begin_seeking(true).unwrap();
        }
    }
    // Pretend the goal has already consumed the whole share.
    r.store.get_mut(id).unwrap().budget_spent = 50.0;
    r.stats.autonomy_spent = 50.0;
    let out = r.tick().unwrap();
    assert!(matches!(out, TickOutcome::Idle(_)), "got {out:?}");
}

#[test]
fn t6_pivot_supersedes_without_siblings() {
    let mut r = runner(6);
    let parent = ready(&mut r, "stuck goal on family d", 1.0);
    let before = r.store.all().len();
    let child = r.pivot(parent).unwrap();
    assert_eq!(r.store.get(parent).unwrap().state, GoalState::Pivoted);
    assert_eq!(r.store.get(child).unwrap().parent, Some(parent));
    assert_eq!(r.store.all().len(), before + 1);
    assert_eq!(r.store.live_count(), 1, "the parent is closed, so one live goal remains");
}

#[test]
fn t7_habituation_prunes_inert_goals() {
    let mut r = runner(7);
    let id = ready(&mut r, "goal that will never move", 0.06);
    // Repeated inert evaluations decay salience below the floor.
    for _ in 0..40 {
        if let Some(g) = r.store.get_mut(id) {
            g.inert_streak = 3;
            g.habituate(0.0, 0.85);
        }
    }
    let habituated = r.store.get(id).map(|g| g.is_habituated(0.05)).unwrap_or(true);
    assert!(habituated, "salience must fall below the prune threshold");
    let out = r.tick().unwrap();
    assert!(matches!(out, TickOutcome::Habituated { .. } | TickOutcome::Idle(_)), "got {out:?}");
    assert_eq!(r.stats.steps, 0, "a habituated goal must never execute");
}

#[test]
fn t8_contradiction_supersedes_and_lowers_confidence() {
    let mut m = Memory::new();
    let old = m.learn("vendor upload returns 429", "sleep 2 then retry the same endpoint", 0.0);
    // Build up support for the old advice.
    for _ in 0..10 {
        m.note_use(old, true, 1.0);
    }
    let before = m.get(old).unwrap().confidence();
    let new = m.revise("vendor upload returns 429", "use the bulk endpoint in a single request", 20.0);
    let after = m.get(old).unwrap().confidence();
    assert!(before > 0.0);
    assert_eq!(after, 0.0, "superseded advice has no standing");
    assert!(m.get(new).unwrap().is_current());
    // The old advice is still retrievable, but only as labelled history.
    let current = m.recall("vendor upload returns 429", 5, false, 21.0);
    assert!(!current.iter().any(|h| h.lesson.id == old));
    let history = m.recall("vendor upload returns 429", 5, true, 21.0);
    assert!(history.iter().any(|h| h.historical && h.lesson.id == old));
    assert!(!history.is_empty() && !history[0].historical);
}

#[test]
fn t9_screening_demotes_unverifiable_proposals() {
    let mut r = runner(9);
    // Empty spec: no executable check. Must be demoted to a one-step probe, not accepted.
    let id = r
        .propose(
            "vague aspiration with no verifier",
            DriveKind::Competence,
            Origin::SelfGenerated,
            Check::new("", 0.0, 1.0),
            None,
        )
        .unwrap();
    let g = r.store.get(id).unwrap();
    assert_eq!(g.budget_units, 1.0, "unverifiable goals are probes");
    assert_eq!(r.stats.screened_out, 1);
}

#[test]
fn selection_returns_none_rather_than_a_bad_choice() {
    let r = Runner::new(10);
    let sel = Selector::default().select(
        &[],
        &r.drives,
        &r.measured,
        100.0,
        100.0,
        0.0,
    );
    assert!(sel.chosen.is_none());
    assert!(sel.idle_reason.is_some());
}

#[test]
fn decision_rule_needs_hysteresis() {
    let err = |p: f64, o: f64| PredictionError::new(agi_core::Id(0), p, o, 1, 0.2, "t");
    // One inert evaluation is not enough to pivot.
    let first = Evaluation::score(agi_core::Id(0), vec![err(0.5, 0.5)], false, 0, 0, 1.0, true, &Policy::default())
        .unwrap();
    assert_eq!(first.decision, Decision::Continue);
    // Two consecutive inert evaluations are.
    let second = Evaluation::score(agi_core::Id(0), vec![err(0.5, 0.5)], false, 1, 0, 1.0, true, &Policy::default())
        .unwrap();
    assert_eq!(second.decision, Decision::Pivot);
}

#[test]
fn calibration_is_measured_not_asserted() {
    let pairs: Vec<(f64, f64)> = (0..50).map(|i| (0.5, if i % 2 == 0 { 0.9 } else { 0.1 })).collect();
    let ece = calibration_error(&pairs, 10);
    assert!(ece > 0.3, "a systematically wrong forecaster must show up: {ece}");
    let perfect: Vec<(f64, f64)> = (0..50).map(|i| {
        let p = i as f64 / 50.0;
        (p, p)
    }).collect();
    assert!(calibration_error(&perfect, 10) < 0.05);
}

#[test]
fn horizon_ratio_distinguishes_independent_from_correlated_failure() {
    assert!((INDEPENDENT_HORIZON_RATIO - (0.8f64).ln() / (0.5f64).ln()).abs() < 1e-12);
    // A measured system below the constant has correlated failures (it gets stuck).
    let measured = horizon_ratio(12.0, 1.2);
    assert!(measured < INDEPENDENT_HORIZON_RATIO);
    // Recovery machinery should push it toward the constant.
    let recovered = horizon_ratio(12.0, 3.8);
    assert!(recovered > measured && recovered <= INDEPENDENT_HORIZON_RATIO + 0.02);
}

#[test]
fn journals_are_deterministic_for_a_fixed_seed() {
    let run = |seed: u64| {
        let mut r = runner(seed);
        for i in 0..12 {
            let _ = r.propose(
                format!("family {i} needs work"),
                DriveKind::Coverage,
                Origin::Frontier,
                Check::new("t", 0.0, 1.0),
                None,
            );
            let _ = r.tick();
        }
        (r.journal.digest(), r.report())
    };
    let (d1, rep1) = run(99);
    let (d2, rep2) = run(99);
    assert_eq!(d1, d2, "same seed must give the same record");
    assert_eq!(rep1.ticks, rep2.ticks);
    let (d3, _) = run(100);
    assert_ne!(d1, d3, "different seeds should diverge");
}
