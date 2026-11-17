//! End-to-end trace: propose, screen, select, execute, evaluate, pivot, habituate.
//!
//! Run with `cargo run --example demo`. The output is deliberately verbose — this is what
//! you read when you are deciding whether the kernel behaves like the brief says it does,
//! as opposed to whether the tests pass.

use agi_core::drives::DriveKind;
use agi_core::goals::{Check, GoalState, Origin};
use agi_core::runner::{Runner, TickOutcome};

fn main() {
    let mut r = Runner::new(11);
    r.config.total_budget = 60.0;

    println!("== a satisfied system idles ==");
    println!("  tick -> {:?}", r.tick().unwrap());
    println!("  live goals: {}\n", r.store.live_count());

    println!("== four sources propose, with different provenance ==");
    let frontier = r
        .propose(
            "family A: raise verified pass rate",
            DriveKind::Competence,
            Origin::Frontier,
            Check::new("family_a::pass", 0.35, 0.90).with_forecast(vec![0.4, 0.5, 0.62, 0.75, 0.88]),
            None,
        )
        .unwrap();
    let pattern = r
        .propose(
            "recurring fixture flake in family B",
            DriveKind::Coverage,
            Origin::PatternDetection,
            Check::new("family_b::flake_rate", 0.30, 0.02),
            None,
        )
        .unwrap();
    let dream = r
        .propose(
            "a simulated refactor that might remove the retry loop",
            DriveKind::Cost,
            Origin::DreamSimulation,
            Check::new("cost::per_success", 22.0, 8.0),
            None,
        )
        .unwrap();
    let human = r
        .propose(
            "human asked: document the retry policy",
            DriveKind::Cost,
            Origin::Human,
            Check::new("docs::retry_policy_present", 0.0, 1.0),
            None,
        )
        .unwrap();

    for (name, id) in [("frontier", frontier), ("pattern", pattern), ("dream", dream), ("human", human)] {
        let g = r.store.get(id).unwrap();
        println!("  {name:9} {} {:?}", g.label(), g.state);
    }
    println!("  note: the dream goal is CRAVING, not SEEKING — it cannot execute until a probe agrees\n");

    println!("== a check is the door to execution ==");
    r.run_check(frontier, 0.38).unwrap();
    r.run_check(pattern, 0.29).unwrap();
    r.run_check(human, 1.0).unwrap();
    // Promote through the validated door.
    for id in [frontier, pattern, human] {
        let g = r.store.get_mut(id).unwrap();
        if g.state == GoalState::Screened {
            g.salience = match id {
                x if x == frontier => 1.0,
                x if x == pattern => 0.95,
                _ => 0.8,
            };
            g.begin_seeking(true).unwrap();
        }
    }

    println!("== ten ticks ==");
    for i in 0..10 {
        match r.tick().unwrap() {
            TickOutcome::Executed { goal, value, progress } => {
                println!("  {i:2} executed {goal} -> observed {value:.3} (progress {progress:.2})")
            }
            TickOutcome::Idle(why) => println!("  {i:2} idle: {why}"),
            TickOutcome::Achieved { goal } => println!("  {i:2} achieved {goal}"),
            TickOutcome::Pivoted { parent, child } => {
                println!("  {i:2} pivoted {parent} -> {child} (parent superseded)")
            }
            TickOutcome::Habituated { goal } => println!("  {i:2} habituated {goal}"),
            TickOutcome::Failed { goal, reason } => println!("  {i:2} FAILED {goal}: {reason}"),
        }
    }

    println!("\n== drive urgencies after the run ==");
    for d in r.drives.iter() {
        println!(
            "  {:12} setpoint {:6.2} last {:6.2} satiation {:.2} learnability {:.2}",
            d.kind.name(),
            d.setpoint,
            d.last_value,
            d.satiation,
            d.learnability
        );
    }

    println!("\n== memory: a contradiction supersedes ==");
    let old = r.memory.learn("vendor upload returns 429", "sleep 2 then retry same endpoint", 0.0);
    let new = r.memory.revise("vendor upload returns 429", "use the bulk endpoint in one request", 50.0);
    println!(
        "  old {old} confidence {:.2} current={}",
        r.memory.get(old).unwrap().confidence(),
        r.memory.get(old).unwrap().is_current()
    );
    println!(
        "  new {new} confidence {:.2} current={}",
        r.memory.get(new).unwrap().confidence(),
        r.memory.get(new).unwrap().is_current()
    );
    let hits = r.memory.recall("vendor upload returns 429", 3, true, 51.0);
    for h in &hits {
        println!(
            "  recall: {}{} — {}",
            if h.historical { "[historical] " } else { "[current] " },
            h.lesson.advice,
            format!("score {:.3}", h.score)
        );
    }

    println!("\n== report ==");
    println!("{}", r.report());
    println!(
        "\nhorizon ratio constant for independent steps: {:.3}",
        agi_core::runner::INDEPENDENT_HORIZON_RATIO
    );

    println!("\n== first 12 journal events ==");
    for e in r.journal.iter().take(12) {
        println!(
            "  t={:5.1} {:>10} {:?} {}",
            e.at,
            e.kind.name(),
            e.goal,
            e.detail
        );
    }
}
