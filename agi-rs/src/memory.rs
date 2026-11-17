//! Revision-aware memory.
//!
//! The dominant long-horizon memory failure is not failing to recall, it is **reusing
//! knowledge that has since been invalidated** — published error analyses attribute the
//! majority of a strong agent's recommendation errors to outdated memory being kept rather
//! than forgotten.
//!
//! The trap is a merge rule that is right for duplicates and wrong for contradictions. Two
//! lessons saying "retry after sleeping" and "use the bulk endpoint instead" are not the
//! same lesson; folding them together keeps whichever arrived first and *raises* confidence
//! because support adds up. So this module separates the two operations:
//!
//! * **`absorb`** — evidence for the *same* advice. Support adds; confidence rises. Correct.
//! * **`supersede`** — *contradicting* advice. The newer lesson wins, the older is marked
//!   `Superseded { by, at }`, and its confidence **falls** rather than rising.
//!
//! Forgetting here is by relevance, never by capacity, and `recall` returns a current-state
//! view: superseded advice is retrievable, labelled as historical, and never presented as
//! current.

use std::collections::HashMap;

use crate::types::{hash_of, jaccard, Id, EPS};

/// Whether a lesson is currently believed, or was replaced.
#[derive(Debug, Clone, PartialEq)]
pub enum Validity {
    /// Currently true.
    Current,
    /// Replaced by another lesson at a given simulation time.
    Superseded { by: Id, at: f64 },
}

impl Validity {
    pub fn is_current(&self) -> bool {
        matches!(self, Validity::Current)
    }
}

/// A reusable piece of knowledge with evidence and an explicit validity.
#[derive(Debug, Clone)]
pub struct Lesson {
    pub id: Id,
    /// The situation in which this applies.
    pub trigger: String,
    /// What to do about it.
    pub advice: String,
    /// What failed and what worked, for `fix` lessons.
    pub failed: Option<String>,
    pub validity: Validity,
    /// How many independent observations support the advice.
    pub support: u32,
    /// Retrievals that preceded success / failure. Coarse but useful.
    pub helped: u32,
    pub hurt: u32,
    pub created: f64,
    pub last_used: f64,
    pub tags: Vec<String>,
}

impl Lesson {
    pub fn new(id: Id, trigger: impl Into<String>, advice: impl Into<String>) -> Self {
        Lesson {
            id,
            trigger: trigger.into(),
            advice: advice.into(),
            failed: None,
            validity: Validity::Current,
            support: 1,
            helped: 0,
            hurt: 0,
            created: 0.0,
            last_used: 0.0,
            tags: Vec::new(),
        }
    }

    /// Posterior mean of "retrieving this helps" under a uniform Beta prior.
    pub fn utility(&self) -> f64 {
        (self.helped as f64 + 1.0) / (self.helped as f64 + self.hurt as f64 + 2.0)
    }

    /// Confidence combines support and utility, and collapses when superseded.
    ///
    /// The collapse is the point: a superseded lesson must never out-rank its replacement
    /// just because it accumulated more observations first.
    pub fn confidence(&self) -> f64 {
        if !self.validity.is_current() {
            return 0.0;
        }
        let evidence = self.support as f64 / (self.support as f64 + 1.0);
        evidence * self.utility()
    }

    pub fn is_current(&self) -> bool {
        self.validity.is_current()
    }
}

/// One retrieval result.
#[derive(Debug, Clone)]
pub struct Recall<'a> {
    pub lesson: &'a Lesson,
    pub score: f64,
    /// Whether this is historical (superseded) advice shown for context only.
    pub historical: bool,
}

/// The store: lessons, revision edges, and retrieval.
#[derive(Debug, Clone, Default)]
pub struct Memory {
    lessons: HashMap<Id, Lesson>,
    /// Revisions: `(older, newer, reason)`.
    revisions: Vec<(Id, Id, String)>,
    /// Relevance scores, decayed so that forgetting is by relevance.
    relevance: HashMap<Id, f64>,
    next: u64,
}

impl Memory {
    pub fn new() -> Self {
        Memory::default()
    }

    /// Store a new lesson. Returns its id.
    pub fn learn(&mut self, trigger: impl Into<String>, advice: impl Into<String>, now: f64) -> Id {
        let id = Id(self.next);
        self.next += 1;
        let mut lesson = Lesson::new(id, trigger, advice);
        lesson.created = now;
        self.relevance.insert(id, 1.0);
        self.lessons.insert(id, lesson);
        id
    }

    pub fn get(&self, id: Id) -> Option<&Lesson> {
        self.lessons.get(&id)
    }

    pub fn len(&self) -> usize {
        self.lessons.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lessons.is_empty()
    }

    /// Fold a duplicate into an existing lesson: evidence adds up, confidence rises.
    ///
    /// Only legal when the advice is *equivalent*. Callers should use [`Memory::revise`]
    /// when advice differs, which decides between absorb and supersede.
    pub fn absorb(&mut self, keeper: Id, other: Id) -> bool {
        if keeper == other {
            return false;
        }
        let (other_advice, other_support) = match self.lessons.get(&other) {
            Some(l) => (l.advice.clone(), l.support),
            None => return false,
        };
        let keeper_advice = match self.lessons.get(&keeper) {
            Some(l) => l.advice.clone(),
            None => return false,
        };
        if jaccard(&keeper_advice, &other_advice) < 0.6 {
            // Not equivalent: absorbing would silently drop different advice.
            return false;
        }
        if let Some(k) = self.lessons.get_mut(&keeper) {
            k.support += other_support;
            k.last_used = k.last_used.max(0.0);
        }
        if let Some(o) = self.lessons.get_mut(&other) {
            o.validity = Validity::Superseded { by: keeper, at: 0.0 };
        }
        self.lessons.remove(&other);
        self.relevance.remove(&other);
        true
    }

    /// Record a revision: `newer` replaces `older` because of `reason`.
    ///
    /// This is the operation the reference implementation lacked. The older lesson is kept
    /// (auditable, retrievable as history) but its confidence goes to zero and it can never
    /// again be presented as current advice.
    pub fn supersede(&mut self, older: Id, newer: Id, reason: impl Into<String>, now: f64) -> bool {
        if older == newer {
            return false;
        }
        if !self.lessons.contains_key(&older) || !self.lessons.contains_key(&newer) {
            return false;
        }
        if let Some(o) = self.lessons.get_mut(&older) {
            o.validity = Validity::Superseded { by: newer, at: now };
        }
        self.revisions.push((older, newer, reason.into()));
        // The replacement inherits the older's evidence base but not its confidence: new
        // advice starts thin and has to earn support on its own.
        if let Some(n) = self.lessons.get_mut(&newer) {
            n.last_used = now;
        }
        true
    }

    /// Learn-or-revise in one call: the default path for a new observation about a known
    /// situation.
    ///
    /// If a current lesson with a similar trigger but *different* advice exists, the new one
    /// supersedes it. If the advice is equivalent, evidence is absorbed. This is where the
    /// "support goes up for contradicted advice" bug is fixed: a contradiction never
    /// increments the old lesson's support.
    pub fn revise(
        &mut self,
        trigger: impl Into<String>,
        advice: impl Into<String>,
        now: f64,
    ) -> Id {
        let trigger = trigger.into();
        let advice = advice.into();

        let related: Option<Id> = self
            .lessons
            .values()
            .filter(|l| l.is_current())
            .filter(|l| jaccard(&l.trigger, &trigger) >= 0.5)
            .max_by(|a, b| {
                jaccard(&a.trigger, &trigger)
                    .partial_cmp(&jaccard(&b.trigger, &trigger))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|l| l.id);

        let new_id = self.learn(trigger, advice.clone(), now);

        if let Some(old_id) = related {
            let same_advice = self
                .lessons
                .get(&old_id)
                .map(|l| jaccard(&l.advice, &advice) >= 0.6)
                .unwrap_or(false);
            if same_advice {
                // Duplicate: absorb into the older lesson and drop the new one. This keeps
                // ids stable and lets support accumulate honestly.
                if let Some(k) = self.lessons.get_mut(&old_id) {
                    k.support += 1;
                    k.last_used = now;
                }
                self.lessons.remove(&new_id);
                self.relevance.remove(&new_id);
                return old_id;
            }
            // Contradiction: supersede. The old lesson keeps its history and loses its
            // standing.
            self.supersede(old_id, new_id, format!("contradicted at t={now:.0}"), now);
        }
        new_id
    }

    /// Relevance-based forgetting: drop current lessons whose relevance has decayed below
    /// `epsilon`. Never drops by capacity, which is what makes it safe under volume.
    pub fn forget_irrelevant(&mut self, epsilon: f64) -> Vec<Id> {
        let doomed: Vec<Id> = self
            .relevance
            .iter()
            .filter(|(_, r)| **r < epsilon)
            .map(|(id, _)| *id)
            .collect();
        for id in &doomed {
            self.lessons.remove(id);
            self.relevance.remove(id);
        }
        doomed
    }

    /// Decay relevance for everything not retrieved this window.
    pub fn decay(&mut self, factor: f64) {
        for r in self.relevance.values_mut() {
            *r *= factor;
        }
    }

    /// Retrieve the most relevant lessons for a query.
    ///
    /// Current lessons always outrank superseded ones; superseded lessons are returned only
    /// when `include_history` is set, and they are flagged so the caller can say "this used
    /// to be true" rather than presenting it as advice.
    pub fn recall(&self, query: &str, k: usize, include_history: bool, now: f64) -> Vec<Recall<'_>> {
        let mut scored: Vec<Recall<'_>> = self
            .lessons
            .values()
            .filter_map(|l| {
                let historical = !l.is_current();
                if historical && !include_history {
                    return None;
                }
                let text_sim = jaccard(&l.trigger, query).max(jaccard(&l.advice, query) * 0.5);
                if text_sim < 0.05 {
                    return None;
                }
                let relevance = self.relevance.get(&l.id).copied().unwrap_or(1.0);
                let score = if historical {
                    text_sim * relevance * 0.25 // history is context, never advice
                } else {
                    text_sim * relevance * l.confidence().max(0.2)
                };
                Some(Recall { lesson: l, score, historical })
            })
            .collect();
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        // No read-side mutation here: recall stays `&self` and relevance is bumped by
        // `note_use` once the outcome is known, so relevance tracks usefulness rather than
        // fan-out.
        let _ = now;
        scored
    }

    /// Mark that a recalled lesson was actually used. Called by the runner after an outcome
    /// is known, so relevance tracks usefulness rather than fan-out.
    pub fn note_use(&mut self, id: Id, helped: bool, now: f64) {
        if let Some(l) = self.lessons.get_mut(&id) {
            if helped {
                l.helped += 1;
            } else {
                l.hurt += 1;
            }
            l.last_used = now;
        }
        if let Some(r) = self.relevance.get_mut(&id) {
            *r = (*r * 1.2).min(1.0);
        }
    }

    /// Re-screen hook: which current lessons rest on evidence that has been superseded?
    ///
    /// A goal whose supporting lesson was later invalidated must be re-screened, otherwise
    /// the system acts on advice it has already learned is wrong.
    pub fn invalidated(&self) -> Vec<Id> {
        self.revisions
            .iter()
            .filter(|(_, newer, _)| {
                self.lessons.get(newer).map(|l| l.is_current()).unwrap_or(false)
            })
            .map(|(older, _, _)| *older)
            .collect()
    }

    pub fn revisions(&self) -> &[(Id, Id, String)] {
        &self.revisions
    }

    /// Stable digest of the current knowledge state, for provenance and drift auditing.
    pub fn digest(&self) -> u64 {
        let mut current: Vec<(u64, u64)> = self
            .lessons
            .values()
            .filter(|l| l.is_current())
            .map(|l| (hash_of(&l.trigger), hash_of(&l.advice)))
            .collect();
        current.sort();
        hash_of(&current)
    }
}

/// Similarity helper exposed for the screening gate, so redundancy is decided by the same
/// measure everywhere.
pub fn redundancy(a: &str, b: &str) -> f64 {
    jaccard(a, b)
}

/// Whether two bits of advice are equivalent, and can therefore be absorbed.
pub fn equivalent(a: &str, b: &str) -> bool {
    jaccard(a, b) >= 0.6 && (a.len() as f64 - b.len() as f64).abs() < EPS.max(20.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_lessons_accumulate_support() {
        let mut m = Memory::new();
        let a = m.learn("upload to vendor api fails", "sleep 2 then retry", 1.0);
        let b = m.learn("upload to vendor api fails", "sleep 2 then retry", 2.0);
        assert!(m.absorb(a, b));
        assert_eq!(m.get(a).unwrap().support, 2);
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn contradicting_advice_never_absorbs() {
        let mut m = Memory::new();
        let a = m.learn("upload to vendor api fails", "sleep 2 then retry", 1.0);
        let b = m.learn("upload to vendor api fails", "use bulk endpoint instead", 2.0);
        assert!(!m.absorb(a, b), "different advice must not be merged");
        assert_eq!(m.get(a).unwrap().support, 1, "support must not rise on contradiction");
        assert!(m.get(b).unwrap().is_current());
    }

    #[test]
    fn supersession_lowers_confidence_and_hides_advice() {
        // T8: contradicting evidence supersedes and lowers confidence in the old advice.
        let mut m = Memory::new();
        let old = m.learn("vendor upload returns 429", "sleep 2 then retry the same endpoint", 1.0);
        for _ in 0..5 {
            m.absorb(old, old); // pretend it accumulated support
        }
        let before = m.get(old).unwrap().confidence();
        let new = m.revise("vendor upload returns 429", "use the bulk endpoint in one request", 10.0);
        let after = m.get(old).unwrap().confidence();
        assert!(after < before, "confidence must fall: {before} -> {after}");
        assert_eq!(after, 0.0);
        assert!(m.get(new).unwrap().is_current());
        assert_eq!(m.revisions().len(), 1);
    }

    #[test]
    fn revise_absorbs_equivalent_advice() {
        let mut m = Memory::new();
        let a = m.revise("vendor upload returns 429", "sleep two seconds then retry", 1.0);
        let b = m.revise("vendor upload returns 429", "sleep two seconds then retry", 2.0);
        assert_eq!(a, b, "equivalent advice should collapse to one lesson");
        assert_eq!(m.get(a).unwrap().support, 2);
    }

    #[test]
    fn recall_prefers_current_and_labels_history() {
        let mut m = Memory::new();
        let old = m.learn("vendor upload returns 429", "sleep 2 then retry same endpoint", 1.0);
        let _new = m.revise("vendor upload returns 429", "use bulk endpoint in one request", 10.0);
        let hits = m.recall("vendor upload returns 429", 5, false, 11.0);
        assert!(hits.iter().all(|h| !h.historical), "history must be excluded by default");
        assert!(!hits.iter().any(|h| h.lesson.id == old));
        let with_history = m.recall("vendor upload returns 429", 5, true, 11.0);
        assert!(with_history.iter().any(|h| h.historical && h.lesson.id == old));
        let top = &with_history[0];
        assert!(!top.historical, "current advice must outrank history");
    }

    #[test]
    fn forgetting_is_by_relevance_not_capacity() {
        let mut m = Memory::new();
        let ids: Vec<Id> = (0..50)
            .map(|i| m.learn(format!("situation number {i} about widgets"), format!("do thing {i}"), 0.0))
            .collect();
        // Everything decays equally: nothing is deleted by volume.
        m.decay(0.9);
        assert!(m.forget_irrelevant(0.0).is_empty(), "volume alone must not delete");
        let dropped = m.forget_irrelevant(0.5);
        assert_eq!(dropped.len(), 50, "relevance below threshold deletes");
        assert_eq!(m.len(), 0);
        let _ = ids;
    }

    #[test]
    fn invalidated_reports_superseded_evidence() {
        let mut m = Memory::new();
        let old = m.learn("family a flakes", "retry twice", 0.0);
        m.revise("family a flakes", "fix the fixture instead of retrying", 1.0);
        assert_eq!(m.invalidated(), vec![old]);
    }

    #[test]
    fn digest_changes_when_knowledge_changes() {
        let mut m = Memory::new();
        let a = m.digest();
        m.learn("something new", "do it", 0.0);
        assert_ne!(a, m.digest());
    }
}
