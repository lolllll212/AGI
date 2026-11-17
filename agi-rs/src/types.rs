//! Identifiers, a deterministic RNG, and the small numeric helpers used elsewhere.
//!
//! No global mutable state: ids and randomness are owned by the [`IdGen`] you pass in, so
//! two runs with the same seed produce byte-identical journals. That is what makes the
//! invariant tests meaningful.

use std::collections::hash_map::DefaultHasher;
use std::hash::Hash;

/// A short, stable identifier. Displayed as `g0`, `g1`, ...; never reused within a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Id(pub u64);

impl std::fmt::Display for Id {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            n if n < 10_000 => write!(f, "g{n}"),
            n => write!(f, "g{n:x}"),
        }
    }
}

/// Monotonic id source plus a seeded xorshift RNG.
#[derive(Debug, Clone)]
pub struct IdGen {
    next: u64,
    state: u64,
}

impl IdGen {
    pub fn new(seed: u64) -> Self {
        // xorshift is degenerate for a zero state; escape it.
        IdGen { next: 0, state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed } }
    }

    /// Next identifier.
    pub fn id(&mut self) -> Id {
        let id = Id(self.next);
        self.next += 1;
        id
    }

    /// Next pseudo-random `u64`.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in `[low, high)`.
    pub fn range(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.unit()
    }

    /// True with probability `p`.
    pub fn chance(&mut self, p: f64) -> bool {
        self.unit() < p
    }
}

/// Stable hash of any hashable value, for similarity and provenance digests.
pub fn hash_of<T: Hash>(value: &T) -> u64 {
    let mut h = DefaultHasher::new();
    value.hash(&mut h);
    h.finish()
}

/// Clamp to `[0, 1]`. Every probability in this crate goes through here, because a
/// probability outside its range is a silent source of impossible arithmetic downstream.
pub fn unit_clamp(x: f64) -> f64 {
    if x.is_nan() {
        0.0
    } else {
        x.clamp(0.0, 1.0)
    }
}

/// `x` if finite and non-negative, else `fallback`.
pub fn non_negative(x: f64, fallback: f64) -> f64 {
    if x.is_finite() && x >= 0.0 {
        x
    } else {
        fallback
    }
}

/// Smallest denominator used to avoid division by zero.
pub const EPS: f64 = 1e-9;

/// Jaccard similarity over token sets — the cheap "is this redundant" test.
pub fn jaccard(a: &str, b: &str) -> f64 {
    fn tokens(s: &str) -> Vec<String> {
        s.split(|c: char| !c.is_alphanumeric())
            // Keep short tokens that carry a digit: "12" and "13" are the whole difference
            // between two otherwise identical statements, and dropping them makes unrelated
            // goals look redundant.
            .filter(|t| t.len() > 2 || t.chars().any(|c| c.is_ascii_digit()))
            .map(|t| t.to_lowercase())
            .collect()
    }
    let ta = tokens(a);
    let tb = tokens(b);
    if ta.is_empty() || tb.is_empty() {
        return if ta.is_empty() && tb.is_empty() { 1.0 } else { 0.0 };
    }
    let mut shared = 0usize;
    let mut used = vec![false; tb.len()];
    for x in &ta {
        for (i, y) in tb.iter().enumerate() {
            if !used[i] && x == y {
                used[i] = true;
                shared += 1;
                break;
            }
        }
    }
    let union = ta.len() + tb.len() - shared;
    shared as f64 / union as f64
}

/// Least-squares slope of `ys` against `0..n`. Used for residual trends and metric slopes;
/// the brief insists on slope rather than level, because level confounds with baseline.
pub fn slope(ys: &[f64]) -> f64 {
    let n = ys.len();
    if n < 2 {
        return 0.0;
    }
    let nf = n as f64;
    let mean_x = (nf - 1.0) / 2.0;
    let mean_y = ys.iter().sum::<f64>() / nf;
    let mut num = 0.0;
    let mut den = 0.0;
    for (i, y) in ys.iter().enumerate() {
        let dx = i as f64 - mean_x;
        num += dx * (y - mean_y);
        den += dx * dx;
    }
    if den.abs() < EPS {
        0.0
    } else {
        num / den
    }
}

/// Wilson score interval for a binomial proportion. Reporting a success rate without an
/// interval is how a 2-of-3 result becomes "67% reliable" in a slide deck.
pub fn wilson(successes: u32, trials: u32, z: f64) -> (f64, f64) {
    if trials == 0 {
        return (0.0, 1.0);
    }
    let n = trials as f64;
    let p = successes as f64 / n;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let centre = (p + z2 / (2.0 * n)) / denom;
    let half = z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt()) / denom;
    (unit_clamp(centre - half), unit_clamp(centre + half))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_monotonic_and_unique() {
        let mut ids = IdGen::new(1);
        let a = ids.id();
        let b = ids.id();
        assert!(a < b);
        assert_ne!(a, b);
    }

    #[test]
    fn same_seed_same_stream() {
        let mut a = IdGen::new(42);
        let mut b = IdGen::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn unit_is_in_range() {
        let mut g = IdGen::new(7);
        for _ in 0..10_000 {
            let u = g.unit();
            assert!((0.0..1.0).contains(&u), "out of range: {u}");
        }
    }

    #[test]
    fn jaccard_bounds_and_identity() {
        assert!((jaccard("retry upload vendor", "retry upload vendor") - 1.0).abs() < 1e-9);
        assert!(jaccard("alpha beta gamma", "delta epsilon zeta") < 0.01);
        let mid = jaccard("retry upload vendor api", "retry upload vendor service");
        assert!(mid > 0.3 && mid < 0.9, "mid similarity was {mid}");
    }

    #[test]
    fn slope_detects_direction() {
        assert!(slope(&[0.0, -1.0, -2.0, -3.0]) < -0.9);
        assert!(slope(&[1.0, 2.0, 3.0]) > 0.9);
        assert_eq!(slope(&[2.0, 2.0, 2.0]), 0.0);
    }

    #[test]
    fn wilson_brackets_the_estimate() {
        let (lo, hi) = wilson(15, 20, 1.96);
        assert!(lo < 0.75 && hi > 0.75, "interval {lo}..{hi} missed 0.75");
        let (lo, hi) = wilson(0, 0, 1.96);
        assert_eq!((lo, hi), (0.0, 1.0));
    }
}
