//! Drives: homeostatic deficiencies with setpoints.
//!
//! The single most important rule in this module: **a drive is not a reward.** A drive is
//! a signed deviation from a setpoint, and it contributes urgency only while dissatisfied.
//! Once the world is at or beyond the setpoint the deficiency floors at zero, so "more
//! competence" cannot become an unbounded objective.
//!
//! The second rule: **satiation decays.** A drive that has been satisfied for a long time
//! contributes a little less each tick, which rotates attention to whatever else is
//! starved. Without this, one axis monopolises arbitration forever.
//!
//! Formal anchor: Keramati & Gutkin (2014), homeostatic reinforcement learning — reward is
//! the reduction of drive deviation, not the drive itself.

use crate::types::{unit_clamp, non_negative, EPS};

/// Which deficiency a goal addresses. Closed set on purpose: a bounded drive space is what
/// makes selection tractable and auditable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DriveKind {
    /// Verified success rate is below target.
    Competence,
    /// Too few task families have ever been solved.
    Coverage,
    /// Spend per verified success is above ceiling.
    Cost,
    /// Reducible model error is above the comfortable band.
    Uncertainty,
}

impl DriveKind {
    pub const ALL: [DriveKind; 4] = [
        DriveKind::Competence,
        DriveKind::Coverage,
        DriveKind::Cost,
        DriveKind::Uncertainty,
    ];

    pub fn name(self) -> &'static str {
        match self {
            DriveKind::Competence => "competence",
            DriveKind::Coverage => "coverage",
            DriveKind::Cost => "cost",
            DriveKind::Uncertainty => "uncertainty",
        }
    }

    /// Whether a *higher* measured value means *more* satisfied. Cost is inverted: lower
    /// spend per success is better, and getting this backwards silently inverts a drive.
    pub fn higher_is_better(self) -> bool {
        // Cost and uncertainty are ceilings: lower is better. Competition and coverage are
        // floors. Getting this backwards silently inverts a drive, so it is spelled out.
        !matches!(self, DriveKind::Cost | DriveKind::Uncertainty)
    }
}

/// One drive: a measured quantity, a setpoint, a gain, and its current satisfaction.
#[derive(Debug, Clone)]
pub struct Drive {
    pub kind: DriveKind,
    /// The level treated as "satisfied". For `Cost` this is a ceiling; for the others a
    /// floor.
    pub setpoint: f64,
    /// Scales how loudly this deficiency speaks relative to the other three.
    pub gain: f64,
    /// Multiplier in `(0, 1]` that decays while satisfied and resets when the drive
    /// contributes. This is the anti-monopoly mechanism.
    pub satiation: f64,
    /// Per-tick decay applied to `satiation` while the drive is satisfied.
    pub satiation_decay: f64,
    /// Last measured value, for reporting.
    pub last_value: f64,
    /// Learnability in `[0, 1]`: how much of the remaining deficiency is actually
    /// reducible. Guards against the noisy-TV failure, where an agent chases irreducible
    /// uncertainty forever.
    pub learnability: f64,
}

impl Drive {
    pub fn new(kind: DriveKind, setpoint: f64, gain: f64) -> Self {
        Drive {
            kind,
            setpoint,
            gain,
            satiation: 1.0,
            satiation_decay: 0.98,
            last_value: setpoint,
            learnability: 1.0,
        }
    }

    pub fn with_learnability(mut self, learnability: f64) -> Self {
        self.learnability = unit_clamp(learnability);
        self
    }

    pub fn with_satiation_decay(mut self, decay: f64) -> Self {
        self.satiation_decay = unit_clamp(decay);
        self
    }

    pub fn with_gain(mut self, gain: f64) -> Self {
        self.gain = non_negative(gain, 1.0);
        self
    }

    /// Signed deficiency in `[0, 1]`. Zero when satisfied — this floor is the whole point.
    ///
    /// Scale is `|setpoint|` with a floor of `EPS`, so a drive with setpoint 0.9 treats
    /// 0.45 as 50% deficient. For `Cost`, a measured value *above* the ceiling is a
    /// deficiency; below it, zero.
    pub fn deficiency(&self, measured: f64) -> f64 {
        let scale = self.setpoint.abs().max(EPS);
        let raw = if self.kind.higher_is_better() {
            (self.setpoint - measured) / scale
        } else {
            (measured - self.setpoint) / scale
        };
        unit_clamp(raw)
    }

    /// Urgency: deficiency x gain x satiation x learnability.
    ///
    /// Read the four factors as: how far off are we, how much do we care, how recently has
    /// this axis had attention, and can anything be done about it.
    pub fn urgency(&self, measured: f64) -> f64 {
        self.deficiency(measured) * self.gain * self.satiation * self.learnability
    }

    /// Advance satiation by one tick. Satisfied drives fade; dissatisfied ones stay eager.
    pub fn tick(&mut self, measured: f64) {
        self.last_value = measured;
        if self.deficiency(measured) <= EPS {
            self.satiation = unit_clamp(self.satiation * self.satiation_decay);
        }
    }

    /// Called when a goal addressing this drive makes progress: attention is being spent,
    /// so the "recently ignored" penalty resets.
    pub fn note_attention(&mut self) {
        self.satiation = 1.0;
    }
}

/// The kernel: four drives, their last measurements, and the urgency snapshot.
#[derive(Debug, Clone)]
pub struct DriveSet {
    drives: Vec<Drive>,
}

impl DriveSet {
    /// Four drives with defensible starting setpoints. These are configuration, not
    /// emergent properties: the kernel is deliberately small and explicit so the system
    /// stays auditable.
    pub fn default_kernel() -> Self {
        DriveSet {
            drives: vec![
                // Target 85% verified success rate on active families.
                Drive::new(DriveKind::Competence, 0.85, 1.0),
                // Aim for 8 families solved at least once; saturating coverage.
                Drive::new(DriveKind::Coverage, 8.0, 0.5),
                // Ceiling of 20 units of cost per verified success.
                Drive::new(DriveKind::Cost, 20.0, 0.6),
                // Comfortable band: some reducible uncertainty is healthy, none is blind.
                Drive::new(DriveKind::Uncertainty, 0.25, 0.7),
            ],
        }
    }

    pub fn with(drives: Vec<Drive>) -> Self {
        DriveSet { drives }
    }

    pub fn get(&self, kind: DriveKind) -> &Drive {
        self.drives
            .iter()
            .find(|d| d.kind == kind)
            .expect("drive set is constructed with all four kinds")
    }

    pub fn get_mut(&mut self, kind: DriveKind) -> &mut Drive {
        self.drives
            .iter_mut()
            .find(|d| d.kind == kind)
            .expect("drive set is constructed with all four kinds")
    }

    pub fn iter(&self) -> impl Iterator<Item = &Drive> {
        self.drives.iter()
    }

    /// Urgency for every drive given fresh measurements. A missing measurement leaves the
    /// drive at its last value rather than inventing one — the runtime measures, the model
    /// never reports.
    pub fn urgencies(&self, measured: &[f64; 4]) -> [f64; 4] {
        [
            self.get(DriveKind::Competence).urgency(measured[0]),
            self.get(DriveKind::Coverage).urgency(measured[1]),
            self.get(DriveKind::Cost).urgency(measured[2]),
            self.get(DriveKind::Uncertainty).urgency(measured[3]),
        ]
    }

    /// Index of the most urgent drive, if any drive is actually dissatisfied.
    pub fn most_urgent(&self, measured: &[f64; 4]) -> Option<(DriveKind, f64)> {
        let mut best: Option<(DriveKind, f64)> = None;
        for (i, kind) in DriveKind::ALL.iter().enumerate() {
            let u = self.get(*kind).urgency(measured[i]);
            if u > EPS && best.map_or(true, |(_, b)| u > b) {
                best = Some((*kind, u));
            }
        }
        best
    }

    /// Advance every drive one tick.
    pub fn tick(&mut self, measured: &[f64; 4]) {
        for (i, kind) in DriveKind::ALL.iter().enumerate() {
            self.get_mut(*kind).tick(measured[i]);
        }
    }

    /// Mark that attention was paid to a drive (a goal addressing it ran).
    pub fn note_attention(&mut self, kind: DriveKind) {
        self.get_mut(kind).note_attention();
    }

    /// Total urgency: a single scalar for "is anything worth doing right now".
    pub fn total_urgency(&self, measured: &[f64; 4]) -> f64 {
        self.urgencies(measured).iter().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deficiency_floors_at_setpoint() {
        let d = Drive::new(DriveKind::Competence, 0.85, 1.0);
        assert_eq!(d.deficiency(0.85), 0.0, "at setpoint there is no deficiency");
        assert_eq!(d.deficiency(0.99), 0.0, "beyond setpoint still zero, never negative");
        assert!((d.deficiency(0.0) - 1.0).abs() < 1e-9, "zero competence is a full deficiency");
    }

    #[test]
    fn satisfied_drives_contribute_no_urgency() {
        let d = Drive::new(DriveKind::Competence, 0.85, 1.0);
        assert_eq!(d.urgency(0.95), 0.0);
        assert!(d.urgency(0.80) > 0.0);
    }

    #[test]
    fn cost_drive_is_inverted() {
        let d = Drive::new(DriveKind::Cost, 20.0, 1.0);
        assert_eq!(d.deficiency(5.0), 0.0, "cheap is satisfied");
        assert!(d.deficiency(40.0) > 0.9, "expensive is a deficiency");
    }

    #[test]
    fn satiation_decays_only_while_satisfied_and_resets_on_attention() {
        let mut d = Drive::new(DriveKind::Competence, 0.85, 1.0);
        for _ in 0..10 {
            d.tick(0.95); // satisfied
        }
        assert!(d.satiation < 1.0, "satisfied drive should fade: {}", d.satiation);
        for _ in 0..10 {
            d.tick(0.10); // starving
        }
        let starved_satiation = d.satiation;
        d.note_attention();
        assert_eq!(d.satiation, 1.0);
        assert!(starved_satiation < 1.0);
    }

    #[test]
    fn unlearnable_uncertainty_is_suppressed() {
        let naive = Drive::new(DriveKind::Uncertainty, 0.25, 1.0);
        let filtered = Drive::new(DriveKind::Uncertainty, 0.25, 1.0).with_learnability(0.0);
        assert!(naive.urgency(0.9) > 0.5);
        assert_eq!(filtered.urgency(0.9), 0.0, "noisy tv: irreducible uncertainty must not compete");
    }

    #[test]
    fn kernel_prefers_the_most_starved_drive() {
        let set = DriveSet::default_kernel();
        // competence 0.0 (full deficiency), coverage 8 (satisfied), cost cheap, uncertainty low
        let measured = [0.0, 8.0, 1.0, 0.0];
        let (kind, _) = set.most_urgent(&measured).expect("something is starving");
        assert_eq!(kind, DriveKind::Competence);
    }

    #[test]
    fn fully_satisfied_kernel_has_nothing_urgent() {
        let set = DriveSet::default_kernel();
        let measured = [0.95, 12.0, 1.0, 0.0];
        assert_eq!(set.most_urgent(&measured), None, "a satisfied system must be able to idle");
    }
}
