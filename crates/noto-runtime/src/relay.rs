//! What the robot felt, back into how it acts. A hand that felt an item start to slip in it — the pads' stick ratio fell
//! under the alarm and the grip read the friction off them (`mu_seen`) — remembers that of the item and the state of
//! its pads (dry, wet), and grips that item harder from the start the next time instead of waiting to slip again.
//!
//! Only what the robot sensed comes back: whether its pads slipped and what friction they read when they did, counted
//! per item and pad state. Never a deviation from a person's value (the benchmark's reference axes), and never the
//! friction the simulator drew for a grasp, which the hands cannot know.

use noto_sds::{Scope, SdsStore, Track};
use serde::Serialize;

/// The scope a grasp's feel of an item goes under: the item and the state of the pads.
pub fn pad_scope(item: &str, pads: &str) -> Scope {
    Scope::new(Track::Act, "grasp:pad", &world_sim::grip_key(item, pads))
}

/// How many table-planned grasps of an item, its pads in one state, the hands take before they trust their own feel of
/// it [추정: the act ledger keeps an item's scope from 6].
pub const GRIP_PRIOR_MIN_N: u64 = 8;

/// The friction the hands expect of an item from their own grasps of it, and what that rests on.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct GripPrior {
    /// What the grip plans with (≤ the table).
    pub mu: f32,
    /// The material's table value, what it planned with before.
    pub table_mu: f32,
    /// The share of grasps that slipped a little (an incipient slip), and the friction read off the pads then.
    pub slip_rate: f32,
    pub mu_seen: Option<f32>,
    pub grasps: u64,
}

/// The table's friction `table_mu` moved toward what the pads read at an incipient slip, as far as such slips came
/// while the grip still planned with the table: every such grasp slipping puts it at that reading, none leaves the
/// table. The share is of table-planned grasps only, so a prior that stops the slips does not talk itself back up to
/// the table that caused them. It never goes above the table — what the hands learn makes them grip harder, never
/// softer. `None` before `GRIP_PRIOR_MIN_N` table-planned grasps of this item and pad state.
pub fn grip_prior(sds: &SdsStore, item: &str, pads: &str, table_mu: f32) -> Option<GripPrior> {
    let scope = pad_scope(item, pads);
    let ((rate, n), at) = sds.agreement_rate_scoped(&scope, "incipient")?;
    // Only this item on these pads: a wider scope mixes items and pad states.
    if at != scope.key_secondary() || n < GRIP_PRIOR_MIN_N {
        return None;
    }
    let seen = sds.exact_baseline(&scope, "mu_seen").map(|b| b.0 as f32).filter(|m| m.is_finite() && *m > 0.0);
    let rate = (rate as f32).clamp(0.0, 1.0);
    let mu = match seen {
        Some(s) if s < table_mu => table_mu - rate * (table_mu - s),
        _ => table_mu,
    };
    Some(GripPrior { mu: mu.clamp(0.05, table_mu), table_mu, slip_rate: rate, mu_seen: seen, grasps: n })
}

impl crate::runtime::Runtime {
    /// The learned grip of `item` on pads in state `pads` into the world, where the next grasp of it plans with it.
    pub(crate) fn relay_grip(&mut self, item: world_sim::ItemKind, pads: &str) {
        let key = world_sim::grip_key(item.id(), pads);
        match grip_prior(&self.sds, item.id(), pads, self.world.pad_mu(item)) {
            Some(p) if p.mu < p.table_mu => {
                self.world.grip_mu.insert(key, p.mu);
            }
            _ => {
                self.world.grip_mu.remove(&key);
            }
        }
    }

    /// Every item's learned grip, on every state of the pads, into the world: at start, from the ledger.
    pub(crate) fn relay_all_grips(&mut self) {
        for item in world_sim::ItemKind::ALL {
            for pads in PAD_STATES {
                self.relay_grip(item, pads);
            }
        }
    }

    /// What the hands have learned to expect, for the screen: item, pad state and the prior.
    pub fn grip_priors(&self) -> Vec<(world_sim::ItemKind, &'static str, GripPrior)> {
        world_sim::ItemKind::ALL
            .into_iter()
            .flat_map(|item| PAD_STATES.into_iter().filter_map(move |pads| grip_prior(&self.sds, item.id(), pads, self.world.pad_mu(item)).map(|p| (item, pads, p))))
            .collect()
    }
}

/// The states of the pads a grip is learned under (`PadSurface::id`).
pub const PAD_STATES: [&str; 3] = ["dry", "wet", "dusty"];

/// What the hands have felt of an item on pads in a state so far, learned enough or not: the share of table-planned
/// grasps that slipped a little, how many there were (0 until the ledger keeps the item's own scope), and the friction
/// the pads read when they slipped.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct GripFeel {
    pub slip_rate: f32,
    pub grasps: u64,
    pub mu_seen: Option<f32>,
}

pub fn grip_feel(sds: &SdsStore, item: &str, pads: &str) -> GripFeel {
    let scope = pad_scope(item, pads);
    let (slip_rate, grasps) = match sds.agreement_rate_scoped(&scope, "incipient") {
        Some(((rate, n), at)) if at == scope.key_secondary() => (rate as f32, n),
        _ => (0.0, 0),
    };
    let mu_seen = sds.exact_baseline(&scope, "mu_seen").map(|b| b.0 as f32);
    GripFeel { slip_rate, grasps, mu_seen }
}

/// Records what one grasp felt: whether the pads slipped a little (counted for grasps the grip planned with the table,
/// `table_planned`), and the friction they read when they did (from any grasp).
pub fn record_grasp_feel(sds: &mut SdsStore, item: &str, pads: &str, table_planned: bool, incipient_slips: f32, mu_seen: Option<f32>) {
    let scope = pad_scope(item, pads);
    if table_planned {
        sds.record_agreement(&scope, "incipient", incipient_slips > 0.0);
    }
    if let Some(m) = mu_seen {
        sds.record_baseline(&scope, "mu_seen", m as f64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_hands_grip_harder_after_feeling_an_item_slip_and_never_softer() {
        let mut sds = SdsStore::memory();
        // Too few grasps say nothing yet.
        for _ in 0..3 {
            record_grasp_feel(&mut sds, "log", "wet", true, 1.0, Some(0.42));
        }
        assert!(grip_prior(&sds, "log", "wet", 0.60).is_none());
        // Five of eight grasps of a wet log slipped a little, reading 0.42 off the pads.
        for i in 0..5 {
            record_grasp_feel(&mut sds, "log", "wet", true, if i % 3 == 0 { 1.0 } else { 0.0 }, (i % 3 == 0).then_some(0.42));
        }
        let p = grip_prior(&sds, "log", "wet", 0.60).expect("eight grasps");
        assert_eq!(p.grasps, 8);
        assert!((p.slip_rate - 5.0 / 8.0).abs() < 1e-6, "{p:?}");
        assert!((p.mu - (0.60 - p.slip_rate * (0.60 - 0.42))).abs() < 1e-4 && p.mu < 0.5, "{p:?}");
        // Grasps planned with that prior stop slipping; they do not count against the table's share, so the prior holds.
        for _ in 0..20 {
            record_grasp_feel(&mut sds, "log", "wet", false, 0.0, None);
        }
        assert_eq!(grip_prior(&sds, "log", "wet", 0.60), Some(p));
        // Another item, or the same item dry, has its own feel.
        assert!(grip_prior(&sds, "log", "dry", 0.60).is_none() && grip_prior(&sds, "stone", "wet", 0.83).is_none());
        // Pads that read more than the table never make the grip softer.
        let mut sds = SdsStore::memory();
        for _ in 0..8 {
            record_grasp_feel(&mut sds, "stone", "dry", true, 1.0, Some(0.95));
        }
        assert_eq!(grip_prior(&sds, "stone", "dry", 0.83).unwrap().mu, 0.83);
        // Grasps that never slip leave the table as it is.
        let mut sds = SdsStore::memory();
        for _ in 0..8 {
            record_grasp_feel(&mut sds, "parcel", "dry", true, 0.0, None);
        }
        assert_eq!(grip_prior(&sds, "parcel", "dry", 0.60).unwrap().mu, 0.60);
    }

    #[test]
    fn the_relay_reads_no_reference_axis_even_when_one_is_there() {
        // A benchmark's deviation axes on the same scope, poisoned: the prior must not move with them.
        let mut clean = SdsStore::memory();
        let mut poisoned = SdsStore::memory();
        for sds in [&mut clean, &mut poisoned] {
            for i in 0..8 {
                record_grasp_feel(sds, "log", "wet", true, if i % 2 == 0 { 1.0 } else { 0.0 }, (i % 2 == 0).then_some(0.40));
            }
        }
        let scope = pad_scope("log", "wet");
        for _ in 0..8 {
            for suffix in ["dev", "absdev", "dev2"] {
                poisoned.record_baseline(&scope, &format!("mu_seen:{suffix}"), -40.0);
                poisoned.record_baseline(&scope, &format!("incipient:{suffix}"), 40.0);
            }
            poisoned.record_agreement(&scope, &format!("incipient:{}", "in_band"), false);
        }
        assert_eq!(grip_prior(&clean, "log", "wet", 0.60), grip_prior(&poisoned, "log", "wet", 0.60));
        // That the file does not even name them is `source_split::decisions_never_read_reference_axes`: this is a
        // decision file (`DECISION_FILES`).
    }
}
