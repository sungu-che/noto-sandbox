//! What the hands feel of the world's items: how often a grasp slips a little in them (an incipient slip, the pads'
//! stick under the alarm) on dry and on wet pads, what friction the pads read then, and what a grip planned with a
//! prior learned from that feel (`GraspInput::mu_prior`) does to the next grasps.

use noto_core::act::grasp::{self, GraspInput};
use noto_core::act::material;
use noto_core::act::reach::ReachMode;
use noto_core::{BaseKind, FingerCount, NotoConfig, Surface};
use world_sim::world::material_of;
use world_sim::{ItemKind, World, WorldConfig};

fn walker() -> World {
    let mut hand = NotoConfig::preset(FingerCount::Five);
    hand.base = BaseKind::HumanFoot;
    World::new(WorldConfig { seed: 3, cells: 32, noto: hand, parcels: Some(0), ..WorldConfig::default() }).unwrap()
}

#[derive(Debug, Default)]
struct Feel {
    grasps: u32,
    incipient: u32,
    slipped: u32,
    seen: Vec<f32>,
    grip_n: f32,
}

impl Feel {
    fn rate(&self) -> f32 {
        self.incipient as f32 / self.grasps.max(1) as f32
    }

    fn grip_mean(&self) -> f32 {
        self.grip_n / self.grasps.max(1) as f32
    }

    fn seen_mean(&self) -> Option<f32> {
        (!self.seen.is_empty()).then(|| self.seen.iter().sum::<f32>() / self.seen.len() as f32)
    }

    /// The prior the runtime's relay learns from this feel (`noto_runtime::relay::grip_prior`).
    fn prior(&self, table: f32) -> f32 {
        match self.seen_mean() {
            Some(s) if s < table => (table - self.rate() * (table - s)).clamp(0.05, table),
            _ => table,
        }
    }
}

fn feel(w: &World, kind: ItemKind, ground: Surface, prior: Option<f32>, seeds: std::ops::Range<u64>) -> Feel {
    let spec = kind.spec();
    let mut f = Feel::default();
    for seed in seeds {
        let inp = GraspInput {
            body: &w.body,
            hand: &w.hand,
            material: material_of(kind),
            mass_kg: spec.mass_kg,
            size_m: spec.piece_size_m(1.0),
            ground,
            two_arms: false,
            cargo_kg: 0.0,
            aim_sigma_m: 0.0,
            seed,
            mode: ReachMode::Front,
            aim_offset_m: [0.0; 2],
            memory_sigma_m: 0.0,
            lead: None,
            look_first: false,
            mu_prior: prior,
        };
        let r = grasp::grasp(&inp);
        if matches!(r.outcome.code, Some("reach_ground") | Some("reach_side")) {
            continue;
        }
        f.grasps += 1;
        if r.metric("incipient_slips").unwrap_or(0.0) > 0.0 {
            f.incipient += 1;
        }
        if r.outcome.code == Some("grasp_slip") {
            f.slipped += 1;
        }
        if let Some(m) = r.metric("mu_seen") {
            f.seen.push(m);
        }
        f.grip_n += r.metric("grip_n").unwrap_or(0.0);
    }
    f
}

#[test]
fn a_grip_planned_with_what_the_pads_read_slips_far_less() {
    let w = walker();
    let table = material::pad_on(material_of(ItemKind::Log)).mean;
    // A log on wet sand, gripped for the table's friction: most grasps slip a little in the hand before the reflex
    // squeezes harder, none lets go.
    let before = feel(&w, ItemKind::Log, Surface::WetSand, None, 0..60);
    assert!(before.rate() > 0.4 && before.slipped == 0, "{before:?}");
    // Under a real grip the wet pads read well under the table (it is for 10 N on dry pads).
    let seen = before.seen_mean().expect("the pads read a friction when they slipped");
    assert!(seen < 0.8 * table, "{seen} vs {table}");
    // The grip planned with what they read: few slips, still none let go, a firmer hold.
    let prior = before.prior(table);
    assert!(prior < table && prior > seen, "{prior}");
    let after = feel(&w, ItemKind::Log, Surface::WetSand, Some(prior), 60..120);
    assert!(after.rate() < 0.4 * before.rate() && after.slipped == 0, "{after:?} vs {before:?}");
    assert!(after.grip_mean() > before.grip_mean(), "{} vs {}", after.grip_mean(), before.grip_mean());
    // A prior above the table changes nothing: what the hands learn never makes them grip softer.
    let same = feel(&w, ItemKind::Log, Surface::WetSand, Some(table + 0.3), 0..60);
    assert_eq!((same.incipient, same.grip_mean()), (before.incipient, before.grip_mean()));
}

const ITEMS: [ItemKind; 5] = [ItemKind::Log, ItemKind::Stone, ItemKind::IronOre, ItemKind::Coal, ItemKind::Parcel];

#[test]
#[ignore]
fn print_what_the_hands_feel_of_each_item() {
    let w = walker();
    for kind in ITEMS {
        let table = material::pad_on(material_of(kind)).mean;
        for ground in [Surface::Dirt, Surface::WetSand] {
            let before = feel(&w, kind, ground, None, 0..200);
            let p = before.prior(table);
            let after = feel(&w, kind, ground, Some(p), 200..400);
            println!(
                "[GRIP] {:<8} {:<9} 표 {:.2} · 미세 미끄럼 {:>5.1}% (읽은 마찰 {}) · 놓침 {:>4.1}% → 사전값 {:.2} · 미세 미끄럼 {:>5.1}% · 놓침 {:>4.1}%",
                kind.id(),
                ground.id(),
                table,
                before.rate() * 100.0,
                before.seen_mean().map_or("—".to_string(), |s| format!("{s:.2}")),
                before.slipped as f32 / before.grasps.max(1) as f32 * 100.0,
                p,
                after.rate() * 100.0,
                after.slipped as f32 / after.grasps.max(1) as f32 * 100.0,
            );
        }
    }
}
