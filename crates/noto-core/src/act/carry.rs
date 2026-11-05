//! A load riding on a moving robot. Nothing holds it there but friction and the hands, so what the body does reaches
//! it: braking and pulling away, the swing of a corner, the jolt of a step. A load on tines is a block on a plate: it
//! stays while its inertia is less than the static friction on its weight, then slides on kinetic friction until it
//! sticks again or its centre passes the tine tips or the outer edge of a tine. A load in the hands has a grip on top
//! of that, and a limit: when the force it takes to keep it there passes what the hands can give, it is gone.
//!
//! A step reaches the load through the legs. Walking off a step with straight knees the body falls the whole height
//! and stops within the give of the sole and a nearly locked knee; a robot that saw the step coming bends its knees a
//! stride ahead and lowers itself instead, so the same step costs a fraction of the shock.

use serde::Serialize;

use crate::balance::{Body, G};
use crate::dims::Dims;

use super::material::{self, ItemMaterial};

/// The share of the slip limit a careful robot uses in a corner or a stop (`LoadSpec::safe_accel`) [추정].
pub const SAFE_SHARE: f32 = 0.85;
/// A lift truck's mast tips back while it travels, and the load leans on the carriage (deg) [추정: 5° of rear tilt].
pub const MAST_TILT_DEG: f32 = 5.0;
/// The share of the hands' strongest grip they keep up on the way (the rest is left in reserve) [추정].
pub const TRANSIT_GRIP: f32 = 0.35;
/// How far a load may ride off the middle of the palms before the hands lose it (m) [추정].
pub const PALM_SLACK_M: f32 = 0.06;
/// The load pressing into the chest stops there (m) [추정].
pub const CHEST_SLACK_M: f32 = 0.03;
/// The relative speed below which a sliding load is at rest on its support (m/s).
const STICK_MPS: f32 = 0.01;
/// The give in a leg not ready for the ground: the sole, the ankle and what the knee yields by itself, for a 1.21 m
/// body (m) [추정: a 10 cm step down not seen comes to about 1.5 g over its weight at a 1.65 m body's hips].
pub const LEG_GIVE_M: f32 = 0.048;
/// The most the hips go down for a step (m) [추정].
pub const CROUCH_MAX_M: f32 = 0.18;
/// The share of a step's height the hips go down for it [추정].
pub const CROUCH_SHARE: f32 = 0.8;
/// The share of the crouch the knees still have left to give on touchdown [추정].
pub const CROUCH_GIVE: f32 = 0.6;
/// How long a walker takes to rise onto a step (s), and how much harder a stiff leg meets its riser [추정].
pub const STEP_UP_S: f32 = 0.6;
pub const STIFF_STEP_UP: f32 = 3.0;
/// The forward lean that goes with a crouch, per metre the hips go down (deg/m) [추정: 2° for 10 cm].
pub const LEAN_DEG_PER_M: f32 = 20.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Holder {
    Tines,
    Hands,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct LoadSpec {
    pub holder: Holder,
    pub mass_kg: f32,
    pub mu_s: f32,
    pub mu_k: f32,
    /// What presses the load on besides its weight (N): the hands' transit grip.
    pub clamp_n: f32,
    /// The most force the hold can give in any direction (N); infinite on tines.
    pub hold_n: f32,
    /// How far the load's centre may go, forward and sideways, before it comes off, and back until something stops it.
    pub fwd_m: f32,
    pub side_m: f32,
    pub back_m: f32,
    /// How much the support leans the load back on (m/s²): the mast's tilt.
    pub tilt_back_mps2: f32,
}

impl LoadSpec {
    /// A load `len_m` long on the tines. Layers above the first lie on each other, not on steel.
    pub fn tines(d: &Dims, item: ItemMaterial, mass_kg: f32, len_m: f32, layers: usize) -> LoadSpec {
        let (s, k) = material::on_steel(item);
        let (mu_s, mu_k) = if layers > 1 {
            let (a, b) = material::on_itself(item);
            (s.mean.min(a.mean), k.mean.min(b.mean))
        } else {
            (s.mean, k.mean)
        };
        LoadSpec {
            holder: Holder::Tines,
            mass_kg,
            mu_s,
            mu_k,
            clamp_n: 0.0,
            hold_n: f32::INFINITY,
            fwd_m: (d.tine_len_m - 0.5 * len_m).max(0.05),
            side_m: 0.5 * (d.tine_spread_m + d.tine_w_m),
            back_m: 0.0,
            tilt_back_mps2: G * MAST_TILT_DEG.to_radians().sin(),
        }
    }

    /// A load in two hands. `hold_n` is the friction force the hands can give at full grip on this item.
    pub fn hands(item: ItemMaterial, mass_kg: f32, size_m: [f32; 3], hold_n: f32) -> LoadSpec {
        let pad = material::pad_on(item);
        let along = 0.5 * size_m[0].max(size_m[2]);
        LoadSpec { holder: Holder::Hands, mass_kg, mu_s: pad.mean, mu_k: material::KINETIC_SHARE * pad.mean, clamp_n: TRANSIT_GRIP * hold_n / pad.mean.max(0.05), hold_n, fwd_m: along + PALM_SLACK_M, side_m: along + PALM_SLACK_M, back_m: CHEST_SLACK_M, tilt_back_mps2: 0.0 }
    }

    /// The same load with its friction drawn for this haul: `u` is a standard normal draw.
    pub fn with_draw(self, cov: f32, u: f32) -> LoadSpec {
        let k = (1.0 + cov * u).clamp(0.5, 1.5);
        LoadSpec { mu_s: self.mu_s * k, mu_k: self.mu_k * k, ..self }
    }

    fn normal_n(&self, a_z: f32) -> f32 {
        (self.mass_kg * (G + a_z) + self.clamp_n).max(0.0)
    }

    /// The horizontal acceleration of the support at which the load starts to slide (m/s²), going straight.
    pub fn slip_accel(&self) -> f32 {
        self.mu_s * self.normal_n(0.0) / self.mass_kg.max(1e-3)
    }

    /// The horizontal acceleration a careful robot keeps to with this load on (m/s²).
    pub fn safe_accel(&self) -> f32 {
        SAFE_SHARE * self.slip_accel()
    }

    /// The force it takes to keep the load with the body under these accelerations (N).
    pub fn force_n(&self, a: [f32; 2], a_z: f32) -> f32 {
        self.mass_kg * (a[0] * a[0] + a[1] * a[1] + (G + a_z) * (G + a_z)).sqrt()
    }
}

/// Where the load is on its support, as it rides.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct LoadState {
    /// Off its seat: forward, left (m).
    pub shift: [f32; 2],
    /// Sliding: forward, left (m/s).
    pub vel: [f32; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Lost {
    /// Past the tips, or out of the front of the hands.
    Forward,
    /// Off the side.
    Side,
    /// The hold could not give the force.
    Grip,
}

impl Lost {
    pub fn id(self) -> &'static str {
        match self {
            Lost::Forward => "forward",
            Lost::Side => "side",
            Lost::Grip => "grip",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct RideStep {
    pub state: LoadState,
    /// The inertia over the static friction: 1 is where it starts to slide.
    pub use_ratio: f32,
    pub sliding: bool,
    pub lost: Option<Lost>,
}

/// One step of a load on its support. `a` is the support's acceleration in its own frame (forward, left), `a_z` its
/// upward acceleration.
pub fn ride(spec: &LoadSpec, st: LoadState, a: [f32; 2], a_z: f32, dt: f32) -> RideStep {
    if spec.force_n(a, a_z) > spec.hold_n {
        return RideStep { state: st, use_ratio: 1.0, sliding: false, lost: Some(Lost::Grip) };
    }
    let n = spec.normal_n(a_z);
    let m = spec.mass_kg.max(1e-3);
    // The inertia the load feels on the support, against the lean of the mast.
    let f = [-a[0] - spec.tilt_back_mps2, -a[1]];
    let f_mag = f[0].hypot(f[1]);
    let limit = spec.mu_s * n / m;
    let use_ratio = f_mag / limit.max(1e-3);
    let (mut v, mut s) = (st.vel, st.shift);
    let moving = v[0].hypot(v[1]) > STICK_MPS;
    if !moving && f_mag <= limit {
        return RideStep { state: LoadState { shift: s, vel: [0.0, 0.0] }, use_ratio, sliding: false, lost: None };
    }
    let fric = spec.mu_k * n / m;
    let speed = v[0].hypot(v[1]);
    let dir = if speed > 1e-4 {
        [v[0] / speed, v[1] / speed]
    } else if f_mag > 1e-4 {
        [f[0] / f_mag, f[1] / f_mag]
    } else {
        [0.0, 0.0]
    };
    let acc = [f[0] - fric * dir[0], f[1] - fric * dir[1]];
    let nv = [v[0] + acc[0] * dt, v[1] + acc[1] * dt];
    // Friction stops a slide, it does not turn it round.
    let reversed = nv[0] * v[0] + nv[1] * v[1] < 0.0 && f_mag <= limit;
    v = if reversed { [0.0, 0.0] } else { nv };
    s = [s[0] + v[0] * dt, s[1] + v[1] * dt];
    if s[0] < -spec.back_m {
        s[0] = -spec.back_m;
        v[0] = v[0].max(0.0);
    }
    let lost = if s[0] > spec.fwd_m {
        Some(Lost::Forward)
    } else if s[1].abs() > spec.side_m {
        Some(Lost::Side)
    } else {
        None
    };
    RideStep { state: LoadState { shift: s, vel: v }, use_ratio, sliding: true, lost }
}

/// A sudden change of the support's velocity (`dv`, forward and left, m/s), over faster than a tick: the load keeps
/// going at what the friction does not take off in `dt_s`.
pub fn jolt(spec: &LoadSpec, st: LoadState, dv: [f32; 2], dt_s: f32) -> RideStep {
    let dt = dt_s.max(1e-4);
    let a = [dv[0] / dt, dv[1] / dt];
    let mut r = ride(spec, st, a, 0.0, dt);
    if r.lost.is_none() && dt < 0.05 {
        // What is left of the slide after the jolt runs out on kinetic friction over the next moment.
        let mut t = 0.0;
        while t < 0.6 && r.lost.is_none() && r.state.vel[0].hypot(r.state.vel[1]) > STICK_MPS {
            let again = ride(spec, r.state, [0.0, 0.0], 0.0, 0.02);
            r = RideStep { use_ratio: r.use_ratio, sliding: true, ..again };
            t += 0.02;
        }
    }
    r
}

/// How far the hips go down for a step `h_m` high.
pub fn crouch_for(h_m: f32, d: &Dims) -> f32 {
    (CROUCH_SHARE * h_m.abs()).min(CROUCH_MAX_M.min(0.25 * d.hip_m))
}

/// The lean that goes with a crouch of `crouch_m` (deg).
pub fn crouch_lean_deg(crouch_m: f32) -> f32 {
    LEAN_DEG_PER_M * crouch_m
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct StepShock {
    /// The peak upward acceleration of the hips (m/s²).
    pub a_z: f32,
    pub stroke_m: f32,
    pub drop_m: f32,
}

/// What a step reaches the body with. `h_m` is the step (up is positive) and `crouch_m` how far the knees were bent for
/// it when the foot came down.
pub fn step_shock(body: &Body, h_m: f32, crouch_m: f32) -> StepShock {
    let k = body.dims.k;
    let give = LEG_GIVE_M * k + CROUCH_GIVE * crouch_m;
    if h_m < 0.0 {
        // Down: what the bent knees did not take down beforehand is a fall, stopped within their give.
        let drop = (-h_m - crouch_m).max(0.0);
        let v2 = 2.0 * G * drop;
        StepShock { a_z: v2 / (2.0 * give.max(1e-3)), stroke_m: give, drop_m: drop }
    } else {
        let t = STEP_UP_S * k.sqrt();
        let lift = 4.0 * h_m / (t * t);
        let stiff = if crouch_m >= CROUCH_SHARE * h_m.min(CROUCH_MAX_M) - 1e-3 { 1.0 } else { STIFF_STEP_UP };
        StepShock { a_z: lift * stiff, stroke_m: give, drop_m: 0.0 }
    }
}

/// The shock of a wheel or a track taking a curb `h_m` high at `speed` (m/s): it rises over the wheel's own arc.
pub fn curb_shock(body: &Body, h_m: f32, speed: f32) -> StepShock {
    let r = body.dims.wheel_r_m.max(0.02);
    let h = h_m.abs().min(r);
    let contact = ((2.0 * r * h - h * h).max(0.0)).sqrt();
    let v = speed.max(0.05);
    let t = (contact / v).max(0.05);
    StepShock { a_z: 2.0 * h / (t * t), stroke_m: 0.0, drop_m: h }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::BaseKind;
    use crate::hand::FingerCount;
    use crate::parts::BodyPlan;

    fn truck() -> Body {
        Body::new(&BodyPlan::standard(), BaseKind::Wheel, FingerCount::Five)
    }

    fn walker() -> Body {
        Body::new(&BodyPlan::standard(), BaseKind::HumanFoot, FingerCount::Five)
    }

    fn log_on_tines() -> LoadSpec {
        LoadSpec::tines(&truck().dims, ItemMaterial::Wood, 4.2, 0.5, 1)
    }

    #[test]
    fn a_load_stays_put_below_the_friction_and_slides_past_it() {
        let s = log_on_tines();
        let limit = s.slip_accel();
        assert!(limit > 2.0 && limit < 8.0, "{limit}");
        let held = ride(&s, LoadState::default(), [-0.8 * limit, 0.0], 0.0, 0.1);
        assert!(!held.sliding && held.state.shift == [0.0, 0.0] && held.lost.is_none(), "{held:?}");
        let sliding = ride(&s, LoadState::default(), [-1.6 * limit, 0.0], 0.0, 0.1);
        assert!(sliding.sliding && sliding.state.shift[0] > 0.0 && sliding.use_ratio > 1.0, "{sliding:?}");
    }

    #[test]
    fn the_mast_tilt_keeps_a_load_on_under_braking_it_would_not_stand_on_a_flat_plate() {
        let s = log_on_tines();
        let flat = LoadSpec { tilt_back_mps2: 0.0, ..s };
        let a = [-(s.slip_accel() + 0.5 * s.tilt_back_mps2), 0.0];
        assert!(!ride(&s, LoadState::default(), a, 0.0, 0.1).sliding);
        assert!(ride(&flat, LoadState::default(), a, 0.0, 0.1).sliding);
    }

    #[test]
    fn pulling_away_presses_the_load_on_the_backrest_and_it_does_not_fall_off() {
        let s = log_on_tines();
        let r = ride(&s, LoadState::default(), [3.0 * s.slip_accel(), 0.0], 0.0, 0.2);
        assert_eq!(r.state.shift[0], 0.0, "the backrest stops it");
        assert!(r.lost.is_none());
    }

    #[test]
    fn hard_braking_runs_a_load_off_the_tips_and_a_gentle_stop_does_not() {
        let s = log_on_tines();
        let mut st = LoadState::default();
        let mut lost = None;
        for _ in 0..60 {
            let r = ride(&s, st, [-2.0 * s.slip_accel(), 0.0], 0.0, 0.05);
            st = r.state;
            lost = lost.or(r.lost);
        }
        assert_eq!(lost, Some(Lost::Forward), "{st:?}");
        let mut st = LoadState::default();
        for _ in 0..60 {
            let r = ride(&s, st, [-0.9 * s.slip_accel(), 0.0], 0.0, 0.05);
            assert!(r.lost.is_none());
            st = r.state;
        }
        assert_eq!(st.shift, [0.0, 0.0]);
    }

    #[test]
    fn a_corner_taken_too_fast_slides_a_load_off_the_side_of_the_tines() {
        let s = log_on_tines();
        let mut st = LoadState::default();
        let mut lost = None;
        for _ in 0..100 {
            let r = ride(&s, st, [0.0, 1.5 * s.slip_accel()], 0.0, 0.05);
            st = r.state;
            lost = lost.or(r.lost);
        }
        assert_eq!(lost, Some(Lost::Side));
    }

    #[test]
    fn a_load_that_stops_sliding_stays_where_it_stopped() {
        let s = log_on_tines();
        let r = jolt(&s, LoadState::default(), [-1.4, 0.0], 0.02);
        assert!(r.lost.is_none() && r.state.shift[0] > 0.02, "{r:?}");
        assert_eq!(ride(&s, LoadState { vel: [0.0, 0.0], ..r.state }, [0.0, 0.0], 0.0, 0.1).state.shift, r.state.shift);
    }

    #[test]
    fn a_stack_slides_sooner_than_one_piece_on_the_steel() {
        let d = truck().dims;
        let one = LoadSpec::tines(&d, ItemMaterial::IronOre, 4.0, 0.1, 1);
        let two = LoadSpec::tines(&d, ItemMaterial::IronOre, 4.0, 0.1, 2);
        assert!(two.mu_s <= one.mu_s, "{two:?} {one:?}");
    }

    #[test]
    fn the_hands_lose_what_they_cannot_hold_and_a_grip_stops_a_slide() {
        let hold = 60.0;
        let s = LoadSpec::hands(ItemMaterial::Wood, 4.2, [0.5, 0.14, 0.14], hold);
        let bare = LoadSpec { clamp_n: 0.0, ..s };
        assert!(s.slip_accel() > bare.slip_accel());
        assert!(ride(&s, LoadState::default(), [0.0, 0.0], 0.0, 0.1).lost.is_none());
        assert_eq!(ride(&s, LoadState::default(), [0.0, 0.0], 60.0, 0.1).lost, Some(Lost::Grip));
    }

    #[test]
    fn bent_knees_take_most_of_a_step_down() {
        let b = walker();
        let h = -0.15;
        let stiff = step_shock(&b, h, 0.0);
        let bent = step_shock(&b, h, crouch_for(h, &b.dims));
        assert!(stiff.a_z > 18.0, "falls the height and stops in the sole: {stiff:?}");
        assert!(bent.a_z < 0.2 * stiff.a_z, "{bent:?} vs {stiff:?}");
        assert!(crouch_for(0.5, &b.dims) <= CROUCH_MAX_M);
        assert!((crouch_for(0.10, &b.dims) - 0.08).abs() < 1e-5);
        let up_stiff = step_shock(&b, 0.15, 0.0);
        let up_bent = step_shock(&b, 0.15, crouch_for(0.15, &b.dims));
        assert!(up_bent.a_z < up_stiff.a_z);
    }

    #[test]
    fn a_curb_jolts_more_the_faster_it_is_taken() {
        let b = truck();
        assert!(curb_shock(&b, 0.05, 1.0).a_z > 3.0 * curb_shock(&b, 0.05, 0.3).a_z);
    }

    #[test]
    fn cargo_shift_relays_to_zmp_and_toe_patch() {
        use crate::act::cocarry;
        use crate::balance::{FeedForward, Posture, Source};
        use crate::kin::{self, KinLoads};
        use crate::parts::Tier;
        // A mid walker with a 2.4 kg log in both hands, 35 cm before its body.
        let b = Body::new(&BodyPlan::uniform(Tier::Mid), BaseKind::HumanFoot, FingerCount::Five);
        let pose = cocarry::carry_pose(&b, cocarry::carry_hand_z(&b));
        let arms = (pose.shoulder_deg, pose.elbow_deg);
        let ff = FeedForward::default();
        let brake_at = |a: f32| Posture { source: Source::Accel, accel_mps2: -a, moving: true, ..Posture::quiet(2.4) };
        // Hard braking (1.5 m/s²) is fine for the load in the bay, where the models used to put it, and runs the ZMP off
        // the toes for the load in the hands, where it is.
        assert!(b.solve(&brake_at(1.5), &ff, 1e3).margin > 0.2);
        assert!(b.solve(&brake_at(1.5).held_in_hands(arms, [0.0, 0.0]), &ff, 1e3).margin < 0.0);
        // So a walker carrying in its hands plans its stops for the load there. With a log the stop it ramps to anyway
        // (top speed in half a second) is the gentler; a heavier load in the hands leaves less room than that.
        let firm = crate::surface::Surface::Firm;
        let plan = |kg: f32, hands: bool| {
            let limits = b.drive_limits(&BaseKind::HumanFoot.spec(), kg);
            let from = if hands { b.quiet(kg).held_in_hands(arms, [0.0, 0.0]) } else { b.quiet(kg) };
            b.planned_accel_from(&limits, &from, 0.0, firm, -1.0, 0.5)
        };
        let in_hands = plan(2.4, true);
        assert!(in_hands <= plan(2.4, false) + 1e-6 && in_hands > 0.2, "{in_hands}");
        for kg in [4.0, 6.0] {
            assert!(plan(kg, true) < 0.8 * plan(kg, false), "{kg} kg: {} vs {}", plan(kg, true), plan(kg, false));
        }
        // It brakes at that, and the log slides 4 cm forward in its hands.
        let braking = brake_at(in_hands);
        let bay = b.solve(&braking, &ff, 1e3);
        let held = b.solve(&braking.held_in_hands(arms, [0.0, 0.0]), &ff, 1e3);
        let slid = b.solve(&braking.held_in_hands(arms, [0.04, 0.0]), &ff, 1e3);
        assert!(held.margin >= 0.0, "{}", held.margin);
        let shoulder = |r: &crate::balance::BalanceReport| r.joint("shoulder_r").map_or(0.0, |j| j.peak_ratio);
        // In the bay the arms hang empty; in the hands they hold it, and more the further out it slid.
        assert!(shoulder(&held) > shoulder(&bay) + 0.05, "{} vs {}", shoulder(&held), shoulder(&bay));
        assert!(shoulder(&slid) > shoulder(&held), "{} vs {}", shoulder(&slid), shoulder(&held));
        assert!(slid.cargo_x_m > held.cargo_x_m + 0.039 && held.cargo_x_m > bay.cargo_x_m + 0.1, "{} {} {}", slid.cargo_x_m, held.cargo_x_m, bay.cargo_x_m);
        // The slide is not planned: the feed-forward does not see it coming and the margin dips for it.
        assert!(slid.margin < held.margin, "{} vs {}", slid.margin, held.margin);
        assert!((slid.ff_predicted_m - held.ff_predicted_m).abs() < 1e-6);
        // The same moment in the joint statics: the ZMP goes forward by the log's share of the slide, onto the toes.
        let loads = |shift: [f32; 2]| KinLoads::carrying(2.4, 2, [-in_hands, 0.0, 0.0], shift);
        let at = |shift: [f32; 2]| kin::solve(&b, &kin::KinPose { cargo_kg: 0.0, ..pose }, &loads(shift));
        let (still, moved) = (at([0.0, 0.0]), at([0.04, 0.0]));
        let m = b.mass_kg() + 2.4;
        let want = 2.4 / m * 0.04;
        assert!((moved.zmp[0] - still.zmp[0] - want).abs() < 0.3 * want, "ZMP +{:.4} m vs {want:.4}", moved.zmp[0] - still.zmp[0]);
        let toe = |r: &kin::KinReport| r.contact("foot_r").and_then(|c| c.sole).map_or(0.0, |s| s.patches[3].normal_n);
        assert!(toe(&moved) > toe(&still), "toe patch {} vs {}", toe(&moved), toe(&still));
        let sh = |r: &kin::KinReport| r.joint("shoulder_r").map_or(0.0, |j| j.ratio);
        assert!(sh(&moved) > sh(&still), "{} vs {}", sh(&moved), sh(&still));
        // Sideways it hangs more on the nearer hand.
        let left = kin::solve(&b, &kin::KinPose { cargo_kg: 0.0, ..pose }, &KinLoads::carrying(2.4, 2, [0.0; 3], [0.0, 0.05]));
        assert!(left.joint("shoulder_l").unwrap().ratio > left.joint("shoulder_r").unwrap().ratio);
        let side = b.solve(&Posture::quiet(2.4).held_in_hands(arms, [0.0, 0.05]), &ff, 1e3);
        assert!(shoulder_l(&side) > shoulder(&side) && side.com[1] > held.com[1]);
        fn shoulder_l(r: &crate::balance::BalanceReport) -> f32 {
            r.joint("shoulder_l").map_or(0.0, |j| j.peak_ratio)
        }
    }
}
