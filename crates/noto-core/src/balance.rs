use serde::{Deserialize, Serialize};

use crate::base::{BaseKind, BaseSpec};
use crate::dims::Dims;
use crate::hand::FingerCount;
use crate::morph::{column_kg, mounted_belly, mounted_chest, mounted_head, Manip, Morphology};
use crate::parts::*;
use crate::surface::{self, Contact, Surface};

pub const G: f32 = 9.81;
pub const REACTION_DELAY_S: f32 = 0.10;
pub const MARGIN_KEEP: f32 = 0.2;
pub const KNEE_SAFETY: f32 = 1.5;
pub const MAST_KG: f32 = 1.0;
pub const GRIP_SAFETY: f32 = 2.0;
pub const STOP_DISTANCE_M: f32 = 0.5;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    Quiet,
    ArmRaise,
    ArmLift,
    FootLift,
    StepUp,
    Slope,
    CarryFront,
    Accel,
}

impl Source {
    pub const ALL: [Source; 8] = [Source::Quiet, Source::ArmRaise, Source::ArmLift, Source::FootLift, Source::StepUp, Source::Slope, Source::CarryFront, Source::Accel];

    pub fn id(self) -> &'static str {
        match self {
            Source::Quiet => "quiet",
            Source::ArmRaise => "arm_raise",
            Source::ArmLift => "arm_lift",
            Source::FootLift => "foot_lift",
            Source::StepUp => "step_up",
            Source::Slope => "slope",
            Source::CarryFront => "carry_front",
            Source::Accel => "accel",
        }
    }

    pub fn label_ko(self) -> &'static str {
        match self {
            Source::Quiet => "정지 자세",
            Source::ArmRaise => "팔 들기 (채취)",
            Source::ArmLift => "팔로 들어올리기",
            Source::FootLift => "발 들기 (보행)",
            Source::StepUp => "턱 오르기",
            Source::Slope => "경사",
            Source::CarryFront => "앞쪽 적재",
            Source::Accel => "가감속",
        }
    }

    pub fn segment(self) -> Segment {
        match self {
            Source::ArmRaise | Source::ArmLift => Segment::Arm,
            Source::FootLift | Source::Slope => Segment::Foot,
            Source::StepUp | Source::Accel => Segment::Leg,
            Source::CarryFront | Source::Quiet => Segment::Belly,
        }
    }

    pub fn duration_s(self) -> f32 {
        match self {
            Source::Quiet => 10.0,
            Source::ArmRaise => 0.45,
            Source::ArmLift => 1.0,
            Source::FootLift => 0.6,
            Source::StepUp => 0.6,
            Source::Slope => 1.0,
            Source::CarryFront => 2.0,
            Source::Accel => 0.5,
        }
    }

    pub fn from_id(id: &str) -> Option<Source> {
        Source::ALL.into_iter().find(|s| s.id() == id)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Posture {
    pub source: Source,
    pub arm_elev_deg: [f32; 2],
    pub elbow_deg: [f32; 2],
    pub hand_load_kg: [f32; 2],
    pub cargo_kg: f32,
    pub slope_deg: f32,
    pub step_m: f32,
    pub single_support: bool,
    pub accel_mps2: f32,
    #[serde(default)]
    pub ground: Surface,
    #[serde(default)]
    pub moving: bool,
    /// The load (`cargo_kg`) is in the two hands, where the arms (`arm_elev_deg`, `elbow_deg`) hold it before the
    /// body, not in its bay: a walker carrying in front of it. Its weight then hangs on the shoulders.
    #[serde(default)]
    pub cargo_in_hands: bool,
    /// How far the load has ridden off its seat (forward, left; m): `act::carry::LoadState::shift`. A slide is not
    /// planned, so the feed-forward does not see it coming.
    #[serde(default)]
    pub cargo_shift_m: [f32; 2],
}

impl Posture {
    pub fn quiet(cargo_kg: f32) -> Posture {
        Posture {
            source: if cargo_kg > 0.0 { Source::CarryFront } else { Source::Quiet },
            arm_elev_deg: [0.0, 0.0],
            elbow_deg: [10.0, 10.0],
            hand_load_kg: [0.0, 0.0],
            cargo_kg,
            slope_deg: 0.0,
            step_m: 0.0,
            single_support: false,
            accel_mps2: 0.0,
            ground: Surface::Firm,
            moving: false,
            cargo_in_hands: false,
            cargo_shift_m: [0.0, 0.0],
        }
    }

    /// This posture with the load held in the hands at the arm angles `(shoulder, elbow)` (deg), ridden `shift_m` off
    /// the middle of the palms.
    pub fn held_in_hands(self, arms_deg: ([f32; 2], [f32; 2]), shift_m: [f32; 2]) -> Posture {
        Posture { arm_elev_deg: arms_deg.0, elbow_deg: arms_deg.1, cargo_in_hands: true, cargo_shift_m: shift_m, ..self }
    }

    pub fn propelling(&self) -> bool {
        self.moving || self.single_support || self.accel_mps2 != 0.0
    }

    pub fn arm_raise(cargo_kg: f32) -> Posture {
        Posture { source: Source::ArmRaise, arm_elev_deg: [75.0, 0.0], elbow_deg: [15.0, 10.0], ..Posture::quiet(cargo_kg) }
    }

    pub fn arm_lift(load_kg: f32, two_arms: bool, cargo_kg: f32) -> Posture {
        let (l0, l1, e1) = if two_arms { (load_kg / 2.0, load_kg / 2.0, 30.0) } else { (load_kg, 0.0, 0.0) };
        Posture { source: Source::ArmLift, arm_elev_deg: [30.0, e1], elbow_deg: [60.0, if two_arms { 60.0 } else { 10.0 }], hand_load_kg: [l0, l1], ..Posture::quiet(cargo_kg) }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeedForward {
    pub enabled: bool,
    pub gain: f32,
}

impl Default for FeedForward {
    fn default() -> Self {
        FeedForward { enabled: true, gain: 1.0 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JointLoad {
    pub joint: String,
    pub segment: Segment,
    pub actuator: String,
    pub torque_nm: f32,
    pub cont_nm: f32,
    pub peak_nm: f32,
    pub effort: f32,
    pub peak_ratio: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Coupling {
    pub source: Source,
    pub target: Segment,
    pub axis: String,
    pub value: f32,
    pub unit: String,
    pub value_noff: f32,
    pub effort: Option<f32>,
    pub effort_noff: Option<f32>,
    pub torque_nm: Option<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BalanceReport {
    pub source: Source,
    pub cargo_x_m: f32,
    pub mass_kg: f32,
    pub hand_load_kg: f32,
    pub cargo_kg: f32,
    pub slope_deg: f32,
    pub com: [f32; 3],
    pub zmp: [f32; 2],
    pub support: [f32; 4],
    pub margin_x: f32,
    pub margin_y: f32,
    pub margin: f32,
    pub margin_noff: f32,
    pub shift_uncomp_m: f32,
    pub restore_frac: f32,
    pub trunk_lean_deg: f32,
    pub lateral_shift_m: f32,
    pub lateral_com_shift_m: f32,
    pub head_pitch_deg: f32,
    pub head_roll_deg: f32,
    pub joints: Vec<JointLoad>,
    pub effort_sum: f32,
    pub feasible: bool,
    pub limiting: Option<String>,
    pub contact: Contact,
    pub ff_predicted_m: f32,
    pub ff_gain: f32,
    pub couplings: Vec<Coupling>,
    #[serde(default)]
    pub traction: Traction,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Traction {
    pub mu: f32,
    pub contact_use: [f32; 2],
    pub push_slide_n: f32,
    pub pull_slide_n: f32,
    pub push_tip_n: f32,
    pub pull_tip_n: f32,
    pub push_max_n: f32,
    pub pull_max_n: f32,
    pub push_slides_first: bool,
    pub pull_slides_first: bool,
    pub hand_z_m: f32,
}

impl BalanceReport {
    pub fn ref_ctx(&self, body_mass_kg: f32) -> RefCtx {
        let m = body_mass_kg.max(1.0);
        match self.source {
            Source::CarryFront => RefCtx { load_ratio: self.cargo_kg / m, load_dist_m: self.cargo_x_m, slope_deg: self.slope_deg },
            _ => RefCtx { load_ratio: self.hand_load_kg / m, load_dist_m: 0.0, slope_deg: self.slope_deg },
        }
    }

    pub fn joint(&self, name: &str) -> Option<&JointLoad> {
        self.joints.iter().find(|j| j.joint == name)
    }

    pub fn worst(&self) -> Option<&JointLoad> {
        self.joints.iter().max_by(|a, b| a.peak_ratio.total_cmp(&b.peak_ratio))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DriveLimits {
    pub max_speed_mps: f32,
    pub max_slope_deg: f32,
    pub max_step_m: f32,
    pub power_scale: f32,
    pub accel_limit_mps2: f32,
    pub capacity_kg: f32,
    pub reasons: Vec<String>,
}

#[derive(Clone, Copy, Debug)]
struct Pt {
    seg: Segment,
    group: Group,
    m: f32,
    p: [f32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Group {
    Upper,
    Pelvis,
    Lower,
    Swing,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Body {
    pub plan: BodyPlan,
    pub base: BaseKind,
    pub fingers: FingerCount,
    pub morph: Morphology,
    pub legged: bool,
    pub head: HeadPreset,
    pub chest: ChestPreset,
    pub belly: BellyPreset,
    pub arm: ArmPreset,
    pub leg: LegPreset,
    pub chassis: ChassisPreset,
    pub lift: Option<LiftPreset>,
    pub gripper: Option<GripperPreset>,
    pub hand_kg: f32,
    pub support_len_m: f32,
    pub support_width_m: f32,
    pub dims: Dims,
    /// Joint limits a design set.
    pub limits: crate::rig::RigLimits,
}

struct Eval {
    pts: Vec<Pt>,
    mass: f32,
    com: [f32; 3],
    support: [f32; 4],
    pitch: f32,
}

fn rot_xz(p: [f32; 3], pivot: [f32; 3], a: f32) -> [f32; 3] {
    let (s, c) = a.sin_cos();
    let (x, z) = (p[0] - pivot[0], p[2] - pivot[2]);
    [pivot[0] + x * c + z * s, p[1], pivot[2] - x * s + z * c]
}

impl Body {
    pub fn new(plan: &BodyPlan, base: BaseKind, fingers: FingerCount) -> Body {
        let morph = plan.morph.fit(base);
        let leg = plan.leg_preset();
        let legged = morph.legged();
        let spec = base.spec();
        let dims = plan.dims(base);
        let rig = match plan.build {
            crate::dims::Build::Rig(id) => crate::rig::get(id),
            _ => None,
        };
        // A design's own masses stand in for the parts': its links are what was drawn.
        let m = rig.as_ref().map_or_else(crate::rig::RigMasses::default, |r| r.masses);
        let mut leg = leg;
        leg.mass_kg = m.leg.unwrap_or(leg.mass_kg);
        leg.foot_mass_kg = m.foot.unwrap_or(leg.foot_mass_kg);
        leg.foot_len_m = m.foot_len_m.unwrap_or(leg.foot_len_m);
        leg.foot_width_m = m.foot_width_m.unwrap_or(leg.foot_width_m);
        leg.com_down_m = m.leg_com_down_m.unwrap_or(leg.com_down_m);
        let (support_len_m, support_width_m) = if legged {
            (leg.foot_len_m, dims.stance_m + leg.foot_width_m)
        } else {
            (dims.wheelbase_m.unwrap_or(spec.footprint_m), dims.chassis_w_m)
        };
        let gripper = (morph.manip == Manip::Gripper1).then(|| plan.gripper_preset());
        let hand_kg = match morph.manip {
            Manip::None => 0.0,
            Manip::Gripper1 => gripper.as_ref().map_or(0.0, |g| g.mass_kg),
            Manip::Arms2 | Manip::Arm1 => m.hand.unwrap_or(hand_mass_kg(fingers) * plan.k().powi(3)),
        };
        let mut head = mounted_head(plan.head_preset(), morph.head);
        head.mass_kg = m.head.unwrap_or(head.mass_kg);
        head.com_up_m = m.head_com_up_m.unwrap_or(head.com_up_m);
        let mut chest = mounted_chest(plan.chest_preset(), morph.torso);
        if let Some(total) = m.chest {
            // The trunk link holds the chest and its battery, in their proportion.
            let share = chest.mass_kg / (chest.mass_kg + chest.battery_kg).max(1e-3);
            chest.mass_kg = total * share;
            chest.battery_kg = total * (1.0 - share);
        }
        chest.com_up_m = m.chest_com_up_m.unwrap_or(chest.com_up_m);
        let mut belly = mounted_belly(plan.belly_preset(), morph.torso);
        belly.mass_kg = m.belly.unwrap_or(belly.mass_kg);
        belly.com_down_m = m.belly_com_down_m.unwrap_or(belly.com_down_m);
        let mut arm = plan.arm_preset();
        arm.mass_kg = m.arm.unwrap_or(arm.mass_kg);
        arm.com_m = m.arm_com_m.unwrap_or(arm.com_m);
        let mut chassis = plan.chassis_preset();
        chassis.mass_kg = m.chassis.unwrap_or(chassis.mass_kg);
        chassis.com_m = m.chassis_com_m.unwrap_or(chassis.com_m);
        Body {
            plan: BodyPlan { morph: Morphology { lower: plan.morph.lower, ..morph }, ..*plan },
            base,
            fingers,
            morph,
            legged,
            head,
            chest,
            belly,
            arm,
            leg,
            chassis,
            lift: BodyPlan { morph, ..*plan }.lift_preset(base),
            gripper,
            hand_kg,
            support_len_m,
            support_width_m,
            dims,
            limits: rig.as_ref().map_or_else(crate::rig::RigLimits::default, |r| r.limits),
        }
    }

    pub fn mast_kg(&self) -> f32 {
        column_kg(&self.dims, self.morph.torso, self.morph.head) + self.lift.as_ref().map_or(0.0, |l| l.mass_kg)
    }

    pub fn mass_kg(&self) -> f32 {
        let lower = if self.legged() { 2.0 * (self.leg.mass_kg + self.leg.foot_mass_kg) } else { self.chassis.mass_kg + self.mast_kg() };
        let cargo = self.cargo_geom().map_or(0.0, |c| c.struct_kg);
        self.head.mass_kg + self.chest.mass_kg + self.chest.battery_kg + self.belly.mass_kg + self.arm_count() as f32 * (self.arm.mass_kg + self.hand_kg) + lower + cargo
    }

    pub fn battery_j(&self) -> f64 {
        // A truck runs on its traction battery [추정: 48 V 375 Ah], the rider on nothing of its own.
        self.dims.payload_kg.map_or(self.chest.battery_wh, |_| 18_000.0) as f64 * 3600.0
    }

    pub fn com_height_m(&self) -> f32 {
        self.build(&self.quiet(0.0)).com[2]
    }

    pub fn arm_split(&self) -> (f32, f32, f32) {
        let m_fore = 0.25 * self.arm.mass_kg;
        let m_up = self.arm.mass_kg - m_fore;
        let r_up = ((self.arm.com_m * self.arm.mass_kg - m_fore * (self.dims.upper_arm_m + 0.45 * self.dims.forearm_m)) / m_up).max(0.02);
        (m_up, r_up, m_fore)
    }

    pub fn support_rect(&self, post: &Posture, pitch: f32) -> [f32; 4] {
        if self.legged() {
            let c = post.slope_deg.to_radians().cos();
            let (x0, x1) = (-0.25 * self.leg.foot_len_m * c, 0.75 * self.leg.foot_len_m * c);
            if post.single_support {
                let ys = -self.dims.stance_m / 2.0;
                [x0, x1, ys - self.leg.foot_width_m / 2.0, ys + self.leg.foot_width_m / 2.0]
            } else {
                [x0, x1, -self.support_width_m / 2.0, self.support_width_m / 2.0]
            }
        } else {
            let h = 0.5 * self.support_len_m * pitch.cos();
            [-h, h, -self.support_width_m / 2.0, self.support_width_m / 2.0]
        }
    }

    pub fn base_pitch(&self, post: &Posture) -> f32 {
        if self.legged() {
            0.0
        } else {
            post.slope_deg.to_radians() + (post.step_m / self.support_len_m.max(0.1)).atan()
        }
    }

    fn arm_points(&self, side: usize, elev_deg: f32, elbow_deg: f32, load: f32) -> (Vec<Pt>, [f32; 3], [f32; 3]) {
        let (m_up, r_up, m_fore) = self.arm_split();
        let a = elev_deg.to_radians();
        let b = (elev_deg + elbow_deg).to_radians();
        let du = [a.sin(), 0.0, -a.cos()];
        let df = [b.sin(), 0.0, -b.cos()];
        let at = |o: [f32; 3], d: [f32; 3], r: f32| [o[0] + d[0] * r, o[1] + d[1] * r, o[2] + d[2] * r];
        let (sh, elbow, hand) = self.arm_ends(side, elev_deg, elbow_deg);
        let mut v = vec![
            Pt { seg: Segment::Arm, group: Group::Upper, m: m_up, p: at(sh, du, r_up) },
            Pt { seg: Segment::Arm, group: Group::Upper, m: m_fore, p: at(elbow, df, 0.45 * self.dims.forearm_m) },
            Pt { seg: Segment::Hand, group: Group::Upper, m: self.hand_kg, p: hand },
        ];
        if load > 0.0 {
            v.push(Pt { seg: Segment::Hand, group: Group::Upper, m: load, p: hand });
        }
        (v, sh, elbow)
    }

    /// Whether the load rides in the hands (`Posture::cargo_in_hands`) on a body that has them.
    pub(crate) fn holds_cargo(&self, post: &Posture) -> bool {
        post.cargo_in_hands && post.cargo_kg > 0.0 && self.arm_count() > 0
    }

    /// The share of a load held in the hands that hangs on arm `side` (kg), and where it hangs (body frame). Two hands
    /// split it by the lever rule about where its centre has slid between them; one hand takes all of it.
    pub(crate) fn held_on(&self, post: &Posture, side: usize, shift: [f32; 2]) -> Option<(f32, [f32; 3])> {
        if !self.holds_cargo(post) || side >= self.arm_count() {
            return None;
        }
        let (_, _, hand) = self.arm_ends(side, post.arm_elev_deg[side], post.elbow_deg[side]);
        let kg = if self.arm_count() >= 2 {
            let (r, l) = (self.arm_ends(0, post.arm_elev_deg[0], post.elbow_deg[0]).2, self.arm_ends(1, post.arm_elev_deg[1], post.elbow_deg[1]).2);
            let span = (l[1] - r[1]).max(0.05);
            let y = 0.5 * (r[1] + l[1]) + shift[1];
            let share_l = ((y - r[1]) / span).clamp(0.0, 1.0);
            post.cargo_kg * if side == 0 { 1.0 - share_l } else { share_l }
        } else {
            post.cargo_kg
        };
        Some((kg, [hand[0] + shift[0], hand[1], hand[2]]))
    }

    /// Shoulder, elbow and palm of arm `side` at these angles (body frame).
    fn arm_ends(&self, side: usize, elev_deg: f32, elbow_deg: f32) -> ([f32; 3], [f32; 3], [f32; 3]) {
        let sh = self.shoulder_mount(side);
        let a = elev_deg.to_radians();
        let b = (elev_deg + elbow_deg).to_radians();
        let at = |o: [f32; 3], d: [f32; 3], r: f32| [o[0] + d[0] * r, o[1] + d[1] * r, o[2] + d[2] * r];
        let elbow = at(sh, [a.sin(), 0.0, -a.cos()], self.dims.upper_arm_m);
        (sh, elbow, at(elbow, [b.sin(), 0.0, -b.cos()], self.dims.forearm_m + 0.06))
    }

    /// How far ahead of the base the carried load's centre is now (m): the palms and its slide when it is held, the
    /// bay otherwise.
    pub fn cargo_x_now(&self, post: &Posture) -> f32 {
        if !self.holds_cargo(post) {
            return self.dims.cargo_x_m + post.cargo_shift_m[0];
        }
        let held: Vec<(f32, [f32; 3])> = (0..self.arm_count()).filter_map(|s| self.held_on(post, s, post.cargo_shift_m)).collect();
        let kg: f32 = held.iter().map(|h| h.0).sum();
        held.iter().map(|h| h.0 * h.1[0]).sum::<f32>() / kg.max(1e-6)
    }

    fn points(&self, post: &Posture) -> Vec<Pt> {
        let mut v = Vec::new();
        let up = |seg, m, p| Pt { seg, group: Group::Upper, m, p };
        if self.head_on_trunk() {
            if let Some(h) = self.head_com() {
                v.push(up(Segment::Head, self.head.mass_kg, h));
            }
        }
        if self.has_torso() {
            v.push(up(Segment::Chest, self.chest.mass_kg, [0.0, 0.0, self.dims.waist_m + self.chest.com_up_m]));
            let belly_z = self.dims.waist_m - self.belly.com_down_m;
            if self.plan.battery_low {
                v.push(Pt { seg: Segment::Belly, group: Group::Pelvis, m: self.chest.battery_kg, p: [0.0, 0.0, belly_z] });
            } else {
                v.push(up(Segment::Chest, self.chest.battery_kg, [0.0, 0.0, self.dims.waist_m + self.chest.com_up_m]));
            }
            v.push(Pt { seg: Segment::Belly, group: Group::Pelvis, m: self.belly.mass_kg, p: [0.0, 0.0, belly_z] });
        }
        if post.cargo_kg > 0.0 && !self.holds_cargo(post) {
            let (p, on_pelvis) = self.cargo_point();
            let p = [p[0] + post.cargo_shift_m[0], p[1] + post.cargo_shift_m[1], p[2]];
            v.push(Pt { seg: Segment::Belly, group: if on_pelvis { Group::Pelvis } else { Group::Lower }, m: post.cargo_kg, p });
        }
        for side in 0..self.arm_count() {
            let (pts, _, _) = self.arm_points(side, post.arm_elev_deg[side], post.elbow_deg[side], post.hand_load_kg[side]);
            v.extend(pts);
            if let Some((kg, p)) = self.held_on(post, side, post.cargo_shift_m) {
                v.push(Pt { seg: Segment::Hand, group: Group::Upper, m: kg, p });
            }
        }
        for mm in self.mount_masses() {
            v.push(Pt { seg: mm.segment, group: Group::Lower, m: mm.kg, p: mm.at });
        }
        if self.legged() {
            for (i, y) in [-self.dims.stance_m / 2.0, self.dims.stance_m / 2.0].into_iter().enumerate() {
                let swing = post.single_support && i == 1;
                let lift = if swing { post.step_m.max(0.05) } else { 0.0 };
                let g = if swing { Group::Swing } else { Group::Lower };
                v.push(Pt { seg: Segment::Leg, group: g, m: self.leg.mass_kg, p: [if swing { 0.10 } else { 0.0 }, y, self.dims.hip_m - self.leg.com_down_m + lift * 0.5] });
                v.push(Pt { seg: Segment::Foot, group: g, m: self.leg.foot_mass_kg, p: [0.25 * self.leg.foot_len_m + if swing { 0.15 } else { 0.0 }, y, 0.04 + lift] });
            }
        } else {
            if let Some((m, p)) = self.column() {
                v.push(Pt { seg: Segment::Leg, group: Group::Lower, m, p });
            }
            v.push(Pt { seg: Segment::Foot, group: Group::Lower, m: self.chassis.mass_kg, p: [0.0, 0.0, self.chassis.com_m] });
        }
        v
    }

    fn evaluate(&self, post: &Posture, lean: f32, shift_y: f32) -> Eval {
        let mut pts = self.points(post);
        let pivot = [0.0, 0.0, self.dims.waist_m];
        for p in pts.iter_mut() {
            if p.group == Group::Upper {
                p.p = rot_xz(p.p, pivot, lean);
            }
            if matches!(p.group, Group::Upper | Group::Pelvis | Group::Swing) {
                p.p[1] += shift_y;
            }
        }
        let pitch = self.base_pitch(post);
        if pitch != 0.0 {
            for p in pts.iter_mut() {
                p.p = rot_xz(p.p, [0.0, 0.0, 0.0], -pitch);
            }
        }
        let mass: f32 = pts.iter().map(|p| p.m).sum();
        let mut com = [0.0f32; 3];
        for p in &pts {
            for k in 0..3 {
                com[k] += p.m * p.p[k] / mass;
            }
        }
        let support = self.support_rect(post, pitch);
        Eval { pts, mass, com, support, pitch }
    }

    fn home_x(&self) -> f32 {
        if self.legged() {
            0.15 * self.leg.foot_len_m
        } else {
            0.0
        }
    }

    fn ankle_band(&self) -> f32 {
        if self.legged() {
            0.10 * self.leg.foot_len_m
        } else {
            0.02
        }
    }

    pub fn nominal_shift(&self, post: &Posture) -> f32 {
        let m = self.mass_kg() + post.cargo_kg + post.hand_load_kg.iter().sum::<f32>();
        let arm_len = self.dims.upper_arm_m + self.dims.forearm_m + 0.06;
        let mut s = 0.0;
        for i in 0..self.arm_count() {
            let a = post.arm_elev_deg[i].to_radians();
            let reach = arm_len * a.sin();
            // A load held in the hands as planned: where the palms are, not where it may have slid.
            let held = self.held_on(post, i, [0.0, 0.0]).map_or(0.0, |h| h.0);
            s += 0.05 * m * 0.53 * reach + (post.hand_load_kg[i] + held) * reach;
        }
        if !self.holds_cargo(post) {
            s += post.cargo_kg * self.cargo_point().0[0];
        }
        s / m
    }

    fn build(&self, post: &Posture) -> Eval {
        self.evaluate(post, 0.0, 0.0)
    }

    pub fn solve(&self, post: &Posture, ff: &FeedForward, grip_capacity_n: f32) -> BalanceReport {
        let (rest_elev, rest_elbow) = self.rest_arms_deg();
        // What the body would be without this source's push. Arms holding a load are how a walker carries, unless the
        // carry is the source; a slide of the load never is.
        let base_post = if self.holds_cargo(post) && post.source != Source::CarryFront {
            Posture { hand_load_kg: [0.0, 0.0], cargo_shift_m: [0.0, 0.0], ..*post }
        } else {
            Posture { arm_elev_deg: rest_elev, elbow_deg: rest_elbow, hand_load_kg: [0.0, 0.0], cargo_kg: if post.source == Source::CarryFront { 0.0 } else { post.cargo_kg }, cargo_shift_m: [0.0, 0.0], ..*post }
        };
        let uncomp = self.build(post);
        let reference = self.build(&base_post);
        let home = self.home_x();
        let target = if post.accel_mps2 != 0.0 {
            let s = uncomp.support;
            0.5 * (s[0] + s[1]) + 0.5 * uncomp.com[2] * post.accel_mps2 / G
        } else {
            home
        };
        let delta = uncomp.com[0] - target;
        let band = self.ankle_band();
        let excess = delta - delta.clamp(-band, band);
        let upper_m: f32 = uncomp.pts.iter().filter(|p| p.group == Group::Upper).map(|p| p.m).sum();
        let upper_h: f32 = uncomp.pts.iter().filter(|p| p.group == Group::Upper).map(|p| p.m * (p.p[2] - self.dims.waist_m)).sum::<f32>() / upper_m.max(1e-3);
        let max_lean = self.belly.max_lean_deg.to_radians();
        let lean = if excess.abs() > 1e-5 && max_lean > 0.0 {
            (-(excess * uncomp.mass) / (upper_m * upper_h.max(0.05))).clamp(-1.0, 1.0).asin().clamp(-max_lean, max_lean)
        } else {
            0.0
        };
        let shift_y = if self.legged() && post.single_support {
            let moved: f32 = uncomp.pts.iter().filter(|p| p.group != Group::Lower).map(|p| p.m).sum();
            (-self.dims.stance_m / 2.0 - uncomp.com[1]) * uncomp.mass / moved.max(1e-3)
        } else {
            0.0
        };
        let fin = self.evaluate(post, lean, shift_y);

        let shift_uncomp = uncomp.com[0] - reference.com[0];
        let restore = if (uncomp.com[0] - home).abs() > 1e-4 { ((uncomp.com[0] - fin.com[0]) / (uncomp.com[0] - home)).clamp(-1.0, 2.0) } else { 0.0 };

        let predicted = self.nominal_shift(post) - self.nominal_shift(&base_post);
        let anticipated = if ff.enabled && shift_uncomp.abs() > 1e-5 { (ff.gain * predicted / shift_uncomp).clamp(0.0, 2.0) } else { 0.0 };
        let lag = (REACTION_DELAY_S / post.source.duration_s()).min(1.0);
        let eps_ff = lag * (1.0 - anticipated).abs();
        let eps_noff = lag;

        let a = post.accel_mps2;
        let zmp = [fin.com[0] - fin.com[2] * a / G, fin.com[1]];
        let half_x = 0.5 * (fin.support[1] - fin.support[0]);
        let half_y = 0.5 * (fin.support[3] - fin.support[2]);
        let cx = 0.5 * (fin.support[0] + fin.support[1]);
        let cy = 0.5 * (fin.support[2] + fin.support[3]);
        let mx = 1.0 - (zmp[0] - cx).abs() / half_x.max(1e-3);
        let my = 1.0 - (zmp[1] - cy).abs() / half_y.max(1e-3);
        let dip = |eps: f32| eps * shift_uncomp.abs() / half_x.max(1e-3);
        let margin = mx.min(my) - dip(eps_ff);
        let margin_noff = mx.min(my) - dip(eps_noff);

        let keep = 1.0 - self.head.gaze_stabilization;
        let head_pitch = (fin.pitch + lean).to_degrees() * keep;
        let head_roll = if self.legged() && post.single_support { (shift_y / self.dims.hip_m).atan().to_degrees() * keep } else { 0.0 };

        let joints = self.joint_loads(post, &fin, lean, shift_y, grip_capacity_n);
        let comp_joint = |j: &JointLoad, eps: f32| {
            if matches!(j.joint.as_str(), "waist" | "ankle" | "hip_roll" | "drive") {
                j.peak_ratio * (1.0 + eps)
            } else {
                j.peak_ratio
            }
        };
        let effort_sum: f32 = joints.iter().map(|j| comp_joint(j, eps_ff).powi(2)).sum();
        let worst = joints.iter().max_by(|a, b| comp_joint(a, eps_ff).total_cmp(&comp_joint(b, eps_ff)));
        let (mut feasible, mut limiting) = (true, None);
        if let Some(w) = worst {
            if comp_joint(w, eps_ff) > 1.0 {
                feasible = false;
                limiting = Some(format!("{} {:.1}/{:.1} N·m ({})", w.joint, w.torque_nm * (1.0 + eps_ff), w.peak_nm, w.actuator));
            }
        }
        let contact = surface::contact(self.base, post.ground.props(), post.slope_deg, post.accel_mps2, post.propelling());
        if contact.utilisation > 1.0 {
            feasible = false;
            limiting = Some(format!("미끄러짐: 필요 마찰 {:.2} > {} {:.2}", contact.required_mu, post.ground.label_ko(), contact.available_mu));
        }
        if margin < 0.0 {
            feasible = false;
            limiting = Some(format!("균형 여유 {:.2} < 0", margin));
        }
        if let Some(s) = (self.arm_count()..2).find(|s| post.hand_load_kg[*s] > 0.0) {
            feasible = false;
            limiting = Some(format!("{} 없음: 손 하중 {:.1} kg를 들 팔이 없음", if s == 0 { "오른팔" } else { "왼팔" }, post.hand_load_kg[s]));
        }

        let mut rep = BalanceReport {
            cargo_x_m: self.cargo_x_now(post),
            source: post.source,
            mass_kg: fin.mass,
            hand_load_kg: post.hand_load_kg.iter().sum(),
            cargo_kg: post.cargo_kg,
            slope_deg: post.slope_deg,
            com: fin.com,
            zmp,
            support: fin.support,
            margin_x: mx,
            margin_y: my,
            margin,
            margin_noff,
            shift_uncomp_m: shift_uncomp,
            restore_frac: restore,
            trunk_lean_deg: lean.to_degrees(),
            lateral_shift_m: shift_y,
            lateral_com_shift_m: fin.com[1] - uncomp.com[1],
            head_pitch_deg: head_pitch,
            head_roll_deg: head_roll,
            joints,
            effort_sum,
            feasible,
            limiting,
            contact,
            ff_predicted_m: predicted,
            ff_gain: ff.gain,
            couplings: Vec::new(),
            traction: Traction::default(),
        };
        rep.traction = self.traction(post, fin.mass, zmp, fin.support);
        rep.couplings = self.couplings(post, &rep, eps_ff, eps_noff, &reference);
        rep
    }

    pub fn traction(&self, post: &Posture, mass_kg: f32, zmp: [f32; 2], support: [f32; 4]) -> Traction {
        let mu = surface::available_mu(self.base, post.ground.props());
        let th = post.slope_deg.to_radians();
        let w = mass_kg * G;
        let n = w * th.cos();
        let shear = w * th.sin() + mass_kg * post.accel_mps2;
        let use_ = if n > 1e-6 && mu > 1e-6 { shear.abs() / (mu * n) } else { 0.0 };
        let contact_use = if self.legged && post.single_support { [use_, 0.0] } else { [use_, use_] };
        let e = post.arm_elev_deg[0].to_radians();
        let f = (post.arm_elev_deg[0] + post.elbow_deg[0]).to_radians();
        let hand_z = (self.dims.shoulder_m - self.dims.upper_arm_m * e.cos() - (self.dims.forearm_m + 0.06) * f.cos()).max(0.05);
        let room = crate::contact::friction_room_n(mu, n, shear);
        let (push_slide, pull_slide) = (room[0], room[1]);
        let push_tip = ((zmp[0] - support[0]).max(0.0) * w / hand_z).max(0.0);
        let pull_tip = ((support[1] - zmp[0]).max(0.0) * w / hand_z).max(0.0);
        Traction {
            mu,
            contact_use,
            push_slide_n: push_slide,
            pull_slide_n: pull_slide,
            push_tip_n: push_tip,
            pull_tip_n: pull_tip,
            push_max_n: push_slide.min(push_tip),
            pull_max_n: pull_slide.min(pull_tip),
            push_slides_first: push_slide < push_tip,
            pull_slides_first: pull_slide < pull_tip,
            hand_z_m: hand_z,
        }
    }

    fn joint_loads(&self, post: &Posture, e: &Eval, lean: f32, shift_y: f32, grip_capacity_n: f32) -> Vec<JointLoad> {
        let a = post.accel_mps2;
        let lever = |p: &Pt, j: [f32; 3]| (p.p[0] - j[0]) - (a / G) * (p.p[2] - j[2]);
        let moment = |pts: &mut dyn Iterator<Item = &Pt>, j: [f32; 3]| -> f32 { G * pts.map(|p| p.m * lever(p, j)).sum::<f32>() };
        let jl = |joint: &str, seg: Segment, act: &Actuator, tau: f32| {
            let t = tau.abs();
            JointLoad { joint: joint.into(), segment: seg, actuator: act.name.into(), torque_nm: t, cont_nm: act.cont_nm, peak_nm: act.peak_nm, effort: t / act.cont_nm.max(1e-3), peak_ratio: t / act.peak_nm.max(1e-3) }
        };
        let tilt = |p: [f32; 3]| {
            let mut q = rot_xz(p, [0.0, 0.0, self.dims.waist_m], lean);
            q[1] += shift_y;
            if e.pitch != 0.0 {
                q = rot_xz(q, [0.0, 0.0, 0.0], -e.pitch);
            }
            q
        };
        let mut out = Vec::new();
        for side in 0..self.arm_count() {
            let (mut pts, sh, el) = self.arm_points(side, post.arm_elev_deg[side], post.elbow_deg[side], post.hand_load_kg[side]);
            if let Some((kg, p)) = self.held_on(post, side, post.cargo_shift_m) {
                pts.push(Pt { seg: Segment::Hand, group: Group::Upper, m: kg, p });
            }
            let pts: Vec<Pt> = pts.into_iter().map(|p| Pt { p: tilt(p.p), ..p }).collect();
            let (sh, el) = (tilt(sh), tilt(el));
            let tag = if side == 0 { "r" } else { "l" };
            out.push(jl(&format!("shoulder_{tag}"), Segment::Arm, &self.arm.shoulder, moment(&mut pts.iter(), sh)));
            out.push(jl(&format!("elbow_{tag}"), Segment::Arm, &self.arm.elbow, moment(&mut pts.iter().skip(1), el)));
            let load = post.hand_load_kg[side];
            if load > 0.0 {
                let need = load * (G + a.abs()) * GRIP_SAFETY;
                let cap = grip_capacity_n.max(1e-3);
                out.push(JointLoad { joint: format!("grip_{tag}"), segment: Segment::Hand, actuator: "손가락 마찰".into(), torque_nm: need, cont_nm: cap, peak_nm: cap, effort: need / cap, peak_ratio: need / cap });
            }
        }
        let waist = tilt([0.0, 0.0, self.dims.waist_m]);
        if let Some(pitch) = &self.belly.pitch {
            out.push(jl("waist", Segment::Belly, pitch, moment(&mut e.pts.iter().filter(|p| p.group == Group::Upper), waist)));
        }
        if let Some(pivot) = self.neck_pivot() {
            let neck = tilt(pivot);
            out.push(jl("neck", Segment::Head, &self.head.neck, moment(&mut e.pts.iter().filter(|p| p.seg == Segment::Head), neck)));
        }
        if self.legged() {
            let n_support = if post.single_support { 1.0 } else { 2.0 };
            let above_feet: f32 = e.mass - e.pts.iter().filter(|p| p.seg == Segment::Foot).map(|p| p.m).sum::<f32>();
            let x_ank = 0.0;
            let tau_ankle = G * e.pts.iter().filter(|p| p.seg != Segment::Foot && p.group != Group::Swing).map(|p| p.m * lever(p, [x_ank, 0.0, 0.08])).sum::<f32>() / n_support;
            out.push(jl("ankle", Segment::Foot, &self.leg.ankle, tau_ankle));
            let d = if post.step_m > 0.0 { step_knee_lever(&self.dims, post.step_m) } else if post.single_support { self.dims.knee_lever_walk_m } else { self.dims.knee_lever_stand_m };
            let leg_m = 2.0 * self.leg.mass_kg;
            out.push(jl("knee", Segment::Leg, &self.leg.knee, (above_feet - leg_m * 0.5) * G * d / n_support));
            if post.single_support {
                let upper: f32 = e.mass - e.pts.iter().filter(|p| matches!(p.seg, Segment::Leg | Segment::Foot)).map(|p| p.m).sum::<f32>();
                out.push(jl("hip_roll", Segment::Leg, &self.leg.knee, upper * G * self.dims.hip_half_m));
            }
        } else {
            let slope = post.slope_deg.to_radians();
            let crr = if post.propelling() { surface::rolling_resistance(self.base, post.ground.props()) } else { 0.0 };
            let force = e.mass * (G * (slope.sin() + crr * slope.cos()) + a.abs());
            let f = self.chassis.drive_force_n.max(1e-3);
            out.push(JointLoad { joint: "drive".into(), segment: Segment::Foot, actuator: self.chassis.name.into(), torque_nm: force, cont_nm: f, peak_nm: 2.0 * f, effort: force / f, peak_ratio: force / (2.0 * f) });
        }
        out
    }

    fn couplings(&self, post: &Posture, r: &BalanceReport, eps_ff: f32, eps_noff: f32, reference: &Eval) -> Vec<Coupling> {
        let src = post.source;
        let foot_len = if self.legged() { self.leg.foot_len_m } else { self.support_len_m };
        let j = |name: &str| r.joint(name);
        let cell = |target: Segment, axis: &str, unit: &str, value: f32, value_noff: f32, joint: Option<&JointLoad>, comp: bool| {
            let k = |eps: f32| joint.map(|j| if comp { j.peak_ratio * (1.0 + eps) } else { j.peak_ratio });
            Coupling { source: src, target, axis: axis.into(), value, unit: unit.into(), value_noff, effort: k(eps_ff), effort_noff: k(eps_noff), torque_nm: joint.map(|j| j.torque_nm) }
        };
        let quiet_ankle = {
            let mut q = Posture { source: Source::Quiet, ..self.quiet(post.cargo_kg) };
            if self.holds_cargo(post) {
                q = q.held_in_hands((post.arm_elev_deg, post.elbow_deg), [0.0, 0.0]);
            }
            let e = self.evaluate(&q, 0.0, 0.0);
            self.joint_loads(&q, &e, 0.0, 0.0, 1.0).into_iter().find(|j| j.joint == "ankle" || j.joint == "drive").map(|j| j.torque_nm).unwrap_or(0.0)
        };
        let ankle = j("ankle").or(j("drive"));
        let ankle_rise = ankle.map(|a| if quiet_ankle > 1e-3 { a.torque_nm / quiet_ankle - 1.0 } else { 0.0 }).unwrap_or(0.0);
        let mut v = Vec::new();
        let head_tilt = r.head_pitch_deg.abs().max(r.head_roll_deg.abs());
        v.push(cell(Segment::Head, "tilt_deg", "°", head_tilt, head_tilt, j("neck"), false));
        v.push(cell(Segment::Chest, "lean_deg", "°", r.trunk_lean_deg, r.trunk_lean_deg, None, false));
        v.push(cell(Segment::Belly, "restore_frac", "", r.restore_frac, r.restore_frac * (1.0 - eps_noff), j("waist"), true));
        match src {
            Source::ArmRaise | Source::ArmLift => {
                v.push(cell(Segment::Arm, "peak_ratio", "", j("shoulder_r").map_or(0.0, |x| x.peak_ratio), j("shoulder_r").map_or(0.0, |x| x.peak_ratio), j("shoulder_r"), false));
            }
            _ => {}
        }
        if let Some(g) = j("grip_r") {
            v.push(cell(Segment::Hand, "grip_effort", "", g.effort, g.effort, Some(g), false));
        }
        if self.legged() {
            let leg_joint = if post.single_support && post.step_m == 0.0 { j("hip_roll") } else { j("knee") };
            if let Some(lj) = leg_joint {
                v.push(cell(Segment::Leg, "peak_ratio", "", lj.peak_ratio, lj.peak_ratio, Some(lj), lj.joint == "hip_roll"));
            }
            if post.single_support {
                let frac = r.lateral_com_shift_m.abs() / (self.dims.stance_m / 2.0);
                v.push(cell(Segment::Belly, "shift_frac", "", frac, frac, None, false));
            }
        }
        let shift = (r.shift_uncomp_m) / foot_len;
        v.push(cell(Segment::Foot, "com_shift_frac", "", shift, shift, ankle, true));
        v.push(cell(Segment::Foot, "ankle_rise", "", ankle_rise, ankle_rise * (1.0 + eps_noff), ankle, true));
        v.push(cell(Segment::Foot, "margin", "", r.margin, r.margin_noff, None, false));
        if src == Source::Slope {
            let x_ref = reference.com[0];
            let cop = (r.zmp[0] - x_ref) / foot_len;
            v.push(cell(Segment::Foot, "cop_shift_frac", "", cop, cop, ankle, true));
        }
        v
    }

    pub fn drive_limits(&self, spec: &BaseSpec, cargo_kg: f32) -> DriveLimits {
        let m = self.mass_kg() + cargo_kg;
        let ref_m = Body::new(&BodyPlan { build: self.plan.build, ..BodyPlan::default() }, self.base, FingerCount::Five).mass_kg();
        let mut reasons = Vec::new();
        let mut speed = spec.max_speed_mps;
        let mut slope = spec.max_slope_deg;
        let mut step = spec.max_step_m;
        let quiet = self.solve(&self.quiet(cargo_kg), &FeedForward::default(), 1.0);
        let half = 0.5 * (quiet.support[1] - quiet.support[0]);
        let accel = (G * (quiet.margin_x - MARGIN_KEEP).max(0.02) * half / quiet.com[2].max(0.1)).min(3.0);
        let capacity;
        if self.legged() {
            let v_fr = (self.leg.froude * G * self.dims.hip_m).sqrt();
            if v_fr < speed {
                speed = v_fr;
                reasons.push(format!("보행 속도 한계 Froude {:.2} → {:.2} m/s", self.leg.froude, v_fr));
            }
            let need = m * G * self.dims.knee_lever_walk_m;
            let s = self.leg.knee.cont_nm / need;
            if s < 1.0 {
                let f = s * s;
                speed *= f;
                reasons.push(format!("무릎 연속 토크 {:.1} < 필요 {:.1} N·m → 속도 ×{:.2}", self.leg.knee.cont_nm, need, f));
            }
            if self.leg.knee.peak_nm < KNEE_SAFETY * need {
                speed = speed.min(0.02);
                reasons.push(format!("무릎 피크 {:.1} < {:.1}×필요 {:.1} N·m: 거의 걸을 수 없음", self.leg.knee.peak_nm, KNEE_SAFETY, need));
            }
            let d_max = self.leg.knee.peak_nm / (KNEE_SAFETY * m * G);
            let h_max = ((d_max / self.dims.hip_m - 0.10) / 0.40 * self.dims.hip_m).max(0.0);
            if h_max < step {
                step = h_max;
                reasons.push(format!("무릎 피크로 오를 수 있는 턱 {:.0} cm", h_max * 100.0));
            }
            let firm = Surface::Firm.props();
            let slip = surface::friction_slope_limit_deg(self.base, firm);
            if slip < slope {
                slope = slip;
                reasons.push(format!("단단한 바닥에서 발 마찰 {:.2} − 보행 전단 {:.3}의 미끄럼 한계 {slip:.0}°", firm.mu_foot, surface::GAIT_RCOF));
            }
            capacity = (self.leg.knee.cont_nm / (G * self.dims.knee_lever_walk_m) - self.mass_kg()).max(0.0);
        } else {
            if self.chassis.max_speed_mps < speed {
                speed = self.chassis.max_speed_mps;
                reasons.push(format!("섀시 최고 속도 {:.2} m/s ({})", speed, self.chassis.name));
            }
            let trac = surface::climb_limit_deg(self.base, Surface::Firm.props(), Some(self.chassis.drive_force_n / m));
            if trac < slope {
                slope = trac.max(0.0);
                reasons.push(format!("단단한 바닥에서 구동력 {:.0} N·마찰로 오를 수 있는 경사 {:.0}°", self.chassis.drive_force_n, slope));
            }
            let mut tip = 0.0;
            let mut d = 0.5;
            while d <= 45.0 {
                let p = Posture { source: Source::Slope, slope_deg: d, ..self.quiet(cargo_kg) };
                if self.solve(&p, &FeedForward::default(), 1.0).margin_x < MARGIN_KEEP {
                    break;
                }
                tip = d;
                d += 0.5;
            }
            if tip < slope {
                slope = tip;
                reasons.push(format!("키 {:.2} m 무게중심과 바퀴 간 거리 {:.2} m로 넘어지지 않는 경사 {tip:.1}°", quiet.com[2], self.support_len_m));
            }
            let v_bal = (2.0 * accel * STOP_DISTANCE_M).sqrt();
            if v_bal < speed {
                speed = v_bal;
                reasons.push(format!("{STOP_DISTANCE_M} m 안에 서려면(감속 {accel:.2} m/s²) {v_bal:.2} m/s"));
            }
            let structural = match self.chassis.tier {
                Tier::Budget | Tier::BudgetPlus => 8.0,
                Tier::Mid => 30.0,
                Tier::Pro => 100.0,
            };
            capacity = match self.dims.payload_kg {
                Some(rated) => rated,
                None => (structural - (self.mass_kg() - self.chassis.mass_kg)).max(0.0),
            };
        }
        let capacity_kg = match self.cargo_geom() {
            Some(c) => match c.payload_kg {
                Some(p) => {
                    if p < capacity {
                        reasons.push(format!("짐칸 정격 하중 {p:.0} kg"));
                    }
                    capacity.min(p)
                }
                None => capacity.min(self.dims.payload_kg.unwrap_or(self.base.carry_capacity_kg())),
            },
            None => {
                reasons.push("적재 공간 없음 (집게로 든 것만 운반)".to_string());
                0.0
            }
        };
        DriveLimits { max_speed_mps: speed.max(0.01), max_slope_deg: slope.max(0.0), max_step_m: step.max(0.0), power_scale: m / ref_m, accel_limit_mps2: accel, capacity_kg, reasons }
    }

    pub fn planned_accel(&self, spec: &BaseSpec, slope_deg: f32, cargo_kg: f32, ground: Surface, dir: f32, min_ramp_s: f32) -> f32 {
        self.planned_accel_within(&self.drive_limits(spec, cargo_kg), slope_deg, cargo_kg, ground, dir, min_ramp_s)
    }

    pub fn planned_accel_within(&self, l: &DriveLimits, slope_deg: f32, cargo_kg: f32, ground: Surface, dir: f32, min_ramp_s: f32) -> f32 {
        self.planned_accel_from(l, &self.quiet(cargo_kg), slope_deg, ground, dir, min_ramp_s)
    }

    /// `planned_accel_within` for a body standing as `held` stands (the arms and where the load is: in the bay, or in
    /// the hands before the body, which leaves the feet less room to brake on).
    pub fn planned_accel_from(&self, l: &DriveLimits, held: &Posture, slope_deg: f32, ground: Surface, dir: f32, min_ramp_s: f32) -> f32 {
        let p = Posture { source: if slope_deg.abs() >= 3.0 { Source::Slope } else { Source::Quiet }, slope_deg, ground, cargo_shift_m: [0.0, 0.0], ..*held };
        let r = self.solve(&p, &FeedForward::default(), 1e3);
        let cargo_kg = held.cargo_kg;
        let s = r.support;
        let keep = MARGIN_KEEP * 0.5 * (s[1] - s[0]);
        let room = if dir > 0.0 { r.zmp[0] - (s[0] + keep) } else { (s[1] - keep) - r.zmp[0] };
        let mut a = (G * room.max(0.0) / r.com[2].max(0.1)).min(l.accel_limit_mps2);
        let props = ground.props();
        let slope = slope_deg.to_radians();
        if !self.legged() && dir > 0.0 {
            let m = self.mass_kg() + cargo_kg;
            let trac = self.chassis.drive_force_n / m - G * (slope.sin() + surface::rolling_resistance(self.base, props) * slope.cos());
            a = a.min(trac.max(0.0));
        }
        a = a.min(surface::traction_accel_limit(self.base, props, if dir > 0.0 { slope_deg } else { -slope_deg }));
        (l.max_speed_mps / min_ramp_s.max(1e-3)).min(a).max(0.05)
    }

    pub fn capacity_on(&self, spec: &BaseSpec, ground: Surface, slope_deg: f32) -> f32 {
        let cap = self.drive_limits(spec, 0.0).capacity_kg;
        if self.legged() {
            return cap;
        }
        let th = slope_deg.to_radians();
        let need = th.sin() + surface::rolling_resistance(self.base, ground.props()) * th.cos();
        if need <= 1e-3 {
            return cap;
        }
        cap.min((self.chassis.drive_force_n / (G * need) - self.mass_kg()).max(0.0))
    }

    pub fn lift(&self, load_kg: f32, cargo_kg: f32, ff: &FeedForward, grip_capacity_n: f32) -> (BalanceReport, bool) {
        if !self.has_manipulator() {
            let mut r = self.solve(&Posture { source: Source::ArmLift, ..self.quiet(cargo_kg) }, ff, grip_capacity_n);
            r.feasible = false;
            r.limiting = Some("조작기 없음 (no_manipulator): 물건을 들 팔·집게가 없음".to_string());
            return (r, false);
        }
        let one = self.solve(&self.lift_posture(load_kg, false, cargo_kg), ff, grip_capacity_n);
        if one.feasible || self.arm_count() < 2 {
            return (one, false);
        }
        let two = self.solve(&self.lift_posture(load_kg, true, cargo_kg), ff, grip_capacity_n);
        if two.feasible {
            (two, true)
        } else {
            (one, false)
        }
    }
}

pub fn step_knee_lever(d: &Dims, h: f32) -> f32 {
    d.hip_m * (0.10 + 0.40 * h / d.hip_m)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RefCtx {
    pub load_ratio: f32,
    pub load_dist_m: f32,
    pub slope_deg: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Reference {
    pub value: f32,
    pub tolerance: f32,
    pub cite: &'static str,
}

fn r(value: f32, tolerance: f32, cite: &'static str) -> Option<Reference> {
    Some(Reference { value, tolerance: tolerance.max(1e-3), cite })
}

pub fn reference(source: Source, target: Segment, axis: &str, ctx: &RefCtx) -> Option<Reference> {
    const R5: f32 = 0.067;
    let lr = (ctx.load_ratio / R5).max(0.0);
    match (source, target, axis) {
        (_, Segment::Head, "tilt_deg") => r(0.0, if matches!(source, Source::FootLift | Source::StepUp) { 8.5 } else { 5.0 }, "Pozzo 1990 (머리 <20°), 보행 피치 1.5–8.5°"),
        (Source::ArmRaise, Segment::Arm, "peak_ratio") => r(0.15, 0.05, "어깨 ~15% (유도: Winter 질량, Marcondes 2019)"),
        (Source::ArmRaise, Segment::Foot, "com_shift_frac") => r(0.056, 0.017, "CoM +1.5 cm / 발 26.6 cm (유도)"),
        (Source::ArmRaise, Segment::Foot, "ankle_rise") => r(0.33, 0.10, "발목 +33% (유도)"),
        (Source::ArmRaise, Segment::Belly, "restore_frac") => r(0.2, 0.2, "무부하: 지지 분절이 뒤로 보상하나 다 막지 못함 (Aruin & Latash 1995)"),
        (Source::ArmLift, Segment::Arm, "peak_ratio") => {
            let v = (0.15 + 0.44 * lr).min(1.0);
            r(v, 0.3 * v, "어깨 무부하 15% → 5 kg 59%, 하중비에 선형 (유도)")
        }
        (Source::ArmLift, Segment::Foot, "com_shift_frac") => {
            let v = (1.5 + 4.2 * lr) / 26.6;
            r(v, 0.3 * v, "CoM +1.5 → +5.7 cm (5 kg), 하중비에 선형 (유도)")
        }
        (Source::ArmLift, Segment::Belly, "restore_frac") => {
            let v = (0.2 + 0.7 * lr.min(1.0)).min(0.9);
            r(v, 0.3, "5 kg: 골반·몸통 ≈ 6 cm 뒤로 = 거의 전부 회복 (유도)")
        }
        (Source::FootLift, Segment::Belly, "shift_frac") => r(1.0, 0.2, "좌우 CoM 이동 = 발 중심 거리의 절반 (유도)"),
        (Source::FootLift, Segment::Leg, "peak_ratio") => r(0.55, 0.10, "지지 고관절 외전 최대의 50–60% (유도)"),
        (Source::FootLift, Segment::Foot, "margin") => r(0.32, 0.12, "한 발 b_min 15.5 ± 6 mm / 발 반폭 48 mm (Hof 2005, 유도)"),
        (Source::StepUp, Segment::Leg, "peak_ratio") => r(0.46, 0.10, "무릎 최대의 42–50% (Lin 2004, 유도)"),
        (Source::Slope, Segment::Foot, "cop_shift_frac") => {
            let v = if ctx.slope_deg >= 0.0 { -0.006 * ctx.slope_deg } else { -0.008 * ctx.slope_deg };
            r(v, 0.3 * v.abs().max(0.01), "Atsawakaewmongkhon 2024 CoP 이동 (유도)")
        }
        (Source::CarryFront, Segment::Belly, "restore_frac") => r(0.9, 0.3, "앞쪽 짐: 뒤로 기울여 CoM 회복 (Hudson 2018 원리, 유도)"),
        (Source::CarryFront, Segment::Foot, "com_shift_frac") => {
            let v = 0.133 * (ctx.load_ratio * ctx.load_dist_m) / (0.133 * 0.30);
            r(v, 0.3 * v.max(0.01), "10 kg @ 30 cm: 보상 없으면 CoM 3.5 cm (유도)")
        }
        (_, Segment::Foot, "margin") => r(0.48, 0.15, "두 발 b_min 64 mm / 지지 반폭 133 mm (Hof 2005, 유도)"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(tier: Tier, base: BaseKind) -> Body {
        Body::new(&BodyPlan::uniform(tier), base, FingerCount::Five)
    }

    #[test]
    fn tier_masses_and_com_heights_match_the_research_totals() {
        for (tier, want) in [(Tier::Budget, 7.7), (Tier::Mid, 20.1), (Tier::Pro, 29.6)] {
            let b = body(tier, BaseKind::HumanFoot);
            let m = b.mass_kg();
            assert!((m - want).abs() / want < 0.15, "{tier:?}: {m} kg vs {want}");
            let h = b.com_height_m();
            assert!((0.5..0.75).contains(&h), "{tier:?}: CoM {h} m");
        }
        let chest = body(Tier::Mid, BaseKind::HumanFoot);
        let low = Body::new(&BodyPlan { battery_low: true, ..BodyPlan::default() }, BaseKind::HumanFoot, FingerCount::Five);
        let d = chest.com_height_m() - low.com_height_m();
        assert!(d > 0.01 && d < 0.04, "{d}");
    }

    #[test]
    fn an_arm_raise_moves_the_com_as_the_lever_rule_says() {
        let b = body(Tier::Mid, BaseKind::Wheel);
        let ff = FeedForward { enabled: false, gain: 1.0 };
        let r = b.solve(&Posture::arm_raise(0.0), &ff, 50.0);
        let (pts, _, _) = b.arm_points(0, 75.0, 15.0, 0.0);
        let (down, _, _) = b.arm_points(0, 0.0, 10.0, 0.0);
        let dm: f32 = pts.iter().zip(&down).map(|(a, d)| a.m * (a.p[0] - d.p[0])).sum();
        assert!((r.shift_uncomp_m - dm / b.mass_kg()).abs() < 1e-4, "{} vs {}", r.shift_uncomp_m, dm / b.mass_kg());
        assert!(r.shift_uncomp_m > 0.0);
        assert!(r.joint("shoulder_r").unwrap().torque_nm > r.joint("shoulder_l").unwrap().torque_nm);
    }

    #[test]
    fn what_can_be_lifted_follows_the_arm_preset() {
        let grip = 200.0;
        let ff = FeedForward::default();
        let budget = body(Tier::Budget, BaseKind::Wheel);
        let mid = body(Tier::Mid, BaseKind::Wheel);
        let pro = body(Tier::Pro, BaseKind::Wheel);
        assert!(!budget.lift(2.4, 0.0, &ff, grip).0.feasible, "a bus-servo elbow cannot lift a log");
        assert!(mid.lift(2.4, 0.0, &ff, grip).0.feasible);
        let (iron_mid, two) = mid.lift(4.0, 0.0, &ff, grip);
        assert!(iron_mid.feasible && two, "mid arms lift iron ore only with both arms: {:?}", iron_mid.limiting);
        let (iron_pro, two_pro) = pro.lift(4.0, 0.0, &ff, grip);
        assert!(iron_pro.feasible && !two_pro, "pro shoulder and elbow lift iron ore one-handed: {:?}", iron_pro.limiting);
        let cap = |elev: f32| {
            let mut kg = 0.0;
            while kg < 20.0 {
                let p = Posture { arm_elev_deg: [elev, 0.0], elbow_deg: [0.0, 10.0], hand_load_kg: [kg + 0.1, 0.0], ..Posture::arm_lift(0.0, false, 0.0) };
                if !mid.solve(&p, &ff, 1e6).joint("shoulder_r").map_or(false, |j| j.peak_ratio <= 1.0) {
                    break;
                }
                kg += 0.1;
            }
            kg
        };
        let (near, far) = (cap(30.0), cap(90.0));
        let ratio = near / far;
        assert!(ratio > 1.5 && ratio < 2.6, "reach 0.24 m vs 0.48 m: {near} kg vs {far} kg");
    }

    #[test]
    fn a_tall_robot_on_a_short_wheelbase_tips_early_and_a_waist_lean_helps() {
        let spec = BaseKind::Wheel.spec();
        let stiff = Body::new(&BodyPlan { belly: Tier::Budget, ..BodyPlan::default() }, BaseKind::Wheel, FingerCount::Five);
        let lean = body(Tier::Pro, BaseKind::Wheel);
        let a = stiff.drive_limits(&spec, 0.0);
        let b = Body::new(&BodyPlan { belly: Tier::Pro, ..BodyPlan::default() }, BaseKind::Wheel, FingerCount::Five).drive_limits(&spec, 0.0);
        assert!(a.max_slope_deg < spec.max_slope_deg, "{:?}", a.reasons);
        assert!(b.max_slope_deg > a.max_slope_deg, "a waist that can lean holds a steeper slope: {} vs {}", b.max_slope_deg, a.max_slope_deg);
        let h = stiff.com_height_m();
        let tip = ((1.0 - MARGIN_KEEP) * 0.5 * spec.footprint_m / h).atan().to_degrees();
        assert!((a.max_slope_deg - tip).abs() < 2.0, "{} vs {tip}", a.max_slope_deg);
        let _ = lean;
    }

    #[test]
    fn walking_speed_and_steps_follow_the_leg_preset_and_the_load() {
        let spec = BaseKind::HumanFoot.spec();
        let mid = body(Tier::Mid, BaseKind::HumanFoot);
        let empty = mid.drive_limits(&spec, 0.0);
        let loaded = mid.drive_limits(&spec, 6.0);
        assert!(empty.max_step_m >= 0.15, "a mid leg climbs the 15 cm curb: {:?}", empty.reasons);
        assert!(loaded.max_speed_mps <= empty.max_speed_mps);
        let plus = Body::new(&BodyPlan { leg: Tier::BudgetPlus, ..BodyPlan::default() }, BaseKind::HumanFoot, FingerCount::Five);
        let (p0, p2) = (plus.drive_limits(&spec, 0.0), plus.drive_limits(&spec, 2.0));
        assert!(p0.max_step_m < 0.10 && p0.max_step_m > 0.03, "{:?}", p0.reasons);
        assert!(p2.max_step_m < p0.max_step_m);
        let budget = body(Tier::Budget, BaseKind::HumanFoot).drive_limits(&spec, 0.0);
        assert!(budget.max_speed_mps < 0.05, "bus-servo knees cannot carry the robot: {:?}", budget.reasons);
        let pro = body(Tier::Pro, BaseKind::HumanFoot).drive_limits(&spec, 0.0);
        assert!(pro.capacity_kg > empty.capacity_kg);
    }

    #[test]
    fn a_foot_lift_shifts_the_pelvis_over_the_stance_foot() {
        let b = body(Tier::Mid, BaseKind::HumanFoot);
        let p = Posture { source: Source::FootLift, single_support: true, ..Posture::quiet(0.0) };
        let r = b.solve(&p, &FeedForward::default(), 50.0);
        assert!((r.com[1] + crate::dims::Dims::COMPACT.stance_m / 2.0).abs() < 0.02, "CoM over the right foot: {:?}", r.com);
        assert!(r.margin > 0.0, "{r:?}");
        let hip = r.joint("hip_roll").unwrap();
        assert!(hip.peak_ratio > 0.05 && hip.peak_ratio < 1.0, "{}", hip.peak_ratio);
        let hr = reference(Source::FootLift, Segment::Leg, "peak_ratio", &RefCtx::default()).unwrap();
        assert!(hip.peak_ratio < hr.value - hr.tolerance, "{} vs {hr:?}", hip.peak_ratio);
        let frac = r.couplings.iter().find(|c| c.axis == "shift_frac").unwrap().value;
        let rf = reference(Source::FootLift, Segment::Belly, "shift_frac", &RefCtx::default()).unwrap();
        assert!((frac - rf.value).abs() <= rf.tolerance, "{frac}");
    }

    #[test]
    fn the_feed_forward_reduces_the_dip_and_learning_the_gain_removes_it() {
        let b = body(Tier::Mid, BaseKind::HumanFoot);
        let p = Posture::arm_lift(2.0, false, 0.0);
        let off = b.solve(&p, &FeedForward { enabled: false, gain: 1.0 }, 100.0);
        let nominal = b.solve(&p, &FeedForward::default(), 100.0);
        let learned_gain = nominal.shift_uncomp_m / nominal.ff_predicted_m;
        let learned = b.solve(&p, &FeedForward { enabled: true, gain: learned_gain }, 100.0);
        assert!(nominal.margin >= off.margin - 1e-6);
        assert!((learned.margin - learned.margin_noff) >= (nominal.margin - nominal.margin_noff) - 1e-6);
        assert!((learned.margin - (learned.margin_x.min(learned.margin_y))).abs() < 1e-4, "a learned gain leaves no dip");
        let c = learned.couplings.iter().find(|c| c.target == Segment::Foot && c.axis == "ankle_rise").unwrap();
        assert!(c.effort.unwrap() <= c.effort_noff.unwrap());
    }

    #[test]
    fn traction_bounds_what_the_hands_may_push_and_a_slope_uses_it_up() {
        let b = body(Tier::Mid, BaseKind::HumanFoot);
        let flat = b.solve(&Posture::arm_lift(0.0, true, 0.0), &FeedForward::default(), 100.0).traction;
        assert!(flat.mu > 0.0 && flat.contact_use[0] < 1e-6 && flat.contact_use[1] < 1e-6, "{flat:?}");
        assert!((flat.push_max_n - flat.push_slide_n.min(flat.push_tip_n)).abs() < 1e-4);
        assert_eq!(flat.push_slides_first, flat.push_slide_n < flat.push_tip_n);
        assert!(flat.push_max_n > 0.0 && flat.pull_max_n > 0.0 && flat.hand_z_m > 0.3);
        assert!(!flat.push_slides_first && !flat.pull_slides_first, "{flat:?}");
        let up = b.solve(&Posture { slope_deg: 10.0, ..Posture::arm_lift(0.0, true, 0.0) }, &FeedForward::default(), 100.0).traction;
        assert!(up.contact_use[0] > 0.2 && up.contact_use[0] < 1.0, "{up:?}");
        assert!(up.push_slide_n < flat.push_slide_n && up.pull_slide_n > flat.push_slide_n - 1e-3, "{up:?}");
        let ice = b.solve(&Posture { ground: Surface::WetSand, ..Posture::arm_lift(0.0, true, 0.0) }, &FeedForward::default(), 100.0).traction;
        assert!(ice.mu < flat.mu && ice.push_slide_n < flat.push_slide_n);
    }

    #[test]
    fn references_cover_the_named_cells_and_scale_with_load() {
        let ctx = RefCtx { load_ratio: 0.067, ..Default::default() };
        let r5 = reference(Source::ArmLift, Segment::Arm, "peak_ratio", &ctx).unwrap();
        assert!((r5.value - 0.59).abs() < 1e-3);
        let r0 = reference(Source::ArmLift, Segment::Arm, "peak_ratio", &RefCtx::default()).unwrap();
        assert!((r0.value - 0.15).abs() < 1e-3);
        assert!(reference(Source::ArmRaise, Segment::Hand, "grip_effort", &ctx).is_none(), "no human number for grip: baseline only");
        let up = reference(Source::Slope, Segment::Foot, "cop_shift_frac", &RefCtx { slope_deg: 10.0, ..Default::default() }).unwrap();
        assert!(up.value < 0.0, "toes-up moves the CoP back");
    }
}
