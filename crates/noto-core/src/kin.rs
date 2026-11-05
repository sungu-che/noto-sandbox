use serde::{Deserialize, Serialize};

use crate::balance::{Body, JointLoad, G};
use crate::base::BaseKind;
use crate::dynamics::{BodyPose, JointPoint, LinkBox};
use crate::dims::Dims;
use crate::morph::{column_top_m, HeadMount, Torso};
use crate::parts::*;
use crate::Source;


pub const KNEE_FLEX_MAX_DEG: f32 = 135.0;
pub const ANKLE_DORSI_MAX_DEG: f32 = 30.0;
pub const HIP_FLEX_MAX_DEG: f32 = 106.0;
pub const WAIST_PITCH_MAX_DEG: f32 = 70.0;
pub const NECK_PITCH_MAX_DEG: f32 = 45.0;
/// Cervical rotation each way: 80° (AAOS, *Joint Motion*, 1965), as the other limits here are a person's. At 90° a
/// glance turned the head past what a neck turns.
pub const NECK_PAN_MAX_DEG: f32 = 80.0;
pub const SHOULDER_ABD_MAX_DEG: f32 = 129.0;
pub const SHOULDER_ADD_MAX_DEG: f32 = 19.5;
pub const HIP_ABD_MAX_DEG: f32 = 30.0;
pub const TRUNK_THIGH_FOLD_MAX_DEG: f32 = 150.0;
pub const GROUND_CLEARANCE_M: f32 = 0.01;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct KinJoint {
    pub name: String,
    pub segment: Segment,
    pub pos: [f32; 3],
    pub angle_deg: Option<f32>,
    pub actuator: String,
    pub torque: f32,
    pub peak: f32,
    pub cont: f32,
    pub ratio: f32,
    pub effort: f32,
    pub in_range: bool,
    pub structural: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ContactPatch {
    pub name: String,
    pub normal_n: f32,
    pub shear_n: f32,
    pub cop: [f32; 2],
    pub rect: [f32; 4],
    pub pressure_kpa: Option<[f32; 2]>,
    pub loaded_len_m: Option<f32>,
    pub shear_y_n: f32,
    pub torsion_nm: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sole: Option<crate::contact::SoleContact>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct KinPose {
    pub hip: [f32; 2],
    pub feet: [[f32; 2]; 2],
    pub pelvis_y: f32,
    pub pelvis_deg: f32,
    pub waist_deg: f32,
    pub neck_deg: f32,
    #[serde(default)]
    pub neck_pan_deg: f32,
    pub shoulder_deg: [f32; 2],
    pub elbow_deg: [f32; 2],
    #[serde(default)]
    pub shoulder_abd_deg: [f32; 2],
    #[serde(default)]
    pub stance_out_m: f32,
    pub base_pitch_deg: f32,
    pub cargo_kg: f32,
    pub closure: f32,
    #[serde(default)]
    pub lift_m: f32,
    #[serde(default)]
    pub toe_deg: [f32; 2],
    /// Fork tine height above the ground (m) on a wheeled or tracked base (`act::fork`); 0 on legs.
    #[serde(default)]
    pub fork_m: f32,
}

impl KinPose {
    pub fn stand(body: &Body, cargo_kg: f32) -> KinPose {
        KinPose::stand_at(body.dims.hip_m, cargo_kg)
    }

    /// Standing, for a compact body (`Dims::COMPACT`).
    pub fn stand_compact(cargo_kg: f32) -> KinPose {
        KinPose::stand_at(Dims::COMPACT.hip_m, cargo_kg)
    }

    fn stand_at(hip_m: f32, cargo_kg: f32) -> KinPose {
        KinPose {
            hip: [0.0, hip_m],
            feet: [[0.0, 0.0], [0.0, 0.0]],
            pelvis_y: 0.0,
            pelvis_deg: 0.0,
            waist_deg: 0.0,
            neck_deg: 0.0,
            neck_pan_deg: 0.0,
            shoulder_deg: [0.0, 0.0],
            elbow_deg: [10.0, 10.0],
            shoulder_abd_deg: [0.0, 0.0],
            stance_out_m: 0.0,
            base_pitch_deg: 0.0,
            cargo_kg,
            closure: 0.0,
            lift_m: 0.0,
            toe_deg: [0.0, 0.0],
            fork_m: 0.0,
        }
    }

    pub fn trunk_deg(&self) -> f32 {
        self.pelvis_deg + self.waist_deg
    }

    pub fn lerp(a: &KinPose, b: &KinPose, u: f32) -> KinPose {
        let l = |x: f32, y: f32| x + (y - x) * u;
        let l2 = |x: [f32; 2], y: [f32; 2]| [l(x[0], y[0]), l(x[1], y[1])];
        KinPose {
            hip: l2(a.hip, b.hip),
            feet: [l2(a.feet[0], b.feet[0]), l2(a.feet[1], b.feet[1])],
            pelvis_y: l(a.pelvis_y, b.pelvis_y),
            pelvis_deg: l(a.pelvis_deg, b.pelvis_deg),
            waist_deg: l(a.waist_deg, b.waist_deg),
            neck_deg: l(a.neck_deg, b.neck_deg),
            neck_pan_deg: l(a.neck_pan_deg, b.neck_pan_deg),
            shoulder_deg: l2(a.shoulder_deg, b.shoulder_deg),
            elbow_deg: l2(a.elbow_deg, b.elbow_deg),
            shoulder_abd_deg: l2(a.shoulder_abd_deg, b.shoulder_abd_deg),
            stance_out_m: l(a.stance_out_m, b.stance_out_m),
            base_pitch_deg: l(a.base_pitch_deg, b.base_pitch_deg),
            cargo_kg: l(a.cargo_kg, b.cargo_kg),
            closure: l(a.closure, b.closure),
            lift_m: l(a.lift_m, b.lift_m),
            toe_deg: l2(a.toe_deg, b.toe_deg),
            fork_m: l(a.fork_m, b.fork_m),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct KinLoads {
    pub hand_force: [[f32; 3]; 2],
    pub bumper_force: [f32; 3],
    pub accel: [f32; 3],
    #[serde(default, skip_serializing_if = "is_zero3")]
    pub body_force: [f32; 3],
    #[serde(default, skip_serializing_if = "is_zero3")]
    pub body_point: [f32; 3],
    /// Where the held load's centre sits off the palms (forward, left, up; m): its weight and inertia (`hand_force`)
    /// act there, so a load that slid forward in the hands loads the shoulders more and moves the ZMP.
    #[serde(default, skip_serializing_if = "is_zero3")]
    pub cargo_shift_m: [f32; 3],
}

fn is_zero3(v: &[f32; 3]) -> bool {
    *v == [0.0; 3]
}

impl KinLoads {
    pub fn holding(kg: f32, two_arms: bool, a_up: f32) -> KinLoads {
        let f = -kg * (G + a_up);
        if two_arms {
            KinLoads { hand_force: [[0.0, 0.0, 0.5 * f], [0.0, 0.0, 0.5 * f]], ..KinLoads::default() }
        } else {
            KinLoads { hand_force: [[0.0, 0.0, f], [0.0; 3]], ..KinLoads::default() }
        }
    }

    /// `holding`, ridden `shift_m` off the palms (`cargo_shift_m`).
    pub fn holding_shifted(kg: f32, two_arms: bool, a_up: f32, shift_m: [f32; 3]) -> KinLoads {
        KinLoads { cargo_shift_m: shift_m, ..KinLoads::holding(kg, two_arms, a_up) }
    }

    /// A load of `kg` carried in the hands of a body accelerating at `accel` (forward, left, up; m/s²) — its weight and
    /// its inertia on the palms, split between `arms` hands — with the body's own `accel`, ridden `shift_m` off the palms.
    pub fn carrying(kg: f32, arms: usize, accel: [f32; 3], shift_m: [f32; 2]) -> KinLoads {
        let f = [-kg * accel[0], -kg * accel[1], -kg * (G + accel[2])];
        let hand_force = if arms >= 2 { [f.map(|x| 0.5 * x); 2] } else { [f, [0.0; 3]] };
        KinLoads { hand_force, accel, cargo_shift_m: [shift_m[0], shift_m[1], 0.0], ..KinLoads::default() }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct KinReport {
    pub joints: Vec<KinJoint>,
    pub contacts: Vec<ContactPatch>,
    pub com: [f32; 3],
    pub zmp: [f32; 2],
    pub support: [f32; 4],
    pub margin: f32,
    pub grf: [f32; 3],
    pub mass_kg: f32,
    pub hands: [[f32; 3]; 2],
    pub knees: [[f32; 3]; 2],
    pub bumper: Option<[f32; 3]>,
    pub feasible: bool,
    pub limiting: Option<String>,
    pub out_of_range: Vec<String>,
    pub effort_sum: f32,
    pub pose: KinPose,
    #[serde(skip)]
    pub boxes: Vec<LinkBox>,
}

impl KinReport {
    pub fn joint(&self, name: &str) -> Option<&KinJoint> {
        self.joints.iter().find(|j| j.name == name)
    }

    pub fn contact(&self, name: &str) -> Option<&ContactPatch> {
        self.contacts.iter().find(|c| c.name == name)
    }

    pub fn worst(&self) -> Option<&KinJoint> {
        self.joints.iter().filter(|j| !j.structural).max_by(|a, b| a.ratio.total_cmp(&b.ratio))
    }

    pub fn joint_loads(&self) -> Vec<JointLoad> {
        self.joints
            .iter()
            .filter(|j| !j.structural)
            .map(|j| JointLoad { joint: j.name.clone(), segment: j.segment, actuator: j.actuator.clone(), torque_nm: j.torque.abs(), cont_nm: j.cont, peak_nm: j.peak, effort: j.effort, peak_ratio: j.ratio })
            .collect()
    }

    pub fn body_pose(&self, t: f32, source: Source) -> BodyPose {
        let joints = self
            .joints
            .iter()
            .map(|j| JointPoint {
                name: j.name.clone(),
                kind: if j.structural { "structure" } else if j.name.starts_with("drive") { "statics" } else { "sim" },
                pos: j.pos,
                q_deg: j.angle_deg,
                tau_nm: j.torque.abs(),
                peak_nm: j.peak,
                ratio: j.ratio,
                saturated: j.ratio > 1.0,
                stalled: false,
            })
            .collect();
        BodyPose { t, source, boxes: self.boxes.clone(), joints, com: self.com, zmp: self.zmp, support: self.support, margin: self.margin, grf: self.grf }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Rom {
    pub knee_max: f32,
    pub ankle_dorsi_max: f32,
    pub hip_flex_max: f32,
    pub waist_max: f32,
    pub neck_max: f32,
    #[serde(default)]
    pub pan_max: f32,
    pub shoulder: [f32; 2],
    pub elbow_max: f32,
    pub shoulder_abd: [f32; 2],
    pub stance_out_max_m: f32,
    pub lift_max_m: f32,
}

impl Rom {
    pub fn of(body: &Body) -> Rom {
        let (neck_max, pan_max) = body.head_range_deg();
        let l = &body.limits;
        Rom {
            knee_max: l.knee_max.unwrap_or(KNEE_FLEX_MAX_DEG),
            ankle_dorsi_max: l.ankle_dorsi_max.unwrap_or(ANKLE_DORSI_MAX_DEG),
            hip_flex_max: l.hip_flex_max.unwrap_or(HIP_FLEX_MAX_DEG),
            waist_max: if body.belly.pitch.is_some() { l.waist_max.unwrap_or(WAIST_PITCH_MAX_DEG) } else { 0.0 },
            neck_max: l.neck_max.unwrap_or(neck_max),
            pan_max,
            shoulder: l.shoulder.unwrap_or([-60.0, 180.0]),
            elbow_max: l.elbow_max.unwrap_or(150.0),
            shoulder_abd: [-SHOULDER_ADD_MAX_DEG, SHOULDER_ABD_MAX_DEG],
            stance_out_max_m: if body.legged() { ((body.dims.hip_m - body.dims.ankle_m) * HIP_ABD_MAX_DEG.to_radians().sin() + body.dims.hip_half_m - 0.5 * body.dims.stance_m).max(0.0) } else { 0.0 },
            lift_max_m: if body.legged() { 0.0 } else { body.lift.as_ref().map_or(0.0, |l| l.stroke_m) },
        }
    }
}

type V3 = [f32; 3];

fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scale(a: V3, s: f32) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn up_dir(deg: f32) -> V3 {
    let a = deg.to_radians();
    [a.sin(), 0.0, a.cos()]
}
fn down_dir(deg: f32) -> V3 {
    let a = deg.to_radians();
    [a.sin(), 0.0, -a.cos()]
}
fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn fore_aft(deg: f32) -> V3 {
    let a = deg.to_radians();
    [a.cos(), 0.0, -a.sin()]
}
pub fn rotate(v: V3, k: V3, theta: f32) -> V3 {
    let (s, c) = theta.sin_cos();
    add(add(scale(v, c), scale(cross(k, v), s)), scale(k, dot(k, v) * (1.0 - c)))
}
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct HeadCamera {
    pub pos: [f32; 3],
    pub pitch_down_deg: f32,
    pub pan_deg: f32,
}


pub fn head_camera(body: &Body, pose: &KinPose) -> HeadCamera {
    camera(body, pose)
}

fn neck_of(body: &Body, pose: &KinPose) -> (V3, f32) {
    if body.morph.torso == Torso::Mast {
        return ([0.0, pose.pelvis_y, body.dims.column_top_m() - pose.lift_m], 0.0);
    }
    let hip_c: V3 = if body.legged() { [pose.hip[0], pose.pelvis_y, pose.hip[1]] } else { [body.dims.seat_x_m, pose.pelvis_y, body.dims.hip_m - pose.lift_m] };
    let waist = add(hip_c, scale(up_dir(pose.pelvis_deg), body.dims.pelvis_m()));
    let trunk = pose.trunk_deg();
    let v = up_dir(trunk);
    (add(waist, scale(v, body.dims.trunk_m() + body.dims.neck_up_m)), trunk)
}

fn lens(d: &Dims, pivot: V3, head_deg: f32, pan_deg: f32) -> V3 {
    let pan = pan_deg.to_radians();
    let fwd = fore_aft(head_deg);
    let ahead = [fwd[0] * pan.cos(), pan.sin(), fwd[2] * pan.cos()];
    add(add(pivot, scale(up_dir(head_deg), d.camera_up_m)), scale(ahead, d.camera_fwd_m))
}

pub fn camera(body: &Body, pose: &KinPose) -> HeadCamera {
    let pitch = if body.legged() { 0.0 } else { pose.base_pitch_deg };
    let at = |eye: V3, head: f32, pan: f32| HeadCamera { pos: if pitch != 0.0 { pitch_about_origin(eye, pitch) } else { eye }, pitch_down_deg: head - pitch, pan_deg: pan };
    match body.camera_mount() {
        HeadMount::Neck => {
            let (neck, trunk) = neck_of(body, pose);
            let head = trunk + pose.neck_deg;
            at(lens(&body.dims, neck, head, pose.neck_pan_deg), head, pose.neck_pan_deg)
        }
        HeadMount::Fixed { tilt_deg } => {
            let (neck, trunk) = neck_of(body, pose);
            let head = trunk + tilt_deg as f32;
            at(lens(&body.dims, neck, head, 0.0), head, 0.0)
        }
        HeadMount::MastCamera { height_mm, pan_tilt } => {
            let pivot = [body.mast_camera_x(), 0.0, height_mm as f32 / 1000.0 - body.dims.camera_up_m];
            let (tilt, pan) = if pan_tilt { (pose.neck_deg, pose.neck_pan_deg) } else { (0.0, 0.0) };
            at(lens(&body.dims, pivot, tilt, pan), tilt, pan)
        }
        HeadMount::ChassisFront { height_mm } => at([0.5 * body.support_len_m, 0.0, height_mm as f32 / 1000.0], 0.0, 0.0),
        HeadMount::None => {
            let p = if body.has_torso() || body.has_column() { neck_of(body, pose).0 } else { [0.0, 0.0, body.dims.deck_z_m()] };
            at(p, 0.0, 0.0)
        }
    }
}

fn pitch_about_origin(p: V3, deg: f32) -> V3 {
    let (s, c) = (-deg.to_radians()).sin_cos();
    [p[0] * c + p[2] * s, p[1], -p[0] * s + p[2] * c]
}

pub fn knee_ik(leg: &Dims, hip: [f32; 2], ankle: [f32; 2]) -> Option<([f32; 2], f32)> {
    let (dx, dz) = (ankle[0] - hip[0], ankle[1] - hip[1]);
    let d = (dx * dx + dz * dz).sqrt();
    if d > leg.thigh_m() + leg.shank_m() + 1e-4 || d < 1e-3 {
        return None;
    }
    let d = d.min(leg.thigh_m() + leg.shank_m());
    let h = (leg.thigh_m() * leg.thigh_m() - 0.25 * d * d).max(0.0).sqrt();
    let (mx, mz) = (hip[0] + 0.5 * dx, hip[1] + 0.5 * dz);
    let (mut px, mut pz) = (-dz / d, dx / d);
    if px < 0.0 {
        px = -px;
        pz = -pz;
    }
    let knee = [mx + h * px, mz + h * pz];
    let interior = (1.0 - 0.5 * d * d / (leg.thigh_m() * leg.thigh_m())).clamp(-1.0, 1.0).acos();
    Some((knee, (std::f32::consts::PI - interior).to_degrees()))
}

pub fn support_margin(support: [f32; 4], zmp: [f32; 2]) -> f32 {
    let (hx, hy) = (0.5 * (support[1] - support[0]), 0.5 * (support[3] - support[2]));
    let (cx, cy) = (0.5 * (support[0] + support[1]), 0.5 * (support[2] + support[3]));
    (1.0 - (zmp[0] - cx).abs() / hx.max(1e-3)).min(1.0 - (zmp[1] - cy).abs() / hy.max(1e-3))
}

pub fn knee_ik3(leg: &Dims, hip: [f32; 3], ankle: [f32; 3]) -> Option<([f32; 3], f32)> {
    let d = sub(ankle, hip);
    let len = dot(d, d).sqrt();
    if len > leg.thigh_m() + leg.shank_m() + 1e-4 || len < 1e-3 {
        return None;
    }
    let len = len.min(leg.thigh_m() + leg.shank_m());
    let u = scale(d, 1.0 / len);
    let fwd = sub([1.0, 0.0, 0.0], scale(u, u[0]));
    let f = scale(fwd, 1.0 / dot(fwd, fwd).sqrt().max(1e-6));
    let h = (leg.thigh_m() * leg.thigh_m() - 0.25 * len * len).max(0.0).sqrt();
    let knee = add(add(hip, scale(d, 0.5)), scale(f, h));
    let interior = (1.0 - 0.5 * len * len / (leg.thigh_m() * leg.thigh_m())).clamp(-1.0, 1.0).acos();
    Some((knee, (std::f32::consts::PI - interior).to_degrees()))
}

#[derive(Clone, Copy, PartialEq)]
enum Grp {
    Head,
    Trunk,
    Pelvis,
    Upper(usize),
    Fore(usize),
    Hand(usize),
    Thigh(usize),
    Shank(usize),
    Foot(usize),
    Base,
}

struct Mass {
    group: Grp,
    m: f32,
    p: V3,
}

pub fn thigh_share(body: &Body) -> f32 {
    let full = body.dims.thigh_m() + 0.45 * body.dims.shank_m();
    ((full - body.leg.com_down_m) / (full - 0.45 * body.dims.thigh_m())).clamp(0.3, 0.95)
}

pub fn solve(body: &Body, pose: &KinPose, loads: &KinLoads) -> KinReport {
    solve_with(body, pose, loads, true)
}

pub fn solve_fast(body: &Body, pose: &KinPose, loads: &KinLoads) -> KinReport {
    solve_with(body, pose, loads, false)
}

fn solve_with(body: &Body, pose: &KinPose, loads: &KinLoads, draw: bool) -> KinReport {
    let k = body.dims.k;
    let rom = Rom::of(body);
    let legged = body.legged();
    let torso = body.morph.torso;
    let n_arms = body.arm_count();
    let mut masses: Vec<Mass> = Vec::new();
    let mut boxes: Vec<LinkBox> = Vec::new();
    let mut out_of_range: Vec<String> = Vec::new();
    let tag = |s: usize| if s == 0 { "r" } else { "l" };

    let wide = pose.stance_out_m > 1e-6;
    let half = body.dims.stance_m * 0.5 + pose.stance_out_m.max(0.0);
    let foot_y = [-half, half];
    let mut knees = [[0.0f32; 3]; 2];
    let mut ankles = [[0.0f32; 3]; 2];
    let mut hips = [[0.0f32; 3]; 2];
    let mut stance = [false; 2];
    let mut leg_angles = [(0.0f32, 0.0f32, 0.0f32); 2];
    let mut hip_abd_deg = [0.0f32; 2];
    let hip_c: V3 = if legged { [pose.hip[0], pose.pelvis_y, pose.hip[1]] } else { [body.dims.seat_x_m, pose.pelvis_y, body.dims.hip_m - pose.lift_m] };
    if legged {
        let share = thigh_share(body);
        for s in 0..2 {
            let lift = pose.feet[s][1].max(0.0);
            stance[s] = lift <= 1e-4;
            let hip2 = [hip_c[0], hip_c[2]];
            let rock = if stance[s] && pose.toe_deg[s] > 1e-3 { crate::contact::rocker_ankle_shift(body.leg.foot_len_m, body.dims.ankle_m, pose.toe_deg[s]) } else { [0.0, 0.0] };
            let ankle2 = [pose.feet[s][0] + rock[0], body.dims.ankle_m + lift + rock[1]];
            let y = foot_y[s] + if stance[s] { 0.0 } else { pose.pelvis_y };
            hips[s] = [hip_c[0], hip_c[1] + if s == 0 { -body.dims.hip_half_m } else { body.dims.hip_half_m }, hip_c[2]];
            let too_long = |out: &mut Vec<String>| out.push(format!("leg_{} 다리 길이를 넘는 자세", tag(s)));
            let (knee, ky, knee_flex) = if wide {
                match knee_ik3(&body.dims, hips[s], [ankle2[0], y, ankle2[1]]) {
                    Some((k, flex)) => ([k[0], k[2]], k[1], flex),
                    None => {
                        too_long(&mut out_of_range);
                        ([0.5 * (hip2[0] + ankle2[0]), 0.5 * (hip2[1] + ankle2[1])], 0.5 * (y + hips[s][1]), 0.0)
                    }
                }
            } else {
                let (k, flex) = knee_ik(&body.dims, hip2, ankle2).unwrap_or_else(|| {
                    too_long(&mut out_of_range);
                    let (dx, dz) = (ankle2[0] - hip2[0], ankle2[1] - hip2[1]);
                    let d = (dx * dx + dz * dz).sqrt().max(1e-3);
                    ([hip2[0] + dx / d * body.dims.thigh_m(), hip2[1] + dz / d * body.dims.thigh_m()], 0.0)
                });
                (k, y, flex)
            };
            knees[s] = [knee[0], ky, knee[1]];
            ankles[s] = [ankle2[0], y, ankle2[1]];
            if wide {
                hip_abd_deg[s] = (y - hips[s][1]).abs().atan2((hips[s][2] - ankle2[1]).max(1e-3)).to_degrees();
                if hip_abd_deg[s] > HIP_ABD_MAX_DEG + 0.5 {
                    out_of_range.push(format!("hip_roll_{} 외전 {:.0}° > {:.0}°", tag(s), hip_abd_deg[s], HIP_ABD_MAX_DEG));
                }
            }
            let shank = (knee[0] - ankle2[0]).atan2(knee[1] - ankle2[1]).to_degrees() - if stance[s] { pose.toe_deg[s].max(0.0) } else { 0.0 };
            let thigh = (knee[0] - hip2[0]).atan2(hip2[1] - knee[1]).to_degrees();
            leg_angles[s] = (shank, knee_flex, pose.pelvis_deg + thigh);
            masses.push(Mass { group: Grp::Thigh(s), m: body.leg.mass_kg * share, p: add(hips[s], scale(sub(knees[s], hips[s]), 0.45)) });
            masses.push(Mass { group: Grp::Shank(s), m: body.leg.mass_kg * (1.0 - share), p: add(knees[s], scale(sub(ankles[s], knees[s]), 0.45)) });
            let fl = body.leg.foot_len_m;
            let sole = ankles[s][2] - body.dims.ankle_m;
            let foot_c = [ankles[s][0] + 0.25 * fl, y, sole + 0.03 * k];
            masses.push(Mass { group: Grp::Foot(s), m: body.leg.foot_mass_kg, p: foot_c });
            boxes.push(limb_box(if s == 0 { "thigh_r" } else { "thigh_l" }, Segment::Leg, hips[s], knees[s], 0.05 * k));
            boxes.push(limb_box(if s == 0 { "shank_r" } else { "shank_l" }, Segment::Leg, knees[s], ankles[s], 0.045 * k));
            boxes.push(LinkBox { name: if s == 0 { "foot_r" } else { "foot_l" }, segment: Segment::Foot, center: foot_c, axes: EYE3, half: [fl * 0.5, body.leg.foot_width_m * 0.5, 0.03 * k] });
            if knee_flex > rom.knee_max + 0.5 {
                out_of_range.push(format!("knee_{} {:.0}° > {:.0}°", tag(s), knee_flex, rom.knee_max));
            }
            if stance[s] && shank > rom.ankle_dorsi_max + 0.5 {
                out_of_range.push(format!("ankle_{} 배측굴곡 {:.0}° > {:.0}°", tag(s), shank, rom.ankle_dorsi_max));
            }
            if leg_angles[s].2 > rom.hip_flex_max + 0.5 {
                out_of_range.push(format!("hip_pitch_{} {:.0}° > {:.0}°", tag(s), leg_angles[s].2, rom.hip_flex_max));
            }
        }
    } else {
        if let Some((m, p)) = body.column() {
            masses.push(Mass { group: Grp::Base, m, p });
        }
        masses.push(Mass { group: Grp::Base, m: body.chassis.mass_kg, p: [0.0, 0.0, body.chassis.com_m] });
        for mm in body.mount_masses() {
            masses.push(Mass { group: if mm.segment == Segment::Head { Grp::Head } else { Grp::Base }, m: mm.kg, p: mm.at });
        }
    }

    let u = up_dir(pose.pelvis_deg);
    let waist = add(hip_c, scale(u, body.dims.pelvis_m()));
    let trunk_deg = if torso == Torso::Humanoid { pose.trunk_deg() } else { 0.0 };
    let v = up_dir(trunk_deg);
    let (shoulder_c, neck) = match torso {
        Torso::Humanoid => {
            let shoulder_c = add(waist, scale(v, body.dims.trunk_m()));
            (shoulder_c, add(shoulder_c, scale(v, body.dims.neck_up_m)))
        }
        Torso::Mast => ([0.0, pose.pelvis_y, body.dims.shoulder_m - pose.lift_m], [0.0, pose.pelvis_y, body.dims.column_top_m() - pose.lift_m]),
        Torso::None => (body.shoulder_at(0, pose.lift_m), [0.0, 0.0, body.dims.deck_z_m()]),
    };
    let neck = match body.camera_mount() {
        HeadMount::MastCamera { height_mm, .. } => [body.mast_camera_x(), 0.0, height_mm as f32 / 1000.0 - body.dims.camera_up_m],
        _ => neck,
    };
    let head_tilt = match body.camera_mount() {
        HeadMount::Neck => pose.neck_deg,
        HeadMount::Fixed { tilt_deg } => tilt_deg as f32,
        HeadMount::MastCamera { pan_tilt: true, .. } => pose.neck_deg,
        _ => 0.0,
    };
    let head_dir = up_dir(trunk_deg + head_tilt);
    if torso == Torso::Humanoid {
        let belly_p = add(hip_c, scale(u, body.dims.pelvis_m() - body.belly.com_down_m));
        masses.push(Mass { group: Grp::Pelvis, m: body.belly.mass_kg, p: belly_p });
        let chest_p = add(waist, scale(v, body.chest.com_up_m));
        masses.push(Mass { group: Grp::Trunk, m: body.chest.mass_kg, p: chest_p });
        if body.plan.battery_low {
            masses.push(Mass { group: Grp::Pelvis, m: body.chest.battery_kg, p: belly_p });
        } else {
            masses.push(Mass { group: Grp::Trunk, m: body.chest.battery_kg, p: chest_p });
        }
    }
    if pose.cargo_kg > 0.0 {
        let (at, on_pelvis) = body.cargo_point();
        if on_pelvis {
            let fwd = [u[2], 0.0, -u[0]];
            let bay = add(add(hip_c, scale(u, body.dims.cargo_z_m() - body.dims.hip_m)), scale(fwd, body.dims.cargo_x_m));
            masses.push(Mass { group: Grp::Pelvis, m: pose.cargo_kg, p: bay });
        } else {
            masses.push(Mass { group: Grp::Base, m: pose.cargo_kg, p: at });
        }
    }
    if body.head_on_trunk() {
        masses.push(Mass { group: Grp::Head, m: body.head.mass_kg, p: add(neck, scale(head_dir, body.head.com_up_m)) });
    } else if let (true, Some(m)) = (body.has_neck(), masses.iter_mut().find(|m| m.group == Grp::Head)) {
        m.p = add(neck, scale(head_dir, body.head.com_up_m));
    }
    if torso == Torso::Humanoid {
        boxes.push(oriented_box("pelvis", Segment::Belly, add(hip_c, scale(u, 0.06 * k)), pose.pelvis_deg, [0.09 * k, 0.14 * k, 0.07 * k]));
        boxes.push(oriented_box("trunk", Segment::Chest, add(waist, scale(v, 0.5 * body.dims.trunk_m() + 0.01 * k)), trunk_deg, [0.10 * k, 0.16 * k, 0.5 * body.dims.trunk_m() + 0.01 * k]));
    }
    if body.head_on_trunk() {
        boxes.push(oriented_box("head", Segment::Head, add(neck, scale(head_dir, 0.07 * k)), trunk_deg + head_tilt, [0.07 * k, 0.08 * k, 0.07 * k]));
    } else if matches!(body.camera_mount(), HeadMount::MastCamera { .. }) {
        boxes.push(oriented_box("head", Segment::Head, add(neck, scale(head_dir, 0.05)), head_tilt, [0.04, 0.07, 0.04]));
    }
    if body.mast_arms() {
        let c = body.shoulder_at(0, pose.lift_m);
        boxes.push(oriented_box("carriage", Segment::Leg, [body.mast_camera_x(), 0.0, c[2]], 0.0, [0.04, 0.05, 0.04]));
    }

    let (m_up, r_up, m_fore) = body.arm_split();
    let mut hands = [[0.0f32; 3]; 2];
    let mut shoulders = [[0.0f32; 3]; 2];
    let mut elbows = [[0.0f32; 3]; 2];
    let mut flex_axes = [[0.0f32, 1.0, 0.0]; 2];
    let trunk_fwd = fore_aft(trunk_deg);
    for s in 0..2 {
        let sh = match torso {
            Torso::Humanoid | Torso::Mast => {
                let y = if s == 0 { -body.dims.shoulder_half_m } else { body.dims.shoulder_half_m };
                [shoulder_c[0], shoulder_c[1] + y, shoulder_c[2]]
            }
            Torso::None => body.shoulder_at(s, pose.lift_m),
        };
        if s >= n_arms {
            let f = loads.hand_force[s];
            let n = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt();
            if n > 1e-3 {
                out_of_range.push(format!("{} 없음: 손 하중 {:.0} N을 받을 팔이 없음", if s == 0 { "오른팔" } else { "왼팔" }, n));
            }
            shoulders[s] = sh;
            elbows[s] = sh;
            hands[s] = sh;
            continue;
        }
        let up_ang = pose.shoulder_deg[s] - trunk_deg;
        let mut du = down_dir(up_ang);
        let mut df = down_dir(up_ang + pose.elbow_deg[s]);
        let abd = pose.shoulder_abd_deg[s];
        if abd.abs() > 1e-6 {
            let theta = if s == 0 { -abd } else { abd }.to_radians();
            du = rotate(du, trunk_fwd, theta);
            df = rotate(df, trunk_fwd, theta);
            flex_axes[s] = rotate([0.0, 1.0, 0.0], trunk_fwd, theta);
        }
        let el = add(sh, scale(du, body.dims.upper_arm_m));
        let wrist = add(el, scale(df, body.dims.forearm_m));
        let hand = add(wrist, scale(df, body.dims.palm_m));
        shoulders[s] = sh;
        elbows[s] = el;
        hands[s] = hand;
        masses.push(Mass { group: Grp::Upper(s), m: m_up, p: add(sh, scale(du, r_up)) });
        masses.push(Mass { group: Grp::Fore(s), m: m_fore, p: add(el, scale(df, 0.45 * body.dims.forearm_m)) });
        masses.push(Mass { group: Grp::Hand(s), m: body.hand_kg, p: hand });
        boxes.push(limb_box(if s == 0 { "upper_arm_r" } else { "upper_arm_l" }, Segment::Arm, sh, el, 0.035 * k));
        boxes.push(limb_box(if s == 0 { "forearm_r" } else { "forearm_l" }, Segment::Arm, el, wrist, 0.03 * k));
        boxes.extend(hand_boxes(s, wrist, df, flex_axes[s], pose.closure, body));
        if !(rom.shoulder[0] - 0.5..=rom.shoulder[1] + 0.5).contains(&pose.shoulder_deg[s]) {
            out_of_range.push(format!("shoulder_{} {:.0}°", tag(s), pose.shoulder_deg[s]));
        }
        if !(-0.5..=rom.elbow_max + 0.5).contains(&pose.elbow_deg[s]) {
            out_of_range.push(format!("elbow_{} {:.0}°", tag(s), pose.elbow_deg[s]));
        }
        if !(rom.shoulder_abd[0] - 0.5..=rom.shoulder_abd[1] + 0.5).contains(&abd) {
            out_of_range.push(format!("shoulder_abd_{} {:.0}° (범위 {:.0}…{:.0}°)", tag(s), abd, rom.shoulder_abd[0], rom.shoulder_abd[1]));
        }
    }
    if legged && wide && pose.stance_out_m > rom.stance_out_max_m + 1e-3 {
        out_of_range.push(format!("발 벌림 {:.2} m > {:.2} m (고관절 외전 범위)", pose.stance_out_m, rom.stance_out_max_m));
    }
    if pose.waist_deg.abs() > rom.waist_max + 0.5 {
        out_of_range.push(if rom.waist_max == 0.0 { format!("허리 피치 관절 없음 (요청 {:.0}°)", pose.waist_deg) } else { format!("waist_pitch {:.0}° > {:.0}°", pose.waist_deg, rom.waist_max) });
    }
    if pose.neck_deg.abs() > rom.neck_max + 0.5 {
        out_of_range.push(if rom.neck_max == 0.0 { format!("목 관절 없음 (요청 {:.0}°)", pose.neck_deg) } else { format!("neck {:.0}° > {:.0}°", pose.neck_deg, rom.neck_max) });
    }
    if pose.neck_pan_deg.abs() > rom.pan_max + 0.5 {
        out_of_range.push(if rom.pan_max == 0.0 { format!("목 팬 관절 없음 (요청 {:.0}°)", pose.neck_pan_deg) } else { format!("neck_pan {:.0}° > {:.0}°", pose.neck_pan_deg, rom.pan_max) });
    }
    if legged {
        for s in 0..2 {
            let fold = leg_angles[s].2 + pose.waist_deg;
            if fold > TRUNK_THIGH_FOLD_MAX_DEG + 0.5 {
                out_of_range.push(format!("몸통이 허벅지에 닿음 (고관절 {:.0}° + 허리 {:.0}° > {:.0}°)", leg_angles[s].2, pose.waist_deg, TRUNK_THIGH_FOLD_MAX_DEG));
                break;
            }
        }
    }

    let pitch = if legged { 0.0 } else { pose.base_pitch_deg };
    let tilt = |p: V3| if pitch != 0.0 { pitch_about_origin(p, pitch) } else { p };
    if pitch != 0.0 {
        for m in masses.iter_mut() {
            m.p = tilt(m.p);
        }
        for b in boxes.iter_mut() {
            b.center = tilt(b.center);
            for a in b.axes.iter_mut() {
                *a = tilt(*a);
            }
        }
    }
    let hands = hands.map(tilt);
    let shoulders = shoulders.map(tilt);
    let elbows = elbows.map(tilt);
    let (waist, neck) = (tilt(waist), tilt(neck));
    let bumper = (!legged).then(|| tilt([0.5 * body.support_len_m + body.dims.wheel_r_m + body.dims.bumper_gap_m, 0.0, 0.5 * (body.dims.bumper_z_m[0] + body.dims.bumper_z_m[1])]));
    for b in &boxes {
        if matches!(b.segment, Segment::Hand | Segment::Foot) {
            continue;
        }
        let low = b.center[2] - (0..3).map(|i| b.axes[i][2].abs() * b.half[i]).sum::<f32>();
        if low < GROUND_CLEARANCE_M - 0.005 {
            out_of_range.push(format!("{} 땅에 닿음 ({:.2} m)", b.name, low));
            break;
        }
    }
    if !legged {
        boxes.extend(base_boxes(body, pitch));
    }

    let a = loads.accel;
    let g_eff: V3 = [-a[0], -a[1], -(G + a[2])];
    let mass: f32 = masses.iter().map(|m| m.m).sum();
    let mut com = [0.0f32; 3];
    for m in &masses {
        com = add(com, scale(m.p, m.m / mass.max(1e-6)));
    }
    let mut hand_f = [0, 1].map(|s| if s < n_arms { loads.hand_force[s] } else { [0.0; 3] });
    // A held load that slid off the palms: its weight acts where it went (forward, up), and a slide sideways between
    // two hands hangs more of it on the nearer one (the lever rule between the palms).
    let shift = tilt(loads.cargo_shift_m);
    let grips = hands.map(|h| add(h, [shift[0], 0.0, shift[2]]));
    if shift[1] != 0.0 && n_arms >= 2 {
        let total = add(hand_f[0], hand_f[1]);
        let span = (hands[1][1] - hands[0][1]).max(0.05);
        let left = ((0.5 * (hands[0][1] + hands[1][1]) + shift[1] - hands[0][1]) / span).clamp(0.0, 1.0);
        hand_f = [scale(total, 1.0 - left), scale(total, left)];
    }
    let mut s_force = scale(g_eff, mass);
    let mut m_o = [0.0f32; 3];
    for m in &masses {
        m_o = add(m_o, cross(m.p, scale(g_eff, m.m)));
    }
    for s in 0..2 {
        s_force = add(s_force, hand_f[s]);
        m_o = add(m_o, cross(grips[s], hand_f[s]));
    }
    if let Some(b) = bumper {
        s_force = add(s_force, loads.bumper_force);
        m_o = add(m_o, cross(b, loads.bumper_force));
    }
    if loads.body_force != [0.0; 3] {
        s_force = add(s_force, loads.body_force);
        m_o = add(m_o, cross(loads.body_point, loads.body_force));
    }
    let fz = s_force[2].min(-1e-6);
    let zmp = [-m_o[1] / fz, m_o[0] / fz];
    let grf = scale(s_force, -1.0);

    let moment = |j: V3, groups: &[Grp], hand_sides: &[usize]| -> V3 {
        let mut t = [0.0f32; 3];
        for m in masses.iter().filter(|m| groups.contains(&m.group)) {
            t = add(t, cross(sub(m.p, j), scale(g_eff, m.m)));
        }
        for &s in hand_sides {
            t = add(t, cross(sub(grips[s], j), hand_f[s]));
        }
        t
    };
    let jnt = |name: &str, seg: Segment, pos: V3, angle: Option<f32>, act: Option<&Actuator>, tau: f32, in_range: bool| -> KinJoint {
        match act {
            Some(a) => KinJoint { name: name.into(), segment: seg, pos, angle_deg: angle, actuator: a.name.into(), torque: tau, peak: a.peak_nm, cont: a.cont_nm, ratio: tau.abs() / a.peak_nm.max(1e-3), effort: tau.abs() / a.cont_nm.max(1e-3), in_range, structural: false },
            None => KinJoint { name: name.into(), segment: seg, pos, angle_deg: angle, actuator: "구조 (피치 관절 없음)".into(), torque: tau, peak: 0.0, cont: 0.0, ratio: 0.0, effort: 0.0, in_range, structural: true },
        }
    };
    let arm_groups = |s: usize| [Grp::Upper(s), Grp::Fore(s), Grp::Hand(s)];
    let head_grp: &[Grp] = if body.head_on_trunk() { &[Grp::Head] } else { &[] };
    let mut joints: Vec<KinJoint> = Vec::new();
    if torso == Torso::Humanoid {
        let above_waist: Vec<Grp> = head_grp.iter().copied().chain([Grp::Trunk, Grp::Upper(0), Grp::Fore(0), Grp::Hand(0), Grp::Upper(1), Grp::Fore(1), Grp::Hand(1)]).collect();
        joints.push(jnt("waist_pitch", Segment::Belly, waist, Some(pose.waist_deg), body.belly.pitch.as_ref(), moment(waist, &above_waist, &[0, 1])[1], pose.waist_deg.abs() <= rom.waist_max + 0.5));
    }
    if body.has_neck() {
        joints.push(jnt("neck", Segment::Head, neck, Some(pose.neck_deg), Some(&body.head.neck), moment(neck, &[Grp::Head], &[])[1], pose.neck_deg.abs() <= rom.neck_max + 0.5));
    }
    if let (false, Some(lift)) = (legged, body.lift.as_ref()) {
        let carried: Vec<Grp> = head_grp.iter().copied().chain([Grp::Trunk, Grp::Pelvis, Grp::Upper(0), Grp::Fore(0), Grp::Hand(0), Grp::Upper(1), Grp::Fore(1), Grp::Hand(1)]).collect();
        let f: f32 = -(masses.iter().filter(|m| carried.contains(&m.group)).map(|m| m.m * g_eff[2]).sum::<f32>() + hand_f[0][2] + hand_f[1][2]);
        let in_range = (-1e-3..=rom.lift_max_m + 1e-3).contains(&pose.lift_m);
        if !in_range {
            out_of_range.push(format!("torso_lift {:.2} m > {:.2} m", pose.lift_m, rom.lift_max_m));
        }
        let pos = match torso {
            Torso::Humanoid => hip_c,
            Torso::Mast => [0.0, pose.pelvis_y, body.dims.shoulder_m - pose.lift_m],
            Torso::None => [body.mast_camera_x(), 0.0, body.shoulder_at(0, pose.lift_m)[2]],
        };
        joints.push(KinJoint { name: "torso_lift".into(), segment: Segment::Leg, pos, angle_deg: None, actuator: lift.name.into(), torque: f, peak: lift.force_n, cont: lift.force_n, ratio: f.abs() / lift.force_n.max(1e-3), effort: f.abs() / lift.force_n.max(1e-3), in_range, structural: false });
    }
    for s in 0..n_arms {
        let abducted = pose.shoulder_abd_deg[s].abs() > 1e-6;
        let about = |m: V3| if abducted { dot(m, flex_axes[s]) } else { m[1] };
        let m_sh = moment(shoulders[s], &arm_groups(s), &[s]);
        joints.push(jnt(&format!("shoulder_{}", tag(s)), Segment::Arm, shoulders[s], Some(pose.shoulder_deg[s]), Some(&body.arm.shoulder), about(m_sh), true));
        joints.push(jnt(&format!("elbow_{}", tag(s)), Segment::Arm, elbows[s], Some(pose.elbow_deg[s]), Some(&body.arm.elbow), about(moment(elbows[s], &[Grp::Fore(s), Grp::Hand(s)], &[s])), true));
        if pose.shoulder_abd_deg[s].abs() > 0.5 {
            joints.push(jnt(&format!("shoulder_abd_{}", tag(s)), Segment::Arm, shoulders[s], Some(pose.shoulder_abd_deg[s]), Some(&body.arm.shoulder), dot(m_sh, trunk_fwd), (rom.shoulder_abd[0] - 0.5..=rom.shoulder_abd[1] + 0.5).contains(&pose.shoulder_abd_deg[s])));
        }
    }

    let mut contacts = Vec::new();
    let support;
    if legged {
        let fl = body.leg.foot_len_m;
        let fw = body.leg.foot_width_m;
        let planted: [f32; 2] = std::array::from_fn(|s| if stance[s] { pose.feet[s][0] } else { ankles[s][0] });
        let rects: Vec<[f32; 4]> = (0..2).map(|s| [planted[s] - 0.25 * fl, planted[s] + 0.75 * fl, ankles[s][1] - fw * 0.5, ankles[s][1] + fw * 0.5]).collect();
        let on: Vec<usize> = (0..2).filter(|s| stance[*s]).collect();
        let on = if on.is_empty() { vec![0] } else { on };
        let sole_model = draw.then(|| crate::contact::SoleModel::of(body));
        let rocker_x = |s: usize| -> Option<[f32; 2]> {
            (stance[s] && pose.toe_deg[s] > 1e-3).then(|| {
                let hinge = planted[s] + (crate::contact::TOE_HINGE_FL - crate::contact::ANKLE_FL) * fl;
                [hinge - (crate::contact::TOE_HINGE_FL - crate::contact::SOLE_BOUNDS_FL[2]) * fl, rects[s][1]]
            })
        };
        support = on.iter().fold([f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::NEG_INFINITY], |r, &s| {
            let x = rocker_x(s).unwrap_or([rects[s][0], rects[s][1]]);
            [r[0].min(x[0]), r[1].max(x[1]), r[2].min(rects[s][2]), r[3].max(rects[s][3])]
        });
        let shares: [f32; 2] = if on.len() == 2 {
            let r = ((ankles[1][1] - zmp[1]) / (ankles[1][1] - ankles[0][1]).max(1e-3)).clamp(0.0, 1.0);
            [r, 1.0 - r]
        } else {
            [if on[0] == 0 { 1.0 } else { 0.0 }, if on[0] == 1 { 1.0 } else { 0.0 }]
        };
        let above_hip = [Grp::Head, Grp::Trunk, Grp::Pelvis, Grp::Upper(0), Grp::Fore(0), Grp::Hand(0), Grp::Upper(1), Grp::Fore(1), Grp::Hand(1)];
        for s in 0..2 {
            let share = shares[s];
            let f = scale(grf, share);
            let rect = rects[s];
            let xr = rocker_x(s).unwrap_or([rect[0], rect[1]]);
            let cop = [zmp[0].clamp(xr[0], xr[1]), if on.len() == 2 { ankles[s][1] } else { zmp[1].clamp(rect[2], rect[3]) }];
            let sole = sole_model.as_ref().filter(|_| stance[s]).map(|m| crate::contact::sole_contact(m, planted[s], f[2].max(0.0), [f[0], f[1]], 0.0, cop[0], pose.toe_deg[s], 0.0));
            contacts.push(ContactPatch { name: format!("foot_{}", tag(s)), normal_n: if stance[s] { f[2].max(0.0) } else { 0.0 }, shear_n: if stance[s] { f[0] } else { 0.0 }, cop, rect, pressure_kpa: None, loaded_len_m: None, shear_y_n: if stance[s] { f[1] } else { 0.0 }, torsion_nm: 0.0, sole });
            let cop3 = [cop[0], cop[1], 0.0];
            let (ank_t, knee_t, hip_t, roll_t) = if stance[s] {
                let ankle_m = add(cross(sub(cop3, ankles[s]), f), moment(ankles[s], &[Grp::Foot(s)], &[]));
                let knee_m = add(cross(sub(cop3, knees[s]), f), moment(knees[s], &[Grp::Foot(s), Grp::Shank(s)], &[]));
                let hm = moment(hips[s], &above_hip, &[0, 1]);
                let roll = if on.len() == 1 {
                    hm[0]
                } else if wide {
                    add(cross(sub(cop3, hips[s]), f), moment(hips[s], &[Grp::Thigh(s), Grp::Shank(s), Grp::Foot(s)], &[]))[0]
                } else {
                    0.0
                };
                (ankle_m[1], knee_m[1], hm[1] * share, roll)
            } else {
                (0.0, moment(knees[s], &[Grp::Shank(s), Grp::Foot(s)], &[])[1], moment(hips[s], &[Grp::Thigh(s), Grp::Shank(s), Grp::Foot(s)], &[])[1], 0.0)
            };
            let (dorsi, kflex, hflex) = leg_angles[s];
            joints.push(jnt(&format!("hip_pitch_{}", tag(s)), Segment::Leg, hips[s], Some(hflex), Some(&body.leg.knee), hip_t, hflex <= rom.hip_flex_max + 0.5));
            joints.push(jnt(&format!("hip_roll_{}", tag(s)), Segment::Leg, hips[s], Some(hip_abd_deg[s]), Some(&body.leg.knee), roll_t, hip_abd_deg[s] <= HIP_ABD_MAX_DEG + 0.5));
            joints.push(jnt(&format!("knee_{}", tag(s)), Segment::Leg, knees[s], Some(kflex), Some(&body.leg.knee), knee_t, kflex <= rom.knee_max + 0.5));
            joints.push(jnt(&format!("ankle_{}", tag(s)), Segment::Foot, ankles[s], Some(dorsi), Some(&body.leg.ankle), ank_t, !stance[s] || dorsi <= rom.ankle_dorsi_max + 0.5));
        }
    } else {
        let len = body.support_len_m * pitch.to_radians().cos();
        let w = body.support_width_m;
        support = [-0.5 * len, 0.5 * len, -0.5 * w, 0.5 * w];
        let n = grf[2].max(0.0);
        let shear = grf[0];
        let f = body.chassis.drive_force_n;
        if body.base == BaseKind::Track {
            let e = zmp[0];
            let area = len * 2.0 * body.dims.belt_w_m;
            let p0 = n / area.max(1e-4) / 1000.0;
            let (rear, front, loaded) = if e.abs() <= len / 6.0 {
                (p0 * (1.0 - 6.0 * e / len), p0 * (1.0 + 6.0 * e / len), len)
            } else {
                let l = (3.0 * (0.5 * len - e.abs())).max(1e-3);
                let peak = 2.0 * n / (l * 2.0 * body.dims.belt_w_m) / 1000.0;
                if e > 0.0 { (0.0, peak, l) } else { (peak, 0.0, l) }
            };
            contacts.push(ContactPatch { name: "track".into(), normal_n: n, shear_n: shear, cop: zmp, rect: support, pressure_kpa: Some([rear, front]), loaded_len_m: Some(loaded), shear_y_n: grf[1], torsion_nm: 0.0, sole: None });
            joints.push(KinJoint { name: "drive".into(), segment: Segment::Foot, pos: tilt([0.0, 0.0, body.dims.wheel_r_m]), angle_deg: None, actuator: body.chassis.name.into(), torque: shear, peak: 2.0 * f, cont: f, ratio: shear.abs() / (2.0 * f).max(1e-3), effort: shear.abs() / f.max(1e-3), in_range: true, structural: false });
        } else {
            let (xr, xf) = (-0.5 * len, 0.5 * len);
            let nf = (n * (zmp[0] - xr) / (xf - xr)).clamp(0.0, n);
            for (name, jname, x, nn) in [("axle_front", "drive_front", xf, nf), ("axle_rear", "drive_rear", xr, n - nf)] {
                let sh = if n > 1e-6 { shear * nn / n } else { 0.0 };
                contacts.push(ContactPatch { name: name.into(), normal_n: nn, shear_n: sh, cop: [x, 0.0], rect: [x - 0.02, x + 0.02, -0.5 * w, 0.5 * w], pressure_kpa: None, loaded_len_m: None, shear_y_n: if n > 1e-6 { grf[1] * nn / n } else { 0.0 }, torsion_nm: 0.0, sole: None });
                let fa = 0.5 * f;
                joints.push(KinJoint { name: jname.into(), segment: Segment::Foot, pos: tilt([x, 0.0, body.dims.wheel_r_m]), angle_deg: None, actuator: body.chassis.name.into(), torque: sh, peak: 2.0 * fa, cont: fa, ratio: sh.abs() / (2.0 * fa).max(1e-3), effort: sh.abs() / fa.max(1e-3), in_range: true, structural: false });
            }
        }
    }

    let margin = support_margin(support, zmp);
    let mut feasible = out_of_range.is_empty();
    let mut limiting = out_of_range.first().map(|r| format!("관절 범위: {r}"));
    if let Some(w) = joints.iter().filter(|j| !j.structural).max_by(|a, b| a.ratio.total_cmp(&b.ratio)) {
        if w.ratio > 1.0 {
            feasible = false;
            limiting.get_or_insert_with(|| format!("{} {:.1}/{:.1} N·m ({})", w.name, w.torque.abs(), w.peak, w.actuator));
        }
    }
    if margin < 0.0 {
        feasible = false;
        limiting.get_or_insert_with(|| format!("균형 여유 {margin:.2} < 0"));
    }
    let effort_sum = joints.iter().filter(|j| !j.structural).map(|j| j.ratio * j.ratio).sum();
    if !draw {
        boxes.clear();
    }
    KinReport { joints, contacts, com, zmp, support, margin, grf, mass_kg: mass, hands, knees, bumper, feasible, limiting, out_of_range, effort_sum, pose: *pose, boxes }
}

const EYE3: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

fn oriented_box(name: &'static str, segment: Segment, center: V3, deg: f32, half: V3) -> LinkBox {
    let z = up_dir(deg);
    let x = [z[2], 0.0, -z[0]];
    LinkBox { name, segment, center, axes: [x, [0.0, 1.0, 0.0], z], half }
}

fn limb_box(name: &'static str, segment: Segment, a: V3, b: V3, w: f32) -> LinkBox {
    let d = sub(b, a);
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-6);
    let z = scale(d, 1.0 / len);
    let x0 = if z[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
    let c = cross(z, x0);
    let cl = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt().max(1e-6);
    let y = scale(c, 1.0 / cl);
    let x = cross(y, z);
    LinkBox { name, segment, center: scale(add(a, b), 0.5), axes: [x, y, z], half: [w, w, len * 0.5] }
}

fn hand_boxes(side: usize, wrist: V3, dir: V3, axis: V3, closure: f32, body: &Body) -> Vec<LinkBox> {
    let n = if body.gripper_only() { 2 } else { u8::from(body.fingers) as usize };
    let k = body.dims.k;
    let mut v = vec![limb_box(if side == 0 { "palm_r" } else { "palm_l" }, Segment::Hand, wrist, add(wrist, scale(dir, 0.10 * k)), 0.035 * k)];
    let palm_n = cross(dir, axis);
    let palm_n = if palm_n[0] < 0.0 { scale(palm_n, -1.0) } else { palm_n };
    let segs: &[f32] = if n == 2 { &[0.050, 0.035] } else { &[0.040, 0.025, 0.018] };
    let flex_max = [90.0f32, 100.0, 80.0];
    let base = add(wrist, scale(dir, 0.10 * k));
    for f in 0..n {
        let off = (-0.035 + 0.07 * f as f32 / (n.max(2) - 1) as f32) * k;
        let mut p = add(base, scale(axis, off));
        let mut ang = 0.0f32;
        for (i, len) in segs.iter().enumerate() {
            ang += closure.clamp(0.0, 1.0) * flex_max[i.min(2)];
            let a = ang.to_radians();
            let d = add(scale(dir, a.cos()), scale(palm_n, a.sin()));
            let q = add(p, scale(d, *len * k));
            v.push(limb_box(if side == 0 { "finger_r" } else { "finger_l" }, Segment::Hand, p, q, 0.009 * k));
            p = q;
        }
    }
    v
}

fn base_boxes(body: &Body, pitch_deg: f32) -> Vec<LinkBox> {
    let len = body.support_len_m;
    let w = body.support_width_m;
    let p = |q: V3| pitch_about_origin(q, pitch_deg);
    let ax = EYE3.map(|a| pitch_about_origin(a, pitch_deg));
    let mut v = vec![LinkBox { name: "chassis", segment: Segment::Foot, center: p([0.0, 0.0, body.dims.wheel_r_m + 0.05]), axes: ax, half: [len * 0.5, w * 0.5 - body.dims.belt_w_m, 0.05] }];
    let deck = body.dims.wheel_r_m + 0.1;
    match (body.morph.torso, body.camera_mount()) {
        (Torso::Humanoid, _) => v.push(LinkBox { name: "mast", segment: Segment::Leg, center: p([0.0, 0.0, 0.5 * (deck + body.dims.hip_m)]), axes: ax, half: [0.03, 0.03, 0.5 * (body.dims.hip_m - body.dims.wheel_r_m - 0.1)] }),
        (Torso::Mast, head) => {
            let top = column_top_m(&body.dims, head);
            v.push(LinkBox { name: "column", segment: Segment::Leg, center: p([0.0, 0.0, 0.5 * (deck + top)]), axes: ax, half: [0.04, 0.04, 0.5 * (top - deck)] });
        }
        (Torso::None, HeadMount::MastCamera { height_mm, .. }) => {
            let top = height_mm as f32 / 1000.0 - body.dims.camera_up_m;
            v.push(LinkBox { name: "pole", segment: Segment::Leg, center: p([body.mast_camera_x(), 0.0, 0.5 * (deck + top)]), axes: ax, half: [0.01, 0.01, 0.5 * (top - deck).max(0.0)] });
        }
        _ => {}
    }
    v.push(LinkBox { name: "bumper", segment: Segment::Foot, center: p([0.5 * len + body.dims.wheel_r_m + body.dims.bumper_gap_m - 0.01, 0.0, 0.5 * (body.dims.bumper_z_m[0] + body.dims.bumper_z_m[1])]), axes: ax, half: [0.01, w * 0.5, 0.5 * (body.dims.bumper_z_m[1] - body.dims.bumper_z_m[0])] });
    for b in body.deck_boxes() {
        v.push(LinkBox { center: p(b.center), axes: ax, ..b });
    }
    if body.base == BaseKind::Track {
        for y in [-0.5 * w + 0.5 * body.dims.belt_w_m, 0.5 * w - 0.5 * body.dims.belt_w_m] {
            v.push(LinkBox { name: "track", segment: Segment::Foot, center: p([0.0, y, body.dims.wheel_r_m]), axes: ax, half: [len * 0.5 + body.dims.wheel_r_m, body.dims.belt_w_m * 0.5, body.dims.wheel_r_m] });
        }
    } else {
        for x in [-0.5 * len, 0.5 * len] {
            for y in [-0.5 * w + 0.5 * body.dims.belt_w_m, 0.5 * w - 0.5 * body.dims.belt_w_m] {
                v.push(LinkBox { name: "wheel", segment: Segment::Foot, center: p([x, y, body.dims.wheel_r_m]), axes: ax, half: [body.dims.wheel_r_m, body.dims.belt_w_m * 0.5, body.dims.wheel_r_m] });
            }
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hand::FingerCount;

    fn body(tier: Tier, base: BaseKind) -> Body {
        Body::new(&BodyPlan::uniform(tier), base, FingerCount::Five)
    }

    #[test]
    fn standing_matches_the_balance_model() {
        for base in [BaseKind::HumanFoot, BaseKind::Wheel] {
            let b = body(Tier::Mid, base);
            let r = solve(&b, &KinPose::stand_compact(0.0), &KinLoads::default());
            assert!((r.mass_kg - b.mass_kg()).abs() < 1e-3, "{base:?}: {} vs {}", r.mass_kg, b.mass_kg());
            let q = b.solve(&crate::Posture::quiet(0.0), &crate::FeedForward { enabled: false, gain: 1.0 }, 1e3);
            assert!((r.com[2] - q.com[2]).abs() < 0.03, "{base:?} CoM height {} vs {}", r.com[2], q.com[2]);
            assert!((r.grf[2] - r.mass_kg * G).abs() < 1e-2);
            assert!(r.feasible && r.margin > 0.3, "{base:?}: {:?} margin {}", r.limiting, r.margin);
        }
    }

    #[test]
    fn the_head_camera_follows_the_bend_the_neck_and_the_slope() {
        let b = body(Tier::Pro, BaseKind::HumanFoot);
        let stand = head_camera(&b, &KinPose::stand_compact(0.0));
        assert!((stand.pos[2] - 1.14).abs() < 0.01 && stand.pitch_down_deg.abs() < 1e-4, "{stand:?}");
        let bent = head_camera(&b, &KinPose { waist_deg: 60.0, neck_deg: 30.0, ..KinPose::stand_compact(0.0) });
        assert!(bent.pos[2] < stand.pos[2] - 0.15 && bent.pos[0] > stand.pos[0] + 0.25, "{bent:?}");
        assert!((bent.pitch_down_deg - 90.0).abs() < 1e-3);
        let left = head_camera(&b, &KinPose { neck_pan_deg: 60.0, ..KinPose::stand_compact(0.0) });
        assert!(left.pos[1] > 0.05 && left.pan_deg == 60.0, "{left:?}");
        let w = body(Tier::Mid, BaseKind::Wheel);
        let up = head_camera(&w, &KinPose { base_pitch_deg: 10.0, ..KinPose::stand_compact(0.0) });
        assert!((up.pitch_down_deg + 10.0).abs() < 1e-3);
    }

    #[test]
    fn the_knee_follows_the_hip_and_ankle() {
        let (k, flex) = knee_ik(&Dims::COMPACT, [0.0, Dims::COMPACT.hip_m], [0.0, crate::dims::Dims::COMPACT.ankle_m]).unwrap();
        assert!(k[0].abs() < 1e-4 && flex.abs() < 0.1, "straight leg: {k:?} {flex}");
        let (k, flex) = knee_ik(&Dims::COMPACT, [-0.05, Dims::COMPACT.hip_m - 0.25], [0.0, crate::dims::Dims::COMPACT.ankle_m]).unwrap();
        assert!(k[0] > 0.0 && flex > 60.0, "{k:?} {flex}");
        assert!(knee_ik(&Dims::COMPACT, [0.0, Dims::COMPACT.hip_m + 0.05], [0.0, crate::dims::Dims::COMPACT.ankle_m]).is_none(), "beyond the leg's length");
    }

    #[test]
    fn a_load_in_the_hand_loads_the_shoulder_elbow_and_moves_the_zmp() {
        let b = body(Tier::Mid, BaseKind::HumanFoot);
        let mut p = KinPose::stand_compact(0.0);
        p.shoulder_deg = [30.0, 0.0];
        p.elbow_deg = [60.0, 10.0];
        let empty = solve(&b, &p, &KinLoads::default());
        let loaded = solve(&b, &p, &KinLoads::holding(2.4, false, 0.0));
        let sh = |r: &KinReport| r.joint("shoulder_r").unwrap().torque.abs();
        let el = |r: &KinReport| r.joint("elbow_r").unwrap().torque.abs();
        assert!(sh(&loaded) > sh(&empty) + 2.4 * G * 0.2, "{} vs {}", sh(&loaded), sh(&empty));
        assert!(el(&loaded) > el(&empty));
        assert!(loaded.zmp[0] > empty.zmp[0], "the load in front moves the ZMP forward");
        let bal = b.solve(&crate::Posture::arm_lift(2.4, false, 0.0), &crate::FeedForward { enabled: false, gain: 1.0 }, 1e3);
        let bs = bal.joint("shoulder_r").unwrap().torque_nm;
        assert!((sh(&loaded) - bs).abs() / bs < 0.10, "kin {} vs balance {bs}", sh(&loaded));
    }

    #[test]
    fn a_rigid_trunk_cannot_pitch_and_a_squat_bends_the_knees() {
        let mid = body(Tier::Mid, BaseKind::Wheel);
        let mut p = KinPose::stand_compact(0.0);
        p.waist_deg = 30.0;
        let r = solve(&mid, &p, &KinLoads::default());
        assert!(!r.feasible && r.limiting.as_deref().unwrap_or("").contains("허리"), "{:?}", r.limiting);
        let pro = body(Tier::Pro, BaseKind::Wheel);
        let r = solve(&pro, &p, &KinLoads::default());
        assert!(r.feasible, "{:?}", r.limiting);
        assert!(r.joint("waist_pitch").unwrap().torque.abs() > 5.0);
        let legs = body(Tier::Mid, BaseKind::HumanFoot);
        let stand = solve(&legs, &KinPose::stand_compact(0.0), &KinLoads::default());
        let mut sq = KinPose::stand_compact(0.0);
        sq.hip = [-0.08, crate::dims::Dims::COMPACT.hip_m - 0.2];
        sq.pelvis_deg = 30.0;
        let r = solve(&legs, &sq, &KinLoads::default());
        fn k(r: &KinReport) -> &KinJoint {
            r.joint("knee_r").unwrap()
        }
        assert!(k(&r).angle_deg.unwrap() > 50.0 && k(&r).torque.abs() > k(&stand).torque.abs(), "{:?}", k(&r));
    }

    #[test]
    fn wheel_and_track_loads_follow_the_zmp() {
        let b = body(Tier::Mid, BaseKind::Wheel);
        let mut p = KinPose::stand_compact(0.0);
        p.shoulder_deg = [80.0, 80.0];
        p.elbow_deg = [0.0, 0.0];
        let r = solve(&b, &p, &KinLoads::holding(2.0, true, 0.0));
        let (f, rr) = (r.contact("axle_front").unwrap(), r.contact("axle_rear").unwrap());
        assert!(f.normal_n > rr.normal_n, "arms forward load the front axle: {} vs {}", f.normal_n, rr.normal_n);
        assert!((f.normal_n + rr.normal_n - r.grf[2]).abs() < 1e-2);
        let t = body(Tier::Mid, BaseKind::Track);
        let r = solve(&t, &p, &KinLoads::holding(2.0, true, 0.0));
        let tr = r.contact("track").unwrap();
        let pk = tr.pressure_kpa.unwrap();
        assert!(pk[1] > pk[0], "the front of the track carries more: {pk:?}");
        let push = solve(&b, &KinPose::stand_compact(0.0), &KinLoads { bumper_force: [-30.0, 0.0, 0.0], ..KinLoads::default() });
        assert!((push.grf[0] - 30.0).abs() < 1e-3, "{:?}", push.grf);
    }
}
