use super::*;
use noto_core::act::focus::{self, FocusCtx, FocusFrame};
use noto_core::act::material::{self, ItemMaterial};
use noto_core::act::reach::ReachMode;
use noto_core::act::{fork, gait, grasp as agrasp, push as apush, ActEvent, ActKind, ActRollout};
use noto_core::kin::{self as hkin, KinLoads, KinPose};

pub const GET_UP_S: f32 = 6.0;
pub const PUSH_START_SLACK_M: f32 = 0.40;
pub const SHUFFLE_MPS: f32 = 0.2;
pub const SIDE_SHUFFLE_MAX_M: f32 = 0.5;
pub const PILE_PIECE_H_M: f32 = 0.12;
/// How long a slipping foot takes to catch (s) [추정: about a reaction].
const REACTION_JOLT_S: f32 = 0.1;

#[derive(Clone, Debug)]
pub struct ActPlay {
    pub rollout: ActRollout,
    pub start_s: f64,
    pub elapsed_s: f32,
    pub played_s: f32,
    pub object: Option<(ItemKind, f32)>,
    pub origin: (f32, f32, f32),
    /// A fork act drives the base: it turns to `yaw_to` over `turn_s`, then keeps the item where the act's sample
    /// says it is in the body frame, by moving along `yaw_to` from where it stood.
    pub drive: Option<ActDrive>,
    /// The drop the act is lifting off the ground (hidden from the scene while it rides the tines).
    pub lifting: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActDrive {
    pub from: (f32, f32, f32),
    pub yaw_to: f32,
    pub turn_s: f32,
    pub object_x0: f32,
}

/// How fast the base turns to face an item before the tines go to it at most (rad/s) [추정]; slower where the ground
/// and motors give a slower pivot or none (`World::fork_align_rad_s`).
pub const FORK_ALIGN_RAD_S: f32 = 1.2;

impl ActPlay {
    fn new(rollout: ActRollout, start_s: f64, object: Option<(ItemKind, f32)>, origin: (f32, f32, f32)) -> ActPlay {
        ActPlay { rollout, start_s, elapsed_s: 0.0, played_s: -1.0, object, origin, drive: None, lifting: None }
    }

    pub fn t(&self) -> f32 {
        self.elapsed_s.clamp(0.0, self.rollout.duration_s)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GazeHold {
    pub pose: KinPose,
    pub x: f32,
    pub z: f32,
    pub since_s: f64,
}

#[derive(Clone, Debug, Default)]
pub struct WalkState {
    pub since_step_m: f32,
    pub steps: u32,
    pub stance: usize,
    pub slips: u32,
    pub plan: Option<(gait::GaitPlan, Ground, i32)>,
    pub last_mu: Option<f32>,
    /// The step it is going over (m): the swinging foot clears that.
    pub step_lift_m: f32,
}

#[derive(Clone, Debug)]
pub struct PushState {
    pub drop: u32,
    pub path: Vec<(f32, f32)>,
    pub length_m: f32,
    pub done_m: f32,
    pub end_m: f32,
    pub lead_m: f32,
}

pub fn material_of(kind: ItemKind) -> ItemMaterial {
    match kind {
        ItemKind::Log => ItemMaterial::Wood,
        ItemKind::Stone => ItemMaterial::Stone,
        ItemKind::Coal => ItemMaterial::Coal,
        ItemKind::IronOre => ItemMaterial::IronOre,
        ItemKind::Parcel => ItemMaterial::Cardboard,
    }
}

pub fn camera_of(sensor: &str) -> (f32, f32, f32) {
    if sensor.contains("OAK") {
        (432.0, 0.075, 0.08)
    } else {
        (424.0, 0.050, 0.08)
    }
}

fn arm_error_m(arm: &noto_core::parts::ArmPreset) -> f32 {
    if arm.shoulder.backdrivable {
        0.005
    } else {
        0.010
    }
}

impl World {
    pub fn aim_sigma_m(&self) -> f32 {
        let (f, b, sub) = camera_of(self.body.head.sensor);
        let range = (SHOULDER_RANGE_M * SHOULDER_RANGE_M + 0.3f32 * 0.3).sqrt();
        let depth = material::stereo_depth_sigma(range, f, b, sub);
        (depth * depth + arm_error_m(&self.body.arm).powi(2)).sqrt()
    }

    pub fn pad_mu(&self, kind: ItemKind) -> f32 {
        material::pad_on(material_of(kind)).mean
    }

    pub(super) fn start_grasp(&mut self, drop: u32, two_arms: bool, aim: Option<(f32, f32)>, side: bool, memory_sigma_m: f32) -> Result<(f32, Option<(f32, f32)>), ActionError> {
        let d = self.drop_item(drop).cloned().ok_or(ActionError::UnknownTarget)?;
        let spec = d.kind.spec();
        let size = spec.piece_size_m(d.amount);
        let seed = self.act_rng.next_u64();
        let ground = self.ground_at(self.terrain.cell_of(d.x, d.z).unwrap_or((0, 0))).surface();
        let at = aim.map(|(x, z)| self.true_of(x, z)).unwrap_or((d.x, d.z));
        let heading = if side { self.agent.yaw } else { (at.1 - self.agent.z).atan2(at.0 - self.agent.x) };
        let (sh, ch) = heading.sin_cos();
        let (ox, oz) = (at.0 - d.x, at.1 - d.z);
        let offset = [ox * ch + oz * sh, ox * sh - oz * ch];
        let mode = if side { ReachMode::Side } else { ReachMode::Front };
        let lead = self.gaze.take().filter(|g| !side && (g.x - at.0).hypot(g.z - at.1) < 0.5).map(|g| g.pose);
        let inp = agrasp::GraspInput {
            body: &self.body,
            hand: &self.hand,
            material: material_of(d.kind),
            mass_kg: d.mass_kg(),
            size_m: size,
            ground,
            two_arms,
            cargo_kg: self.carried_kg(),
            aim_sigma_m: self.aim_sigma_m(),
            seed,
            mode,
            aim_offset_m: offset,
            memory_sigma_m,
            look_first: lead.is_none(),
            lead,
        };
        let mut rollout = agrasp::grasp(&inp);
        if matches!(rollout.outcome.code, Some("reach_ground") | Some("reach_side")) {
            let need = size[1] + noto_core::act::reach::PALM_CLEARANCE_M;
            return Err(ActionError::CannotReach { lowest_m: rollout.metric("lowest_palm_z").unwrap_or(0.0), need_m: need, reason: rollout.outcome.reason_ko.clone().unwrap_or_default() });
        }
        let planned = rollout.samples.first().and_then(|s| s.object).map_or([0.3, 0.0, 0.0], |o| o);
        let (step_x, step_z, step) = if side {
            let (rx, rz) = (at.0 - self.agent.x, at.1 - self.agent.z);
            let (fwd, left) = (rx * ch + rz * sh, rx * sh - rz * ch);
            let (df, dl) = (fwd - planned[0], left - planned[1]);
            let n = df.hypot(dl);
            if n > SIDE_SHUFFLE_MAX_M {
                return Err(ActionError::OutOfReach { distance_m: n, reach_m: SIDE_SHUFFLE_MAX_M });
            }
            (df * ch + dl * sh, df * sh - dl * ch, n)
        } else {
            let dist = (at.0 - self.agent.x).hypot(at.1 - self.agent.z);
            let k = (dist - planned[0]) / dist.max(1e-3);
            ((at.0 - self.agent.x) * k, (at.1 - self.agent.z) * k, (dist - planned[0]).abs())
        };
        let shuffle = step / SHUFFLE_MPS;
        if shuffle > 0.05 && !(self.can_step_to(self.agent.x + step_x, self.agent.z + step_z) && self.pose_check().clear(self.agent.x + step_x, self.agent.z + step_z, self.agent.yaw)) {
            let dist = (at.0 - self.agent.x).hypot(at.1 - self.agent.z);
            return Err(ActionError::OutOfReach { distance_m: dist, reach_m: (dist - step).max(0.0) });
        }
        if shuffle > 0.05 {
            for s in rollout.samples.iter_mut() {
                s.t += shuffle;
            }
            for e in rollout.events.iter_mut() {
                e.t += shuffle;
            }
            rollout.duration_s += shuffle;
            rollout.metrics.push(("shuffle_s", shuffle));
            self.agent.x += step_x;
            self.agent.z += step_z;
        }
        let dur = rollout.duration_s;
        let mut play = ActPlay::new(rollout, self.time_s, Some((d.kind, d.amount)), (self.agent.x, self.agent.z, heading));
        play.lifting = Some(drop);
        self.act = Some(play);
        Ok((dur, (!side).then_some(at)))
    }

    /// How fast the base turns to the item inside a fork act: as an ordinary turn here does (the skid-steer pivot the
    /// ground and the motors allow, a creep where it cannot pivot), never faster than `FORK_ALIGN_RAD_S`. At a fixed
    /// rate a track on soft ground spun on the spot inside the act, which a turn outside it could not do.
    pub fn fork_align_rad_s(&self) -> f32 {
        let caps = self.turn_caps();
        if caps.can_pivot() {
            caps.pivot_rad_s.min(FORK_ALIGN_RAD_S)
        } else {
            motion::CREEP_RAD_S
        }
    }

    /// A look at the ground point (`x`, `z`) — at the middle of an item lying there, if there is one.
    pub(super) fn start_look(&mut self, x: f32, z: f32) -> f32 {
        let (dx, dz) = (x - self.agent.x, z - self.agent.z);
        let heading = dz.atan2(dx);
        let here = self.agent_ground_m();
        let h = self
            .drops
            .iter()
            .filter(|d| (d.x - x).hypot(d.z - z) < 0.35)
            .min_by(|a, b| (a.x - x).hypot(a.z - z).total_cmp(&(b.x - x).hypot(b.z - z)))
            .map_or(0.05, |d| self.drop_base_m(d) - here + 0.5 * d.kind.spec().piece_size_m(d.amount)[1]);
        if fork::applies(&self.body) {
            let size = self.drops.iter().filter(|d| (d.x - x).hypot(d.z - z) < 0.35).map(|d| d.kind.spec().piece_size_m(d.amount)).next().unwrap_or([0.1; 3]);
            let base = (h - 0.5 * size[1]).max(0.0);
            let turn_s = wrap_angle(heading - self.agent.yaw).abs() / self.fork_align_rad_s();
            let r = fork::look(&self.body, self.fork_m_now(), dx.hypot(dz), size, base, self.carried_kg(), turn_s);
            let dur = r.duration_s;
            let mut play = ActPlay::new(r, self.time_s, None, (self.agent.x, self.agent.z, heading));
            play.drive = Some(ActDrive { from: (self.agent.x, self.agent.z, self.agent.yaw), yaw_to: heading, turn_s, object_x0: dx.hypot(dz) });
            self.act = Some(play);
            return dur;
        }
        let from = self.body.stand(self.carried_kg());
        let r = noto_core::act::gaze::look(&self.body, &from, [dx.hypot(dz), 0.0, h]);
        let dur = r.duration_s;
        self.act = Some(ActPlay::new(r, self.time_s, None, (self.agent.x, self.agent.z, heading)));
        dur
    }

    /// Tine height now (m above the ground under the base): from the fork act playing, the tines held down after a
    /// look, or travel height (loaded or empty).
    pub fn fork_m_now(&self) -> f32 {
        if let Some(p) = self.act.as_ref().filter(|p| p.rollout.kind == ActKind::Fork) {
            if let Some(s) = p.rollout.sample_at(p.t()) {
                return s.pose.fork_m;
            }
        }
        if let Some(g) = self.gaze.filter(|_| fork::applies(&self.body)) {
            return g.pose.fork_m;
        }
        // On the move the tines ride at travel height, over curbs, whether they carry anything or not.
        if self.carried_kg() > AMOUNT_EPS || matches!(self.active, Some((_, Active::Move(_)))) {
            self.body.dims.fork_carry_m
        } else {
            self.body.dims.fork_rest_m
        }
    }

    /// The drop an act is lifting off the ground right now: on the tines while the mast raises it, or in the hands
    /// from lift-off until it goes into the bay.
    pub fn lifted_drop(&self) -> Option<u32> {
        self.act
            .as_ref()
            .filter(|p| p.rollout.sample_at(p.t()).is_some_and(|s| matches!((p.rollout.kind, s.phase), (ActKind::Fork, "raise") | (ActKind::Grasp, "lift" | "stow"))))
            .and_then(|p| p.lifting)
    }

    /// What the hands hold off the ground now, as (kind, piece size, centre in the body frame): the item from
    /// lift-off until it goes into the bay, where the eyes follow it.
    pub fn hand_load(&self) -> Option<(ItemKind, [f32; 3], [f32; 3])> {
        if let Some(p) = self.act.as_ref().filter(|p| p.rollout.kind == ActKind::Grasp && p.lifting.is_some()) {
            if let (Some(s), Some((k, amount))) = (p.rollout.sample_at(p.t()).filter(|s| matches!(s.phase, "lift" | "stow")), p.object) {
                return Some((k, k.spec().piece_size_m(amount), s.object?));
            }
        }
        self.carried_load()
    }

    /// What a walker carries in its two hands, as `hand_load`: on the middle of the hands, where the ride has moved it.
    pub fn carried_load(&self) -> Option<(ItemKind, [f32; 3], [f32; 3])> {
        if !self.body.legged() {
            return None;
        }
        let (k, amount) = *self.agent.carried.iter().rev().find(|(_, n)| *n > AMOUNT_EPS)?;
        let size = k.spec().piece_size_m(amount.min(1.0));
        let (pose, loads, _) = self.act_pose()?;
        let r = hkin::solve_fast(&self.body, &pose, &loads);
        let (a, b) = (r.hands[0], r.hands[1]);
        let s = self.ride.state.shift;
        Some((k, size, [0.5 * (a[0] + b[0]) + s[0] + 0.5 * size[2], 0.5 * (a[1] + b[1]) + s[1], 0.5 * (a[2] + b[2]) + 0.5 * size[1]]))
    }

    /// What rides on the tines now, as (kind, piece size, centre in the body frame), bottom first.
    pub fn fork_load(&self) -> Vec<(ItemKind, [f32; 3], [f32; 3])> {
        if !fork::applies(&self.body) {
            return Vec::new();
        }
        let play = self.act.as_ref().filter(|p| p.rollout.kind == ActKind::Fork);
        let sample = play.and_then(|p| p.rollout.sample_at(p.t()));
        let mut out = Vec::new();
        let mut z = self.fork_m_now() + self.body.dims.tine_t_m;
        match (play.and_then(|p| p.object), sample.and_then(|s| s.object), sample.map(|s| s.phase)) {
            (Some((k, a)), Some(o), Some("raise")) => {
                let size = k.spec().piece_size_m(a);
                out.push((k, size, o));
                z = o[2] + 0.5 * size[1];
            }
            (Some(_), Some(o), Some("lower_load" | "back_out" | "rest")) => {
                let mut zz = o[2];
                for (k, n) in self.agent.carried.iter().filter(|(_, n)| *n > AMOUNT_EPS) {
                    let size = k.spec().piece_size_m(n.min(1.0));
                    out.push((*k, size, [o[0], 0.0, zz]));
                    zz += size[1];
                }
                return out;
            }
            _ => {}
        }
        for (k, n) in self.agent.carried.iter().filter(|(_, n)| *n > AMOUNT_EPS) {
            let mut rest = *n;
            while rest > AMOUNT_EPS && out.len() < 6 {
                let a = rest.min(1.0);
                let size = k.spec().piece_size_m(a);
                let s = self.ride.state.shift;
                out.push((*k, size, [fork::load_x(&self.body, size[0].max(size[2])) + s[0], s[1], z + 0.5 * size[1]]));
                z += size[1];
                rest -= a;
            }
        }
        out
    }

    /// Picks the drop up on the tines (wheeled and tracked bases). Returns the act's length.
    pub(super) fn start_fork_pick(&mut self, drop: u32) -> Result<f32, ActionError> {
        let d = self.drop_item(drop).cloned().ok_or(ActionError::UnknownTarget)?;
        let size = d.kind.spec().piece_size_m(d.amount);
        let (dx, dz) = (d.x - self.agent.x, d.z - self.agent.z);
        let dist = dx.hypot(dz);
        let heading = dz.atan2(dx);
        let base = (self.drop_base_m(&d) - self.agent_ground_m()).max(0.0);
        let turn_s = wrap_angle(heading - self.agent.yaw).abs() / self.fork_align_rad_s();
        let r = fork::pick(&self.body, self.fork_m_now(), dist, d.mass_kg(), size, base, self.carried_kg(), turn_s);
        if !r.outcome.ok {
            let why = r.outcome.reason_ko.clone().unwrap_or_default();
            return Err(ActionError::CannotReach { lowest_m: base, need_m: base, reason: why });
        }
        let dur = r.duration_s;
        self.agent.grip_margin = r.metric("tip_ratio").map(|t| ((t - 1.0) / t.max(1e-3)).clamp(0.0, 1.0));
        let mut play = ActPlay::new(r, self.time_s, Some((d.kind, d.amount)), (self.agent.x, self.agent.z, heading));
        play.drive = Some(ActDrive { from: (self.agent.x, self.agent.z, self.agent.yaw), yaw_to: heading, turn_s, object_x0: dist });
        play.lifting = Some(drop);
        self.act = Some(play);
        // The pick starts from the look's lowered tines; the look is over.
        self.gaze = None;
        Ok(dur)
    }

    /// Whether the tines can lift the item here (wheeled and tracked bases), with what they carry already.
    pub fn fork_check(&self, kind: ItemKind, amount: f32) -> Result<fork::ForkFit, String> {
        fork::check(&self.body, kind.spec().mass_kg * amount, kind.spec().piece_size_m(amount), 0.0, self.carried_kg())
    }

    /// Keeps the head where the look left it until the base moves again (or a pick-up starts from it).
    pub(super) fn hold_gaze(&mut self, play: Option<&ActPlay>, at: (f32, f32)) {
        self.gaze = play.filter(|p| p.rollout.outcome.ok).and_then(|p| p.rollout.samples.last()).map(|s| GazeHold { pose: s.pose, x: at.0, z: at.1, since_s: self.time_s });
    }

    /// The phase of what the body is doing now: an act's phase, or "hold" while the gaze is held after a look.
    pub fn act_phase(&self) -> Option<(ActKind, &'static str)> {
        if let Some(p) = &self.act {
            return p.rollout.sample_at(p.t()).map(|s| (p.rollout.kind, s.phase));
        }
        self.gaze.map(|_| (ActKind::Look, "hold"))
    }

    pub(super) fn can_step_to(&self, x: f32, z: f32) -> bool {
        let Some(to) = self.terrain.cell_of(x, z) else { return false };
        let from = self.agent_cell();
        if to == from {
            return true;
        }
        let near = (to.0 as i32 - from.0 as i32).abs() <= 1 && (to.1 as i32 - from.1 as i32).abs() <= 1;
        near && !self.blocked_cells().contains(&to) && self.plan_route_avoiding(Goal::Cell(to), &HashSet::new()).is_some_and(|p| p.edges.len() == 1)
    }

    pub(super) fn start_release(&mut self, pieces: &[(ItemKind, f32)], waypoint: Option<u32>) -> f32 {
        if fork::applies(&self.body) && !pieces.is_empty() {
            let pile_n = waypoint.and_then(|w| self.waypoint(w)).map_or(0usize, |w| render::stock_pieces(&w.stock).len());
            let surface = if waypoint.is_some() { (pile_n / 4) as f32 * PILE_PIECE_H_M } else { 0.0 };
            let mass: f32 = pieces.iter().map(|(k, a)| k.spec().mass_kg * a).sum();
            let (k0, a0) = pieces[0];
            let size = k0.spec().piece_size_m(a0.min(1.0));
            let mut r = fork::set_down(&self.body, self.fork_m_now(), mass, size, surface, self.carried_kg());
            let lx = fork::load_x(&self.body, size[0].max(size[2]));
            for i in 0..pieces.len() {
                r.metrics.push(("land_x", lx));
                r.metrics.push(("land_y", (i as f32 - (pieces.len() as f32 - 1.0) * 0.5) * 0.13));
            }
            let dur = r.duration_s;
            let mut play = ActPlay::new(r, self.time_s, Some((k0, a0)), (self.agent.x, self.agent.z, self.agent.yaw));
            play.drive = Some(ActDrive { from: (self.agent.x, self.agent.z, self.agent.yaw), yaw_to: self.agent.yaw, turn_s: 0.0, object_x0: lx });
            self.act = Some(play);
            return dur;
        }
        let mut all: Option<ActRollout> = None;
        let mut cargo = self.carried_kg();
        let pile_n = waypoint.and_then(|w| self.waypoint(w)).map_or(0usize, |w| render::stock_pieces(&w.stock).len());
        let ground = match waypoint.and_then(|w| self.waypoint(w)) {
            Some(w) => self.ground_at(self.terrain.cell_of(w.x, w.z).unwrap_or((0, 0))).surface(),
            None => self.ground_here().surface(),
        };
        for (i, (k, a)) in pieces.iter().enumerate() {
            let mass = k.spec().mass_kg * a;
            cargo = (cargo - mass).max(0.0);
            let surface_z = if waypoint.is_some() { ((pile_n + i) / 4) as f32 * PILE_PIECE_H_M } else { 0.0 };
            let inp = agrasp::ReleaseInput { body: &self.body, material: material_of(*k), mass_kg: mass, size_m: k.spec().piece_size_m(*a), ground, surface_z_m: surface_z, cargo_kg: cargo, aim_sigma_m: self.aim_sigma_m(), seed: self.act_rng.next_u64() };
            let r = agrasp::release(&inp);
            all = Some(match all {
                None => r,
                Some(mut acc) => {
                    let t0 = acc.duration_s;
                    acc.samples.extend(r.samples.into_iter().map(|mut s| {
                        s.t += t0;
                        s
                    }));
                    acc.events.extend(r.events.into_iter().map(|mut e| {
                        e.t += t0;
                        e
                    }));
                    acc.duration_s += r.duration_s;
                    acc.metrics.extend(r.metrics);
                    acc
                }
            });
        }
        let Some(r) = all else { return DROP_TIME_S };
        let dur = r.duration_s;
        let first = pieces.first().copied();
        self.act = Some(ActPlay::new(r, self.time_s, first, (self.agent.x, self.agent.z, self.agent.yaw)));
        dur
    }

    pub(super) fn act_advance(&mut self, dt: f32, events: &mut Vec<TimedEvent>) {
        let Some(p) = self.act.as_mut() else { return };
        p.elapsed_s += dt;
        let t = p.t();
        let out: Vec<WorldEvent> = p.rollout.events_between(p.played_s, t).map(|e| act_event(p.rollout.kind, e)).collect();
        p.played_s = t;
        let to = p.drive.and_then(|d| {
            let ox = p.rollout.sample_at(t)?.object?[0];
            let u = if d.turn_s > 0.0 { (t / d.turn_s).min(1.0) } else { 1.0 };
            let along = d.object_x0 - ox;
            Some((d.from.0 + along * d.yaw_to.cos(), d.from.1 + along * d.yaw_to.sin(), d.from.2 + wrap_angle(d.yaw_to - d.from.2) * u))
        });
        for ev in out {
            events.push(self.stamp(ev));
        }
        // The odometry takes this motion with the rest of the step (`World::step`), once. The base drives itself here
        // too, and what stands in its way stops it here as anywhere.
        if let Some(to) = to.filter(|to| self.pose_check().clear(to.0, to.1, to.2)) {
            (self.agent.x, self.agent.z, self.agent.yaw) = (to.0, to.1, wrap_angle(to.2));
            self.shove_loose();
        }
    }

    pub(super) fn act_finish(&mut self, events: &mut Vec<TimedEvent>) {
        let Some(p) = self.act.as_mut() else { return };
        let end = p.rollout.duration_s;
        let out: Vec<WorldEvent> = p.rollout.events_between(p.played_s, end).map(|e| act_event(p.rollout.kind, e)).collect();
        p.elapsed_s = p.elapsed_s.max(end);
        p.played_s = end;
        for ev in out {
            events.push(self.stamp(ev));
        }
    }

    pub(super) fn end_act(&mut self) -> Option<ActPlay> {
        let p = self.act.take()?;
        self.last_act = Some(p.clone());
        Some(p)
    }

    pub(super) fn grasp_failed(&mut self, drop: u32, p: &ActPlay) {
        let h = p.rollout.metric("drop_height_m").unwrap_or(0.0);
        if h <= 0.0 {
            return;
        }
        let Some(kind) = p.object.map(|o| o.0) else { return };
        let ground = self.ground_here().surface();
        let e = material::restitution(material_of(kind), ground);
        let theta = self.act_rng.next_f32() * 30f32.to_radians();
        let dir = self.act_rng.next_f32() * std::f32::consts::TAU;
        let r = 2.0 * e * e * h * (2.0 * theta).sin();
        if let Some(d) = self.drops.iter_mut().find(|d| d.id == drop) {
            d.x += r * dir.cos();
            d.z += r * dir.sin();
        }
        if let Some(i) = self.drops.iter().position(|d| d.id == drop) {
            let d = &self.drops[i];
            if !d.support.is_ground() && (self.surface_m(d.x, d.z) - d.y_m).abs() > 0.03 {
                self.drops[i].support = crate::site::Support::Ground;
            }
        }
    }

    pub(super) fn release_landing(&self, p: &ActPlay, index: usize) -> Option<(f32, f32)> {
        let xs: Vec<f32> = p.rollout.metrics.iter().filter(|(k, _)| *k == "land_x").map(|(_, v)| *v).collect();
        let ys: Vec<f32> = p.rollout.metrics.iter().filter(|(k, _)| *k == "land_y").map(|(_, v)| *v).collect();
        let (lx, ly) = (*xs.get(index)?, *ys.get(index)?);
        let (ox, oz, yaw) = p.origin;
        let (s, c) = yaw.sin_cos();
        Some((ox + c * lx + s * ly, oz + s * lx - c * ly))
    }

    pub(super) fn walk_steps(&mut self, walked_m: f32, e: &nav::EdgePlan, events: &mut Vec<TimedEvent>) -> Option<String> {
        if !self.body.legged() || walked_m <= 0.0 {
            return None;
        }
        let ground = self.ground_at(e.to);
        let props = *ground.surface().props();
        let slope = e.slope_deg.round() as i32;
        let need_plan = self.walk.plan.as_ref().map_or(true, |(_, g, s)| *g != ground || *s != slope);
        if need_plan {
            let spec = self.drive_spec();
            let g = gait::plan(&self.body, spec.max_speed_mps, noto_core::surface::available_mu(self.body.base, &props), e.slope_deg);
            self.walk.plan = Some((g, ground, slope));
        }
        let (g, _, _) = self.walk.plan.clone().expect("set above");
        self.walk.since_step_m += walked_m;
        while self.walk.since_step_m >= g.step_m {
            self.walk.since_step_m -= g.step_m;
            self.walk.steps += 1;
            self.walk.stance = 1 - self.walk.stance;
            let mu = draw_lognormal(&mut self.act_rng, noto_core::surface::available_mu(self.body.base, &props), gait::GROUND_MU_COV);
            self.walk.last_mu = Some(mu);
            let margin = mu - g.rcof;
            let p = noto_core::surface::slip_probability(margin);
            if self.act_rng.next_f32() >= p {
                continue;
            }
            self.walk.slips += 1;
            let (fell, slide) = gait::slip_outcome(&self.body.dims, &g, mu);
            if self.crew_turn.is_none() {
                self.odom.foot_slid(slide);
            }
            let side = if self.walk.stance == 0 { "오른" } else { "왼" };
            if fell {
                let label = format!("{side}발이 미끄러져 넘어짐: 마찰 {mu:.2} < 필요 {:.2}, {:.0} mm", g.rcof, slide * 1000.0);
                events.push(self.stamp(WorldEvent::Act { act: "walk".into(), kind: "fall".into(), label: label.clone(), ok: false, value: Some(slide), parts: vec!["foot".into(), "foot_joints".into(), "waist".into()] }));
                let worker = self.acting_id();
                self.spill_all(worker, events);
                self.fallen_until = Some(self.time_s + GET_UP_S as f64);
                return Some(label);
            }
            // The foot going from under it jerks the load about as far as it slid; what that did to the load goes in
            // the slip's record, and a load it shook loose follows it.
            let worker = self.acting_id();
            let mut shaken = Vec::new();
            self.jolt_cargo([0.0, slide / REACTION_JOLT_S], REACTION_JOLT_S, worker, "foot_slip", &mut shaken);
            let load = match self.ride.jolt.filter(|j| j.cause == "foot_slip" && j.time_s == self.time_s) {
                Some(j) if j.lost => " · 손의 짐을 놓침".to_string(),
                Some(j) => format!(" · 손 짐 충격 {:.1} m/s² (마찰 이용률 {:.0}%, {:.0} mm 밀림)", j.dv_mps[0].hypot(j.dv_mps[1]) / j.dt_s.max(1e-3), j.use_ratio * 100.0, j.moved_m * 1000.0),
                None => String::new(),
            };
            let label = format!("{side}발 미끄러짐 {:.0} mm → 회복 (마찰 {mu:.2} < 필요 {:.2}){load}", slide * 1000.0, g.rcof);
            events.push(self.stamp(WorldEvent::Act { act: "walk".into(), kind: "foot_slip".into(), label, ok: true, value: Some(slide), parts: vec!["foot".into(), "foot_joints".into()] }));
            events.extend(shaken);
        }
        None
    }

    pub(super) fn act_pose(&self) -> Option<(KinPose, KinLoads, Source)> {
        if let Some(p) = &self.act {
            let s = p.rollout.sample_at(p.t())?;
            let src = match p.rollout.kind {
                ActKind::Push | ActKind::Drive | ActKind::Fork => Source::Quiet,
                _ => Source::ArmLift,
            };
            return Some((s.pose, s.loads, src));
        }
        if self.body.legged() && matches!(&self.active, Some((_, Active::Move(_)))) && self.agent.speed_mps > 0.0 {
            let (g, _, _) = self.walk.plan.clone()?;
            let u = (self.walk.since_step_m / g.step_m.max(1e-3)).clamp(0.0, 1.0);
            let (mut pose, loads, _) = gait::pose_at(&self.body, &g, self.walk.stance, u, self.carried_kg());
            // The foot in the air goes over a step, not through it.
            let lift = (self.walk.step_lift_m + 0.03) / gait::SWING_LIFT_M;
            if lift > 1.0 {
                pose.feet[1 - self.walk.stance][1] *= lift;
            }
            return Some((self.crouched(self.carrying(pose)), self.carried_loads(loads.accel), Source::FootLift));
        }
        if self.body.legged() && self.carried_kg() > AMOUNT_EPS {
            let held = self.gaze.map_or(KinPose::stand(&self.body, self.carried_kg()), |g| g.pose);
            return Some((self.crouched(self.carrying(held)), self.carried_loads([0.0; 3]), Source::Quiet));
        }
        self.gaze.map(|g| (g.pose, KinLoads::default(), Source::Quiet))
    }

    /// `pose` with the load in two hands before the body, and the head tipped to the ground a stride ahead while it
    /// walks, where the next step is. Empty hands swing as they do.
    fn carrying(&self, mut pose: KinPose) -> KinPose {
        if self.carried_kg() <= AMOUNT_EPS {
            return pose;
        }
        let held = noto_core::act::cocarry::carry_pose(&self.body, noto_core::act::cocarry::carry_hand_z(&self.body));
        pose.shoulder_deg = held.shoulder_deg;
        pose.elbow_deg = held.elbow_deg;
        pose.closure = held.closure;
        // The load is in the hands (`carried_loads`), not in the bay.
        if self.body.arm_count() > 0 {
            pose.cargo_kg = 0.0;
        }
        if matches!(&self.active, Some((_, Active::Move(_)))) {
            let cam = hkin::head_camera(&self.body, &pose);
            let want = cam.pos[2].atan2(0.5 * (steps::STEP_LOOK_M[0] + steps::STEP_LOOK_M[1])).to_degrees();
            pose.neck_deg = (pose.neck_deg + want - cam.pitch_down_deg).clamp(0.0, 35.0);
        }
        pose
    }

    /// The loads on a walker carrying in its hands, the body accelerating at `accel` (forward, left, up): the load's
    /// weight and inertia on the palms, where it has slid in them. Empty hands carry nothing.
    fn carried_loads(&self, accel: [f32; 3]) -> KinLoads {
        let kg = self.carried_kg();
        if kg <= AMOUNT_EPS || self.body.arm_count() == 0 {
            return KinLoads { accel, ..KinLoads::default() };
        }
        KinLoads::carrying(kg, self.body.arm_count(), accel, self.ride.state.shift)
    }

    pub(super) fn act_body_pose(&self) -> Option<BodyPose> {
        let (pose, loads, src) = self.act_pose()?;
        let r = hkin::solve(&self.body, &pose, &loads);
        Some(r.body_pose(0.0, src))
    }

    pub fn focus_frame(&self) -> FocusFrame {
        let walking = self.act.is_none() && self.body.legged && matches!(&self.active, Some((_, Active::Move(_)))) && self.agent.speed_mps > 0.0;
        let (pose, loads, kind, sample, event, object) = match (&self.act, &self.last_act) {
            (Some(p), _) => {
                let t = p.t();
                let s = p.rollout.sample_at(t);
                let ev = p.rollout.events.iter().filter(|e| e.t <= t + 1e-6).last();
                (s.map_or(KinPose::stand(&self.body, self.carried_kg()), |s| s.pose), s.map_or(KinLoads::default(), |s| s.loads), Some(p.rollout.kind), s.cloned(), ev.cloned(), p.object)
            }
            (None, Some(p)) if self.active.is_none() && self.time_s - (p.start_s + p.rollout.duration_s as f64) < 3.0 => {
                let t = p.t();
                let s = p.rollout.sample_at(t);
                let ev = p.rollout.events.last();
                (s.map_or(KinPose::stand(&self.body, self.carried_kg()), |s| s.pose), s.map_or(KinLoads::default(), |s| s.loads), Some(p.rollout.kind), s.cloned(), ev.cloned(), p.object)
            }
            _ => match self.act_pose() {
                Some((pose, loads, _)) => (pose, loads, Some(ActKind::Walk), None, None, None),
                None => {
                    let moving = matches!(&self.active, Some((_, Active::Move(_)))) && self.agent.speed_mps > 0.0;
                    let pose = KinPose { base_pitch_deg: if self.body.legged() { 0.0 } else { self.posture_now().0.slope_deg }, ..KinPose::stand(&self.body, self.carried_kg()) };
                    (pose, KinLoads::default(), moving.then_some(ActKind::Drive), None, None, None)
                }
            },
        };
        let r = hkin::solve_fast(&self.body, &pose, &loads);
        let props = self.ground_here().surface().props();
        let drive = (!self.body.legged()).then(|| noto_core::act::drive::state(&self.body, props, self.posture_now().0.slope_deg, 0.0, self.carried_kg(), 0.0));
        let obj = object.map(|(k, a)| {
            let s = k.spec().piece_size_m(a);
            ([0.5 * s[0], 0.5 * s[1], 0.5 * s[2]], k.spec().mass_kg * a, material_of(k).shape())
        });
        focus::frame(&FocusCtx { body: &self.body, hand: &self.hand, report: &r, sample: sample.as_ref(), action: kind, event: event.as_ref(), ground_mu: noto_core::surface::available_mu(self.body.base, props), drive, object: obj, step_mu: if walking { self.walk.last_mu } else { None } })
    }

    pub fn reach_ground(&self, kind: ItemKind, amount: f32) -> Result<(), String> {
        let size = kind.spec().piece_size_m(amount);
        let goal = noto_core::act::reach::ReachGoal::ground_object(size[1], kind.spec().mass_kg * amount, false, self.carried_kg());
        let plan = noto_core::act::reach::plan(&self.body, &goal);
        if plan.ok {
            return Ok(());
        }
        let both = noto_core::act::reach::plan(&self.body, &noto_core::act::reach::ReachGoal { two_arms: true, ..goal });
        if both.ok {
            Ok(())
        } else {
            Err(plan.reason_ko.unwrap_or_else(|| "바닥에 손이 닿지 않음".into()))
        }
    }

    pub fn side_reach(&self, kind: ItemKind, amount: f32) -> Option<(f32, f32)> {
        let key = (kind, (amount * 1000.0).round() as i32, (self.carried_kg() * 10.0).round() as i32);
        if let Some(v) = self.side_cache.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
            return *v;
        }
        let size = kind.spec().piece_size_m(amount);
        let goal = noto_core::act::reach::ReachGoal::ground_object(size[1], kind.spec().mass_kg * amount, false, self.carried_kg());
        let plan = noto_core::act::reach::plan_side(&self.body, &goal, 0.5 * size[2]);
        let v = plan.ok.then_some((plan.object_x, plan.object_y));
        self.side_cache.lock().unwrap_or_else(|e| e.into_inner()).insert(key, v);
        v
    }

    pub fn stock_pushed(&mut self, drop: u32, waypoint: u32, events: &mut Vec<TimedEvent>) -> Option<f32> {
        let d = self.drop_item(drop)?.clone();
        let w = self.waypoint(waypoint)?;
        if (d.x - w.x).hypot(d.z - w.z) > DELIVER_REACH_M {
            return None;
        }
        self.drops.retain(|x| x.id != drop);
        if let Some(w) = self.waypoints.iter_mut().find(|w| w.id == waypoint) {
            add_stock(&mut w.stock, d.kind, d.amount);
        }
        events.push(self.stamp(WorldEvent::Delivered { waypoint: waypoint_id(waypoint), item: d.kind, amount: d.amount }));
        Some(d.amount)
    }

    pub fn push_route(&self, drop: u32, x: f32, z: f32) -> Result<(nav::Plan, (f32, f32, f32)), ActionError> {
        if self.body.legged() {
            return Err(ActionError::CannotPush { reason: "다리형은 범퍼가 없어 바닥의 물체를 밀지 않음 (들어서 옮김)".into() });
        }
        let d = self.drop_item(drop).ok_or(ActionError::UnknownTarget)?;
        if !d.support.is_ground() {
            return Err(ActionError::CannotPush { reason: format!("{}의 물체는 범퍼로 밀 수 없음", d.support.label_ko()) });
        }
        let start = self.terrain.cell_of(d.x, d.z).ok_or(ActionError::UnknownTarget)?;
        let nodes = self.blocked_cells();
        let t = &self.terrain;
        let blocked = |cx: usize, cz: usize| nodes.contains(&(cx, cz)) || t.curb_m(cx, cz) > 0.0;
        let spec = self.drive_spec();
        let load = nav::PushLoad { mass_kg: d.kind.spec().mass_kg * d.amount, material: material_of(d.kind) };
        let plan = nav::plan_with(&spec, t, &blocked, start, Goal::Within { x, z, reach_m: 0.5 }, Some(load)).ok_or(ActionError::NoPath)?;
        let side = d.kind.spec().piece_size_m(d.amount)[0];
        let lead = apush::lead_m(&self.body, side);
        let (hx, hz) = plan.cells.get(1).map(|c| cell_center(c.0, c.1)).unwrap_or((x, z));
        let yaw = (hz - d.z).atan2(hx - d.x);
        Ok((plan, (d.x - lead * yaw.cos(), d.z - lead * yaw.sin(), yaw)))
    }

    pub(super) fn start_push(&mut self, drop: u32, x: f32, z: f32) -> Result<PushState, ActionError> {
        let (plan, (lx, lz, yaw)) = self.push_route(drop, x, z)?;
        let off = (lx - self.agent.x).hypot(lz - self.agent.z);
        if off > PUSH_START_SLACK_M {
            return Err(ActionError::OutOfReach { distance_m: off, reach_m: PUSH_START_SLACK_M });
        }
        let d = self.drop_item(drop).cloned().ok_or(ActionError::UnknownTarget)?;
        let size = d.kind.spec().piece_size_m(d.amount);
        let radius = apush::planned_radius_m(&self.body, material_of(d.kind), size[0]);
        let mut path = vec![(d.x, d.z)];
        path.extend(plan.cells.iter().skip(1).map(|c| cell_center(c.0, c.1)));
        let mut legs: Vec<apush::PushLeg> = Vec::new();
        let mut heading = yaw;
        for w in path.windows(2) {
            let (a, b) = (w[0], w[1]);
            let h = (b.1 - a.1).atan2(b.0 - a.0);
            let turn = wrap_angle(h - heading).to_degrees();
            heading = h;
            let cell = self.terrain.cell_of(b.0, b.1).unwrap_or((0, 0));
            let rise = self.terrain.height_m(cell.0, cell.1) - self.terrain.cell_of(a.0, a.1).map_or(0.0, |c| self.terrain.height_m(c.0, c.1));
            let len = (b.0 - a.0).hypot(b.1 - a.1);
            let slope = rise.atan2(len.max(1e-3)).to_degrees();
            let ground = self.ground_at(cell).surface();
            match legs.last_mut() {
                Some(l) if turn.abs() < 1.0 && l.ground == ground && (l.slope_deg - slope).abs() < 1.0 => l.len_m += len,
                _ => legs.push(apush::PushLeg { turn_deg: turn, radius_m: if turn.abs() < 1.0 { 0.0 } else { radius }, len_m: len, ground, slope_deg: slope, blocked: false }),
            }
        }
        let inp = apush::PushInput { body: &self.body, material: material_of(d.kind), mass_kg: d.mass_kg(), size_m: size, cargo_kg: self.carried_kg(), legs, aim_sigma_m: self.aim_sigma_m(), seed: self.act_rng.next_u64() };
        let mut rollout = apush::push(&inp);
        let creep = off / SHUFFLE_MPS;
        if creep > 0.05 {
            for s in rollout.samples.iter_mut() {
                s.t += creep;
            }
            for e in rollout.events.iter_mut() {
                e.t += creep;
            }
            rollout.duration_s += creep;
            rollout.metrics.push(("line_up_s", creep));
        }
        let length: f32 = path.windows(2).map(|w| (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1)).sum();
        let end_m = if rollout.outcome.ok { length } else { rollout.metric("pushed_m").unwrap_or(0.0).min(length) };
        let lead = apush::lead_m(&self.body, size[0]);
        self.agent.yaw = yaw;
        self.act = Some(ActPlay::new(rollout, self.time_s, Some((d.kind, d.amount)), (self.agent.x, self.agent.z, yaw)));
        Ok(PushState { drop, path, length_m: length, done_m: 0.0, end_m, lead_m: lead })
    }

    pub(super) fn advance_push(&mut self, ps: &mut PushState, remaining: f32, events: &mut Vec<TimedEvent>, done: &mut Option<(bool, Option<String>)>) -> f32 {
        let Some(p) = self.act.as_ref() else {
            *done = Some((false, Some("push_lost".into())));
            return 0.0;
        };
        let dur = p.rollout.duration_s.max(1e-3);
        let t_now = p.t();
        let dt = remaining.min((dur - t_now).max(0.0));
        let t_next = t_now + dt;
        let pace = if p.rollout.outcome.ok { ps.length_m / dur } else { ps.end_m / dur };
        ps.done_m = (pace * t_next).min(ps.end_m);
        let (ix, iz, h) = point_along(&ps.path, ps.done_m);
        if let Some(d) = self.drops.iter_mut().find(|d| d.id == ps.drop) {
            d.x = ix;
            d.z = iz;
        }
        self.agent.x = ix - ps.lead_m * h.cos();
        self.agent.z = iz - ps.lead_m * h.sin();
        self.agent.yaw = h;
        self.agent.speed_mps = pace;
        let power = self.drive_spec().power_w as f64;
        let push_n = p.rollout.metric("push_peak_n").unwrap_or(0.0) as f64;
        self.add_energy(power * dt as f64 + push_n * pace as f64 * dt as f64);
        self.act_advance(dt, events);
        if t_next >= dur - 1e-4 {
            let ok = self.act.as_ref().map_or(false, |p| p.rollout.outcome.ok);
            let code = self.act.as_ref().and_then(|p| p.rollout.outcome.code).map(String::from);
            if !ok {
                let off = self.act.as_ref().and_then(|p| p.rollout.metric("offset_m")).unwrap_or(0.0);
                if let Some(d) = self.drops.iter_mut().find(|d| d.id == ps.drop) {
                    d.x += -h.sin() * off.clamp(-0.4, 0.4);
                    d.z += h.cos() * off.clamp(-0.4, 0.4);
                }
            }
            self.end_act();
            *done = Some((ok, code));
        }
        dt.max(1e-4)
    }
}

const SHOULDER_RANGE_M: f32 = 1.05;

fn point_along(path: &[(f32, f32)], s: f32) -> (f32, f32, f32) {
    let mut left = s.max(0.0);
    let n = path.len().saturating_sub(1);
    for i in 0..n {
        let (a, b) = (path[i], path[i + 1]);
        let len = (b.0 - a.0).hypot(b.1 - a.1);
        let h = (b.1 - a.1).atan2(b.0 - a.0);
        if left <= len || i + 1 == n {
            let u = (left / len.max(1e-6)).clamp(0.0, 1.0);
            return (a.0 + (b.0 - a.0) * u, a.1 + (b.1 - a.1) * u, h);
        }
        left -= len;
    }
    let p = path.last().copied().unwrap_or((0.0, 0.0));
    (p.0, p.1, 0.0)
}

fn act_event(kind: ActKind, e: &ActEvent) -> WorldEvent {
    WorldEvent::Act { act: kind.id().into(), kind: e.kind.into(), label: e.label_ko.clone(), ok: e.ok, value: e.value, parts: e.parts.iter().map(|p| p.id().to_string()).collect() }
}

fn draw_lognormal(rng: &mut Rng, mean: f32, cov: f32) -> f32 {
    let s2 = (1.0 + cov * cov).ln();
    let mu = mean.ln() - 0.5 * s2;
    let u1 = rng.next_f32().max(1e-7);
    let u2 = rng.next_f32();
    let z = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
    (mu + s2.sqrt() * z).exp()
}
