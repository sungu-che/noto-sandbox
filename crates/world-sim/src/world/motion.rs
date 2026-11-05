//! Driving the planned cell path like a vehicle or a walker, not like a cursor: the base follows the path with a
//! pure-pursuit steering law (Coulter, "Implementation of the Pure Pursuit Path Tracking Algorithm", CMU-RI-TR-92-01,
//! 1992), so it rounds corners on arcs at the speed the arc allows (`steer::TurnCaps`), stops to turn only where the
//! path bends more than it can steer, turns there the way the base can (scrub pivot, K-turn, step turn), and reaches
//! a goal close behind it in reverse. The cell edges keep their physics (speed on the ground and slope, curb pauses,
//! footsteps, slip): the edge under the base sets the speed at each moment.
//!
//! The body keeps clear of what it can hit (`clear`): the route's corners are pushed to the middle of the gaps, a step
//! that would bring the body's outline against a tree, a rock or a wall follows the route line instead of cutting the
//! corner, and a turn in place is taken only where its sweep is clear. Where there is no room to turn and the way on is
//! behind (a dead end between obstacles), the base does not spin on the spot: it backs straight out the way it came
//! in, as a driver reverses out of a blind alley, to the first place it can turn, turns there and goes on forward. A
//! turn asked for in such a place (the search looking about, facing an item) backs out first the same way.
//!
//! Nothing moves the body without asking first. Every pose it takes passes `PoseCheck`, against the solids of the map,
//! the other robots and what lies on the ground; where neither the pursuit arc nor the route line will do, the way is
//! shut: the base stops there. If it was going fast enough to touch, that is a collision, and `act::collide::bump` says
//! what it did to the robot and its load (a fall, a jolt). Then it plans again round what stopped it.

use super::*;
use crate::clear::{Plate, Solid, SolidKind};
use crate::steer::{self, Seg, TurnCaps, TurnKind};
use noto_core::act::collide::{self, BumpInput, Obstacle};

pub const LOOKAHEAD_MIN_M: f32 = 0.35;
pub const LOOKAHEAD_MAX_M: f32 = 1.0;
pub const LOOKAHEAD_S: f32 = 0.6;
pub const ALPHA_STOP_SKID_RAD: f32 = 1.2;
pub const ALPHA_STOP_LEGS_RAD: f32 = 1.05;
pub const REVERSE_BEHIND_RAD: f32 = 2.0;
pub const REVERSE_STRAIGHT_RAD: f32 = 0.8;
pub const SUBSTEP_S: f32 = 0.05;
pub const CREEP_RAD_S: f32 = 0.25;
pub const MIN_CURVE_MPS: f32 = 0.08;
pub const GOAL_TOL_M: f32 = 0.06;
pub const MAX_STOP_TURNS: u32 = 6;
pub const STALL_S: f32 = 2.5;
pub const FINAL_APPROACH_M: f32 = 0.45;
pub const STEER_MAX_RAD: f32 = 0.75;
pub const STEER_RATE_RAD_S: f32 = 1.0;
/// How far apart the places a backing base tries to turn at are (m).
pub const BACK_OUT_STEP_M: f32 = 0.1;
/// The farthest a base backs straight out of a dead end looking for room to turn (m) [추정].
pub const BACK_OUT_MAX_M: f32 = 6.0;
/// How many times a move plans again round what shut its way before it gives up.
pub const MAX_REPLANS: u32 = 8;
/// How far short of its reach a body stopped by what it was going for still counts as there (m).
pub const ARRIVED_SHORT_M: f32 = 0.15;
/// How far short of it a body settles for when a robot that will not move stands in the way (m).
pub const ROBOT_SHORT_M: f32 = 0.4;
/// A move that has not taken the body `STUCK_M` from where it was in `STUCK_S` is going nowhere, whatever the body is
/// busy with: a truck in a tight spot can shunt back and forth for minutes (m, s) [추정].
pub const STUCK_M: f32 = 1.0;
pub const STUCK_S: f64 = 40.0;
/// How far a stopped base backs off to make room to plan again from (m), longest first.
pub const BACK_OFF_M: [f32; 3] = [1.2, 0.8, 0.4];
/// How many routes are tried for one that takes the whole body.
pub const FIT_TRIES: usize = 6;
/// Slower than this a touch is a nudge, not a collision (m/s) [추정].
pub const BUMP_MIN_MPS: f32 = 0.12;
/// How long a robot waits for another in its way before it plans round it (s).
pub const ROBOT_WAIT_S: f32 = 2.0;

#[derive(Clone, Debug)]
pub(crate) struct SegRun {
    pub kind: &'static str,
    pub label_ko: String,
    pub segs: Vec<Seg>,
    pub i: usize,
    pub u: f32,
    pub start: (f32, f32, f32),
    pub wait_left: f32,
    pub forced: bool,
    /// Stopped short: its next pose ran into something that was not there when it was planned.
    pub blocked: bool,
    /// The share of full speed it is doing, built up from rest with a load on.
    pub pace: f32,
}

impl SegRun {
    pub fn new(m: steer::Maneuver, at: (f32, f32, f32)) -> SegRun {
        SegRun { kind: m.kind, label_ko: m.label_ko, segs: m.segs, i: 0, u: 0.0, start: at, wait_left: 0.0, forced: false, blocked: false, pace: 0.0 }
    }

    pub fn done(&self) -> bool {
        self.i >= self.segs.len()
    }

    pub fn reversing(&self) -> bool {
        matches!(self.segs.get(self.i), Some(Seg::Drive { dir, .. }) if *dir < 0.0)
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Follow {
    pub path: Vec<(f32, f32)>,
    pub cum: Vec<f32>,
    pub edge_at: Vec<usize>,
    pub s: f32,
    pub reverse: bool,
    pub man: Option<SegRun>,
    pub note: Option<(&'static str, String)>,
    pub turns: u32,
    pub on_rails: bool,
    pub best_s: f32,
    pub stall_s: f32,
    pub stop_within: Option<((f32, f32), f32)>,
    pub steer: f32,
    /// How long it has stood for another robot in the way (s).
    pub robot_wait_s: f32,
    /// How fast it is turning to the route line (rad/s), built up and let down with a load on.
    pub spin: f32,
    /// It is backing off to plan again once it has.
    pub then_replan: bool,
}

impl Follow {
    pub fn total_m(&self) -> f32 {
        self.cum.last().copied().unwrap_or(0.0)
    }

    pub fn point_at(&self, s: f32) -> (f32, f32) {
        let s = s.clamp(0.0, self.total_m());
        let k = self.cum.partition_point(|c| *c <= s).clamp(1, self.path.len().max(2) - 1);
        let (a, b) = (self.path[k - 1], self.path[k]);
        let len = (self.cum[k] - self.cum[k - 1]).max(1e-6);
        let u = ((s - self.cum[k - 1]) / len).clamp(0.0, 1.0);
        (a.0 + (b.0 - a.0) * u, a.1 + (b.1 - a.1) * u)
    }

    /// The points still ahead on the route.
    pub fn ahead(&self) -> Vec<(f32, f32)> {
        self.path.iter().zip(&self.cum).filter(|(_, c)| **c > self.s).map(|(p, _)| *p).collect()
    }

    /// Lengths along the route again after its points moved.
    pub fn measure(&mut self) {
        self.cum = vec![0.0];
        for w in self.path.windows(2) {
            self.cum.push(self.cum.last().unwrap() + (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1));
        }
    }

    pub fn edge_index(&self, s: f32) -> usize {
        let k = self.cum.partition_point(|c| *c <= s.max(0.0)).clamp(1, self.path.len().max(2) - 1);
        self.edge_at.get(k - 1).copied().unwrap_or(0)
    }

    fn project(&self, p: (f32, f32), from: f32, ahead: f32) -> f32 {
        let (mut best, mut best_d) = (from, f32::INFINITY);
        for k in 1..self.path.len() {
            if self.cum[k] < from - 1e-4 || self.cum[k - 1] > from + ahead {
                continue;
            }
            let (a, b) = (self.path[k - 1], self.path[k]);
            let (dx, dz) = (b.0 - a.0, b.1 - a.1);
            let l2 = (dx * dx + dz * dz).max(1e-9);
            let u = (((p.0 - a.0) * dx + (p.1 - a.1) * dz) / l2).clamp(0.0, 1.0);
            let (qx, qz) = (a.0 + dx * u, a.1 + dz * u);
            let d = (p.0 - qx).hypot(p.1 - qz);
            let s = self.cum[k - 1] + u * l2.sqrt();
            if d < best_d && s >= from - 1e-4 {
                (best, best_d) = (s, d);
            }
        }
        best.max(from)
    }

    /// How much the path turns in total (rad), corner by corner.
    pub fn turning(&self) -> f32 {
        self.path.windows(3).map(|w| steer::wrap((w[2].1 - w[1].1).atan2(w[2].0 - w[1].0) - (w[1].1 - w[0].1).atan2(w[1].0 - w[0].0)).abs()).sum()
    }
}

pub(crate) fn follow_of(start: (f32, f32), edges: &[EdgePlan]) -> Follow {
    let mut path = vec![start];
    let mut edge_at = Vec::new();
    for (i, e) in edges.iter().enumerate() {
        let p = cell_center(e.to.0, e.to.1);
        if (p.0 - path.last().unwrap().0).hypot(p.1 - path.last().unwrap().1) < 1e-4 {
            continue;
        }
        path.push(p);
        edge_at.push(i);
    }
    let mut f = Follow { path, edge_at, ..Follow::default() };
    f.measure();
    f
}

/// What stopped a pose.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Stop {
    Solid(f32, Solid),
    Robot(u32),
}

/// Whether the base may take a pose: its centre on dry ground of the map, off blocked cells and no curb or ledge
/// away from the height it measures from, and its body `clear::TOUCH_GAP_M` off anything it can hit (or no closer than
/// it already is, when it starts closer, so a body caught in a tight spot can only get out of it). What lies loose on
/// the ground is not in this: it gives way (`shove_loose`).
pub(crate) struct PoseCheck {
    map: std::sync::Arc<ClearMap>,
    fp: Footprint,
    blocked: HashSet<Cell>,
    here: Cell,
    pub h0: f32,
    /// The most the ground may change from the height it measures from: a step, and a cell's worth of the steepest
    /// slope. What it cannot climb stands as a ledge in the map; this keeps the base's centre on its own level.
    climb: f32,
    gap: f32,
    others: Vec<(u32, Plate)>,
}

impl PoseCheck {
    pub fn ok(&self, t: &Terrain, x: f32, z: f32, yaw: f32) -> bool {
        self.ok_from(t, x, z, yaw, self.h0)
    }

    pub fn ok_from(&self, t: &Terrain, x: f32, z: f32, yaw: f32, h_ref: f32) -> bool {
        let centre = match t.cell_of(x, z) {
            Some(c) if c == self.here && (h_ref - self.h0).abs() < 1e-6 => true,
            Some(c) => !t.is_water(c.0, c.1) && !self.blocked.contains(&c) && (t.height_m(c.0, c.1) + t.curb_m(c.0, c.1) - h_ref).abs() <= self.climb,
            None => false,
        };
        centre && self.clear(x, z, yaw)
    }

    /// Whether the body keeps its gap at that pose.
    pub fn clear(&self, x: f32, z: f32, yaw: f32) -> bool {
        room_at(&self.fp, &self.map, &self.others, x, z, yaw) >= self.gap
    }

    /// What is nearest the body at that pose, and how far (m; negative inside it).
    pub fn stopped_by(&self, x: f32, z: f32, yaw: f32) -> Option<Stop> {
        let plates = body_plates(&self.fp, x, z, yaw);
        let robot = self.others.iter().flat_map(|(id, o)| plates.iter().map(move |p| (*id, p.gap(o)))).min_by(|a, b| a.1.total_cmp(&b.1));
        let solid = self.fp.nearest(&self.map, x, z, yaw);
        if let Some((id, _)) = robot.filter(|r| r.1 <= solid.map_or(f32::INFINITY, |s| s.0) && r.1 < clear::TOUCH_GAP_M) {
            return Some(Stop::Robot(id));
        }
        solid.filter(|s| s.0 < clear::TOUCH_GAP_M).map(|s| Stop::Solid(s.0, s.1))
    }

    /// How high the ground stands at the cell under (`x`, `z`) (m).
    pub(crate) fn ground_m(t: &Terrain, x: f32, z: f32) -> f32 {
        t.cell_of(x, z).map_or(0.0, |c| t.height_m(c.0, c.1) + t.curb_m(c.0, c.1))
    }
}

/// As far as `ds` from `from` toward `to`: a base does not jump back onto a line it has come off.
fn step_toward(from: (f32, f32), to: (f32, f32), ds: f32) -> (f32, f32) {
    let d = (to.0 - from.0).hypot(to.1 - from.1);
    if d <= ds {
        return to;
    }
    (from.0 + (to.0 - from.0) * ds / d, from.1 + (to.1 - from.1) * ds / d)
}

/// The hull, and the tines when it has them.
fn body_plates(fp: &Footprint, x: f32, z: f32, yaw: f32) -> Vec<Plate> {
    let mut v = vec![fp.hull(x, z, yaw)];
    v.extend(fp.tines(x, z, yaw).map(|t| t.0));
    v
}

fn room_at(fp: &Footprint, map: &ClearMap, others: &[(u32, Plate)], x: f32, z: f32, yaw: f32) -> f32 {
    let mut r = fp.room(map, x, z, yaw);
    if !others.is_empty() {
        let plates = body_plates(fp, x, z, yaw);
        for (_, o) in others {
            for p in &plates {
                r = r.min(p.gap(o));
            }
        }
    }
    r
}

impl World {
    pub fn turn_caps(&self) -> TurnCaps {
        steer::caps(&self.drive_spec(), self.body.support_len_m, self.body.support_width_m, self.body.chassis.drive_force_n, self.ground_here().surface().props())
    }

    pub(crate) fn is_track(&self) -> bool {
        self.hand.base.kind == noto_core::BaseKind::Track
    }

    /// Whether this body has room to stand at a pose.
    pub fn stands_clear(&self, x: f32, z: f32, yaw: f32) -> bool {
        self.footprint().room(&self.clear_map(), x, z, yaw) >= clear::TOUCH_GAP_M
    }

    /// Whether `body` could stand at a spot and turn right round there.
    pub fn turns_clear(&self, body: &Body, x: f32, z: f32) -> bool {
        let (fp, map) = (Footprint::of(body), self.clear_map());
        (0..8).all(|k| fp.room(&map, x, z, k as f32 * std::f32::consts::FRAC_PI_4) >= clear::TOUCH_GAP_M)
    }

    /// What the base may do with its body here (`PoseCheck`), measured from where it stands now.
    pub(crate) fn pose_check(&self) -> PoseCheck {
        let here = self.agent_cell();
        let map = self.clear_map();
        let fp = self.footprint();
        let others: Vec<(u32, Plate)> = self.other_robots().iter().map(|o| (o.id, Plate::of_pose(o.x, o.z, o.yaw, o.front, o.rear, o.half_w))).collect();
        let now = room_at(&fp, &map, &others, self.agent.x, self.agent.z, self.agent.yaw);
        PoseCheck {
            gap: clear::TOUCH_GAP_M.min(now - 1e-3),
            map,
            fp,
            blocked: self.blocked_cells(),
            here,
            h0: self.terrain.height_m(here.0, here.1) + self.terrain.curb_m(here.0, here.1),
            climb: {
                let spec = self.drive_spec();
                spec.max_step_m + 0.01 + CELL_M * spec.max_slope_deg.to_radians().tan()
            },
            others,
        }
    }

    /// Whether the base, at (`x`, `z`) facing `yaw`, has room to turn in place to face `to` (a pivot or a K-turn whose
    /// sweep is clear). `h_ref` is the ground height the manoeuvre measures curbs from.
    pub(crate) fn room_to_turn(&self, check: &PoseCheck, x: f32, z: f32, yaw: f32, to: f32, h_ref: f32) -> bool {
        let caps = self.turn_caps();
        let v = self.drive_spec().max_speed_mps;
        steer::turn_in_place((x, z, yaw), wrap_angle(to - yaw), &caps, self.is_track(), v, &|px, pz, pyaw| check.ok_from(&self.terrain, px, pz, pyaw, h_ref)).is_some()
    }

    /// The first spot straight behind the base (every `BACK_OUT_STEP_M`, at most `BACK_OUT_MAX_M`, over ground it may
    /// back across) where it has room to turn to face `want(x, z)`: how far back it is, and the turn there (none when
    /// `want` is within `ok_rad` of the heading). `None` when the way back is blocked first.
    fn back_out_spot(&self, check: &PoseCheck, ok_rad: f32, want: &dyn Fn(f32, f32) -> f32) -> Option<(f32, steer::Maneuver)> {
        let caps = self.turn_caps();
        let v = self.drive_spec().max_speed_mps;
        let (x0, z0, yaw) = (self.agent.x, self.agent.z, self.agent.yaw);
        let (sn, c) = yaw.sin_cos();
        let mut h_prev = check.h0;
        let n = (BACK_OUT_MAX_M / steer::SAMPLE_M).ceil() as usize;
        let per_try = (BACK_OUT_STEP_M / steer::SAMPLE_M).round().max(1.0) as usize;
        for k in 1..=n {
            let d = k as f32 * steer::SAMPLE_M;
            let (x, z) = (x0 - c * d, z0 - sn * d);
            if !check.ok_from(&self.terrain, x, z, yaw, h_prev) {
                return None;
            }
            h_prev = PoseCheck::ground_m(&self.terrain, x, z);
            if k % per_try != 0 {
                continue;
            }
            let delta = wrap_angle(want(x, z) - yaw);
            if delta.abs() <= ok_rad {
                return Some((d, steer::Maneuver { kind: "none", label_ko: String::new(), segs: vec![], time_s: 0.0 }));
            }
            let h = h_prev;
            if let Some(m) = steer::turn_in_place((x, z, yaw), delta, &caps, self.is_track(), v, &|px, pz, pyaw| check.ok_from(&self.terrain, px, pz, pyaw, h)) {
                return Some((d, m));
            }
        }
        None
    }

    /// Backing `d` m straight out, then `turn`, as one manoeuvre from where the base stands.
    fn backed_turn(&self, d: f32, turn: steer::Maneuver, label_ko: String) -> SegRun {
        let caps = self.turn_caps();
        let mut segs = steer::reverse_line(d);
        segs.extend(turn.segs);
        let time_s = steer::maneuver_time(&segs, &caps, self.drive_spec().max_speed_mps);
        SegRun::new(steer::Maneuver { kind: "reverse", label_ko, segs, time_s }, (self.agent.x, self.agent.z, self.agent.yaw))
    }

    fn back_word(&self) -> &'static str {
        if self.turn_caps().kind == TurnKind::Legs {
            "뒷걸음"
        } else {
            "후진"
        }
    }

    /// A dead end: there is no room to turn here and the way on is behind. The base backs straight out, the way it
    /// came in, as a driver reverses out of a blind alley, to the first spot where its sweep is clear to turn toward
    /// the route, and turns there (`back_out_spot`); with no such spot it reverses along the route instead. Returns
    /// whether it found the spot.
    pub(crate) fn back_out(&self, f: &mut Follow, check: &PoseCheck) -> bool {
        let caps = self.turn_caps();
        let stop = if caps.kind == TurnKind::Legs { ALPHA_STOP_LEGS_RAD } else { ALPHA_STOP_SKID_RAD };
        let total = f.total_m();
        let how = self.back_word();
        let route = |x: f32, z: f32| {
            let near = f.project((x, z), 0.0, total);
            let ahead = f.point_at((near + LOOKAHEAD_MAX_M).min(total));
            (ahead.1 - z).atan2(ahead.0 - x)
        };
        let spot = self.back_out_spot(check, stop, &route);
        f.steer = 0.0;
        match spot {
            Some((d, turn)) => {
                let label_ko = format!("막다른 길: 돌 자리가 없어 {d:.1} m 곧게 {how}해 빠져나온 뒤 돎");
                f.note = Some(("dead_end", label_ko.clone()));
                f.man = Some(self.backed_turn(d, turn, label_ko));
                true
            }
            None => {
                f.reverse = true;
                f.note = Some(("dead_end", format!("막다른 길: 돌 자리가 없어 길을 따라 끝까지 {total:.1} m {how}")));
                false
            }
        }
    }

    /// How to turn to face `target` where the base stands: in place when its sweep is clear, else backed straight out
    /// of the tight spot first (a dead end between obstacles: no spinning against them), and only when even that is
    /// blocked, a forced scrub on the spot.
    pub(crate) fn turn_run(&self, target: f32) -> SegRun {
        let pose = (self.agent.x, self.agent.z, self.agent.yaw);
        let delta = wrap_angle(target - self.agent.yaw);
        let caps = self.turn_caps();
        let v = self.drive_spec().max_speed_mps;
        let check = self.pose_check();
        if let Some(m) = steer::turn_in_place(pose, delta, &caps, self.is_track(), v, &|x, z, yaw| check.ok(&self.terrain, x, z, yaw)) {
            return SegRun::new(m, pose);
        }
        if let Some((d, turn)) = self.back_out_spot(&check, FACE_DONE_RAD, &|_, _| target) {
            let how = self.back_word();
            return self.backed_turn(d, turn, format!("자리가 좁아 제자리에서 {:.0}° 돌지 않고 {d:.1} m 곧게 {how}한 뒤 돎", delta.abs().to_degrees()));
        }
        SegRun {
            forced: true,
            ..SegRun::new(steer::Maneuver { kind: "pivot", label_ko: format!("자리가 좁아 제자리에서 억지로 {:.0}° 비벼 돎 (스키드 조향 힘 부족)", delta.abs().to_degrees()), segs: vec![Seg::Pivot { angle: delta }], time_s: 0.0 }, pose)
        }
    }

    /// Plays `run` for up to `remaining` s. Returns the time used.
    pub(crate) fn advance_run(&mut self, run: &mut SegRun, remaining: f32) -> f32 {
        if run.wait_left > 0.0 {
            let t = run.wait_left.min(remaining);
            run.wait_left -= t;
            self.agent.speed_mps = 0.0;
            self.add_energy(0.2 * self.drive_spec().power_w as f64 * t as f64);
            return t;
        }
        let Some(seg) = run.segs.get(run.i).copied() else { return 0.0 };
        let caps = self.turn_caps();
        let v_fwd = self.drive_spec().max_speed_mps * if run.kind == "kturn" { 0.5 } else { 1.0 };
        let (total, speed) = match seg {
            Seg::Pivot { angle } => {
                let rate = if caps.can_pivot() { caps.pivot_rad_s } else { CREEP_RAD_S };
                (angle.abs() / rate.max(0.05), 0.0)
            }
            Seg::Drive { .. } => {
                let t = steer::seg_time(&seg, &caps, v_fwd);
                (t, seg.length_m() / t.max(1e-6))
            }
        };
        let left = (1.0 - run.u) * total;
        let t = left.min(remaining).max(0.0);
        // With a load on, a turn or a shunt starts and stops within what its friction stands.
        let pace = self.load_pace(run, &seg, total, t);
        run.u = if total > 1e-6 { (run.u + pace * t / total).min(1.0) } else { 1.0 };
        let (x, z, yaw) = steer::along(run.start, &seg, run.u);
        // A manoeuvre is planned clear, but the world moves on while it runs, and a forced scrub has no plan to go by.
        if !self.pose_check().clear(x, z, wrap_angle(yaw)) {
            run.blocked = true;
            run.i = run.segs.len();
            run.wait_left = 0.0;
            self.agent.speed_mps = 0.0;
            return t.max(1e-4);
        }
        let moved = (x - self.agent.x).hypot(z - self.agent.z);
        (self.agent.x, self.agent.z, self.agent.yaw) = (x, z, wrap_angle(yaw));
        self.shove_loose();
        self.agent.speed_mps = speed * pace;
        let scrub = match (seg, caps.kind) {
            (Seg::Pivot { .. }, TurnKind::Skid) => if run.forced { 2.0 } else { 1.3 },
            _ => 1.0,
        };
        self.add_energy(self.drive_spec().power_w as f64 * scrub * t as f64);
        if moved > 0.0 && self.body.legged() {
            self.walk.since_step_m += moved;
        }
        if run.u >= 1.0 - 1e-6 {
            run.i += 1;
            run.u = 0.0;
            run.start = (self.agent.x, self.agent.z, self.agent.yaw);
            let flips = match (seg, run.segs.get(run.i)) {
                (Seg::Drive { dir: a, .. }, Some(Seg::Drive { dir: b, .. })) => a != *b,
                _ => false,
            };
            if flips {
                run.wait_left = steer::SHIFT_S;
            }
        }
        t
    }

    /// The share of a manoeuvre's full speed a loaded base keeps this moment: it builds up and comes down at no more
    /// than the load stands, reckoned where the load sits for a turn on the spot.
    fn load_pace(&self, run: &mut SegRun, seg: &Seg, total: f32, dt: f32) -> f32 {
        let Some(a) = self.load_accel() else { return 1.0 };
        let (full, left) = match *seg {
            Seg::Pivot { angle } => {
                let lever = self.load_lever_m();
                (angle.abs() * lever / total.max(1e-3), (1.0 - run.u) * angle.abs() * lever)
            }
            Seg::Drive { .. } => (seg.length_m() / total.max(1e-3), (1.0 - run.u) * seg.length_m()),
        };
        let up = run.pace + a * dt / full.max(1e-3);
        let down = (2.0 * a * left.max(0.0)).sqrt() / full.max(1e-3) + 0.05;
        run.pace = up.min(down).clamp(0.05, 1.0);
        run.pace
    }

    /// The route of `edges` as the body will drive it: pushed off what it passes and fitted to the body.
    fn fitted_route(&self, edges: &[EdgePlan]) -> Follow {
        let mut f = follow_of((self.agent.x, self.agent.z), edges);
        let map = self.clear_map();
        let fp = self.footprint();
        clear::centre_path(&map, &mut f.path, fp.half_w_m + clear::MIN_GAP_M, fp.half_w_m + clear::COMFORT_GAP_M);
        clear::fit_path(&map, &fp, &mut f.path, clear::TOUCH_GAP_M + 0.01);
        f.measure();
        f
    }

    /// Where a planned route will not take the whole body: the cells of the corners and stretches where it swings into
    /// something. The planner sees a width; a lift truck's tines reach well out in front of that, and swing.
    fn route_faults(&self, plan: &nav::Plan) -> Vec<Cell> {
        let f = self.fitted_route(&plan.edges);
        let (map, fp) = (self.clear_map(), self.footprint());
        let hits = |x: f32, z: f32, yaw: f32| fp.room(&map, x, z, yaw) < clear::TOUCH_GAP_M;
        let mut out = Vec::new();
        for i in 1..f.path.len().saturating_sub(1) {
            let (a, b, c) = (f.path[i - 1], f.path[i], f.path[i + 1]);
            let (inward, outward) = ((b.1 - a.1).atan2(b.0 - a.0), (c.1 - b.1).atan2(c.0 - b.0));
            let turn = wrap_angle(outward - inward);
            let round = (0..=4).any(|k| hits(b.0, b.1, inward + turn * k as f32 / 4.0));
            let along = (1..4).any(|k| {
                let u = k as f32 / 4.0;
                hits(b.0 + (c.0 - b.0) * u, b.1 + (c.1 - b.1) * u, outward)
            });
            if round || along {
                out.extend(self.terrain.cell_of(b.0, b.1));
            }
        }
        out
    }

    /// `plan` or, where the whole body will not go along it, the route that keeps off the corners it will not make. The
    /// cells of `avoid` stay shut. The first plan stands when there is nothing better.
    pub(crate) fn fit_plan(&self, plan: nav::Plan, goal: Goal, avoid: &HashSet<Cell>) -> nav::Plan {
        let here = self.agent_cell();
        let mut shut = avoid.clone();
        let mut now = plan;
        let first = now.clone();
        for _ in 0..FIT_TRIES {
            let faults: Vec<Cell> = self.route_faults(&now).into_iter().filter(|c| *c != here && !matches!(goal, Goal::Cell(g) if g == *c)).collect();
            if faults.is_empty() {
                return now;
            }
            shut.extend(faults);
            match self.plan_route_avoiding(goal, &shut) {
                Some(p) => now = p,
                None => return first,
            }
        }
        first
    }

    pub(crate) fn start_follow(&self, edges: &[EdgePlan]) -> Follow {
        let mut f = self.fitted_route(edges);
        let caps = self.turn_caps();
        if f.path.len() < 2 || caps.kind == TurnKind::Holonomic {
            return f;
        }
        let first = (f.path[1].1 - f.path[0].1).atan2(f.path[1].0 - f.path[0].0);
        let delta = wrap_angle(first - self.agent.yaw);
        let stop = if caps.kind == TurnKind::Legs { ALPHA_STOP_LEGS_RAD } else { ALPHA_STOP_SKID_RAD };
        if delta.abs() > REVERSE_BEHIND_RAD && f.total_m() <= caps.reverse_max_m && f.turning() <= REVERSE_STRAIGHT_RAD {
            f.reverse = true;
            let how = if caps.kind == TurnKind::Legs { "뒷걸음" } else { "후진" };
            f.note = Some(("reverse", format!("목표가 뒤쪽 {:.1} m: 돌지 않고 {how}", f.total_m())));
        } else if delta.abs() > stop {
            let check = self.pose_check();
            if !self.room_to_turn(&check, self.agent.x, self.agent.z, self.agent.yaw, first, check.h0) {
                self.back_out(&mut f, &check);
            }
        }
        f
    }

    /// One step of path following. Returns the time used.
    pub(crate) fn advance_follow(&mut self, m: &mut MoveState, remaining: f32, events: &mut Vec<TimedEvent>, done: &mut Option<(bool, Option<String>)>) -> f32 {
        let (now, here) = (self.time_s, (self.agent.x, self.agent.z));
        match m.anchor {
            Some((at, _)) if (at.0 - here.0).hypot(at.1 - here.1) >= STUCK_M => m.anchor = Some((here, now)),
            Some((_, since)) if now - since >= STUCK_S => {
                self.agent.speed_mps = 0.0;
                *done = Some((false, Some("blocked".into())));
                return 0.0;
            }
            Some(_) => {}
            None => m.anchor = Some((here, now)),
        }
        if let Some(run) = m.follow.man.as_mut() {
            let mut r = run.clone();
            let span: f32 = r.segs.iter().map(|s| s.length_m()).sum();
            let t = self.advance_run(&mut r, remaining);
            let blocked = r.blocked;
            m.follow.man = (!r.done() || r.wait_left > 0.0).then_some(r);
            if m.follow.man.is_none() {
                m.follow.s = m.follow.project((self.agent.x, self.agent.z), m.follow.s, 1.5 + span);
            }
            m.elapsed_s += t;
            if blocked {
                m.follow.then_replan = false;
                let at = (self.agent.x, self.agent.z, self.agent.yaw);
                return self.way_shut(m, at, t.max(1e-4), events, done);
            }
            if m.follow.man.is_none() && std::mem::take(&mut m.follow.then_replan) {
                self.plan_again(m, events, done);
            }
            return t.max(1e-4);
        }
        let total = m.follow.total_m();
        if m.edges.is_empty() || total <= GOAL_TOL_M {
            *done = Some((true, None));
            return 0.0;
        }
        let k = m.follow.edge_index(m.follow.s).min(m.edges.len() - 1);
        if k != m.idx {
            m.idx = k;
            m.announced_pause = false;
            m.progress_s = 0.0;
        }
        let e = m.edges[k];
        self.agent.terrain_margin = Some(e.margin);
        self.walk.step_lift_m = e.step_m;
        let power = self.drive_spec().power_w as f64;
        if e.pause_s > 0.0 && !m.announced_pause {
            m.announced_pause = true;
            m.pause_left = e.pause_s;
            events.push(self.stamp(WorldEvent::CurbClimbed { height_m: e.step_m }));
            let worker = self.acting_id();
            self.take_step(&e, worker, events);
        }
        if m.pause_left > 0.0 {
            let t = m.pause_left.min(remaining);
            m.pause_left -= t;
            self.agent.speed_mps = 0.0;
            self.add_energy(power * t as f64);
            return t;
        }
        let goal = *m.follow.path.last().expect("a path has an end");
        let pos = (self.agent.x, self.agent.z);
        if let Some((p, reach)) = m.follow.stop_within {
            if (pos.0 - p.0).hypot(pos.1 - p.1) <= reach {
                self.agent.speed_mps = 0.0;
                *done = Some((true, None));
                return 0.0;
            }
        }
        let left = (total - m.follow.s).max(0.0).min(total);
        if left <= GOAL_TOL_M || (pos.0 - goal.0).hypot(pos.1 - goal.1) <= GOAL_TOL_M {
            // The last few centimetres are an evening-up of the books, which a load on board would feel as a jolt.
            if self.load_accel().is_none() && self.pose_check().clear(goal.0, goal.1, self.agent.yaw) {
                (self.agent.x, self.agent.z) = goal;
            }
            m.idx = m.edges.len() - 1;
            *done = Some((true, None));
            return 0.0;
        }
        let caps = self.turn_caps();
        let v_edge = e.horizontal_m / e.move_time_s.max(1e-6);
        if m.follow.on_rails {
            return self.advance_on_rails(m, &e, remaining, left, v_edge, events, done);
        }
        let check = self.pose_check();
        let to_goal = (pos.0 - goal.0).hypot(pos.1 - goal.1);
        let ld = (v_edge * LOOKAHEAD_S).clamp(LOOKAHEAD_MIN_M, LOOKAHEAD_MAX_M).min(if caps.kind == TurnKind::Legs { 0.6 } else { LOOKAHEAD_MAX_M });
        let target = if to_goal < FINAL_APPROACH_M { goal } else { m.follow.point_at(m.follow.s + ld) };
        let (tx, tz) = (target.0 - pos.0, target.1 - pos.1);
        let dist = tx.hypot(tz).max(1e-4);
        let heading = if m.follow.reverse { self.agent.yaw + std::f32::consts::PI } else { self.agent.yaw };
        let alpha = wrap_angle(tz.atan2(tx) - heading);
        let stop_at = if caps.kind == TurnKind::Legs { ALPHA_STOP_LEGS_RAD } else { ALPHA_STOP_SKID_RAD };
        let near_end = to_goal < FINAL_APPROACH_M && alpha.abs() > 0.6;
        let stalled = m.follow.stall_s > STALL_S;
        if caps.kind != TurnKind::Holonomic && (alpha.abs() > stop_at || near_end || stalled) && dist > 0.03 {
            // A loaded base comes to a stand before it turns: it does not stop dead with its load going on.
            if let Some(a) = self.load_accel().filter(|_| self.agent.speed_mps > 0.15) {
                let dt = remaining.min(SUBSTEP_S);
                let v = (self.agent.speed_mps - a * dt).max(0.0);
                let dir = if m.follow.reverse { -1.0 } else { 1.0 };
                let (s, c) = self.agent.yaw.sin_cos();
                let (nx, nz) = (pos.0 + dir * c * v * dt, pos.1 + dir * s * v * dt);
                if self.free_ahead(nx, nz) && check.ok(&self.terrain, nx, nz, self.agent.yaw) {
                    (self.agent.x, self.agent.z) = (nx, nz);
                    self.agent.speed_mps = v;
                    m.elapsed_s += dt;
                    return dt;
                }
            }
            m.follow.stall_s = 0.0;
            m.follow.steer = 0.0;
            if m.follow.turns >= MAX_STOP_TURNS {
                m.follow.on_rails = true;
                return 1e-4;
            }
            let want = tz.atan2(tx) - if m.follow.reverse { std::f32::consts::PI } else { 0.0 };
            if !m.follow.reverse && !near_end && alpha.abs() > stop_at && !self.room_to_turn(&check, pos.0, pos.1, self.agent.yaw, wrap_angle(want), check.h0) {
                // No room to turn round between the obstacles and the way on is behind: back out instead of spinning.
                self.back_out(&mut m.follow, &check);
                return 1e-4;
            }
            m.follow.turns += 1;
            let run = self.turn_run(wrap_angle(want));
            m.follow.note = Some((run.kind, run.label_ko.clone()));
            self.agent.speed_mps = 0.0;
            m.follow.man = Some(run);
            return 1e-4;
        }
        let dt = remaining.min(SUBSTEP_S);
        let mut v = if m.follow.reverse { v_edge.min(caps.reverse_mps) } else { v_edge };
        let mut kappa = 2.0 * alpha.sin() / dist;
        if caps.kind == TurnKind::Skid && caps.min_radius_m > 0.0 {
            kappa = kappa.clamp(-1.0 / caps.min_radius_m, 1.0 / caps.min_radius_m);
        }
        if caps.kind == TurnKind::Skid {
            // The steering lock swings at a finite rate and stops at a limit, as a hydraulic tractor's does, so the
            // curvature ramps instead of jumping to whatever the pursuit law asks for in one tick.
            let l = self.body.support_len_m.max(0.1);
            let want = (kappa * l).atan().clamp(-STEER_MAX_RAD, STEER_MAX_RAD);
            let step = STEER_RATE_RAD_S * dt;
            m.follow.steer += (want - m.follow.steer).clamp(-step, step);
            kappa = m.follow.steer.tan() / l;
        }
        if caps.kind != TurnKind::Holonomic && kappa.abs() > 1e-3 {
            v = v.min(caps.speed_on_radius(1.0 / kappa.abs(), v).max(MIN_CURVE_MPS));
        }
        // With a load on the corner and the stop are taken within what its friction stands (`cargo`), and a step it has
        // seen at a walk. The ramps are the base's own: a body does not reach its speed at once.
        let load = self.load_accel();
        if let Some(a) = load.filter(|_| kappa.abs() > 1e-3) {
            v = v.min((a / kappa.abs()).sqrt().max(MIN_CURVE_MPS));
        }
        if let Some(limit) = self.step_speed_limit() {
            v = v.min(limit);
        }
        // It stops where it comes within reach, which can be well short of the route's end.
        let stop_in = m.follow.stop_within.map_or(left, |(p, reach)| left.min(((pos.0 - p.0).hypot(pos.1 - p.1) - reach).max(0.0)));
        let brake = self.planned_accel(e.slope_deg, self.ground_at(e.to), -1.0).max(0.3).min(load.unwrap_or(f32::INFINITY));
        v = v.min((2.0 * brake * stop_in).sqrt() + 0.05);
        let pull = self.planned_accel(e.slope_deg, self.ground_at(e.to), 1.0).max(0.3).min(load.unwrap_or(f32::INFINITY));
        let was = self.agent.speed_mps;
        v = v.min(was + pull * dt);
        if load.is_some() {
            v = v.max(was - brake * dt);
        }
        let ds = (v * dt).min(left + 0.01);
        let before = (self.agent.x, self.agent.z, self.agent.yaw);
        let (nx, nz, nyaw) = match caps.kind {
            TurnKind::Holonomic => {
                let turn = (caps.pivot_rad_s * dt).min(alpha.abs()) * alpha.signum();
                (pos.0 + ds * tx / dist, pos.1 + ds * tz / dist, self.agent.yaw + turn)
            }
            _ if m.follow.reverse => steer::along(before, &Seg::Drive { dir: -1.0, kappa: -kappa, len: ds }, 1.0),
            _ => steer::along(before, &Seg::Drive { dir: 1.0, kappa, len: ds }, 1.0),
        };
        // The pursuit arc cuts corners: where it would bring the body against something, keep to the route line, and
        // turn toward it no faster than the base turns. Where that will not do either, the way is shut.
        let (nx, nz, nyaw) = if self.free_ahead(nx, nz) && check.ok(&self.terrain, nx, nz, nyaw) {
            (nx, nz, nyaw)
        } else {
            let line = m.follow.point_at(m.follow.s + ds);
            let ahead = m.follow.point_at(m.follow.s + ds + 0.1);
            let want = (ahead.1 - line.1).atan2(ahead.0 - line.0) + if m.follow.reverse { std::f32::consts::PI } else { 0.0 };
            let err = wrap_angle(want - self.agent.yaw);
            let p = step_toward(pos, line, ds);
            let (mut rate, mut ramp) = (caps.pivot_rad_s.max(CREEP_RAD_S), f32::INFINITY);
            if let Some(a) = load {
                let lever = self.load_lever_m();
                (rate, ramp) = (rate.min((a / lever).sqrt()), a / lever);
            }
            m.follow.spin += ((err / dt).clamp(-rate, rate) - m.follow.spin).clamp(-ramp * dt, ramp * dt);
            let yaw = self.agent.yaw + m.follow.spin * dt;
            // The route line is the plan's own: the planner has judged its ground, and only the body is left to check.
            if self.free_ahead(p.0, p.1) && check.clear(p.0, p.1, yaw) {
                (p.0, p.1, yaw)
            } else if err.abs() > 1e-3 && check.clear(pos.0, pos.1, yaw) {
                (pos.0, pos.1, yaw)
            } else {
                return self.way_shut(m, (nx, nz, nyaw), dt, events, done);
            }
        };
        (self.agent.x, self.agent.z, self.agent.yaw) = (nx, nz, wrap_angle(nyaw));
        self.shove_loose();
        m.follow.s = m.follow.project((nx, nz), m.follow.s, ds + 0.3);
        if m.follow.s > m.follow.best_s + 0.02 {
            (m.follow.best_s, m.follow.stall_s) = (m.follow.s, 0.0);
        } else {
            m.follow.stall_s += dt;
        }
        let moved = (nx - pos.0).hypot(nz - pos.1);
        self.agent.speed_mps = moved / dt.max(1e-6);
        m.progress_s += dt;
        m.elapsed_s += dt;
        if !self.body.legged() {
            let props = self.ground_at(e.to).surface().props();
            self.odom_over = noto_core::act::cruise_slip(&self.drive_spec(), props, e.slope_deg).map_or(0.0, |s| s / (1.0 - s).max(0.05));
        }
        let worker = self.acting_id();
        let follow = std::mem::take(&mut m.follow);
        self.watch_steps(&follow, moved, dt, worker, events);
        m.follow = follow;
        if let Some(why) = self.walk_steps(moved, &e, events) {
            *done = Some((false, Some(format!("fall: {why}"))));
            return dt;
        }
        self.add_energy(power * e.energy_factor as f64 * dt as f64);
        dt
    }

    /// What lies loose on the ground gives way to the body: a truck's bumper shoves a log along, nothing that lies about
    /// here weighs enough to stop one, and a walker's shins do the same to what it cannot step over. It goes out by the
    /// nearest side of the hull, if there is ground to go to.
    pub(crate) fn shove_loose(&mut self) {
        let fp = self.footprint();
        let hull = fp.hull(self.agent.x, self.agent.z, self.agent.yaw);
        let v = (-hull.u.1, hull.u.0);
        let held = self.act.as_ref().and_then(|p| p.lifting);
        let mut moved = Vec::new();
        for d in self.drops.iter().filter(|d| d.support.is_ground() && Some(d.id) != held) {
            let size = d.kind.spec().piece_size_m(d.amount);
            if size[1] <= fp.over_m {
                continue;
            }
            let plate = Plate { c: (d.x, d.z), u: (1.0, 0.0), half: (0.5 * size[0], 0.5 * size[2]) };
            if hull.gap(&plate) >= 0.0 {
                continue;
            }
            let reach = 0.5 * size[0].max(size[2]) + 0.01;
            let (rx, rz) = (d.x - hull.c.0, d.z - hull.c.1);
            let (along, across) = (rx * hull.u.0 + rz * hull.u.1, rx * v.0 + rz * v.1);
            let (ea, ec) = (hull.half.0 + reach - along.abs(), hull.half.1 + reach - across.abs());
            let (px, pz) = if ea < ec { (hull.u.0 * ea * along.signum(), hull.u.1 * ea * along.signum()) } else { (v.0 * ec * across.signum(), v.1 * ec * across.signum()) };
            moved.push((d.id, d.x + px, d.z + pz));
        }
        let blocked = if moved.is_empty() { HashSet::new() } else { self.blocked_cells() };
        for (id, x, z) in moved {
            let open = self.terrain.cell_of(x, z).is_some_and(|c| !self.terrain.is_water(c.0, c.1) && !blocked.contains(&c));
            if let Some(d) = self.drops.iter_mut().find(|d| d.id == id).filter(|_| open) {
                (d.x, d.z) = (x, z);
            }
        }
    }

    /// The robot in the first worker's place in the motion code: the first worker, or the sub-worker having its turn.
    pub(crate) fn acting_id(&self) -> u32 {
        self.crew_turn.unwrap_or(1)
    }

    /// The pose ahead (`at`) cannot be taken. The base stops; if it touches, that is a collision. Then it plans again.
    fn way_shut(&mut self, m: &mut MoveState, at: (f32, f32, f32), dt: f32, events: &mut Vec<TimedEvent>, done: &mut Option<(bool, Option<String>)>) -> f32 {
        let v = self.agent.speed_mps;
        let stop = self.pose_check().stopped_by(at.0, at.1, at.2);
        self.agent.speed_mps = 0.0;
        m.follow.steer = 0.0;
        // Against the very thing it was going for, and all but there: it is as near as its body lets it come. From
        // further off a rock by the target is only something to go round.
        if let (Some((p, reach)), Some(Stop::Solid(_, s))) = (m.follow.stop_within, stop) {
            if s.rect.dist(p.0, p.1) <= 0.3 && (self.agent.x - p.0).hypot(self.agent.z - p.1) <= reach + ARRIVED_SHORT_M {
                *done = Some((true, None));
                return dt.max(1e-4);
            }
        }
        match stop {
            Some(Stop::Robot(id)) => {
                // Bodies do not push through each other: wait for it, ask it to stand aside, then go round it.
                if self.crew_turn.is_none() {
                    let dir = self.travel_dir();
                    self.crew_make_way(id, dir);
                }
                m.follow.robot_wait_s += dt;
                if m.follow.robot_wait_s < ROBOT_WAIT_S {
                    return dt.max(1e-4);
                }
                m.follow.robot_wait_s = 0.0;
                // Someone stands at the very place, and it is nearly there: no nearer while they do.
                if m.follow.stop_within.is_some_and(|(p, reach)| (self.agent.x - p.0).hypot(self.agent.z - p.1) <= reach + ROBOT_SHORT_M) {
                    *done = Some((true, None));
                    return dt.max(1e-4);
                }
            }
            Some(Stop::Solid(gap, solid)) if gap <= 0.0 && v >= BUMP_MIN_MPS => {
                if let Some(why) = self.bump_into(&solid, v, events) {
                    *done = Some((false, Some(format!("fall: {why}"))));
                    return dt.max(1e-4);
                }
            }
            _ => {}
        }
        self.replan_move(m, events, done);
        dt.max(1e-4)
    }

    /// What running into `solid` at `speed` does: the shock of it, through the same impact model the benchmarks score,
    /// and the jolt to the load. A fall is what comes back.
    fn bump_into(&mut self, solid: &Solid, speed: f32, events: &mut Vec<TimedEvent>) -> Option<String> {
        let g = self.agent_ground_m();
        let (obstacle, what) = match solid.kind {
            SolidKind::Tree => (Obstacle::Wood, "나무"),
            SolidKind::Rock => (Obstacle::Rock, "바위"),
            SolidKind::Ledge => (Obstacle::Concrete, "단차"),
            SolidKind::Fixture | SolidKind::Water => (Obstacle::Concrete, "벽"),
        };
        let inp = BumpInput {
            body: &self.body,
            pose: self.current_kin_pose(),
            speed_mps: speed,
            heading_deg: 0.0,
            obstacle,
            obstacle_top_m: (solid.y1 - g).max(0.02),
            obstacle_bottom_m: (solid.y0 - g).max(0.0),
            obstacle_speed_mps: 0.0,
            ground: self.ground_here().surface(),
            max_step_m: self.drive_spec().max_step_m,
        };
        let c = collide::bump(&inp);
        let peak_n = c.pulse.as_ref().map_or(0.0, |p| p.peak_n);
        let contact_s = c.pulse.as_ref().map_or(0.02, |p| p.contact_s);
        let fell = c.tip.is_some();
        let worker = self.acting_id();
        let label = format!("{what}에 {speed:.1} m/s로 부딪힘: {peak_n:.0} N · {:.1} g{}", c.a_peak_g, if fell { " → 넘어짐" } else { "" });
        events.push(self.stamp(WorldEvent::Bump { worker, what: what.into(), peak_n, a_g: c.a_peak_g, speed_mps: speed, fell, label: label.clone() }));
        self.jolt_cargo([-speed, 0.0], contact_s, worker, "bump", events);
        if !fell {
            return None;
        }
        self.spill_all(worker, events);
        self.fallen_until = Some(self.time_s + GET_UP_S as f64);
        Some(label)
    }

    /// Plans again round what shut the way; from the second time, from a little further back, as a driver who cannot
    /// make the turn backs up first.
    fn replan_move(&mut self, m: &mut MoveState, events: &mut Vec<TimedEvent>, done: &mut Option<(bool, Option<String>)>) {
        m.replans += 1;
        if m.goal.is_none() || m.replans > MAX_REPLANS {
            *done = Some((false, Some("blocked".into())));
            return;
        }
        if m.replans >= 2 {
            if let Some(run) = self.back_off() {
                m.follow.man = Some(run);
                m.follow.then_replan = true;
                return;
            }
        }
        self.plan_again(m, events, done);
    }

    /// Straight back from where it stands, as far as is clear.
    fn back_off(&self) -> Option<SegRun> {
        let check = self.pose_check();
        let (x0, z0, yaw) = (self.agent.x, self.agent.z, self.agent.yaw);
        let (s, c) = yaw.sin_cos();
        let free = |d: f32| {
            let n = (d / steer::SAMPLE_M).ceil() as usize;
            (1..=n).map(|k| k as f32 * steer::SAMPLE_M).all(|t| self.free_ahead(x0 - c * t, z0 - s * t) && check.clear(x0 - c * t, z0 - s * t, yaw))
        };
        let d = BACK_OFF_M.into_iter().find(|d| free(*d))?;
        let label = format!("길이 막혀 {d:.1} m {}한 뒤 다시 계획", self.back_word());
        Some(self.backed_turn(d, steer::Maneuver { kind: "none", label_ko: String::new(), segs: vec![], time_s: 0.0 }, label))
    }

    /// Plans a new route from here, round the stretch just ahead that shut the way.
    fn plan_again(&mut self, m: &mut MoveState, events: &mut Vec<TimedEvent>, done: &mut Option<(bool, Option<String>)>) {
        let Some(goal) = m.goal else {
            *done = Some((false, Some("blocked".into())));
            return;
        };
        let here = self.agent_cell();
        let mut shut = self.loose_cells(goal);
        shut.extend(self.still_others(self.footprint().half_w_m + 0.15));
        for d in [0.3f32, 0.6, 0.9] {
            let p = m.follow.point_at(m.follow.s + d);
            shut.extend(self.terrain.cell_of(p.0, p.1).filter(|c| *c != here));
        }
        let plan = self.plan_route_avoiding(goal, &shut).or_else(|| self.plan_route(goal));
        let Some(plan) = plan.map(|p| self.fit_plan(p, goal, &shut)) else {
            *done = Some((false, Some("blocked".into())));
            return;
        };
        let stop = m.follow.stop_within;
        let mut follow = self.start_follow(&plan.edges);
        follow.stop_within = stop;
        let (replans, elapsed_s) = (m.replans, m.elapsed_s);
        *m = MoveState { edges: plan.edges, idx: 0, pause_left: 0.0, progress_s: 0.0, announced_pause: false, elapsed_s, follow, anchor: m.anchor, held_s: 0.0, held_by: None, reroutes: m.reroutes, goal: m.goal, replans };
        let worker = self.acting_id();
        events.push(self.stamp(WorldEvent::Hold { worker, other: 0, gap_m: 0.0, label: format!("길이 막혀 돌아가는 길로 다시 계획 ({replans}/{MAX_REPLANS})") }));
    }

    /// Fallback after too many stops to turn: slide along the path line itself, facing along it.
    #[allow(clippy::too_many_arguments)]
    fn advance_on_rails(&mut self, m: &mut MoveState, e: &EdgePlan, remaining: f32, left: f32, v_edge: f32, events: &mut Vec<TimedEvent>, done: &mut Option<(bool, Option<String>)>) -> f32 {
        let dt = remaining.min(SUBSTEP_S);
        let load = self.load_accel();
        let pull = self.planned_accel(e.slope_deg, self.ground_at(e.to), 1.0).max(0.3).min(load.unwrap_or(f32::INFINITY));
        let brake = self.planned_accel(e.slope_deg, self.ground_at(e.to), -1.0).max(0.3).min(load.unwrap_or(f32::INFINITY));
        let v = v_edge.min((2.0 * brake * left).sqrt() + 0.05).min(self.agent.speed_mps + pull * dt);
        let ds = (v * dt).min(left);
        let pos = (self.agent.x, self.agent.z);
        let line = m.follow.point_at(m.follow.s + ds);
        let ahead = m.follow.point_at(m.follow.s + ds + 0.1);
        let p = step_toward(pos, line, ds);
        let flip = if m.follow.reverse { std::f32::consts::PI } else { 0.0 };
        let rate = self.turn_caps().pivot_rad_s.max(CREEP_RAD_S).min(load.map_or(f32::INFINITY, |a| (a / self.load_lever_m()).sqrt()));
        let yaw = if (ahead.0 - p.0).hypot(ahead.1 - p.1) > 1e-4 { self.agent.yaw + wrap_angle((ahead.1 - p.1).atan2(ahead.0 - p.0) + flip - self.agent.yaw).clamp(-rate * dt, rate * dt) } else { self.agent.yaw };
        if !self.pose_check().ok(&self.terrain, p.0, p.1, yaw) {
            return self.way_shut(m, (p.0, p.1, yaw), dt, events, done);
        }
        self.agent.yaw = wrap_angle(yaw);
        (self.agent.x, self.agent.z) = p;
        m.follow.s = m.follow.project(p, m.follow.s, ds + 0.3);
        let moved = (p.0 - pos.0).hypot(p.1 - pos.1);
        self.agent.speed_mps = moved / dt.max(1e-6);
        m.elapsed_s += dt;
        if let Some(why) = self.walk_steps(moved, e, events) {
            *done = Some((false, Some(format!("fall: {why}"))));
        }
        self.add_energy(self.drive_spec().power_w as f64 * e.energy_factor as f64 * dt as f64);
        dt
    }

    fn free_ahead(&self, x: f32, z: f32) -> bool {
        match self.terrain.cell_of(x, z) {
            Some(c) => !self.terrain.is_water(c.0, c.1) && !self.blocked_cells().contains(&c),
            None => false,
        }
    }

    pub fn motion_view(&self) -> Option<MotionView> {
        match &self.active {
            Some((_, Active::Move(m))) => {
                if let Some(r) = &m.follow.man {
                    return Some(MotionView { kind: r.kind.into(), label_ko: r.label_ko.clone(), reversing: r.reversing() });
                }
                if m.follow.reverse {
                    return Some(MotionView { kind: "reverse".into(), label_ko: m.follow.note.as_ref().map(|n| n.1.clone()).unwrap_or_default(), reversing: true });
                }
                (self.agent.speed_mps > 0.0).then(|| MotionView { kind: "follow".into(), label_ko: String::new(), reversing: false })
            }
            Some((_, Active::Turn { run, .. })) => Some(MotionView { kind: run.kind.into(), label_ko: run.label_ko.clone(), reversing: run.reversing() }),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::TerrainKind;
    use noto_core::{BaseKind, FingerCount, NotoConfig};

    fn world(base: BaseKind) -> World {
        let mut hand = NotoConfig::preset(FingerCount::Five);
        hand.base = base;
        let mut w = World::new(WorldConfig { seed: 11, cells: 64, noto: hand, body: BodyPlan::default(), terrain: TerrainKind::Open, parcels: Some(0), ..WorldConfig::default() }).unwrap();
        w.nodes.clear();
        w.drops.clear();
        let (cx, cz) = w.agent_cell();
        (w.agent.x, w.agent.z) = cell_center(cx, cz);
        w.agent.yaw = 0.0;
        w
    }

    fn run(w: &mut World, max_ticks: usize) -> Vec<(f32, f32, f32, f32, bool)> {
        let mut trace = Vec::new();
        for _ in 0..max_ticks {
            w.tick();
            let rev = w.motion_view().is_some_and(|m| m.reversing);
            trace.push((w.agent.x, w.agent.z, w.agent.yaw, w.agent.speed_mps, rev));
            if !w.busy() {
                break;
            }
        }
        trace
    }

    #[test]
    fn a_wheeled_base_rounds_a_corner_without_stopping() {
        let mut w = world(BaseKind::Wheel);
        let (x0, z0) = (w.agent.x, w.agent.z);
        let goal = (x0 + 3.0, z0 + 3.0);
        w.command(Action::MoveTo { x: goal.0, z: goal.1, reach_m: 0.0 }).unwrap();
        let tr = run(&mut w, 400);
        assert!(!w.busy(), "arrived");
        assert!((w.agent.x - goal.0).hypot(w.agent.z - goal.1) < 0.3);
        let moving: Vec<&(f32, f32, f32, f32, bool)> = tr.iter().skip(3).take(tr.len().saturating_sub(8)).collect();
        assert!(moving.iter().all(|t| t.3 > 0.05), "it never halts mid-route: {:?}", moving.iter().map(|t| t.3).collect::<Vec<_>>());
        let jumps = tr.windows(2).map(|p| wrap_angle(p[1].2 - p[0].2).abs()).fold(0.0f32, f32::max);
        assert!(jumps < 0.35, "the heading turns smoothly, at most {jumps} rad a tick");
    }

    #[test]
    fn the_steering_lock_swings_at_a_finite_rate() {
        let mut w = world(BaseKind::Track);
        let (x0, z0) = (w.agent.x, w.agent.z);
        w.command(Action::MoveTo { x: x0 + 3.0, z: z0 + 3.0, reach_m: 0.0 }).unwrap();
        let mut steer = 0.0f32;
        for _ in 0..400 {
            w.tick();
            if let Some(Active::Move(m)) = w.active.as_ref().map(|a| &a.1) {
                assert!((m.follow.steer - steer).abs() < STEER_RATE_RAD_S * 0.2, "the lock jumped {} rad", m.follow.steer - steer);
                steer = m.follow.steer;
            }
            if !w.busy() {
                break;
            }
        }
        assert!(!w.busy());
    }

    #[test]
    fn a_goal_close_behind_is_reached_in_reverse() {
        for (base, back_m) in [(BaseKind::Wheel, 1.5), (BaseKind::Track, 1.5), (BaseKind::HumanFoot, 1.0)] {
            let mut w = world(base);
            let (x0, z0) = (w.agent.x, w.agent.z);
            w.command(Action::MoveTo { x: x0 - back_m, z: z0, reach_m: 0.0 }).unwrap();
            let tr = run(&mut w, 300);
            assert!(!w.busy(), "{base:?} arrived");
            assert!(tr.iter().any(|t| t.4), "{base:?} backs up");
            assert!(wrap_angle(w.agent.yaw).abs() < 0.2, "{base:?} still faces the way it did: {}", w.agent.yaw);
        }
    }

    #[test]
    fn a_track_on_soft_ground_turns_round_with_a_k_turn() {
        let mut w = world(BaseKind::Track);
        let (cx, cz) = w.agent_cell();
        for dz in -4i32..=4 {
            for dx in -4i32..=4 {
                let (x, z) = ((cx as i32 + dx) as usize, (cz as i32 + dz) as usize);
                w.terrain.set_ground(x, z, crate::kinds::Ground::Dirt);
            }
        }
        assert!(w.turn_caps().prefers_kturn(true), "{:?}", w.turn_caps());
        w.command(Action::Turn { yaw_deg: 150.0 }).unwrap();
        assert_eq!(w.motion_view().map(|m| m.kind), Some("kturn".into()));
        let tr = run(&mut w, 400);
        assert!(!w.busy());
        assert!(tr.iter().any(|t| t.4) && tr.iter().any(|t| !t.4 && t.3 > 0.01), "forward and reverse legs");
        assert!((wrap_angle(w.agent.yaw) - 150f32.to_radians()).abs() < 0.05, "{}", w.agent.yaw.to_degrees());
        let mut walker = world(BaseKind::HumanFoot);
        walker.command(Action::Turn { yaw_deg: 150.0 }).unwrap();
        assert_eq!(walker.motion_view().map(|m| m.kind), Some("pivot".into()), "legs step round where they stand");
    }

    fn plant(w: &mut World, kind: crate::kinds::NodeKind, c: Cell) {
        let (x, z) = cell_center(c.0, c.1);
        let id = 900 + w.nodes.len() as u32;
        w.nodes.push(Node { id, kind, x, z, hp: 3.0, max_hp: 3.0 });
    }

    /// The least room the body's outline had at any tick of the run (m; 0 is touching or inside something).
    fn drive_room(w: &mut World, max_ticks: usize) -> (f32, Vec<(f32, f32, f32, f32, bool, String)>) {
        let mut worst = f32::INFINITY;
        let mut trace = Vec::new();
        for _ in 0..max_ticks {
            w.tick();
            let map = w.clear_map();
            worst = worst.min(w.footprint().room(&map, w.agent.x, w.agent.z, w.agent.yaw));
            let m = w.motion_view();
            trace.push((w.agent.x, w.agent.z, w.agent.yaw, w.agent.speed_mps, m.as_ref().is_some_and(|m| m.reversing), m.map(|m| m.kind).unwrap_or_default()));
            if !w.busy() {
                break;
            }
        }
        (worst, trace)
    }

    /// A pocket one cell wide round the base: rocks left and right of it and across its end, open behind it. Returns
    /// where the pocket opens (x).
    fn pocket(w: &mut World) -> f32 {
        use crate::kinds::NodeKind;
        let (cx, cz) = w.agent_cell();
        for x in cx - 3..=cx + 2 {
            plant(w, NodeKind::Boulder, (x, cz - 1));
            plant(w, NodeKind::Boulder, (x, cz + 1));
        }
        plant(w, NodeKind::Boulder, (cx + 2, cz));
        cell_center(cx - 3, cz).0
    }

    #[test]
    fn in_a_dead_end_between_trees_the_base_backs_out_instead_of_spinning() {
        for base in [BaseKind::Wheel, BaseKind::Track, BaseKind::HumanFoot] {
            let mut w = world(base);
            let pocket_end_x = pocket(&mut w);
            let (x0, z0) = (w.agent.x, w.agent.z);
            let fp = w.footprint();
            assert!(fp.room(&w.clear_map(), x0, z0, 0.0) > 0.0 && w.clear_map().at(x0, z0) < fp.sweep_m(), "{base:?}: it fits, but cannot turn round in it");
            let goal = (x0 - 3.5, z0 + 2.0);
            w.command(Action::MoveTo { x: goal.0, z: goal.1, reach_m: 0.0 }).unwrap();
            assert_eq!(w.motion_view().map(|m| m.kind), Some("reverse".into()), "{base:?} backs out at once");
            let (worst, tr) = drive_room(&mut w, 900);
            assert!(!w.busy(), "{base:?} got out");
            assert!((w.agent.x - goal.0).hypot(w.agent.z - goal.1) < 0.3, "{base:?} arrived: {:?}", (w.agent.x, w.agent.z));
            let inside: Vec<_> = tr.iter().filter(|t| t.0 > pocket_end_x).collect();
            assert!(inside.iter().all(|t| t.4 || t.3 < 1e-3), "{base:?} only reverses while in the pocket");
            assert!(inside.iter().all(|t| wrap_angle(t.2).abs() < 0.35), "{base:?} does not turn round in the pocket: {:?}", inside.iter().map(|t| t.2).collect::<Vec<_>>());
            assert!(inside.iter().all(|t| t.5 != "pivot" && t.5 != "kturn"));
            assert!(tr.iter().any(|t| !t.4 && t.3 > 0.05), "{base:?} drives forward again once out");
            assert!(worst > 0.0, "{base:?} never touches a tree: {worst}");
        }
    }

    #[test]
    fn a_turn_asked_for_in_a_dead_end_backs_out_of_it_first() {
        // The search turns the base round in steps to look about; in a pocket it must not spin against the trees.
        for (base, deg) in [(BaseKind::Wheel, 45.0f32), (BaseKind::Track, 180.0), (BaseKind::HumanFoot, 180.0), (BaseKind::Wheel, -120.0)] {
            let mut w = world(base);
            let pocket_end_x = pocket(&mut w);
            w.command(Action::Turn { yaw_deg: deg }).unwrap();
            let m = w.motion_view().unwrap();
            assert!(m.kind == "reverse" && m.reversing, "{base:?} {deg}°: it backs out first: {m:?}");
            let (worst, tr) = drive_room(&mut w, 600);
            assert!(!w.busy(), "{base:?} {deg}° turned");
            assert!((wrap_angle(w.agent.yaw - deg.to_radians())).abs() < 0.06, "{base:?} {deg}°: faces {:.0}°", w.agent.yaw.to_degrees());
            let inside: Vec<_> = tr.iter().filter(|t| t.0 > pocket_end_x).collect();
            assert!(inside.iter().all(|t| wrap_angle(t.2).abs() < 0.05), "{base:?} {deg}°: no turning inside the pocket");
            assert!(worst > 0.0, "{base:?} {deg}°: never touches a tree: {worst}");
            let mut free = world(base);
            free.command(Action::Turn { yaw_deg: deg }).unwrap();
            assert_ne!(free.motion_view().map(|m| m.kind), Some("reverse".into()), "{base:?} {deg}°: in the open it turns where it stands");
        }
    }

    #[test]
    fn the_body_keeps_clear_of_trees_and_rocks_it_passes_between() {
        use crate::kinds::NodeKind;
        for base in [BaseKind::Wheel, BaseKind::Track, BaseKind::HumanFoot] {
            let mut w = world(base);
            let (cx, cz) = w.agent_cell();
            // A grove the straight line runs through: trees and boulders a cell apart, staggered, with roomier lanes.
            for (k, dx) in (2..12).step_by(2).enumerate() {
                for dz in [-4i32, -2, 0, 2, 4] {
                    let kind = if (dz / 2 + k as i32) % 2 == 0 { NodeKind::Tree } else { NodeKind::Boulder };
                    plant(&mut w, kind, (cx + dx, (cz as i32 + dz + (k as i32 % 2)) as usize));
                }
            }
            let (x0, z0) = (w.agent.x, w.agent.z);
            let goal = cell_center(cx + 14, cz);
            let fp = w.footprint();
            // What the old planner did: cell centres past the trunks, the body's outline into them.
            let blocked = w.blocked_cells();
            let old = nav::plan(&w.drive_spec(), &w.terrain, &|a, b| blocked.contains(&(a, b)), (cx, cz), Goal::Cell(w.terrain.cell_of(goal.0, goal.1).unwrap())).unwrap();
            let map = w.clear_map();
            let old_room = old.cells.windows(2).map(|p| {
                let (a, b) = (cell_center(p[0].0, p[0].1), cell_center(p[1].0, p[1].1));
                fp.room(&map, b.0, b.1, (b.1 - a.1).atan2(b.0 - a.0))
            }).fold(f32::INFINITY, f32::min);
            let new = w.plan_route(Goal::Cell(w.terrain.cell_of(goal.0, goal.1).unwrap())).unwrap();
            let new_room = new.cells.iter().skip(1).map(|c| map.best_in(*c)).fold(f32::INFINITY, f32::min);
            assert!(new_room >= fp.half_w_m + clear::TIGHT_GAP_M, "{base:?}: every cell of the route has room for the body: {new_room}");
            w.command(Action::MoveTo { x: goal.0, z: goal.1, reach_m: 0.0 }).unwrap();
            let (worst, _) = drive_room(&mut w, 1500);
            assert!(!w.busy() && (w.agent.x - goal.0).hypot(w.agent.z - goal.1) < 0.3, "{base:?} arrived");
            println!("[CLEAR] {base:?}: old route outline room {old_room:.3} m, driven {worst:.3} m (from {:.1},{:.1})", x0, z0);
            assert!(old_room < clear::MIN_GAP_M, "{base:?}: the old route brushed the grove ({old_room})");
            assert!(worst >= clear::TOUCH_GAP_M - 1e-3, "{base:?}: the body kept clear all the way: {worst}");
        }
    }
}
