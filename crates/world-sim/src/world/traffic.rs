//! Robots sharing one map: who stands where (the first worker, its rider when it has stepped off, the sub-workers),
//! what each one's eye sees ahead on its way, and how they keep clear of each other. Routes go round the robots that
//! stand still; a robot on the move stops for a robot its camera sees in its way — inside the camera's horizontal
//! field of view, within its stopping distance, not hidden behind a tree or a boulder — and when the way does not
//! clear it routes round that robot. A humanoid gives way sooner than a vehicle, and both give way to the first
//! worker, so two robots meeting head on do not wait for each other forever.
//!
//! A lift truck's rider fells a node on foot (`Rider`): the truck stops short of the node, the rider steps down, walks
//! over, swings at it, walks back and climbs on again; the head camera goes with the rider.

use std::collections::HashSet;

use serde::Serialize;

use super::*;

/// Half the head camera's horizontal field of view, which the way-ahead check looks over (deg) [직접: 70° camera].
pub const SIGHT_HALF_FOV_DEG: f32 = 35.0;
/// The farthest the way-ahead check looks (m).
pub const SIGHT_RANGE_M: f32 = 6.0;
/// How long a robot the eye saw stays in mind after it left the view (s): until the next off-the-ball glance
/// (`gaze::SCAN_PERIOD_S`) looks again, where it was going when seen.
pub const NOTICE_KEEP_S: f32 = noto_core::act::gaze::SCAN_PERIOD_S;
/// Clearance kept between two robots' outlines when one stops for the other (m) [추정: a hand's breadth beyond the
/// bodies], plus the distance it covers in a second at its speed (reaction and braking).
pub const KEEP_CLEAR_M: f32 = 0.45;
/// Seconds a robot waits for its way to clear before it routes round: humanoids give way first.
pub const YIELD_HUMANOID_S: f32 = 1.5;
pub const YIELD_VEHICLE_S: f32 = 3.0;
/// A rider getting off and on a lift truck (s), and walking (m/s) [추정: three points of contact take 3–4 s; a
/// careful walk on site 0.8 m/s].
pub const DISMOUNT_S: f32 = 3.0;
pub const REMOUNT_S: f32 = 3.0;
pub const RIDER_WALK_MPS: f32 = 0.8;
/// The radius a standing rider keeps others at (m).
pub const RIDER_R_M: f32 = 0.28;
/// How far from a node a lift truck stops for its rider to walk over and fell it, and the farthest it may (m): the
/// tines' tips stay clear of the trunk.
pub const RIDER_STOP_M: f32 = 1.8;
pub const RIDER_REACH_M: f32 = 3.5;

/// The distance `RIDER_STOP_M` for this truck: its tine tips and a margin, if they reach further.
pub fn rider_stop_m(body: &noto_core::balance::Body) -> f32 {
    RIDER_STOP_M.max(crate::clear::Footprint::of(body).front_m + 0.3)
}

/// The distance `RIDER_REACH_M` for this truck.
pub fn rider_reach_m(body: &noto_core::balance::Body) -> f32 {
    RIDER_REACH_M + rider_stop_m(body) - RIDER_STOP_M
}

/// The person who rides a truck, on foot.
pub fn rider_body(body: &noto_core::balance::Body) -> noto_core::balance::Body {
    use noto_core::{BaseKind, BodyPlan, Morphology};
    noto_core::balance::Body::new(&BodyPlan { morph: Morphology::humanoid(BaseKind::HumanFoot), ..body.plan }, BaseKind::HumanFoot, body.fingers)
}

/// A standing rider's eye height (m).
pub fn rider_eye_m(body: &noto_core::balance::Body) -> f32 {
    let on_foot = rider_body(body);
    noto_core::kin::head_camera(&on_foot, &noto_core::kin::KinPose::stand(&on_foot, 0.0)).pos[2]
}
/// The id the first worker's dismounted rider goes by among the robots.
pub const RIDER_ID: u32 = 1000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RiderPhase {
    Dismount,
    WalkOut,
    Mine,
    WalkBack,
    Remount,
}

impl RiderPhase {
    pub fn label_ko(self) -> &'static str {
        match self {
            RiderPhase::Dismount => "하차",
            RiderPhase::WalkOut => "걸어가기",
            RiderPhase::Mine => "채취",
            RiderPhase::WalkBack => "돌아오기",
            RiderPhase::Remount => "재탑승",
        }
    }
}

/// A lift truck's rider on foot, felling a node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rider {
    pub x: f32,
    pub z: f32,
    pub yaw: f32,
    pub phase: RiderPhase,
    pub t: f32,
    /// Where it steps down and back on (beside the seat), where it stands to swing, and the node.
    pub seat: (f32, f32),
    pub stand: (f32, f32),
    pub node: (f32, f32),
    pub failed: bool,
    /// A step aside round a robot standing in its way, walked before the rest; and how long it has waited.
    pub via: Option<(f32, f32)>,
    pub held_s: f32,
}

impl Rider {
    /// The rider of a truck at (x, z) facing `yaw`, about to step down and fell the node at `node`.
    pub fn start(x: f32, z: f32, yaw: f32, node: (f32, f32)) -> Rider {
        let (s, c) = yaw.sin_cos();
        // Steps down on the side nearer the node.
        let side = if (node.0 - x) * -s + (node.1 - z) * c >= 0.0 { 1.0 } else { -1.0 };
        let seat = (x - s * 0.7 * side, z + c * 0.7 * side);
        let (dx, dz) = (node.0 - seat.0, node.1 - seat.1);
        let d = dx.hypot(dz).max(1e-3);
        let keep = (MINE_REACH_M - 0.25).min(d);
        let stand = (node.0 - dx / d * keep, node.1 - dz / d * keep);
        Rider { x, z, yaw, phase: RiderPhase::Dismount, t: 0.0, seat, stand, node, failed: false, via: None, held_s: 0.0 }
    }

    /// On foot, away from the seat.
    pub fn off(&self) -> bool {
        matches!(self.phase, RiderPhase::WalkOut | RiderPhase::Mine | RiderPhase::WalkBack)
    }

    /// Walks toward `to` for `dt`; true once there.
    pub fn walk_toward(&mut self, to: (f32, f32), dt: f32) -> bool {
        let (dx, dz) = (to.0 - self.x, to.1 - self.z);
        let d = dx.hypot(dz);
        if d < 1e-3 {
            return true;
        }
        let step = (RIDER_WALK_MPS * dt).min(d);
        self.yaw = dz.atan2(dx);
        self.x += dx / d * step;
        self.z += dz / d * step;
        step >= d - 1e-4
    }

    /// The rider's body as solids in the world, standing (on foot) at its place: the same person as on the seat.
    pub fn boxes(&self, ground_m: f32, body: &noto_core::balance::Body) -> Vec<crate::render::Obb> {
        use noto_core::kin::{self, KinLoads, KinPose};
        let on_foot = rider_body(body);
        let links = kin::solve(&on_foot, &KinPose::stand(&on_foot, 0.0), &KinLoads::default()).boxes;
        let parts = noto_core::rig::dress(&on_foot, &links, &noto_core::rig::Palette { accent: [236, 190, 40], ..Default::default() });
        let (s, c) = self.yaw.sin_cos();
        let at = |p: [f32; 3]| [self.x + c * p[0] + s * p[1], ground_m + p[2], self.z + s * p[0] - c * p[1]];
        let dir = |v: [f32; 3]| [c * v[0] + s * v[1], v[2], s * v[0] - c * v[1]];
        parts.iter().map(|p| crate::render::Obb::placed(p, at, dir)).collect()
    }
}

/// A robot as the others see it: its outline on the ground. A lift truck is long and narrow, and a disc round it
/// would cover the bay it waits at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RobotOutline {
    pub id: u32,
    pub x: f32,
    pub z: f32,
    pub yaw: f32,
    pub front: f32,
    pub rear: f32,
    pub half_w: f32,
    pub moving: bool,
    pub humanoid: bool,
    /// Its velocity (m/s along x, z): what an eye that sees it moving for a fixation can tell.
    pub vel: (f32, f32),
}

impl RobotOutline {
    /// A round outline, for someone on foot.
    fn round(id: u32, x: f32, z: f32, r: f32, moving: bool, vel: (f32, f32)) -> RobotOutline {
        RobotOutline { id, x, z, yaw: 0.0, front: r, rear: r, half_w: r, moving, humanoid: true, vel }
    }

    fn of(id: u32, x: f32, z: f32, yaw: f32, fp: crate::clear::Footprint, moving: bool, humanoid: bool, speed: f32) -> RobotOutline {
        RobotOutline { id, x, z, yaw, front: fp.front_m, rear: fp.rear_m, half_w: fp.half_w_m, moving, humanoid, vel: (speed * yaw.cos(), speed * yaw.sin()) }
    }

    /// Whether a point lies within `pad` of the outline.
    pub fn near(&self, p: (f32, f32), pad: f32) -> bool {
        let (s, c) = self.yaw.sin_cos();
        let (dx, dz) = (p.0 - self.x, p.1 - self.z);
        let (along, across) = (dx * c + dz * s, dz * c - dx * s);
        along <= self.front + pad && along >= -self.rear - pad && across.abs() <= self.half_w + pad
    }

    /// The farthest the outline reaches from the robot's centre.
    pub fn reach_m(&self) -> f32 {
        self.front.max(self.rear).hypot(self.half_w)
    }
}

/// The robot looking: where it is, which way it goes and how fast, how wide its way is (half its width) and how far
/// its body reaches ahead along it, its eye height, and how far its head has turned off the way (`gaze`, rad, to its
/// left positive as `KinPose::neck_pan_deg`: an off-the-ball glance). A body's left lies at its yaw − 90° on the map,
/// so the eye looks along `dir − gaze` (`Looker::eye_dir`), as the head camera's view does (`World::head_view`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Looker {
    pub id: u32,
    pub at: (f32, f32),
    pub dir: f32,
    pub half_w: f32,
    pub front: f32,
    pub speed: f32,
    pub eye_m: f32,
    pub gaze: f32,
}

impl Looker {
    /// Where the eye points on the map (rad).
    pub fn eye_dir(&self) -> f32 {
        self.dir - self.gaze
    }
}

/// A robot an eye saw: where and how fast it was going, and when.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Noticed {
    pub id: u32,
    pub at: (f32, f32),
    pub vel: (f32, f32),
    pub time_s: f64,
}

impl Noticed {
    /// Where it should be at `now`, going on as it went.
    pub fn at_time(&self, now: f64) -> (f32, f32) {
        let age = (now - self.time_s).max(0.0) as f32;
        (self.at.0 + self.vel.0 * age, self.at.1 + self.vel.1 * age)
    }
}

/// Keeps in `mind` what the eye sees `now` and what it saw within `NOTICE_KEEP_S`.
pub fn remember(mind: &mut Vec<Noticed>, seen: Vec<Noticed>, now: f64) {
    mind.retain(|n| now - n.time_s <= NOTICE_KEEP_S as f64 && !seen.iter().any(|s| s.id == n.id));
    mind.extend(seen);
}

/// Whether robot `o` is robot `me` itself or its rider on foot.
fn owns(me: u32, o: u32) -> bool {
    o == me || (me == 1 && o == RIDER_ID) || (me == RIDER_ID && o == 1) || (me > 1 && me < RIDER_ID && o == RIDER_ID + me) || (me > RIDER_ID && o == me - RIDER_ID)
}

/// Whether a robot that has stopped for `other` routes round it (the other waits): the lower rank does, and of two
/// of one rank the higher id — so of two meeting head on, exactly one goes round.
pub fn gives_way(me: u32, me_humanoid: bool, other: u32, other_humanoid: bool) -> bool {
    let (a, b) = (rank(me, me_humanoid), rank(other, other_humanoid));
    a > b || (a == b && me > other)
}

/// Which of two robots gives way: the lower rank waits first (the first worker 0, vehicles 1, humanoids 2).
pub fn rank(id: u32, humanoid: bool) -> u8 {
    match id {
        1 | RIDER_ID => 0,
        _ if humanoid => 2,
        _ => 1,
    }
}

impl World {
    /// The radius the first worker keeps others at: its outline's half width and half its length.
    pub fn agent_radius_m(&self) -> f32 {
        let fp = self.footprint();
        fp.half_w_m.max(0.5 * (fp.front_m + fp.rear_m)).max(0.3)
    }

    /// Every robot on the map: the first worker (and its rider while on foot) and the sub-workers.
    pub fn robot_outlines(&self) -> Vec<RobotOutline> {
        // On the move: going somewhere, even while stopped for another robot (so that two which stopped for each
        // other still tell who gives way).
        let agent_moving = matches!(self.active, Some((_, Active::Move(_))));
        let rider_vel = |r: &Rider| if matches!(r.phase, RiderPhase::WalkOut | RiderPhase::WalkBack) { (RIDER_WALK_MPS * r.yaw.cos(), RIDER_WALK_MPS * r.yaw.sin()) } else { (0.0, 0.0) };
        let mut v = vec![RobotOutline::of(1, self.agent.x, self.agent.z, self.agent.yaw, self.footprint(), agent_moving, self.body.legged(), self.agent.speed_mps)];
        if let Some(r) = self.rider.filter(|r| r.off()) {
            v.push(RobotOutline::round(RIDER_ID, r.x, r.z, RIDER_R_M, !matches!(r.phase, RiderPhase::Mine), rider_vel(&r)));
        }
        for w in &self.crew {
            v.push(RobotOutline::of(w.id, w.x, w.z, w.yaw, w.footprint, matches!(w.job, crate::crew::Job::Go { .. }), w.form == crate::crew::Form::Humanoid, w.speed_mps));
            if let Some(r) = w.rider.filter(|r| r.off()) {
                v.push(RobotOutline::round(RIDER_ID + w.id, r.x, r.z, RIDER_R_M, !matches!(r.phase, RiderPhase::Mine), rider_vel(&r)));
            }
        }
        v
    }

    /// How far robot `id` has its head turned off its way now for an off-the-ball glance (rad, left positive): on the
    /// move with nothing in its hands, the sub-workers when `scan_crew`, the first worker when `scan_first`. Each robot
    /// keeps its own beat, so that a crew does not look the same way at once.
    pub fn glance_rad(&self, id: u32) -> f32 {
        let (body, moving, empty) = if id == 1 && self.crew_turn.is_none() {
            (&self.body, self.scan_first && matches!(self.active, Some((_, Active::Move(_)))) && self.agent.speed_mps > 0.05 && self.act.is_none(), self.carried_kg() <= AMOUNT_EPS)
        } else if let Some(w) = self.crew.iter().find(|w| w.id == id) {
            (&w.body, self.scan_crew && matches!(w.job, crate::crew::Job::Go { .. }) && w.speed_mps > 0.05, w.carried_kg() <= AMOUNT_EPS)
        } else {
            return 0.0;
        };
        if !moving || !empty {
            return 0.0;
        }
        let beat = (id as f32 * 0.61).rem_euclid(noto_core::act::gaze::SCAN_PERIOD_S);
        noto_core::act::gaze::scan_pan_deg(body, self.time_s as f32 + beat).to_radians()
    }

    /// The robots the eye of `me` sees now: inside its camera's horizontal view about where its head points (the way
    /// plus its glance), within sight, not behind a node.
    pub fn robots_seen(&self, me: &Looker) -> Vec<Noticed> {
        let half = SIGHT_HALF_FOV_DEG.to_radians();
        self.robot_outlines()
            .into_iter()
            .filter(|o| !owns(me.id, o.id))
            .filter(|o| {
                let (dx, dz) = (o.x - me.at.0, o.z - me.at.1);
                wrap_angle(dz.atan2(dx) - me.eye_dir()).abs() <= half && dx.hypot(dz) <= SIGHT_RANGE_M && self.line_of_sight(me.at, (o.x, o.z), me.eye_m.min(1.2))
            })
            .map(|o| Noticed { id: o.id, at: (o.x, o.z), vel: o.vel, time_s: self.time_s })
            .collect()
    }

    /// Whether the straight line from `a` to `b` (both at `h` above the ground) passes clear of every node.
    pub fn line_of_sight(&self, a: (f32, f32), b: (f32, f32), h: f32) -> bool {
        let (dx, dz) = (b.0 - a.0, b.1 - a.1);
        let len = dx.hypot(dz);
        if len < 1e-3 {
            return true;
        }
        let base = self.surface_m(a.0, a.1) + h;
        self.nodes.iter().all(|n| {
            let r = if n.kind == crate::kinds::NodeKind::Tree { 0.35 } else { 0.55 };
            let t = (((n.x - a.0) * dx + (n.z - a.1) * dz) / (len * len)).clamp(0.0, 1.0);
            let (px, pz) = (a.0 + dx * t, a.1 + dz * t);
            // A node at either end is what is looked at, not what hides it.
            let at_end = t <= 1e-3 || t >= 1.0 - 1e-3;
            at_end || (n.x - px).hypot(n.z - pz) > r || self.surface_m(n.x, n.z) + 1.0 < base - 0.6
        })
    }

    /// The nearest robot the eye of `me` sees in its way: in the camera's horizontal field of view about where its head
    /// points (or close enough to touch), not hidden behind a node — or one it saw a moment ago (`mind`, at most
    /// `NOTICE_KEEP_S` old) where it was going then — and standing on the stretch of its path (`path`, the points it
    /// goes through next) it covers before it could stop: its body's reach ahead and the stopping distance. (its id,
    /// the path length to it in m). A robot the path passes clear of is passed, not waited for.
    pub fn robot_in_way(&self, me: &Looker, path: &[(f32, f32)], mind: &[Noticed]) -> Option<(u32, f32)> {
        let look = me.front + KEEP_CLEAR_M + me.speed.max(0.3) * 1.0;
        let half = SIGHT_HALF_FOV_DEG.to_radians();
        let own = |o: u32| owns(me.id, o);
        // The path ahead every 0.15 m, with the distance along it.
        let mut samples: Vec<(f32, f32, f32)> = Vec::new();
        let (mut prev, mut run) = (me.at, 0.0f32);
        for &p in path.iter().chain(std::iter::once(&(me.at.0 + me.dir.cos() * 1e-3, me.at.1 + me.dir.sin() * 1e-3)).filter(|_| path.is_empty())) {
            let seg = (p.0 - prev.0).hypot(p.1 - prev.1);
            let n = (seg / 0.15).ceil().max(1.0) as usize;
            for k in 1..=n {
                let u = k as f32 / n as f32;
                let s = run + seg * u;
                if s > look {
                    break;
                }
                samples.push((prev.0 + (p.0 - prev.0) * u, prev.1 + (p.1 - prev.1) * u, s));
            }
            run += seg;
            prev = p;
            if run > look {
                break;
            }
        }
        if path.is_empty() || samples.is_empty() {
            // Standing, or no path: only a robot pressing on it.
            samples = vec![(me.at.0, me.at.1, 0.0)];
        }
        self.robot_outlines()
            .into_iter()
            .filter(|o| !own(o.id))
            .filter_map(|o| {
                let (dx, dz) = (o.x - me.at.0, o.z - me.at.1);
                let d = dx.hypot(dz);
                let ahead = wrap_angle(dz.atan2(dx) - me.dir).abs();
                let looked = wrap_angle(dz.atan2(dx) - me.eye_dir()).abs();
                let touching = o.near(me.at, me.half_w + 0.05);
                let seen = ((looked <= half && d <= SIGHT_RANGE_M) || touching) && self.line_of_sight(me.at, (o.x, o.z), me.eye_m.min(1.2));
                // Out of view now but seen a moment ago: it is where it was going then, as far as this robot knows.
                let o = match mind.iter().find(|n| n.id == o.id).filter(|n| !seen && self.time_s - n.time_s <= NOTICE_KEEP_S as f64) {
                    Some(n) => {
                        let (x, z) = n.at_time(self.time_s);
                        RobotOutline { x, z, ..o }
                    }
                    None if seen => o,
                    None => return None,
                };
                // Bodies that would overlap: the path's points within both half widths of the robot's centre.
                let hit = samples.iter().find(|s| o.near((s.0, s.1), me.half_w - 0.05)).map(|s| s.2);
                hit.or(touching.then_some(0.0).filter(|_| ahead <= half)).map(|s| (o.id, s))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
    }

    /// The cells robots standing still occupy, widened by `pad` (the half width of the robot routing round them and a
    /// margin), for a route to go round; all but `except`, and never the cell `keep` (where the router stands).
    pub fn still_robot_cells(&self, except: &[u32], keep: Option<(usize, usize)>, pad: f32) -> HashSet<(usize, usize)> {
        self.robot_cells(except, keep, pad, true)
    }

    /// `still_robot_cells` with moving robots too when `still_only` is false, widened by `pad`.
    pub fn robot_cells(&self, except: &[u32], keep: Option<(usize, usize)>, pad: f32, still_only: bool) -> HashSet<(usize, usize)> {
        let mut out = HashSet::new();
        for o in self.robot_outlines().into_iter().filter(|o| !except.contains(&o.id) && (!still_only || !o.moving)) {
            let (cx, cz) = match self.terrain.cell_of(o.x, o.z) {
                Some(c) => c,
                None => continue,
            };
            let span = ((o.reach_m() + pad) / CELL_M).ceil() as i32 + 1;
            for dz in -span..=span {
                for dx in -span..=span {
                    let (x, z) = (cx as i32 + dx, cz as i32 + dz);
                    if x < 0 || z < 0 || x as usize >= self.terrain.w || z as usize >= self.terrain.d {
                        continue;
                    }
                    let c = (x as usize, z as usize);
                    let (px, pz) = crate::terrain::cell_center(c.0, c.1);
                    if o.near((px, pz), pad) && Some(c) != keep {
                        out.insert(c);
                    }
                }
            }
        }
        out
    }

    /// The robots a body in the first worker's place has to miss: all but itself and its own rider.
    pub(crate) fn other_robots(&self) -> Vec<RobotOutline> {
        if self.crew_turn.is_some() {
            return self.turn_others.clone();
        }
        self.robot_outlines().into_iter().filter(|o| o.id != 1 && o.id != RIDER_ID).collect()
    }

    /// The cells of the robots standing still that a body in the first worker's place has to go round, widened by `pad`.
    pub(crate) fn still_others(&self, pad: f32) -> HashSet<(usize, usize)> {
        let keep = self.terrain.cell_of(self.agent.x, self.agent.z);
        let mut out = HashSet::new();
        for o in self.other_robots().into_iter().filter(|o| !o.moving) {
            let Some((cx, cz)) = self.terrain.cell_of(o.x, o.z) else { continue };
            let span = ((o.reach_m() + pad) / CELL_M).ceil() as i32 + 1;
            for dz in -span..=span {
                for dx in -span..=span {
                    let (x, z) = (cx as i32 + dx, cz as i32 + dz);
                    if x < 0 || z < 0 || x as usize >= self.terrain.w || z as usize >= self.terrain.d {
                        continue;
                    }
                    let c = (x as usize, z as usize);
                    if o.near(crate::terrain::cell_center(c.0, c.1), pad) && Some(c) != keep {
                        out.insert(c);
                    }
                }
            }
        }
        out
    }

    /// The first worker's way ahead: its heading, or backwards when it reverses.
    pub(crate) fn travel_dir(&self) -> f32 {
        let reversing = self.motion_view().is_some_and(|m| m.reversing);
        if reversing {
            wrap_angle(self.agent.yaw + std::f32::consts::PI)
        } else {
            self.agent.yaw
        }
    }

    /// A rider walking out to the node or back to the seat for `dt`, stopping for a robot it sees in its way and
    /// stepping round one that stays there. True once it has arrived.
    pub(crate) fn rider_walk(&self, id: u32, r: &mut Rider, dt: f32) -> bool {
        let out = r.phase == RiderPhase::WalkOut;
        let goal = if out { r.stand } else { r.seat };
        // A person on foot goes round what stands in the way, not through it. Going out, anywhere it can swing from will
        // do: the spot picked from the seat may have a rock on it.
        if r.via.is_none() && !self.rider_line_clear((r.x, r.z), goal) {
            if out && (r.x - r.node.0).hypot(r.z - r.node.1) <= MINE_REACH_M - 0.05 {
                return true;
            }
            r.via = if out { self.rider_detour((r.x, r.z), r.node, MINE_REACH_M - 0.1) } else { self.rider_detour((r.x, r.z), goal, 0.35) };
            if r.via.is_none() && out {
                (r.phase, r.failed) = (RiderPhase::WalkBack, true);
                return false;
            }
        }
        let to = r.via.unwrap_or(goal);
        let me = Looker { id, at: (r.x, r.z), dir: (to.1 - r.z).atan2(to.0 - r.x), half_w: RIDER_R_M, front: RIDER_R_M, speed: RIDER_WALK_MPS, eye_m: rider_eye_m(&self.body), gaze: 0.0 };
        let path: Vec<(f32, f32)> = r.via.into_iter().chain(std::iter::once(goal)).collect();
        if let Some((other, _)) = self.robot_in_way(&me, &path, &[]) {
            r.held_s += dt;
            if r.held_s >= YIELD_HUMANOID_S {
                // Step round it on the side nearer the goal.
                if let Some(o) = self.robot_outlines().into_iter().find(|o| o.id == other) {
                    let (dx, dz) = (goal.0 - r.x, goal.1 - r.z);
                    let n = dx.hypot(dz).max(1e-3);
                    let (px, pz) = (-dz / n, dx / n);
                    let side = if (o.x - r.x) * px + (o.z - r.z) * pz > 0.0 { -1.0 } else { 1.0 };
                    let off = o.reach_m() + RIDER_R_M + 0.3;
                    r.via = Some((o.x + px * off * side, o.z + pz * off * side));
                }
                r.held_s = 0.0;
            }
            return false;
        }
        r.held_s = 0.0;
        if r.walk_toward(to, dt) {
            if r.via.take().is_some() {
                return false;
            }
            return true;
        }
        false
    }

    fn rider_line_clear(&self, a: (f32, f32), b: (f32, f32)) -> bool {
        self.clear_map().along(a, b) >= (RIDER_R_M + clear::TOUCH_GAP_M).min(self.clear_map().at(a.0, a.1))
    }

    /// The next point of a route on foot from `a` to within `reach_m` of `b`, fitted to the rider's shoulders.
    fn rider_detour(&self, a: (f32, f32), b: (f32, f32), reach_m: f32) -> Option<(f32, f32)> {
        let spec = noto_core::BaseKind::HumanFoot.spec();
        let (map, blocked) = (self.clear_map(), self.blocked_cells());
        let fp = crate::clear::Footprint::of(&rider_body(&self.body));
        let room = nav::Room { map: &map, need_m: fp.half_w_m + clear::TIGHT_GAP_M, soft_m: fp.half_w_m + clear::COMFORT_GAP_M };
        let start = self.terrain.cell_of(a.0, a.1)?;
        let plan = nav::plan_room(&spec, &self.terrain, &|x, z| blocked.contains(&(x, z)), Some(&room), start, Goal::Within { x: b.0, z: b.1, reach_m }, None)?;
        let mut path = vec![a];
        // The cell it stands in only when that is all the route is: walked to every time, it would never leave it.
        path.extend(plan.cells.iter().skip(usize::from(plan.cells.len() > 1)).map(|c| crate::terrain::cell_center(c.0, c.1)));
        clear::fit_path(&map, &fp, &mut path, clear::TOUCH_GAP_M);
        path.iter().skip(1).find(|p| (p.0 - a.0).hypot(p.1 - a.1) > 0.05).copied().or(Some(b))
    }

    /// The first worker's rider steps down, walks to the node, fells it, walks back and climbs on: one tick of it.
    pub(super) fn advance_rider(&mut self, node: u32, remaining: f32, events: &mut Vec<TimedEvent>, done: &mut Option<(bool, Option<String>)>) -> f32 {
        let Some(mut r) = self.rider else { return 0.0 };
        let t = remaining;
        self.agent.speed_mps = 0.0;
        match r.phase {
            RiderPhase::Dismount => {
                r.t += t;
                self.add_energy(MANIPULATION_POWER_W * t as f64);
                if r.t >= DISMOUNT_S {
                    (r.phase, r.t, r.x, r.z) = (RiderPhase::WalkOut, 0.0, r.seat.0, r.seat.1);
                    events.push(self.stamp(WorldEvent::Rider { worker: 1, phase: "dismount".into(), label: format!("탑승자 하차: 지게차를 노드에서 {:.1} m에 세우고 내림", (self.agent.x - r.node.0).hypot(self.agent.z - r.node.1)) }));
                }
            }
            RiderPhase::WalkOut | RiderPhase::WalkBack => {
                if self.rider_walk(RIDER_ID, &mut r, t) {
                    if r.phase == RiderPhase::WalkOut {
                        r.phase = RiderPhase::Mine;
                        events.push(self.stamp(WorldEvent::Rider { worker: 1, phase: "walk".into(), label: "탑승자가 노드 앞까지 걸어감".into() }));
                    } else {
                        (r.phase, r.t) = (RiderPhase::Remount, 0.0);
                    }
                }
                self.add_energy(MANIPULATION_POWER_W * t as f64);
            }
            RiderPhase::Mine => match self.node(node).map(|n| (n.x, n.z)) {
                None => {
                    (r.phase, r.failed) = (RiderPhase::WalkBack, true);
                }
                Some((nx, nz)) => {
                    r.yaw = (nz - r.z).atan2(nx - r.x);
                    let rate = self.mining_rate();
                    self.add_energy(MANIPULATION_POWER_W * t as f64);
                    self.agent.strike_acc += t;
                    while self.agent.strike_acc >= 1.0 {
                        self.agent.strike_acc -= 1.0;
                        self.agent.strikes += 1;
                    }
                    let n = self.nodes.iter_mut().find(|n| n.id == node).expect("looked up above");
                    n.hp -= (t * rate).min(n.hp);
                    if n.hp <= 1e-5 {
                        self.break_node(node, events);
                        r.phase = RiderPhase::WalkBack;
                        events.push(self.stamp(WorldEvent::Rider { worker: 1, phase: "mine".into(), label: "탑승자가 걸어서 채취를 마침 — 지게차로 돌아감".into() }));
                    }
                }
            },
            RiderPhase::Remount => {
                r.t += t;
                self.add_energy(MANIPULATION_POWER_W * t as f64);
                if r.t >= REMOUNT_S {
                    self.rider = None;
                    events.push(self.stamp(WorldEvent::Rider { worker: 1, phase: "remount".into(), label: "탑승자 재탑승 — 포크 작업으로".into() }));
                    *done = Some(if r.failed { (false, Some("unknown_target".into())) } else { (true, None) });
                    return t;
                }
            }
        }
        self.rider = Some(r);
        t
    }
}
