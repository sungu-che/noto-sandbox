//! Sub-workers (서브 작업자): robots added around the first worker (the world's `agent`, worker 1) that share its world —
//! the trees they fell, the drops they pick up, the loading bay they stage at, the waypoints they stock. Each runs a
//! lighter simulation than the first worker where the work is concerned: it swings at a node at the first worker's
//! mining rate per hand, and picks up, stages and delivers in fixed times — no joint dynamics, no camera frames of its
//! own. On the move it is no lighter. It plans on its own clearance map with its own base and follows that plan on the
//! first worker's code (`world::turns`): the same checks against what it can hit, the same collisions, the same load
//! coming off in the same corners. The runtime's crew dispatcher (`noto-runtime::crew`) hands out the jobs; the heavy
//! vision pipeline runs for one robot at a time, so a crew costs about one robot's vision.
//!
//! A loading bay (상차장) is a waypoint whose deliveries become drops on the ground in a row of slots in front of it:
//! where humanoids put down what they harvested and a lift truck's tines take it from.

use noto_core::balance::Body;
use noto_core::kin::{self, KinLoads, KinPose};
use noto_core::rig::{self, Palette, Solid};
use noto_core::{BaseKind, BaseSpec, BodyPlan, FingerCount, Morphology};
use serde::{Deserialize, Serialize};

use crate::clear::{self, Footprint};
use crate::kinds::ItemKind;
use crate::nav::{self, Goal};
use crate::odometry::Odometry;
use crate::render::Obb;
use crate::snapshot::{drop_id, node_id, waypoint_id, CarryView};
use crate::terrain::cell_center;
use crate::world::traffic::{self, Rider, RiderPhase};
use crate::world::{add_stock, CargoRide, DropItem, MoveState, Stride, TimedEvent, WalkState, Waypoint, World, WorldEvent, AMOUNT_EPS, DELIVER_REACH_M, MINE_REACH_M, PICKUP_REACH_M};

/// At most this many sub-workers (with the first worker, nine robots on one map).
pub const CREW_MAX: usize = 8;
/// How many times a sub-worker routes round robots in its way on one move before the move fails.
pub const MAX_REROUTES: u32 = 6;
/// Seconds to pick an item up by hand, to put it down at a bay slot, to hand it over or deliver it.
pub const HAND_TIME_S: f32 = 1.0;
/// Seconds for a lift truck to put its tines under a staged item and lift it [추정: the first worker's fork act runs
/// 2–4 s from lowering to raising].
pub const FORK_TIME_S: f32 = 2.5;
/// Seconds for a lift truck to set its load down at a waypoint.
pub const UNLOAD_TIME_S: f32 = 2.0;
/// Power while working in place (W), as the first worker's `MANIPULATION_POWER_W`.
pub const WORK_POWER_W: f64 = 30.0;
/// The slots a bay keeps its staged items in: a row across its front, this far apart (m) [추정: a log is 0.50 m long;
/// a hand's width between them leaves room for the tines].
pub const BAY_SLOT_M: f32 = 0.55;
pub const BAY_SLOTS: usize = 6;
/// How far in front of the bay the row of slots lies (m).
pub const BAY_ROW_AHEAD_M: f32 = 0.6;
/// How far a flag may be for a sub-worker's eye to put its odometry right by it (m): the first worker's
/// `RELOCALIZE_RANGE_M`.
pub const FLAG_FIX_RANGE_M: f32 = 8.0;

/// A sub-worker in the first worker's view (`World::peers_in_view`): where the eye has it from the body (forward and
/// left, m: its true offset; the error of the look is the runtime's to put on it), how far, and what it says of
/// where it is on the map and how sure it is (its odometry and `Worker::pose_sigma_m`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct PeerSight {
    pub id: u32,
    pub rel_m: (f32, f32),
    pub range_m: f32,
    pub said: (f32, f32),
    pub said_sigma_m: f32,
}

/// What one look at a flag `d_m` away is worth to a robot's eye (m, 1σ): its head sensor's range law along the ray
/// (`stereo_depth_sigma`) and a pixel across it — the law the first worker's look at a flag is weighed by.
pub fn flag_look_sigma_m(body: &Body, d_m: f32) -> f32 {
    let (f, b, sub) = crate::world::camera_of(body.head.sensor);
    let along = noto_core::act::material::stereo_depth_sigma(d_m, f, b, sub);
    (along * along + (d_m / f.max(1.0)).powi(2)).sqrt().max(0.005)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Form {
    /// Legs and two hands: fells, picks up, carries, stages.
    Humanoid,
    /// A lift truck (wheels, mast, tines) with a rider: hauls what is staged, and can fell and pick up as the first
    /// worker's wheeled bodies do.
    Forklift,
}

impl Form {
    pub fn base(self) -> BaseKind {
        match self {
            Form::Humanoid => BaseKind::HumanFoot,
            Form::Forklift => BaseKind::Wheel,
        }
    }

    pub fn of_base(b: BaseKind) -> Form {
        match b {
            BaseKind::HumanFoot => Form::Humanoid,
            _ => Form::Forklift,
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            Form::Humanoid => "humanoid",
            Form::Forklift => "forklift",
        }
    }

    pub fn from_id(s: &str) -> Option<Form> {
        match s {
            "humanoid" | "human" | "휴머노이드" => Some(Form::Humanoid),
            "forklift" | "lift_truck" | "지게차" => Some(Form::Forklift),
            _ => None,
        }
    }

    pub fn label_ko(self) -> &'static str {
        match self {
            Form::Humanoid => "휴머노이드",
            Form::Forklift => "지게차",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Job {
    Idle,
    /// Walk or drive until within `reach_m` of (x, z).
    Go { x: f32, z: f32, reach_m: f32 },
    /// Swing at a node until it breaks.
    Mine { node: u32 },
    /// Pick a drop up (a lift truck: with its tines).
    Pick { drop: u32 },
    /// Put everything carried down in the bay's free slots.
    Stage { bay: u32 },
    /// Put everything carried into a waypoint's stock.
    Deliver { waypoint: u32 },
    /// Stand facing (x, z) until given another job.
    Park { x: f32, z: f32 },
}

impl Job {
    pub fn label_ko(&self) -> String {
        match self {
            Job::Idle => "대기".into(),
            Job::Go { .. } => "이동".into(),
            Job::Mine { node } => format!("{} 채취", node_id(*node)),
            Job::Pick { drop } => format!("{} 줍기", drop_id(*drop)),
            Job::Stage { .. } => "상차장에 내려놓기".into(),
            Job::Deliver { waypoint } => format!("{} 인도", waypoint_id(*waypoint)),
            Job::Park { .. } => "정차 대기".into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Worker {
    pub id: u32,
    pub form: Form,
    /// Cameras on its eye: 1 (monocular, TIPSv2 range) or 2 (stereo pair, SGBM range).
    pub cameras: u8,
    pub x: f32,
    pub z: f32,
    pub yaw: f32,
    pub speed_mps: f32,
    pub carried: Vec<(ItemKind, f32)>,
    pub energy_j: f64,
    pub capacity_kg: f32,
    pub job: Job,
    /// The last job's outcome (ok, what happened), for the dispatcher to take.
    pub outcome: Option<(bool, String)>,
    /// How much its last delivery or staging put down.
    pub last_moved: f32,
    /// A lift truck's rider on foot, felling a node (`traffic::Rider`).
    pub rider: Option<Rider>,
    /// Its body: the first worker's build, in its own form.
    pub body: Body,
    /// How far it has walked, for its stride.
    pub walked_m: f32,
    /// Stopped for a robot its eye sees in its way: for how long, for which; and how many times it routed round.
    pub held_s: f32,
    pub held_by: Option<u32>,
    pub reroutes: u32,
    felled: Option<String>,
    pub(crate) spec: BaseSpec,
    pub(crate) footprint: Footprint,
    /// The move it is on.
    pub(crate) mv: Option<MoveState>,
    pub walk: WalkState,
    pub ride: CargoRide,
    pub stride: Stride,
    pub fallen_until: Option<f64>,
    work_left_s: f32,
    /// The robots its eye saw a moment ago (`traffic::Noticed`).
    pub noticed: Vec<traffic::Noticed>,
    /// Where it believes it is: its own odometry, put right when its eye catches a flag (`fix_from_flags`). It moves by
    /// the world's place for it, as the sub-workers always have; this is what it can tell another robot.
    pub odom: Odometry,
    /// What the wheels slipped on its last step, for its odometry.
    slip_over: f32,
}

impl Worker {
    fn new(id: u32, form: Form, cameras: u8, x: f32, z: f32, yaw: f32, plan: &BodyPlan, fingers: FingerCount) -> Worker {
        let body = Body::new(&BodyPlan { morph: Morphology::humanoid(form.base()), ..*plan }, form.base(), fingers);
        let spec = form.base().spec();
        let capacity_kg = body.drive_limits(&spec, 0.0).capacity_kg;
        Worker {
            id,
            form,
            cameras: cameras.clamp(1, 2),
            x,
            z,
            yaw,
            speed_mps: 0.0,
            carried: Vec::new(),
            energy_j: 0.0,
            capacity_kg,
            job: Job::Idle,
            outcome: None,
            last_moved: 0.0,
            rider: None,
            walked_m: 0.0,
            held_s: 0.0,
            held_by: None,
            reroutes: 0,
            felled: None,
            spec,
            footprint: Footprint::of(&body),
            body,
            mv: None,
            walk: WalkState::default(),
            ride: CargoRide::default(),
            stride: Stride::default(),
            fallen_until: None,
            work_left_s: 0.0,
            noticed: Vec::new(),
            odom: Odometry::new(crate::odometry::OdomCfg::exact(), id as u64, form.base() == BaseKind::HumanFoot, (x, z, yaw)),
            slip_over: 0.0,
        }
    }

    /// Puts it in another body where it stands. The job it was on stops.
    pub(crate) fn rebody(&mut self, plan: BodyPlan, base: BaseKind, fingers: FingerCount) {
        self.body = Body::new(&BodyPlan { morph: Morphology::humanoid(base), ..plan }, base, fingers);
        self.form = Form::of_base(base);
        self.spec = base.spec();
        self.capacity_kg = self.body.drive_limits(&self.spec, 0.0).capacity_kg;
        self.footprint = Footprint::of(&self.body);
        self.walk = WalkState::default();
        self.ride = CargoRide::default();
        self.stride = Stride::default();
        self.mv = None;
        self.rider = None;
        self.job = Job::Idle;
        self.outcome = Some((false, "몸이 바뀌어 하던 일을 멈춤".into()));
        self.speed_mps = 0.0;
    }

    pub fn carried_kg(&self) -> f32 {
        self.carried.iter().map(|(k, n)| k.spec().mass_kg * n).sum()
    }

    /// Its outline on the ground (`clear::Footprint`).
    pub fn footprint(&self) -> Footprint {
        self.footprint
    }

    pub fn carried_count(&self) -> f32 {
        self.carried.iter().map(|(_, n)| n).sum()
    }

    /// Whether it can take one more `item` on its hands or tines.
    pub fn room_for(&self, item: ItemKind) -> bool {
        self.carried_kg() + item.spec().mass_kg <= self.capacity_kg + 1e-3
    }

    pub fn busy(&self) -> bool {
        !matches!(self.job, Job::Idle | Job::Park { .. })
    }

    /// The radius it keeps others at: its outline's half width and half its length.
    pub fn radius_m(&self) -> f32 {
        self.footprint.half_w_m.max(0.5 * (self.footprint.front_m + self.footprint.rear_m)).max(0.3)
    }

    /// Its eye height (a humanoid's head, a lift truck rider's on the seat).
    pub fn eye_m(&self) -> f32 {
        kin::head_camera(&self.body, &KinPose::stand(&self.body, 0.0)).pos[2]
    }

    /// How far off its odometry may be by now (m, 1σ): the drift spec over what it travelled since a flag put it
    /// right, the law the first worker's memory starts from before the SDS learns its own (`vision::DriftRate::spec`).
    pub fn pose_sigma_m(&self) -> f32 {
        let per_m = if self.form == Form::Humanoid { crate::odometry::LEG_SCALE_SD } else { crate::odometry::WHEEL_SCALE_SD };
        let d = self.odom.since_fix_m;
        ((per_m * d).powi(2) + crate::odometry::RANDOM_WALK_M.powi(2) * d).sqrt()
    }

    /// The pose its body is drawn in: a stride while it walks, the load held out in front of it.
    fn draw_pose(&self) -> KinPose {
        let mut p = KinPose::stand(&self.body, 0.0);
        let d = &self.body.dims;
        let carrying = !self.carried.is_empty();
        if carrying {
            let held = noto_core::act::cocarry::carry_pose(&self.body, noto_core::act::cocarry::carry_hand_z(&self.body));
            p.shoulder_deg = held.shoulder_deg;
            p.elbow_deg = held.elbow_deg;
            p.closure = held.closure;
        }
        if self.speed_mps > 0.01 && self.form == Form::Humanoid {
            let stride = 0.34 * d.hip_m;
            let phase = std::f32::consts::TAU * self.walked_m / (2.0 * stride);
            let (s, c) = phase.sin_cos();
            let reach = 0.5 * stride;
            p.feet = [[reach * s, (0.06 * d.k * c).max(0.0)], [-reach * s, (-0.06 * d.k * c).max(0.0)]];
            p.hip[0] = 0.0;
            p.hip[1] = d.hip_m - 0.03 * d.k - self.stride.crouch_m;
            if !carrying {
                p.shoulder_deg = [-22.0 * s, 22.0 * s];
                p.elbow_deg = [18.0, 18.0];
            }
        }
        p
    }

    /// Its outside as solids in the world, on ground height `g`.
    fn boxes(&self, g: f32) -> Vec<Obb> {
        let palette = Palette { accent: colour(self.id, self.form), truck: if self.form == Form::Forklift { colour(self.id, self.form) } else { rig::TRUCK_YELLOW }, ..Palette::default() };
        let pose = self.draw_pose();
        let mut parts = if self.form == Form::Humanoid {
            rig::dress(&self.body, &kin::solve(&self.body, &pose, &KinLoads::default()).boxes, &palette)
        } else {
            let mut v = rig::truck_parts(&self.body, &palette);
            if !self.rider.is_some_and(|r| r.off()) {
                v.extend(rig::rider_parts(&self.body, &pose, &palette));
            }
            let travelling = self.mv.is_some() || !self.carried.is_empty();
            v.extend(rig::fork_parts(&self.body, if travelling { self.body.dims.fork_carry_m } else { self.body.dims.fork_rest_m }));
            v
        };
        let level = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        if let Some(&(item, _)) = self.carried.first() {
            let size = item.spec().size_m;
            let n = self.carried_count().ceil().clamp(1.0, 4.0);
            let half = [0.5 * size[2], 0.5 * size[0], 0.5 * n * size[1]];
            let shift = self.ride.state.shift;
            let center = if self.form == Form::Forklift {
                [noto_core::act::fork::load_x(&self.body, size[2]) + shift[0], shift[1], self.body.dims.fork_carry_m + self.body.dims.tine_t_m + half[2]]
            } else {
                let r = kin::solve_fast(&self.body, &pose, &KinLoads::default());
                [0.5 * (r.hands[0][0] + r.hands[1][0]) + half[0] + shift[0], shift[1], 0.5 * (r.hands[0][2] + r.hands[1][2]) + half[2]]
            };
            parts.push(rig::VisualPart { solid: Solid::Box, center, axes: level, half, color: item.spec().color });
        }
        let (s, c) = self.yaw.sin_cos();
        let at = |p: [f32; 3]| [self.x + c * p[0] + s * p[1], g + p[2], self.z + s * p[0] - c * p[1]];
        let dir = |v: [f32; 3]| [c * v[0] + s * v[1], v[2], s * v[0] - c * v[1]];
        parts.iter().map(|p| Obb::placed(p, at, dir)).collect()
    }

    /// Where a lift truck's tines take a load from, or where a humanoid holds it: a little ahead of its body.
    pub fn front(&self) -> (f32, f32) {
        let ahead = self.footprint.front_m.max(0.3);
        (self.x + ahead * self.yaw.cos(), self.z + ahead * self.yaw.sin())
    }

    pub fn carry_views(&self) -> Vec<CarryView> {
        World::carry_views(&self.carried)
    }

    pub fn status_ko(&self) -> String {
        if let Some(r) = self.rider {
            return format!("탑승자 {} (채취)", r.phase.label_ko());
        }
        let load = if self.carried.is_empty() {
            String::new()
        } else {
            format!(" · 적재 {:.1} kg", self.carried_kg())
        };
        format!("{}{load}", self.job.label_ko())
    }
}

fn wrap(a: f32) -> f32 {
    let mut a = a % std::f32::consts::TAU;
    if a > std::f32::consts::PI {
        a -= std::f32::consts::TAU;
    } else if a < -std::f32::consts::PI {
        a += std::f32::consts::TAU;
    }
    a
}

/// A distinct colour per sub-worker (the first worker keeps the robot's own).
fn colour(id: u32, form: Form) -> [u8; 3] {
    const HUMANOID: [[u8; 3]; 4] = [[236, 128, 52], [52, 170, 190], [168, 96, 210], [96, 186, 82]];
    const FORKLIFT: [[u8; 3]; 2] = [[232, 192, 40], [214, 150, 32]];
    match form {
        Form::Humanoid => HUMANOID[id as usize % HUMANOID.len()],
        Form::Forklift => FORKLIFT[id as usize % FORKLIFT.len()],
    }
}

impl World {
    /// Adds sub-workers of the given forms around the first worker, each on a free cell its base can reach from
    /// the first worker's. Returns their ids (2, 3, …).
    pub fn add_crew(&mut self, forms: &[Form], cameras: u8) -> Vec<u32> {
        let mut ids = Vec::new();
        let blocked = self.blocked_cells();
        let map = self.clear_map();
        let start = self.terrain.cell_of(self.agent.x, self.agent.z).unwrap_or((0, 0));
        for &form in forms {
            if self.crew.len() >= CREW_MAX {
                break;
            }
            let id = self.crew.iter().map(|w| w.id).max().unwrap_or(1) + 1;
            let spec = form.base().spec();
            // Clear of the others' outlines: a truck stands further off than a person.
            let probe = Body::new(&BodyPlan { morph: Morphology::humanoid(form.base()), ..self.cfg.body }, form.base(), self.cfg.noto.fingers);
            let fp = Footprint::of(&probe);
            let my_r = fp.half_w_m.max(0.5 * (fp.front_m + fp.rear_m)).max(0.3);
            let mine = self.footprint();
            let first_r = mine.half_w_m.max(0.5 * (mine.front_m + mine.rear_m)).max(0.3);
            let taken: Vec<(f32, f32, f32)> = self.crew.iter().map(|w| (w.x, w.z, w.radius_m())).chain(std::iter::once((self.agent.x, self.agent.z, first_r))).collect();
            let stand_off = (my_r + first_r + 0.4).max(1.6);
            let spot = [1.0f32, 1.4, 1.8, 2.3].into_iter().map(|f| f * stand_off).find_map(|r| {
                (0..12).find_map(|k| {
                    // Behind and beside the first worker first, so the way ahead stays open.
                    let a = self.agent.yaw + std::f32::consts::PI + (k as f32 * 0.5 + 0.25) * if k % 2 == 0 { 1.0 } else { -1.0 } * 0.5;
                    let (x, z) = (self.agent.x + r * a.cos(), self.agent.z + r * a.sin());
                    let c = self.terrain.cell_of(x, z)?;
                    if blocked.contains(&c) || taken.iter().any(|&(tx, tz, tr)| (tx - x).hypot(tz - z) < (my_r + tr + 0.2).max(1.0)) {
                        return None;
                    }
                    let (px, pz) = cell_center(c.0, c.1);
                    if fp.room(&map, px, pz, self.agent.yaw) < clear::TOUCH_GAP_M {
                        return None;
                    }
                    nav::plan(&spec, &self.terrain, &|cx, cz| blocked.contains(&(cx, cz)), start, Goal::Cell(c))?;
                    let (cx, cz) = cell_center(c.0, c.1);
                    Some((cx, cz))
                })
            });
            let Some((x, z)) = spot else { continue };
            let yaw = self.agent.yaw;
            let mut w = Worker::new(id, form, cameras, x, z, yaw, &self.cfg.body, self.cfg.noto.fingers);
            // It starts where it was put, and from there knows where it is only by its own odometry and the flags.
            w.odom = Odometry::new(self.cfg.odometry, self.cfg.seed ^ (0xC2E_0000 + id as u64), form.base() == BaseKind::HumanFoot, (x, z, yaw));
            self.crew.push(w);
            ids.push(id);
        }
        ids
    }

    /// Takes every sub-worker off the map; what they carried goes down where they stood.
    pub fn remove_crew(&mut self) -> usize {
        let crew = std::mem::take(&mut self.crew);
        for w in &crew {
            for (k, n) in &w.carried {
                if *n > AMOUNT_EPS {
                    let d = DropItem { amount: *n, ..DropItem::whole(self.next_drop_id(), *k, w.x, w.z) };
                    self.drops.push(d);
                }
            }
        }
        crew.len()
    }

    /// Takes one sub-worker off the map; what it carried goes down where it stood.
    pub fn remove_worker(&mut self, id: u32) -> bool {
        let Some(i) = self.crew.iter().position(|w| w.id == id) else { return false };
        let w = self.crew.remove(i);
        for (k, n) in &w.carried {
            if *n > AMOUNT_EPS {
                let d = DropItem { amount: *n, ..DropItem::whole(self.next_drop_id(), *k, w.x, w.z) };
                self.drops.push(d);
            }
        }
        true
    }

    pub fn crew_worker(&self, id: u32) -> Option<&Worker> {
        self.crew.iter().find(|w| w.id == id)
    }

    pub fn crew_worker_mut(&mut self, id: u32) -> Option<&mut Worker> {
        self.crew.iter_mut().find(|w| w.id == id)
    }

    /// Gives a sub-worker a job; a move plans its route now (an error when there is none).
    pub fn crew_command(&mut self, id: u32, job: Job) -> Result<(), String> {
        let route = match &job {
            Job::Go { x, z, reach_m } => {
                let i = self.crew.iter().position(|w| w.id == id).ok_or_else(|| format!("작업자 {id}가 없습니다"))?;
                let w = &self.crew[i];
                if (w.x - x).hypot(w.z - z) <= *reach_m {
                    None
                } else {
                    Some(self.crew_route(i, *x, *z, *reach_m).ok_or_else(|| format!("작업자 {id}: ({x:.1}, {z:.1})로 가는 길이 없습니다"))?)
                }
            }
            _ => None,
        };
        let w = self.crew_worker(id).ok_or_else(|| format!("작업자 {id}가 없습니다"))?;
        let (wx, wz) = (w.x, w.z);
        let check = match &job {
            Job::Mine { node } => self.node(*node).map(|n| (n.x - wx).hypot(n.z - wz) <= if w.form == Form::Forklift { traffic::rider_reach_m(&w.body) } else { MINE_REACH_M + 0.35 }).ok_or("unknown_target"),
            Job::Pick { drop } => self.drop_item(*drop).map(|d| (d.x - wx).hypot(d.z - wz) <= if w.form == Form::Forklift { (PICKUP_REACH_M + 1.2).max(noto_core::act::fork::standoff_x(&w.body, 0.5) + 0.3) } else { PICKUP_REACH_M + 0.35 }).ok_or("unknown_target"),
            // A lift truck sets a load down at the end of its tines.
            Job::Deliver { waypoint } | Job::Stage { bay: waypoint } => self.waypoint(*waypoint).map(|p| (p.x - wx).hypot(p.z - wz) <= (DELIVER_REACH_M + 0.6).max(if w.form == Form::Forklift { w.footprint.front_m + 0.3 } else { 0.0 })).ok_or("unknown_target"),
            _ => Ok(true),
        };
        match check {
            Err(code) => return Err(format!("작업자 {id}: 대상이 없습니다 ({code})")),
            Ok(false) => return Err(format!("작업자 {id}: 대상에 손이 닿지 않습니다")),
            Ok(true) => {}
        }
        let work = match &job {
            Job::Pick { .. } if w.form == Form::Forklift => FORK_TIME_S,
            Job::Pick { .. } | Job::Stage { .. } => HAND_TIME_S,
            Job::Deliver { .. } if w.form == Form::Forklift => UNLOAD_TIME_S,
            Job::Deliver { .. } => HAND_TIME_S,
            _ => 0.0,
        };
        let node_at = match &job {
            Job::Mine { node } => self.node(*node).map(|n| (n.x, n.z)),
            _ => None,
        };
        let w = self.crew_worker_mut(id).expect("looked up above");
        w.mv = route;
        w.work_left_s = work;
        w.outcome = None;
        w.held_s = 0.0;
        w.held_by = None;
        w.reroutes = 0;
        // A lift truck's rider steps down and fells it on foot.
        w.rider = match (w.form, node_at) {
            (Form::Forklift, Some(at)) => Some(Rider::start(w.x, w.z, w.yaw, at)),
            _ => None,
        };
        w.felled = None;
        w.job = job;
        Ok(())
    }

    /// The outcome of a sub-worker's last job, once (ok, what happened).
    pub fn crew_take_outcome(&mut self, id: u32) -> Option<(bool, String)> {
        self.crew_worker_mut(id)?.outcome.take()
    }

    fn crew_route(&mut self, i: usize, x: f32, z: f32, reach_m: f32) -> Option<MoveState> {
        let w = &self.crew[i];
        let start = self.terrain.cell_of(w.x, w.z)?;
        let robots = self.still_robot_cells(&[w.id, traffic::RIDER_ID + w.id], Some(start), w.footprint.half_w_m + 0.15);
        self.worker_move(i, x, z, reach_m, &robots).or_else(|| self.worker_move(i, x, z, reach_m, &std::collections::HashSet::new()))
    }

    /// Advances every sub-worker by `dt`, and what it believes of where it is.
    pub(crate) fn step_crew(&mut self, dt: f32, events: &mut Vec<TimedEvent>) {
        for i in 0..self.crew.len() {
            let before = (self.crew[i].x, self.crew[i].z, self.crew[i].yaw);
            self.step_worker(i, dt, events);
            let w = &mut self.crew[i];
            let over = std::mem::take(&mut w.slip_over);
            w.odom.update(before, (w.x, w.z, w.yaw), dt, over);
            self.fix_from_flags(i);
        }
    }

    /// A sub-worker whose eye catches a flag (a fixed waypoint, not a loading bay) within `FLAG_FIX_RANGE_M` puts its
    /// odometry right by it, as the first worker does (`noto_runtime::Runtime::relocalize_from`): a Kalman step between
    /// what its drift says it may be off by and what its eye's range law says the look is worth.
    fn fix_from_flags(&mut self, i: usize) {
        let w = &self.crew[i];
        let (at, dir, gaze, eye) = ((w.x, w.z), w.yaw, self.glance_rad(w.id), w.eye_m());
        let half = traffic::SIGHT_HALF_FOV_DEG.to_radians();
        let near = self
            .waypoints
            .iter()
            .filter(|p| !p.key.starts_with("bay"))
            .map(|p| (p, (p.x - at.0).hypot(p.z - at.1)))
            .filter(|(p, d)| *d <= FLAG_FIX_RANGE_M && wrap((p.z - at.1).atan2(p.x - at.0) - (dir - gaze)).abs() <= half && self.line_of_sight(at, (p.x, p.z), eye.min(1.2)))
            .map(|(_, d)| d)
            .min_by(|a, b| a.total_cmp(b));
        let Some(d) = near else { return };
        let p_pose = w.pose_sigma_m().powi(2);
        let p_obs = flag_look_sigma_m(&w.body, d).powi(2);
        let k = p_pose / (p_pose + p_obs).max(1e-12);
        // The look's own error: the same for the same robot, place and moment.
        let n = |salt: u64| {
            let h = crate::rng::hash2(self.cfg.seed ^ salt ^ w.id as u64, (w.x * 100.0) as i32, (self.time_s * 10.0) as i32);
            ((h & 0xFFFF) as f32 / 65535.0 - 0.5) * 12f32.sqrt() * p_obs.sqrt() / std::f32::consts::SQRT_2
        };
        let (ex, ez) = (w.x - w.odom.x + n(0x51), w.z - w.odom.z + n(0xA7));
        if k * ex.hypot(ez) > 1e-3 {
            self.crew[i].odom.correct(k * ex, k * ez);
        }
    }

    /// The sub-workers the first worker's head camera `cam` has in view now — inside its horizontal view (the neck's
    /// pan included), within `FLAG_FIX_RANGE_M`, not behind a node — each with what it says of where it is and how
    /// sure it is (`PeerSight`): a landmark that moves, for the first worker's odometry.
    pub fn peers_in_view(&self, cam: &crate::render::Camera) -> Vec<PeerSight> {
        let view = self.head_view(cam).view;
        let (x, z, yaw) = (self.agent.x, self.agent.z, self.agent.yaw);
        let half = 0.5 * cam.fov_deg.to_radians() - 0.05;
        let eye = self.standing_eye_m();
        self.crew
            .iter()
            .filter_map(|w| {
                let (dx, dz) = (w.x - x, w.z - z);
                let range_m = dx.hypot(dz);
                let off = wrap(dz.atan2(dx) - view.yaw).abs();
                (range_m <= FLAG_FIX_RANGE_M && off <= half && self.line_of_sight((x, z), (w.x, w.z), eye.min(1.2))).then(|| {
                    let (s, c) = yaw.sin_cos();
                    // Forward and left of the body (left is yaw − 90° on the map).
                    let rel_m = (dx * c + dz * s, dx * s - dz * c);
                    PeerSight { id: w.id, rel_m, range_m, said: (w.odom.x, w.odom.z), said_sigma_m: w.pose_sigma_m() }
                })
            })
            .collect()
    }

    fn crew_event(&self, id: u32, kind: &str, label: String, item: Option<ItemKind>, amount: f32, ok: bool) -> TimedEvent {
        self.stamp(WorldEvent::Crew { worker: id, kind: kind.into(), label, item, amount, ok })
    }

    fn step_worker(&mut self, i: usize, dt: f32, events: &mut Vec<TimedEvent>) {
        let job = self.crew[i].job.clone();
        let id = self.crew[i].id;
        let finish = |w: &mut Worker, ok: bool, what: String| {
            w.outcome = Some((ok, what));
            w.job = Job::Idle;
            w.speed_mps = 0.0;
            w.mv = None;
        };
        if self.crew[i].fallen_until.is_some_and(|t| self.time_s < t) {
            self.crew[i].speed_mps = 0.0;
            return;
        }
        if !matches!(job, Job::Go { .. }) {
            self.crew[i].ride.at_rest();
        }
        match job {
            Job::Idle => self.crew[i].speed_mps = 0.0,
            Job::Park { x, z } => {
                self.crew[i].speed_mps = 0.0;
                self.crew_turn_to(i, x, z, dt);
            }
            Job::Go { x, z, reach_m } => {
                if self.crew_hold(i, dt, (x, z, reach_m), events) {
                    return;
                }
                let Some(mut m) = self.crew[i].mv.take() else {
                    finish(&mut self.crew[i], true, "도착".into());
                    return;
                };
                let was = (self.crew[i].x, self.crew[i].z);
                let step = self.worker_step(i, &mut m, dt, events);
                let w = &mut self.crew[i];
                w.slip_over = step.slip_over;
                w.walked_m += (w.x - was.0).hypot(w.z - was.1);
                if step.lost {
                    let d = self.drops.last().map_or(String::new(), |d| drop_id(d.id));
                    finish(&mut self.crew[i], false, format!("cargo_lost:{d}"));
                    return;
                }
                match step.done {
                    Some((true, _)) => finish(w, true, "도착".into()),
                    Some((false, why)) => finish(w, false, why.unwrap_or_else(|| "blocked".into())),
                    None => w.mv = Some(m),
                }
            }
            Job::Mine { node } if self.crew[i].rider.is_some() => self.crew_rider_step(i, node, dt, events),
            Job::Mine { node } => {
                let Some(n) = self.node(node) else {
                    finish(&mut self.crew[i], false, format!("{} 없음 (이미 쓰러짐)", node_id(node)));
                    return;
                };
                let (nx, nz) = (n.x, n.z);
                self.crew_turn_to(i, nx, nz, dt);
                let w = &mut self.crew[i];
                w.energy_j += WORK_POWER_W * dt as f64;
                // The first worker's rate for five fingers, a hand per robot.
                let rate = 1.0f32;
                let n = self.nodes.iter_mut().find(|n| n.id == node).expect("looked up above");
                n.hp -= (rate * dt).min(n.hp);
                if n.hp <= 1e-5 {
                    let (kind, drops) = self.fell(node);
                    let label = format!("{} {} 쓰러뜨림 → {}", kind.spec().label_ko, node_id(node), drops.iter().map(|d| drop_id(*d)).collect::<Vec<_>>().join(", "));
                    events.push(self.crew_event(id, "node_broken", label, Some(kind.spec().drop), drops.len() as f32, true));
                    finish(&mut self.crew[i], true, drops.iter().map(|d| drop_id(*d)).collect::<Vec<_>>().join(","));
                }
            }
            Job::Pick { drop } => {
                let w = &mut self.crew[i];
                w.work_left_s -= dt;
                w.energy_j += WORK_POWER_W * dt as f64;
                if w.work_left_s > 0.0 {
                    return;
                }
                let Some(pos) = self.drops.iter().position(|d| d.id == drop) else {
                    finish(&mut self.crew[i], false, format!("{} 없음", drop_id(drop)));
                    return;
                };
                if !self.crew[i].room_for(self.drops[pos].kind) {
                    finish(&mut self.crew[i], false, "적재 한계".into());
                    return;
                }
                let d = self.drops.remove(pos);
                let w = &mut self.crew[i];
                add_stock(&mut w.carried, d.kind, d.amount);
                let how = if w.form == Form::Forklift { "포크로 들어 올림" } else { "주움" };
                let label = format!("{} {} {how}", d.kind.spec().label_ko, drop_id(drop));
                finish(w, true, drop_id(drop));
                events.push(self.crew_event(id, "picked", label, Some(d.kind), d.amount, true));
            }
            Job::Stage { bay } => {
                let w = &mut self.crew[i];
                w.work_left_s -= dt;
                w.energy_j += WORK_POWER_W * dt as f64;
                if w.work_left_s > 0.0 {
                    return;
                }
                let load = std::mem::take(&mut self.crew[i].carried);
                self.crew[i].last_moved = load.iter().map(|(_, n)| n).sum();
                let mut ids = Vec::new();
                for (k, n) in load {
                    ids.extend(self.stage_at_bay(bay, k, n));
                }
                let label = format!("상차장에 {}개 내려놓음 ({})", ids.len(), ids.iter().map(|d| drop_id(*d)).collect::<Vec<_>>().join(", "));
                finish(&mut self.crew[i], true, ids.iter().map(|d| drop_id(*d)).collect::<Vec<_>>().join(","));
                events.push(self.crew_event(id, "staged", label, None, ids.len() as f32, true));
            }
            Job::Deliver { waypoint } => {
                let w = &mut self.crew[i];
                w.work_left_s -= dt;
                w.energy_j += WORK_POWER_W * dt as f64;
                if w.work_left_s > 0.0 {
                    return;
                }
                let load = std::mem::take(&mut self.crew[i].carried);
                self.crew[i].last_moved = load.iter().map(|(_, n)| n).sum();
                let name = self.waypoint(waypoint).map(|p| p.label_ko.clone()).unwrap_or_default();
                for (k, n) in load.into_iter().filter(|(_, n)| *n > AMOUNT_EPS) {
                    if let Some(p) = self.waypoints.iter_mut().find(|p| p.id == waypoint) {
                        add_stock(&mut p.stock, k, n);
                    }
                    let label = format!("{name}에 {} {} 인도", k.spec().label_ko, crate::world::amount_text(n));
                    events.push(self.crew_event(id, "delivered", label, Some(k), n, true));
                }
                finish(&mut self.crew[i], true, name);
            }
        }
    }

    /// Turns a standing sub-worker toward (`x`, `z`) at a walker's pace, as far as its body has room to swing.
    fn crew_turn_to(&mut self, i: usize, x: f32, z: f32, dt: f32) {
        let w = &self.crew[i];
        let turn = wrap((z - w.z).atan2(x - w.x) - w.yaw).clamp(-1.2 * dt, 1.2 * dt);
        if turn.abs() < 1e-4 {
            return;
        }
        let yaw = wrap(w.yaw + turn);
        if self.as_worker(i, |world| world.pose_check().clear(world.agent.x, world.agent.z, yaw)) {
            self.crew[i].yaw = yaw;
        }
    }

    /// The way-ahead check of a sub-worker on the move: a robot its eye sees in its way stops it; one that stays has
    /// it route round (a humanoid sooner than a lift truck). True while it waits.
    fn crew_hold(&mut self, i: usize, dt: f32, goal: (f32, f32, f32), events: &mut Vec<TimedEvent>) -> bool {
        let w = &self.crew[i];
        let (id, form) = (w.id, w.form);
        let path: Vec<(f32, f32)> = w.mv.as_ref().map_or(Vec::new(), |m| m.follow.ahead());
        let dir = match path.first() {
            Some(l) if (l.0 - w.x).hypot(l.1 - w.z) > 1e-3 => (l.1 - w.z).atan2(l.0 - w.x),
            _ => w.yaw,
        };
        let me = traffic::Looker { id, at: (w.x, w.z), dir, half_w: w.footprint.half_w_m, front: w.footprint.front_m, speed: w.spec.max_speed_mps.min(1.0), eye_m: w.eye_m(), gaze: self.glance_rad(id) };
        let seen = self.robots_seen(&me);
        let now = self.time_s;
        traffic::remember(&mut self.crew[i].noticed, seen, now);
        let mind = self.crew[i].noticed.clone();
        let Some((other, gap)) = self.robot_in_way(&me, &path, &mind) else {
            let w = &mut self.crew[i];
            w.held_s = 0.0;
            w.held_by = None;
            return false;
        };
        let w = &mut self.crew[i];
        w.speed_mps = 0.0;
        if w.held_by != Some(other) {
            w.held_by = Some(other);
            w.held_s = 0.0;
            let label = format!("시야에 작업자 {} {gap:.1} m 앞 — 멈춰 기다림", if other >= traffic::RIDER_ID { format!("{}의 탑승자", other - traffic::RIDER_ID) } else { other.to_string() });
            events.push(self.stamp(WorldEvent::Hold { worker: id, other, gap_m: gap, label }));
        }
        let blocker = self.robot_outlines().into_iter().find(|o| o.id == other);
        let (other_humanoid, other_moving) = blocker.map_or((false, false), |o| (o.humanoid, o.moving));
        // An idle sub-worker in the way steps aside.
        if !other_moving && self.crew_make_way(other, dir) {
            events.push(self.stamp(WorldEvent::Hold { worker: other, other: id, gap_m: gap, label: format!("작업자 {id}의 길에서 비켜 섬") }));
        }
        let w = &mut self.crew[i];
        w.held_s += dt;
        let patience = if form == Form::Humanoid { traffic::YIELD_HUMANOID_S } else { traffic::YIELD_VEHICLE_S };
        // Of two on the move only the one that gives way routes round, the other waits for it to pass; a robot that
        // stands still is routed round by whoever it is in the way of.
        if w.held_s < patience || (other_moving && !traffic::gives_way(id, form == Form::Humanoid, other, other_humanoid)) {
            return true;
        }
        if w.reroutes >= MAX_REROUTES {
            // Blocked every way round: the move fails, and whoever gave it picks another target.
            w.outcome = Some((false, format!("작업자 {other}에 막혀 갈 길이 없음")));
            w.job = Job::Idle;
            w.mv = None;
            return true;
        }
        let (start, half) = (self.terrain.cell_of(w.x, w.z), w.footprint.half_w_m);
        let mut extra = self.still_robot_cells(&[id, traffic::RIDER_ID + id], start, half + 0.15);
        let others: Vec<u32> = self.robot_outlines().iter().map(|o| o.id).filter(|o| *o != other).collect();
        extra.extend(self.robot_cells(&others, start, half + 0.15, false));
        let route = self.worker_move(i, goal.0, goal.1, goal.2, &extra);
        let w = &mut self.crew[i];
        w.held_s = 0.0;
        // Counted whether a way round turned up or not: a robot that stays and leaves none ends the move above.
        w.reroutes += 1;
        if let Some(r) = route {
            w.mv = Some(r);
            events.push(self.stamp(WorldEvent::Hold { worker: id, other, gap_m: gap, label: format!("작업자 {other}이(가) 비키지 않아 돌아가는 길로 다시 계획") }));
        }
        true
    }

    /// Asks a sub-worker standing idle in another robot's way (that robot going along `dir`) to step aside, across
    /// that way; true when it goes.
    pub fn crew_make_way(&mut self, id: u32, dir: f32) -> bool {
        let Some(w) = self.crew_worker(id) else { return false };
        if !matches!(w.job, Job::Idle) {
            return false;
        }
        let (x, z) = (w.x, w.z);
        for d in [1.3f32, 2.0] {
            for side in [1.0f32, -1.0] {
                let a = dir + side * std::f32::consts::FRAC_PI_2;
                if self.crew_command(id, Job::Go { x: x + d * a.cos(), z: z + d * a.sin(), reach_m: 0.3 }).is_ok() {
                    return true;
                }
            }
        }
        false
    }

    /// A sub-worker lift truck's rider fells a node on foot: one tick of stepping down, walking, swinging, walking
    /// back, climbing on.
    fn crew_rider_step(&mut self, i: usize, node: u32, dt: f32, events: &mut Vec<TimedEvent>) {
        let Some(mut r) = self.crew[i].rider else { return };
        let id = self.crew[i].id;
        self.crew[i].speed_mps = 0.0;
        self.crew[i].energy_j += WORK_POWER_W * dt as f64;
        match r.phase {
            RiderPhase::Dismount => {
                r.t += dt;
                if r.t >= traffic::DISMOUNT_S {
                    (r.phase, r.t, r.x, r.z) = (RiderPhase::WalkOut, 0.0, r.seat.0, r.seat.1);
                    events.push(self.stamp(WorldEvent::Rider { worker: id, phase: "dismount".into(), label: "탑승자 하차: 지게차를 세우고 내림".into() }));
                }
            }
            RiderPhase::WalkOut | RiderPhase::WalkBack => {
                if self.rider_walk(traffic::RIDER_ID + id, &mut r, dt) {
                    if r.phase == RiderPhase::WalkOut {
                        r.phase = RiderPhase::Mine;
                    } else {
                        (r.phase, r.t) = (RiderPhase::Remount, 0.0);
                    }
                }
            }
            RiderPhase::Mine => match self.node(node).map(|n| (n.x, n.z)) {
                None => {
                    (r.phase, r.failed) = (RiderPhase::WalkBack, true);
                }
                Some((nx, nz)) => {
                    r.yaw = (nz - r.z).atan2(nx - r.x);
                    let n = self.nodes.iter_mut().find(|n| n.id == node).expect("looked up above");
                    n.hp -= dt.min(n.hp);
                    if n.hp <= 1e-5 {
                        let (kind, drops) = self.fell(node);
                        let label = format!("탑승자가 걸어가 {} {} 쓰러뜨림 → {}", kind.spec().label_ko, node_id(node), drops.iter().map(|d| drop_id(*d)).collect::<Vec<_>>().join(", "));
                        events.push(self.crew_event(id, "node_broken", label, Some(kind.spec().drop), drops.len() as f32, true));
                        self.crew[i].felled = Some(drops.iter().map(|d| drop_id(*d)).collect::<Vec<_>>().join(","));
                        r.phase = RiderPhase::WalkBack;
                    }
                }
            },
            RiderPhase::Remount => {
                r.t += dt;
                if r.t >= traffic::REMOUNT_S {
                    events.push(self.stamp(WorldEvent::Rider { worker: id, phase: "remount".into(), label: "탑승자 재탑승".into() }));
                    let w = &mut self.crew[i];
                    w.rider = None;
                    let felled = w.felled.take();
                    w.outcome = Some(match (r.failed, felled) {
                        (false, Some(d)) => (true, d),
                        _ => (false, format!("{} 없음 (이미 쓰러짐)", node_id(node))),
                    });
                    w.job = Job::Idle;
                    return;
                }
            }
        }
        self.crew[i].rider = Some(r);
    }

    /// A loading bay: a waypoint whose deliveries become drops in the row of slots in front of it, facing `face`.
    pub fn add_bay(&mut self, x: f32, z: f32, face: (f32, f32), label_ko: &str) -> u32 {
        let id = self.waypoints.iter().map(|p| p.id).max().unwrap_or(0) + 1;
        let n = self.waypoints.iter().filter(|p| p.key.starts_with("bay")).count() + 1;
        self.waypoints.push(Waypoint { id, key: format!("bay{n}"), label_ko: label_ko.into(), x, z, stock: Vec::new() });
        self.bay_faces.push((id, (face.1 - z).atan2(face.0 - x)));
        id
    }

    pub fn is_bay(&self, waypoint: u32) -> bool {
        self.waypoint(waypoint).is_some_and(|p| p.key.starts_with("bay"))
    }

    /// The `k`-th slot of a bay's row.
    pub fn bay_slot(&self, bay: u32, k: usize) -> Option<(f32, f32)> {
        let p = self.waypoint(bay)?;
        let face = self.bay_faces.iter().find(|(id, _)| *id == bay).map_or(0.0, |(_, a)| *a);
        let (fx, fz) = (face.cos(), face.sin());
        let (rx, rz) = (-fz, fx);
        let off = (k as f32 - (BAY_SLOTS as f32 - 1.0) * 0.5) * BAY_SLOT_M;
        Some((p.x + fx * BAY_ROW_AHEAD_M + rx * off, p.z + fz * BAY_ROW_AHEAD_M + rz * off))
    }

    /// The drops lying in a bay's slots now.
    pub fn staged_at(&self, bay: u32) -> Vec<u32> {
        let slots: Vec<(f32, f32)> = (0..BAY_SLOTS).filter_map(|k| self.bay_slot(bay, k)).collect();
        self.drops.iter().filter(|d| slots.iter().any(|(x, z)| (d.x - x).hypot(d.z - z) < 0.5 * BAY_SLOT_M)).map(|d| d.id).collect()
    }

    /// Puts `amount` of `item` down at a bay, a whole one per free slot (a fraction in one); the rest goes to the
    /// bay's stock when the row is full.
    pub fn stage_at_bay(&mut self, bay: u32, item: ItemKind, amount: f32) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut left = amount;
        while left > AMOUNT_EPS {
            let piece = left.min(1.0);
            let occupied = |w: &World, x: f32, z: f32| w.drops.iter().any(|d| (d.x - x).hypot(d.z - z) < 0.5 * BAY_SLOT_M);
            let Some((x, z)) = (0..BAY_SLOTS).filter_map(|k| self.bay_slot(bay, k)).find(|&(x, z)| !occupied(self, x, z)) else {
                if let Some(p) = self.waypoints.iter_mut().find(|p| p.id == bay) {
                    add_stock(&mut p.stock, item, left);
                }
                break;
            };
            let id = self.next_drop_id();
            self.drops.push(DropItem { amount: piece, ..DropItem::whole(id, item, x, z) });
            ids.push(id);
            left -= piece;
        }
        ids
    }

    /// The bodies of the sub-workers as solids, for every camera to draw (id 0: not an object the scan names).
    pub fn crew_boxes(&self) -> Vec<Obb> {
        self.crew.iter().flat_map(|w| w.boxes(self.surface_m(w.x, w.z))).collect()
    }

    pub fn crew_views(&self) -> Vec<crate::snapshot::CrewView> {
        self.crew
            .iter()
            .map(|w| crate::snapshot::CrewView {
                id: w.id,
                form: w.form,
                cameras: w.cameras,
                pos: crate::snapshot::Pos3 { x: w.x, y: self.surface_m(w.x, w.z), z: w.z },
                yaw_deg: w.yaw.to_degrees(),
                speed_mps: w.speed_mps,
                carrying: w.carry_views(),
                carried_kg: w.carried_kg(),
                capacity_kg: w.capacity_kg,
                job: w.job.clone(),
                status_ko: w.status_ko(),
                energy_j: w.energy_j,
                rider: w.rider.filter(|r| r.off()).map(|r| crate::snapshot::Pos3 { x: r.x, y: self.surface_m(r.x, r.z), z: r.z }),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kinds::NodeKind;
    use crate::terrain::TerrainKind;
    use crate::world::{Node, WorldConfig};

    fn world() -> World {
        World::new(WorldConfig { seed: 3, parcels: Some(0), terrain: TerrainKind::Open, ..WorldConfig::default() }).unwrap()
    }

    fn run(w: &mut World, id: u32, max_s: f32) -> (bool, String) {
        let mut t = 0.0;
        while t < max_s {
            w.tick();
            t += crate::world::TICK_S;
            if let Some(o) = w.crew_take_outcome(id) {
                return o;
            }
        }
        panic!("worker {id} still on {:?} after {max_s} s", w.crew_worker(id).map(|w| &w.job));
    }

    #[test]
    fn sub_workers_stand_around_the_first_worker_on_free_cells() {
        let mut w = world();
        let ids = w.add_crew(&[Form::Humanoid, Form::Humanoid, Form::Forklift], 1);
        assert_eq!(ids, vec![2, 3, 4]);
        for c in &w.crew {
            let d = (c.x - w.agent.x).hypot(c.z - w.agent.z);
            assert!((1.0..9.0).contains(&d), "worker {} stands {d:.2} m from the first", c.id);
            assert!(!w.blocked_cells().contains(&w.terrain.cell_of(c.x, c.z).unwrap()));
        }
        for (a, b) in [(0, 1), (0, 2), (1, 2)] {
            assert!((w.crew[a].x - w.crew[b].x).hypot(w.crew[a].z - w.crew[b].z) >= 1.0, "apart");
        }
        assert!(w.crew[2].capacity_kg > w.crew[0].capacity_kg, "a lift truck takes more than two hands");
        assert!(w.crew_boxes().len() > 100, "the bodies are drawn in full");
        let many = w.add_crew(&[Form::Humanoid; 9], 2);
        assert_eq!(w.crew.len(), CREW_MAX, "{many:?}");
        assert_eq!(w.remove_crew(), CREW_MAX);
    }

    #[test]
    fn a_humanoid_fells_a_tree_and_stages_its_logs_at_a_bay_for_the_tines() {
        let mut w = world();
        w.nodes.clear();
        w.drops.clear();
        let (ax, az, yaw) = (w.agent.x, w.agent.z, w.agent.yaw);
        let tree = (ax + 3.0 * yaw.cos(), az + 3.0 * yaw.sin());
        w.nodes.push(Node { id: 500, kind: NodeKind::Tree, x: tree.0, z: tree.1, hp: 3.0, max_hp: 3.0 });
        let id = w.add_crew(&[Form::Humanoid], 1)[0];
        w.crew_command(id, Job::Go { x: tree.0, z: tree.1, reach_m: MINE_REACH_M - 0.1 }).unwrap();
        assert!(run(&mut w, id, 60.0).0);
        w.crew_command(id, Job::Mine { node: 500 }).unwrap();
        let (ok, drops) = run(&mut w, id, 20.0);
        assert!(ok && w.node(500).is_none(), "felled: {drops}");
        let logs: Vec<u32> = w.drops.iter().filter(|d| d.kind == ItemKind::Log).map(|d| d.id).collect();
        assert_eq!(logs.len(), 3);
        let bay = w.add_bay(ax + 1.0, az, (tree.0, tree.1), "상차장");
        assert!(w.is_bay(bay));
        for &d in &logs {
            let p = w.drop_item(d).map(|d| (d.x, d.z)).unwrap();
            w.crew_command(id, Job::Go { x: p.0, z: p.1, reach_m: PICKUP_REACH_M - 0.1 }).unwrap();
            assert!(run(&mut w, id, 30.0).0);
            w.crew_command(id, Job::Pick { drop: d }).unwrap();
            let (ok, why) = run(&mut w, id, 5.0);
            assert!(ok, "{why}");
            let b = w.waypoint(bay).map(|p| (p.x, p.z)).unwrap();
            w.crew_command(id, Job::Go { x: b.0, z: b.1, reach_m: DELIVER_REACH_M - 0.3 }).unwrap();
            assert!(run(&mut w, id, 30.0).0);
            w.crew_command(id, Job::Stage { bay }).unwrap();
            assert!(run(&mut w, id, 5.0).0);
        }
        let staged = w.staged_at(bay);
        assert_eq!(staged.len(), 3, "three logs in the bay's row");
        let slots: Vec<(f32, f32)> = staged.iter().map(|d| w.drop_item(*d).map(|d| (d.x, d.z)).unwrap()).collect();
        for (a, b) in [(0, 1), (1, 2), (0, 2)] {
            assert!((slots[a].0 - slots[b].0).hypot(slots[a].1 - slots[b].1) >= BAY_SLOT_M - 1e-3, "one log a slot");
        }
        assert!(w.crew[0].energy_j > 0.0);
    }

    const PICK_REACH_TRUCK_TEST: f32 = 2.4;

    #[test]
    fn a_lift_truck_takes_staged_logs_on_its_tines_and_delivers_them() {
        let mut w = world();
        w.nodes.clear();
        w.drops.clear();
        let (ax, az) = (w.agent.x, w.agent.z);
        let bay = w.add_bay(ax + 6.0, az + 1.0, (ax + 8.0, az + 1.0), "상차장");
        let staged = w.stage_at_bay(bay, ItemKind::Log, 2.0);
        assert_eq!(staged.len(), 2);
        let id = w.add_crew(&[Form::Forklift], 2)[0];
        for d in staged {
            let p = w.drop_item(d).map(|d| (d.x, d.z)).unwrap();
            w.crew_command(id, Job::Go { x: p.0, z: p.1, reach_m: PICK_REACH_TRUCK_TEST }).unwrap();
            let (ok, why) = run(&mut w, id, 40.0);
            assert!(ok, "{why}");
            w.crew_command(id, Job::Pick { drop: d }).unwrap();
            let (ok, why) = run(&mut w, id, 5.0);
            assert!(ok, "{why}");
        }
        assert_eq!(w.crew[0].carried_count(), 2.0);
        let wh = w.waypoint_by_key("warehouse").map(|p| (p.id, p.x, p.z)).unwrap();
        w.crew_command(id, Job::Go { x: wh.1, z: wh.2, reach_m: DELIVER_REACH_M - 0.3 }).unwrap();
        let (ok, why) = run(&mut w, id, 200.0);
        assert!(ok, "drove to the warehouse: {why} at {:?}", (w.crew[0].x, w.crew[0].z));
        w.crew_command(id, Job::Deliver { waypoint: wh.0 }).unwrap();
        assert!(run(&mut w, id, 5.0).0);
        assert_eq!(w.waypoint(wh.0).unwrap().stock_of(ItemKind::Log), 2.0);
        assert!(w.crew[0].carried.is_empty());
        let mut ev = Vec::new();
        w.step_crew(0.1, &mut ev);
        assert!(w.crew_command(id, Job::Mine { node: 999 }).is_err(), "no such node");
    }

    #[test]
    fn a_lift_truck_rider_steps_down_fells_the_tree_on_foot_and_climbs_on_again() {
        let mut w = world();
        w.nodes.clear();
        w.drops.clear();
        let id = w.add_crew(&[Form::Forklift], 1)[0];
        let (x, z) = { let c = w.crew_worker(id).unwrap(); (c.x, c.z) };
        // Two metres off, on the side away from the first worker.
        let away = (z - w.agent.z).atan2(x - w.agent.x);
        let tree = (x + 2.0 * away.cos(), z + 2.0 * away.sin());
        w.nodes.push(Node { id: 600, kind: NodeKind::Tree, x: tree.0, z: tree.1, hp: 2.0, max_hp: 2.0 });
        assert!(w.crew_command(id, Job::Mine { node: 600 }).is_ok(), "2 m away: the rider walks the rest");
        let (mut phases, mut off_far) = (Vec::new(), 0.0f32);
        let mut events = Vec::new();
        for _ in 0..400 {
            events.extend(w.tick());
            let c = w.crew_worker(id).unwrap();
            assert!((c.x - x).abs() < 1e-4 && (c.z - z).abs() < 1e-4, "the truck stays where it stopped");
            if let Some(r) = c.rider {
                if phases.last() != Some(&r.phase) {
                    phases.push(r.phase);
                }
                if r.off() {
                    off_far = off_far.max((r.x - x).hypot(r.z - z));
                    assert!(w.rider_boxes(None).len() > 10, "the rider on foot is drawn");
                }
            }
            if let Some(o) = w.crew_take_outcome(id) {
                assert!(o.0, "{o:?}");
                break;
            }
        }
        assert_eq!(phases, vec![RiderPhase::Dismount, RiderPhase::WalkOut, RiderPhase::Mine, RiderPhase::WalkBack, RiderPhase::Remount]);
        assert!(off_far > 0.9, "the rider walked over to the tree: {off_far:.2} m");
        assert!(w.node(600).is_none() && w.crew_worker(id).unwrap().rider.is_none());
        let said: Vec<String> = events.iter().filter_map(|e| match &e.event { WorldEvent::Rider { label, .. } => Some(label.clone()), _ => None }).collect();
        assert!(said.iter().any(|l| l.contains("하차")) && said.iter().any(|l| l.contains("재탑승")), "{said:?}");
    }

    #[test]
    fn two_robots_meeting_head_on_stop_for_each_other_and_route_round() {
        let mut w = world();
        w.nodes.clear();
        w.drops.clear();
        let ids = w.add_crew(&[Form::Humanoid, Form::Humanoid], 1);
        let (ax, az, yaw) = (w.agent.x, w.agent.z, w.agent.yaw);
        // Put them 6 m apart on one line ahead of the first worker, each walking to where the other stands.
        let p = |d: f32| (ax + d * yaw.cos(), az + d * yaw.sin());
        let (a, b) = (p(4.0), p(10.0));
        for (id, at) in [(ids[0], a), (ids[1], b)] {
            let c = w.crew_worker_mut(id).unwrap();
            (c.x, c.z) = at;
        }
        w.crew_command(ids[0], Job::Go { x: b.0, z: b.1, reach_m: 0.3 }).unwrap();
        w.crew_command(ids[1], Job::Go { x: a.0, z: a.1, reach_m: 0.3 }).unwrap();
        let (mut closest, mut held, mut rerouted) = (f32::INFINITY, false, false);
        let mut arrived = (false, false);
        for _ in 0..1500 {
            for e in w.tick() {
                if let WorldEvent::Hold { label, .. } = &e.event {
                    held = true;
                    rerouted |= label.contains("돌아가는 길");
                }
            }
            let (c0, c1) = (w.crew_worker(ids[0]).unwrap(), w.crew_worker(ids[1]).unwrap());
            closest = closest.min((c0.x - c1.x).hypot(c0.z - c1.z) - c0.radius_m() - c1.radius_m());
            arrived.0 |= w.crew_take_outcome(ids[0]).is_some_and(|o| o.0);
            arrived.1 |= w.crew_take_outcome(ids[1]).is_some_and(|o| o.0);
            if arrived.0 && arrived.1 {
                break;
            }
        }
        assert!(held, "one saw the other in its way and stopped");
        assert!(rerouted, "the one giving way routed round");
        assert!(closest > 0.0, "their outlines never touched: closest gap {closest:.2} m");
        assert!(arrived.0 && arrived.1, "both got where they were going");
    }

    #[test]
    fn a_robot_hidden_behind_a_tree_is_not_seen_and_one_in_the_open_is() {
        let mut w = world();
        w.nodes.clear();
        let id = w.add_crew(&[Form::Humanoid], 1)[0];
        let (x, z) = (w.agent.x, w.agent.z);
        {
            let c = w.crew_worker_mut(id).unwrap();
            (c.x, c.z) = (x + 1.6, z);
        }
        let fp = w.footprint();
        let me = |dir: f32| traffic::Looker { id: 1, at: (x, z), dir, half_w: fp.half_w_m, front: fp.front_m, speed: 1.0, eye_m: 1.4, gaze: 0.0 };
        let line = |dir: f32| vec![(x + 3.0 * dir.cos(), z + 3.0 * dir.sin())];
        assert_eq!(w.robot_in_way(&me(0.0), &line(0.0), &[]).map(|s| s.0), Some(id), "in the open, on the way");
        assert_eq!(w.robot_in_way(&me(std::f32::consts::PI), &line(std::f32::consts::PI), &[]), None, "behind the eye: not seen");
        assert_eq!(w.robot_in_way(&me(1.2), &line(1.2), &[]), None, "the path passes clear of it: passed, not waited for");
        w.nodes.push(Node { id: 700, kind: NodeKind::Tree, x: x + 0.9, z, hp: 3.0, max_hp: 3.0 });
        assert!(!w.line_of_sight((x, z), (x + 1.6, z), 1.2), "the trunk is in the line");
    }

    #[test]
    fn off_the_ball_neck_scan_spots_side_hazards() {
        let mut w = world();
        w.nodes.clear();
        let id = w.add_crew(&[Form::Humanoid], 1)[0];
        let (x, z) = (w.agent.x, w.agent.z);
        // Round a corner: the way goes 0.6 m ahead, then turns right (+z is a body's right going along +x); a robot
        // stands just round it, 49° off the way.
        {
            let c = w.crew_worker_mut(id).unwrap();
            (c.x, c.z) = (x + 0.6, z + 0.7);
        }
        let fp = w.footprint();
        let me = |gaze: f32| traffic::Looker { id: 1, at: (x, z), dir: 0.0, half_w: fp.half_w_m, front: fp.front_m, speed: 1.0, eye_m: 1.4, gaze };
        let path = vec![(x + 0.6, z), (x + 0.6, z + 1.5)];
        // Looking along the way it is outside the 70° view, and the robot walks on into it.
        assert!(w.robots_seen(&me(0.0)).is_empty());
        assert_eq!(w.robot_in_way(&me(0.0), &path, &[]), None);
        // A glance 25° to the left does not see it; one to the right catches it, and the way-ahead check stops for it.
        let glance = noto_core::act::gaze::SCAN_PAN_DEG.to_radians();
        assert!(w.robots_seen(&me(glance)).is_empty());
        let seen = w.robots_seen(&me(-glance));
        assert_eq!(seen.iter().map(|n| n.id).collect::<Vec<_>>(), vec![id]);
        assert_eq!(w.robot_in_way(&me(-glance), &path, &[]).map(|s| s.0), Some(id));
        // The head camera with the neck turned the same way looks where the eye's glance looks: its view turns by minus
        // the pan on the map (`World::head_view`), as the eye does.
        let pan = noto_core::kin::KinPose { neck_pan_deg: -noto_core::act::gaze::SCAN_PAN_DEG, ..noto_core::kin::KinPose::stand(&w.body, 0.0) };
        let hc = noto_core::kin::head_camera(&w.body, &pan);
        assert!((-hc.pan_deg.to_radians() - me(-glance).eye_dir()).abs() < 1e-5);
        // With the head ahead again it stays in mind where it stood, until the next glance is due.
        let mut mind = Vec::new();
        traffic::remember(&mut mind, seen, w.time_s);
        assert_eq!(w.robot_in_way(&me(0.0), &path, &mind).map(|s| s.0), Some(id));
        w.time_s += traffic::NOTICE_KEEP_S as f64 + 0.1;
        traffic::remember(&mut mind, Vec::new(), w.time_s);
        assert!(mind.is_empty() && w.robot_in_way(&me(0.0), &path, &mind).is_none(), "an old glance is forgotten");
        // One seen walking off the way is expected where it went, not where it was.
        let walking = vec![traffic::Noticed { id, at: (x + 0.6, z + 0.7), vel: (1.5, 0.0), time_s: w.time_s - 1.0 }];
        assert_eq!(w.robot_in_way(&me(0.0), &path, &walking), None);
    }

    #[test]
    fn a_sub_worker_on_the_move_glances_each_way_and_the_first_worker_only_when_asked() {
        let mut w = world();
        w.nodes.clear();
        w.drops.clear();
        let id = w.add_crew(&[Form::Humanoid], 1)[0];
        let (cx, cz) = {
            let c = w.crew_worker(id).unwrap();
            (c.x, c.z)
        };
        w.crew_command(id, Job::Go { x: cx + 6.0, z: cz, reach_m: 0.2 }).unwrap();
        let (mut left, mut right) = (false, false);
        for _ in 0..60 {
            w.tick();
            let g = w.glance_rad(id).to_degrees();
            left |= g > 20.0;
            right |= g < -20.0;
        }
        assert!(left && right, "it glanced both ways on its way");
        w.scan_crew = false;
        w.crew_command(id, Job::Go { x: cx, z: cz, reach_m: 0.2 }).unwrap();
        for _ in 0..20 {
            w.tick();
            assert_eq!(w.glance_rad(id), 0.0);
        }
        // The first worker's head camera feeds its vision: it glances only when asked.
        let (x, z) = (w.agent.x, w.agent.z);
        let away = (0..8)
            .map(|k| k as f32 * std::f32::consts::FRAC_PI_4)
            .map(|a| (x + 4.0 * a.cos(), z + 4.0 * a.sin()))
            .find(|t| w.clone().command(crate::world::Action::MoveTo { x: t.0, z: t.1, reach_m: 0.0 }).is_ok())
            .expect("somewhere to go");
        w.command(crate::world::Action::MoveTo { x: away.0, z: away.1, reach_m: 0.0 }).unwrap();
        let mut glanced = false;
        for _ in 0..300 {
            w.tick();
            glanced |= w.glance_rad(1) != 0.0;
            if !w.busy() {
                break;
            }
        }
        assert!(!glanced);
        w.scan_first = true;
        w.command(crate::world::Action::MoveTo { x, z, reach_m: 0.0 }).unwrap();
        for _ in 0..30 {
            w.tick();
            glanced |= w.glance_rad(1).abs() > 0.3;
        }
        assert!(glanced);
    }

    #[test]
    fn the_first_worker_sees_a_peer_where_it_stands_with_what_it_says_of_itself() {
        let mut w = world();
        w.nodes.clear();
        let id = w.add_crew(&[Form::Humanoid], 1)[0];
        let (x, z, yaw) = (w.agent.x, w.agent.z, w.agent.yaw);
        let put = |w: &mut World, fwd: f32, left: f32| {
            let (s, c) = yaw.sin_cos();
            let c2 = w.crew_worker_mut(id).unwrap();
            (c2.x, c2.z) = (x + fwd * c + left * s, z + fwd * s - left * c);
        };
        let cam = crate::render::Camera::default();
        put(&mut w, 3.0, 0.8);
        let seen = w.peers_in_view(&cam);
        assert_eq!(seen.len(), 1);
        assert!((seen[0].rel_m.0 - 3.0).abs() < 1e-4 && (seen[0].rel_m.1 - 0.8).abs() < 1e-4, "{:?}", seen[0]);
        let c = w.crew_worker(id).unwrap();
        assert_eq!(seen[0].said, (c.odom.x, c.odom.z));
        // Behind the head, or past the flags' reach, it is not a landmark.
        put(&mut w, -3.0, 0.0);
        assert!(w.peers_in_view(&cam).is_empty());
        put(&mut w, FLAG_FIX_RANGE_M + 0.5, 0.0);
        assert!(w.peers_in_view(&cam).is_empty());
    }

    #[test]
    fn a_sub_worker_drifts_on_its_own_odometry_and_a_flag_in_view_puts_it_right() {
        let mut w = World::new(WorldConfig { seed: 3, parcels: Some(0), terrain: TerrainKind::Open, odometry: crate::odometry::OdomCfg { enabled: true, scale: 4.0 }, ..WorldConfig::default() }).unwrap();
        w.nodes.clear();
        w.drops.clear();
        let flags: Vec<(f32, f32)> = w.waypoints.iter().filter(|p| !p.key.starts_with("bay")).map(|p| (p.x, p.z)).collect();
        let id = w.add_crew(&[Form::Humanoid], 1)[0];
        let err = |w: &World| {
            let c = w.crew_worker(id).unwrap();
            ((c.odom.x - c.x).hypot(c.odom.z - c.z), c.pose_sigma_m(), c.odom.since_fix_m)
        };
        assert_eq!(err(&w).0, 0.0, "it starts where it was put");
        // Walking away from every flag, its belief drifts and its σ grows with the distance.
        let (cx, cz) = {
            let c = w.crew_worker(id).unwrap();
            (c.x, c.z)
        };
        let far = (0..16)
            .map(|k| k as f32 * std::f32::consts::TAU / 16.0)
            .map(|a| (cx + 8.0 * a.cos(), cz + 8.0 * a.sin()))
            .filter(|t| w.terrain.cell_of(t.0, t.1).is_some())
            .max_by(|a, b| {
                let d = |t: &(f32, f32)| flags.iter().map(|f| (f.0 - t.0).hypot(f.1 - t.1)).fold(f32::INFINITY, f32::min);
                d(a).total_cmp(&d(b))
            })
            .unwrap();
        w.crew_command(id, Job::Go { x: far.0, z: far.1, reach_m: 0.3 }).unwrap();
        assert!(run(&mut w, id, 60.0).0);
        let (drift, sigma, travelled) = err(&w);
        assert!(travelled > 4.0 && sigma > 0.05 && drift > 0.0, "{drift} {sigma} {travelled}");
        // Back to a flag: once its eye catches it near enough, the belief comes back close to where it is.
        let flag = flags.iter().min_by(|a, b| (a.0 - far.0).hypot(a.1 - far.1).total_cmp(&(b.0 - far.0).hypot(b.1 - far.1))).copied().unwrap();
        w.crew_command(id, Job::Go { x: flag.0, z: flag.1, reach_m: 2.0 }).unwrap();
        assert!(run(&mut w, id, 90.0).0);
        let (after, sigma_after, _) = err(&w);
        assert!(sigma_after < sigma && after < drift.max(0.05), "{after} vs {drift}; σ {sigma_after} vs {sigma}");
    }

    #[test]
    fn the_first_worker_delivering_to_a_bay_stages_its_load_on_the_ground() {
        let mut w = world();
        w.drops.clear();
        let (ax, az) = (w.agent.x, w.agent.z);
        let bay = w.add_bay(ax + 0.8, az, (ax + 2.0, az), "상차장");
        crate::world::add_stock(&mut w.agent.carried, ItemKind::Log, 2.0);
        let ev = w.command(crate::world::Action::Deliver { waypoint: bay, item: None }).unwrap();
        let mut all = ev;
        for _ in 0..600 {
            all.extend(w.tick());
        }
        assert!(all.iter().any(|e| matches!(e.event, WorldEvent::Delivered { .. })), "the first worker's driver still hears the delivery");
        assert_eq!(w.staged_at(bay).len(), 2, "two logs on the ground in the bay's row");
        assert_eq!(w.waypoint(bay).unwrap().stock_of(ItemKind::Log), 0.0);
    }
}
