use std::sync::Mutex;
use std::collections::{BTreeMap, HashSet};

use noto_core::balance::{BalanceReport, Body, DriveLimits, FeedForward, Posture, Source, GRIP_SAFETY};
use noto_core::dynamics::{self, BodyPose, DynCfg, DynDigest, Rollout};
use noto_core::{compute_body, BalanceSummary, BodyInputs, BodyLimits, BodyPlan, BodyStatus, ConfigError, Hand, NotoConfig};
use serde::{Deserialize, Serialize};

use crate::clear::{self, ClearMap, Footprint};
use crate::fixture::Fixture;
use crate::kinds::{Ground, ItemKind, NodeKind, MIN_PIECE};
use crate::nav::{self, Cell, EdgePlan, Goal};
use crate::odometry::{OdomCfg, Odometry};
use crate::render;
use crate::rng::Rng;
use crate::site::{self, Site, SiteCfg, Support};
use crate::snapshot::*;
use crate::terrain::{cell_center, FlatZone, RuggedLayout, Terrain, TerrainKind, CELL_M};

mod acts;
pub mod cargo;
mod eye;
pub mod hands;
mod motion;
pub mod steps;
pub mod traffic;
mod turns;
pub use acts::{camera_of, material_of, ActPlay, GazeHold, PushState, WalkState, GET_UP_S, PUSH_START_SLACK_M, SIDE_SHUFFLE_MAX_M};
pub use cargo::CargoRide;
pub use steps::{StepSeen, Stride};
pub(crate) use motion::Follow;
use motion::{PoseCheck, SegRun};
pub use traffic::{RiderPhase, Rider, RobotOutline, RIDER_REACH_M, RIDER_STOP_M};
pub use eye::{chassis_accel_rms, exposure_s, profile, roughness_gd, sensor_timing, sensor_timing_at, strike_peak_mps2, strike_scale, EyeState, HeadView, SensorTiming, Shake, EXPOSURE_MAX_S, SCENE_LUX};

pub const TICK_S: f32 = 0.1;
pub const MINE_REACH_M: f32 = 0.9;
pub const PICKUP_REACH_M: f32 = 0.7;
pub const DELIVER_REACH_M: f32 = 1.6;
pub const PICKUP_TIME_S: f32 = 1.0;
pub const DELIVER_TIME_S: f32 = 1.0;
pub const MANIPULATION_POWER_W: f64 = 30.0;
pub const TURN_RATE_RAD_S: f32 = 1.2;
const FACE_DONE_RAD: f32 = 0.03;
pub const TWO_ARM_LIFT_FACTOR: f32 = 1.5;
pub const ACCEL_TIME_S: f32 = 0.5;
const BALANCE_RESAMPLE_TICKS: u64 = 50;
pub const DROP_TIME_S: f32 = 1.0;
pub const DROP_AHEAD_M: f32 = 0.45;
pub const AMOUNT_EPS: f32 = 1e-3;
pub const CAMP_PARCEL_AHEAD_M: f32 = 2.0;
pub const CAMP_PARCEL_RIGHT_M: f32 = 0.30;
pub const PARCELS_PER_CELL: f32 = 1.0 / 700.0;
/// The gap in the curb wall (cells).
pub const GAP_CELLS: usize = 5;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldConfig {
    pub seed: u64,
    pub cells: usize,
    #[serde(alias = "hand")]
    pub noto: NotoConfig,
    #[serde(default)]
    pub body: BodyPlan,
    #[serde(default)]
    pub terrain: TerrainKind,
    #[serde(default)]
    pub odometry: OdomCfg,
    #[serde(default)]
    pub parcels: Option<usize>,
    #[serde(default)]
    pub lighting: crate::light::Lighting,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub site: Option<SiteCfg>,
}

impl Default for WorldConfig {
    fn default() -> Self {
        WorldConfig { seed: 7, cells: 64, noto: NotoConfig::preset(noto_core::FingerCount::Five), body: BodyPlan::standard(), terrain: TerrainKind::Open, odometry: OdomCfg::default(), parcels: None, lighting: crate::light::Lighting::Flat, site: None }
    }
}

#[derive(Clone, Debug)]
pub struct BalanceEvent {
    pub tick: u64,
    pub time_s: f64,
    pub state: String,
    pub report: BalanceReport,
    pub dynamics: Option<DynDigest>,
    pub posture: Posture,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub id: u32,
    pub kind: NodeKind,
    pub x: f32,
    pub z: f32,
    pub hp: f32,
    pub max_hp: f32,
}

#[derive(Clone, Debug)]
pub struct DropItem {
    pub id: u32,
    pub kind: ItemKind,
    pub x: f32,
    pub z: f32,
    pub amount: f32,
    pub support: Support,
    pub y_m: f32,
}

impl DropItem {
    pub fn whole(id: u32, kind: ItemKind, x: f32, z: f32) -> DropItem {
        DropItem { id, kind, x, z, amount: 1.0, support: Support::Ground, y_m: 0.0 }
    }

    pub fn on(id: u32, kind: ItemKind, x: f32, z: f32, support: Support, y_m: f32) -> DropItem {
        DropItem { id, kind, x, z, amount: 1.0, support, y_m }
    }

    pub fn mass_kg(&self) -> f32 {
        self.kind.spec().mass_kg * self.amount
    }
}

#[derive(Clone, Debug)]
pub struct Waypoint {
    pub id: u32,
    pub key: String,
    pub label_ko: String,
    pub x: f32,
    pub z: f32,
    pub stock: Vec<(ItemKind, f32)>,
}

impl Waypoint {
    pub fn stock_of(&self, item: ItemKind) -> f32 {
        self.stock.iter().filter(|(k, _)| *k == item).map(|(_, n)| *n).sum()
    }
}

#[derive(Clone, Debug)]
pub struct Agent {
    pub x: f32,
    pub z: f32,
    pub yaw: f32,
    pub carried: Vec<(ItemKind, f32)>,
    pub energy_used_j: f64,
    pub strikes: u32,
    pub speed_mps: f32,
    pub grip_margin: Option<f32>,
    pub terrain_margin: Option<f32>,
    strike_acc: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Action {
    MoveTo { x: f32, z: f32, reach_m: f32 },
    Turn { yaw_deg: f32 },
    Face { x: f32, z: f32 },
    Look { x: f32, z: f32 },
    Mine { node: u32 },
    PickUp {
        drop: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        aim: Option<(f32, f32)>,
        #[serde(default, skip_serializing_if = "is_false")]
        side: bool,
        #[serde(default, skip_serializing_if = "is_zero")]
        memory_sigma_m: f32,
    },
    Deliver { waypoint: u32, item: Option<ItemKind> },
    Drop { item: Option<ItemKind>, amount: Option<f32> },
    Split { drop: u32, portion: f32 },
    Push { drop: u32, x: f32, z: f32 },
}

fn is_false(b: &bool) -> bool {
    !*b
}

fn is_zero(x: &f32) -> bool {
    *x == 0.0
}

impl Action {
    pub fn pick(drop: u32) -> Action {
        Action::PickUp { drop, aim: None, side: false, memory_sigma_m: 0.0 }
    }

    pub fn label(&self) -> String {
        match self {
            Action::MoveTo { x, z, reach_m } => format!("move_to({x:.1},{z:.1},reach={reach_m:.1})"),
            Action::Turn { yaw_deg } => format!("turn({yaw_deg:.0})"),
            Action::Face { x, z } => format!("face({x:.1},{z:.1})"),
            Action::Look { x, z } => format!("look({x:.1},{z:.1})"),
            Action::Mine { node } => format!("mine({})", node_id(*node)),
            Action::PickUp { drop, side, .. } => format!("{}({})", if *side { "pick_up_side" } else { "pick_up" }, drop_id(*drop)),
            Action::Deliver { waypoint, item } => match item {
                Some(i) => format!("deliver({},{})", waypoint_id(*waypoint), i.id()),
                None => format!("deliver({})", waypoint_id(*waypoint)),
            },
            Action::Drop { item, amount } => match (item, amount) {
                (Some(i), Some(a)) => format!("drop({}×{a:.2})", i.id()),
                (Some(i), None) => format!("drop({})", i.id()),
                (None, _) => "drop(all)".to_string(),
            },
            Action::Split { drop, portion } => format!("split({},{portion:.2})", drop_id(*drop)),
            Action::Push { drop, x, z } => format!("push({}→{x:.1},{z:.1})", drop_id(*drop)),
        }
    }
}

pub fn amount_text(a: f32) -> String {
    if (a - a.round()).abs() < AMOUNT_EPS {
        format!("{}", a.round() as i64)
    } else {
        let s = format!("{a:.2}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "code", rename_all = "snake_case")]
pub enum ActionError {
    Busy,
    BatteryDepleted,
    UnknownTarget,
    NoPath,
    OutOfReach { distance_m: f32, reach_m: f32 },
    GraspUnstable { margin: f32 },
    CapacityExceeded { carried_kg: f32, item_kg: f32, capacity_kg: f32 },
    NothingToDeliver,
    LiftOverload { limiting: String, margin: f32 },
    NothingToDrop,
    NotDivisible,
    PieceTooSmall { amount: f32 },
    CannotReach { lowest_m: f32, need_m: f32, reason: String },
    CannotPush { reason: String },
    Fallen,
}

impl ActionError {
    pub fn code(&self) -> &'static str {
        match self {
            ActionError::Busy => "busy",
            ActionError::BatteryDepleted => "battery_depleted",
            ActionError::UnknownTarget => "unknown_target",
            ActionError::NoPath => "no_path",
            ActionError::OutOfReach { .. } => "out_of_reach",
            ActionError::GraspUnstable { .. } => "grasp_unstable",
            ActionError::CapacityExceeded { .. } => "capacity_exceeded",
            ActionError::NothingToDeliver => "nothing_to_deliver",
            ActionError::LiftOverload { .. } => "lift_overload",
            ActionError::NothingToDrop => "nothing_to_drop",
            ActionError::NotDivisible => "not_divisible",
            ActionError::PieceTooSmall { .. } => "piece_too_small",
            ActionError::CannotReach { .. } => "reach_ground",
            ActionError::CannotPush { .. } => "push_impossible",
            ActionError::Fallen => "fallen",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorldEvent {
    ActionStarted { action: String },
    ActionDone { action: String, ok: bool, reason: Option<String> },
    CurbClimbed { height_m: f32 },
    NodeBroken { node: String, kind: NodeKind, drops: Vec<String> },
    ItemPicked { drop: String, item: ItemKind, mass_kg: f32, grip_margin: f32, amount: f32 },
    Delivered { waypoint: String, item: ItemKind, amount: f32 },
    ItemDropped { item: ItemKind, amount: f32, waypoint: Option<String>, drops: Vec<String> },
    ItemSplit { drop: String, piece: String, item: ItemKind, amount: f32, left: f32 },
    BatteryDepleted,
    Act { act: String, kind: String, label: String, ok: bool, value: Option<f32>, parts: Vec<String> },
    /// What a sub-worker did (`crew`): felled a node, picked up, staged at a bay, delivered. The first worker's
    /// driver hears only its own `NodeBroken` / `Delivered`.
    Crew { worker: u32, kind: String, label: String, item: Option<ItemKind>, amount: f32, ok: bool },
    /// A lift truck's rider stepping down, walking, felling, climbing on (`traffic::Rider`).
    Rider { worker: u32, phase: String, label: String },
    /// A robot stopped for another its eye saw in its way (`gap_m` between the outlines), or routed round it.
    Hold { worker: u32, other: u32, gap_m: f32, label: String },
    /// A robot ran into something solid.
    Bump { worker: u32, what: String, peak_n: f32, a_g: f32, speed_mps: f32, fell: bool, label: String },
    /// A load started to slide on its support.
    CargoSlip { worker: u32, item: ItemKind, shift_m: f32, use_ratio: f32, label: String },
    /// A piece of the load came off and lies on the ground as `drop`.
    CargoLost { worker: u32, item: ItemKind, amount: f32, drop: String, cause: String, label: String },
    /// A step ahead, seen in time to bend the knees for it.
    StepSeen { worker: u32, height_m: f32, dist_m: f32, crouch_m: f32, source: String, label: String },
    /// A step taken with a load on.
    StepTaken { worker: u32, height_m: f32, crouch_m: f32, a_z: f32, label: String },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TimedEvent {
    pub tick: u64,
    pub time_s: f64,
    pub event: WorldEvent,
}

#[derive(Clone, Debug)]
pub(crate) struct MoveState {
    pub(crate) edges: Vec<EdgePlan>,
    idx: usize,
    pause_left: f32,
    progress_s: f32,
    announced_pause: bool,
    elapsed_s: f32,
    pub(crate) follow: Follow,
    /// Stopped for a robot in the way: for how long, for which, and how many times it routed round one.
    held_s: f32,
    held_by: Option<u32>,
    reroutes: u32,
    goal: Option<Goal>,
    /// How many times its way was shut and it planned again (`motion`).
    replans: u32,
    /// Where it last was a good way from, and when it was there (`motion::STUCK_S`).
    anchor: Option<((f32, f32), f64)>,
}

impl MoveState {
    pub(crate) fn new(edges: Vec<EdgePlan>, follow: Follow, goal: Goal) -> MoveState {
        MoveState { edges, idx: 0, pause_left: 0.0, progress_s: 0.0, announced_pause: false, elapsed_s: 0.0, follow, anchor: None, held_s: 0.0, held_by: None, reroutes: 0, goal: Some(goal), replans: 0 }
    }
}

#[derive(Clone, Debug)]
enum WorkKind {
    Mine { node: u32 },
    Look { at: (f32, f32) },
    Pick { drop: u32, two_arms: bool, face: Option<(f32, f32)> },
    Deliver { waypoint: u32, item: Option<ItemKind> },
    Drop { item: Option<ItemKind>, amount: Option<f32>, waypoint: Option<u32> },
    Split { drop: u32, portion: f32 },
}

#[derive(Clone, Debug)]
struct WorkState {
    kind: WorkKind,
    remaining_s: f32,
}

#[derive(Clone, Debug)]
enum Active {
    Move(MoveState),
    Turn { target: f32, run: SegRun },
    Work(WorkState),
    Push(PushState),
}

pub struct World {
    pub cfg: WorldConfig,
    pub hand: Hand,
    pub terrain: Terrain,
    pub nodes: Vec<Node>,
    pub drops: Vec<DropItem>,
    pub waypoints: Vec<Waypoint>,
    pub agent: Agent,
    pub tick: u64,
    pub time_s: f64,
    grasps: Vec<GraspInfo>,
    active: Option<(Action, Active)>,
    rng: Rng,
    next_drop: u32,
    pub body: Body,
    limits: Mutex<Option<(u32, DriveLimits)>>,
    pub balance: Option<(String, BalanceReport)>,
    balance_since: u64,
    balance_events: Vec<BalanceEvent>,
    pub ff_gains: BTreeMap<Source, f32>,
    pub ff_enabled: bool,
    /// The friction the hands expect of an item on pads in a state (`grip_key`), learned from what they felt of it:
    /// the grasp plans with it instead of the table (`GraspInput::mu_prior`).
    pub grip_mu: BTreeMap<String, f32>,
    pub heat: BTreeMap<String, f32>,
    pub wear: BTreeMap<String, f32>,
    pub motion: Option<Motion>,
    last_posture: Posture,
    pub layout: Option<RuggedLayout>,
    pub act: Option<ActPlay>,
    pub last_act: Option<ActPlay>,
    act_rng: Rng,
    pub fallen_until: Option<f64>,
    pub walk: WalkState,
    pub odom: Odometry,
    odom_over: f32,
    pub eye: EyeState,
    side_cache: Mutex<std::collections::HashMap<(ItemKind, i32, i32), Option<(f32, f32)>>>,
    pub site: Option<Site>,
    pub fixtures: Vec<Fixture>,
    pub gaze: Option<GazeHold>,
    /// The eye is the fork carriage camera instead of the rider's head (a wheeled base checking an item at the floor).
    pub fork_cam: bool,
    /// The clearance maps for the nodes standing now (`clear::ClearMap`): one per kind of body, keyed by both.
    room: Mutex<Vec<(u64, std::sync::Arc<ClearMap>)>>,
    /// Sub-workers sharing this world with the first worker (`crew`).
    pub crew: Vec<crate::crew::Worker>,
    /// A lift truck's rider on foot, felling a node (`traffic::Rider`).
    pub rider: Option<traffic::Rider>,
    /// The way each loading bay faces (bay waypoint id, yaw): its row of slots lies across that.
    pub(crate) bay_faces: Vec<(u32, f32)>,
    /// How the load is riding (`cargo`), and the steps ahead (`steps`).
    pub ride: CargoRide,
    pub stride: Stride,
    /// The sub-worker standing in for the first worker in the motion code this moment, and the robots it must miss.
    pub(crate) crew_turn: Option<u32>,
    pub(crate) turn_others: Vec<traffic::RobotOutline>,
    /// Off-the-ball glances (`gaze::scan_pan_deg`) on the move with empty hands: the sub-workers' eyes, which only keep
    /// them clear of each other, glance by default; the first worker's head camera, whose frames its vision reads,
    /// only when asked.
    pub scan_crew: bool,
    pub scan_first: bool,
    /// The robots the first worker's eye saw a moment ago (`traffic::Noticed`).
    pub noticed: Vec<traffic::Noticed>,
}

impl Clone for World {
    fn clone(&self) -> World {
        World {
            cfg: self.cfg.clone(),
            hand: self.hand.clone(),
            terrain: self.terrain.clone(),
            nodes: self.nodes.clone(),
            drops: self.drops.clone(),
            waypoints: self.waypoints.clone(),
            agent: self.agent.clone(),
            tick: self.tick,
            time_s: self.time_s,
            grasps: self.grasps.clone(),
            active: self.active.clone(),
            rng: self.rng.clone(),
            next_drop: self.next_drop,
            body: self.body.clone(),
            limits: Mutex::new(self.limits.lock().unwrap_or_else(|e| e.into_inner()).clone()),
            balance: self.balance.clone(),
            balance_since: self.balance_since,
            balance_events: self.balance_events.clone(),
            ff_gains: self.ff_gains.clone(),
            ff_enabled: self.ff_enabled,
            grip_mu: self.grip_mu.clone(),
            heat: self.heat.clone(),
            wear: self.wear.clone(),
            motion: self.motion.clone(),
            last_posture: self.last_posture,
            layout: self.layout.clone(),
            act: self.act.clone(),
            last_act: self.last_act.clone(),
            act_rng: self.act_rng.clone(),
            fallen_until: self.fallen_until,
            walk: self.walk.clone(),
            odom: self.odom.clone(),
            odom_over: self.odom_over,
            eye: self.eye.clone(),
            side_cache: Mutex::new(self.side_cache.lock().unwrap_or_else(|e| e.into_inner()).clone()),
            site: self.site.clone(),
            fixtures: self.fixtures.clone(),
            gaze: self.gaze,
            fork_cam: self.fork_cam,
            room: Mutex::new(self.room.lock().unwrap_or_else(|e| e.into_inner()).clone()),
            crew: self.crew.clone(),
            rider: self.rider,
            bay_faces: self.bay_faces.clone(),
            ride: self.ride,
            stride: self.stride.clone(),
            crew_turn: self.crew_turn,
            turn_others: self.turn_others.clone(),
            scan_crew: self.scan_crew,
            scan_first: self.scan_first,
            noticed: self.noticed.clone(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Motion {
    pub rollout: Rollout,
    pub posture: Posture,
    pub start_s: f64,
    pub state: String,
}

fn wrap_angle(a: f32) -> f32 {
    let mut a = a % std::f32::consts::TAU;
    if a > std::f32::consts::PI {
        a -= std::f32::consts::TAU;
    } else if a < -std::f32::consts::PI {
        a += std::f32::consts::TAU;
    }
    a
}

/// The key a learned grip goes by (`World::grip_mu`): the item and the state of the pads on it (`PadSurface::id`).
pub fn grip_key(item: &str, pads: &str) -> String {
    format!("{item}:{pads}")
}

pub(crate) fn add_stock(list: &mut Vec<(ItemKind, f32)>, kind: ItemKind, n: f32) {
    if let Some(e) = list.iter_mut().find(|(k, _)| *k == kind) {
        e.1 += n;
        if (e.1 - e.1.round()).abs() < AMOUNT_EPS {
            e.1 = e.1.round();
        }
    } else {
        list.push((kind, n));
    }
}

impl World {
    pub fn new(cfg: WorldConfig) -> Result<World, ConfigError> {
        let hand = cfg.noto.build()?.scaled(cfg.body.k());
        let cells = cfg.cells.clamp(32, 160);
        let size = cells as f32 * CELL_M;

        let base_pos = (0.14 * size, 0.14 * size);
        let wh_pos = (0.78 * size, 0.66 * size);
        let wall_cx = (0.46 * cells as f32).round() as usize;
        let wall_x = (wall_cx as f32 + 0.5) * CELL_M;

        let mut zones: Vec<FlatZone> = vec![(base_pos.0, base_pos.1, 3.5), (wh_pos.0, wh_pos.1, 3.5)];
        let mut zc = 1.0;
        while zc < size {
            zones.push((wall_x, zc, 1.4));
            zc += 2.0;
        }
        // Wide enough for a lift truck to drive through without scraping: a 1 m gap fits one only to the centimetre.
        let gap_from = (0.86 * cells as f32) as usize;
        let gap_to = gap_from + GAP_CELLS;
        let gap = (wall_x, (gap_from as f32 + 0.5 * GAP_CELLS as f32) * CELL_M);
        let way = [base_pos, (base_pos.0, gap.1), gap, wh_pos];
        let site_plan = cfg.site.as_ref().map(|sc| {
            let keep = site::Keep { base: base_pos, warehouse: wh_pos, wall_cx, way: way.to_vec(), way_m: 2.2 };
            site::plan(cfg.seed, cells, sc, &keep)
        });
        let site_zones = site_plan.as_ref().map_or(Vec::new(), |p| p.zones());
        zones.extend(site_zones.iter().copied());
        let mut terrain = Terrain::generate(cfg.seed, cells, cells, &zones);

        for (px, pz) in [base_pos, wh_pos] {
            for cz in 0..cells {
                for cx in 0..cells {
                    let (x, z) = cell_center(cx, cz);
                    if ((x - px).powi(2) + (z - pz).powi(2)).sqrt() <= 2.0 {
                        terrain.set_ground(cx, cz, Ground::Gravel);
                    }
                }
            }
        }

        for cz in 0..cells {
            if (gap_from..gap_to).contains(&cz) {
                continue;
            }
            let f = cz as f32 / cells as f32;
            let cm = if f < 0.30 {
                5
            } else if f < 0.55 {
                10
            } else {
                15
            };
            terrain.set_curb_cm(wall_cx, cz, cm);
        }

        let layout = (cfg.terrain == TerrainKind::Rugged).then(|| {
            let mut clear: Vec<FlatZone> = vec![(base_pos.0, base_pos.1, 3.5), (wh_pos.0, wh_pos.1, 3.5)];
            clear.extend(site_zones.iter().copied());
            terrain.stamp_rugged(cfg.seed, &clear, &way, 2.2)
        });
        let mut next_fixture = 1u32;
        let (mut site, fixtures) = match &site_plan {
            Some(p) => {
                let (s, f) = site::build(p, &mut terrain, &mut next_fixture);
                (Some(s), f)
            }
            None => (None, Vec::new()),
        };
        let reserved: Vec<site::Rect> = site.as_ref().map_or(Vec::new(), |s| s.reserved.clone());
        let on_site = |cx: usize, cz: usize| reserved.iter().any(|r| r.contains((cx, cz)));

        let mut rng = Rng::new(cfg.seed ^ 0xA5A5_5A5A_1234_8765);
        let mut occupied: HashSet<Cell> = HashSet::new();
        let mut nodes = Vec::new();
        let mut next_node = 1u32;
        let wanted = [
            (NodeKind::Tree, (cells * cells / 140).max(4)),
            (NodeKind::Boulder, (cells * cells / 300).max(3)),
            (NodeKind::CoalOre, (cells * cells / 520).max(2)),
            (NodeKind::IronOre, (cells * cells / 700).max(2)),
        ];
        let in_zone = |x: f32, z: f32| {
            [base_pos, wh_pos].iter().any(|&(px, pz)| ((x - px).powi(2) + (z - pz).powi(2)).sqrt() < 3.8)
        };
        for (kind, count) in wanted {
            let mut placed = 0;
            let mut tries = 0;
            while placed < count && tries < 6000 {
                tries += 1;
                let cx = rng.below(cells as u32) as usize;
                let cz = rng.below(cells as u32) as usize;
                let (x, z) = cell_center(cx, cz);
                let h = terrain.height_cm(cx, cz);
                let strict = tries < 3000;
                let height_ok = match kind {
                    NodeKind::Tree => h < 70,
                    NodeKind::Boulder => true,
                    NodeKind::CoalOre | NodeKind::IronOre => !strict || h >= 20,
                };
                let near_wall = cx.abs_diff(wall_cx) <= 1;
                if !height_ok
                    || near_wall
                    || on_site(cx, cz)
                    || in_zone(x, z)
                    || terrain.curb_m(cx, cz) > 0.0
                    || terrain.is_water(cx, cz)
                    || occupied.contains(&(cx, cz))
                {
                    continue;
                }
                occupied.insert((cx, cz));
                let hp = kind.break_time_s();
                nodes.push(Node { id: next_node, kind, x, z, hp, max_hp: hp });
                next_node += 1;
                placed += 1;
            }
        }
        let site_blocked: HashSet<Cell> = crate::fixture::blocked(&fixtures).collect();
        if layout.is_some() {
            let wheel = noto_core::BaseKind::Wheel.spec();
            let start = terrain.cell_of(base_pos.0, base_pos.1).unwrap_or((0, 0));
            let taken: HashSet<Cell> = occupied.union(&site_blocked).copied().collect();
            let times = nav::travel_times(&wheel, &terrain, &|x, z| taken.contains(&(x, z)), start, f32::INFINITY);
            let reachable = |cx: usize, cz: usize| -> bool {
                (-1i32..=1).any(|dz| (-1i32..=1).any(|dx| {
                    let (nx, nz) = (cx as i32 + dx, cz as i32 + dz);
                    terrain.in_bounds(nx, nz) && times[terrain.idx(nx as usize, nz as usize)].is_finite()
                }))
            };
            for kind in NodeKind::ALL {
                let mut ok = nodes.iter().filter(|n| n.kind == kind).filter(|n| terrain.cell_of(n.x, n.z).map_or(false, |(cx, cz)| reachable(cx, cz))).count();
                let mut tries = 0;
                while ok < 2 && tries < 4000 {
                    tries += 1;
                    let (cx, cz) = (rng.below(cells as u32) as usize, rng.below(cells as u32) as usize);
                    let (x, z) = cell_center(cx, cz);
                    let i = terrain.idx(cx, cz);
                    if !times[i].is_finite() || occupied.contains(&(cx, cz)) || in_zone(x, z) || cx.abs_diff(wall_cx) <= 1 || terrain.curb_m(cx, cz) > 0.0 || on_site(cx, cz) {
                        continue;
                    }
                    if kind == NodeKind::Tree && terrain.height_cm(cx, cz) >= 70 {
                        continue;
                    }
                    occupied.insert((cx, cz));
                    let hp = kind.break_time_s();
                    nodes.push(Node { id: next_node, kind, x, z, hp, max_hp: hp });
                    next_node += 1;
                    ok += 1;
                }
            }
        }

        let waypoints = vec![
            Waypoint { id: 1, key: "base_camp".into(), label_ko: "베이스캠프".into(), x: base_pos.0, z: base_pos.1, stock: vec![] },
            Waypoint { id: 2, key: "warehouse".into(), label_ko: "창고".into(), x: wh_pos.0, z: wh_pos.1, stock: vec![] },
        ];
        let yaw = (wh_pos.1 - base_pos.1).atan2(wh_pos.0 - base_pos.0);
        let agent = Agent {
            x: base_pos.0,
            z: base_pos.1,
            yaw,
            carried: vec![],
            energy_used_j: 0.0,
            strikes: 0,
            speed_mps: 0.0,
            grip_margin: None,
            terrain_margin: None,
            strike_acc: 0.0,
        };

        if let Some(s) = site.as_mut() {
            let taken: HashSet<Cell> = occupied.union(&site_blocked).copied().collect();
            let start = terrain.cell_of(base_pos.0, base_pos.1).unwrap_or((0, 0));
            let legs = nav::travel_times(&noto_core::BaseKind::HumanFoot.spec(), &terrain, &|x, z| taken.contains(&(x, z)), start, f32::INFINITY);
            let wheels = nav::travel_times(&noto_core::BaseKind::Wheel.spec(), &terrain, &|x, z| taken.contains(&(x, z)), start, f32::INFINITY);
            let on = |times: &[f32], cells: &[Cell]| cells.iter().any(|c| times[terrain.idx(c.0, c.1)].is_finite());
            for h in s.houses.iter_mut() {
                h.reachable_legs = on(&legs, &h.porch);
                h.reachable_wheels = on(&wheels, &h.porch);
            }
            let lost: Vec<u32> = s.houses.iter().filter(|h| !h.reachable_legs).map(|h| h.id).collect();
            s.slots.retain(|sl| !matches!(sl.owner, crate::fixture::Owner::House(id) if lost.contains(&id)));
        }

        let mut drops = Vec::new();
        let mut next_drop = 1u32;
        let n_parcels = cfg.parcels.unwrap_or(((cells * cells) as f32 * PARCELS_PER_CELL).round().max(3.0) as usize);
        if n_parcels > 0 {
            let mut prng = Rng::new(cfg.seed ^ 0x9A2C_E1B0_77D3_4F15);
            let (s, c) = yaw.sin_cos();
            let (px, pz) = (base_pos.0 + CAMP_PARCEL_AHEAD_M * c - CAMP_PARCEL_RIGHT_M * s, base_pos.1 + CAMP_PARCEL_AHEAD_M * s + CAMP_PARCEL_RIGHT_M * c);
            drops.push(DropItem::whole(next_drop, ItemKind::Parcel, px, pz));
            next_drop += 1;
            if let Some(s) = &site {
                for sl in site::choose_slots(s, n_parcels - drops.len(), &mut prng) {
                    drops.push(DropItem::on(next_drop, ItemKind::Parcel, sl.x, sl.z, sl.support, sl.y_m));
                    next_drop += 1;
                }
            }
            let wheel = noto_core::BaseKind::Wheel.spec();
            let start = terrain.cell_of(base_pos.0, base_pos.1).unwrap_or((0, 0));
            let taken: HashSet<Cell> = occupied.union(&site_blocked).copied().collect();
            let times = nav::travel_times(&wheel, &terrain, &|x, z| taken.contains(&(x, z)), start, f32::INFINITY);
            let mut tries = 0;
            while drops.len() < n_parcels && tries < 4000 {
                tries += 1;
                let (cx, cz) = (prng.below(cells as u32) as usize, prng.below(cells as u32) as usize);
                let (x, z) = cell_center(cx, cz);
                if !times[terrain.idx(cx, cz)].is_finite() || occupied.contains(&(cx, cz)) || in_zone(x, z) || cx.abs_diff(wall_cx) <= 1 || terrain.curb_m(cx, cz) > 0.0 || on_site(cx, cz) {
                    continue;
                }
                occupied.insert((cx, cz));
                let (jx, jz) = (prng.range_f32(-0.12, 0.12), prng.range_f32(-0.12, 0.12));
                drops.push(DropItem::whole(next_drop, ItemKind::Parcel, x + jx, z + jz));
                next_drop += 1;
            }
        }

        let grasps = ItemKind::ALL.iter().map(|&k| top_grasp_info(&hand, k, 1.0)).collect();

        let (cfg_body, base_kind, fingers, cfg_seed) = (cfg.body, cfg.noto.base, cfg.noto.fingers, cfg.seed);
        let odom = Odometry::new(cfg.odometry, cfg_seed, base_kind == noto_core::BaseKind::HumanFoot, (agent.x, agent.z, agent.yaw));
        Ok(World {
            cfg,
            hand,
            terrain,
            nodes,
            drops,
            waypoints,
            agent,
            tick: 0,
            time_s: 0.0,
            grasps,
            active: None,
            rng,
            next_drop,
            body: Body::new(&cfg_body, base_kind, fingers),
            limits: Mutex::new(None),
            balance: None,
            balance_since: 0,
            balance_events: Vec::new(),
            ff_gains: BTreeMap::new(),
            ff_enabled: true,
            grip_mu: BTreeMap::new(),
            heat: BTreeMap::new(),
            wear: BTreeMap::new(),
            motion: None,
            last_posture: Posture::quiet(0.0),
            layout,
            act: None,
            last_act: None,
            act_rng: Rng::new(cfg_seed ^ 0x51D3_A6C2_7F10_88E5),
            fallen_until: None,
            walk: WalkState::default(),
            odom,
            odom_over: 0.0,
            eye: EyeState::default(),
            side_cache: Mutex::new(std::collections::HashMap::new()),
            site,
            fixtures,
            gaze: None,
            fork_cam: false,
            room: Mutex::new(Vec::new()),
            crew: Vec::new(),
            rider: None,
            bay_faces: Vec::new(),
            ride: CargoRide::default(),
            stride: Stride::default(),
            crew_turn: None,
            turn_others: Vec::new(),
            scan_crew: true,
            scan_first: false,
            noticed: Vec::new(),
        })
    }

    pub fn reset_world_keep_body(&mut self) -> Result<(), ConfigError> {
        let mut fresh = World::new(self.cfg.clone())?;
        let crew: Vec<(crate::crew::Form, u8)> = self.crew.iter().map(|w| (w.form, w.cameras)).collect();
        for (form, cameras) in crew {
            fresh.add_crew(&[form], cameras);
        }
        fresh.agent.energy_used_j = self.agent.energy_used_j;
        fresh.agent.strikes = self.agent.strikes;
        fresh.agent.strike_acc = self.agent.strike_acc;
        fresh.heat = std::mem::take(&mut self.heat);
        fresh.wear = std::mem::take(&mut self.wear);
        fresh.ff_gains = std::mem::take(&mut self.ff_gains);
        fresh.ff_enabled = self.ff_enabled;
        fresh.grip_mu = std::mem::take(&mut self.grip_mu);
        fresh.scan_crew = self.scan_crew;
        fresh.scan_first = self.scan_first;
        fresh.tick = self.tick;
        fresh.time_s = self.time_s;
        *self = fresh;
        Ok(())
    }

    pub fn reset_body(&mut self) {
        self.agent.energy_used_j = 0.0;
        self.agent.strikes = 0;
        self.agent.strike_acc = 0.0;
        self.agent.grip_margin = None;
        self.agent.terrain_margin = None;
        self.heat.clear();
        self.wear.clear();
        self.balance = None;
        self.motion = None;
        self.last_posture = Posture::quiet(self.carried_kg());
        self.fallen_until = None;
        self.gaze = None;
    }

    pub fn grasp_info(&self, item: ItemKind) -> &GraspInfo {
        self.grasps.iter().find(|g| g.item == item).expect("every item has a grasp entry")
    }

    pub fn grasp_table(&self) -> &[GraspInfo] {
        &self.grasps
    }

    pub fn believed_pose(&self) -> (f32, f32, f32) {
        self.odom.pose()
    }

    fn true_pose(&self) -> (f32, f32, f32) {
        (self.agent.x, self.agent.z, self.agent.yaw)
    }

    pub fn belief_of(&self, x: f32, z: f32) -> (f32, f32) {
        self.odom.belief_of(self.true_pose(), (x, z))
    }

    pub fn true_of(&self, x: f32, z: f32) -> (f32, f32) {
        self.odom.true_of(self.true_pose(), (x, z))
    }

    pub fn true_yaw_of(&self, yaw: f32) -> f32 {
        self.odom.true_yaw(self.agent.yaw, yaw)
    }

    pub fn belief_error_m(&self) -> f32 {
        (self.odom.x - self.agent.x).hypot(self.odom.z - self.agent.z)
    }

    pub fn relocalize(&mut self, dx: f32, dz: f32) {
        self.odom.correct(dx, dz);
    }

    pub fn node(&self, id: u32) -> Option<&Node> {
        self.nodes.iter().find(|n| n.id == id)
    }

    pub fn drop_item(&self, id: u32) -> Option<&DropItem> {
        self.drops.iter().find(|d| d.id == id)
    }

    pub fn waypoint(&self, id: u32) -> Option<&Waypoint> {
        self.waypoints.iter().find(|w| w.id == id)
    }

    pub fn waypoint_by_key(&self, key: &str) -> Option<&Waypoint> {
        self.waypoints.iter().find(|w| w.key == key)
    }

    pub fn base_spec(&self) -> &noto_core::BaseSpec {
        &self.hand.base
    }

    pub fn carried_kg(&self) -> f32 {
        self.agent.carried.iter().map(|(k, n)| k.spec().mass_kg * *n).sum()
    }

    pub fn carried_count(&self, item: ItemKind) -> u32 {
        (self.carried_amount(item) + AMOUNT_EPS).floor() as u32
    }

    pub fn carried_amount(&self, item: ItemKind) -> f32 {
        self.agent.carried.iter().filter(|(k, _)| *k == item).map(|e| e.1).sum()
    }

    pub fn grasp_for(&self, item: ItemKind, amount: f32) -> GraspInfo {
        if (amount - 1.0).abs() < AMOUNT_EPS {
            return self.grasp_info(item).clone();
        }
        top_grasp_info(&self.hand, item, amount)
    }

    pub fn grip_capacity_for(&self, item: ItemKind, amount: f32) -> f32 {
        let g = self.grasp_for(item, amount);
        g.margin * item.spec().mass_kg * amount * noto_core::balance::G * GRIP_SAFETY
    }

    pub fn split_time_s(&self, item: ItemKind, amount: f32) -> f32 {
        item.split_time_s() * amount.max(MIN_PIECE).powf(2.0 / 3.0) / self.mining_rate().max(0.05)
    }

    pub fn waypoint_in_reach(&self) -> Option<u32> {
        self.waypoints
            .iter()
            .map(|w| (w.id, ((w.x - self.agent.x).powi(2) + (w.z - self.agent.z).powi(2)).sqrt()))
            .filter(|(_, d)| *d <= DELIVER_REACH_M)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(id, _)| id)
    }

    pub fn capacity_kg(&self) -> f32 {
        self.limits().capacity_kg
    }

    pub fn battery_capacity_j(&self) -> f64 {
        self.body.battery_j()
    }

    pub fn limits(&self) -> DriveLimits {
        let key = self.carried_kg().to_bits();
        let mut cache = self.limits.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((k, l)) = cache.as_ref() {
            if *k == key {
                return l.clone();
            }
        }
        let l = self.body.drive_limits(&self.hand.base, self.carried_kg());
        *cache = Some((key, l.clone()));
        l
    }

    pub fn drive_spec(&self) -> noto_core::BaseSpec {
        self.drive_spec_for(self.carried_kg())
    }

    pub fn drive_spec_for(&self, cargo_kg: f32) -> noto_core::BaseSpec {
        let l = if (cargo_kg - self.carried_kg()).abs() < 1e-6 { self.limits() } else { self.body.drive_limits(&self.hand.base, cargo_kg) };
        let mass = self.body.mass_kg() + cargo_kg;
        noto_core::BaseSpec {
            max_speed_mps: l.max_speed_mps,
            max_slope_deg: l.max_slope_deg,
            max_step_m: l.max_step_m,
            power_w: if self.body.dims.payload_kg.is_some() { self.body.chassis.power_w } else { self.hand.base.power_w } * l.power_scale,
            drive_accel_mps2: (!self.body.legged()).then(|| self.body.chassis.drive_force_n / mass),
            mass_kg: Some(mass),
            walk_com_z_m: self.body.legged().then(|| noto_core::act::gait::walking_com_z(&self.body)),
            walk_foot_m: self.body.legged().then_some(self.body.leg.foot_len_m),
            walk_dims: self.body.legged().then_some(self.body.dims),
            ..self.hand.base
        }
    }

    pub fn ground_at(&self, cell: Cell) -> Ground {
        if cell.0 < self.terrain.w && cell.1 < self.terrain.d {
            self.terrain.ground_at(cell.0, cell.1)
        } else {
            Ground::Gravel
        }
    }

    pub fn ground_here(&self) -> Ground {
        self.ground_at(self.agent_cell())
    }

    pub fn accel_time_s(&self) -> f32 {
        let l = self.limits();
        (l.max_speed_mps / l.accel_limit_mps2.max(0.05)).max(ACCEL_TIME_S)
    }

    pub fn planned_accel(&self, slope_deg: f32, ground: Ground, dir: f32) -> f32 {
        // A load in the hands, 35 cm before the body, leaves the feet less room to stop on than one in a bay.
        let stand = self.with_load_held(self.body.quiet(self.carried_kg()));
        self.body.planned_accel_from(&self.limits(), &stand, slope_deg, ground.surface(), dir, ACCEL_TIME_S)
    }

    pub fn braking_distance_m(&self, slope_deg: f32, ground: Ground) -> f32 {
        let v = self.drive_spec().max_speed_mps;
        v * v / (2.0 * self.planned_accel(slope_deg, ground, -1.0))
    }

    fn remaining_m(&self, m: &MoveState) -> f32 {
        (m.follow.total_m() - m.follow.s).max(0.0)
    }

    fn ff_for(&self, s: Source) -> FeedForward {
        FeedForward { enabled: self.ff_enabled, gain: self.ff_gains.get(&s).copied().unwrap_or(1.0) }
    }

    pub fn grip_capacity_n(&self, item: ItemKind) -> f32 {
        let g = self.grasp_info(item);
        g.margin * item.spec().mass_kg * noto_core::balance::G * GRIP_SAFETY
    }

    pub fn take_balance_events(&mut self) -> Vec<BalanceEvent> {
        std::mem::take(&mut self.balance_events)
    }

    pub fn battery_ratio(&self) -> f32 {
        (1.0 - self.agent.energy_used_j / self.battery_capacity_j()).clamp(0.0, 1.0) as f32
    }

    pub fn battery_depleted(&self) -> bool {
        self.agent.energy_used_j >= self.battery_capacity_j()
    }

    pub fn mining_rate(&self) -> f32 {
        let hot = ["shoulder_r", "elbow_r"].iter().map(|j| self.heat.get(*j).copied().unwrap_or(0.0)).fold(0.0f32, f32::max);
        u8::from(self.hand.config.fingers) as f32 / 5.0 / hot.max(1.0)
    }

    pub fn busy(&self) -> bool {
        self.active.is_some()
    }

    pub fn current_action(&self) -> Option<String> {
        self.active.as_ref().map(|(a, _)| a.label())
    }

    fn agent_cell(&self) -> Cell {
        self.terrain.cell_of(self.agent.x, self.agent.z).unwrap_or((0, 0))
    }

    pub fn blocked_cells(&self) -> HashSet<Cell> {
        let mut b: HashSet<Cell> = self.nodes.iter().filter_map(|n| self.terrain.cell_of(n.x, n.z)).collect();
        b.extend(crate::fixture::blocked(&self.fixtures));
        b
    }

    pub fn drop_base_m(&self, d: &DropItem) -> f32 {
        if d.support.is_ground() {
            self.surface_m(d.x, d.z)
        } else {
            d.y_m
        }
    }

    pub fn support_top_m(&self, s: Support) -> Option<f32> {
        self.site.as_ref()?.support_top_m(&self.fixtures, s)
    }

    pub fn search_prior(&self) -> Option<Vec<f32>> {
        self.site.as_ref().map(|s| s.prior(&self.terrain))
    }

    pub fn agent_ground_m(&self) -> f32 {
        match self.terrain.cell_of(self.agent.x, self.agent.z) {
            Some((cx, cz)) => self.terrain.height_m(cx, cz) + self.terrain.curb_m(cx, cz),
            None => 0.0,
        }
    }

    pub fn surface_m(&self, x: f32, z: f32) -> f32 {
        match self.terrain.cell_of(x, z) {
            Some((cx, cz)) => self.terrain.height_m(cx, cz) + self.terrain.curb_m(cx, cz),
            None => 0.0,
        }
    }

    /// A route that keeps the body clear of trees, rocks, walls and water: the half width and `clear::MIN_GAP_M` where
    /// it can, `clear::TIGHT_GAP_M` where nothing roomier gets there, and no route where the body does not fit. It goes
    /// round what lies on the ground if it can.
    pub fn plan_route(&self, goal: Goal) -> Option<nav::Plan> {
        let mut robots = HashSet::new();
        if !self.crew.is_empty() {
            let keep = self.terrain.cell_of(self.agent.x, self.agent.z);
            robots = self.still_robot_cells(&[1, traffic::RIDER_ID], keep, self.footprint().half_w_m + 0.15);
        }
        let mut both = robots.clone();
        both.extend(self.loose_cells(goal));
        self.plan_route_avoiding(goal, &both).or_else(|| self.plan_route_avoiding(goal, &robots)).or_else(|| self.plan_route_avoiding(goal, &HashSet::new()))
    }

    /// The cells things lie in on the ground, for a route to go round; not those by the goal, or under the robot.
    pub(crate) fn loose_cells(&self, goal: Goal) -> HashSet<Cell> {
        let here = self.agent_cell();
        let at = match goal {
            Goal::Cell(c) => cell_center(c.0, c.1),
            Goal::Within { x, z, .. } => (x, z),
        };
        let reach = match goal {
            Goal::Cell(_) => 0.45,
            Goal::Within { reach_m, .. } => reach_m + 0.3,
        };
        let mut out = HashSet::new();
        for d in self.drops.iter().filter(|d| d.support.is_ground() && (d.x - at.0).hypot(d.z - at.1) > reach) {
            let half = 0.5 * d.kind.spec().piece_size_m(d.amount)[0];
            for dx in [-half, 0.0, half] {
                out.extend(self.terrain.cell_of(d.x + dx, d.z).filter(|c| *c != here));
            }
        }
        out
    }

    /// `plan_route` with `extra` cells blocked as well (the robots standing in the way).
    pub fn plan_route_avoiding(&self, goal: Goal, extra: &HashSet<Cell>) -> Option<nav::Plan> {
        let mut blocked = self.blocked_cells();
        blocked.extend(extra.iter().copied());
        let map = self.clear_map();
        let (spec, fp) = (self.drive_spec(), self.footprint());
        let plan = [clear::MIN_GAP_M, clear::TIGHT_GAP_M].into_iter().find_map(|gap| {
            let room = nav::Room { map: &map, need_m: fp.half_w_m + gap, soft_m: fp.half_w_m + clear::COMFORT_GAP_M };
            nav::plan_room(&spec, &self.terrain, &|x, z| blocked.contains(&(x, z)), Some(&room), self.agent_cell(), goal, None)
        })?;
        Some(self.to_cell_centre(plan, goal))
    }

    /// The planner counts cells by their centres, so a body in a corner of a cell whose centre is within reach has an
    /// empty route and is still out of reach. It goes to the centre.
    fn to_cell_centre(&self, mut plan: nav::Plan, goal: Goal) -> nav::Plan {
        let Goal::Within { x, z, reach_m } = goal else { return plan };
        let (ax, az) = (self.agent.x, self.agent.z);
        let c = self.agent_cell();
        let (cx, cz) = cell_center(c.0, c.1);
        let d = (cx - ax).hypot(cz - az);
        if plan.edges.is_empty() && (ax - x).hypot(az - z) > reach_m && d > motion::GOAL_TOL_M {
            let move_time_s = d / self.drive_spec().max_speed_mps.max(0.1);
            plan.edges.push(nav::EdgePlan { from: c, to: c, horizontal_m: d, move_time_s, pause_s: 0.0, step_m: 0.0, slope_deg: 0.0, margin: 1.0, ground_limit_deg: 90.0, energy_factor: 1.0 });
            plan.time_s += move_time_s;
            plan.distance_m += d;
        }
        plan
    }

    /// The body's outline on the ground (`clear::Footprint`), with what rides on its tines.
    pub fn footprint(&self) -> Footprint {
        let load: f32 = self.agent.carried.iter().filter(|(_, n)| *n > AMOUNT_EPS).map(|(k, n)| k.spec().piece_size_m(n.min(1.0))[1] * n.ceil()).sum();
        Footprint::of(&self.body).laden(load)
    }

    /// How far everything this body can hit is from each point (`clear::ClearMap`), built again when a node goes. A
    /// ledge is one for the base that cannot take it, so each kind of body has its own map.
    pub fn clear_map(&self) -> std::sync::Arc<ClearMap> {
        let spec = self.drive_spec();
        let top = Footprint::of(&self.body).top_m.max(self.standing_eye_m() + 0.2).max(self.body.dims.roof_m.unwrap_or(0.0));
        let nodes = self.nodes.iter().fold(self.nodes.len() as u64 ^ (self.fixtures.len() as u64) << 32, |h, n| (h ^ n.id as u64).wrapping_mul(0x0000_0100_0000_01B3));
        let key = [spec.max_step_m.to_bits(), spec.max_slope_deg.to_bits(), top.to_bits(), spec.kind as u32].into_iter().fold(nodes, |h, b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01B3));
        let mut slot = self.room.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, m)) = slot.iter().find(|(k, _)| *k == key) {
            return m.clone();
        }
        let map = std::sync::Arc::new(ClearMap::build_for(&self.terrain, &self.nodes, &self.fixtures, top, Some(&spec)));
        if slot.len() >= 8 {
            slot.remove(0);
        }
        slot.push((key, map.clone()));
        map
    }

    pub fn command(&mut self, action: Action) -> Result<Vec<TimedEvent>, ActionError> {
        if self.active.is_some() {
            return Err(ActionError::Busy);
        }
        if self.battery_depleted() {
            return Err(ActionError::BatteryDepleted);
        }
        if self.fallen_until.is_some_and(|t| self.time_s < t) {
            return Err(ActionError::Fallen);
        }
        let before = (self.agent.x, self.agent.z, self.agent.yaw);
        let state = match &action {
            Action::MoveTo { x, z, reach_m } => {
                let goal = if *reach_m <= 0.0 {
                    match self.terrain.cell_of(*x, *z) {
                        Some(c) => Goal::Cell(c),
                        None => return Err(ActionError::UnknownTarget),
                    }
                } else {
                    Goal::Within { x: *x, z: *z, reach_m: *reach_m }
                };
                let plan = self.plan_route(goal).ok_or(ActionError::NoPath)?;
                let plan = self.fit_plan(plan, goal, &self.loose_cells(goal));
                self.odom.new_move();
                let mut follow = self.start_follow(&plan.edges);
                follow.stop_within = (*reach_m > 0.0).then_some(((*x, *z), *reach_m));
                Active::Move(MoveState::new(plan.edges, follow, goal))
            }
            Action::Turn { yaw_deg } => {
                let target = wrap_angle(yaw_deg.to_radians());
                Active::Turn { target, run: self.turn_run(target) }
            }
            Action::Face { x, z } => {
                let target = (z - self.agent.z).atan2(x - self.agent.x);
                Active::Turn { target, run: self.turn_run(target) }
            }
            Action::Look { x, z } => {
                let t = self.start_look(*x, *z);
                Active::Work(WorkState { kind: WorkKind::Look { at: (*x, *z) }, remaining_s: t })
            }
            Action::Mine { node } => {
                let (nx, nz, hp) = self.node(*node).map(|n| (n.x, n.z, n.hp)).ok_or(ActionError::UnknownTarget)?;
                if noto_core::act::fork::applies(&self.body) {
                    // A lift truck's rider steps down and fells it on foot, then climbs on again.
                    self.check_reach(nx, nz, traffic::rider_reach_m(&self.body))?;
                    self.rider = Some(traffic::Rider::start(self.agent.x, self.agent.z, self.agent.yaw, (nx, nz)));
                } else {
                    self.check_reach(nx, nz, MINE_REACH_M)?;
                }
                Active::Work(WorkState { kind: WorkKind::Mine { node: *node }, remaining_s: hp })
            }
            Action::PickUp { drop, .. } if noto_core::act::fork::applies(&self.body) => {
                let (dx, dz, kind, amount) = {
                    let d = self.drop_item(*drop).ok_or(ActionError::UnknownTarget)?;
                    (d.x, d.z, d.kind, d.amount)
                };
                let size = kind.spec().piece_size_m(amount);
                self.check_reach(dx, dz, PICKUP_REACH_M.max(noto_core::act::fork::standoff_x(&self.body, size[0].max(size[2])) + 0.2))?;
                self.check_support(*drop)?;
                let item_kg = kind.spec().mass_kg * amount;
                if self.carried_kg() + item_kg > self.capacity_kg() + 1e-4 {
                    return Err(ActionError::CapacityExceeded { carried_kg: self.carried_kg(), item_kg, capacity_kg: self.capacity_kg() });
                }
                let t = self.start_fork_pick(*drop)?;
                Active::Work(WorkState { kind: WorkKind::Pick { drop: *drop, two_arms: false, face: Some((dx, dz)) }, remaining_s: t })
            }
            Action::PickUp { drop, aim, side, memory_sigma_m } => {
                let (dx, dz, kind, amount) = {
                    let d = self.drop_item(*drop).ok_or(ActionError::UnknownTarget)?;
                    (d.x, d.z, d.kind, d.amount)
                };
                self.check_reach(dx, dz, PICKUP_REACH_M)?;
                self.check_support(*drop)?;
                let info = self.grasp_for(kind, amount);
                self.agent.grip_margin = Some(info.margin);
                if !info.stable && 2.0 * info.margin < 1.0 {
                    return Err(ActionError::GraspUnstable { margin: info.margin });
                }
                let item_kg = kind.spec().mass_kg * amount;
                if self.carried_kg() + item_kg > self.capacity_kg() + 1e-4 {
                    return Err(ActionError::CapacityExceeded {
                        carried_kg: self.carried_kg(),
                        item_kg,
                        capacity_kg: self.capacity_kg(),
                    });
                }
                let (rep, mut two_arms) = self.body.lift(item_kg, self.carried_kg(), &self.ff_for(Source::ArmLift), self.grip_capacity_for(kind, amount));
                if !rep.feasible {
                    return Err(ActionError::LiftOverload { limiting: rep.limiting.clone().unwrap_or_default(), margin: rep.margin });
                }
                if !info.stable {
                    two_arms = true;
                }
                let (t, face) = self.start_grasp(*drop, two_arms, *aim, *side, *memory_sigma_m)?;
                Active::Work(WorkState { kind: WorkKind::Pick { drop: *drop, two_arms, face }, remaining_s: t })
            }
            Action::Drop { item, amount } => {
                let have: f32 = self.agent.carried.iter().filter(|(k, n)| *n > AMOUNT_EPS && item.map_or(true, |i| i == *k)).map(|(_, n)| *n).sum();
                if have <= AMOUNT_EPS || amount.is_some_and(|a| a <= AMOUNT_EPS) {
                    return Err(ActionError::NothingToDrop);
                }
                let waypoint = self.waypoint_in_reach();
                let pieces = self.pieces_to_set_down(*item, *amount);
                let t = self.start_release(&pieces, waypoint);
                Active::Work(WorkState { kind: WorkKind::Drop { item: *item, amount: *amount, waypoint }, remaining_s: t })
            }
            Action::Split { drop, portion } => {
                let (dx, dz, kind, amount) = {
                    let d = self.drop_item(*drop).ok_or(ActionError::UnknownTarget)?;
                    (d.x, d.z, d.kind, d.amount)
                };
                self.check_reach(dx, dz, PICKUP_REACH_M)?;
                if !kind.spec().divisible {
                    return Err(ActionError::NotDivisible);
                }
                let piece = amount * portion;
                if !(0.0..1.0).contains(portion) || piece < MIN_PIECE - AMOUNT_EPS || amount - piece < MIN_PIECE - AMOUNT_EPS {
                    return Err(ActionError::PieceTooSmall { amount: piece.min(amount - piece) });
                }
                Active::Work(WorkState { kind: WorkKind::Split { drop: *drop, portion: *portion }, remaining_s: self.split_time_s(kind, amount) })
            }
            Action::Deliver { waypoint, item } => {
                let w = self.waypoint(*waypoint).ok_or(ActionError::UnknownTarget)?;
                self.check_reach(w.x, w.z, DELIVER_REACH_M)?;
                let any = self.agent.carried.iter().any(|(k, n)| *n > AMOUNT_EPS && item.map_or(true, |i| i == *k));
                if !any {
                    return Err(ActionError::NothingToDeliver);
                }
                let pieces = self.pieces_to_set_down(*item, None);
                let t = self.start_release(&pieces, Some(*waypoint));
                Active::Work(WorkState { kind: WorkKind::Deliver { waypoint: *waypoint, item: *item }, remaining_s: t })
            }
            Action::Push { drop, x, z } => Active::Push(self.start_push(*drop, *x, *z)?),
        };
        let after = (self.agent.x, self.agent.z, self.agent.yaw);
        if after != before {
            self.odom.update(before, after, 0.0, 0.0);
        }
        if !matches!(action, Action::PickUp { .. } | Action::Look { .. }) {
            self.gaze = None;
        }
        let label = action.label();
        self.active = Some((action, state));
        Ok(vec![self.stamp(WorldEvent::ActionStarted { action: label })])
    }

    pub fn cancel(&mut self) -> Vec<TimedEvent> {
        self.end_act();
        self.rider = None;
        match self.active.take() {
            Some((a, _)) => {
                self.agent.speed_mps = 0.0;
                self.agent.terrain_margin = None;
                vec![self.stamp(WorldEvent::ActionDone { action: a.label(), ok: false, reason: Some("cancelled".into()) })]
            }
            None => vec![],
        }
    }

    fn check_support(&self, drop: u32) -> Result<(), ActionError> {
        let Some(d) = self.drop_item(drop) else { return Err(ActionError::UnknownTarget) };
        let here = self.agent_ground_m();
        let at = self.drop_base_m(d);
        match d.support {
            Support::Ground => Ok(()),
            Support::Step { .. } if (at - here).abs() <= 0.03 => Ok(()),
            Support::Step { .. } => Err(ActionError::CannotReach { lowest_m: here, need_m: at, reason: format!("{}에 있어 같은 높이에 올라서야 집을 수 있음", d.support.label_ko()) }),
            Support::Shelf { .. } | Support::Table { .. } => Err(ActionError::CannotReach { lowest_m: here, need_m: at, reason: format!("{}의 물체를 집는 손 뻗기는 아직 모델에 없음", d.support.label_ko()) }),
        }
    }

    fn check_reach(&self, x: f32, z: f32, reach: f32) -> Result<(), ActionError> {
        let d = ((x - self.agent.x).powi(2) + (z - self.agent.z).powi(2)).sqrt();
        if d > reach {
            Err(ActionError::OutOfReach { distance_m: d, reach_m: reach })
        } else {
            Ok(())
        }
    }

    pub(crate) fn stamp(&self, event: WorldEvent) -> TimedEvent {
        TimedEvent { tick: self.tick, time_s: self.time_s, event }
    }

    pub fn tick(&mut self) -> Vec<TimedEvent> {
        let ev = self.step(TICK_S);
        self.tick += 1;
        ev
    }

    pub fn step(&mut self, dt: f32) -> Vec<TimedEvent> {
        self.update_balance(dt);
        let before = (self.agent.x, self.agent.z, self.agent.yaw);
        let mut events = Vec::new();
        let mut remaining = dt;
        let mut guard = 0;
        while remaining > 1e-6 && self.active.is_some() && guard < 64 {
            guard += 1;
            if self.battery_depleted() {
                events.push(self.stamp(WorldEvent::BatteryDepleted));
                self.finish(false, Some("battery_depleted".into()), &mut events);
                break;
            }
            let used = self.advance(remaining, &mut events);
            remaining -= used.max(1e-6);
        }
        if self.active.is_none() {
            self.agent.speed_mps = 0.0;
        }
        // A piece coming off stops the robot where it is: whoever drives it decides what next.
        let driven = matches!(self.active, Some((_, Active::Move(_) | Active::Turn { .. })));
        if self.ride_cargo(dt, driven, 1, &mut events) && driven {
            self.finish(false, Some("cargo_lost".into()), &mut events);
        }
        // What the load did in the hands this tick reaches the whole body in it: a slide of a centimetre or more is
        // solved again with the load where it went, not at the next tick's posture.
        let s = (self.ride.state.shift, self.ride.solved_shift);
        if self.carries_in_hands() && (s.0[0] - s.1[0]).hypot(s.0[1] - s.1[1]) >= cargo::RESOLVE_SHIFT_M {
            self.solve_balance(0.0, true);
        }
        let after = (self.agent.x, self.agent.z, self.agent.yaw);
        self.odom.update(before, after, dt, self.odom_over);
        self.odom_over = 0.0;
        self.eye.advance(before, after, dt);
        if !self.crew.is_empty() {
            self.step_crew(dt, &mut events);
        }
        self.time_s += dt as f64;
        events
    }

    fn finish(&mut self, ok: bool, reason: Option<String>, events: &mut Vec<TimedEvent>) {
        if let Some((a, _)) = self.active.take() {
            self.agent.speed_mps = 0.0;
            self.agent.terrain_margin = None;
            events.push(self.stamp(WorldEvent::ActionDone { action: a.label(), ok, reason }));
        }
    }

    fn rotate_toward(&mut self, target: f32, max_dt: f32) -> f32 {
        let caps = self.turn_caps();
        let rate = if caps.can_pivot() { caps.pivot_rad_s.min(TURN_RATE_RAD_S) } else { motion::CREEP_RAD_S };
        let err = wrap_angle(target - self.agent.yaw);
        let turn = (rate * max_dt).min(err.abs());
        self.agent.yaw = wrap_angle(self.agent.yaw + turn * err.signum());
        turn / rate
    }

    fn advance(&mut self, remaining: f32, events: &mut Vec<TimedEvent>) -> f32 {
        let (action, mut state) = self.active.take().expect("checked by the caller");
        let used;
        let mut done: Option<(bool, Option<String>)> = None;
        match &mut state {
            Active::Turn { target, run } => {
                if run.done() && run.wait_left <= 0.0 {
                    if wrap_angle(*target - self.agent.yaw).abs() <= FACE_DONE_RAD {
                        self.agent.yaw = *target;
                    }
                    self.agent.speed_mps = 0.0;
                    done = Some(if run.blocked { (false, Some("blocked".into())) } else { (true, None) });
                    used = 0.0;
                } else {
                    used = self.advance_run(run, remaining).max(1e-4);
                }
            }
            Active::Move(m) => {
                used = self.advance_move(m, remaining, events, &mut done);
            }
            Active::Work(w) => {
                used = self.advance_work(w, remaining, events, &mut done);
            }
            Active::Push(ps) => {
                used = self.advance_push(ps, remaining, events, &mut done);
            }
        }
        self.active = Some((action, state));
        if let Some((ok, reason)) = done {
            self.finish(ok, reason, events);
        }
        used
    }

    fn add_energy(&mut self, j: f64) {
        self.agent.energy_used_j += j;
    }

    fn advance_move(&mut self, m: &mut MoveState, remaining: f32, events: &mut Vec<TimedEvent>, done: &mut Option<(bool, Option<String>)>) -> f32 {
        if m.idx >= m.edges.len() {
            *done = Some((true, None));
            return 0.0;
        }
        if !self.crew.is_empty() && self.hold_for_robot(m, remaining, events) {
            return remaining;
        }
        self.advance_follow(m, remaining, events, done)
    }

    /// The way-ahead check of a move: the head camera's view about the way it is going. A robot in it within the
    /// stopping distance stops the move; one that stays routes it round (the others give way to the first worker, so
    /// most clear by themselves).
    fn hold_for_robot(&mut self, m: &mut MoveState, dt: f32, events: &mut Vec<TimedEvent>) -> bool {
        let fp = self.footprint();
        let reversing = self.motion_view().is_some_and(|m| m.reversing);
        let me = traffic::Looker { id: 1, at: (self.agent.x, self.agent.z), dir: self.travel_dir(), half_w: fp.half_w_m, front: if reversing { fp.rear_m } else { fp.front_m }, speed: self.agent.speed_mps, eye_m: self.standing_eye_m(), gaze: self.glance_rad(1) };
        let seen = self.robots_seen(&me);
        traffic::remember(&mut self.noticed, seen, self.time_s);
        let path: Vec<(f32, f32)> = m.edges[m.idx.min(m.edges.len())..].iter().map(|e| crate::terrain::cell_center(e.to.0, e.to.1)).collect();
        let Some((other, gap)) = self.robot_in_way(&me, &path, &self.noticed) else {
            m.held_s = 0.0;
            m.held_by = None;
            return false;
        };
        self.agent.speed_mps = 0.0;
        if m.held_by != Some(other) {
            m.held_by = Some(other);
            m.held_s = 0.0;
            events.push(self.stamp(WorldEvent::Hold { worker: 1, other, gap_m: gap, label: format!("시야에 작업자 {other}이(가) {gap:.1} m 앞 — 멈춰 기다림") }));
        }
        // An idle sub-worker in the way steps aside for the first worker.
        let dir = self.travel_dir();
        if self.crew_make_way(other, dir) {
            events.push(self.stamp(WorldEvent::Hold { worker: other, other: 1, gap_m: gap, label: "작업자 1의 길에서 비켜 섬".into() }));
        }
        m.held_s += dt;
        if m.held_s >= traffic::YIELD_VEHICLE_S && m.reroutes < 3 {
            if let Some(goal) = m.goal {
                let keep = self.terrain.cell_of(self.agent.x, self.agent.z);
                let mut extra = self.still_robot_cells(&[1, traffic::RIDER_ID], keep, fp.half_w_m + 0.15);
                // Round the one in the way by both bodies' widths, whether it moves or not.
                let others: Vec<u32> = self.robot_outlines().iter().map(|o| o.id).filter(|o| *o != other).collect();
                extra.extend(self.robot_cells(&others, keep, fp.half_w_m + 0.15, false));
                if let Some(plan) = self.plan_route_avoiding(goal, &extra) {
                    let stop = m.follow.stop_within;
                    let mut follow = self.start_follow(&plan.edges);
                    follow.stop_within = stop;
                    *m = MoveState { edges: plan.edges, idx: 0, pause_left: 0.0, progress_s: 0.0, announced_pause: false, elapsed_s: m.elapsed_s, follow, anchor: m.anchor, held_s: 0.0, held_by: None, reroutes: m.reroutes + 1, goal: m.goal, replans: m.replans };
                    events.push(self.stamp(WorldEvent::Hold { worker: 1, other, gap_m: gap, label: format!("작업자 {other}이(가) 비키지 않아 돌아가는 길로 다시 계획") }));
                    return true;
                }
            }
            m.held_s = 0.0;
        }
        true
    }

    fn advance_work(&mut self, w: &mut WorkState, remaining: f32, events: &mut Vec<TimedEvent>, done: &mut Option<(bool, Option<String>)>) -> f32 {
        if let (WorkKind::Mine { node }, true) = (&w.kind, self.rider.is_some()) {
            return self.advance_rider(*node, remaining, events, done);
        }
        let (tx, tz) = match w.kind {
            WorkKind::Mine { node } => match self.node(node) {
                Some(n) => (n.x, n.z),
                None => {
                    *done = Some((false, Some("unknown_target".into())));
                    return 0.0;
                }
            },
            WorkKind::Look { at } => at,
            WorkKind::Pick { drop, face, .. } => match (self.drop_item(drop), face) {
                (Some(_), Some(p)) => p,
                (Some(_), None) => (self.agent.x + self.agent.yaw.cos(), self.agent.z + self.agent.yaw.sin()),
                (None, _) => {
                    *done = Some((false, Some("unknown_target".into())));
                    return 0.0;
                }
            },
            WorkKind::Deliver { waypoint, .. } => match self.waypoint(waypoint) {
                Some(p) => (p.x, p.z),
                None => {
                    *done = Some((false, Some("unknown_target".into())));
                    return 0.0;
                }
            },
            WorkKind::Split { drop, .. } => match self.drop_item(drop) {
                Some(d) => (d.x, d.z),
                None => {
                    *done = Some((false, Some("unknown_target".into())));
                    return 0.0;
                }
            },
            WorkKind::Drop { waypoint, .. } => match waypoint.and_then(|w| self.waypoint(w)) {
                Some(p) => (p.x, p.z),
                None => (self.agent.x + self.agent.yaw.cos(), self.agent.z + self.agent.yaw.sin()),
            },
        };
        let heading = (tz - self.agent.z).atan2(tx - self.agent.x);
        let err = wrap_angle(heading - self.agent.yaw).abs();
        // A fork act turns and drives the base itself: once it runs (or when it is a pick, which aligns first), the
        // base no longer turns to face the target on its own.
        let fork_steers = self.act.as_ref().is_some_and(|p| p.drive.is_some() && (p.elapsed_s > 0.0 || matches!(w.kind, WorkKind::Pick { .. })));
        if err > FACE_DONE_RAD && !fork_steers {
            let t = self.rotate_toward(heading, remaining);
            self.add_energy(MANIPULATION_POWER_W * t as f64);
            return t;
        }

        let t = w.remaining_s.min(remaining);
        self.add_energy(MANIPULATION_POWER_W * t as f64);
        self.act_advance(t, events);
        match w.kind.clone() {
            WorkKind::Mine { node } => {
                let rate = self.mining_rate();
                let work = (t * rate).max(1e-6);
                self.agent.strike_acc += t;
                while self.agent.strike_acc >= 1.0 {
                    self.agent.strike_acc -= 1.0;
                    self.agent.strikes += 1;
                }
                let n = self.nodes.iter_mut().find(|n| n.id == node).expect("looked up above");
                n.hp -= work.min(n.hp);
                w.remaining_s = n.hp / rate;
                if n.hp <= 1e-5 {
                    self.break_node(node, events);
                    *done = Some((true, None));
                }
            }
            WorkKind::Look { at } => {
                w.remaining_s -= t;
                if w.remaining_s <= 1e-6 {
                    self.act_finish(events);
                    let play = self.end_act();
                    self.hold_gaze(play.as_ref(), at);
                    *done = Some((true, None));
                }
            }
            WorkKind::Pick { drop, .. } => {
                w.remaining_s -= t;
                if w.remaining_s <= 1e-6 {
                    self.act_finish(events);
                    let play = self.end_act();
                    match play.as_ref().map(|p| (p.rollout.outcome.ok, p.rollout.outcome.code)) {
                        Some((false, code)) => {
                            if let Some(p) = &play {
                                self.grasp_failed(drop, p);
                            }
                            *done = Some((false, Some(code.unwrap_or("grasp_failed").to_string())));
                        }
                        _ => {
                            self.complete_pick(drop, events);
                            *done = Some((true, None));
                        }
                    }
                }
            }
            WorkKind::Deliver { waypoint, item } => {
                w.remaining_s -= t;
                if w.remaining_s <= 1e-6 {
                    self.act_finish(events);
                    self.end_act();
                    self.complete_deliver(waypoint, item, events);
                    *done = Some((true, None));
                }
            }
            WorkKind::Drop { item, amount, waypoint } => {
                w.remaining_s -= t;
                if w.remaining_s <= 1e-6 {
                    self.act_finish(events);
                    let play = self.end_act();
                    self.complete_drop(item, amount, waypoint, play.as_ref(), events);
                    *done = Some((true, None));
                }
            }
            WorkKind::Split { drop, portion } => {
                self.agent.strike_acc += t;
                while self.agent.strike_acc >= 1.0 {
                    self.agent.strike_acc -= 1.0;
                    self.agent.strikes += 1;
                }
                w.remaining_s -= t;
                if w.remaining_s <= 1e-6 {
                    self.complete_split(drop, portion, events);
                    *done = Some((true, None));
                }
            }
        }
        t
    }

    fn break_node(&mut self, id: u32, events: &mut Vec<TimedEvent>) {
        let (kind, drops) = self.fell(id);
        events.push(self.stamp(WorldEvent::NodeBroken { node: node_id(id), kind, drops: drops.into_iter().map(drop_id).collect() }));
    }

    /// Takes a node off the map and drops what it gives around where it stood (the first worker's swing or a
    /// sub-worker's).
    pub(crate) fn fell(&mut self, id: u32) -> (NodeKind, Vec<u32>) {
        let pos = self.nodes.iter().position(|n| n.id == id).expect("node exists");
        let node = self.nodes.remove(pos);
        let spec = node.kind.spec();
        let mut ids = Vec::new();
        for i in 0..spec.drop_count {
            let a = (i as f32 / spec.drop_count as f32) * std::f32::consts::TAU + self.rng.range_f32(0.0, 1.0);
            // Around the stump, one piece's diagonal apart whatever the angle: no two lie inside one another.
            let size = spec.drop.spec().size_m;
            let r = if spec.drop_count == 1 { 0.0 } else { (0.5 * (size[0].hypot(size[2]) + 0.02) / (std::f32::consts::PI / spec.drop_count as f32).sin()).max(0.13) };
            let (x, z) = self.free_spot((node.x + r * a.cos(), node.z + r * a.sin()));
            let d = DropItem::whole(self.next_drop, spec.drop, x, z);
            self.next_drop += 1;
            ids.push(d.id);
            self.drops.push(d);
        }
        (node.kind, ids)
    }

    pub(crate) fn next_drop_id(&mut self) -> u32 {
        let id = self.next_drop;
        self.next_drop += 1;
        id
    }

    fn complete_pick(&mut self, id: u32, events: &mut Vec<TimedEvent>) {
        if let Some(pos) = self.drops.iter().position(|d| d.id == id) {
            let d = self.drops.remove(pos);
            add_stock(&mut self.agent.carried, d.kind, d.amount);
            let margin = self.agent.grip_margin.unwrap_or(0.0);
            events.push(self.stamp(WorldEvent::ItemPicked { drop: drop_id(id), item: d.kind, mass_kg: d.mass_kg(), grip_margin: margin, amount: d.amount }));
        }
    }

    fn complete_deliver(&mut self, waypoint: u32, item: Option<ItemKind>, events: &mut Vec<TimedEvent>) {
        let mut moved: Vec<(ItemKind, f32)> = Vec::new();
        self.agent.carried.retain_mut(|(k, n)| {
            if *n > AMOUNT_EPS && item.map_or(true, |i| i == *k) {
                moved.push((*k, *n));
                false
            } else {
                *n > AMOUNT_EPS
            }
        });
        let bay = self.is_bay(waypoint);
        for (k, n) in moved {
            if bay {
                // A loading bay: the load goes down in its row of slots, for the tines to take.
                self.stage_at_bay(waypoint, k, n);
            } else if let Some(w) = self.waypoints.iter_mut().find(|w| w.id == waypoint) {
                add_stock(&mut w.stock, k, n);
            }
            events.push(self.stamp(WorldEvent::Delivered { waypoint: waypoint_id(waypoint), item: k, amount: n }));
        }
    }

    fn take_cargo(&mut self, item: Option<ItemKind>, amount: Option<f32>) -> Vec<(ItemKind, f32)> {
        let mut left = amount.unwrap_or(f32::INFINITY);
        let mut taken: Vec<(ItemKind, f32)> = Vec::new();
        for (k, n) in self.agent.carried.iter_mut() {
            if left <= AMOUNT_EPS || *n <= AMOUNT_EPS || !item.map_or(true, |i| i == *k) {
                continue;
            }
            let a = n.min(left);
            *n -= a;
            left -= a;
            taken.push((*k, a));
        }
        self.agent.carried.retain(|(_, n)| *n > AMOUNT_EPS);
        taken
    }

    fn complete_drop(&mut self, item: Option<ItemKind>, amount: Option<f32>, waypoint: Option<u32>, play: Option<&ActPlay>, events: &mut Vec<TimedEvent>) {
        let taken = self.take_cargo(item, amount);
        let mut piece_index = 0usize;
        for (k, a) in taken {
            if let Some(wid) = waypoint {
                if let Some(w) = self.waypoints.iter_mut().find(|w| w.id == wid) {
                    add_stock(&mut w.stock, k, a);
                }
                events.push(self.stamp(WorldEvent::ItemDropped { item: k, amount: a, waypoint: Some(waypoint_id(wid)), drops: vec![] }));
                continue;
            }
            let ahead = (self.agent.x + DROP_AHEAD_M * self.agent.yaw.cos(), self.agent.z + DROP_AHEAD_M * self.agent.yaw.sin());
            let on_land = |w: &World, x: f32, z: f32| w.terrain.cell_of(x, z).is_some_and(|(cx, cz)| !w.terrain.is_water(cx, cz) && !w.blocked_cells().contains(&(cx, cz)));
            let (bx, bz) = if on_land(self, ahead.0, ahead.1) { ahead } else { (self.agent.x, self.agent.z) };
            let mut pieces = Vec::new();
            let mut rest = a;
            while rest > AMOUNT_EPS {
                pieces.push(rest.min(1.0));
                rest -= 1.0;
            }
            let (sx, sz) = (-self.agent.yaw.sin(), self.agent.yaw.cos());
            let mut ids = Vec::new();
            for (j, p) in pieces.iter().enumerate() {
                let off = (j as f32 - (pieces.len() as f32 - 1.0) * 0.5) * 0.13;
                let (lx, lz) = play
                    .and_then(|pl| self.release_landing(pl, piece_index))
                    .filter(|&(x, z)| on_land(self, x, z))
                    .unwrap_or((bx + sx * off, bz + sz * off));
                piece_index += 1;
                let d = DropItem { id: self.next_drop, kind: k, x: lx, z: lz, amount: *p, support: Support::Ground, y_m: 0.0 };
                self.next_drop += 1;
                ids.push(drop_id(d.id));
                self.drops.push(d);
            }
            events.push(self.stamp(WorldEvent::ItemDropped { item: k, amount: a, waypoint: None, drops: ids }));
        }
    }

    fn complete_split(&mut self, id: u32, portion: f32, events: &mut Vec<TimedEvent>) {
        let Some(d) = self.drops.iter_mut().find(|d| d.id == id) else { return };
        let piece = d.amount * portion;
        d.amount -= piece;
        let (kind, left, x, z, support, y_m) = (d.kind, d.amount, d.x, d.z, d.support, d.y_m);
        let (ax, az) = (self.agent.x - x, self.agent.z - z);
        let n = ax.hypot(az).max(1e-3);
        let new = DropItem { id: self.next_drop, kind, x: x + 0.12 * ax / n, z: z + 0.12 * az / n, amount: piece, support, y_m };
        self.next_drop += 1;
        events.push(self.stamp(WorldEvent::ItemSplit { drop: drop_id(id), piece: drop_id(new.id), item: kind, amount: piece, left }));
        self.drops.push(new);
    }

    pub fn terrain_grid(&self) -> TerrainGrid {
        let t = &self.terrain;
        let mut heights = Vec::with_capacity(t.w * t.d);
        let mut curbs = Vec::with_capacity(t.w * t.d);
        let mut ground = Vec::with_capacity(t.w * t.d);
        for cz in 0..t.d {
            for cx in 0..t.w {
                heights.push(t.height_cm(cx, cz) as i16);
                curbs.push((t.curb_m(cx, cz) * 100.0).round() as u8);
                ground.push(t.ground_at(cx, cz).code());
            }
        }
        TerrainGrid { cells_w: t.w, cells_d: t.d, cell_m: CELL_M, heights_cm: heights, curb_cm: curbs, ground, kind: self.cfg.terrain, curb_kind: t.curb_kind_codes().to_vec() }
    }

    pub(crate) fn carry_views(list: &[(ItemKind, f32)]) -> Vec<CarryView> {
        list.iter()
            .filter(|(_, n)| *n > AMOUNT_EPS)
            .map(|(k, n)| CarryView {
                item: *k,
                label_ko: k.spec().label_ko.into(),
                count: (*n + AMOUNT_EPS).floor() as u32,
                amount: *n,
                mass_kg: k.spec().mass_kg * *n,
                unit_kg: k.spec().mass_kg,
            })
            .collect()
    }

    pub fn snapshot(&self) -> WorldSnapshot {
        let (w, d) = self.terrain.size_m();
        let at = |x: f32, z: f32| Pos3 { x, y: self.surface_m(x, z), z };
        WorldSnapshot {
            tick: self.tick,
            time_s: self.time_s,
            seed: self.cfg.seed,
            bounds: Bounds { width_m: w, depth_m: d, cells_w: self.terrain.w, cells_d: self.terrain.d, cell_m: CELL_M },
            agent: AgentView {
                pos: at(self.agent.x, self.agent.z),
                yaw_deg: self.agent.yaw.to_degrees(),
                speed_mps: self.agent.speed_mps,
                base: format!("{:?}", self.hand.base.kind).to_lowercase(),
                fingers: u8::from(self.hand.config.fingers),
                carrying: Self::carry_views(&self.agent.carried),
                carried_kg: self.carried_kg(),
                capacity_kg: self.capacity_kg(),
                battery_ratio: self.battery_ratio(),
                action: self.current_action(),
                motion: self.motion_view(),
                belief: self.odom.cfg.enabled.then(|| {
                    let (x, z, yaw) = self.believed_pose();
                    BeliefView { x, z, yaw_deg: yaw.to_degrees(), error_m: self.belief_error_m(), since_fix_m: self.odom.since_fix_m }
                }),
            },
            waypoints: self
                .waypoints
                .iter()
                .map(|p| WaypointView {
                    id: waypoint_id(p.id),
                    key: p.key.clone(),
                    label_ko: p.label_ko.clone(),
                    pos: at(p.x, p.z),
                    stock: Self::carry_views(&p.stock),
                })
                .collect(),
            nodes: self
                .nodes
                .iter()
                .map(|n| NodeView { id: node_id(n.id), kind: n.kind, label_ko: n.kind.spec().label_ko.into(), pos: at(n.x, n.z), hp: (n.hp / n.max_hp).clamp(0.0, 1.0) })
                .collect(),
            drops: self
                .drops
                .iter()
                .map(|dr| DropView { id: drop_id(dr.id), item: dr.kind, label_ko: dr.kind.spec().label_ko.into(), pos: Pos3 { x: dr.x, y: self.drop_base_m(dr), z: dr.z }, amount: dr.amount, support: dr.support })
                .collect(),
            terrain: self.cfg.terrain,
            site: self.site.as_ref().map(|s| s.view(&self.fixtures, &self.terrain)),
            crew: self.crew_views(),
            rider: self.rider.filter(|r| r.off()).map(|r| Pos3 { x: r.x, y: self.surface_m(r.x, r.z), z: r.z }),
        }
    }

    pub fn posture_now(&self) -> (Posture, String) {
        let (post, state) = self.posture_action();
        // Between acts a walker's load is in its hands before it, as far as it has slid there.
        let between_acts = matches!(&self.active, None | Some((_, Active::Move(_) | Active::Turn { .. })) | Some((_, Active::Work(WorkState { kind: WorkKind::Look { .. }, .. }))));
        let post = if between_acts { self.with_load_held(post) } else { post };
        (Posture { ground: self.ground_here().surface(), ..post }, state)
    }

    fn posture_action(&self) -> (Posture, String) {
        let cargo = self.carried_kg();
        let bucket = |deg: f32, step: f32| -> String {
            if step > 0.0 {
                "step".into()
            } else if deg.abs() < 3.0 {
                "flat".into()
            } else {
                format!("{}_{}", if deg > 0.0 { "up" } else { "down" }, if deg.abs() < 10.0 { "lo" } else { "hi" })
            }
        };
        match &self.active {
            Some((_, Active::Move(m))) => {
                let Some(e) = m.edges.get(m.idx) else { return (Posture::quiet(cargo), "flat|move".into()) };
                let rise = self.terrain.height_m(e.to.0, e.to.1) - self.terrain.height_m(e.from.0, e.from.1) + self.terrain.ramp_rise_m(e.from, e.to).unwrap_or(0.0);
                let slope = if rise >= 0.0 { e.slope_deg } else { -e.slope_deg };
                let ground = self.ground_here();
                let mut p = Posture { slope_deg: slope, moving: true, ..Posture::quiet(cargo) };
                let b = bucket(slope, if m.pause_left > 0.0 { e.step_m } else { 0.0 });
                let mut b_brake = false;
                if m.pause_left > 0.0 {
                    p.source = Source::StepUp;
                    p.step_m = e.step_m;
                    p.single_support = self.body.legged();
                } else if m.elapsed_s < self.drive_spec().max_speed_mps / self.planned_accel(slope, ground, 1.0) {
                    p.source = Source::Accel;
                    p.accel_mps2 = self.planned_accel(slope, ground, 1.0);
                } else if self.remaining_m(m) < self.braking_distance_m(slope, ground) {
                    p.source = Source::Accel;
                    p.accel_mps2 = -self.planned_accel(slope, ground, -1.0);
                    b_brake = true;
                } else if self.body.legged() {
                    p.source = Source::FootLift;
                    p.single_support = true;
                } else if slope.abs() >= 3.0 {
                    p.source = Source::Slope;
                }
                (p, if b_brake { format!("{b}|brake") } else { format!("{b}|move") })
            }
            Some((_, Active::Work(w))) => match &w.kind {
                WorkKind::Mine { .. } => (Posture::arm_raise(cargo), "flat|mine".into()),
                WorkKind::Look { .. } => (Posture::quiet(cargo), "flat|look".into()),
                WorkKind::Pick { drop, two_arms, .. } => {
                    let kg = self.drop_item(*drop).map_or(0.0, |d| d.mass_kg());
                    (Posture::arm_lift(kg, *two_arms, cargo), "flat|pick".into())
                }
                WorkKind::Deliver { .. } | WorkKind::Drop { .. } => {
                    let kg = self.agent.carried.iter().map(|(k, n)| k.spec().mass_kg * n.min(1.0)).fold(0.0f32, f32::max);
                    let tag = if matches!(w.kind, WorkKind::Drop { .. }) { "flat|drop" } else { "flat|deliver" };
                    (Posture::arm_lift(kg, false, (cargo - kg).max(0.0)), tag.into())
                }
                WorkKind::Split { .. } => (Posture::arm_raise(cargo), "flat|split".into()),
            },
            Some((_, Active::Push(_))) => (Posture { moving: true, ..Posture::quiet(cargo) }, "flat|push".into()),
            _ => (Posture::quiet(cargo), "flat|idle".into()),
        }
    }

    fn update_balance(&mut self, dt: f32) {
        self.solve_balance(dt, false);
    }

    /// The balance now, its heat and wear over `dt`, and its motion when the posture changed: `slid` when the load moved
    /// in the hands since the last solve, which changes the posture without changing its name.
    fn solve_balance(&mut self, dt: f32, slid: bool) {
        let (post, state) = self.posture_now();
        self.ride.solved_shift = post.cargo_shift_m;
        let grip = match &self.active {
            Some((_, Active::Work(WorkState { kind: WorkKind::Pick { drop, .. }, .. }))) => self.drop_item(*drop).map(|d| self.grip_capacity_for(d.kind, d.amount)),
            _ => None,
        }
        .unwrap_or(1e3);
        let rep = self.body.solve(&post, &self.ff_for(post.source), grip);
        for j in &rep.joints {
            let Some(act) = self.joint_actuator(&j.joint) else { continue };
            let h = self.heat.entry(j.joint.clone()).or_insert(0.0);
            *h += (dt / act.thermal_tau_s).min(1.0) * (j.effort * j.effort - *h);
            *self.wear.entry(j.joint.clone()).or_insert(0.0) += noto_core::wear::damage(&act, j.torque_nm, dt);
        }
        let key = format!("{}|{state}", post.source.id());
        let changed = slid || self.balance.as_ref().map_or(true, |(k, _)| *k != key);
        let digest = if changed {
            let roll = dynamics::simulate(&self.body, &self.last_posture, &post, &self.ff_for(post.source), grip, &DynCfg::default());
            let d = roll.digest();
            self.motion = Some(Motion { rollout: roll, posture: post, start_s: self.time_s, state: state.clone() });
            self.last_posture = post;
            Some(d)
        } else {
            None
        };
        if changed || self.tick.saturating_sub(self.balance_since) >= BALANCE_RESAMPLE_TICKS {
            self.balance_since = self.tick;
            if self.balance_events.len() < 2000 {
                self.balance_events.push(BalanceEvent { tick: self.tick, time_s: self.time_s, state: state.clone(), report: rep.clone(), dynamics: digest, posture: post });
            }
        }
        self.balance = Some((key, rep));
    }

    pub fn body_pose(&self) -> Option<BodyPose> {
        if let Some(p) = self.act_body_pose() {
            return Some(p);
        }
        match &self.motion {
            Some(m) => m.rollout.pose_at(&self.body, &m.posture, (self.time_s - m.start_s) as f32),
            None => {
                let q = Posture::quiet(self.carried_kg());
                let r = dynamics::simulate(&self.body, &q, &q, &FeedForward::default(), 1e3, &DynCfg { settle_s: 0.05, ..DynCfg::default() });
                r.pose_at(&self.body, &q, 0.0)
            }
        }
    }

    pub fn joint_actuator(&self, joint: &str) -> Option<noto_core::parts::Actuator> {
        match joint {
            n if n.starts_with("shoulder") => Some(self.body.arm.shoulder),
            n if n.starts_with("elbow") => Some(self.body.arm.elbow),
            "waist" => self.body.belly.pitch,
            "neck" => Some(self.body.head.neck),
            "knee" | "hip_roll" => Some(self.body.leg.knee),
            "ankle" => Some(self.body.leg.ankle),
            _ => None,
        }
    }

    pub fn joint_load(&self) -> Option<(String, f32, f32, f32)> {
        let (_, r) = self.balance.as_ref()?;
        r.joints
            .iter()
            .filter(|j| !j.joint.starts_with("grip"))
            .map(|j| {
                let rms = self.heat.get(&j.joint).copied().unwrap_or(0.0).max(0.0).sqrt();
                (j.joint.clone(), j.peak_ratio.max(rms), j.peak_ratio, rms)
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
    }

    pub fn most_worn(&self) -> Option<(String, f32)> {
        self.wear.iter().max_by(|a, b| a.1.total_cmp(b.1)).map(|(k, v)| (k.clone(), *v))
    }

    pub fn balance_summary(&self) -> Option<BalanceSummary> {
        let (key, r) = self.balance.as_ref()?;
        let worst = r.worst();
        Some(BalanceSummary {
            plan: self.cfg.body.id(),
            source: r.source.id().into(),
            source_ko: r.source.label_ko().into(),
            state: key.splitn(2, '|').nth(1).unwrap_or("").to_string(),
            margin: r.margin,
            margin_noff: r.margin_noff,
            trunk_lean_deg: r.trunk_lean_deg,
            head_tilt_deg: r.head_pitch_deg.abs().max(r.head_roll_deg.abs()),
            worst_joint: worst.map(|w| w.joint.clone()),
            worst_ratio: worst.map_or(0.0, |w| w.peak_ratio),
            effort_sum: r.effort_sum,
            mass_kg: r.mass_kg,
            com_height_m: r.com[2],
            heat_max: self.heat.values().copied().fold(0.0, f32::max),
            load_joint: self.joint_load().map(|l| l.0),
            load_index: self.joint_load().map_or(0.0, |l| l.1),
            load_rms: self.joint_load().map_or(0.0, |l| l.3),
            joints: r.joints.clone(),
            couplings: r.couplings.clone(),
            ground: self.ground_here().label_ko().into(),
            contact: Some(r.contact),
        })
    }

    pub fn body_inputs(&self, vision_confidence: Option<f32>, frame_ms: Option<f32>) -> BodyInputs {
        BodyInputs {
            base: self.hand.base.kind,
            morph: self.body.morph,
            fingers: u8::from(self.hand.config.fingers),
            vision_confidence,
            frame_ms,
            energy_used_j: self.agent.energy_used_j,
            carried_kg: self.carried_kg(),
            grip_margin: self.agent.grip_margin,
            joint_strikes: self.agent.strikes,
            arm_wear: ["shoulder_r", "elbow_r"].iter().filter_map(|j| self.wear.get(*j)).copied().fold(0.0, f32::max),
            wear_max: self.most_worn().map_or(0.0, |w| w.1),
            wear_joint: self.most_worn().map(|w| w.0),
            speed_factor: (self.agent.speed_mps / self.drive_spec().max_speed_mps).clamp(0.0, 1.0),
            terrain_margin: self.agent.terrain_margin,
            limits: Some({
                let l = self.limits();
                BodyLimits { battery_j: self.battery_capacity_j(), capacity_kg: l.capacity_kg, max_speed_mps: l.max_speed_mps, max_slope_deg: l.max_slope_deg, max_step_m: l.max_step_m }
            }),
            body_mass_kg: Some(self.body.mass_kg()),
            capacity_here: Some((self.capacity_here(), self.ground_here().label_ko().to_string())),
            balance: self.balance_summary(),
        }
    }

    pub fn capacity_here(&self) -> f32 {
        let slope = self.posture_now().0.slope_deg;
        self.body.capacity_on(&self.hand.base, self.ground_here().surface(), slope).min(self.capacity_kg())
    }

    pub fn body_status(&self, vision_confidence: Option<f32>, frame_ms: Option<f32>) -> BodyStatus {
        compute_body(&self.body_inputs(vision_confidence, frame_ms))
    }

    fn pieces_to_set_down(&self, item: Option<ItemKind>, amount: Option<f32>) -> Vec<(ItemKind, f32)> {
        let mut left = amount.unwrap_or(f32::INFINITY);
        let mut out = Vec::new();
        for (k, n) in &self.agent.carried {
            if !item.map_or(true, |i| i == *k) {
                continue;
            }
            let mut rest = n.min(left);
            left -= rest;
            while rest > AMOUNT_EPS {
                out.push((*k, rest.min(1.0)));
                rest -= 1.0;
            }
        }
        out
    }
}

fn top_grasp_info(hand: &Hand, item: ItemKind, amount: f32) -> GraspInfo {
    let s = item.spec();
    let mass = s.mass_kg * amount;
    let mu = noto_core::act::material::pad_on(material_of(item)).mean;
    let cap = noto_core::act::grasp::capacity(hand, s.piece_size_m(amount), mass, mu);
    let need = mass * noto_core::balance::G * GRIP_SAFETY;
    let margin = if need > 1e-6 { mu * cap.squeeze_n / need } else { 99.0 };
    GraspInfo { item, stable: cap.contacts >= 2 && margin >= 1.0, margin: margin.min(99.0), max_payload_kg: mu * cap.squeeze_n / (noto_core::balance::G * GRIP_SAFETY), grasp_type: cap.grasp_type }
}

#[cfg(test)]
mod tests {
    use super::*;
    use noto_core::{BaseKind, FingerCount};

    fn world(fingers: FingerCount, base: BaseKind) -> World {
        let mut hand = NotoConfig::preset(fingers);
        hand.base = base;
        World::new(WorldConfig { seed: 7, cells: 64, noto: hand, body: BodyPlan::standard(), terrain: TerrainKind::Open, parcels: Some(0), ..WorldConfig::default() }).unwrap()
    }

    #[test]
    fn a_pick_never_shuffles_the_body_onto_ground_it_cannot_drive_onto() {
        let mut w = world(FingerCount::Five, BaseKind::HumanFoot);
        let (cx, cz) = w.agent_cell();
        let (ax, az) = (w.agent.x, w.agent.z);
        assert!(w.can_step_to(ax + 0.1, az), "within its own cell");
        assert!(w.can_step_to(ax + CELL_M, az), "onto a level neighbour");
        let h = (w.terrain.height_m(cx, cz) * 100.0).round() as i32;
        w.terrain.set_height_cm(cx + 1, cz, h + 90);
        assert!(!w.can_step_to(ax + CELL_M, az));
        assert!(!w.can_step_to(ax + 2.0 * CELL_M, az));
    }

    #[test]
    fn parcels_lie_about_and_one_waits_in_front_of_the_base_camp() {
        let mut hand = NotoConfig::preset(FingerCount::Five);
        hand.base = BaseKind::HumanFoot;
        let w = World::new(WorldConfig { noto: hand, ..WorldConfig::default() }).unwrap();
        let parcels: Vec<&DropItem> = w.drops.iter().filter(|d| d.kind == ItemKind::Parcel).collect();
        assert_eq!(parcels.len(), w.drops.len(), "only parcels lie about at the start");
        assert!(parcels.len() >= 3, "{}", parcels.len());
        let p = parcels[0];
        let (s, c) = w.agent.yaw.sin_cos();
        let (rx, rz) = (p.x - w.agent.x, p.z - w.agent.z);
        let (ahead, right) = (rx * c + rz * s, -rx * s + rz * c);
        assert!((ahead - CAMP_PARCEL_AHEAD_M).abs() < 1e-3 && (right - CAMP_PARCEL_RIGHT_M).abs() < 1e-3, "{ahead} {right}");
        let blocked = w.blocked_cells();
        assert!(parcels.iter().all(|d| w.terrain.cell_of(d.x, d.z).is_some_and(|c| !blocked.contains(&c) && !w.terrain.is_water(c.0, c.1))));
        let again = World::new(WorldConfig { noto: hand, ..WorldConfig::default() }).unwrap();
        assert!(w.drops.iter().zip(&again.drops).all(|(a, b)| (a.x, a.z, a.id) == (b.x, b.z, b.id)));
        assert!(w.next_drop > parcels.iter().map(|d| d.id).max().unwrap());
        let bare = World::new(WorldConfig { noto: hand, parcels: Some(0), ..WorldConfig::default() }).unwrap();
        assert!(w.nodes.iter().zip(&bare.nodes).all(|(a, b)| (a.x, a.z) == (b.x, b.z)));
    }

    #[test]
    fn the_belief_drifts_while_the_robot_drives_and_maps_back() {
        let mut w = World::new(WorldConfig { parcels: Some(0), ..WorldConfig::default() }).unwrap();
        assert!(w.belief_error_m() < 1e-6, "it starts on its pad");
        let camp = w.waypoint_by_key("base_camp").unwrap().clone();
        let goal = (camp.x + 6.0, camp.z + 2.0);
        w.command(Action::MoveTo { x: goal.0, z: goal.1, reach_m: 0.0 }).unwrap();
        run(&mut w, 120.0);
        let err = w.belief_error_m();
        assert!(err > 1e-4 && err < 0.5, "{err}");
        assert!((6.0..13.0).contains(&w.odom.travelled_m), "{}", w.odom.travelled_m);
        let (bx, bz, _) = w.believed_pose();
        let me = w.belief_of(w.agent.x, w.agent.z);
        assert!((me.0 - bx).abs() < 1e-4 && (me.1 - bz).abs() < 1e-4);
        let p = w.true_of(10.0, 9.0);
        let q = w.belief_of(p.0, p.1);
        assert!((q.0 - 10.0).abs() < 1e-3 && (q.1 - 9.0).abs() < 1e-3);
        let mut e = World::new(WorldConfig { parcels: Some(0), odometry: OdomCfg::exact(), ..WorldConfig::default() }).unwrap();
        e.command(Action::MoveTo { x: goal.0, z: goal.1, reach_m: 0.0 }).unwrap();
        run(&mut e, 120.0);
        assert!(e.belief_error_m() < 1e-3);
    }

    fn run(w: &mut World, max_s: f32) -> Vec<TimedEvent> {
        let mut all = Vec::new();
        let mut t = 0.0;
        while w.busy() && t < max_s {
            all.extend(w.tick());
            t += TICK_S;
        }
        all
    }

    fn ok_done(events: &[TimedEvent]) -> bool {
        events.iter().any(|e| matches!(&e.event, WorldEvent::ActionDone { ok: true, .. }))
    }

    fn nearest_node(w: &World, kind: NodeKind) -> u32 {
        w.nodes
            .iter()
            .filter(|n| n.kind == kind)
            .min_by(|a, b| {
                let da = (a.x - w.agent.x).hypot(a.z - w.agent.z);
                let db = (b.x - w.agent.x).hypot(b.z - w.agent.z);
                da.total_cmp(&db)
            })
            .unwrap()
            .id
    }

    fn mine_nearest(w: &mut World, kind: NodeKind) -> Vec<TimedEvent> {
        let id = nearest_node(w, kind);
        let (x, z) = {
            let n = w.node(id).unwrap();
            (n.x, n.z)
        };
        w.command(Action::MoveTo { x, z, reach_m: 0.8 }).expect("route to the node");
        let mut ev = run(w, 600.0);
        w.command(Action::Mine { node: id }).expect("in reach");
        ev.extend(run(w, 600.0));
        ev
    }

    #[test]
    fn same_seed_gives_the_same_world() {
        let a = world(FingerCount::Five, BaseKind::Track);
        let b = world(FingerCount::Five, BaseKind::Track);
        assert_eq!(a.nodes.len(), b.nodes.len());
        for (x, y) in a.nodes.iter().zip(&b.nodes) {
            assert_eq!((x.kind, x.x, x.z), (y.kind, y.x, y.z));
        }
    }

    #[test]
    fn world_has_every_resource_and_two_waypoints() {
        let w = world(FingerCount::Five, BaseKind::Track);
        for k in NodeKind::ALL {
            assert!(w.nodes.iter().filter(|n| n.kind == k).count() >= 2, "{k:?}");
        }
        assert_eq!(w.waypoints.len(), 2);
        let cells: HashSet<_> = w.nodes.iter().map(|n| w.terrain.cell_of(n.x, n.z).unwrap()).collect();
        assert_eq!(cells.len(), w.nodes.len(), "one node per cell");
    }

    #[test]
    fn robot_starts_at_the_base_camp_on_level_ground() {
        let w = world(FingerCount::Five, BaseKind::Wheel);
        let camp = w.waypoint_by_key("base_camp").unwrap();
        assert!((w.agent.x - camp.x).abs() < 1e-4 && (w.agent.z - camp.z).abs() < 1e-4);
        assert!(w.agent_ground_m().abs() < 1e-6);
    }

    #[test]
    fn mining_a_tree_takes_the_time_the_finger_count_implies() {
        for (fingers, expect) in [(FingerCount::Five, 3.0f32), (FingerCount::Two, 7.5)] {
            // A walker swings at it from where it stands; a lift truck's rider (tracks) steps down, walks over, swings,
            // walks back and climbs on.
            for base in [BaseKind::HumanFoot, BaseKind::Track] {
                let mut w = world(fingers, base);
                let id = nearest_node(&w, NodeKind::Tree);
                let n = w.node(id).unwrap().clone();
                w.agent.x = n.x - 0.6;
                w.agent.z = n.z;
                w.agent.yaw = 0.0;
                let t0 = w.time_s;
                w.command(Action::Mine { node: id }).unwrap();
                let (mut walked, mut off) = (0.0f32, false);
                let mut ev = Vec::new();
                for _ in 0..600 {
                    // Steps walked on foot (stepping down beside the seat is not walking).
                    let before = w.rider.filter(|r| r.off()).map(|r| (r.x, r.z));
                    ev.extend(w.tick());
                    if let Some(r) = w.rider {
                        off |= r.off();
                        if let Some(a) = before.filter(|_| r.off()) {
                            walked += (r.x - a.0).hypot(r.z - a.1);
                        }
                    }
                    if !w.busy() {
                        break;
                    }
                }
                assert!(ok_done(&ev), "{ev:?}");
                let dt = (w.time_s - t0) as f32;
                let want = match base {
                    BaseKind::HumanFoot => expect,
                    _ => expect + traffic::DISMOUNT_S + traffic::REMOUNT_S + walked / traffic::RIDER_WALK_MPS,
                };
                // Each of the rider's four phase changes ends on a tick: up to a tick each on top.
                let tol = if base == BaseKind::HumanFoot { 0.25 } else { 0.25 + 4.0 * TICK_S };
                assert!((dt - want).abs() < tol, "{fingers:?} {base:?}: {dt} s vs {want} s");
                assert_eq!(off, base != BaseKind::HumanFoot, "only the lift truck's rider steps down");
                assert!(w.rider.is_none(), "back on the seat");
                assert!(w.node(id).is_none());
                assert_eq!(w.drops.len(), 3);
            }
        }
    }

    #[test]
    fn out_of_reach_actions_fail_immediately() {
        let mut w = world(FingerCount::Five, BaseKind::Track);
        let id = w.nodes[0].id;
        w.agent.x = 0.0;
        w.agent.z = 0.0;
        match w.command(Action::Mine { node: id }) {
            Err(ActionError::OutOfReach { distance_m, reach_m }) => assert!(distance_m > reach_m),
            other => panic!("{other:?}"),
        }
        assert!(matches!(w.command(Action::Mine { node: 9999 }), Err(ActionError::UnknownTarget)));
    }

    #[test]
    fn a_second_command_while_busy_is_refused() {
        let mut w = world(FingerCount::Five, BaseKind::Track);
        w.command(Action::Turn { yaw_deg: 170.0 }).unwrap();
        assert_eq!(w.command(Action::Turn { yaw_deg: 0.0 }), Err(ActionError::Busy));
        let ev = w.cancel();
        assert!(matches!(&ev[0].event, WorldEvent::ActionDone { ok: false, .. }));
        assert!(!w.busy());
    }

    #[test]
    fn driving_uses_the_energy_the_drive_model_predicts() {
        let mut w = world(FingerCount::Five, BaseKind::Track);
        let camp = w.waypoint_by_key("base_camp").unwrap().clone();
        let goal = (camp.x + 3.0, camp.z);
        let plan = w.plan_route(Goal::Cell(w.terrain.cell_of(goal.0, goal.1).unwrap())).unwrap();
        w.agent.yaw = 0.0;
        w.command(Action::MoveTo { x: goal.0, z: goal.1, reach_m: 0.0 }).unwrap();
        let ev = run(&mut w, 120.0);
        assert!(ok_done(&ev));
        assert!((w.agent.x - cell_center(w.terrain.cell_of(goal.0, goal.1).unwrap().0, 0).0).abs() < 0.3);
        assert!(w.agent.energy_used_j >= plan.energy_j as f64 * 0.8 - 1.0, "the base drives from where it stands, not from its cell's centre: {} vs {}", w.agent.energy_used_j, plan.energy_j);
        assert!(w.agent.energy_used_j <= plan.energy_j as f64 * 1.15 + 20.0, "{} vs {}", w.agent.energy_used_j, plan.energy_j);
    }

    #[test]
    fn a_felled_trees_logs_do_not_lie_inside_one_another() {
        let mut w = world(FingerCount::Five, BaseKind::HumanFoot);
        mine_nearest(&mut w, NodeKind::Tree);
        let size = ItemKind::Log.spec().size_m;
        let logs: Vec<_> = w.drops.iter().filter(|d| d.kind == ItemKind::Log).collect();
        assert_eq!(logs.len(), 3);
        for (i, a) in logs.iter().enumerate() {
            for b in &logs[i + 1..] {
                assert!((a.x - b.x).abs() >= size[0] || (a.z - b.z).abs() >= size[2], "{:?} and {:?} overlap", (a.x, a.z), (b.x, b.z));
            }
        }
    }

    #[test]
    fn five_finger_legs_carry_a_log_to_the_warehouse() {
        let mut w = world(FingerCount::Five, BaseKind::HumanFoot);
        let ev = mine_nearest(&mut w, NodeKind::Tree);
        assert!(ev.iter().any(|e| matches!(e.event, WorldEvent::NodeBroken { .. })));
        let drop = w.drops[0].clone();
        w.command(Action::MoveTo { x: drop.x, z: drop.z, reach_m: 0.5 }).unwrap();
        run(&mut w, 60.0);
        w.command(Action::pick(drop.id)).unwrap();
        let ev = run(&mut w, 20.0);
        assert!(ev.iter().any(|e| matches!(e.event, WorldEvent::ItemPicked { .. })), "{ev:?}");
        assert_eq!(w.carried_count(ItemKind::Log), 1);
        for k in ["bend_bottom", "contact", "lift_off", "stowed"] {
            assert!(ev.iter().any(|e| matches!(&e.event, WorldEvent::Act { kind, .. } if kind == k)), "no {k}");
        }
        let wh = w.waypoint_by_key("warehouse").unwrap().clone();
        w.command(Action::MoveTo { x: wh.x, z: wh.z, reach_m: 1.0 }).unwrap();
        let ev = run(&mut w, 600.0);
        assert!(ok_done(&ev));
        w.command(Action::Deliver { waypoint: wh.id, item: Some(ItemKind::Log) }).unwrap();
        run(&mut w, 20.0);
        assert_eq!(w.carried_count(ItemKind::Log), 0);
        let stock = &w.waypoint(wh.id).unwrap().stock;
        assert_eq!(stock, &vec![(ItemKind::Log, 1.0)]);
    }

    #[test]
    fn delivery_without_cargo_is_refused() {
        let mut w = world(FingerCount::Five, BaseKind::Track);
        assert_eq!(w.command(Action::Deliver { waypoint: 1, item: None }), Err(ActionError::NothingToDeliver));
    }

    #[test]
    fn the_curb_wall_costs_a_wheel_a_long_detour_but_a_foot_crosses_it() {
        let wheel = world(FingerCount::Five, BaseKind::Wheel);
        let foot = world(FingerCount::Five, BaseKind::HumanFoot);
        let wh = wheel.waypoint_by_key("warehouse").unwrap();
        let goal = Goal::Within { x: wh.x, z: wh.z, reach_m: 1.0 };
        let pw = wheel.plan_route(goal).expect("wheel reaches the warehouse through the gap");
        let pf = foot.plan_route(goal).expect("foot route");
        assert!(pw.distance_m > pf.distance_m + 8.0, "wheel {} m, foot {} m", pw.distance_m, pf.distance_m);
        assert!(pf.edges.iter().any(|e| e.step_m > 0.0));
        assert!(pw.edges.iter().all(|e| e.step_m == 0.0));
    }

    #[test]
    fn grasp_table_reflects_the_hand() {
        let five = world(FingerCount::Five, BaseKind::Track);
        let two = world(FingerCount::Two, BaseKind::Track);
        for k in ItemKind::ALL {
            let a = five.grasp_info(k);
            let b = two.grasp_info(k);
            assert!(a.margin.is_finite() && b.margin.is_finite());
        }
        let light = five.grasp_info(ItemKind::Coal).margin;
        let heavy = five.grasp_info(ItemKind::Stone).margin;
        assert!(light >= heavy * 0.99, "coal {light} vs stone {heavy}");
    }

    #[test]
    fn item_catalogue_spans_what_the_hands_can_hold() {
        let holdable = |f: FingerCount| {
            let w = world(f, BaseKind::Track);
            ItemKind::ALL.into_iter().filter(|&k| w.grasp_info(k).stable || 2.0 * w.grasp_info(k).margin >= 1.0).count()
        };
        let (two, five) = (holdable(FingerCount::Two), holdable(FingerCount::Five));
        assert!(five >= 3, "a five-finger hand should hold at least three of four items, holds {five}");
        assert!(two <= five, "a two-finger hand holds no more items ({two} vs {five})");
        for k in ItemKind::ALL {
            let (a, b) = (world(FingerCount::Two, BaseKind::Track).grasp_info(k).margin, world(FingerCount::Five, BaseKind::Track).grasp_info(k).margin);
            assert!(a <= b + 1e-3, "{k:?}: two fingers {a} vs five {b}");
        }
        assert!(two >= 1, "the lightest item should be holdable even with two fingers");
    }

    #[test]
    fn an_unstable_grasp_blocks_the_pickup() {
        let mut w = world(FingerCount::Two, BaseKind::HumanFoot);
        let bad = ItemKind::ALL.into_iter().find(|&k| 2.0 * w.grasp_for(k, 8.0).margin < 1.0).expect("the two-finger hand cannot hold every item eight times over");
        w.drops.push(DropItem { amount: 8.0, ..DropItem::whole(500, bad, w.agent.x + 0.4, w.agent.z) });
        match w.command(Action::pick(500)) {
            Err(ActionError::GraspUnstable { margin }) => assert!(margin < 1.0 + 1e-3),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn capacity_limit_stops_the_pickup() {
        let mut w = world(FingerCount::Five, BaseKind::Mecanum);
        let item = ItemKind::ALL.into_iter().find(|&k| w.grasp_info(k).stable).expect("some item is holdable");
        let cap = w.capacity_kg();
        add_stock(&mut w.agent.carried, item, (cap / item.spec().mass_kg).floor());
        w.drops.push(DropItem::whole(501, item, w.agent.x + 0.4, w.agent.z));
        assert!(matches!(w.command(Action::pick(501)), Err(ActionError::CapacityExceeded { .. })));
    }

    #[test]
    fn mining_counts_joint_strikes_and_drains_the_battery() {
        let mut w = world(FingerCount::Five, BaseKind::Track);
        let before = w.battery_ratio();
        mine_nearest(&mut w, NodeKind::Tree);
        assert!(w.agent.strikes >= 3);
        assert!(w.battery_ratio() < before);
        let body = w.body_status(Some(0.8), Some(12.0));
        assert_eq!(body.parts.len(), 7);
    }

    #[test]
    fn a_depleted_battery_stops_everything() {
        let mut w = world(FingerCount::Five, BaseKind::Track);
        w.agent.energy_used_j = w.battery_capacity_j() + 1.0;
        assert_eq!(w.command(Action::Turn { yaw_deg: 10.0 }), Err(ActionError::BatteryDepleted));
        let mut w = world(FingerCount::Five, BaseKind::Track);
        w.command(Action::Turn { yaw_deg: 170.0 }).unwrap();
        w.agent.energy_used_j = w.battery_capacity_j() + 1.0;
        let ev = w.tick();
        assert!(ev.iter().any(|e| e.event == WorldEvent::BatteryDepleted));
        assert!(!w.busy());
    }

    #[test]
    fn snapshot_uses_prefixed_string_ids_and_serialises() {
        let w = world(FingerCount::Five, BaseKind::Track);
        let s = w.snapshot();
        assert!(s.nodes.iter().all(|n| n.id.starts_with('n')));
        assert!(s.waypoints.iter().all(|p| p.id.starts_with('w')));
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains("\"base\":\"track\""));
        let g = w.terrain_grid();
        assert_eq!(g.heights_cm.len(), g.cells_w * g.cells_d);
    }

    #[test]
    fn facing_a_point_turns_the_agent_toward_it() {
        let mut w = world(FingerCount::Five, BaseKind::Track);
        w.agent.yaw = 0.0;
        w.command(Action::Face { x: w.agent.x, z: w.agent.z + 5.0 }).unwrap();
        run(&mut w, 30.0);
        assert!((w.agent.yaw - std::f32::consts::FRAC_PI_2).abs() < 0.05);
    }

    fn world_with(body: BodyPlan, base: BaseKind) -> World {
        let mut hand = NotoConfig::preset(FingerCount::Five);
        hand.base = base;
        World::new(WorldConfig { seed: 7, cells: 64, noto: hand, body, terrain: TerrainKind::Open, ..WorldConfig::default() }).unwrap()
    }

    #[test]
    fn the_arm_preset_decides_what_can_be_picked_up() {
        let place = |w: &mut World, kind: ItemKind| {
            w.drops.push(DropItem::whole(600, kind, w.agent.x + 0.4, w.agent.z));
            w.command(Action::pick(600))
        };
        let mut budget = world_with(BodyPlan { arm: noto_core::Tier::Budget, ..BodyPlan::standard() }, BaseKind::HumanFoot);
        match place(&mut budget, ItemKind::Log) {
            Err(ActionError::LiftOverload { limiting, .. }) => assert!(limiting.contains("elbow"), "{limiting}"),
            other => panic!("{other:?}"),
        }
        let mut mid = world_with(BodyPlan::standard(), BaseKind::HumanFoot);
        assert!(place(&mut mid, ItemKind::Log).is_ok());
        let ev = run(&mut mid, 20.0);
        assert!(ev.iter().any(|e| matches!(e.event, WorldEvent::ItemPicked { .. })), "{ev:?}");
        let mut iron = world_with(BodyPlan { build: noto_core::dims::Build::Standard, ..BodyPlan::uniform(noto_core::Tier::Pro) }, BaseKind::HumanFoot);
        let t0 = iron.time_s;
        place(&mut iron, ItemKind::IronOre).unwrap();
        run(&mut iron, 20.0);
        assert!((iron.time_s - t0) as f32 >= PICKUP_TIME_S * TWO_ARM_LIFT_FACTOR - 0.15);
        let mut track = world_with(BodyPlan::standard(), BaseKind::Track);
        assert!(place(&mut track, ItemKind::Log).is_ok(), "a track lifts the log on its forks instead of reaching for it");
        let ev = run(&mut track, 20.0);
        assert!(ev.iter().any(|e| matches!(e.event, WorldEvent::ItemPicked { .. })), "{ev:?}");
        assert!(track.current_kin_pose().neck_deg.abs() < 1e-3, "the rider never bends to the floor");
    }

    #[test]
    fn mining_records_an_arm_raise_and_a_hot_budget_arm_mines_slower() {
        let mut w = world(FingerCount::Five, BaseKind::Track);
        mine_nearest(&mut w, NodeKind::Tree);
        let events = w.take_balance_events();
        let raise = events.iter().find(|e| e.report.source == Source::ArmRaise).expect("an arm-raise state");
        assert_eq!(raise.state, "flat|mine");
        assert!(raise.report.couplings.iter().any(|c| c.target == noto_core::Segment::Foot));
        let mut cold = world_with(BodyPlan { arm: noto_core::Tier::Budget, ..BodyPlan::default() }, BaseKind::Track);
        cold.heat.insert("elbow_r".into(), 2.0);
        assert!((cold.mining_rate() - 0.5).abs() < 1e-6, "a joint at heat 2 halves the strike rate");
    }

    fn rugged(seed: u64, base: BaseKind) -> World {
        let mut hand = NotoConfig::preset(FingerCount::Five);
        hand.base = base;
        World::new(WorldConfig { seed, cells: 64, noto: hand, body: BodyPlan::standard(), terrain: TerrainKind::Rugged, ..WorldConfig::default() }).unwrap()
    }

    #[test]
    fn the_rugged_map_has_sea_cliffs_and_ridges_and_stays_connected() {
        for seed in 1..=12u64 {
            let w = rugged(seed, BaseKind::Wheel);
            let l = w.layout.as_ref().expect("rugged layout");
            assert!(l.water_cells > 20, "seed {seed}: a sea");
            assert!(!l.mesas.is_empty() && !l.ridges.is_empty(), "seed {seed}: {l:?}");
            let wh = w.waypoint_by_key("warehouse").unwrap();
            assert!(w.plan_route(Goal::Within { x: wh.x, z: wh.z, reach_m: 1.0 }).is_some(), "seed {seed}: no wheel route to the warehouse");
            let t = &w.terrain;
            let cliff = (0..t.d).any(|z| (0..t.w - 1).any(|x| (t.height_cm(x, z) - t.height_cm(x + 1, z)).abs() >= 100));
            assert!(cliff, "seed {seed}: no cliff");
            let start = t.cell_of(w.agent.x, w.agent.z).unwrap();
            let blocked = w.blocked_cells();
            let times = nav::travel_times(&BaseKind::Wheel.spec(), t, &|x, z| blocked.contains(&(x, z)), start, f32::INFINITY);
            for k in NodeKind::ALL {
                let ok = w.nodes.iter().filter(|n| n.kind == k).filter(|n| {
                    let (cx, cz) = t.cell_of(n.x, n.z).unwrap();
                    (-1i32..=1).any(|dz| (-1i32..=1).any(|dx| t.in_bounds(cx as i32 + dx, cz as i32 + dz) && times[t.idx((cx as i32 + dx) as usize, (cz as i32 + dz) as usize)].is_finite()))
                }).count();
                assert!(ok >= 2, "seed {seed}: {k:?} has {ok} reachable nodes");
                assert!(w.nodes.iter().all(|n| !t.is_water(t.cell_of(n.x, n.z).unwrap().0, t.cell_of(n.x, n.z).unwrap().1)), "no node in the sea");
            }
        }
        let (a, b) = (rugged(5, BaseKind::Track), rugged(5, BaseKind::Track));
        assert_eq!(a.terrain_grid(), b.terrain_grid());
        assert!(world(FingerCount::Five, BaseKind::Track).layout.is_none());
    }

    #[test]
    fn pads_are_gravel_the_beach_is_dry_sand_behind_wet_sand_and_the_ground_reaches_the_posture() {
        let w = rugged(4, BaseKind::Wheel);
        let t = &w.terrain;
        let camp = w.waypoint_by_key("base_camp").unwrap();
        assert_eq!(w.ground_at(t.cell_of(camp.x, camp.z).unwrap()), Ground::Gravel);
        let count = |g: Ground| (0..t.d).flat_map(|z| (0..t.w).map(move |x| (x, z))).filter(|&(x, z)| t.ground_at(x, z) == g).count();
        assert!(count(Ground::WetSand) > 0 && count(Ground::Sand) > 0, "a beach with both bands");
        let shore = t.water_distance_m(8.0);
        for z in 0..t.d {
            for x in 0..t.w {
                match t.ground_at(x, z) {
                    Ground::WetSand => assert!(shore[t.idx(x, z)] <= 1.05 + 1e-3),
                    Ground::Sand => assert!(shore[t.idx(x, z)] > 1.05 && shore[t.idx(x, z)] <= 2.5 + 1e-3),
                    _ => {}
                }
            }
        }
        let (p, _) = w.posture_now();
        assert_eq!(p.ground, noto_core::Surface::Gravel);
        let mut flat = world(FingerCount::Five, BaseKind::Wheel);
        let cell = flat.terrain.cell_of(flat.agent.x, flat.agent.z).unwrap();
        let ahead = (cell.0 + 2, cell.1);
        let cost = |g: Ground| {
            let mut w = flat.clone();
            for c in [(cell.0 + 1, cell.1), ahead] {
                w.terrain.set_ground(c.0, c.1, g);
            }
            w.plan_route(Goal::Cell(ahead)).map(|p| p.energy_j / p.time_s.max(1e-6))
        };
        let (gravel, sand) = (cost(Ground::Gravel).unwrap(), cost(Ground::Sand).unwrap());
        assert!(sand > gravel * 1.2, "sand {sand} W vs gravel {gravel} W");
        flat.terrain.set_ground(ahead.0, ahead.1, Ground::Sand);
        flat.command(Action::MoveTo { x: cell_center(ahead.0, ahead.1).0, z: cell_center(ahead.0, ahead.1).1, reach_m: 0.0 }).unwrap();
        for _ in 0..200 {
            flat.tick();
            if flat.ground_here() == Ground::Sand {
                break;
            }
        }
        let s = flat.body_status(None, None);
        let slip = s.effects.iter().find(|e| e.id == "slip").unwrap();
        assert!(slip.detail.contains("모래") && slip.detail.contains("마찰 이용률"), "{}", slip.detail);
    }

    #[test]
    fn nothing_drives_into_the_sea_and_the_ramp_is_for_tracks_and_legs() {
        let w = rugged(3, BaseKind::Track);
        let t = &w.terrain;
        let water = (0..t.d).flat_map(|z| (0..t.w).map(move |x| (x, z))).find(|&(x, z)| t.is_water(x, z)).unwrap();
        assert!(w.plan_route(Goal::Cell(water)).is_none());
        let e = (0..t.d).flat_map(|z| (0..t.w - 1).map(move |x| (x, z))).find(|&(x, z)| {
            let d = t.height_cm(x + 1, z) - t.height_cm(x, z);
            d == crate::terrain::RAMP_RISE_CM && t.ground_at(x + 1, z) == Ground::Rock && t.height_cm(x + 1, z) > 0
        });
        if let Some((x, z)) = e {
            assert!(nav::edge(&BaseKind::Track.spec(), t, (x, z), (x + 1, z)).is_some());
            assert!(nav::edge(&BaseKind::Wheel.spec(), t, (x, z), (x + 1, z)).is_none());
        }
    }

    #[test]
    fn carried_items_can_be_set_down_on_the_ground_or_on_a_pile() {
        let mut w = world(FingerCount::Five, BaseKind::Track);
        assert_eq!(w.command(Action::Drop { item: None, amount: None }), Err(ActionError::NothingToDrop));
        add_stock(&mut w.agent.carried, ItemKind::Log, 2.0);
        add_stock(&mut w.agent.carried, ItemKind::Coal, 1.0);
        w.command(Action::Drop { item: Some(ItemKind::Log), amount: Some(1.0) }).unwrap();
        let ev = run(&mut w, 15.0);
        assert!(ev.iter().any(|e| matches!(&e.event, WorldEvent::ItemDropped { waypoint: Some(_), .. })));
        assert_eq!(w.waypoint_by_key("base_camp").unwrap().stock_of(ItemKind::Log), 1.0);
        assert_eq!(w.carried_count(ItemKind::Log), 1);
        w.agent.x += 5.0;
        w.agent.yaw = 0.0;
        let before = w.drops.len();
        w.command(Action::Drop { item: None, amount: None }).unwrap();
        let ev = run(&mut w, 30.0);
        assert!(ev.iter().any(|e| matches!(&e.event, WorldEvent::ItemDropped { waypoint: None, .. })));
        assert_eq!(w.drops.len(), before + 2, "a log and a coal lump");
        assert!(w.agent.carried.is_empty() && w.carried_kg() == 0.0);
        let d = w.drops.last().unwrap();
        assert!((d.x - w.agent.x).abs() < w.footprint().front_m + 0.5 && d.x > w.agent.x, "ahead of the robot: {} vs {}", d.x, w.agent.x);
        assert!(ev.iter().any(|e| matches!(&e.event, WorldEvent::Act { kind, .. } if kind == "fork_out")), "a track sets the load down with its forks and backs out");
    }

    #[test]
    fn a_stone_breaks_in_half_and_the_half_is_easier_to_hold() {
        // The waist that bends takes the hands down to something this low (the middle tier's trunk does not bend).
        let mut w = world_with(BodyPlan { build: noto_core::dims::Build::Standard, ..BodyPlan::uniform(noto_core::Tier::Pro) }, BaseKind::HumanFoot);
        w.agent.yaw = 0.0;
        w.drops.push(DropItem::whole(700, ItemKind::Stone, w.agent.x + 0.4, w.agent.z));
        w.drops.push(DropItem::whole(701, ItemKind::Log, w.agent.x - 0.4, w.agent.z));
        assert_eq!(w.command(Action::Split { drop: 701, portion: 0.5 }), Err(ActionError::NotDivisible));
        assert!(matches!(w.command(Action::Split { drop: 700, portion: 0.05 }), Err(ActionError::PieceTooSmall { .. })));
        let whole = w.grasp_for(ItemKind::Stone, 1.0);
        let half = w.grasp_for(ItemKind::Stone, 0.5);
        assert!(half.margin > whole.margin, "half {} vs whole {}", half.margin, whole.margin);
        let t0 = w.time_s;
        w.command(Action::Split { drop: 700, portion: 0.5 }).unwrap();
        let ev = run(&mut w, 30.0);
        let split = ev.iter().find_map(|e| match &e.event { WorldEvent::ItemSplit { piece, amount, left, .. } => Some((piece.clone(), *amount, *left)), _ => None }).expect("split");
        assert!((split.1 - 0.5).abs() < 1e-5 && (split.2 - 0.5).abs() < 1e-5);
        assert!(((w.time_s - t0) as f32 - w.split_time_s(ItemKind::Stone, 1.0)).abs() < 0.3);
        let piece: u32 = split.0[1..].parse().unwrap();
        assert!((w.drop_item(piece).unwrap().amount - 0.5).abs() < 1e-5);
        assert!((w.drop_item(700).unwrap().mass_kg() - 1.3).abs() < 1e-4);
        if half.stable {
            w.command(Action::pick(piece)).unwrap();
            run(&mut w, 20.0);
            assert!((w.carried_amount(ItemKind::Stone) - 0.5).abs() < 1e-5);
            assert_eq!(w.carried_count(ItemKind::Stone), 0, "half a stone is not a whole one");
        }
    }

    #[test]
    fn a_bus_servo_arm_wears_in_hours_and_a_qdd_arm_barely() {
        let mine = |plan: BodyPlan| {
            let mut w = world_with(plan, BaseKind::Track);
            mine_nearest(&mut w, NodeKind::Boulder);
            let s = w.body_status(None, None);
            (w.wear.get("shoulder_r").copied().unwrap_or(0.0), w.agent.strikes, s)
        };
        let (budget, strikes, s) = mine(BodyPlan::uniform(noto_core::Tier::Budget));
        let (mid, _, _) = mine(BodyPlan::default());
        assert!(strikes > 0 && budget > 0.0 && mid > 0.0);
        assert!(budget > 100.0 * mid, "spur-gear servos against a planetary QDD: {budget} vs {mid}");
        let arm = s.parts.iter().find(|p| p.id == "arm_r").unwrap();
        assert!(arm.detail.contains("수명") && arm.ratio < 1.0, "{}", arm.detail);
        let strain = s.effects.iter().find(|e| e.id == "strain").unwrap();
        assert!(strain.detail.contains("RMS"), "{}", strain.detail);
    }

    #[test]
    fn the_world_and_the_body_reset_on_their_own() {
        let mut w = world(FingerCount::Five, BaseKind::Track);
        let nodes = w.nodes.len();
        mine_nearest(&mut w, NodeKind::Tree);
        add_stock(&mut w.agent.carried, ItemKind::Log, 1.0);
        w.heat.insert("elbow_r".into(), 1.4);
        let (energy, strikes) = (w.agent.energy_used_j, w.agent.strikes);
        let worn = w.wear.get("shoulder_r").copied().unwrap_or(0.0);
        assert!(energy > 0.0 && strikes > 0 && w.nodes.len() < nodes && worn > 0.0);
        w.reset_world_keep_body().unwrap();
        assert_eq!(w.wear.get("shoulder_r"), Some(&worn), "the reducers keep their wear");
        assert_eq!(w.nodes.len(), nodes, "the tree is back");
        assert!(w.drops.is_empty() && w.agent.carried.is_empty());
        assert_eq!((w.agent.energy_used_j, w.agent.strikes), (energy, strikes), "the body keeps its wear");
        assert_eq!(w.heat.get("elbow_r"), Some(&1.4));
        let camp = w.waypoint_by_key("base_camp").unwrap().clone();
        assert!((w.agent.x - camp.x).abs() < 1e-4);
        add_stock(&mut w.agent.carried, ItemKind::Coal, 1.0);
        w.agent.x += 2.0;
        w.reset_body();
        assert_eq!((w.agent.energy_used_j, w.agent.strikes), (0.0, 0));
        assert!(w.heat.is_empty() && w.wear.is_empty() && w.battery_ratio() == 1.0);
        assert_eq!(w.carried_count(ItemKind::Coal), 1, "the body reset keeps the cargo");
        assert!((w.agent.x - camp.x - 2.0).abs() < 1e-4, "and the place");
    }

    #[test]
    fn a_cloned_world_runs_on_its_own() {
        let mut a = world(FingerCount::Five, BaseKind::Track);
        let mut b = a.clone();
        mine_nearest(&mut b, NodeKind::Tree);
        assert!(b.nodes.len() < a.nodes.len() && b.tick > a.tick);
        a.tick();
        assert_eq!(a.tick, 1);
        assert_eq!(a.capacity_kg(), b.capacity_kg());
    }

    #[test]
    fn a_walking_mid_robot_carries_little_and_a_tracked_one_carries_more() {
        let foot = world_with(BodyPlan::default(), BaseKind::HumanFoot);
        let track = world_with(BodyPlan::default(), BaseKind::Track);
        assert!(foot.capacity_kg() < 6.0, "mid legs hold ~24 kg in total: {}", foot.capacity_kg());
        assert!(track.capacity_kg() > foot.capacity_kg());
        assert!((foot.battery_capacity_j() - 324.0 * 3600.0).abs() < 1.0);
        let mut loaded = world_with(BodyPlan::default(), BaseKind::Track);
        let (v0, p0) = (loaded.drive_spec().max_speed_mps, loaded.drive_spec().power_w);
        add_stock(&mut loaded.agent.carried, ItemKind::IronOre, 3.0);
        assert!(loaded.drive_spec().power_w > p0);
        assert!(loaded.drive_spec().max_speed_mps <= v0);
        let s = loaded.body_status(None, None);
        assert!(s.balance.is_none() || s.balance.as_ref().unwrap().mass_kg > 0.0);
    }
}
