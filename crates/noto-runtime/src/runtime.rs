use std::path::Path;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Result};
use noto_core::{BodyStatus, NotoConfig};
use noto_quest::{Labels, Quest, QuestBook};
use noto_sds::{AuditLog, Level, LogLine, SdsStatus, SdsStore};
use noto_search::semantic::PhraseEmbedder;
use noto_search::{Lexicon, ParsedCommand, Parser, Plan};
use noto_ts::Forecast;
use serde::Serialize;
use serde_json::{json, Value};
use world_sim::{amount_text, encode_webp, Action, Camera, ItemKind, NodeKind, TimedEvent, World, WorldConfig, WorldEvent, WorldSnapshot};

use noto_search::text::eul_reul;
use crate::config::{GateCfg, NoiseCfg, RuntimeConfig};
use crate::driver::Task;
use crate::perception::{OraclePerception, Recognizer, ScanInput};
use crate::perception::EntKind;
use crate::precision::{Judged, MemCtx, PrecisionGates};
use crate::vision::{in_view, memory_ok, DriftRate, VisionMark};
use world_sim::world::DELIVER_REACH_M;
use noto_sds::{Scope, Track};

#[derive(Clone, Debug, Serialize)]
pub struct ScanReport {
    pub seq: u64,
    pub tick: u64,
    pub trigger: String,
    pub mode: String,
    pub device: String,
    pub tiles: usize,
    pub embed_ms: f32,
    pub total_ms: f32,
    pub judged: Vec<Judged>,
    pub scene: Vec<(String, f32)>,
    pub filter: crate::perception::ScanFilter,
    /// The camera the scan looked through (the head camera turned as it was, or the carriage camera).
    pub view: world_sim::render::View,
}

impl ScanReport {
    pub fn accepted(&self) -> impl Iterator<Item = &Judged> {
        self.judged.iter().filter(|j| j.accepted)
    }

    /// Whether the scan's boxes still lie on a frame seen through `view`: the same camera, turned and pitched within
    /// 3° of how it was and moved under 0.3 m. A glance aside, the rider's look down or the carriage camera drew their
    /// boxes on the wrong part of the first-person frame.
    pub fn seen_through(&self, view: &world_sim::render::View) -> bool {
        let (a, b) = (&self.view, view);
        let turn = (a.yaw - b.yaw + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
        let moved = ((a.origin[0] - b.origin[0]).powi(2) + (a.origin[1] - b.origin[1]).powi(2) + (a.origin[2] - b.origin[2]).powi(2)).sqrt();
        turn.abs() <= 3f32.to_radians() && (a.pitch - b.pitch).abs() <= 3f32.to_radians() && moved < 0.3
    }

    pub fn confidence(&self) -> Option<f32> {
        let v: Vec<f32> = self.accepted().map(|j| j.cand.score).collect();
        (!v.is_empty()).then(|| v.iter().sum::<f32>() / v.len() as f32)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct EvalCounters {
    pub scans: u64,
    pub raw_total: u64,
    pub raw_correct: u64,
    pub accepted_total: u64,
    pub accepted_correct: u64,
    pub truths: u64,
    pub truths_found_raw: u64,
    pub truths_found_accepted: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct EvalSummary {
    pub raw_precision: Option<f64>,
    pub raw_recall: Option<f64>,
    pub gated_precision: Option<f64>,
    pub gated_recall: Option<f64>,
}

fn ratio(a: u64, b: u64) -> Option<f64> {
    (b > 0).then(|| a as f64 / b as f64)
}

impl EvalCounters {
    pub fn summary(&self) -> EvalSummary {
        EvalSummary {
            raw_precision: ratio(self.raw_correct, self.raw_total),
            raw_recall: ratio(self.truths_found_raw, self.truths),
            gated_precision: ratio(self.accepted_correct, self.accepted_total),
            gated_recall: ratio(self.truths_found_accepted, self.truths),
        }
    }

    fn add(&mut self, truths: &[String], judged: &[Judged]) {
        self.scans += 1;
        self.truths += truths.len() as u64;
        for j in judged {
            let correct = j.cand.is_correct() == Some(true);
            self.raw_total += 1;
            self.raw_correct += correct as u64;
            if j.accepted {
                self.accepted_total += 1;
                self.accepted_correct += correct as u64;
            }
        }
        for t in truths {
            let hit = |only_accepted: bool| judged.iter().any(|j| j.cand.truth.as_deref() == Some(t) && j.cand.is_correct() == Some(true) && (!only_accepted || j.accepted));
            self.truths_found_raw += hit(false) as u64;
            self.truths_found_accepted += hit(true) as u64;
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct VisionStats {
    pub approach_s: f64,
    pub approach_looks: u32,
    pub looks: u32,
    pub blind_acts: u32,
    pub side_picks: u32,
    pub picks: u32,
    pub memory_aim_m: Vec<f32>,
    pub seen_aim_m: Vec<f32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Submission {
    pub ok: bool,
    pub stopped: bool,
    pub quest_id: Option<String>,
    pub error: Option<String>,
    pub error_code: Option<String>,
    pub mode: String,
    pub parsed: ParsedCommand,
}

#[derive(Clone, Debug, Serialize)]
pub struct TsView {
    pub label: String,
    pub history: Vec<f64>,
    pub forecast: Option<Forecast>,
    pub labels: Vec<(String, usize)>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Detection {
    pub label: String,
    pub label_ko: String,
    pub score: f32,
    pub margin: f32,
    pub accepted: bool,
    pub blocked_by: Option<String>,
    pub reason: String,
    pub bbox: [f32; 4],
    pub landmark: Option<u32>,
}

pub struct DepthView {
    pub tick: u64,
    pub width: u32,
    pub height: u32,
    pub mode: String,
    pub device: String,
    pub infer_ms: f32,
    pub metrics: Option<noto_vision::DepthMetrics>,
    pub scale_m: (f32, f32),
    pub max_depth_m: Option<f32>,
    pub rgb_webp: Vec<u8>,
    pub pred_webp: Vec<u8>,
    pub truth_webp: Vec<u8>,
    pub metrics_raw: Option<noto_vision::DepthMetrics>,
    pub raw_webp: Option<Vec<u8>>,
    pub seg_webp: Option<Vec<u8>>,
    pub normals_webp: Option<Vec<u8>>,
    pub align: Option<noto_vision::AlignFit>,
    pub sky_frac: Option<f32>,
    pub flip_tta: bool,
    /// The right camera's frame and what SGBM made of the pair, when the eye is a stereo pair.
    pub right_webp: Option<Vec<u8>>,
    pub stereo: Option<noto_vision::StereoStats>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FrameOut {
    #[serde(skip)]
    pub webp: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub bytes: usize,
    pub encode_ms: f32,
    pub lossless: bool,
    pub tick: u64,
    pub detections: Vec<Detection>,
    pub overlay: Option<Value>,
    /// Where each robot came out on the picture, on the map view.
    pub marks: Vec<world_sim::render::UnitMark>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameView {
    Agent,
    Top,
    Chase { orbit_deg: i16 },
    Focus(world_sim::render::FocusCam),
    Iso { az_step: u8, zoom_pct: u16, dimetric: bool },
    /// The fork carriage camera of a wheeled or tracked base (the head camera on legs).
    Fork,
    /// The head camera's stereo depth, live, as a walker's pick-up HUD shows it beside the picture (a stereo eye only):
    /// SGBM on the pair at the asked size, with the box where the robot believes its target is (`stereo::eye_depth`).
    EyeDepth,
}

pub(crate) struct WorldLabels<'a>(&'a World);

pub(crate) fn world_labels(w: &World) -> WorldLabels<'_> {
    WorldLabels(w)
}

impl Labels for WorldLabels<'_> {
    fn resource_ko(&self, id: &str) -> String {
        ItemKind::from_id(id).map(|k| k.spec().label_ko.to_string()).unwrap_or_else(|| id.to_string())
    }
    fn place_ko(&self, id: &str) -> String {
        self.0.waypoint_by_key(id).map(|w| w.label_ko.clone()).unwrap_or_else(|| id.to_string())
    }
    fn resource_mined(&self, id: &str) -> bool {
        ItemKind::from_id(id).map_or(true, |k| k.from_node())
    }
}

pub fn label_ko(world: &World, label: &str) -> String {
    if let Some(k) = NodeKind::ALL.into_iter().find(|k| k.id() == label) {
        return k.spec().label_ko.to_string();
    }
    if let Some(k) = ItemKind::from_id(label) {
        return k.spec().label_ko.to_string();
    }
    if let Some(w) = world.waypoint_by_key(label) {
        return w.label_ko.clone();
    }
    label.to_string()
}

pub(crate) const RELOCALIZE_RANGE_M: f32 = 8.0;
const FIX_GATE_CHI2: f32 = 9.21;

/// A sub-worker that put the first worker's pose right: which, by how much it looked off, and how much of it the fix
/// took (the Kalman gain), when.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct PeerFixNote {
    pub peer: u32,
    pub gap_m: f32,
    pub gain: f32,
    pub time_s: f64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PoseFix {
    pub dx: f32,
    pub dz: f32,
    pub odo_m: f32,
    pub prev_odo_m: f32,
}

impl PoseFix {
    pub fn share(&self, odo: f32, rate: DriftRate) -> f32 {
        if odo <= self.prev_odo_m {
            return 0.0;
        }
        let var = |d: f32| (rate.per_m * d).powi(2) + rate.walk_m * rate.walk_m * d;
        (var(odo.min(self.odo_m) - self.prev_odo_m) / var(self.odo_m - self.prev_odo_m).max(1e-12)).clamp(0.0, 1.0)
    }

    pub fn shift(&self, x: f32, z: f32, odo: f32, rate: DriftRate) -> (f32, f32) {
        let k = self.share(odo, rate);
        (x + k * self.dx, z + k * self.dz)
    }
}

fn new_world(cfg: &RuntimeConfig) -> Result<World> {
    World::new(WorldConfig { seed: cfg.seed, cells: cfg.cells, noto: cfg.noto, body: cfg.body, terrain: cfg.terrain, odometry: cfg.odometry, parcels: cfg.parcels, lighting: cfg.lighting, site: cfg.site }).map_err(|e| anyhow!("월드를 만들 수 없습니다: {e}"))
}

pub struct Runtime {
    pub cfg: RuntimeConfig,
    pub(crate) nn_live: crate::nnlive::NnLive,
    pub(crate) world: World,
    pub(crate) cam: Camera,
    parser: Parser,
    embedder: Option<Arc<dyn PhraseEmbedder + Send + Sync>>,
    pub(crate) recognizer: Box<dyn Recognizer>,
    pub(crate) gates: PrecisionGates,
    pub(crate) sds: SdsStore,
    pub(crate) audit: AuditLog,
    pub(crate) quests: QuestBook,
    pub(crate) task: Option<Task>,
    pub(crate) last_scan: Option<ScanReport>,
    /// How far the odometry had run at the last look at the ground ahead (m).
    pub(crate) step_odo_m: f32,
    pub(crate) gaze_check: Option<crate::vision::GazeCheck>,
    /// A lift truck's rider and carriage camera on the item it is picking up, each checked against the other.
    pub(crate) fork_cross: Option<crate::vision::CrossCheck>,
    pub(crate) pending_fix: Option<PoseFix>,
    /// The last time a sub-worker in view put this robot's pose right (`relocalize_from_peers`), for the screen.
    pub(crate) last_peer_fix: Option<PeerFixNote>,
    scan_seq: u64,
    last_device: String,
    pub(crate) eval: EvalCounters,
    last_flush_tick: u64,
    pub(crate) focus: Option<String>,
    pub(crate) db: noto_sds::TestDb,
    pub(crate) pass: String,
    pub(crate) run_id: String,
    pub(crate) session_id: String,
    pub(crate) test_seq: u32,
    pub(crate) recording: Option<crate::record::Recording>,
    pub(crate) ml: crate::ml::MlPipeline,
    pub(crate) ml_task: Option<crate::mldrive::MlTask>,
    /// The robots saved from the CAD editor.
    pub(crate) rigs: crate::rigs::RigShelf,
    pub(crate) test: Option<crate::testrun::TestSession>,
    pub test_cfg: crate::testrun::TestCfg,
    pub(crate) live_queue: std::collections::VecDeque<String>,
    pub(crate) vstats: VisionStats,
    pub(crate) iso: crate::views::IsoFollow,
    /// Sub-workers' lanes and the team flow (`crew`).
    pub(crate) crew: crate::crew::Crew,
}

impl Runtime {
    pub fn new(cfg: RuntimeConfig, lexicon_json: &str, data_dir: Option<&Path>) -> Result<Runtime> {
        let lex = Lexicon::from_json(lexicon_json)?;
        let recognizer: Box<dyn Recognizer> = Box::new(OraclePerception::new(&lex, cfg.noise.clone(), cfg.min_pixels));
        let mut ml = crate::ml::MlPipeline::stand_in(&lex, cfg.seed);
        ml.prompting = noto_search::VisionPrompting::from_lexicon_json(lexicon_json)?;
        ml.fix = crate::ml::MlFixCfg::v1(&ml.prompting);
        let parser = Parser::new(lex);
        let world = new_world(&cfg)?;
        let ms = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0);
        let (sds, audit, db) = match data_dir {
            Some(dir) => (SdsStore::open_with(&dir.join("sds"), noto_sds::OpenOptions::stamped(noto_core::bench::ENGINE_REV)), AuditLog::open(&dir.join("logs"), &format!("session_{ms}"), 4000)?, noto_sds::TestDb::open(&dir.join("db"))),
            None => (SdsStore::memory(), AuditLog::memory(4000), noto_sds::TestDb::memory()),
        };
        let cam = cfg.camera.camera();
        let gates = PrecisionGates::new(cfg.gates.clone());
        let mut rt = Runtime {
            cfg,
            world,
            cam,
            parser,
            embedder: None,
            recognizer,
            gates,
            sds,
            audit,
            quests: QuestBook::new(),
            task: None,
            last_scan: None,
            step_odo_m: 0.0,
            gaze_check: None,
            fork_cross: None,
            pending_fix: None,
            last_peer_fix: None,
            scan_seq: 0,
            last_device: String::new(),
            nn_live: crate::nnlive::NnLive::default(),
            eval: EvalCounters::default(),
            last_flush_tick: 0,
            focus: None,
            db,
            pass: "game".into(),
            run_id: format!("s{ms}-live"),
            session_id: format!("s{ms}"),
            test_seq: 0,
            recording: None,
            ml,
            ml_task: None,
            rigs: crate::rigs::RigShelf::open(data_dir.map(|d| d.join("rigs"))),
            test: None,
            test_cfg: crate::testrun::TestCfg::default(),
            live_queue: std::collections::VecDeque::new(),
            vstats: VisionStats::default(),
            iso: crate::views::IsoFollow::default(),
            crew: crate::crew::Crew::default(),
        };
        let msg = format!(
            "월드 시작: seed {} · {}×{}칸 · 베이스 {:?} · 손가락 {} · 인식기 {}",
            rt.cfg.seed,
            rt.world.terrain.w,
            rt.world.terrain.d,
            rt.world.hand.base.kind,
            u8::from(rt.world.hand.config.fingers),
            rt.recognizer.mode()
        );
        rt.log("world", Level::Info, msg, Some(json!({"seed": rt.cfg.seed, "cells": rt.cfg.cells})));
        rt.relay_all_grips();
        Ok(rt)
    }

    pub(crate) fn log(&mut self, cat: &str, level: Level, msg: impl Into<String>, data: Option<Value>) {
        self.audit.log(cat, level, msg, data.map(tidy));
    }

    pub fn note_device(&mut self, msg: &str) {
        self.log("device", Level::Warn, msg, None);
    }

    pub(crate) fn log_world_events(&mut self, events: &[TimedEvent]) {
        for te in events {
            let physics = matches!(te.event, WorldEvent::Bump { .. } | WorldEvent::CargoSlip { .. } | WorldEvent::CargoLost { .. } | WorldEvent::StepSeen { .. } | WorldEvent::StepTaken { .. });
            let (level, msg) = match &te.event {
                WorldEvent::ActionStarted { action } => (Level::Debug, format!("동작 시작 {action}")),
                WorldEvent::ActionDone { action, ok, reason } => (
                    if *ok { Level::Debug } else { Level::Warn },
                    format!("동작 {} {action}{}", if *ok { "완료" } else { "실패" }, reason.as_ref().map(|r| format!(" ({r})")).unwrap_or_default()),
                ),
                WorldEvent::CurbClimbed { height_m } => (Level::Info, format!("연석 {:.0} cm를 넘음", height_m * 100.0)),
                WorldEvent::NodeBroken { node, kind, drops } => {
                    (Level::Info, format!("{}({node}){} 캠 → 아이템 {}개 떨어짐", kind.spec().label_ko, eul_reul(kind.spec().label_ko), drops.len()))
                }
                WorldEvent::ItemPicked { drop, item, mass_kg, grip_margin, amount } => {
                    (Level::Info, format!("{} {drop} 줍기 ({}개, {mass_kg:.1} kg, 마찰 여유 {grip_margin:.2})", item.spec().label_ko, amount_text(*amount)))
                }
                WorldEvent::Delivered { waypoint, item, amount } => (Level::Info, format!("{waypoint}에 {} {}개 인도", item.spec().label_ko, amount_text(*amount))),
                WorldEvent::ItemDropped { item, amount, waypoint, drops } => (
                    Level::Info,
                    match waypoint {
                        Some(w) => format!("{} {}개를 {w} 더미에 내려놓음", item.spec().label_ko, amount_text(*amount)),
                        None => format!("{} {}개를 땅에 내려놓음 ({})", item.spec().label_ko, amount_text(*amount), drops.join(", ")),
                    },
                ),
                WorldEvent::ItemSplit { drop, piece, item, amount, left } => {
                    (Level::Info, format!("{} {drop} 쪼갬 → 조각 {piece} {}개 + 남은 {}개", item.spec().label_ko, amount_text(*amount), amount_text(*left)))
                }
                WorldEvent::BatteryDepleted => (Level::Error, "배터리 방전".to_string()),
                WorldEvent::Act { act, kind, label, ok, .. } => (if *ok { Level::Debug } else { Level::Warn }, format!("[{act}:{kind}] {label}")),
                WorldEvent::Crew { worker, label, ok, .. } => (if *ok { Level::Info } else { Level::Warn }, format!("[작업자 {worker}] {label}")),
                WorldEvent::Rider { worker, label, .. } => (Level::Info, format!("[작업자 {worker}] {label}")),
                WorldEvent::Hold { worker, label, .. } => (Level::Info, format!("[작업자 {worker}] {label}")),
                WorldEvent::Bump { worker, label, fell, .. } => (if *fell { Level::Error } else { Level::Warn }, format!("[작업자 {worker}] {label}")),
                WorldEvent::CargoSlip { worker, label, .. } => (Level::Warn, format!("[작업자 {worker}] {label}")),
                WorldEvent::CargoLost { worker, label, .. } => (Level::Error, format!("[작업자 {worker}] {label}")),
                WorldEvent::StepSeen { worker, label, .. } | WorldEvent::StepTaken { worker, label, .. } => (Level::Info, format!("[작업자 {worker}] {label}")),
            };
            self.audit.log(if physics { "physics" } else { "world" }, level, msg, Some(tidy(json!({"tick": te.tick, "time_s": te.time_s, "event": te.event}))));
        }
    }

    pub(crate) fn flush_sds(&mut self) {
        self.last_flush_tick = self.world.tick;
        for l in self.sds.flush() {
            self.audit.info("sds", l);
        }
    }

    pub fn world(&self) -> &World {
        &self.world
    }

    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }

    pub fn camera(&self) -> &Camera {
        &self.cam
    }

    pub fn gates(&self) -> &PrecisionGates {
        &self.gates
    }

    pub fn sds(&self) -> &SdsStore {
        &self.sds
    }

    pub fn sds_mut(&mut self) -> &mut SdsStore {
        &mut self.sds
    }

    pub fn audit(&self) -> &AuditLog {
        &self.audit
    }

    pub fn quest_book(&self) -> &QuestBook {
        &self.quests
    }

    pub fn quests(&self) -> &[Quest] {
        self.quests.all()
    }

    pub fn eval(&self) -> &EvalCounters {
        &self.eval
    }

    pub fn vision_stats(&self) -> &VisionStats {
        &self.vstats
    }

    pub fn last_scan(&self) -> Option<&ScanReport> {
        self.last_scan.as_ref()
    }

    pub fn parser(&self) -> &Parser {
        &self.parser
    }

    pub fn has_embedder(&self) -> bool {
        self.embedder.is_some()
    }

    pub fn recognizer_mode(&self) -> &str {
        self.recognizer.mode()
    }

    pub fn stand_ins(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if self.recognizer.mode() == "oracle" {
            out.push("recognizer");
        }
        if matches!(self.ml.locate, crate::ml::LocateEngine::StandIn(_)) {
            out.push("locate");
        }
        if matches!(self.ml.depth, crate::ml::DepthEngine::StandIn(_)) {
            out.push("depth");
        }
        if self.ml.tracker.mode() == "region-grow" {
            out.push("track");
        }
        if !self.has_ttm() {
            out.push("ttm");
        }
        out
    }

    pub fn busy(&self) -> bool {
        self.task.is_some() || self.ml_task.is_some() || self.quests.current_id().is_some() || self.crew_busy()
    }

    /// A sub-worker's quest or a team's flow is still running.
    pub fn crew_busy(&self) -> bool {
        self.crew.team.as_ref().is_some_and(|t| !t.done) || self.quests.all().iter().any(|q| q.owner != noto_quest::FIRST_WORKER && !q.is_finished())
    }

    pub fn ml_pipeline(&self) -> &crate::ml::MlPipeline {
        &self.ml
    }

    pub fn ml_pipeline_mut(&mut self) -> &mut crate::ml::MlPipeline {
        &mut self.ml
    }

    pub fn test_db(&self) -> &noto_sds::TestDb {
        &self.db
    }

    pub fn test_db_mut(&mut self) -> &mut noto_sds::TestDb {
        &mut self.db
    }

    pub fn pass(&self) -> &str {
        &self.pass
    }

    pub fn use_tips(&mut self, dir: &Path, device: &candle_core::Device) -> Result<()> {
        if self.stereo() {
            anyhow::bail!("카메라 2대(스테레오)에서는 깊이를 SGBM이 재므로 TIPSv2를 불러오지 않습니다. 단안으로 쓰려면 카메라를 1대로 바꾸세요");
        }
        let t0 = Instant::now();
        let m = noto_vision::TipsDepth::load(dir, device)?;
        self.ml.depth = crate::ml::DepthEngine::Model(Box::new(m));
        self.ml.sync_pyramid();
        let ms = t0.elapsed().as_secs_f32() * 1000.0;
        self.log("embed", Level::Info, format!("TIPSv2 DPT 깊이 모델 로드 ({ms:.0} ms) · {} · {}", noto_ml::device_label(device), dir.display()), None);
        Ok(())
    }

    pub fn use_sam(&mut self, dir: &Path, device: &candle_core::Device) -> Result<()> {
        let t0 = Instant::now();
        let m = noto_vision::SamModel::load(dir, device)?;
        let mode = m.variant().mode();
        self.ml.tracker = Box::new(noto_vision::SamTracker::new(Arc::new(m)));
        self.ml.sync_pyramid();
        let ms = t0.elapsed().as_secs_f32() * 1000.0;
        self.log("embed", Level::Info, format!("{mode} 마스크 모델 로드 ({ms:.0} ms) · {} · {}", noto_ml::device_label(device), dir.display()), None);
        Ok(())
    }

    pub fn use_depth_stand_in(&mut self) {
        if self.stereo() {
            // Two cameras measure depth themselves: there is nothing to stand in for.
            self.use_stereo();
            return;
        }
        self.ml.depth = crate::ml::DepthEngine::StandIn(crate::ml::DepthStandIn::new(self.cfg.seed));
        self.ml.sync_pyramid();
        self.log("embed", Level::Info, "깊이: 대역 모델(stand-in)", None);
    }

    /// Cameras per eye: 1 (monocular, TIPSv2) or 2 (stereo pair, SGBM).
    pub fn cameras(&self) -> u8 {
        self.cfg.camera.cameras.clamp(1, 2)
    }

    pub fn stereo(&self) -> bool {
        self.cameras() == 2
    }

    /// The baseline the renderer puts the right camera at, when the eye is a stereo pair.
    pub(crate) fn stereo_baseline(&self) -> Option<f32> {
        self.stereo().then(|| world_sim::world::camera_of(self.world.body.head.sensor).1)
    }

    /// Puts SGBM in the depth slot, on the rig of the head sensor the body has now. A monocular model that was there
    /// (TIPSv2) is dropped, which frees its memory: two cameras do not load or run it.
    pub fn use_stereo(&mut self) {
        let rig = crate::stereo::rig_for(&self.cam, self.world.body.head.sensor);
        let had_model = matches!(&self.ml.depth, crate::ml::DepthEngine::Model(m) if m.stereo_rig().is_none());
        self.ml.depth = crate::ml::DepthEngine::Model(Box::new(noto_vision::SgbmDepth::new(rig, noto_vision::SgbmCfg::default())));
        self.ml.sync_pyramid();
        let (gw, gh) = noto_vision::stereo_grid(self.cam.width as u32, self.cam.height as u32, self.ml.fix.pyramid.as_ref(), noto_vision::SgbmCfg::default().per_l2);
        let msg = format!(
            "깊이: SGBM 스테레오 (카메라 2대 · 기선 {:.1} cm · 초점 {:.0} px · 매칭 격자 {gw}×{gh}){}",
            rig.baseline_m * 100.0,
            rig.focal_px,
            if had_model { " — TIPSv2를 내려놓음" } else { "" }
        );
        self.log("embed", Level::Info, msg, Some(json!({"baseline_m": rig.baseline_m, "focal_px": rig.focal_px, "grid": [gw, gh], "subpixel_px": rig.subpixel_px})));
    }

    /// Switches the eye between one camera and a stereo pair. Going to two puts SGBM in the depth slot at once; going
    /// back to one leaves the stand-in there until TIPSv2 is loaded again (the app's model step does that).
    pub fn set_cameras(&mut self, n: u8) -> Result<()> {
        if !(1..=2).contains(&n) {
            anyhow::bail!("카메라는 1대(단안) 또는 2대(스테레오)입니다: {n}");
        }
        if n == self.cameras() {
            return Ok(());
        }
        self.cfg.camera.cameras = n;
        if n == 2 {
            self.use_stereo();
        } else {
            self.ml.depth = crate::ml::DepthEngine::StandIn(crate::ml::DepthStandIn::new(self.cfg.seed));
            self.ml.sync_pyramid();
            self.log("embed", Level::Info, "깊이: 카메라 1대(단안) — TIPSv2를 불러와야 합니다 (그 전에는 대역 모델)", None);
        }
        Ok(())
    }

    /// The camera `MemCtx::sigma_obs` hands to `stereo_depth_sigma`: the SGBM rig on its matching grid when the eye
    /// is a stereo pair, else the head sensor's own.
    pub(crate) fn sigma_camera(&self) -> (f32, f32, f32) {
        match self.ml.stereo_rig() {
            Some(rig) if self.stereo() => crate::stereo::sigma_camera(&rig, &self.cam, &noto_vision::SgbmCfg::default()),
            _ => world_sim::world::camera_of(self.world.body.head.sensor),
        }
    }

    pub fn use_region_tracker(&mut self) {
        self.ml.tracker = Box::new(noto_vision::RegionTracker::default());
        self.ml.sync_pyramid();
        self.log("embed", Level::Info, "마스크 추적: 영역 확장(region-grow)", None);
    }

    pub fn set_gates(&mut self, gates: GateCfg) {
        self.cfg.gates = gates.clone();
        self.gates.set_cfg(gates);
        self.log("vision", Level::Info, "정밀도 게이트 설정이 바뀜", Some(serde_json::to_value(&self.cfg.gates).unwrap_or(Value::Null)));
    }

    pub fn set_noise(&mut self, noise: NoiseCfg) {
        self.recognizer.set_noise(&noise);
        self.cfg.noise = noise;
        self.log("vision", Level::Info, "모의 인식기 오류 설정이 바뀜", Some(serde_json::to_value(&self.cfg.noise).unwrap_or(Value::Null)));
    }

    pub fn set_recognizer(&mut self, r: Box<dyn Recognizer>) {
        let mode = r.mode().to_string();
        let scope = r.scope();
        self.recognizer = r;
        self.eval = EvalCounters::default();
        self.gates.clear();
        self.gates.set_recognizer_mode(&scope);
        self.log("vision", Level::Info, format!("인식기 교체: {mode}"), None);
    }

    pub fn use_siglip(&mut self, model: Arc<noto_ml::Siglip2>, bank_dir: Option<&Path>) -> Result<()> {
        let t0 = Instant::now();
        let phrases = Arc::new(crate::embedding::BankedPhrases::new(Arc::clone(&model), bank_dir));
        self.ml.locate = crate::ml::LocateEngine::Model(Box::new(noto_vision::SiglipLocator::with_cfg(Arc::clone(&model), self.ml.fix.locate.clone())));
        self.ml.sync_pyramid();
        let prompting = self.ml.fix.scan_prompting.then(|| self.ml.prompting.clone());
        let mut rec = crate::embedding::EmbeddingPerception::siglip_with(model, self.parser.lexicon(), bank_dir, Default::default(), prompting.as_ref())?;
        if self.ml.fix.scan_calibrated {
            rec.set_calibration(self.ml.calib.iter().find(|t| t.mode == crate::embedding::SCAN_CALIB_MODE).cloned());
        }
        self.attach_embedder(phrases)?;
        self.set_recognizer(Box::new(rec));
        let ms = t0.elapsed().as_secs_f32() * 1000.0;
        self.log("embed", Level::Info, format!("SigLIP2 인식기 준비 ({ms:.0} ms)"), Some(json!({"ms": ms})));
        Ok(())
    }

    pub fn attach_ttm(&mut self, ttm: Option<Arc<noto_ts::ttm::TtmForecaster>>) {
        let msg = match &ttm {
            Some(t) => format!("Granite TTM 연결: 문맥 {} · 예측 {} 스텝", t.context_length(), t.prediction_length()),
            None => "Granite TTM 해제: last/ewma/holt 융합만 사용".to_string(),
        };
        self.gates.set_ttm(ttm);
        self.log("ts", Level::Info, msg, None);
    }

    pub fn has_ttm(&self) -> bool {
        self.gates.has_ttm()
    }

    pub fn load_ttm(&mut self, dir: &Path, device: &candle_core::Device) -> Result<()> {
        let t0 = Instant::now();
        let ttm = noto_ts::ttm::TtmForecaster::load(dir, device).map_err(|e| anyhow!("Granite TTM 을 {} 에서 읽지 못했습니다: {e}", dir.display()))?;
        self.attach_ttm(Some(Arc::new(ttm)));
        let ms = t0.elapsed().as_secs_f32() * 1000.0;
        self.log("ts", Level::Info, format!("Granite TTM 로드 완료 ({ms:.0} ms) · {}", dir.display()), Some(json!({"ms": ms})));
        Ok(())
    }

    pub fn use_oracle(&mut self) {
        let r = OraclePerception::new(self.parser.lexicon(), self.cfg.noise.clone(), self.cfg.min_pixels);
        self.set_recognizer(Box::new(r));
    }

    pub fn use_honest_stand_ins(&mut self) {
        let seed = self.cfg.seed;
        self.ml.use_honest_stand_ins(seed);
        if self.recognizer.mode() == "oracle" {
            let mut r = OraclePerception::new(self.parser.lexicon(), self.cfg.noise.clone(), self.cfg.min_pixels);
            r.set_honest(Some(crate::perception::ScanErrModel::SIGLIP_MEASURED));
            self.set_recognizer(Box::new(r));
        }
    }

    pub fn attach_embedder(&mut self, emb: Arc<dyn PhraseEmbedder + Send + Sync>) -> Result<()> {
        let t0 = Instant::now();
        self.parser.attach_embedder(&*emb)?;
        self.embedder = Some(emb);
        let ms = t0.elapsed().as_secs_f32() * 1000.0;
        self.log("embed", Level::Info, format!("명령어 사전 임베딩 완료 ({ms:.0} ms)"), Some(json!({"ms": ms})));
        Ok(())
    }

    pub fn reset(&mut self, seed: Option<u64>) -> Result<()> {
        self.reset_with(seed, None, None)
    }

    pub fn reset_with(&mut self, seed: Option<u64>, cells: Option<usize>, hand: Option<NotoConfig>) -> Result<()> {
        let mut next = self.cfg.clone();
        if let Some(s) = seed {
            next.seed = s;
        }
        if let Some(c) = cells {
            next.cells = c;
        }
        if let Some(h) = hand {
            next.noto = h;
        }
        let world = new_world(&next)?;
        self.cfg = next;
        self.world = world;
        self.crew = crate::crew::Crew { mode: self.crew.mode, ..crate::crew::Crew::default() };
        self.task = None;
        self.ml_task = None;
        self.quests.clear();
        self.gates.clear();
        self.last_scan = None;
        self.gaze_check = None;
        self.fork_cross = None;
        self.focus = None;
        self.eval = EvalCounters::default();
        self.flush_sds();
        // What the hands learned of the items stays with them in a new world.
        self.relay_all_grips();
        if self.stereo() {
            // The head preset (its sensor's baseline) may have changed with the body.
            self.use_stereo();
        }
        let msg = format!(
            "월드 다시 만듦: seed {} · {}×{}칸 · 베이스 {:?} · 손가락 {}",
            self.cfg.seed,
            self.world.terrain.w,
            self.world.terrain.d,
            self.world.hand.base.kind,
            u8::from(self.world.hand.config.fingers)
        );
        self.log("world", Level::Info, msg, Some(json!({"seed": self.cfg.seed, "cells": self.cfg.cells})));
        Ok(())
    }

    pub fn reset_world(&mut self) -> Result<()> {
        self.cancel();
        self.world.reset_world_keep_body().map_err(|e| anyhow!("월드를 되돌릴 수 없습니다: {e}"))?;
        let mode = self.crew.mode;
        self.crew = crate::crew::Crew { mode, ..crate::crew::Crew::default() };
        for w in &self.world.crew {
            self.crew.lanes.insert(w.id, crate::crew::Lane::idle(w.id, w.form));
        }
        self.task = None;
        self.ml_task = None;
        self.quests.clear();
        self.gates.clear();
        self.last_scan = None;
        self.gaze_check = None;
        self.fork_cross = None;
        self.focus = None;
        self.eval = EvalCounters::default();
        self.flush_sds();
        let msg = format!("월드 리셋: seed {} 처음 상태로 (신체 상태는 유지 · 배터리 {:.0}%)", self.cfg.seed, self.world.battery_ratio() * 100.0);
        self.log("world", Level::Info, msg, Some(json!({"seed": self.cfg.seed, "kept_energy_j": self.world.agent.energy_used_j})));
        Ok(())
    }

    pub fn reset_body(&mut self) {
        let (battery, strikes) = (self.world.battery_ratio(), self.world.agent.strikes);
        let worn = self.world.most_worn().map(|(j, d)| format!(", 가장 닳은 관절 {j} 수명 {:.2}% → 0", d * 100.0)).unwrap_or_default();
        self.world.reset_body();
        self.log("world", Level::Info, format!("신체 상태 리셋: 배터리 {:.0}% → 100%, 채취 타격 {strikes} → 0, 관절 발열 초기화{worn}", battery * 100.0), None);
    }

    pub fn submit_plan(&mut self, plan: Plan, what: &str) -> std::result::Result<String, String> {
        let tick = self.world.tick;
        match self.quests.add(&plan, &WorldLabels(&self.world), tick) {
            Ok(id) => {
                let title = self.quests.get(&id).map(|q| q.title.clone()).unwrap_or_default();
                self.log("quest", Level::Info, format!("퀘스트 {id} 접수 ({what}): {title}"), Some(json!({"quest": id})));
                Ok(id)
            }
            Err(e) => Err(e.to_string()),
        }
    }

    pub fn tick(&mut self) -> Vec<TimedEvent> {
        let events = self.world.tick();
        self.log_world_events(&events);
        self.drive(&events);
        self.crew_tick(&events);
        self.record_balance_events();
        self.record_tick();
        self.record_act_keyframes(&events);
        self.record_act_sds(&events);
        self.record_bench_sds(&events);
        if self.world.tick.saturating_sub(self.last_flush_tick) >= 600 {
            self.flush_sds();
        }
        events
    }

    pub fn step(&mut self, n: u32) -> Vec<LogLine> {
        let before = self.audit.last_seq();
        for _ in 0..n {
            self.tick();
        }
        self.audit.tail(before, 500, None).0
    }

    pub fn run_until_idle(&mut self, max_ticks: u64) -> u64 {
        let mut n = 0;
        while n < max_ticks && self.busy() {
            self.tick();
            n += 1;
        }
        n
    }

    pub fn submit(&mut self, text: &str) -> Submission {
        let emb = self.embedder.clone();
        let parsed = self.parser.parse(text, emb.as_deref().map(|e| e as &dyn PhraseEmbedder));
        let mode = if emb.is_some() { "lexical+embedding" } else { "lexical" }.to_string();
        let plan_json = parsed.plan.as_ref().and_then(|p| serde_json::to_value(p).ok());
        self.log("search", Level::Info, format!("명령: \"{text}\" → {}", plan_brief(&parsed)), Some(json!({"text": text, "plan": plan_json, "mode": mode})));
        for tl in &parsed.trace {
            self.log("search", Level::Debug, format!("{}: {}", tl.stage, tl.detail), None);
        }
        for w in &parsed.warnings {
            self.log("search", Level::Warn, w.clone(), None);
        }
        if let Some(Plan::Fetch { targets, .. }) | Some(Plan::Collect { targets }) = &parsed.plan {
            for t in targets {
                let scope = Scope::new(Track::Search, &t.resource, &format!("{:?}", t.via));
                self.sds.record_baseline(&scope, "score", t.score as f64);
            }
        }
        let mut out = Submission { ok: false, stopped: false, quest_id: None, error: None, error_code: None, mode, parsed };
        match out.parsed.plan.clone() {
            None => {
                let (code, msg) = out.parsed.problem.as_ref().map(|p| (p.code.clone(), p.message.clone())).unwrap_or(("unknown".into(), "명령을 이해하지 못했습니다".into()));
                self.log("search", Level::Warn, format!("해석 실패 [{code}]: {msg}"), None);
                out.error = Some(msg);
                out.error_code = Some(code);
            }
            Some(Plan::Stop) => {
                self.cancel();
                out.ok = true;
                out.stopped = true;
            }
            Some(plan) => {
                let tick = self.world.tick;
                // With sub-workers the command runs over the crew, split (개별) or as a team (협업).
                let added = match self.crew_submit(&plan) {
                    Some(r) => r,
                    None => self.quests.add(&plan, &WorldLabels(&self.world), tick).map_err(|e| e.to_string()),
                };
                match added {
                    Ok(id) => {
                        let title = self.quests.get(&id).map(|q| q.title.clone()).unwrap_or_default();
                        self.log("quest", Level::Info, format!("퀘스트 {id} 접수: {title}"), Some(json!({"quest": id})));
                        out.ok = true;
                        out.quest_id = Some(id);
                    }
                    Err(e) => {
                        out.error = Some(e);
                        out.error_code = Some("quest".into());
                    }
                }
            }
        }
        out
    }

    pub fn submit_preview(&self, text: &str) -> Submission {
        let emb = self.embedder.clone();
        let parsed = self.parser.parse(text, emb.as_deref().map(|e| e as &dyn PhraseEmbedder));
        let mode = if emb.is_some() { "lexical+embedding" } else { "lexical" }.to_string();
        let (error, error_code) = match (&parsed.plan, &parsed.problem) {
            (None, Some(p)) => (Some(p.message.clone()), Some(p.code.clone())),
            _ => (None, None),
        };
        Submission { ok: parsed.plan.is_some(), stopped: false, quest_id: None, error, error_code, mode, parsed }
    }

    pub fn scan_now(&mut self, trigger: &str) -> Result<&ScanReport> {
        let t0 = Instant::now();
        let frame = self.world.render(&self.cam);
        self.scan_frame(trigger, &frame, t0)
    }

    /// `scan_now` on `frame`, the current camera's frame the caller rendered at `t0` and may use again.
    pub(crate) fn scan_frame(&mut self, trigger: &str, frame: &world_sim::Frame, t0: Instant) -> Result<&ScanReport> {
        let view = self.world.head_view(&self.cam).view;
        let mut out = self.recognizer.scan(ScanInput { world: &self.world, frame, camera: &self.cam })?;
        let tick = self.world.tick;
        let oracle = self.recognizer.mode() == "oracle";
        let truths: Vec<String> = if oracle {
            self.world.visible_objects(frame).into_iter().filter(|v| v.pixels >= self.cfg.min_pixels).map(|v| v.entity).collect()
        } else {
            Vec::new()
        };
        let mut cands = std::mem::take(&mut out.candidates);
        for c in cands.iter_mut() {
            c.world = c.world.map(|(x, z)| self.world.belief_of(x, z));
        }
        let mem = self.mem_ctx();
        let judged = self.gates.judge_with(cands, tick, &mut self.sds, &mem);
        self.relocalize_from(&judged);
        self.relocalize_from_peers();
        if oracle {
            self.eval.add(&truths, &judged);
        }
        self.note_look(trigger, String::new(), None, None, None);
        self.scan_seq += 1;
        let report = ScanReport {
            seq: self.scan_seq,
            tick,
            trigger: trigger.to_string(),
            mode: self.recognizer.mode().to_string(),
            device: out.device.clone(),
            tiles: out.tiles,
            embed_ms: out.embed_ms,
            total_ms: t0.elapsed().as_secs_f32() * 1000.0,
            judged,
            scene: out.scene,
            filter: out.filter,
            view,
        };
        self.note_scan(&report);
        self.last_scan = Some(report);
        Ok(self.last_scan.as_ref().expect("just stored"))
    }

    pub(crate) fn me(&self) -> (f32, f32, f32) {
        self.world.believed_pose()
    }

    pub(crate) fn dist_me(&self, x: f32, z: f32) -> f32 {
        let (mx, mz, _) = self.me();
        (x - mx).hypot(z - mz)
    }

    pub(crate) fn base_id(&self) -> String {
        format!("{:?}", self.world.hand.base.kind).to_lowercase()
    }

    pub(crate) fn drift_rate(&self) -> DriftRate {
        DriftRate::learned(&self.sds, &self.base_id(), self.world.body.legged())
    }

    pub(crate) fn pose_sigma_m(&self) -> f32 {
        let r = self.drift_rate();
        let d = self.world.odom.since_fix_m;
        ((r.per_m * d).powi(2) + r.walk_m * r.walk_m * d).sqrt()
    }

    pub(crate) fn mem_ctx(&self) -> MemCtx {
        let base = self.base_id();
        let obs_key = if self.stereo() { format!("{base}|stereo") } else { base.clone() };
        let floor = self.sds.adaptive_baseline(&Scope::new(Track::Memory, &obs_key, "obs"), "scatter_var_m2").map_or(0.0, |b| (b.0 as f32).max(0.0).sqrt());
        MemCtx { odo_m: self.world.odom.travelled_m, rate: self.drift_rate(), base, camera: self.sigma_camera(), obs_floor_m: floor, obs_key }
    }

    fn relocalize_from(&mut self, judged: &[Judged]) {
        if self.pending_fix.is_some() {
            return;
        }
        let mem = self.mem_ctx();
        let p_pose = self.pose_sigma_m().powi(2);
        let mut best: Option<(f32, f32, f32, String, f32)> = None;
        for j in judged.iter().filter(|j| j.accepted && j.cand.kind == EntKind::Waypoint && j.cand.depth_m <= RELOCALIZE_RANGE_M) {
            let (Some((sx, sz)), Some(w)) = (j.cand.world, self.world.waypoint_by_key(&j.cand.link)) else { continue };
            let (ex, ez) = (w.x - sx, w.z - sz);
            let p_obs = mem.sigma_obs(j.cand.depth_m).powi(2);
            if ex * ex + ez * ez > FIX_GATE_CHI2 * (p_pose + p_obs) {
                let msg = format!("위치 보정 안 함: 지점 깃발 {}이 지도와 {:.2} m 달라 보임 — 예상 오차 밖 (잘못 본 깃발)", j.cand.label, ex.hypot(ez));
                self.log("vision", Level::Debug, msg, None);
                continue;
            }
            if best.as_ref().map_or(true, |b| p_obs < b.0) {
                best = Some((p_obs, ex, ez, j.cand.label.clone(), 0.0));
            }
        }
        let Some((p_obs, ex, ez, label, _)) = best else { return };
        let k = p_pose / (p_pose + p_obs).max(1e-9);
        if k * ex.hypot(ez) > 1e-3 {
            let odo = self.world.odom.travelled_m;
            self.pending_fix = Some(PoseFix { dx: k * ex, dz: k * ez, odo_m: odo, prev_odo_m: odo - self.world.odom.since_fix_m });
            let msg = format!("위치 보정: 지점 깃발 {label}이 지도와 {:.2} m 달라 보여 {:.0}%만큼 고침", ex.hypot(ez), k * 100.0);
            self.log("vision", Level::Debug, msg, None);
        }
    }

    /// After the flags, the robots: a sub-worker in the head camera's view is a landmark that moves
    /// (`World::peers_in_view`). The eye puts it where the look says — the camera's range law for its error, as a
    /// flag's — and it says where it is and how sure it is; the gap between the two is this robot's odometry off by as
    /// much as the three σ allow, gated and weighed as a flag is. The best-placed of them corrects the pose; a flag seen
    /// in the same scan goes first. The recognizer does not name robots: the peer is taken as told by its tag and its
    /// word over the radio [추정].
    pub(crate) fn relocalize_from_peers(&mut self) {
        if self.pending_fix.is_some() {
            return;
        }
        let mem = self.mem_ctx();
        let p_pose = self.pose_sigma_m().powi(2);
        let (bx, bz, byaw) = self.me();
        let (s, c) = byaw.sin_cos();
        let tick = self.world.tick;
        let mut best: Option<(f32, f32, f32, u32)> = None;
        for p in self.world.peers_in_view(&self.cam) {
            let p_obs = mem.sigma_obs(p.range_m).powi(2);
            let n = |axis: i32| {
                let h = world_sim::rng::hash2(self.cfg.seed ^ 0x9EE7 ^ ((p.id as u64) << 20), tick as i32, axis);
                ((h & 0xFFFF) as f32 / 65535.0 - 0.5) * 12f32.sqrt() * (0.5 * p_obs).sqrt()
            };
            let (f, l) = (p.rel_m.0 + n(1), p.rel_m.1 + n(2));
            // From the body back onto the map by the pose it believes: forward along its yaw, left at yaw − 90°.
            let seen = (bx + f * c + l * s, bz + f * s - l * c);
            let (ex, ez) = (p.said.0 - seen.0, p.said.1 - seen.1);
            let p_peer = p.said_sigma_m.powi(2);
            if ex * ex + ez * ez > FIX_GATE_CHI2 * (p_pose + p_obs + p_peer) {
                let msg = format!("위치 보정 안 함: 작업자 {}가 말한 자리와 {:.2} m 달라 보임 — 예상 오차 밖", p.id, ex.hypot(ez));
                self.log("vision", Level::Debug, msg, None);
                continue;
            }
            if best.as_ref().map_or(true, |b| p_obs + p_peer < b.0) {
                best = Some((p_obs + p_peer, ex, ez, p.id));
            }
        }
        let Some((p_other, ex, ez, id)) = best else { return };
        let k = p_pose / (p_pose + p_other).max(1e-9);
        if k * ex.hypot(ez) > 1e-3 {
            let odo = self.world.odom.travelled_m;
            self.pending_fix = Some(PoseFix { dx: k * ex, dz: k * ez, odo_m: odo, prev_odo_m: odo - self.world.odom.since_fix_m });
            self.last_peer_fix = Some(PeerFixNote { peer: id, gap_m: ex.hypot(ez), gain: k, time_s: self.world.time_s });
            let msg = format!("위치 보정: 작업자 {id}를 움직이는 기준점으로 — {:.2} m 달라 보여 {:.0}%만큼 고침 (보기 · 동료 σ {:.2} m)", ex.hypot(ez), k * 100.0, p_other.sqrt());
            self.log("vision", Level::Info, msg, Some(json!({"peer": id, "gap_m": ex.hypot(ez), "gain": k})));
        }
    }

    pub(crate) fn apply_fix(&mut self) -> Option<PoseFix> {
        let f = self.pending_fix.take()?;
        self.world.relocalize(f.dx, f.dz);
        let rate = self.drift_rate();
        self.gates.shift_landmarks(|x, z, odo| f.shift(x, z, odo, rate));
        Some(f)
    }

    pub fn to_world(&self, a: Action) -> Action {
        match a {
            Action::MoveTo { x, z, reach_m } => {
                let (x, z) = self.world.true_of(x, z);
                Action::MoveTo { x, z, reach_m }
            }
            Action::Face { x, z } => {
                let (x, z) = self.world.true_of(x, z);
                Action::Face { x, z }
            }
            Action::Look { x, z } => {
                let (x, z) = self.world.true_of(x, z);
                Action::Look { x, z }
            }
            Action::Turn { yaw_deg } => Action::Turn { yaw_deg: self.world.true_yaw_of(yaw_deg.to_radians()).to_degrees() },
            Action::Push { drop, x, z } => {
                let (x, z) = self.world.true_of(x, z);
                Action::Push { drop, x, z }
            }
            other => other,
        }
    }

    pub(crate) fn note_look(&mut self, kind: &str, what: String, sigma_before_m: Option<f32>, sigma_after_m: Option<f32>, innovation_m: Option<f32>) {
        let mark = VisionMark { tick: self.world.tick, kind: kind.to_string(), what, sigma_before_m, sigma_after_m, innovation_m };
        if let Some(rec) = self.recording.as_mut() {
            if rec.vision.len() < crate::record::MAX_VISION_MARKS {
                rec.vision.push(mark);
            }
        }
    }

    pub fn scan_tilted(&mut self, trigger: &str, tilt_deg: f32) -> Result<()> {
        self.scan_gaze(trigger, 0.0, tilt_deg)
    }

    pub fn scan_gaze(&mut self, trigger: &str, pan_deg: f32, tilt_deg: f32) -> Result<()> {
        let level = self.cam;
        self.cam.pitch_deg -= tilt_deg;
        // A look the vision aims takes the head from an off-the-ball glance (`World::glance_rad`): the pan is from the
        // way the body faces, whatever the neck was doing.
        self.cam.pan_deg += pan_deg - self.world.glance_rad(1).to_degrees();
        let r = self.scan_now(trigger).map(|_| ());
        self.cam = level;
        r
    }

    pub(crate) fn place_look(&mut self, cd: &mut u32, x: f32, z: f32) -> bool {
        if *cd > 0 {
            *cd -= 1;
            return false;
        }
        if self.dist_me(x, z) > RELOCALIZE_RANGE_M || !in_view(&self.cam, self.world.standing_eye_m(), self.me(), x, z) {
            return false;
        }
        let intermittent = self.cfg.vision.intermittent;
        if intermittent && memory_ok(self.pose_sigma_m(), DELIVER_REACH_M - crate::driver::PLACE_REACH_M, &self.cfg.vision) {
            return false;
        }
        *cd = if intermittent { self.cfg.vision.look_every_ticks } else { self.cfg.approach_scan_every_ticks };
        let looked = self.scan_gaze("place", 0.0, 0.0).is_ok();
        if looked {
            self.vstats.looks += 1;
        }
        looked
    }

    fn note_scan(&mut self, r: &ScanReport) {
        if r.device != self.last_device {
            self.last_device = r.device.clone();
            self.log("device", Level::Info, format!("인식 장치: {} (모드 {})", r.device, r.mode), None);
        }
        if r.embed_ms > 0.0 {
            self.log("embed", Level::Debug, format!("스캔 #{}: 이미지 조각 {}개 임베딩 {:.1} ms", r.seq, r.tiles, r.embed_ms), None);
        }
        let blocked: Vec<&Judged> = r.judged.iter().filter(|j| !j.accepted).collect();
        let accepted = r.judged.len() - blocked.len();
        let mut why: std::collections::BTreeMap<&str, u32> = std::collections::BTreeMap::new();
        for j in &blocked {
            if let Some(g) = j.blocked_by() {
                *why.entry(g).or_insert(0) += 1;
            }
        }
        let why_txt = if why.is_empty() { String::new() } else { format!(" (차단: {})", why.iter().map(|(g, n)| format!("{g}×{n}")).collect::<Vec<_>>().join(" ")) };
        let level = if r.judged.is_empty() { Level::Debug } else { Level::Info };
        let f = r.filter;
        let filter_txt = if f.is_empty() {
            String::new()
        } else {
            format!(" · 타일 제외 하늘 {} · 원거리 {} · 기하 재분류 {} · 기하 탈락 {} · 교차 억제 {}", f.skipped_sky, f.skipped_far, f.relabeled, f.dropped, f.merged)
        };
        let list: Vec<Value> = r
            .judged
            .iter()
            .take(12)
            .map(|j| json!({"label": j.cand.label, "score": j.cand.score, "margin": j.cand.margin, "accepted": j.accepted, "blocked_by": j.blocked_by(), "landmark": j.landmark}))
            .collect();
        self.log(
            "vision",
            level,
            format!("스캔 #{} [{}]: 후보 {}건 → 수락 {accepted}건{why_txt}{filter_txt}", r.seq, r.trigger, r.judged.len()),
            Some(json!({"seq": r.seq, "tick": r.tick, "total_ms": r.total_ms, "candidates": list, "filter": f})),
        );
        let need = self.cfg.gates.confirm_sightings;
        for j in &r.judged {
            if let (true, Some(id)) = (j.accepted, j.landmark) {
                if self.gates.landmark(id).map_or(false, |l| l.good == need) {
                    if let Some(l) = self.gates.landmark(id) {
                        let msg = format!("랜드마크 L{id} 확인: {} ({}) @ ({:.1}, {:.1}) 점수 {:.2}", l.label, l.link, l.x, l.z, l.score);
                        let level = if self.focus.as_deref() == Some(l.link.as_str()) { Level::Info } else { Level::Debug };
                        self.audit.log("vision", level, msg, None);
                    }
                }
            }
            if let Some(v) = &j.temporal {
                if v.outcome == noto_ts::TemporalOutcome::Held {
                    let msg = format!("시계열 게이트가 {}의 점수 {:.2}를 보류: {}", j.cand.label, v.observed, v.reason);
                    self.audit.log(
                        "ts",
                        Level::Warn,
                        msg,
                        Some(tidy(json!({"label": j.cand.label, "observed": v.observed, "expected": v.expected, "lo": v.lo, "hi": v.hi, "method": v.method}))),
                    );
                }
            }
        }
    }

    pub fn snapshot(&self) -> WorldSnapshot {
        self.world.snapshot()
    }

    pub fn body_status(&self) -> BodyStatus {
        let (conf, ms) = match &self.last_scan {
            Some(r) => (r.confidence(), Some(r.total_ms)),
            None => (None, None),
        };
        self.world.body_status(conf, ms)
    }

    pub fn sds_status(&self) -> SdsStatus {
        self.sds.status(80)
    }

    pub fn ts_view(&self, label: Option<&str>, horizon: usize) -> TsView {
        let labels = self.gates.series_labels();
        let label = label.map(String::from).or_else(|| labels.first().map(|l| l.0.clone())).unwrap_or_default();
        TsView { history: self.gates.series(&label).to_vec(), forecast: self.gates.forecast_for(&label, horizon, &self.sds), label, labels }
    }

    pub fn log_tail(&self, after_seq: u64, limit: usize, cat: Option<&str>) -> (Vec<LogLine>, u64) {
        self.audit.tail(after_seq, limit, cat)
    }

    /// The depth tab: the model's answer on what the head camera sees now, put through what the robot does with it —
    /// ADE20K sky taken out, then aligned to the ground under the camera's own pose (eye height, pitch with the neck's
    /// tilt, chassis roll), with the predicted normals keeping objects out of that fit — next to the raw output and
    /// the renderer's truth on one colour scale.
    pub fn depth_view(&mut self, quality: f32) -> Result<DepthView> {
        let frame = self.world.render(&self.cam);
        let truth = crate::ml::KeyTruth::capture(&self.world, &frame);
        let rgb = noto_vision::Rgb::new(frame.width as u32, frame.height as u32, frame.rgb.clone());
        let intr = crate::record::intrinsics(&self.world, &self.cam);
        let pose = crate::record::odometry(&self.world, &self.cam);
        let t0 = Instant::now();
        let fix = self.ml.fix.clone();
        let right = self.stereo_baseline().map(|b| self.world.render_right(&self.cam, b));
        let (scene, max_depth_m, stereo) = match &mut self.ml.depth {
            crate::ml::DepthEngine::Model(m) if m.stereo_rig().is_some() => {
                let r = right.as_ref().ok_or_else(|| anyhow!("스테레오 깊이(SGBM)에 오른쪽 화면이 없습니다"))?;
                let pair = noto_vision::Rgb::new(r.width as u32, r.height as u32, r.rgb.clone());
                (noto_vision::DepthScene::depth_only(m.infer_pair(&rgb, &pair)?), m.max_depth_m(), m.stereo_stats())
            }
            crate::ml::DepthEngine::Model(m) => (m.infer_scene(&rgb, noto_vision::SceneHeads::ALL)?, m.max_depth_m(), None),
            crate::ml::DepthEngine::StandIn(s) => (noto_vision::DepthScene::depth_only(s.run(&truth)), None, None),
        };
        let infer_ms = t0.elapsed().as_secs_f32() * 1000.0;
        let raw = scene.depth.clone();
        let mut pred = scene.depth.clone();
        // A stereo pair's depth is metric already: no ground fit, and the raw panel would be the same map.
        let align = match (max_depth_m, &fix.align) {
            (Some(max), Some(cfg)) if stereo.is_none() => noto_vision::ground_align_with(&mut pred, scene.normals.as_ref().filter(|_| fix.normals), &intr, &pose, None, Some(max), cfg),
            _ => None,
        };
        let true_map = truth.depth_map();
        let metrics = noto_vision::depth_metrics(&pred, &true_map, 24.0, false);
        let metrics_raw = max_depth_m.filter(|_| stereo.is_none()).and_then(|_| noto_vision::depth_metrics(&raw, &true_map, 24.0, false));
        let (lo, hi) = true_map.range();
        let enc = |img: &noto_vision::Rgb| encode_webp(&img.data, img.width, img.height, quality).map(|e| e.bytes).map_err(|e| anyhow!("WebP 인코딩 실패: {e}"));
        Ok(DepthView {
            tick: self.world.tick,
            width: frame.width as u32,
            height: frame.height as u32,
            mode: self.ml.depth_mode(),
            device: self.ml.depth_device(),
            infer_ms,
            metrics,
            scale_m: (lo, hi),
            max_depth_m,
            rgb_webp: enc(&rgb)?,
            pred_webp: enc(&pred.colorize_in(lo, hi))?,
            truth_webp: enc(&true_map.colorize_in(lo, hi))?,
            metrics_raw,
            raw_webp: max_depth_m.filter(|_| stereo.is_none()).map(|_| enc(&raw.colorize_in(lo, hi))).transpose()?,
            right_webp: right.as_ref().map(|r| encode_webp(&r.rgb, r.width as u32, r.height as u32, quality).map(|e| e.bytes).map_err(|e| anyhow!("WebP 인코딩 실패: {e}"))).transpose()?,
            stereo,
            seg_webp: scene.segmentation.as_ref().map(|s| enc(&s.colorize())).transpose()?,
            normals_webp: scene.normals.as_ref().map(|n| enc(&n.colorize())).transpose()?,
            align,
            sky_frac: scene.segmentation.as_ref().map(|s| s.sky_fraction()),
            flip_tta: false,
        })
    }

    pub fn frame(&mut self, view: FrameView, quality: f32) -> Result<FrameOut> {
        let (w, h) = (self.cam.width as u32, self.cam.height as u32);
        self.frame_sized(view, w, h, quality)
    }

    pub fn frame_sized(&mut self, view: FrameView, width: u32, height: u32, quality: f32) -> Result<FrameOut> {
        let mut overlay = None;
        let mut marks = Vec::new();
        let (frame, detections) = match view {
            FrameView::Agent => {
                let cam = Camera { width: width.clamp(32, 2048) as usize, height: height.clamp(32, 2048) as usize, ..self.cam };
                let f = self.world.render(&cam);
                let now = self.world.head_view(&cam).view;
                let det = self
                    .last_scan
                    .as_ref()
                    .filter(|r| self.world.tick.saturating_sub(r.tick) <= 6 && r.seen_through(&now))
                    .map(|r| {
                        r.judged
                            .iter()
                            .map(|j| Detection {
                                label: j.cand.label.clone(),
                                label_ko: label_ko(&self.world, &j.cand.label),
                                score: j.cand.score,
                                margin: j.cand.margin,
                                accepted: j.accepted,
                                blocked_by: j.blocked_by().map(String::from),
                                reason: j.reason_ko(),
                                bbox: j.cand.bbox,
                                landmark: j.landmark,
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                (f, det)
            }
            FrameView::Fork => {
                let cam = Camera { width: width.clamp(32, 2048) as usize, height: height.clamp(32, 2048) as usize, ..self.cam };
                self.world.fork_cam = noto_core::act::fork::applies(&self.world.body);
                let f = self.world.render(&cam);
                self.world.fork_cam = false;
                (f, Vec::new())
            }
            FrameView::EyeDepth => {
                let cam = Camera { width: width.clamp(32, 1024) as usize, height: height.clamp(32, 1024) as usize, ..self.cam };
                let baseline = self.stereo_baseline().ok_or_else(|| anyhow!("카메라 1대(TIPSv2) 몸에는 스테레오 깊이 화면이 없습니다"))?;
                // The box of the item the last look checked, while that look is fresh (as the procedure's check).
                let target = self
                    .gaze_check
                    .as_ref()
                    .filter(|g| self.world.tick.saturating_sub(g.tick) < 600)
                    .and_then(|g| g.at.zip(self.world.drop_item(g.drop).map(|d| d.kind.spec().piece_size_m(d.amount))));
                let (f, o) = crate::stereo::eye_depth(&self.world, &cam, baseline, target)?;
                overlay = Some(o);
                (f, Vec::new())
            }
            FrameView::Top => {
                let (f, m) = world_sim::render::render_map(&self.world, width.clamp(32, 2048) as usize, height.clamp(32, 2048) as usize);
                marks = m;
                (f, Vec::new())
            }
            FrameView::Chase { orbit_deg } => {
                let cam = Camera { width: width.clamp(32, 2048) as usize, height: height.clamp(32, 2048) as usize, fov_deg: 64.0, max_dist_m: 30.0, ..self.cam };
                let (f, o) = crate::chase::render(&self.world, &cam, orbit_deg as f32);
                overlay = o;
                (f, Vec::new())
            }
            FrameView::Focus(which) => {
                let cam = Camera { width: width.clamp(32, 2048) as usize, height: height.clamp(32, 2048) as usize, fov_deg: 50.0, max_dist_m: 12.0, ..self.cam };
                (world_sim::render::render_focus(&self.world, &cam, which), Vec::new())
            }
            FrameView::Iso { az_step, zoom_pct, dimetric } => {
                let cam = Camera { width: width.clamp(32, 2048) as usize, height: height.clamp(32, 2048) as usize, max_dist_m: 60.0, ..self.cam };
                let (f, o) = crate::views::render_iso(&self.world, &cam, &mut self.iso, az_step, zoom_pct as f32 / 100.0, dimetric);
                overlay = o;
                (f, Vec::new())
            }
        };
        let enc = encode_webp(&frame.rgb, frame.width as u32, frame.height as u32, quality).map_err(|e| anyhow!("WebP 인코딩 실패: {e}"))?;
        Ok(FrameOut {
            width: enc.width,
            height: enc.height,
            bytes: enc.bytes.len(),
            encode_ms: enc.encode_ms,
            lossless: enc.lossless,
            tick: frame.tick,
            detections,
            overlay,
            marks,
            webp: enc.bytes,
        })
    }
}

pub(crate) fn tidy(v: Value) -> Value {
    match v {
        Value::Number(n) => match n.as_f64() {
            Some(x) if n.is_f64() => serde_json::Number::from_f64((x * 1e4).round() / 1e4).map(Value::Number).unwrap_or(Value::Null),
            _ => Value::Number(n),
        },
        Value::Array(a) => Value::Array(a.into_iter().map(tidy).collect()),
        Value::Object(o) => Value::Object(o.into_iter().map(|(k, v)| (k, tidy(v))).collect()),
        other => other,
    }
}

fn plan_brief(p: &ParsedCommand) -> String {
    match &p.plan {
        Some(Plan::Fetch { targets, destination }) => format!(
            "가져오기 {} → {}",
            targets.iter().map(|t| format!("{}×{}", t.resource, t.quantity)).collect::<Vec<_>>().join(", "),
            destination.place
        ),
        Some(Plan::Collect { targets }) => format!("채취 {}", targets.iter().map(|t| format!("{}×{}", t.resource, t.quantity)).collect::<Vec<_>>().join(", ")),
        Some(Plan::Goto { place }) => format!("이동 {}", place.place),
        Some(Plan::Drop { items, place }) => format!(
            "내려놓기 {}{}",
            if items.is_empty() { "전부".to_string() } else { items.iter().map(|t| format!("{}×{}", t.resource, t.quantity.map_or("전부".to_string(), |q| q.to_string()))).collect::<Vec<_>>().join(", ") },
            place.as_ref().map(|p| format!(" @ {}", p.place)).unwrap_or_default()
        ),
        Some(Plan::Stop) => "중지".to_string(),
        None => p.problem.as_ref().map(|x| format!("해석 불가({})", x.code)).unwrap_or_else(|| "해석 불가".into()),
    }
}

#[cfg(test)]
mod peer_tests {
    use super::*;

    fn walker() -> Runtime {
        let lex = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/world_lexicon.json")).expect("lexicon file");
        let mut cfg = crate::config::RuntimeConfig { parcels: Some(0), ..crate::config::RuntimeConfig::default() };
        cfg.noto.base = noto_core::BaseKind::HumanFoot;
        Runtime::new(cfg, &lex, None).expect("runtime")
    }

    /// A walker whose odometry has run 12 cm off over `walked_m` since its last flag, no flag in sight, and a peer 2 m
    /// ahead that says where it stands `said_off` m off, having travelled `since_fix_m` since a flag.
    fn drifted_with_peer(walked_m: f32, said_off: f32, since_fix_m: f32) -> (Runtime, u32) {
        let mut rt = walker();
        let w = rt.world_mut();
        w.nodes.clear();
        w.waypoints.clear();
        let id = w.add_crew(&[world_sim::Form::Humanoid], 1)[0];
        let (x, z, yaw) = (w.agent.x, w.agent.z, w.agent.yaw);
        let c = w.crew_worker_mut(id).unwrap();
        (c.x, c.z) = (x + 2.0 * yaw.cos(), z + 2.0 * yaw.sin());
        c.odom = world_sim::Odometry::new(world_sim::OdomCfg::exact(), 1, true, (c.x + said_off, c.z, c.yaw));
        c.odom.since_fix_m = since_fix_m;
        w.odom.x += 0.12;
        w.odom.since_fix_m = walked_m;
        (rt, id)
    }

    fn belief_err(rt: &Runtime) -> f32 {
        let (bx, bz, _) = rt.world().believed_pose();
        (bx - rt.world().agent.x).hypot(bz - rt.world().agent.z)
    }

    #[test]
    fn peer_landmark_corrects_odometry_drift() {
        let (mut rt, id) = drifted_with_peer(20.0, 0.0, 0.0);
        assert!(rt.world().peers_in_view(&rt.cam).iter().any(|p| p.id == id), "the peer is in view");
        assert!((belief_err(&rt) - 0.12).abs() < 1e-3);
        rt.scan_now("peer").expect("a scan");
        let fix = rt.apply_fix().expect("the peer, just put right by a flag, corrects the drift");
        assert!(fix.dx < -0.05, "it moves the belief back toward where it is: {fix:?}");
        assert!(belief_err(&rt) < 0.04, "12 cm → {:.3} m", belief_err(&rt));
    }

    #[test]
    fn a_peer_that_is_sure_of_a_wrong_place_is_not_taken_and_an_unsure_one_counts_for_little() {
        // Off by 1 m and sure of itself, to a walker 2 m from its last flag: outside what the σ allow, so the gate turns
        // it away. (Twenty metres from a flag the walker's own σ is a third of a metre, and the same peer would pass.)
        let (mut rt, _) = drifted_with_peer(2.0, 1.0, 0.0);
        rt.scan_now("peer").expect("a scan");
        assert!(rt.apply_fix().is_none(), "a peer far off its word is not a landmark");
        assert!((belief_err(&rt) - 0.12).abs() < 1e-3);
        // As unsure as the walker itself (the same 20 m since a flag): the walker meets it about halfway.
        let (mut rt, _) = drifted_with_peer(20.0, 0.0, 20.0);
        rt.scan_now("peer").expect("a scan");
        let fix = rt.apply_fix().expect("a fix");
        let k = fix.dx.abs() / 0.12;
        assert!(k > 0.3 && k < 0.7, "the gain splits the two σ: {k}");
    }
}
