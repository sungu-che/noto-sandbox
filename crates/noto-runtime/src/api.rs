use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use noto_core::{BaseKind, NotoConfig};
use noto_search::{Category, ParsedCommand, Plan, Via};
use serde_json::{json, Value};
use world_sim::snapshot::{CarryView, Pos3, SiteView};
use world_sim::WorldSnapshot;

use noto_sds::LogLine;

use crate::precision::Judged;
use crate::runtime::{label_ko, FrameView, Runtime, Submission};

pub const MIN_CELLS: usize = 32;
pub const MAX_CELLS: usize = 160;

pub const COMMANDS: [&str; 53] = [
    "act_status",
    "units_list",
    "unit_frame",
    "unit_apply",
    "rig_prompt",
    "rig_edit",
    "rig_params",
    "rig_commit",
    "crew_remove",
    "crew_add",
    "crew_clear",
    "crew_mode",
    "crew_status",
    "camera_rig",
    "rig_presets",
    "rig_design",
    "rig_import",
    "rig_keep_design",
    "rig_validate",
    "rig_apply",
    "rig_delete",
    "inventory_drop",
    "world_reset",
    "body_reset",
    "world_terrain",
    "world_new",
    "sim_step",
    "world_frame",
    "command_submit",
    "quest_list",
    "perception_scan",
    "body_status",
    "sds_status",
    "ts_forecast",
    "log_tail",
    "models_status",
    "test_start",
    "test_step",
    "test_status",
    "test_keyframe",
    "test_video",
    "test_stats",
    "test_cancel",
    "body_presets",
    "body_plan_eval",
    "body_plan_set",
    "world_body",
    "tips_use",
    "sam_use",
    "devices",
    "nn_status",
    "physics_status",
    "world_light",
];

const FORECAST_HORIZON: usize = 16;

fn r3(x: f32) -> f64 {
    let v = ((x as f64) * 1000.0).round() / 1000.0;
    if v == 0.0 { 0.0 } else { v }
}

/// Millimetre resolution, for the stereo σ of a near look.
fn r4f(x: f32) -> f64 {
    (x as f64 * 10_000.0).round() / 10_000.0
}

fn r4(x: f64) -> f64 {
    (x * 1e4).round() / 1e4
}

pub fn iso_ms(ms: i64) -> String {
    let secs = ms.div_euclid(1000);
    let milli = ms.rem_euclid(1000);
    let days = secs.div_euclid(86_400);
    let sod = secs.rem_euclid(86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.{milli:03}Z", sod / 3600, sod % 3600 / 60, sod % 60)
}

fn wrap_deg(a: f32) -> f32 {
    let mut a = a % 360.0;
    if a > 180.0 {
        a -= 360.0;
    } else if a <= -180.0 {
        a += 360.0;
    }
    a
}

fn pos(p: &Pos3, depth_m: f32) -> Value {
    json!([r3(p.x), r3(depth_m - p.z), r3(p.y)])
}

fn at2(p: [f32; 2], depth_m: f32) -> Value {
    json!([r3(p[0]), r3(depth_m - p[1])])
}

fn at3(p: [f32; 3], depth_m: f32) -> Value {
    json!([r3(p[0]), r3(depth_m - p[1]), r3(p[2])])
}

fn rect(r: [f32; 4], depth_m: f32) -> Value {
    json!([r3(r[0]), r3(depth_m - r[3]), r3(r[2]), r3(depth_m - r[1])])
}

fn site_json(s: &SiteView, d: f32) -> Value {
    json!({
        "houses": s.houses.iter().map(|h| json!({
            "id": h.id, "label_ko": h.label_ko, "rect": rect(h.rect_m, d), "front_deg": r3(wrap_deg(-h.front_deg)),
            "floor_m": r3(h.floor_m), "door": at2(h.door, d), "porch": rect(h.porch_m, d),
            "stairs": {"risers": h.stairs.risers, "riser_m": r3(h.stairs.riser_m), "tread_m": r3(h.stairs.tread_m), "width_m": r3(h.stairs.width_m),
                       "treads": h.stairs.treads.iter().map(|t| at3(*t, d)).collect::<Vec<_>>()},
            "ramp": h.ramp.as_ref().map(|r| json!({"cells": r.cells.iter().map(|c| at3(*c, d)).collect::<Vec<_>>(), "rise_m": r3(r.rise_m), "run_m": r3(r.run_m), "run_per_rise": r3(r.run_per_rise)})),
            "reachable_legs": h.reachable_legs, "reachable_wheels": h.reachable_wheels,
        })).collect::<Vec<_>>(),
        "warehouse": s.warehouse.as_ref().map(|w| json!({
            "label_ko": w.label_ko, "rect": rect(w.rect_m, d), "door": at2(w.door, d), "door_width_m": r3(w.door_width_m),
            "door_front_deg": r3(wrap_deg(-w.door_front_deg)), "aisle_m": r3(w.aisle_m), "shelves": w.shelves, "tables": w.tables,
            "zones": w.zones.iter().map(|z| json!({"kind": z.kind, "label_ko": z.label_ko, "rect": rect(z.rect_m, d), "em_lx": r3(z.em_lx)})).collect::<Vec<_>>(),
        })),
        "fixtures": s.fixtures.iter().map(|f| json!({
            "id": f.id, "kind": f.kind, "label_ko": f.label_ko, "cells": f.cells.iter().map(|c| at2(*c, d)).collect::<Vec<_>>(),
            "yaw_deg": r3(wrap_deg(-f.yaw_deg)), "base_m": r3(f.base_m), "top_m": r3(f.top_m),
            "levels_m": f.levels_m.iter().map(|l| r3(*l)).collect::<Vec<_>>(), "blocks": f.blocks,
        })).collect::<Vec<_>>(),
        "slots": s.slots,
    })
}

fn carry(c: &CarryView) -> Value {
    json!({"item_id": c.item.id(), "label_ko": c.label_ko, "count": c.count, "amount": r3(c.amount), "mass_kg": r3(c.unit_kg), "total_kg": r3(c.mass_kg)})
}

pub fn snapshot_json(s: &WorldSnapshot) -> Value {
    let d = s.bounds.depth_m;
    let mut v = json!({
        "tick": s.tick,
        "time_s": r3(s.time_s as f32),
        "seed": s.seed,
        "terrain": s.terrain.id(),
        "bounds": [r3(s.bounds.width_m), r3(s.bounds.depth_m)],
        "agent": {
            "pos": pos(&s.agent.pos, d),
            "yaw_deg": r3(wrap_deg(-s.agent.yaw_deg)),
            "speed_mps": r3(s.agent.speed_mps),
            "base": s.agent.base,
            "fingers": s.agent.fingers,
            "carrying": s.agent.carrying.iter().map(carry).collect::<Vec<_>>(),
            "carried_kg": r3(s.agent.carried_kg),
            "capacity_kg": r3(s.agent.capacity_kg),
            "battery_ratio": r3(s.agent.battery_ratio),
            "action": s.agent.action,
            "motion": s.agent.motion.as_ref().map(|m| json!({"kind": m.kind, "label_ko": m.label_ko, "reversing": m.reversing})),
            "belief": s.agent.belief.as_ref().map(|b| json!({"pos": [r3(b.x), r3(d - b.z)], "yaw_deg": r3(wrap_deg(-b.yaw_deg)), "error_m": r3(b.error_m), "since_fix_m": r3(b.since_fix_m)})),
        },
        "waypoints": s.waypoints.iter().map(|w| json!({
            "id": w.key,
            "node_id": w.id,
            "label_ko": w.label_ko,
            "pos": pos(&w.pos, d),
            "stock": w.stock.iter().map(carry).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "nodes": s.nodes.iter().map(|n| json!({
            "id": n.id, "kind": n.kind.id(), "label_ko": n.label_ko, "pos": pos(&n.pos, d), "hp": r3(n.hp),
        })).collect::<Vec<_>>(),
        "drops": s.drops.iter().map(|x| {
            let mut v = json!({"id": x.id, "item_id": x.item.id(), "label_ko": x.label_ko, "pos": pos(&x.pos, d), "amount": r3(x.amount)});
            if !x.support.is_ground() {
                let mut on = json!(x.support);
                on["label_ko"] = json!(x.support.label_ko());
                v["support"] = on;
            }
            v
        }).collect::<Vec<_>>(),
    });
    if let Some(site) = &s.site {
        v["site"] = site_json(site, d);
    }
    if !s.crew.is_empty() {
        v["crew"] = json!(s
            .crew
            .iter()
            .map(|c| json!({
                "id": c.id, "form": c.form, "form_ko": c.form.label_ko(), "cameras": c.cameras,
                "pos": pos(&c.pos, d), "yaw_deg": r3(wrap_deg(-c.yaw_deg)), "speed_mps": r3(c.speed_mps),
                "carrying": c.carrying.iter().map(carry).collect::<Vec<_>>(), "carried_kg": r3(c.carried_kg), "capacity_kg": r3(c.capacity_kg),
                "job": c.job, "status_ko": c.status_ko,
                "rider": c.rider.as_ref().map(|r| json!([r3(r.x), r3(d - r.z)])),
            }))
            .collect::<Vec<_>>());
    }
    if let Some(r) = &s.rider {
        v["rider"] = json!([r3(r.x), r3(d - r.z)]);
    }
    v
}

pub fn log_json(l: &LogLine) -> Value {
    let mut v = json!({"seq": l.seq, "ts": iso_ms(l.ts), "cat": l.cat, "level": l.level, "msg": l.msg});
    if let Some(d) = &l.data {
        v["data"] = d.clone();
    }
    v
}

fn role_ko(c: Category) -> &'static str {
    match c {
        Category::Resource => "아이템",
        Category::Place => "장소",
        Category::Negation => "부정",
        Category::Direction => "도착 표지",
        Category::Source => "출발 표지",
        Category::Verb => "동작",
        Category::Filler => "군더더기",
    }
}

fn stage_of(v: Via) -> &'static str {
    match v {
        Via::Exact => "lexical",
        Via::Tail => "tail",
        Via::Prefix => "prefix",
        Via::Embedding => "embedding",
    }
}

fn parsed_json(rt: &Runtime, p: &ParsedCommand) -> Value {
    let w = rt.world();
    let place = |id: &str| json!({"kind": "waypoint", "id": id, "label_ko": label_ko(w, id)});
    let mean = |v: &[f32]| if v.is_empty() { 0.0 } else { v.iter().sum::<f32>() / v.len() as f32 };
    let targets_json = |targets: &[noto_search::Target]| {
        targets
            .iter()
            .map(|t| {
                let mut v = json!({"item_id": t.resource, "label_ko": label_ko(w, &t.resource), "count": t.quantity, "score": r3(t.score)});
                if !t.origin.is_empty() {
                    v["origin"] = json!(t.origin.iter().map(|o| json!({"id": o.place, "text": o.text})).collect::<Vec<_>>());
                }
                v
            })
            .collect::<Vec<_>>()
    };
    let (action, item_id, item_ko, count, dest, conf, targets) = match &p.plan {
        Some(Plan::Fetch { targets, destination }) => {
            let scores: Vec<f32> = targets.iter().map(|t| t.score).chain(std::iter::once(destination.score)).collect();
            let mut dest = place(&destination.place);
            if destination.implied() {
                dest["implied"] = json!(true);
            }
            summary_of(targets, "mine_and_deliver", Some(dest), mean(&scores), w, targets_json(targets))
        }
        Some(Plan::Collect { targets }) => {
            let scores: Vec<f32> = targets.iter().map(|t| t.score).collect();
            summary_of(targets, "mine", None, mean(&scores), w, targets_json(targets))
        }
        Some(Plan::Goto { place: pl }) => ("goto", None, None, None, Some(place(&pl.place)), pl.score, Vec::new()),
        Some(Plan::Drop { items, place: pl }) => {
            let one = items.len() == 1;
            let item_ko = match items.as_slice() {
                [] => Some("적재함 전부".to_string()),
                [t] if t.quantity.is_some() => Some(label_ko(w, &t.resource)),
                _ => Some(items.iter().map(|t| format!("{} {}", label_ko(w, &t.resource), t.quantity.map_or("전부".to_string(), |q| format!("{q}개")))).collect::<Vec<_>>().join(" + ")),
            };
            let tj = items.iter().map(|t| json!({"item_id": t.resource, "label_ko": label_ko(w, &t.resource), "count": t.quantity, "score": r3(t.score)})).collect::<Vec<_>>();
            let conf = if items.is_empty() { 1.0 } else { mean(&items.iter().map(|t| t.score).collect::<Vec<_>>()) };
            ("drop", items.first().filter(|_| one).map(|t| t.resource.clone()), item_ko, items.first().filter(|_| one).and_then(|t| t.quantity), pl.as_ref().map(|p| place(&p.place)), conf, tj)
        }
        Some(Plan::Stop) => ("stop", None, None, None, None, 1.0, Vec::new()),
        None => ("unknown", None, None, None, None, 0.0, Vec::new()),
    };
    json!({
        "action": action,
        "item_id": item_id,
        "item_label_ko": item_ko,
        "count": count,
        "destination": dest,
        "confidence": r3(conf),
        "targets": targets,
    })
}

type Summary = (&'static str, Option<String>, Option<String>, Option<u32>, Option<Value>, f32, Vec<Value>);

fn summary_of(targets: &[noto_search::Target], action: &'static str, dest: Option<Value>, conf: f32, w: &world_sim::World, tj: Vec<Value>) -> Summary {
    let first = targets.first();
    let one = targets.len() == 1;
    let item_ko = if one {
        first.map(|t| label_ko(w, &t.resource))
    } else {
        Some(targets.iter().map(|t| format!("{} {}개", label_ko(w, &t.resource), t.quantity)).collect::<Vec<_>>().join(" + "))
    };
    (action, first.map(|t| t.resource.clone()), item_ko, if one { first.map(|t| t.quantity) } else { None }, dest, conf, tj)
}

fn trace_json(p: &ParsedCommand, conf: f64) -> Vec<Value> {
    let mut rows = vec![json!({"stage": "normalize", "text": p.normalized, "score": null, "role": null, "accepted": true})];
    for h in &p.hits {
        let ruled_out = p.excluded.iter().any(|e| *e == h.text);
        rows.push(json!({"stage": stage_of(h.via), "text": h.text, "score": r3(h.score), "role": role_ko(h.category), "accepted": !ruled_out}));
    }
    for (value, start, end) in &p.quantities {
        let text = p.tokens.get(*start..*end).map(|t| t.join(" ")).unwrap_or_default();
        rows.push(json!({"stage": "quantity", "text": format!("{text} → {value}"), "score": 1.0, "role": "수량", "accepted": true}));
    }
    for w in &p.unmatched {
        rows.push(json!({"stage": "lexical", "text": w, "score": 0.0, "role": null, "accepted": false}));
    }
    for t in &p.trace {
        rows.push(json!({"stage": t.stage, "text": t.detail, "score": null, "role": null, "accepted": true}));
    }
    for w in &p.warnings {
        rows.push(json!({"stage": "warning", "text": w, "score": null, "role": null, "accepted": false}));
    }
    let end = match (&p.plan, &p.problem) {
        (Some(_), _) => json!({"stage": "assign", "text": "역할 배정 완료", "score": conf, "role": null, "accepted": true}),
        (None, Some(pr)) => json!({"stage": "assign", "text": pr.message, "score": 0.0, "role": null, "accepted": false}),
        (None, None) => json!({"stage": "assign", "text": "해석하지 못함", "score": 0.0, "role": null, "accepted": false}),
    };
    rows.push(end);
    rows
}

fn submission_json(rt: &Runtime, text: &str, s: &Submission) -> Value {
    let parsed = parsed_json(rt, &s.parsed);
    let conf = parsed["confidence"].as_f64().unwrap_or(0.0);
    json!({
        "ok": s.ok,
        "stopped": s.stopped,
        "parsed": parsed,
        "quest_id": s.quest_id,
        "trace": trace_json(&s.parsed, conf),
        "error": s.error,
        "error_code": s.error_code,
        "mode": s.mode,
        "text": text,
        "recording": rt.test_session().is_some_and(|t| t.live && t.phase != crate::TestPhase::Done),
        "queued": false,
    })
}

fn scan_labels(rt: &Runtime, judged: &[Judged], top_k: usize) -> Vec<Value> {
    use std::collections::BTreeMap;
    let mut best: BTreeMap<&str, (&Judged, u32, bool)> = BTreeMap::new();
    for j in judged {
        let e = best.entry(j.cand.label.as_str()).or_insert((j, 0, false));
        e.1 += 1;
        e.2 |= j.accepted;
        if j.cand.score > e.0.cand.score {
            e.0 = j;
        }
    }
    let mut rows: Vec<(&str, (&Judged, u32, bool))> = best.into_iter().collect();
    rows.sort_by(|a, b| b.1 .0.cand.score.total_cmp(&a.1 .0.cand.score));
    rows.into_iter()
        .take(top_k.max(1))
        .map(|(label, (j, n, any_ok))| {
            json!({
                "id": label,
                "label_ko": label_ko(rt.world(), label),
                "score": r3(j.cand.score),
                "margin": r3(j.cand.margin),
                "accepted": any_ok,
                "reason": if any_ok && !j.accepted { "다른 개체가 수락됨".to_string() } else { j.reason_ko() },
                "count": n,
                "blocked_by": j.blocked_by(),
                "floor": r3(j.floor),
                "truth": j.cand.truth_label,
            })
        })
        .collect()
}

fn num(args: &Value, key: &str) -> Option<f64> {
    args.get(key).and_then(Value::as_f64)
}

fn text<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key).and_then(Value::as_str)
}

pub fn stereo_json(st: noto_vision::StereoStats) -> Value {
    let px = st.pixels().max(1) as f32;
    let share = |n: u32| r3(n as f32 / px);
    json!({
        "grid": [st.width, st.height], "disparities": st.disparities, "focal_px": r3(st.focal_px), "baseline_m": r3(st.baseline_m), "ms": r3(st.ms),
        "valid": share(st.valid), "far": share(st.far),
        "rejected": {"border": share(st.border), "saturated": share(st.saturated), "no_texture": share(st.no_texture), "not_unique": share(st.not_unique), "lr_mismatch": share(st.lr_mismatch), "speckle": share(st.speckle)},
    })
}

pub fn device_json(sel: &noto_ml::Selection) -> Value {
    json!({
        "backends": noto_ml::probe_backends(),
        "selected": sel.backend.name(),
        "ordinal": sel.ordinal,
        "reason": sel.reason,
    })
}

impl Runtime {
    fn refuse_during_test(&self) -> Result<(), String> {
        match self.test_session().filter(|t| !t.live).map(|t| t.phase) {
            Some(crate::TestPhase::Game) | Some(crate::TestPhase::Perceive) | Some(crate::TestPhase::Decide) => {
                Err("통합 테스트가 진행 중입니다. 끝나거나 중지한 뒤에 진행하세요".into())
            }
            _ => Ok(()),
        }
    }

    pub fn live_running(&self) -> bool {
        self.test_session().is_some_and(|t| t.live && t.phase != crate::TestPhase::Done)
    }

    fn live_brief(&self) -> Value {
        match self.test_session() {
            None => json!({"active": false, "queued": self.live_queue.len()}),
            Some(t) => json!({
                "active": true, "live": t.live, "id": t.id, "phase": t.phase, "command": t.command, "ml_needed": t.ml_needed,
                "game": t.game.as_ref().map_or_else(|| if t.phase == crate::TestPhase::Game { self.recording.as_ref().map_or(0, |r| r.keyframes.len()) } else { 0 }, |r| r.keyframes.len()),
                "perceived": t.perceive_next,
                "decide": t.decide.as_ref().map_or_else(|| if t.phase == crate::TestPhase::Decide { self.recording.as_ref().map_or(0, |r| r.keyframes.len()) } else { 0 }, |r| r.keyframes.len()),
                "queued": self.live_queue.len(),
            }),
        }
    }

    fn live_pump(&mut self) {
        if self.live_running() || self.busy() {
            return;
        }
        if let Some(cmd) = self.live_queue.pop_front() {
            if let Err(e) = self.live_start(&cmd) {
                self.log("quest", noto_sds::Level::Warn, format!("대기 명령 \"{cmd}\"을 시작하지 못함: {e}"), None);
            }
        }
    }

    pub fn api_world_new(&mut self, seed: u64, size: usize, config: Option<NotoConfig>, base: Option<&str>) -> Result<Value, String> {
        if !(MIN_CELLS..=MAX_CELLS).contains(&size) {
            return Err(format!("size 는 {MIN_CELLS}~{MAX_CELLS} 셀이어야 합니다 (받은 값 {size})"));
        }
        if self.test_session().is_some() {
            let _ = self.api_test_call("test_cancel", &Value::Null);
        }
        self.live_queue.clear();
        let mut hand = config;
        if let Some(b) = base.filter(|b| !b.is_empty()) {
            let kind: BaseKind = serde_json::from_value(Value::String(b.to_string())).map_err(|_| format!("알 수 없는 베이스 '{b}'"))?;
            let mut h = hand.unwrap_or(self.cfg.noto);
            h.base = kind;
            hand = Some(h);
        }
        self.reset_with(Some(seed), Some(size), hand).map_err(|e| e.to_string())?;
        Ok(snapshot_json(&self.snapshot()))
    }

    pub fn api_sim_step(&mut self, ticks: u32) -> Result<Value, String> {
        self.refuse_during_test()?;
        let ticks = ticks.clamp(1, 200);
        let lines = if self.live_running() {
            let before = self.audit.last_seq();
            self.test_advance(ticks).map_err(|e| e.to_string())?;
            self.audit.tail(before, 500, None).0
        } else {
            self.step(ticks)
        };
        self.live_pump();
        let notable: Vec<&LogLine> = lines.iter().filter(|l| l.level != noto_sds::Level::Debug).collect();
        let events: Vec<Value> = notable[notable.len().saturating_sub(50)..].iter().map(|l| log_json(l)).collect();
        Ok(json!({
            "snapshot": snapshot_json(&self.snapshot()),
            "quests": self.api_quest_list()?,
            "body": self.api_body_status()?,
            "events": events,
            "test": self.live_brief(),
        }))
    }

    pub fn api_inventory_drop(&mut self, item: Option<&str>, amount: Option<u32>) -> Result<Value, String> {
        self.refuse_during_test()?;
        if self.world().agent.carried.is_empty() {
            return Err("적재함이 비어 있습니다".into());
        }
        let items = match item.filter(|i| !i.is_empty()) {
            Some(i) => {
                let k = world_sim::ItemKind::from_id(i).ok_or_else(|| format!("알 수 없는 물건 '{i}'"))?;
                if self.world().carried_amount(k) <= world_sim::world::AMOUNT_EPS {
                    return Err(format!("적재함에 {}이(가) 없습니다", k.spec().label_ko));
                }
                vec![noto_search::DropTarget { resource: i.to_string(), quantity: amount, via: Via::Exact, score: 1.0, text: k.spec().label_ko.to_string() }]
            }
            None => vec![],
        };
        let id = self.submit_plan(Plan::Drop { items, place: None }, "인벤토리 버튼")?;
        Ok(json!({"ok": true, "quest_id": id, "quests": self.api_quest_list()?}))
    }

    pub fn api_world_reset(&mut self) -> Result<Value, String> {
        if self.test_session().is_some() {
            let _ = self.api_test_call("test_cancel", &Value::Null);
        }
        self.live_queue.clear();
        self.reset_world().map_err(|e| e.to_string())?;
        Ok(json!({"snapshot": snapshot_json(&self.snapshot()), "quests": self.api_quest_list()?, "body": self.api_body_status()?}))
    }

    pub fn api_body_reset(&mut self) -> Result<Value, String> {
        self.reset_body();
        Ok(json!({"snapshot": snapshot_json(&self.snapshot()), "body": self.api_body_status()?}))
    }

    pub fn api_world_terrain(&self) -> Result<Value, String> {
        let g = self.world().terrain_grid();
        let mut v = json!({
            "kind": g.kind.id(), "cells_w": g.cells_w, "cells_d": g.cells_d, "cell_m": g.cell_m,
            "heights_cm": g.heights_cm, "curb_cm": g.curb_cm, "ground": g.ground,
            "ground_kinds": world_sim::Ground::ALL.iter().map(|k| json!({"code": k.code(), "id": k.id(), "label_ko": k.label_ko(), "color": k.color()})).collect::<Vec<_>>(),
            "layout": self.world().layout,
        });
        if !g.curb_kind.is_empty() {
            v["curb_kind"] = json!(g.curb_kind);
        }
        Ok(v)
    }

    pub fn api_physics_status(&self) -> Value {
        let w = self.world();
        let (post, state) = w.posture_now();
        let g = w.ground_here();
        let bal = w.balance_summary();
        let joints: Vec<Value> = w
            .heat
            .keys()
            .chain(w.wear.keys())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .map(|j| {
                let torque = bal.as_ref().and_then(|b| b.joints.iter().find(|x| &x.joint == j)).map(|x| x.torque_nm);
                let life = w.joint_actuator(j).zip(torque).map(|(a, t)| noto_core::wear::life_h(&a, t));
                json!({
                    "joint": j,
                    "heat": r3(w.heat.get(j).copied().unwrap_or(0.0)),
                    "life_used": w.wear.get(j).copied().unwrap_or(0.0),
                    "life_h_at_torque": life.filter(|l| l.is_finite()),
                    "rating": w.joint_actuator(j).map(|a| noto_core::wear::rating(&a)),
                })
            })
            .collect();
        let live = noto_core::effects::live_scenario(&w.body, &post).map(|s| noto_core::effects::evaluate(&w.body, &s));
        json!({
            "ground": {"id": g.id(), "label_ko": g.label_ko(), "surface": g.surface().props()},
            "state": state,
            "slope_deg": r3(post.slope_deg),
            "moving": post.propelling(),
            "contact": bal.as_ref().and_then(|b| b.contact),
            "joints": joints,
            "load": {"carried_kg": r3(w.carried_kg()), "body_kg": r3(w.body.mass_kg()), "capacity_kg": r3(w.capacity_kg()), "capacity_here_kg": r3(w.capacity_here())},
            "live": live,
            "surfaces": noto_core::surface::SURFACES,
        })
    }

    pub fn api_act_status(&self) -> Value {
        let w = self.world();
        let brief = |r: &noto_core::act::ActRollout, upto: f32| {
            json!({
                "kind": r.kind.id(), "kind_ko": r.kind.label_ko(),
                "duration_s": r3(r.duration_s), "elapsed_s": r3(upto.min(r.duration_s)),
                "outcome": {"ok": r.outcome.ok, "code": r.outcome.code, "reason": r.outcome.reason_ko},
                "events": r.events.iter().filter(|e| e.t <= upto + 1e-6).map(|e| json!({"t": r3(e.t), "kind": e.kind, "label": e.label_ko, "ok": e.ok, "value": e.value.map(r3), "parts": e.parts.iter().map(|p| p.id()).collect::<Vec<_>>()})).collect::<Vec<_>>(),
                "metrics": r.metrics.iter().filter(|(_, v)| v.is_finite()).map(|(k, v)| (k.to_string(), json!(r3(*v)))).collect::<serde_json::Map<String, Value>>(),
                "notes": r.notes,
            })
        };
        json!({
            "now": w.act.as_ref().map(|p| brief(&p.rollout, p.elapsed_s)),
            "last": w.last_act.as_ref().map(|p| brief(&p.rollout, p.rollout.duration_s)),
            "fallen": w.fallen_until.is_some(),
            "walk": {"steps": w.walk.steps, "slips": w.walk.slips},
            "focus": w.focus_frame(),
        })
    }

    pub fn api_world_frame(&mut self, width: u32, height: u32, quality: f32, view: &str) -> Result<Value, String> {
        let view = crate::views::parse_view(view)?;
        let view_id = if view == FrameView::Agent { "agent" } else { "other" };
        // A walker's pick-up HUD draws its arm and its fingers on the item while it bends, grips and lifts: the same
        // focus frame the focus views carry.
        let hands_busy = view == FrameView::Agent
            && !noto_core::act::fork::applies(&self.world.body)
            && matches!(self.world.act_phase(), Some((noto_core::act::ActKind::Grasp, "reach" | "close" | "squeeze" | "load" | "lift" | "stow")));
        let focus = (matches!(view, FrameView::Focus(_)) || hands_busy).then(|| serde_json::to_value(self.world.focus_frame()).unwrap_or(Value::Null));
        let f = self.frame_sized(view, width, height, quality).map_err(|e| e.to_string())?;
        let units = f.marks.iter().map(|m| self.unit_json(m.id).map_or(Value::Null, |mut u| {
            u["u"] = json!(r3(m.u));
            u["v"] = json!(r3(m.v));
            u
        })).collect::<Vec<_>>();
        let dets: Vec<Value> = f
            .detections
            .iter()
            .map(|d| {
                json!({
                    "label_id": d.label, "label_ko": d.label_ko, "score": r3(d.score), "margin": r3(d.margin),
                    "accepted": d.accepted, "blocked_by": d.blocked_by, "reason": d.reason,
                    "bbox": d.bbox.map(r3), "landmark": d.landmark,
                })
            })
            .collect();
        let camera = matches!(view_id, "agent").then(|| self.head_view_json());
        let procedure = matches!(view_id, "agent").then(|| self.procedure_json()).flatten();
        // The top view draws the whole map scaled into the frame (`render_map`): its frame pixels per metre, for the
        // robots' sight fans.
        let px_per_m = (view == FrameView::Top).then(|| {
            let (sx, sz) = self.world().terrain.size_m();
            1.0 / (sx / f.width as f32).max(sz / f.height as f32).max(1e-6)
        });
        Ok(json!({
            "webp_base64": STANDARD.encode(&f.webp),
            "width": f.width, "height": f.height, "bytes": f.bytes,
            "encode_ms": r3(f.encode_ms), "lossless": f.lossless, "tick": f.tick,
            "detections": dets,
            "overlay": f.overlay,
            "focus": focus,
            "camera": camera,
            "procedure": procedure,
            "light": crate::views::light_json(self.world()),
            "units": units,
            "px_per_m": px_per_m.map(r3),
            "relay": self.relay_json(),
        }))
    }

    /// What the first worker's physics and its own feel pass back into how it acts, for the screen: the grip it plans
    /// for the item in its hands (its learned friction, `relay`), what a load slid in its hands does to its whole body
    /// (`World::slide_relay`), the last time a sub-worker in view put its pose right, and its head's glance.
    fn relay_json(&self) -> Value {
        let w = self.world();
        let grip = w.act.as_ref().filter(|p| p.rollout.kind == noto_core::act::ActKind::Grasp).and_then(|p| Some((p.object?.0, p.pads?))).map(|(item, pads)| {
            let table = w.pad_mu(item);
            let feel = crate::relay::grip_feel(&self.sds, item.id(), pads);
            let prior = crate::relay::grip_prior(&self.sds, item.id(), pads, table);
            json!({"item": item.id(), "item_ko": item.spec().label_ko, "pads": pads, "table_mu": r3(table), "mu": r3(prior.map_or(table, |p| p.mu)),
                "slip_rate": r3(feel.slip_rate), "mu_seen": feel.mu_seen.map(r3), "grasps": feel.grasps, "learning": prior.is_none(), "min_grasps": crate::relay::GRIP_PRIOR_MIN_N})
        });
        let fix = self.last_peer_fix.filter(|f| w.time_s - f.time_s <= 6.0).map(|f| json!({"peer": f.peer, "gap_m": r3(f.gap_m), "gain": r3(f.gain), "age_s": r3((w.time_s - f.time_s) as f32)}));
        json!({"grip": grip, "slide": w.slide_relay(), "peer_fix": fix, "glance_deg": r3(w.glance_rad(1).to_degrees())})
    }

    /// A robot's eye on the top view: where it looks on the picture (x right, y down: its yaw less its glance, the
    /// way the head camera turns), the half of its view and how far it looks (`traffic::SIGHT_*`).
    fn sight_json(yaw: f32, glance: f32) -> Value {
        json!({"dir_deg": r3((yaw - glance).to_degrees()), "half_deg": world_sim::world::traffic::SIGHT_HALF_FOV_DEG, "range_m": world_sim::world::traffic::SIGHT_RANGE_M, "glance_deg": r3(glance.to_degrees())})
    }

    /// One robot for the screen: where it is, what it does, what it carries.
    fn unit_json(&self, id: u32) -> Option<Value> {
        let w = self.world();
        let load = |list: &[world_sim::snapshot::CarryView]| list.iter().map(|c| json!({"item": c.item.id(), "label_ko": c.label_ko, "count": c.amount})).collect::<Vec<_>>();
        let rig = |build: noto_core::dims::Build, base: BaseKind| self.rig_name_of(build, base);
        let v = if id == 1 {
            let s = w.snapshot();
            let (rig_id, rig_name) = rig(self.cfg.body.build, self.cfg.noto.base);
            json!({"id": 1, "first": true, "form": world_sim::crew::Form::of_base(self.cfg.noto.base).id(), "base": self.cfg.noto.base, "label_ko": world_sim::crew::Form::of_base(self.cfg.noto.base).label_ko(),
                "x": r3(w.agent.x), "z": r3(w.agent.z), "yaw_deg": r3(w.agent.yaw.to_degrees()), "moving": w.agent.speed_mps > 0.01, "fallen": w.fallen_until.is_some_and(|t| w.time_s < t),
                "status_ko": w.current_action().unwrap_or_else(|| "대기".to_string()), "carried_kg": r3(w.carried_kg()), "capacity_kg": r3(w.capacity_kg()), "battery_ratio": r3(w.battery_ratio()),
                "load": load(&s.agent.carrying), "cameras": self.cameras(), "rig_id": rig_id, "rig_name_ko": rig_name,
                "sight": Self::sight_json(w.agent.yaw, w.glance_rad(1)), "noticed": w.noticed.iter().map(|n| n.id).collect::<Vec<_>>()})
        } else {
            let c = w.crew_worker(id)?;
            let (rig_id, rig_name) = rig(c.body.plan.build, c.body.base);
            json!({"id": id, "first": false, "form": c.form.id(), "base": c.body.base, "label_ko": c.form.label_ko(), "x": r3(c.x), "z": r3(c.z), "yaw_deg": r3(c.yaw.to_degrees()), "moving": c.speed_mps > 0.01,
                "fallen": c.fallen_until.is_some_and(|t| w.time_s < t), "status_ko": c.status_ko(), "carried_kg": r3(c.carried_kg()), "capacity_kg": r3(c.capacity_kg), "battery_ratio": 1.0,
                "load": load(&c.carry_views()), "cameras": c.cameras, "rig_id": rig_id, "rig_name_ko": rig_name,
                "sight": Self::sight_json(c.yaw, w.glance_rad(id)), "noticed": c.noticed.iter().map(|n| n.id).collect::<Vec<_>>(),
                "pose_sigma_m": r3(c.pose_sigma_m()), "belief_err_m": r3((c.odom.x - c.x).hypot(c.odom.z - c.z))})
        };
        Some(v)
    }

    pub fn api_units_list(&self) -> Result<Value, String> {
        let ids: Vec<u32> = std::iter::once(1).chain(self.world().crew.iter().map(|c| c.id)).collect();
        Ok(json!({"units": ids.into_iter().filter_map(|id| self.unit_json(id)).collect::<Vec<_>>()}))
    }

    /// What one robot's camera sees now: its picture, its depth, or a lift truck's fork camera. A sub-worker has no
    /// models of its own, so its depth is the renderer's.
    pub fn api_unit_frame(&mut self, args: &Value) -> Result<Value, String> {
        let unit = num(args, "unit").unwrap_or(1.0) as u32;
        let (width, height) = (num(args, "width").unwrap_or(160.0).clamp(32.0, 1024.0) as usize, num(args, "height").unwrap_or(120.0).clamp(32.0, 1024.0) as usize);
        let quality = num(args, "quality").unwrap_or(60.0) as f32;
        let mode = text(args, "mode").unwrap_or("rgb").to_string();
        let cam = world_sim::Camera { width, height, ..*self.camera() };
        let frame = self.world_mut().render_unit(unit, &cam, mode == "fork").ok_or_else(|| format!("유닛 {unit}가 없습니다"))?;
        let (rgb, depth_name) = if mode == "depth" {
            let map = noto_vision::DepthMap { width: frame.width as u32, height: frame.height as u32, kind: noto_vision::DepthKind::Metric, data: frame.depth.clone() };
            let (lo, hi) = map.range();
            (map.colorize_in(lo, hi).data, if unit == 1 { self.ml.depth_mode() } else { "렌더러 깊이".to_string() })
        } else {
            (frame.rgb.clone(), String::new())
        };
        let e = world_sim::encode_webp(&rgb, frame.width as u32, frame.height as u32, quality).map_err(|e| format!("WebP 인코딩 실패: {e}"))?;
        let label = self.unit_json(unit).and_then(|u| u["label_ko"].as_str().map(str::to_string)).unwrap_or_default();
        Ok(json!({"webp_base64": STANDARD.encode(&e.bytes), "width": frame.width, "height": frame.height, "unit": unit, "mode": mode, "label_ko": label, "depth_name": depth_name,
            "procedure": if unit == 1 { self.procedure_json() } else { None }}))
    }

    /// The pick-up stage by stage, for the first-person HUD. Legs do it as a person does: walk up, bend the head to
    /// look, check with vision, lower the body, grip, lift with the eyes on the item. Wheels and tracks do it as a lift
    /// truck does, with two eyes checked against each other: the rider looks down at the item, looks ahead again while
    /// the tines go down and the carriage camera checks it, the base slides the tines under it with the carriage camera
    /// on it until the item blocks that camera, the rider looks down once more to see it ride on, and the mast lifts
    /// it; setting down runs the other way. Read from the world's act phase (or held look) and the last checks.
    pub fn procedure_json(&self) -> Option<Value> {
        use noto_core::act::ActKind;
        const HAND: [(&str, &str); 6] = [("approach", "다가가기"), ("look", "고개 숙여 응시"), ("check", "비전 확인"), ("bend", "자세 낮춤"), ("grip", "쥐기"), ("lift", "시선 따라 들어올림")];
        const FORK: [(&str, &str); 9] = [
            ("approach", "다가가기"),
            ("align", "정렬"),
            ("rider", "탑승자 시선 내려 확인"),
            ("lower", "포크 내림 · 시선 정면"),
            ("check", "포크 카메라 확인 · 교차 검증"),
            ("insert", "포크 끼우기 · 포크 카메라 추적"),
            ("recheck", "포크 카메라 가림 → 탑승자 재확인"),
            ("seat", "안착"),
            ("raise", "들어올림"),
        ];
        const FORK_DOWN: [(&str, &str); 3] = [("lower_load", "짐 내림"), ("back_out", "후진해 포크 빼기"), ("rest", "포크 올림")];
        let w = self.world();
        let forks = noto_core::act::fork::applies(&w.body);
        let action = w.current_action().unwrap_or_default();
        let target = action.split(['(', ')']).nth(1).filter(|s| s.starts_with('d')).and_then(|s| s.get(1..)?.parse::<u32>().ok());
        let phase = w.act_phase();
        let stage = match phase {
            Some((ActKind::Fork, "rider_look" | "rider_check")) => "rider",
            Some((ActKind::Fork, "recheck_look" | "recheck")) => "recheck",
            Some((ActKind::Fork, p @ ("align" | "lower" | "check" | "insert" | "seat" | "raise" | "lower_load" | "back_out" | "rest"))) => p,
            Some((ActKind::Look, "hold")) if forks => "check",
            Some((ActKind::Look, "look")) | Some((ActKind::Grasp, "look")) => "look",
            Some((ActKind::Look, _)) | Some((ActKind::Grasp, "check")) => "check",
            Some((ActKind::Grasp, "reach")) => "bend",
            Some((ActKind::Grasp, "close" | "squeeze" | "load")) => "grip",
            Some((ActKind::Grasp, _)) => "lift",
            _ => {
                let next = self.task.as_ref().filter(|t| matches!(t.phase, crate::driver::Phase::Collect)).and_then(|t| t.drops.get(t.drop_idx).copied());
                let near = next.and_then(|id| w.drop_item(id)).is_some_and(|d| (d.x - w.agent.x).hypot(d.z - w.agent.z) < 3.0);
                if !(near && action.starts_with("move_to")) {
                    return None;
                }
                "approach"
            }
        };
        let setting_down = matches!(stage, "lower_load" | "back_out" | "rest");
        let list: &[(&str, &str)] = if setting_down { &FORK_DOWN } else if forks { &FORK } else { &HAND };
        let gazed = self.gaze_check.as_ref().map(|g| g.drop);
        let drop = if setting_down { None } else { target.or(gazed).or_else(|| self.task.as_ref().and_then(|t| t.drops.get(t.drop_idx).copied())) };
        let at = list.iter().position(|(id, _)| *id == stage).unwrap_or(0);
        let stages: Vec<Value> = list
            .iter()
            .enumerate()
            .map(|(i, (id, ko))| json!({"id": id, "label_ko": ko, "state": if i < at { "done" } else if i == at { "active" } else { "pending" }}))
            .collect();
        let item_json = |p: &noto_vision::ItemPose| {
            json!({"yaw_deg": r3(p.yaw_deg), "rel_yaw_deg": r3(p.rel_yaw_deg), "length_m": r3(p.length_m), "width_m": r3(p.width_m), "height_m": r3(p.height_m), "top_tilt_deg": r3(p.top_tilt_deg), "elongation": r3(p.elongation), "form": p.form, "points": p.points})
        };
        let check = self.gaze_check.as_ref().filter(|g| Some(g.drop) == drop && w.tick.saturating_sub(g.tick) < 600).map(|g| {
            json!({"ok": g.ok, "label_ko": g.label_ko, "score": g.score.map(r3), "range_m": g.range_m.map(r3), "depth_m": g.depth_m.map(r3), "bbox": g.bbox.map(|b| b.map(r3)), "reason": g.reason, "depth_source": g.depth_source, "depth_sigma_m": g.depth_sigma_m.map(r4f),
                   "depth_valid": g.depth_valid.map(r3), "item": g.item.as_ref().map(item_json)})
        });
        let hv = w.head_view(&self.cam);
        let pose = w.current_kin_pose();
        let label_ko = drop
            .and_then(|id| w.drop_item(id))
            .map(|d| d.kind.spec().label_ko.to_string())
            .or_else(|| setting_down.then(|| w.agent.carried.first().map(|(k, _)| k.spec().label_ko.to_string())).flatten())
            .or_else(|| self.gaze_check.as_ref().map(|g| g.label_ko.clone()));
        let tip = w.act.as_ref().or(w.last_act.as_ref()).filter(|_| forks).and_then(|p| p.rollout.metric("tip_ratio"));
        let sight = |s: &crate::vision::Sighting| {
            json!({"camera": s.camera, "ok": s.ok, "matched": s.matched, "label_ko": s.label_ko, "score": s.score.map(r3), "range_m": s.range_m.map(r3), "depth_m": s.depth_m.map(r3), "bbox": s.bbox.map(|b| b.map(r3)), "neck_deg": r3(s.neck_deg), "reason": s.reason,
                   "depth_source": s.depth_source, "depth_sigma_m": s.depth_sigma_m.map(r4f), "depth_valid": s.depth_valid.map(r3), "sigma_m": s.sigma_m.map(r4f), "item": s.item.as_ref().map(item_json)})
        };
        let cross = self.fork_cross.as_ref().filter(|c| forks && !setting_down && Some(c.drop) == drop).map(|c| {
            json!({
                "rider": c.rider.as_ref().map(sight),
                "fork": c.fork.as_ref().map(sight),
                "recheck": c.recheck.as_ref().map(sight),
                "verdict": c.fork.as_ref().map(|_| c.verdict()),
                "apart_m": c.fork_vs_rider.map(|v| r3(v.0)),
                "gate_m": c.fork_vs_rider.map(|v| r3(v.1)),
                "recheck_apart_m": c.recheck_vs_fork.map(|v| r3(v.0)),
                "recheck_gate_m": c.recheck_vs_fork.map(|v| r3(v.1)),
                "seated": c.seated(),
                "blocked_share": c.blocked_share.map(r3),
            })
        });
        Some(json!({
            "act": if setting_down { "set_down" } else if matches!(phase, Some((ActKind::Look, "look" | "check")) | Some((ActKind::Fork, "align" | "rider_look" | "rider_check" | "lower" | "check"))) && action.starts_with("look") { "look" } else { "pick" },
            "mechanism": if forks { "fork" } else { "hand" },
            "cameras": self.cameras(),
            "depth_procedure": if self.stereo() { "sgbm" } else { "tipsv2" },
            "camera": if forks && matches!(stage, "lower" | "check" | "insert") { "fork" } else { "head" },
            "cross": cross,
            "fork_m": forks.then(|| r3(w.fork_m_now())),
            "tip_ratio": tip.map(r3),
            "target": drop.map(|d| format!("d{d}")),
            "label_ko": label_ko,
            "stage": stage,
            "stages": stages,
            "neck_deg": r3(pose.neck_deg),
            "trunk_deg": r3(pose.trunk_deg()),
            "eye_m": r3(hv.eye_m),
            "standing_eye_m": r3(w.standing_eye_m()),
            "pitch_deg": r3(hv.view.pitch.to_degrees()),
            "check": check,
        }))
    }

    pub fn head_view_json(&self) -> Value {
        let w = self.world();
        let hv = w.head_view(&self.cam);
        let t = world_sim::world::sensor_timing_at(w.body.head.sensor, w.scene_lux());
        let deg = |r: f32| r3(r.to_degrees());
        json!({
            "eye_m": r3(hv.eye_m), "standing_eye_m": r3(w.standing_eye_m()),
            "pitch_deg": deg(hv.view.pitch), "roll_deg": deg(hv.view.roll), "yaw_deg": deg(hv.view.yaw),
            "head_pitch_deg": r3(hv.head_pitch_deg), "pan_deg": r3(hv.pan_deg), "mount_deg": r3(self.cam.pitch_deg),
            "shake": {"source": hv.shake.source, "dz_mm": r3(hv.shake.dz_m * 1000.0), "pitch_deg": deg(hv.shake.pitch), "roll_deg": deg(hv.shake.roll),
                      "rate_dps": [deg(hv.exposure.rate[0]), deg(hv.exposure.rate[1]), deg(hv.exposure.rate[2])]},
            "est_err_deg": {"pitch": deg(hv.view.pitch - hv.est.pitch), "roll": deg(hv.view.roll - hv.est.roll)},
            "blur_px": r3(hv.blur_px),
            "terrain_tilt_deg": [r3(hv.tilt_deg[0]), r3(hv.tilt_deg[1])],
            "sensor": {"name": w.body.head.sensor, "rgb": t.rgb, "exposure_ms": r3(t.exposure_s * 1000.0), "readout_ms": r3(t.readout_s * 1000.0), "gyro_arw_deg": r3(t.gyro_arw_deg), "scene_lux": r3(w.scene_lux())},
            "light": crate::views::light_json(w),
            "ground": w.ground_here().surface().props().id,
        })
    }

    pub fn api_depth_view(&mut self, quality: f32) -> Result<Value, String> {
        let d = self.depth_view(quality).map_err(|e| format!("{e:#}"))?;
        Ok(json!({
            "tick": d.tick, "width": d.width, "height": d.height, "mode": d.mode, "device": d.device, "infer_ms": r3(d.infer_ms),
            "metrics": d.metrics.map(|m| json!({"absrel": r3(m.absrel), "delta1": r3(m.delta1), "rmse_m": r3(m.rmse_m), "median_scale": r3(m.median_scale), "pixels": m.pixels})),
            "scale_m": [r3(d.scale_m.0), r3(d.scale_m.1)], "max_depth_m": d.max_depth_m.map(r3),
            "rgb_webp_base64": STANDARD.encode(&d.rgb_webp), "pred_webp_base64": STANDARD.encode(&d.pred_webp), "truth_webp_base64": STANDARD.encode(&d.truth_webp),
            "metrics_raw": d.metrics_raw.map(|m| json!({"absrel": r3(m.absrel), "delta1": r3(m.delta1), "rmse_m": r3(m.rmse_m), "median_scale": r3(m.median_scale), "pixels": m.pixels})),
            "raw_webp_base64": d.raw_webp.as_ref().map(|b| STANDARD.encode(b)),
            "seg_webp_base64": d.seg_webp.as_ref().map(|b| STANDARD.encode(b)),
            "normals_webp_base64": d.normals_webp.as_ref().map(|b| STANDARD.encode(b)),
            "align": d.align.map(|a| json!({"a": r3(a.a), "b": r3(a.b), "inliers": a.inliers, "candidates": a.candidates, "inlier_share": r3(a.inlier_share), "median_rel_err": r3(a.median_rel_err), "span_m": [r3(a.span_m.0), r3(a.span_m.1)], "off_ground_normals": a.off_ground_normals})),
            "sky": d.sky_frac.map(|f| json!({"frac": r3(f)})),
            "flip_tta": d.flip_tta,
            "camera": self.head_view_json(),
            "cameras": self.cameras(),
            "right_webp_base64": d.right_webp.as_ref().map(|b| STANDARD.encode(b)),
            "stereo": d.stereo.map(stereo_json),
        }))
    }

    pub fn api_command_submit(&mut self, input: &str) -> Result<Value, String> {
        self.api_command_submit_rec(input, false)
    }

    pub fn api_command_submit_rec(&mut self, input: &str, record: bool) -> Result<Value, String> {
        self.refuse_during_test()?;
        if !record {
            let s = self.submit(input);
            return Ok(submission_json(self, input, &s));
        }
        let probe = self.parser().parse(input, None);
        if matches!(probe.plan, Some(Plan::Stop)) {
            self.live_queue.clear();
            if self.live_running() {
                let _ = self.api_test_call("test_cancel", &Value::Null);
            }
            let s = self.submit(input);
            return Ok(submission_json(self, input, &s));
        }
        if probe.plan.is_some() && (self.live_running() || self.busy()) {
            self.live_queue.push_back(input.to_string());
            let n = self.live_queue.len();
            self.log("quest", noto_sds::Level::Info, format!("명령 대기 ({n}번째): \"{input}\" — 지금 기록이 끝나면 시작"), None);
            let s = self.submit_preview(input);
            let mut v = submission_json(self, input, &s);
            v["queued"] = json!(n);
            v["ok"] = json!(true);
            return Ok(v);
        }
        let s = self.live_start(input).map_err(|e| e.to_string())?;
        Ok(submission_json(self, input, &s))
    }

    pub fn api_quest_list(&self) -> Result<Value, String> {
        serde_json::to_value(self.quests()).map_err(|e| e.to_string())
    }

    pub fn api_perception_scan(&mut self, top_k: usize) -> Result<Value, String> {
        self.scan_now("manual").map_err(|e| e.to_string())?;
        let r = self.last_scan().ok_or("스캔 결과가 없습니다")?;
        Ok(json!({
            "labels": scan_labels(self, &r.judged, top_k),
            "tiles": r.tiles,
            "embed_ms": r3(r.embed_ms),
            "total_ms": r3(r.total_ms),
            "device": r.device,
            "mode": r.mode,
            "seq": r.seq,
            "tick": r.tick,
            "candidates": r.judged.len(),
        }))
    }

    pub fn api_body_status(&self) -> Result<Value, String> {
        serde_json::to_value(self.body_status()).map_err(|e| e.to_string())
    }

    pub fn api_sds_status(&self) -> Result<Value, String> {
        let mut v = serde_json::to_value(self.sds_status()).map_err(|e| e.to_string())?;
        let g = self.gates();
        let landmarks = g.landmarks();
        let forecastable: Vec<Value> = g
            .series_labels()
            .into_iter()
            .filter(|(_, n)| *n >= 3)
            .flat_map(|(l, n)| ["score", "margin"].map(|axis| json!({"scope": format!("vision|{l}|"), "axis": axis, "n": n})))
            .collect();
        v["forecastable"] = Value::Array(forecastable);
        v["precision"] = json!({
            "mode": self.recognizer_mode(),
            "gates_enabled": g.cfg().enabled,
            "evaluated": g.counters.evaluated,
            "accepted": g.counters.accepted,
            "blocked_by": g.counters.blocked_by,
            "landmarks": {
                "total": landmarks.len(),
                "confirmed": landmarks.iter().filter(|l| l.confirmed).count(),
                "rejected": landmarks.iter().filter(|l| l.state == crate::precision::LmState::Rejected).count(),
            },
            "eval": self.eval().summary(),
            "counts": self.eval(),
        });
        Ok(v)
    }

    pub fn api_ts_forecast(&self, scope: &str, axis: &str) -> Result<Value, String> {
        let label = scope.split('|').nth(1).unwrap_or("").trim();
        if label.is_empty() {
            return Err(format!("범위 '{scope}'에는 라벨이 없어 예측할 수 없습니다"));
        }
        let (history, fc) = match axis {
            "score" | "score_all" => (self.gates().series(label).to_vec(), self.gates().forecast_for(label, FORECAST_HORIZON, self.sds())),
            "margin" => (self.gates().margin_series(label).to_vec(), self.gates().forecast_margin(label, FORECAST_HORIZON)),
            other => return Err(format!("축 '{other}'은 이력을 따로 두지 않습니다. 이력이 있는 축: score, margin")),
        };
        let Some(fc) = fc else {
            let known: Vec<String> = self.gates().series_labels().into_iter().take(6).map(|(l, n)| format!("{l}({n})")).collect();
            return Err(format!("'{label}'의 {axis} 이력이 아직 없습니다. 인식이 쌓이면 예측할 수 있습니다{}", if known.is_empty() { String::new() } else { format!(" (이력 있는 라벨: {})", known.join(", ")) }));
        };
        Ok(json!({
            "scope": scope, "axis": axis, "label": label,
            "history": history.iter().map(|x| r4(*x)).collect::<Vec<_>>(),
            "point": fc.point.iter().map(|x| r4(*x)).collect::<Vec<_>>(),
            "p10": fc.p10.iter().map(|x| r4(*x)).collect::<Vec<_>>(),
            "p90": fc.p90.iter().map(|x| r4(*x)).collect::<Vec<_>>(),
            "method": fc.method,
            "weights": fc.weights.iter().map(|w| json!({"method": w.method, "weight": r4(w.weight), "rmse": r4(w.rmse)})).collect::<Vec<_>>(),
            "rmse_fused": r4(fc.rmse_fused),
            "holdout": fc.holdout,
            "notes": fc.notes,
        }))
    }

    pub fn api_log_tail(&self, after_seq: u64, limit: usize, category: Option<&str>) -> Result<Value, String> {
        let cat = category.filter(|c| !c.is_empty());
        let (lines, next) = self.log_tail(after_seq, limit.clamp(1, 2000), cat);
        Ok(json!({"lines": lines.iter().map(log_json).collect::<Vec<_>>(), "next_seq": next}))
    }

    pub fn api_model_use(&mut self, which: &str, args: &Value) -> Result<Value, String> {
        let enabled = args.get("enabled").and_then(Value::as_bool).unwrap_or(true);
        let sel = noto_ml::preferred();
        let folder = |env: &str, default: &str| -> std::path::PathBuf {
            text(args, "dir").map(std::path::PathBuf::from).or_else(|| std::env::var(env).ok().map(std::path::PathBuf::from)).unwrap_or_else(|| std::path::PathBuf::from(default))
        };
        match (which, enabled) {
            ("tips", false) => self.use_depth_stand_in(),
            ("sam", false) => self.use_region_tracker(),
            (w, true) => {
                let device = sel.backend.device(sel.ordinal).map_err(|e| format!("{} 장치를 열 수 없습니다: {e:#}", sel.backend.name()))?;
                if w == "tips" {
                    self.use_tips(&folder(noto_vision::tips::DIR_ENV, noto_vision::tips::DEFAULT_DIR), &device).map_err(|e| format!("{e:#}"))?;
                } else {
                    self.use_sam(&folder(noto_vision::sam::DIR_ENV, noto_vision::sam::DEFAULT_DIR), &device).map_err(|e| format!("{e:#}"))?;
                }
            }
            _ => {}
        }
        let mut v = self.api_models_status()?;
        v["device"] = device_json(&sel);
        Ok(v)
    }

    pub fn api_models_status(&self) -> Result<Value, String> {
        let mode = self.recognizer_mode();
        Ok(json!({
            "recognizer": mode,
            "embedding": mode != "oracle",
            "ttm": self.has_ttm(),
            "parser_embedding": self.has_embedder(),
            "devices": {"depth": self.ml.depth_device(), "locate": self.ml.locate_device(), "track": self.ml.tracker.device(), "ttm": self.gates.ttm().map(|t| noto_ml::device_label(t.device()))},
            "ml": {"depth": self.ml.depth_mode(), "locate": self.ml.locate_mode(), "track": self.ml.track_mode()},
            "stand_ins": self.stand_ins(),
            "cameras": self.cameras(),
            "stereo": self.ml.stereo_rig().map(|r| {
                let (f, b, sub) = self.sigma_camera();
                json!({"baseline_m": r3(r.baseline_m), "focal_px": r3(r.focal_px), "grid_focal_px": r3(f), "subpixel_px": r3(sub), "sigma_m_at": [[1.0, r4f(noto_core::act::material::stereo_depth_sigma(1.0, f, b, sub))], [2.0, r4f(noto_core::act::material::stereo_depth_sigma(2.0, f, b, sub))], [4.0, r4f(noto_core::act::material::stereo_depth_sigma(4.0, f, b, sub))]]})
            }),
            "tips_needed": !self.stereo(),
        }))
    }

    /// Cameras per eye: `{cameras: 1}` monocular (TIPSv2), `{cameras: 2}` stereo pair (SGBM, TIPSv2 not loaded).
    pub fn api_camera_rig(&mut self, args: &Value) -> Result<Value, String> {
        if let Some(n) = num(args, "cameras") {
            self.set_cameras(n as u8).map_err(|e| format!("{e:#}"))?;
        }
        self.api_models_status()
    }

    pub fn api_call(&mut self, cmd: &str, args: &Value) -> Result<Value, String> {
        if let Some(r) = self.api_test_call(cmd, args) {
            return r;
        }
        if let Some(r) = self.rig_call(cmd, args) {
            return r;
        }
        if let Some(r) = self.mcp_call(cmd, args) {
            return r;
        }
        match cmd {
            "units_list" => self.api_units_list(),
            "unit_frame" => self.api_unit_frame(args),
            "crew_remove" => {
                let id = num(args, "id").ok_or("id 가 필요합니다")? as u32;
                if !self.world_mut().remove_worker(id) {
                    return Err(format!("작업자 {id}가 없습니다"));
                }
                self.crew.lanes.remove(&id);
                Ok(json!({"ok": true}))
            }
            "world_body" => Ok(json!({"plan": self.cfg.body, "id": self.cfg.body.id(), "balance": self.world().balance_summary(), "limits": {
                "capacity_kg": self.world().capacity_kg(), "battery_wh": self.world().battery_capacity_j() / 3600.0,
                "max_speed_mps": self.world().drive_spec().max_speed_mps, "max_slope_deg": self.world().drive_spec().max_slope_deg, "max_step_m": self.world().drive_spec().max_step_m,
            }, "heat": self.world().heat, "ff_gains": self.world().ff_gains.iter().map(|(k, v)| (k.id().to_string(), *v)).collect::<std::collections::BTreeMap<_, _>>()})),
            "world_new" => {
                let seed = num(args, "seed").filter(|s| *s >= 0.0).ok_or("seed 는 0 이상의 숫자여야 합니다")? as u64;
                let size = num(args, "size").ok_or("size 가 필요합니다")?.max(0.0) as usize;
                let config = match args.get("config") {
                    Some(c) if !c.is_null() => Some(serde_json::from_value::<NotoConfig>(c.clone()).map_err(|e| format!("config(NotoConfig)를 읽을 수 없습니다: {e}"))?),
                    _ => None,
                };
                if let Some(body) = crate::api_test::plan_from(args)? {
                    self.cfg.body = body;
                }
                if let Some(t) = text(args, "terrain").filter(|t| !t.is_empty()) {
                    self.cfg.terrain = world_sim::TerrainKind::from_id(t).ok_or_else(|| format!("알 수 없는 지형 '{t}' (open 또는 rugged)"))?;
                }
                if let Some(n) = num(args, "cameras") {
                    self.set_cameras(n as u8).map_err(|e| format!("{e:#}"))?;
                }
                self.api_world_new(seed, size, config, text(args, "base"))
            }
            "sim_step" => self.api_sim_step(num(args, "ticks").unwrap_or(1.0).max(1.0) as u32),
            "world_frame" => {
                let (w, h) = (num(args, "width").unwrap_or(480.0) as u32, num(args, "height").unwrap_or(320.0) as u32);
                self.api_world_frame(w, h, num(args, "quality").unwrap_or(75.0) as f32, text(args, "view").unwrap_or("agent"))
            }
            "command_submit" => self.api_command_submit_rec(text(args, "text").unwrap_or(""), args.get("record").and_then(Value::as_bool).unwrap_or(false)),
            "inventory_drop" => self.api_inventory_drop(text(args, "item"), num(args, "amount").map(|a| a.max(1.0) as u32)),
            "world_reset" => self.api_world_reset(),
            "body_reset" => self.api_body_reset(),
            "world_terrain" => self.api_world_terrain(),
            "quest_list" => self.api_quest_list(),
            "perception_scan" => self.api_perception_scan(num(args, "topK").unwrap_or(8.0).max(1.0) as usize),
            "body_status" => self.api_body_status(),
            "sds_status" => self.api_sds_status(),
            "ts_forecast" => self.api_ts_forecast(text(args, "scope").unwrap_or(""), text(args, "axis").unwrap_or("")),
            "models_status" => self.api_models_status(),
            "depth_view" => self.api_depth_view(num(args, "quality").unwrap_or(80.0) as f32),
            "tips_use" => self.api_model_use("tips", args),
            "sam_use" => self.api_model_use("sam", args),
            "devices" => Ok(device_json(&noto_ml::preferred())),
            "camera_rig" => self.api_camera_rig(args),
            "crew_status" => Ok(self.crew_json()),
            "crew_add" => {
                let n = num(args, "count").unwrap_or(1.0).max(1.0) as usize;
                let form = text(args, "form").unwrap_or("mixed");
                let cameras = num(args, "cameras").map(|c| c.clamp(1.0, 2.0) as u8);
                self.crew_add(n, form, cameras)?;
                if let Some(m) = text(args, "mode").and_then(crate::crew::TrackMode::from_id) {
                    self.set_track(m);
                }
                Ok(json!({"crew": self.crew_json(), "snapshot": snapshot_json(&self.snapshot())}))
            }
            "crew_clear" => {
                let removed = self.crew_clear();
                Ok(json!({"removed": removed, "crew": self.crew_json(), "snapshot": snapshot_json(&self.snapshot())}))
            }
            "crew_mode" => {
                let m = text(args, "mode").ok_or("mode 가 필요합니다 (solo 또는 team)")?;
                let mode = crate::crew::TrackMode::from_id(m).ok_or_else(|| format!("알 수 없는 작업 방식 '{m}' (solo 또는 team)"))?;
                self.set_track(mode);
                Ok(self.crew_json())
            }
            "nn_status" => Ok(json!({"scenarios": self.nn_live.summary(), "phases": self.nn_live.phases().collect::<Vec<_>>()})),
            "physics_status" => Ok(self.api_physics_status()),
            "act_status" => Ok(self.api_act_status()),
            "world_light" => self.api_world_light(args),
            "log_tail" => self.api_log_tail(num(args, "afterSeq").unwrap_or(0.0).max(0.0) as u64, num(args, "limit").unwrap_or(200.0) as usize, text(args, "category")),
            other => Err(format!("unknown command {other}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RuntimeConfig;
    use noto_core::FingerCount;
    use std::sync::Arc;

    fn lexicon() -> String {
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/world_lexicon.json")).expect("lexicon file")
    }

    fn rt() -> Runtime {
        Runtime::new(RuntimeConfig::default(), &lexicon(), None).unwrap()
    }

    #[test]
    fn iso_time_matches_known_instants() {
        assert_eq!(iso_ms(0), "1970-01-01T00:00:00.000Z");
        assert_eq!(iso_ms(1_700_000_000_123), "2023-11-14T22:13:20.123Z");
        assert_eq!(iso_ms(951_782_400_000), "2000-02-29T00:00:00.000Z");
        assert_eq!(iso_ms(-1), "1969-12-31T23:59:59.999Z");
    }

    #[test]
    fn yaw_is_wrapped_and_flipped() {
        assert_eq!(wrap_deg(190.0), -170.0);
        assert_eq!(wrap_deg(-180.0), 180.0);
        assert_eq!(wrap_deg(-0.0), 0.0);
    }

    #[test]
    fn snapshot_has_the_shape_the_ui_reads() {
        let rt = rt();
        let v = snapshot_json(&rt.snapshot());
        assert_eq!(v["bounds"].as_array().unwrap().len(), 2);
        assert_eq!(v["agent"]["pos"].as_array().unwrap().len(), 3);
        assert!(v["agent"]["yaw_deg"].is_number());
        assert!(v["agent"]["carrying"].as_array().unwrap().is_empty());
        let w0 = &v["waypoints"][0];
        assert!(w0["id"].is_string() && w0["label_ko"].is_string() && w0["pos"].as_array().unwrap().len() == 3);
        let n0 = &v["nodes"][0];
        assert!(n0["kind"].is_string() && n0["hp"].is_number());
        let depth = v["bounds"][1].as_f64().unwrap();
        for n in v["nodes"].as_array().unwrap() {
            let y = n["pos"][1].as_f64().unwrap();
            assert!((0.0..=depth).contains(&y), "node inside the map after the flip: {y}");
        }
    }

    #[test]
    fn a_site_goes_out_through_the_same_flip_and_a_bare_world_sends_none() {
        let mut rt = rt();
        let v = snapshot_json(&rt.snapshot());
        assert!(v.get("site").is_none(), "no site, no key");
        assert!(v["drops"].as_array().unwrap().iter().all(|x| x.get("support").is_none()));
        assert!(rt.api_world_terrain().unwrap().get("curb_kind").is_none());
        let cfg = world_sim::WorldConfig { seed: 4, site: Some(world_sim::SiteCfg::default()), ..world_sim::WorldConfig::default() };
        *rt.world_mut() = world_sim::World::new(cfg).unwrap();
        let s = rt.snapshot();
        let v = snapshot_json(&s);
        let d = s.bounds.depth_m;
        let site = s.site.as_ref().unwrap();
        let h = &site.houses[0];
        let j = &v["site"]["houses"][0];
        assert_eq!(j["rect"][0].as_f64().unwrap(), r3(h.rect_m[0]));
        assert_eq!(j["rect"][1].as_f64().unwrap(), r3(d - h.rect_m[3]), "the far edge becomes the low y");
        assert!(j["rect"][1].as_f64().unwrap() < j["rect"][3].as_f64().unwrap());
        assert_eq!(j["door"][1].as_f64().unwrap(), r3(d - h.door[1]));
        assert_eq!(j["front_deg"].as_f64().unwrap(), r3(wrap_deg(-h.front_deg)));
        assert_eq!(j["stairs"]["risers"], h.stairs.risers);
        assert!(v["site"]["warehouse"]["shelves"].as_u64().unwrap() > 0);
        assert!(v["site"]["fixtures"].as_array().unwrap().iter().any(|f| f["kind"] == "shelf" && f["levels_m"].as_array().unwrap().len() == 3));
        let raised: Vec<&Value> = v["drops"].as_array().unwrap().iter().filter(|x| x.get("support").is_some()).collect();
        assert!(!raised.is_empty());
        for x in raised {
            let id = x["id"].as_str().unwrap();
            let dv = s.drops.iter().find(|q| q.id == id).unwrap();
            assert_eq!(x["pos"][2].as_f64().unwrap(), r3(dv.pos.y));
            assert!(x["support"]["label_ko"].is_string() && x["support"]["on"].is_string());
        }
        let t = rt.api_world_terrain().unwrap();
        assert_eq!(t["curb_kind"].as_array().unwrap().len(), t["heights_cm"].as_array().unwrap().len());
    }

    #[test]
    fn a_parcel_fetch_says_where_to_look_and_that_the_base_camp_was_implied() {
        let mut rt = rt();
        let v = rt.api_command_submit("창고 선반에서 택배 가져와").unwrap();
        assert_eq!(v["ok"], true, "{v}");
        assert_eq!(v["parsed"]["destination"]["id"], "base_camp");
        assert_eq!(v["parsed"]["destination"]["implied"], true);
        let origin: Vec<&str> = v["parsed"]["targets"][0]["origin"].as_array().unwrap().iter().map(|o| o["id"].as_str().unwrap()).collect();
        assert_eq!(origin, vec!["warehouse", "shelf"]);
        let v = rt.api_command_submit("택배를 창고로 가져와").unwrap();
        assert!(v["parsed"]["destination"].get("implied").is_none());
        assert!(v["parsed"]["targets"][0].get("origin").is_none());
    }

    #[test]
    fn the_flip_keeps_the_map_unmirrored() {
        let mut rt = rt();
        let (ax, az) = (rt.world().agent.x, rt.world().agent.z);
        rt.world_mut().agent.yaw = 0.0;
        let d = rt.snapshot().bounds.depth_m;
        let p = pos(&Pos3 { x: ax, y: 0.0, z: az + 2.0 }, d);
        let me = pos(&Pos3 { x: ax, y: 0.0, z: az }, d);
        assert!(p[1].as_f64().unwrap() < me[1].as_f64().unwrap());
        let v = snapshot_json(&rt.snapshot());
        assert_eq!(v["agent"]["yaw_deg"].as_f64().unwrap(), 0.0);
    }

    #[test]
    fn world_new_validates_and_applies_the_hand() {
        let mut rt = rt();
        assert!(rt.api_world_new(1, 8, None, None).is_err());
        assert!(rt.api_world_new(1, 64, None, Some("hoverboard")).is_err());
        let two = NotoConfig::preset(FingerCount::Two);
        let v = rt.api_world_new(5, 48, Some(two), Some("track")).unwrap();
        assert_eq!(v["seed"], 5);
        assert_eq!(v["agent"]["fingers"], 2);
        assert_eq!(v["agent"]["base"], "track");
        assert_eq!(v["bounds"][0].as_f64().unwrap(), 24.0);
        assert!(rt.api_world_new(9, 8, None, None).is_err());
        assert_eq!(rt.snapshot().seed, 5);
    }

    #[test]
    fn command_submit_answers_what_the_command_bar_prints() {
        let mut rt = rt();
        let v = rt.api_command_submit("철광석 2개 캐서 창고로 가져와").unwrap();
        assert_eq!(v["ok"], true);
        assert_eq!(v["parsed"]["action"], "mine_and_deliver");
        assert_eq!(v["parsed"]["item_id"], "iron_ore");
        assert_eq!(v["parsed"]["item_label_ko"], "철광석");
        assert_eq!(v["parsed"]["count"], 2);
        assert_eq!(v["parsed"]["destination"]["kind"], "waypoint");
        assert_eq!(v["parsed"]["destination"]["id"], "warehouse");
        assert!(v["quest_id"].is_string());
        let trace = v["trace"].as_array().unwrap();
        assert_eq!(trace.first().unwrap()["stage"], "normalize");
        assert_eq!(trace.last().unwrap()["stage"], "assign");
        assert!(trace.iter().any(|r| r["role"] == "아이템" && r["accepted"] == true));
        assert!(rt.api_quest_list().unwrap().as_array().unwrap().len() == 1);

        let bad = rt.api_command_submit("춤춰").unwrap();
        assert_eq!(bad["ok"], false);
        assert!(bad["error"].is_string());
        assert_eq!(bad["parsed"]["action"], "unknown");
        assert!(bad["quest_id"].is_null());
    }

    #[test]
    fn several_targets_are_summarised_for_the_one_line_display() {
        let mut rt = rt();
        let v = rt.api_command_submit("원목 2개랑 석탄 1개 창고로 가져와").unwrap();
        assert_eq!(v["ok"], true, "{v}");
        assert_eq!(v["parsed"]["targets"].as_array().unwrap().len(), 2);
        assert!(v["parsed"]["count"].is_null());
        assert!(v["parsed"]["item_label_ko"].as_str().unwrap().contains(" + "));
    }

    #[test]
    fn a_run_through_the_api_moves_the_quest_and_the_log() {
        let mut rt = rt();
        rt.api_command_submit("돌 1개 캐서 베이스캠프로 가져와").unwrap();
        let mut last = Value::Null;
        for _ in 0..40 {
            last = rt.api_sim_step(25).unwrap();
        }
        let q = &last["quests"][0];
        assert!(matches!(q["state"].as_str(), Some("active" | "done")), "{q}");
        assert!(last["snapshot"]["tick"].as_u64().unwrap() >= 1000);
        assert!(last["body"]["parts"].as_array().unwrap().len() == 7);
        let tail = rt.api_log_tail(0, 5, Some("quest")).unwrap();
        let lines = tail["lines"].as_array().unwrap();
        assert!(!lines.is_empty() && lines.iter().all(|l| l["cat"] == "quest"));
        assert!(lines[0]["ts"].as_str().unwrap().contains('T'));
        let next = tail["next_seq"].as_u64().unwrap();
        let again = rt.api_log_tail(next, 5, Some("quest")).unwrap();
        assert!(again["next_seq"].as_u64().unwrap() >= next);
    }

    #[test]
    fn frames_and_scans_carry_boxes_and_reasons() {
        let mut rt = rt();
        let scan = rt.api_perception_scan(5).unwrap();
        assert_eq!(scan["mode"], "oracle");
        let labels = scan["labels"].as_array().unwrap();
        assert!(labels.len() <= 5);
        for l in labels {
            assert!(l["reason"].as_str().unwrap().len() > 3);
            assert!(l["score"].as_f64().unwrap() <= 1.0);
        }
        let f = rt.api_world_frame(320, 200, 70.0, "agent").unwrap();
        assert_eq!(f["width"], 320);
        assert_eq!(f["height"], 200);
        let b64 = f["webp_base64"].as_str().unwrap();
        let bytes = STANDARD.decode(b64).unwrap();
        assert_eq!(&bytes[..4], b"RIFF");
        for d in f["detections"].as_array().unwrap() {
            let b = d["bbox"].as_array().unwrap();
            assert!(b.len() == 4 && b.iter().all(|x| (0.0..=1.0).contains(&x.as_f64().unwrap())));
            assert!(d["label_id"].is_string() && d["reason"].is_string());
        }
        let top = rt.api_world_frame(300, 256, 70.0, "top").unwrap();
        assert_eq!((top["width"].as_u64(), top["height"].as_u64()), (Some(300), Some(256)));
        assert_eq!(top["units"][0]["id"], 1, "{}", top["units"]);
        assert!((0.0..=1.0).contains(&top["units"][0]["u"].as_f64().unwrap()));
    }

    #[test]
    fn a_scan_through_another_gaze_draws_no_boxes_on_the_first_person_frame() {
        let mut rt = rt();
        let boxes = |rt: &mut Runtime| rt.api_world_frame(320, 200, 70.0, "agent").unwrap()["detections"].as_array().map_or(0, Vec::len);
        let ahead = rt.scan_now("test").unwrap().judged.len();
        assert!(ahead > 0, "the oracle sees something ahead");
        assert_eq!(boxes(&mut rt), ahead, "a scan through the camera as it is draws its boxes");
        rt.scan_gaze("glance", 40.0, 0.0).unwrap();
        assert_eq!(boxes(&mut rt), 0, "a glance 40° aside put its boxes on the wrong part of the frame");
        rt.scan_gaze("glance", 0.0, 30.0).unwrap();
        assert_eq!(boxes(&mut rt), 0, "so did a look 30° down");
    }

    #[test]
    fn forecast_asks_for_a_history_that_exists() {
        let mut rt = rt();
        assert!(rt.api_ts_forecast("vision|tree|", "score").is_err());
        assert!(rt.api_ts_forecast("motion|wheel|move", "route_time_s").unwrap_err().contains("이력"));
        for _ in 0..40 {
            rt.scan_now("test").unwrap();
            rt.tick();
        }
        let labels = rt.gates().series_labels();
        let (label, n) = labels.first().cloned().expect("a label has a history after forty scans");
        assert!(n >= 3);
        let v = rt.api_ts_forecast(&format!("vision|{label}|"), "score").unwrap();
        assert_eq!(v["history"].as_array().unwrap().len(), n);
        assert_eq!(v["point"].as_array().unwrap().len(), v["p10"].as_array().unwrap().len());
        assert_eq!(v["point"].as_array().unwrap().len(), v["p90"].as_array().unwrap().len());
        let m = rt.api_ts_forecast(&format!("vision|{label}|"), "margin").unwrap();
        assert_eq!(m["history"].as_array().unwrap().len(), n);
    }

    #[test]
    fn sds_status_tells_what_the_gates_did() {
        let mut rt = rt();
        for _ in 0..20 {
            rt.scan_now("test").unwrap();
        }
        let v = rt.api_sds_status().unwrap();
        assert!(v["ledgers"].is_array() && v["axes"].is_array());
        assert!(v["precision"]["evaluated"].as_u64().unwrap() > 0);
        assert!(v["precision"]["eval"].is_object());
        let fc = v["forecastable"].as_array().unwrap();
        assert!(!fc.is_empty(), "twenty scans leave a history to forecast");
        let first = &fc[0];
        let r = rt.api_ts_forecast(first["scope"].as_str().unwrap(), first["axis"].as_str().unwrap());
        assert!(r.is_ok(), "every listed pair can be forecast: {r:?}");
    }

    #[test]
    fn the_dispatcher_reads_the_ui_argument_names() {
        let mut rt = rt();
        let config = serde_json::to_value(NotoConfig::preset(FingerCount::Three)).unwrap();
        let v = rt.api_call("world_new", &json!({"seed": 3, "size": 64, "config": config, "base": "mecanum"})).unwrap();
        assert_eq!(v["agent"]["fingers"], 3);
        assert_eq!(v["agent"]["base"], "mecanum");
        assert!(rt.api_call("sim_step", &json!({"ticks": 3})).is_ok());
        assert!(rt.api_call("log_tail", &json!({"afterSeq": 0, "limit": 3, "category": null})).is_ok());
        assert!(rt.api_call("perception_scan", &json!({"topK": 3})).is_ok());
        let m = rt.api_call("models_status", &json!({})).unwrap();
        assert_eq!((m["recognizer"].as_str(), m["embedding"].as_bool(), m["ttm"].as_bool()), (Some("oracle"), Some(false), Some(false)));
        assert!(rt.api_call("world_new", &json!({"seed": -1, "size": 64})).is_err());
        assert_eq!(rt.api_call("nope", &json!({})).unwrap_err(), "unknown command nope");
    }

    #[test]
    fn granite_ttm_joins_the_fusion_once_it_is_attached() {
        use candle_core::{DType, Device};
        use candle_nn::{VarBuilder, VarMap};
        use noto_ts::ttm::{TtmConfig, TtmForecaster};

        let cfg: TtmConfig = serde_json::from_value(json!({
            "context_length": 52, "prediction_length": 16, "patch_length": 4, "patch_stride": 4, "num_input_channels": 1,
            "d_model": 16, "num_layers": 2, "expansion_factor": 2, "mode": "common_channel", "gated_attn": true,
            "norm_mlp": "LayerNorm", "scaling": "std", "decoder_num_layers": 2, "decoder_d_model": 8,
            "decoder_mode": "common_channel", "use_decoder": true, "adaptive_patching_levels": 0
        }))
        .unwrap();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let ttm = Arc::new(TtmForecaster::from_varbuilder(cfg, vb, std::path::PathBuf::from("zeros")).expect("model builds"));

        let mut rt = rt();
        for _ in 0..60 {
            rt.scan_now("test").unwrap();
            rt.tick();
        }
        let (label, n) = rt.gates().series_labels().first().cloned().unwrap();
        assert!(n >= 12);
        let scope = format!("vision|{label}|");
        let before = rt.api_ts_forecast(&scope, "score").unwrap();
        assert!(!before["method"].as_str().unwrap().contains("ttm"), "{}", before["method"]);
        assert!(!rt.has_ttm());

        rt.attach_ttm(Some(Arc::clone(&ttm)));
        assert!(rt.has_ttm());
        let after = rt.api_ts_forecast(&scope, "score").unwrap();
        let weights = after["weights"].as_array().unwrap();
        assert!(weights.iter().any(|w| w["method"] == "ttm"), "ttm takes part: {after}");
        assert_eq!(after["point"].as_array().unwrap().len(), 16);
        assert!(after["point"].as_array().unwrap().iter().all(|v| v.as_f64().is_some_and(|x| x.is_finite())));
        let ts = rt.api_log_tail(0, 50, Some("ts")).unwrap();
        assert!(ts["lines"].as_array().unwrap().iter().any(|l| l["msg"].as_str().unwrap().contains("Granite TTM 연결")));
        rt.attach_ttm(None);
        assert!(!rt.has_ttm());
        let again = rt.api_ts_forecast(&scope, "score").unwrap();
        assert!(!again["weights"].as_array().unwrap().iter().any(|w| w["method"] == "ttm"));
    }
}
