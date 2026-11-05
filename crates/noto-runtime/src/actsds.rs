use noto_core::act::{drive, gait, ActKind, ActRollout};
use noto_core::actbench as ab;
use noto_core::balance::{reference, RefCtx, Source, G};
use noto_core::kin;
use noto_core::parts::Segment;
use noto_core::BaseKind;
use noto_sds::{Scope, SdsStore, Track};
use world_sim::{TimedEvent, WorldEvent};

use crate::record::{capture_keyframe, KeyEvent};
use crate::runtime::Runtime;

pub const KEY_KINDS: [&str; 36] = [
    "bend_bottom", "contact", "lift_off", "hold", "slip", "pivot", "drop", "stowed", "release", "impact", "stall", "miss", "reach_fail", "foot_slip", "fall", "breakaway", "turn_slide", "push_lost", "collision", "arrive", "stuck", "wheel_slip", "bump", "tip", "hit", "edge_fall", "yield", "co_drop", "overload_share",
    "incipient_slip", "heel_off", "toe_push", "fork_down", "fork_in", "fork_up", "fork_out",
];

pub const MOTION_EVENTS: [&str; 6] = ["foot_slip", "fall", "wheel_slip", "stuck", "collision", "push_lost"];

pub(crate) fn record_forecast(sds: &mut SdsStore, scope: &Scope, axis: &str, happened: bool, p: f32) {
    let y = if happened { 1.0 } else { 0.0 };
    let p = (p as f64).clamp(0.0, 1.0);
    sds.record_baseline(scope, &format!("{axis}:brier"), (y - p) * (y - p));
    let bin = ((p * 10.0).floor() as usize).min(9);
    sds.record_agreement(scope, &format!("{axis}:bin{bin}"), happened);
    sds.record_baseline(scope, &format!("{axis}:bin{bin}:p"), p);
}

fn code_of(reason: Option<&str>) -> String {
    let r = reason.unwrap_or("failed").split(':').next().unwrap_or("failed").trim();
    let c: String = r.chars().take(32).map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' }).collect();
    if c.is_empty() {
        "failed".into()
    } else {
        c
    }
}
pub const KEY_GAP_S: f64 = 0.1;

fn ts_reference(action: ActKind, metric: &str) -> Option<(f32, f32)> {
    match (action, metric) {
        (ActKind::Grasp, "preload_s") => Some(ab::PRELOAD_S),
        (ActKind::Grasp, "loading_s") => Some(ab::LOADING_S),
        (ActKind::Grasp, "gf_lead_s") => Some(ab::GF_LEAD_S),
        (ActKind::Grasp, "gf_lf_corr") => Some(ab::GF_LF_CORR),
        (ActKind::Grasp, "safety_margin") => Some(ab::SAFETY_MARGIN),
        (ActKind::Release, "place_err_m") => Some((ab::PLACE_ERR_MM.0 / 1000.0, ab::PLACE_ERR_MM.1 / 1000.0)),
        _ => None,
    }
}

impl Runtime {
    pub(crate) fn record_act_keyframes(&mut self, events: &[TimedEvent]) {
        let Some(rec) = self.recording.as_ref() else { return };
        let last = rec.keyframes.iter().rev().find(|k| k.event.is_some()).map_or(f64::NEG_INFINITY, |k| k.time_s);
        let n = rec.keyframes.len();
        if n >= self.test_cfg.record.max_keyframes {
            return;
        }
        let moments: Vec<(String, String, String, bool, Vec<String>)> = events
            .iter()
            .filter_map(|e| match &e.event {
                WorldEvent::Act { act, kind, label, ok, parts, .. } if KEY_KINDS.contains(&kind.as_str()) => Some((act.clone(), kind.clone(), label.clone(), *ok, parts.clone())),
                _ => None,
            })
            .collect();
        let Some((act, kind, label, ok, parts)) = moments.iter().find(|m| !m.3).or(moments.first()).cloned() else {
            return;
        };
        if ok && self.world.time_s - last < KEY_GAP_S {
            return;
        }
        let quality = self.test_cfg.record.quality;
        if let Ok(mut kf) = capture_keyframe(&self.world, &self.cam, quality, n as u32, self.stereo_baseline()) {
            kf.step = self.current_step_label();
            kf.event = Some(KeyEvent { act: act.clone(), kind: kind.clone(), label, ok, parts });
            let value = events.iter().find_map(|e| match &e.event {
                WorldEvent::Act { kind: k, value, .. } if *k == kind => *value,
                _ => None,
            });
            if let Some(v) = value {
                let scope = Scope::new(Track::Act, &format!("{act}:{kind}"), "");
                if let Some((m, sd, _)) = self.sds.adaptive_baseline(&scope, "value") {
                    kf.sds.insert("act".into(), ((v as f64 - m) / sd.max(1e-6)) as f32);
                }
                self.sds.record_baseline(&scope, "value", v as f64);
            }
            if let Some(r) = self.recording.as_mut() {
                // Only the game pass's keyframes are analysed afterwards; a moment in another pass keeps no truth.
                if r.pass != "game" {
                    kf.truth.shed_pixels();
                }
                r.keyframes.push(kf);
            }
        }
    }

    pub(crate) fn record_act_sds(&mut self, events: &[TimedEvent]) {
        self.record_motion_sds(events);
        let done = events.iter().any(|e| matches!(&e.event, WorldEvent::ActionDone { action, .. } if action.starts_with("pick_up") || action.starts_with("drop") || action.starts_with("deliver") || action.starts_with("push")));
        if !done {
            return;
        }
        let Some(p) = self.world.last_act.clone() else { return };
        let r = &p.rollout;
        let action = r.kind;
        let preset = noto_core::bench::form_id(&self.world.body);
        let what = p.object.map(|o| o.0.id().to_string()).unwrap_or_default();
        let s1 = Scope::new(Track::Act, action.id(), &preset);
        self.sds.record_agreement(&s1, "ok", r.outcome.ok);
        self.sds.record_transition(&s1, "attempt", r.outcome.code.unwrap_or("ok"));
        if let Some(pred) = r.metric("p_success") {
            record_forecast(&mut self.sds, &s1, "p_success", r.outcome.ok, pred);
        }
        let s2 = Scope::new(Track::Act, &format!("{}:ts", action.id()), &what);
        for (k, v) in &r.metrics {
            if !v.is_finite() {
                continue;
            }
            match ts_reference(action, k) {
                Some((rf, tol)) if r.outcome.ok => self.sds.record_reference(&s2, k, *v as f64, rf as f64, tol as f64),
                _ => self.sds.record_baseline(&s2, k, *v as f64),
            }
        }
        // What the pads felt of the item goes back into the next grasp of it (`relay`).
        if let (ActKind::Grasp, Some((item, _)), Some(pads)) = (action, p.object, p.pads) {
            let table_planned = r.metric("mu_table").map_or(true, |t| r.metric("mu_est").is_some_and(|e| (e - t).abs() < 1e-6));
            crate::relay::record_grasp_feel(&mut self.sds, item.id(), pads, table_planned, r.metric("incipient_slips").unwrap_or(0.0), r.metric("mu_seen"));
            self.relay_grip(item, pads);
        }
        if action == ActKind::Push {
            self.record_push_layers(r);
            return;
        }
        let bottom = r.samples.iter().min_by(|a, b| {
            let za = kin::solve_fast(&self.world.body, &a.pose, &a.loads).hands[0][2];
            let zb = kin::solve_fast(&self.world.body, &b.pose, &b.loads).hands[0][2];
            za.total_cmp(&zb)
        });
        if let Some(b) = bottom {
            let k = kin::solve_fast(&self.world.body, &b.pose, &b.loads);
            let s3 = Scope::new(Track::Act, &format!("{}>part", action.id()), b.phase);
            for j in k.joints.iter().filter(|j| !j.structural) {
                self.sds.record_baseline(&s3, &format!("{}:ratio", j.name), j.ratio as f64);
            }
            let trunk = k.pose.trunk_deg();
            self.sds.record_reference(&s3, "trunk_deg", trunk as f64, ab::LIFT_TRUNK_DEG.0 as f64, ab::LIFT_TRUNK_DEG.1 as f64);
            if self.world.body.legged() {
                if let Some(h) = k.joint("hip_pitch_r").and_then(|j| j.angle_deg) {
                    self.sds.record_reference(&s3, "hip_flex_deg", h as f64, ab::LIFT_HIP_DEG.0 as f64, ab::LIFT_HIP_DEG.1 as f64);
                }
                if let Some(n) = k.joint("knee_r").and_then(|j| j.angle_deg) {
                    self.sds.record_reference(&s3, "knee_flex_deg", n as f64, ab::LIFT_KNEE_DEG.0 as f64, ab::LIFT_KNEE_DEG.1 as f64);
                }
            }
            let s4 = Scope::new(Track::Act, &format!("nn:{}", action.id()), "foot");
            match reference(Source::ArmLift, Segment::Foot, "margin", &RefCtx::default()) {
                Some(rf) => self.sds.record_reference(&s4, "margin", k.margin as f64, rf.value as f64, rf.tolerance as f64),
                None => self.sds.record_baseline(&s4, "margin", k.margin as f64),
            }
            let s4n = Scope::new(Track::Act, &format!("nn:{}", action.id()), "neck");
            self.sds.record_baseline(&s4n, "camera_pitch_deg", (k.pose.trunk_deg() + k.pose.neck_deg) as f64);
        }
    }

    fn record_push_layers(&mut self, r: &ActRollout) {
        let Some(key) = r.samples.iter().max_by(|a, b| a.peak_ratio.total_cmp(&b.peak_ratio)) else { return };
        let k = kin::solve_fast(&self.world.body, &key.pose, &key.loads);
        let s3 = Scope::new(Track::Act, "push>part", key.phase);
        for j in k.joints.iter().filter(|j| !j.structural) {
            self.sds.record_baseline(&s3, &format!("{}:ratio", j.name), j.ratio as f64);
        }
        let props = *self.world.ground_here().surface().props();
        let slope = self.world.posture_now().0.slope_deg;
        let st = drive::state(&self.world.body, &props, slope, 0.0, self.world.carried_kg(), r.metric("push_peak_n").unwrap_or(0.0));
        let part = if self.world.body.base == BaseKind::Track { "track" } else { "wheel" };
        let s4 = Scope::new(Track::Act, "nn:push", part);
        self.sds.record_baseline(&s4, "bumper>wheel:front_share", (st.front_n / (st.front_n + st.rear_n).max(1e-3)) as f64);
        self.sds.record_baseline(&s4, "bumper>wheel:slip", st.slip as f64);
        self.sds.record_baseline(&s4, "need_over_limit", (st.need_n / st.limit_n.max(1e-3)) as f64);
        self.sds.record_baseline(&Scope::new(Track::Act, "nn:push", "base"), "margin", k.margin as f64);
    }

    fn record_motion_sds(&mut self, events: &[TimedEvent]) {
        let form = noto_core::bench::form_id(&self.world.body);
        for e in events {
            match &e.event {
                WorldEvent::ActionDone { action, ok, reason } if action.starts_with("move_to") => {
                    if reason.as_deref() == Some("cancelled") {
                        continue;
                    }
                    let legged = self.world.body.base == BaseKind::HumanFoot;
                    let s1 = Scope::new(Track::Act, if legged { "walk" } else { "drive" }, &form);
                    self.sds.record_agreement(&s1, "ok", *ok);
                    self.sds.record_transition(&s1, "attempt", &if *ok { "ok".to_string() } else { code_of(reason.as_deref()) });
                    if legged {
                        self.record_walk_layers();
                    } else {
                        self.record_drive_layers();
                    }
                }
                WorldEvent::Act { act, kind, ok, value, .. } if MOTION_EVENTS.contains(&kind.as_str()) => {
                    let ground = self.world.ground_here().id();
                    self.sds.record_agreement(&Scope::new(Track::Act, &format!("event:{kind}"), &form), "ok", *ok);
                    if let Some(v) = value {
                        self.sds.record_baseline(&Scope::new(Track::Act, &format!("event:{kind}:ts"), ground), "value", *v as f64);
                    }
                    if act == "walk" && matches!(kind.as_str(), "foot_slip" | "fall") {
                        self.sds.record_transition(&Scope::new(Track::Act, "walk", &form), "slip", if kind == "fall" { "fall" } else { "recover" });
                    }
                }
                _ => {}
            }
        }
    }

    fn record_walk_layers(&mut self) {
        let Some((g, ground, _)) = self.world.walk.plan.clone() else { return };
        let body = &self.world.body;
        let cargo = self.world.carried_kg();
        let s2 = Scope::new(Track::Act, "walk:ts", ground.id());
        let leg = body.dims.hip_m;
        for (k, v) in [("speed_mps", g.speed), ("step_m", g.step_m), ("step_s", g.step_s), ("ds_frac", g.ds_frac), ("rcof", g.rcof), ("froude", g.speed * g.speed / (G * leg))] {
            self.sds.record_baseline(&s2, k, v as f64);
        }
        let (pose, loads, phase) = gait::pose_at(body, &g, 0, 0.5, cargo);
        let k = kin::solve_fast(body, &pose, &loads);
        let s3 = Scope::new(Track::Act, "walk>part", phase);
        for j in k.joints.iter().filter(|j| !j.structural) {
            self.sds.record_baseline(&s3, &format!("{}:ratio", j.name), j.ratio as f64);
        }
        let mu = noto_core::surface::available_mu(body.base, ground.surface().props());
        let s4 = Scope::new(Track::Act, "nn:walk", "foot");
        self.sds.record_baseline(&s4, "foot>sole:rcof_over_mu", (g.rcof / mu.max(1e-3)) as f64);
        self.sds.record_baseline(&s4, "margin", k.margin as f64);
    }

    fn record_drive_layers(&mut self) {
        let ground = self.world.ground_here();
        let props = *ground.surface().props();
        let slope = self.world.posture_now().0.slope_deg;
        let st = drive::state(&self.world.body, &props, slope, 0.0, self.world.carried_kg(), 0.0);
        let part = if self.world.body.base == BaseKind::Track { "track" } else { "wheel" };
        let s2 = Scope::new(Track::Act, "drive:ts", ground.id());
        self.sds.record_baseline(&s2, "slip_cruise", st.slip as f64);
        self.sds.record_baseline(&s2, "speed_factor", st.speed_factor as f64);
        let s3 = Scope::new(Track::Act, "drive>part", part);
        self.sds.record_baseline(&s3, "front_share_cruise", (st.front_n / (st.front_n + st.rear_n).max(1e-3)) as f64);
        if let Some(p) = st.pressure_kpa {
            self.sds.record_baseline(&s3, "track_pressure_max_kpa", p[0].max(p[1]) as f64);
        }
        let s4 = Scope::new(Track::Act, "nn:drive", part);
        self.sds.record_baseline(&s4, "slope>drive:need_over_limit", (st.need_n / st.limit_n.max(1e-3)) as f64);
        self.sds.record_baseline(&s4, "margin", st.margin as f64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_forecast_is_scored_by_brier_and_reliability_bins() {
        let mut sds = SdsStore::memory();
        let s = Scope::new(Track::Act, "grasp", "mid:humanfoot:f5");
        for (ok, p) in [(true, 0.95), (true, 0.92), (false, 0.97), (true, 0.35), (false, 0.31), (true, 0.93)] {
            record_forecast(&mut sds, &s, "p_success", ok, p);
        }
        let (brier, _, n) = sds.exact_baseline(&s, "p_success:brier").unwrap();
        assert_eq!(n, 6);
        let want = [(1.0f64, 0.95f64), (1.0, 0.92), (0.0, 0.97), (1.0, 0.35), (0.0, 0.31), (1.0, 0.93)].iter().map(|(y, p)| (y - p).powi(2)).sum::<f64>() / 6.0;
        assert!((brier - want).abs() < 1e-6, "{brier} vs {want}");
        let (p9, _, n9) = sds.exact_baseline(&s, "p_success:bin9:p").unwrap();
        assert_eq!(n9, 4);
        assert!((p9 - 0.9425).abs() < 1e-6, "{p9}");
        let st = sds.status(200);
        let rate = st.axes.iter().find(|a| a.scope == "act|grasp|mid:humanfoot:f5" && a.axis == "p_success:bin9").unwrap();
        assert!((rate.mean - 0.75).abs() < 1e-9);
        assert!(sds.exact_baseline(&s, "p_success:bin3:p").is_some());
    }

    fn runtime_on(base: BaseKind) -> Runtime {
        let lex = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/world_lexicon.json")).expect("lexicon file");
        let mut cfg = crate::config::RuntimeConfig { parcels: Some(0), ..crate::config::RuntimeConfig::default() };
        cfg.noto.base = base;
        Runtime::new(cfg, &lex, None).expect("runtime")
    }

    #[test]
    fn a_finished_move_records_its_layers_by_base() {
        for (base, act, ts, part, nn) in [
            (BaseKind::HumanFoot, "walk", ["speed_mps", "step_m", "rcof", "froude"], "walk>part|", ("act|nn:walk|foot", "foot>sole:rcof_over_mu")),
            (BaseKind::Wheel, "drive", ["slip_cruise", "speed_factor", "slip_cruise", "speed_factor"], "drive>part|wheel", ("act|nn:drive|wheel", "slope>drive:need_over_limit")),
        ] {
            let mut rt = runtime_on(base);
            let (x, z, yaw) = (rt.world().agent.x, rt.world().agent.z, rt.world().agent.yaw);
            let go = rt.to_world(world_sim::Action::MoveTo { x: x + 3.0 * yaw.cos(), z: z + 3.0 * yaw.sin(), reach_m: 0.3 });
            rt.world_mut().command(go).expect("the move starts");
            let mut n = 0;
            while rt.world().busy() && n < 3000 {
                rt.tick();
                n += 1;
            }
            let form = noto_core::bench::form_id(&rt.world().body);
            let st = rt.sds().status(100_000);
            let has = |scope: &str, axis: &str| st.axes.iter().any(|a| a.scope.starts_with(scope) && a.axis == axis);
            assert!(has(&format!("act|{act}|{form}"), "ok"), "{act}: the outcome by form");
            for axis in ts {
                assert!(has(&format!("act|{act}:ts|"), axis), "{act}: {axis} by ground");
            }
            assert!(st.axes.iter().any(|a| a.scope.starts_with(&format!("act|{part}")) && a.n > 0), "{act}: the joints or axles");
            assert!(has(nn.0, nn.1) && has(nn.0, "margin"), "{act}: the coupling");
            let other = if act == "walk" { "drive" } else { "walk" };
            assert!(!st.axes.iter().any(|a| a.scope.starts_with(&format!("act|{other}"))), "{act} is not also a {other}");
        }
    }

    #[test]
    fn failure_reasons_become_short_codes() {
        assert_eq!(code_of(Some("fall: 오른발이 미끄러져 넘어짐")), "fall");
        assert_eq!(code_of(Some("battery_depleted")), "battery_depleted");
        assert_eq!(code_of(None), "failed");
        assert_eq!(code_of(Some(": x")), "failed");
        assert!(KEY_KINDS.iter().all(|k| !k.is_empty()) && KEY_KINDS.len() == 36);
        assert!(MOTION_EVENTS.iter().all(|k| KEY_KINDS.contains(k)), "every counted moment is a keyframe moment");
    }
}
