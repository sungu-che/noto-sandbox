use noto_core::kin::{self as hkin, KinPose};
use noto_core::surface::Surface;
use serde::Serialize;

use super::World;
use crate::render::{Camera, Exposure, View};
use crate::rng::hash2;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct EyeState {
    pub travelled_m: f32,
    pub yaw_rate: f32,
}

impl EyeState {
    pub(super) fn advance(&mut self, before: (f32, f32, f32), after: (f32, f32, f32), dt: f32) {
        self.travelled_m += (after.0 - before.0).hypot(after.1 - before.1);
        let mut d = (after.2 - before.2) % std::f32::consts::TAU;
        if d > std::f32::consts::PI {
            d -= std::f32::consts::TAU;
        } else if d < -std::f32::consts::PI {
            d += std::f32::consts::TAU;
        }
        self.yaw_rate = if dt > 0.0 { d / dt } else { 0.0 };
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct Shake {
    pub source: &'static str,
    pub dz_m: f32,
    pub pitch: f32,
    pub roll: f32,
    pub rate: [f32; 3],
}

impl Shake {
    fn still() -> Shake {
        Shake { source: "still", ..Shake::default() }
    }

    fn size(&self) -> f32 {
        self.pitch.abs() + self.roll.abs() + self.dz_m.abs()
    }

    fn add(self, o: Shake) -> Shake {
        let source = if o.size() > self.size() || self.source == "still" { o.source } else { self.source };
        Shake {
            source,
            dz_m: self.dz_m + o.dz_m,
            pitch: self.pitch + o.pitch,
            roll: self.roll + o.roll,
            rate: [self.rate[0] + o.rate[0], self.rate[1] + o.rate[1], self.rate[2] + o.rate[2]],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct HeadView {
    pub view: View,
    pub exposure: Exposure,
    pub est: View,
    pub shake: Shake,
    pub eye_m: f32,
    pub head_pitch_deg: f32,
    pub pan_deg: f32,
    pub blur_px: f32,
    pub tilt_deg: [f32; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct SensorTiming {
    pub rgb: &'static str,
    pub exposure_s: f32,
    pub readout_s: f32,
    pub gyro_arw_deg: f32,
}

pub const SCENE_LUX: f32 = 10_000.0;
pub const RGB_READOUT_S: f32 = 0.0167;
pub const EXPOSURE_MAX_S: f32 = 1.0 / 30.0;

pub fn exposure_s(lux: f32) -> f32 {
    let luminance = 0.18 * lux.max(1.0) / std::f32::consts::PI;
    (0.5 / luminance).min(EXPOSURE_MAX_S)
}

pub fn sensor_timing(sensor: &str) -> SensorTiming {
    sensor_timing_at(sensor, SCENE_LUX)
}

pub fn sensor_timing_at(sensor: &str, lux: f32) -> SensorTiming {
    let exposure = exposure_s(lux);
    if sensor.contains("OAK") {
        SensorTiming { rgb: "IMX214 (rolling)", exposure_s: exposure, readout_s: RGB_READOUT_S, gyro_arw_deg: 0.010 }
    } else {
        SensorTiming { rgb: "OV2740 (rolling)", exposure_s: exposure, readout_s: RGB_READOUT_S, gyro_arw_deg: 0.014 }
    }
}

pub const SWAY_PITCH_PP_DEG: f32 = 4.0;
pub const SWAY_ROLL_PP_DEG: f32 = 4.0;
pub const STRIKE_PEAK_MPS2: f32 = 0.5 * 9.81;
pub const STRIKE_PULSE_S: f32 = 0.033;
pub const MOUNT_HZ: f32 = 13.4;
pub const MOUNT_ZETA: f32 = 0.30;
pub const GYRATION_SHARE: f32 = 0.25;
pub const IMU_WINDOW_S: f32 = 1.0;

pub fn strike_scale(ground: Surface) -> f32 {
    match ground {
        Surface::Firm | Surface::Rock => 1.00,
        Surface::Gravel => 0.92,
        Surface::Dirt => 0.75,
        Surface::WetSand => 0.68,
        Surface::Grass => 0.36,
        Surface::Sand | Surface::Water => 0.17,
    }
}

pub fn roughness_gd(ground: Surface) -> f32 {
    let class = |k: i32| 16.0e-6 * 4f32.powi(k);
    match ground {
        Surface::Firm | Surface::WetSand => class(1),
        Surface::Gravel | Surface::Grass | Surface::Water => class(2),
        Surface::Dirt | Surface::Sand => class(3),
        Surface::Rock => class(5),
    }
}

const N0: f32 = 0.1;
const N_LO: f32 = 0.1;
const N_HI: f32 = 50.0;
const N_WAVES: usize = 48;

pub fn profile(seed: u64, gd: f32, side: u32, s: f32, bridge_m: f32) -> (f32, f32) {
    let ratio = (N_HI / N_LO).powf(1.0 / (N_WAVES - 1) as f32);
    let (mut h, mut dh) = (0.0f32, 0.0f32);
    let mut n = N_LO;
    for k in 0..N_WAVES {
        let dn = n * (ratio.sqrt() - 1.0 / ratio.sqrt());
        let g = gd * (n / N0).powi(-2);
        let x = std::f32::consts::PI * n * bridge_m;
        let bridge = if x < 1e-4 { 1.0 } else { (x.sin() / x).abs() };
        let a = (2.0 * g * dn).sqrt() * bridge;
        let phase = (hash2(seed ^ 0x6E7A_10F3, side as i32, k as i32) % 10_000) as f32 / 10_000.0 * std::f32::consts::TAU;
        let w = std::f32::consts::TAU * n;
        h += a * (w * s + phase).sin();
        dh += a * w * (w * s + phase).cos();
        n *= ratio;
    }
    (h, dh)
}

fn ring(t: f32, peak: f32, pulse_s: f32) -> (f32, f32) {
    let wn = std::f32::consts::TAU * MOUNT_HZ;
    let wd = wn * (1.0 - MOUNT_ZETA * MOUNT_ZETA).sqrt();
    let end = t.min(pulse_s);
    let steps = 24;
    let ds = end / steps as f32;
    let (mut x, mut v) = (0.0f32, 0.0f32);
    for i in 0..steps {
        let s = (i as f32 + 0.5) * ds;
        let a = peak * (std::f32::consts::PI * s / pulse_s).sin();
        let tau = t - s;
        let e = (-MOUNT_ZETA * wn * tau).exp();
        x += a * e * (wd * tau).sin() / wd * ds;
        v += a * e * ((wd * tau).cos() - MOUNT_ZETA * wn / wd * (wd * tau).sin()) * ds;
    }
    (x, v)
}

fn normal(seed: u64, a: i32, b: i32) -> f32 {
    let u1 = ((hash2(seed, a, b) % 1_000_000) as f32 + 0.5) / 1_000_000.0;
    let u2 = ((hash2(seed ^ 0x9E37_79B9, a, b) % 1_000_000) as f32 + 0.5) / 1_000_000.0;
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

impl World {
    pub fn standing_eye_m(&self) -> f32 {
        hkin::head_camera(&self.body, &KinPose::stand(&self.body, 0.0)).pos[2]
    }

    /// The body pose the head camera hangs on right now (an act, a step, a held gaze, or standing).
    pub fn current_kin_pose(&self) -> KinPose {
        self.camera_pose().0
    }

    fn camera_pose(&self) -> (KinPose, bool) {
        if let Some(p) = &self.act {
            if let Some(s) = p.rollout.sample_at(p.t()) {
                return (s.pose, false);
            }
        }
        if let Some((pose, _, src)) = self.act_pose() {
            return (pose, src == noto_core::Source::FootLift);
        }
        let slope = if self.body.legged() { 0.0 } else { self.posture_now().0.slope_deg };
        let glance = if self.crew_turn.is_none() { self.glance_rad(1).to_degrees() } else { 0.0 };
        (KinPose { base_pitch_deg: slope, neck_pan_deg: glance, ..KinPose::stand(&self.body, self.carried_kg()) }, false)
    }

    pub fn head_view(&self, cam: &Camera) -> HeadView {
        if self.fork_cam && noto_core::act::fork::applies(&self.body) {
            return self.fork_view(cam);
        }
        if let Some(r) = self.rider.filter(|r| r.off()) {
            return self.rider_view(cam, &r);
        }
        let (mut pose, walking) = self.camera_pose();
        let (tilt, tilt_rate) = if self.body.legged() {
            ((0.0, 0.0), [0.0, 0.0])
        } else {
            let (x, z, yaw) = (self.agent.x, self.agent.z, self.agent.yaw);
            let now = self.chassis_attitude(x, z, yaw);
            let v = self.agent.speed_mps;
            let dt = 0.05;
            let next = if v > 1e-3 { self.chassis_attitude(x + v * dt * yaw.cos(), z + v * dt * yaw.sin(), yaw) } else { now };
            pose.base_pitch_deg = now.0.to_degrees();
            (now, [(next.0 - now.0) / dt, (next.1 - now.1) / dt])
        };
        let hc = hkin::head_camera(&self.body, &pose);
        let mut pos = hc.pos;
        if walking {
            pos[0] -= pose.hip[0];
        }
        let base = if self.body.legged() { 0.0 } else { pose.base_pitch_deg };
        let head_pitch_deg = hc.pitch_down_deg + base * self.body.head.gaze_stabilization;
        let timing = sensor_timing_at(self.body.head.sensor, self.scene_lux());
        let (shake, window_s) = self.shake();
        let arw = (timing.gyro_arw_deg * window_s.max(0.0).sqrt()).to_radians();
        let (ep, er) = if shake.source == "still" { (0.0, 0.0) } else { (arw * normal(self.cfg.seed, self.tick as i32, 1), arw * normal(self.cfg.seed, self.tick as i32, 2)) };
        let at = self.body_to_world(pos);
        let pan_deg = hc.pan_deg + cam.pan_deg;
        let yaw = self.agent.yaw - pan_deg.to_radians();
        let pitch = cam.pitch_deg.to_radians() - head_pitch_deg.to_radians();
        let view = View { origin: [at[0], at[1] + shake.dz_m, at[2]], yaw, pitch: pitch + shake.pitch, roll: shake.roll + tilt.1 };
        let est = View { pitch: view.pitch + ep, roll: view.roll + er, ..view };
        let keep = 1.0 - self.body.head.gaze_stabilization;
        let rate = [self.eye.yaw_rate, shake.rate[1] + tilt_rate[0] * keep, shake.rate[2] + tilt_rate[1]];
        let exposure = Exposure { rate, exposure_s: timing.exposure_s, readout_s: timing.readout_s };
        let tilt_deg = [tilt.0.to_degrees(), tilt.1.to_degrees()];
        HeadView { view, exposure, est, shake, eye_m: pos[2] + shake.dz_m, head_pitch_deg, pan_deg, blur_px: exposure.blur_px(cam), tilt_deg }
    }

    /// A lift truck's rider on foot: the head camera standing at the rider's eye height, looking where it walks or
    /// down at the node it fells.
    pub fn rider_view(&self, cam: &Camera, r: &super::traffic::Rider) -> HeadView {
        let ground = self.surface_m(r.x, r.z);
        let tilt = if r.phase == super::traffic::RiderPhase::Mine { 20.0f32 } else { 0.0 };
        let eye_m = super::traffic::rider_eye_m(&self.body);
        let view = View { origin: [r.x, ground + eye_m, r.z], yaw: r.yaw - cam.pan_deg.to_radians(), pitch: (cam.pitch_deg - tilt).to_radians(), roll: 0.0 };
        let timing = sensor_timing_at(self.body.head.sensor, self.scene_lux());
        let exposure = Exposure { rate: [0.0; 3], exposure_s: timing.exposure_s, readout_s: timing.readout_s };
        HeadView { view, exposure, est: view, shake: Shake::still(), eye_m, head_pitch_deg: tilt, pan_deg: cam.pan_deg, blur_px: 0.0, tilt_deg: [0.0, 0.0] }
    }

    /// The fork carriage camera: between the tines, a little above them, looking ahead and down at what the tines
    /// will go under. It rides the mast, so it is low when the tines are down.
    pub fn fork_view(&self, cam: &Camera) -> HeadView {
        let pos = noto_core::act::fork::camera_pos(&self.body, self.fork_m_now());
        let at = self.body_to_world(pos);
        let roll = self.chassis_attitude(self.agent.x, self.agent.z, self.agent.yaw).1;
        let view = View { origin: at, yaw: self.agent.yaw - cam.pan_deg.to_radians(), pitch: -noto_core::act::fork::CAMERA_PITCH_DEG.to_radians(), roll };
        let timing = sensor_timing_at(self.body.head.sensor, self.scene_lux());
        let exposure = Exposure { rate: [self.eye.yaw_rate, 0.0, 0.0], exposure_s: timing.exposure_s, readout_s: timing.readout_s };
        HeadView { view, exposure, est: view, shake: Shake::still(), eye_m: pos[2], head_pitch_deg: 0.0, pan_deg: cam.pan_deg, blur_px: exposure.blur_px(cam), tilt_deg: [0.0, roll.to_degrees()] }
    }

    pub fn chassis_attitude(&self, x: f32, z: f32, yaw: f32) -> (f32, f32) {
        let spec = self.drive_spec();
        let reach = spec.max_step_m.max(spec.max_slope_deg.to_radians().tan() * crate::terrain::CELL_M) + 0.01;
        let base = self.surface_m(x, z);
        let (len, wid) = (self.body.support_len_m.max(0.2), self.body.support_width_m.max(0.2));
        let (s, c) = yaw.sin_cos();
        let h = |f: f32, r: f32| self.terrain.surface_smooth_m(x + c * f - s * r, z + s * f + c * r, base, reach);
        let (fr, fl) = (h(0.5 * len, 0.5 * wid), h(0.5 * len, -0.5 * wid));
        let (rr, rl) = (h(-0.5 * len, 0.5 * wid), h(-0.5 * len, -0.5 * wid));
        let pitch = (0.5 * (fr + fl) - 0.5 * (rr + rl)).atan2(len);
        let roll = (0.5 * (fl + rl) - 0.5 * (fr + rr)).atan2(wid);
        (pitch, roll)
    }

    fn shake(&self) -> (Shake, f32) {
        let ground = self.ground_here().surface();
        let speed = self.agent.speed_mps;
        if speed <= 1e-3 {
            return (Shake::still(), 0.0);
        }
        let gd = roughness_gd(ground);
        let seed = self.cfg.seed;
        let stab = self.body.head.gaze_stabilization;
        if self.body.legged() {
            let Some((g, _, _)) = self.walk.plan.as_ref() else { return (Shake::still(), 0.0) };
            let t = self.walk.since_step_m / speed;
            let u = (self.walk.since_step_m / g.step_m.max(1e-3)).clamp(0.0, 1.0);
            let s = sway(u, self.walk.stance, g.step_s, stab)
                .add(self.heel_strike(g, ground, t))
                .add(footfall(seed, gd, g.step_m, self.body.dims.stance_m, self.walk.steps, stab));
            (s, t)
        } else {
            let tracked = matches!(self.body.base, noto_core::BaseKind::Track);
            let bridge = if tracked { self.body.support_len_m } else { 2.0 * self.body.dims.wheel_r_m };
            (rolling(seed, gd, self.eye.travelled_m, speed, self.body.support_len_m.max(0.2), self.body.support_width_m.max(0.2), bridge), IMU_WINDOW_S)
        }
    }

    fn heel_strike(&self, g: &noto_core::act::gait::GaitPlan, ground: Surface, t: f32) -> Shake {
        let s = strike_scale(ground);
        let pulse = STRIKE_PULSE_S / s;
        if pulse > g.ds_frac * g.step_s {
            return Shake::still();
        }
        let (x, v) = ring(t, STRIKE_PEAK_MPS2 * s, pulse);
        let eye = self.standing_eye_m();
        let r2 = (GYRATION_SHARE * eye).powi(2);
        let pitch_arm = 0.5 * g.step_m / r2;
        let side = if self.walk.stance == 0 { -1.0 } else { 1.0 };
        let roll_arm = side * 0.5 * self.body.dims.stance_m / r2;
        Shake { source: "heel_strike", dz_m: x, pitch: pitch_arm * x, roll: roll_arm * x, rate: [v, pitch_arm * v, roll_arm * v] }
    }
}

fn sway(u: f32, stance: usize, step_s: f32, stab: f32) -> Shake {
    let keep = 1.0 - stab;
    let (ap, ar) = ((0.5 * SWAY_PITCH_PP_DEG).to_radians() * keep, (0.5 * SWAY_ROLL_PP_DEG).to_radians() * keep);
    let w = std::f32::consts::TAU / step_s.max(0.1);
    let phase = std::f32::consts::TAU * u;
    let side = if stance == 0 { 1.0 } else { -1.0 };
    let half = std::f32::consts::PI * u;
    Shake {
        source: "sway",
        dz_m: 0.0,
        pitch: ap * phase.cos(),
        roll: side * ar * half.sin(),
        rate: [0.0, -ap * w * phase.sin(), side * ar * 0.5 * w * half.cos()],
    }
}

fn footfall(seed: u64, gd: f32, step_m: f32, stance_m: f32, steps: u32, stab: f32) -> Shake {
    let s_now = steps as f32 * step_m;
    let s_last = s_now - step_m;
    let (side_now, side_last) = if steps % 2 == 0 { (0, 1) } else { (1, 0) };
    let foot = noto_core::act::gait::NOMINAL_FOOT_M;
    let (h_now, _) = profile(seed, gd, side_now, s_now, foot);
    let (h_last, _) = profile(seed, gd, side_last, s_last, foot);
    let dh = h_now - h_last;
    let keep = 1.0 - stab;
    let roll_sign = if side_now == 1 { 1.0 } else { -1.0 };
    Shake {
        source: "footfall",
        dz_m: 0.5 * dh,
        pitch: keep * (dh / step_m.max(0.05)).atan() * 0.5,
        roll: keep * roll_sign * (dh / stance_m).atan() * 0.5,
        rate: [0.0; 3],
    }
}

pub fn chassis_accel_rms(seed: u64, ground: Surface, speed: f32, base_m: f32, track_m: f32, bridge_m: f32) -> f32 {
    let gd = roughness_gd(ground);
    let ds = 0.005f32;
    let n = (60.0 / ds) as usize;
    let heave: Vec<f32> = (0..n).map(|i| rolling(seed, gd, i as f32 * ds, speed, base_m, track_m, bridge_m).dz_m).collect();
    let acc: Vec<f32> = heave.windows(3).map(|w| (w[0] - 2.0 * w[1] + w[2]) / (ds * ds) * speed * speed).collect();
    (acc.iter().map(|a| a * a).sum::<f32>() / acc.len().max(1) as f32).sqrt()
}

pub fn strike_peak_mps2(ground: Surface, ds_s: f32) -> f32 {
    let s = strike_scale(ground);
    if STRIKE_PULSE_S / s > ds_s {
        0.0
    } else {
        STRIKE_PEAK_MPS2 * s
    }
}

fn rolling(seed: u64, gd: f32, s: f32, speed: f32, base_m: f32, track_m: f32, bridge_m: f32) -> Shake {
    let half = 0.5 * base_m;
    let at = |side: u32, x: f32| profile(seed, gd, side, x, bridge_m);
    let (fl, dfl) = at(0, s + half);
    let (fr, dfr) = at(1, s + half);
    let (rl, drl) = at(0, s - half);
    let (rr, drr) = at(1, s - half);
    let heave = 0.25 * (fl + fr + rl + rr);
    let pitch = ((0.5 * (fl + fr) - 0.5 * (rl + rr)) / base_m).atan();
    let roll = ((0.5 * (fl + rl) - 0.5 * (fr + rr)) / track_m).atan();
    let d_heave = 0.25 * (dfl + dfr + drl + drr) * speed;
    let d_pitch = (0.5 * (dfl + dfr) - 0.5 * (drl + drr)) / base_m * speed;
    let d_roll = (0.5 * (dfl + drl) - 0.5 * (dfr + drr)) / track_m * speed;
    Shake { source: "rolling", dz_m: heave, pitch, roll, rate: [d_heave, d_pitch, d_roll] }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::TerrainKind;
    use crate::world::{Action, WorldConfig};
    use noto_core::{BaseKind, BodyPlan, FingerCount, NotoConfig};

    fn world(base: BaseKind) -> World {
        let mut hand = NotoConfig::preset(FingerCount::Five);
        hand.base = base;
        World::new(WorldConfig { seed: 7, cells: 64, noto: hand, body: BodyPlan::default(), terrain: TerrainKind::Open, parcels: Some(0), ..WorldConfig::default() }).unwrap()
    }

    #[test]
    fn the_camera_sits_on_the_head_and_stands_still_at_rest() {
        for base in [BaseKind::HumanFoot, BaseKind::Wheel] {
            let w = world(base);
            let eye = w.standing_eye_m();
            assert!((eye - 1.14).abs() < 0.02, "{base:?}: eye {eye}");
            let cam = Camera::default();
            let hv = w.head_view(&cam);
            assert!((hv.view.pitch - cam.pitch_deg.to_radians()).abs() < 1e-4 && hv.view.roll == 0.0, "{base:?}: {:?}", hv.view);
            assert_eq!(hv.shake.source, "still");
            assert_eq!(hv.view, hv.est);
            assert!((hv.view.origin[1] - (w.agent_ground_m() + eye)).abs() < 1e-4);
        }
    }

    #[test]
    fn walking_shakes_the_camera_within_a_persons_range_and_the_gyro_reads_it() {
        let mut w = world(BaseKind::HumanFoot);
        let (ax, az) = (w.agent.x, w.agent.z);
        let goal = (0..16)
            .map(|k| (k as f32 * std::f32::consts::TAU / 16.0).sin_cos())
            .map(|(s, c)| (ax + 6.0 * c, az + 6.0 * s))
            .find(|&(x, z)| w.clone().command(Action::MoveTo { x, z, reach_m: 0.3 }).is_ok())
            .expect("a reachable point 6 m away");
        w.command(Action::MoveTo { x: goal.0, z: goal.1, reach_m: 0.3 }).unwrap();
        let cam = Camera::default();
        let (mut peak_pitch, mut peak_rate, mut seen, mut peak_err) = (0.0f32, 0.0f32, 0, 0.0f32);
        for _ in 0..40 {
            w.tick();
            let hv = w.head_view(&cam);
            if hv.shake.source == "still" {
                continue;
            }
            seen += 1;
            peak_pitch = peak_pitch.max(hv.shake.pitch.abs());
            peak_rate = peak_rate.max(hv.shake.rate[1].abs());
            peak_err = peak_err.max((hv.view.pitch - hv.est.pitch).abs());
        }
        assert!(seen > 20, "walking ticks {seen}");
        assert!(peak_pitch.to_degrees() > 0.2 && peak_pitch.to_degrees() < 4.5, "pitch shake {}°", peak_pitch.to_degrees());
        assert!(peak_rate.to_degrees() < 90.0, "pitch rate {}°/s", peak_rate.to_degrees());
        assert!(peak_err.to_degrees() < 0.1, "IMU error {}°", peak_err.to_degrees());
    }

    #[test]
    fn harder_ground_strikes_harder_and_rougher_ground_tilts_more() {
        let w = world(BaseKind::HumanFoot);
        let g = noto_core::act::gait::plan(&w.body, 0.6, 0.7, 0.0);
        let peak = |s: Surface| (0..300).map(|i| w.heel_strike(&g, s, i as f32 * 0.001).pitch.abs()).fold(0.0, f32::max);
        assert!(peak(Surface::Firm) > peak(Surface::Grass) && peak(Surface::Grass) > peak(Surface::Sand), "firm {} grass {} sand {}", peak(Surface::Firm), peak(Surface::Grass), peak(Surface::Sand));
        assert_eq!(strike_peak_mps2(Surface::Sand, 0.15), 0.0);
        assert!(strike_peak_mps2(Surface::Firm, 0.15) > 4.0);
        let tilt = |s: Surface| ((0..400).map(|k| footfall(5, roughness_gd(s), 0.3, noto_core::dims::Dims::COMPACT.stance_m, k, 0.55).roll.powi(2)).sum::<f32>() / 400.0).sqrt();
        assert!(tilt(Surface::Rock) > 4.0 * tilt(Surface::Firm), "rock {} firm {}", tilt(Surface::Rock), tilt(Surface::Firm));
        let roll = |s: Surface| ((0..400).map(|i| rolling(5, roughness_gd(s), i as f32 * 0.07, 1.0, 0.4, 0.35, 0.12).pitch.powi(2)).sum::<f32>() / 400.0).sqrt();
        assert!(roll(Surface::Gravel) > 1.5 * roll(Surface::Firm), "gravel {} firm {}", roll(Surface::Gravel), roll(Surface::Firm));
    }

    #[test]
    fn a_chassis_tilts_with_the_ground_and_legs_keep_the_torso_level() {
        for base in [BaseKind::Wheel, BaseKind::Track, BaseKind::HumanFoot] {
            let mut w = world(base);
            w.agent.yaw = 0.0;
            let (cx, cz) = w.agent_cell();
            (w.agent.x, w.agent.z) = crate::terrain::cell_center(cx, cz);
            let h0 = 50;
            for dz in -2i32..=2 {
                for dx in -2i32..=2 {
                    let (x, z) = ((cx as i32 + dx) as usize, (cz as i32 + dz) as usize);
                    w.terrain.set_height_cm(x, z, h0 - 5 * dz);
                }
            }
            let hv = w.head_view(&Camera::default());
            if base == BaseKind::HumanFoot {
                assert_eq!(hv.tilt_deg, [0.0, 0.0], "legs keep the torso level");
                assert!(hv.view.roll.abs() < 1e-6);
            } else {
                assert!(hv.tilt_deg[1] > 3.0 && hv.tilt_deg[1] < 7.0, "{base:?} roll {:?}", hv.tilt_deg);
                assert!(hv.tilt_deg[0].abs() < 0.5, "{base:?} pitch {:?}", hv.tilt_deg);
                assert!((hv.view.roll.to_degrees() - hv.tilt_deg[1]).abs() < 1e-3, "the neck has no roll joint");
            }
        }
        let mut w = world(BaseKind::Wheel);
        w.agent.yaw = 0.0;
        let (cx, cz) = w.agent_cell();
        (w.agent.x, w.agent.z) = crate::terrain::cell_center(cx, cz);
        let h0 = (w.terrain.height_m(cx, cz) * 100.0).round() as i32;
        for dx in -2i32..=2 {
            w.terrain.set_height_cm((cx as i32 + dx) as usize, cz + 1, h0 + 120);
        }
        assert!(w.head_view(&Camera::default()).tilt_deg[1].abs() < 0.5);
    }

    #[test]
    fn turning_blurs_in_dim_light_and_hardly_in_daylight() {
        let mut w = world(BaseKind::Wheel);
        let cam = Camera::default();
        assert!(w.head_view(&cam).blur_px < 0.1);
        let yaw = w.agent.yaw + 1.5;
        w.command(Action::Turn { yaw_deg: yaw.to_degrees() }).unwrap();
        w.tick();
        let hv = w.head_view(&cam);
        assert!(hv.exposure.rate[0].abs() > 1.0, "yaw rate {}", hv.exposure.rate[0]);
        assert!(hv.blur_px > 0.1 && hv.blur_px < 2.0, "daylight blur {} px", hv.blur_px);
        let dim = Exposure { exposure_s: exposure_s(200.0), ..hv.exposure };
        assert!(dim.blur_px(&cam) > 15.0, "dim blur {} px", dim.blur_px(&cam));
        assert!((exposure_s(10_000.0) - 0.00087).abs() < 2e-5 && (exposure_s(500.0) - 0.0175).abs() < 3e-4);
    }

    #[test]
    fn rougher_classes_give_taller_profiles() {
        let rms = |gd: f32| {
            let n = 2000;
            ((0..n).map(|i| profile(7, gd, 0, i as f32 * 0.05, 0.0).0.powi(2)).sum::<f32>() / n as f32).sqrt()
        };
        let (a, d) = (rms(16.0e-6), rms(1024.0e-6));
        assert!((d / a - 8.0).abs() < 0.5, "{a} {d}");
        assert!(a > 0.7e-3 && a < 2.0e-3, "class A rms {a}");
    }

    #[test]
    fn a_long_contact_bridges_short_bumps() {
        let rough = |bridge: f32| {
            let n = 2000;
            (0..n).map(|i| profile(3, 512e-6, 1, i as f32 * 0.01, bridge).1.abs()).sum::<f32>() / n as f32
        };
        assert!(rough(0.4) < 0.5 * rough(0.0), "a track's contact smooths the slope it feels");
    }

    #[test]
    fn the_mount_rings_and_settles() {
        let (x0, _) = ring(0.04, STRIKE_PEAK_MPS2, STRIKE_PULSE_S);
        let (x1, _) = ring(0.40, STRIKE_PEAK_MPS2, STRIKE_PULSE_S);
        assert!(x0.abs() > 10.0 * x1.abs(), "{x0} {x1}");
        let peak = |s: f32| (0..200).map(|i| ring(i as f32 * 0.001, STRIKE_PEAK_MPS2 * s, STRIKE_PULSE_S / s).0.abs()).fold(0.0, f32::max);
        assert!(peak(0.36) < peak(1.0));
    }
}
