use serde::Serialize;

use crate::act::material;
use crate::balance::Body;

pub const PAD_MU_FORCE_EXP: f32 = -0.21;
pub const PAD_MU_REF_N: f32 = 10.0;
pub const PAD_MU_N_RANGE: [f32; 2] = [1.0, 55.0];
pub const PAD_MU_FLOOR: f32 = 0.5;
pub const FINGER_MU_COV: f32 = 0.04;
pub const WET_PAD_FACTOR: f32 = 0.875;
pub const DUSTY_PAD_FACTOR: f32 = 0.75;
pub const STICK_ALARM: f32 = 0.5;
pub const TACTILE_FPS: f32 = 60.0;
pub const DETECT_FRAMES: f32 = 2.0;
pub const PRELOAD_N: f32 = 1.0;

pub fn detect_delay_s() -> f32 {
    DETECT_FRAMES / TACTILE_FPS
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PadSurface {
    #[default]
    Dry,
    Wet,
    Dusty,
}

impl PadSurface {
    pub fn id(self) -> &'static str {
        match self {
            PadSurface::Dry => "dry",
            PadSurface::Wet => "wet",
            PadSurface::Dusty => "dusty",
        }
    }
}

pub fn pad_mu(mu_table: f32, n: f32, surface: PadSurface) -> f32 {
    let n = n.clamp(PAD_MU_N_RANGE[0], PAD_MU_N_RANGE[1]);
    let law = (n / PAD_MU_REF_N).powf(PAD_MU_FORCE_EXP).max(PAD_MU_FLOOR);
    let s = match surface {
        PadSurface::Dry => 1.0,
        PadSurface::Wet => WET_PAD_FACTOR,
        PadSurface::Dusty => DUSTY_PAD_FACTOR,
    };
    mu_table * law * s
}

pub fn patch_radius_m(n: f32) -> f32 {
    material::contact_radius_m(n)
}

pub fn torsion_radius_m(n: f32) -> f32 {
    3.0 * std::f32::consts::PI / 16.0 * patch_radius_m(n)
}

pub fn patch_area_m2(n: f32) -> f32 {
    std::f32::consts::PI * patch_radius_m(n).powi(2)
}

pub fn tangential_modulus_pa() -> f32 {
    material::PAD_E_PA / (2.0 * (2.0 - material::PAD_NU) * (1.0 + material::PAD_NU))
}

pub fn tangential_stiffness(n: f32) -> f32 {
    8.0 * tangential_modulus_pa() * patch_radius_m(n.max(0.05))
}

pub fn preslip_displacement_m(q: f32, mu: f32, n: f32) -> f32 {
    let f = (mu * n).max(1e-9);
    let a = patch_radius_m(n).max(1e-6);
    3.0 * f / (16.0 * tangential_modulus_pa() * a) * (1.0 - (1.0 - (q.abs() / f).min(1.0)).powf(2.0 / 3.0))
}

pub fn limit_use(ft: f32, m: f32, mu: f32, n: f32) -> f32 {
    let f_max = (mu * n).max(1e-9);
    let m_max = (torsion_radius_m(n) * mu * n).max(1e-12);
    ((ft / f_max).powi(2) + (m / m_max).powi(2)).sqrt()
}

pub fn tangential_capacity(m: f32, mu: f32, n: f32) -> f32 {
    let f_max = (mu * n).max(0.0);
    let m_max = (torsion_radius_m(n) * mu * n).max(1e-12);
    f_max * (1.0 - (m / m_max).powi(2)).max(0.0).sqrt()
}

pub fn slip_direction(ft: f32, m: f32, mu: f32, n: f32) -> [f32; 2] {
    let f_max = (mu * n).max(1e-9);
    let m_max = (torsion_radius_m(n) * mu * n).max(1e-12);
    let g = [ft / f_max, m / m_max];
    let l = (g[0] * g[0] + g[1] * g[1]).sqrt();
    if l > 1e-12 {
        [g[0] / l, g[1] / l]
    } else {
        [1.0, 0.0]
    }
}

pub fn stick_ratio(q: f32, q_max: f32) -> f32 {
    if q_max <= 1e-9 {
        return 0.0;
    }
    (1.0 - (q.abs() / q_max).min(1.0)).powf(2.0 / 3.0)
}

pub fn mu_from_stick(q: f32, n: f32, stick: f32) -> f32 {
    let used = 1.0 - stick.clamp(0.0, 1.0).powf(1.5);
    if used <= 1e-4 || n <= 1e-6 {
        f32::INFINITY
    } else {
        q.abs() / (n * used)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SlipState {
    #[default]
    Stick,
    Incipient,
    Slip,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct PadLoad {
    pub n: f32,
    pub mu: f32,
    pub ft: f32,
    pub torsion_nm: f32,
    pub cone_use: f32,
    pub stick: f32,
    pub state: SlipState,
    pub radius_m: f32,
    pub slip_dir: [f32; 2],
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Shared {
    pub pads: Vec<PadLoad>,
    pub capacity_n: f32,
    pub gross_slip: bool,
    pub stick_min: f32,
}

pub fn share_load(normals: &[f32], mus: &[f32], load_n: f32, twist_nm: f32) -> Shared {
    let k = normals.len();
    let mut pads: Vec<PadLoad> = (0..k).map(|i| PadLoad { n: normals[i].max(0.0), mu: mus[i], radius_m: patch_radius_m(normals[i].max(0.0)), ..PadLoad::default() }).collect();
    let on: Vec<usize> = (0..k).filter(|&i| pads[i].n > 1e-6).collect();
    if on.is_empty() {
        return Shared { pads, capacity_n: 0.0, gross_slip: load_n > 1e-6 || twist_nm.abs() > 1e-9, stick_min: 0.0 };
    }
    let m_max: Vec<f32> = (0..k).map(|i| torsion_radius_m(pads[i].n) * pads[i].mu * pads[i].n).collect();
    let mut t_free: Vec<bool> = (0..k).map(|i| pads[i].n > 1e-6).collect();
    let mut t_left = twist_nm.abs();
    for _ in 0..k + 1 {
        let a3: f32 = (0..k).filter(|&i| t_free[i]).map(|i| pads[i].radius_m.powi(3)).sum();
        if a3 <= 1e-15 || t_left <= 1e-12 {
            break;
        }
        let share: Vec<f32> = (0..k).map(|i| if t_free[i] { t_left * pads[i].radius_m.powi(3) / a3 } else { 0.0 }).collect();
        let capped: Vec<usize> = (0..k).filter(|&i| t_free[i] && pads[i].torsion_nm + share[i] >= m_max[i]).collect();
        if capped.is_empty() {
            for i in 0..k {
                pads[i].torsion_nm += share[i];
            }
            t_left = 0.0;
            break;
        }
        for &i in &capped {
            pads[i].torsion_nm = m_max[i];
            t_free[i] = false;
        }
        t_left = (twist_nm.abs() - pads.iter().map(|p| p.torsion_nm).sum::<f32>()).max(0.0);
    }
    let twist_slip = t_left > 1e-9;
    for p in pads.iter_mut() {
        p.torsion_nm = p.torsion_nm.copysign(twist_nm);
    }
    let cap: Vec<f32> = (0..k).map(|i| if pads[i].n > 1e-6 { tangential_capacity(pads[i].torsion_nm, pads[i].mu, pads[i].n) } else { 0.0 }).collect();
    let capacity_n: f32 = cap.iter().sum();
    let stiff: Vec<f32> = (0..k).map(|i| if pads[i].n > 1e-6 { tangential_stiffness(pads[i].n) } else { 0.0 }).collect();
    let mut free: Vec<bool> = (0..k).map(|i| pads[i].n > 1e-6).collect();
    let mut left = load_n.max(0.0);
    for _ in 0..k + 1 {
        let ks: f32 = (0..k).filter(|&i| free[i]).map(|i| stiff[i]).sum();
        if ks <= 1e-12 || left <= 1e-9 {
            break;
        }
        let mut capped = false;
        for i in 0..k {
            if free[i] && left * stiff[i] / ks >= cap[i] {
                capped = true;
            }
        }
        if !capped {
            for i in 0..k {
                if free[i] {
                    pads[i].ft += left * stiff[i] / ks;
                }
            }
            left = 0.0;
            break;
        }
        for i in 0..k {
            if free[i] && left * stiff[i] / ks >= cap[i] {
                pads[i].ft = cap[i];
                free[i] = false;
            }
        }
        left = (load_n.max(0.0) - pads.iter().map(|p| p.ft).sum::<f32>()).max(0.0);
    }
    let gross_slip = (left > 1e-6 && load_n > capacity_n + 1e-6) || twist_slip;
    let mut stick_min = 1.0f32;
    for i in 0..k {
        let p = &mut pads[i];
        if p.n <= 1e-6 {
            continue;
        }
        p.cone_use = limit_use(p.ft, p.torsion_nm, p.mu, p.n);
        p.slip_dir = slip_direction(p.ft, p.torsion_nm, p.mu, p.n);
        p.stick = stick_ratio(p.cone_use, 1.0);
        p.state = if gross_slip || p.cone_use >= 0.999 {
            SlipState::Slip
        } else if p.stick < STICK_ALARM {
            SlipState::Incipient
        } else {
            SlipState::Stick
        };
        if gross_slip {
            p.stick = 0.0;
        }
        stick_min = stick_min.min(p.stick);
    }
    Shared { pads, capacity_n, gross_slip, stick_min }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct FingerContact {
    pub finger: u8,
    pub phalanx: u8,
    pub point: [f32; 3],
    pub normal: [f32; 3],
    pub fn_n: f32,
    pub ft: [f32; 3],
    pub ft_n: f32,
    pub torsion_nm: f32,
    pub mu: f32,
    pub cone_use: f32,
    pub stick: f32,
    pub state: SlipState,
    pub radius_m: f32,
    pub slip_dir: [f32; 2],
    /// How far the pad has sunk in (m, `material::pad_indentation_m`) and the pressure in the middle of its patch (kPa).
    pub indentation_m: f32,
    pub peak_kpa: f32,
}

/// The most pads a hand records per sample: every phalanx of four fingers and the thumb's two [추정]. `ActSample` is
/// copied 50 times a second, so the record is a fixed array; past this the weakest pads are left out of the hold.
pub const MAX_ACTIVE_PADS: usize = 12;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FingerContacts {
    pub n: u8,
    pub c: [FingerContact; MAX_ACTIVE_PADS],
}

impl FingerContacts {
    pub fn push(&mut self, c: FingerContact) {
        if (self.n as usize) < self.c.len() {
            self.c[self.n as usize] = c;
            self.n += 1;
        }
    }

    pub fn as_slice(&self) -> &[FingerContact] {
        &self.c[..self.n as usize]
    }
}

impl Serialize for FingerContacts {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let mut seq = s.serialize_seq(Some(self.n as usize))?;
        for c in self.as_slice() {
            seq.serialize_element(c)?;
        }
        seq.end()
    }
}

pub const SHEAR_DETECT_N: f32 = 0.1;

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct RampHold {
    pub latency_s: Option<f32>,
    pub grip_n: f32,
    pub load_n: f32,
    pub safety_margin: f32,
    pub stick_min: f32,
    pub slipped: bool,
}

pub const RAMP_MAX_S: f32 = 60.0;

pub fn ramp_hold(mu_table: f32, surface: PadSurface, mu_est: f32, safety: f32, grip0_n: f32, pads: usize, rate_n_s: f32, plateau_n: f32, hold_s: f32, rise_s: f32) -> RampHold {
    let dt = 1e-3f32;
    let frame = 1.0 / TACTILE_FPS;
    let delay = detect_delay_s();
    let tau = (rise_s / 2.2).max(1e-3);
    let rate = rate_n_s.max(1e-6);
    let load_at = |t: f32| (rate * t).clamp(0.0, plateau_n.max(0.0));
    let k = pads.max(1);
    let steps = (((plateau_n.max(0.0) / rate).min(RAMP_MAX_S) + hold_s.clamp(0.0, RAMP_MAX_S)) / dt).ceil() as usize;
    let mut grip = grip0_n.max(0.0);
    let mut latency_s = None;
    let mut stick_min = 1.0f32;
    let mut out = RampHold { latency_s: None, grip_n: grip, load_n: 0.0, safety_margin: 1.0, stick_min: 1.0, slipped: false };
    let mut acting = false;
    for i in 1..=steps {
        let t = i as f32 * dt;
        let seen = if t >= delay { load_at(((t - delay) / frame).floor() * frame) } else { 0.0 };
        acting = acting || seen >= SHEAR_DETECT_N;
        let cmd = if acting { grip0_n + safety * seen / (PINCH_FACES * mu_est.max(0.05)) } else { grip0_n };
        grip += (cmd - grip) * (1.0 - (-dt / tau).exp());
        if latency_s.is_none() && grip > grip0_n + 1e-4 {
            latency_s = Some(t);
        }
        let load = load_at(t);
        let n = grip / k as f32;
        let normals = vec![n; 2 * k];
        let mus: Vec<f32> = normals.iter().map(|&ni| pad_mu(mu_table, ni, surface)).collect();
        let st = share_load(&normals, &mus, load, 0.0);
        if load > 1e-6 {
            stick_min = stick_min.min(st.stick_min);
        }
        out = RampHold { latency_s, grip_n: grip, load_n: load, safety_margin: 1.0 - load / st.capacity_n.max(1e-6), stick_min, slipped: st.gross_slip };
        if st.gross_slip {
            break;
        }
    }
    out
}

pub const PINCH_FACES: f32 = 2.0;

pub const SOLE_BOUNDS_FL: [f32; 5] = [0.0, 0.30, 0.55, 0.75, 1.0];
pub const TOE_HINGE_FL: f32 = 0.75;
pub const ANKLE_FL: f32 = 0.25;
pub const HRP2TJ_TOE_KP: f32 = 10.0;
pub const HRP2TJ_TOE_KD: f32 = 5.0;
pub const HRP2TJ_MASS_KG: f32 = 54.0;
pub const HRP2TJ_COM_Z_M: f32 = 0.814;
pub const TOE_DORSI_MAX_DEG: f32 = 60.0;
pub const SQUARE_TORSION: f32 = 0.383;
pub const SOLE_COMPRESS_M: f32 = 0.004;
pub const SOLE_PATCHES: [&str; 4] = ["heel", "midfoot", "metatarsal", "toe"];
pub const SOLE_PATCHES_KO: [&str; 4] = ["뒤꿈치", "중족", "앞발 볼", "발가락"];

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct SoleModel {
    pub foot_len_m: f32,
    pub foot_width_m: f32,
    pub k_per_m: f32,
    pub toe_kp: f32,
    pub toe_kd: f32,
    pub toe_share: f32,
}

impl SoleModel {
    pub fn of(body: &Body) -> SoleModel {
        let fl = body.leg.foot_len_m;
        let fw = body.leg.foot_width_m;
        let k_per_m = crate::act::gait::SOLE_STIFFNESS / (SOLE_BOUNDS_FL[1] * fl).max(1e-3);
        let z = body.com_height_m().max(0.1);
        let kp = HRP2TJ_TOE_KP * (body.mass_kg() * z) / (HRP2TJ_MASS_KG * HRP2TJ_COM_Z_M);
        let tau = HRP2TJ_TOE_KD / HRP2TJ_TOE_KP * (z / HRP2TJ_COM_Z_M).sqrt();
        let l_toe = (1.0 - TOE_HINGE_FL) * fl;
        let k_rot = k_per_m * l_toe.powi(3) / 3.0;
        SoleModel { foot_len_m: fl, foot_width_m: fw, k_per_m, toe_kp: kp, toe_kd: kp * tau, toe_share: kp / (kp + k_rot) }
    }

    fn bounds(&self) -> [f32; 5] {
        SOLE_BOUNDS_FL.map(|f| f * self.foot_len_m)
    }

    pub fn edge_inset_m(&self, f: f32) -> f32 {
        2.0 * f.max(0.0) / (3.0 * self.k_per_m.max(1.0) * SOLE_COMPRESS_M)
    }

    pub fn hinge_m(&self) -> f32 {
        TOE_HINGE_FL * self.foot_len_m
    }

    pub fn static_toe_deg(&self, m: f32) -> f32 {
        (m.max(0.0) / self.toe_kp.max(1e-3)).to_degrees().min(TOE_DORSI_MAX_DEG)
    }

    pub fn toe_step_deg(&self, deg: f32, m: f32, dt: f32) -> f32 {
        if m <= 0.0 {
            return 0.0;
        }
        let th = deg.to_radians();
        let tau = (self.toe_kd / self.toe_kp.max(1e-3)).max(1e-3);
        let target = m / self.toe_kp.max(1e-3);
        let next = target + (th - target) * (-dt / tau).exp();
        next.to_degrees().clamp(0.0, TOE_DORSI_MAX_DEG)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct SolePatch {
    pub name: &'static str,
    pub label_ko: &'static str,
    pub x0: f32,
    pub x1: f32,
    pub normal_n: f32,
    pub shear: [f32; 2],
    pub torsion_nm: f32,
    pub peak_kpa: f32,
    pub cone_use: f32,
    pub slipping: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct SoleContact {
    pub patches: [SolePatch; 4],
    pub rocker: bool,
    pub toe_held: bool,
    pub toe_deg: f32,
    pub toe_torque_nm: f32,
    pub cop_x: f32,
    pub loaded: [f32; 2],
    pub mu: f32,
    pub cone_use: f32,
    pub cone_use_max: f32,
    pub stick_share: f32,
}

fn clipped_line(a: f32, b: f32, u0: f32, u1: f32) -> (f32, f32) {
    let (v0, v1) = if b.abs() < 1e-12 {
        if a > 0.0 {
            (u0, u1)
        } else {
            return (0.0, 0.0);
        }
    } else {
        let r = -a / b;
        if b > 0.0 {
            (u0.max(r), u1)
        } else {
            (u0, u1.min(r))
        }
    };
    if v1 <= v0 {
        return (0.0, 0.0);
    }
    let f = a * (v1 - v0) + 0.5 * b * (v1 * v1 - v0 * v0);
    let m = 0.5 * a * (v1 * v1 - v0 * v0) + b * (v1 * v1 * v1 - v0 * v0 * v0) / 3.0;
    (f, m)
}

fn plate_line(bounds: &[f32], w: &[f32], cop: f32) -> (f32, f32) {
    let (u0, u1) = (bounds[0], bounds[bounds.len() - 1]);
    let h = 0.5 * (u1 - u0).max(1e-6);
    let mid_u = 0.5 * (u0 + u1);
    let at = |psi: f32| {
        let (s, c) = psi.sin_cos();
        let (a, b) = (c - s * mid_u / h, s / h);
        let (mut f, mut m) = (0.0, 0.0);
        for i in 0..w.len() {
            let (fi, mi) = clipped_line(a, b, bounds[i], bounds[i + 1]);
            f += w[i] * fi;
            m += w[i] * mi;
        }
        (a, b, f, m)
    };
    let target = cop.clamp(u0 + 1e-4 * h, u1 - 1e-4 * h);
    let lim = 0.75 * std::f32::consts::PI - 1e-4;
    let (mut lo, mut hi) = (-lim, lim);
    for _ in 0..48 {
        let mid = 0.5 * (lo + hi);
        let (_, _, f, m) = at(mid);
        let c = if f > 1e-12 { m / f } else if mid < 0.0 { u0 } else { u1 };
        if c < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let (a, b, f, _) = at(0.5 * (lo + hi));
    let f = f.max(1e-12);
    (a / f, b / f)
}

pub fn sole_contact(sole: &SoleModel, ankle_x: f32, normal_n: f32, shear: [f32; 2], torsion_nm: f32, cop_x: f32, toe_deg: f32, mu: f32) -> SoleContact {
    let b = sole.bounds();
    let x_back = ankle_x - ANKLE_FL * sole.foot_len_m;
    let mut out = SoleContact { mu, ..SoleContact::default() };
    for i in 0..4 {
        out.patches[i] = SolePatch { name: SOLE_PATCHES[i], label_ko: SOLE_PATCHES_KO[i], x0: x_back + b[i], x1: x_back + b[i + 1], ..SolePatch::default() };
    }
    let f = normal_n.max(0.0);
    if f <= 1e-6 {
        out.cop_x = cop_x;
        return out;
    }
    let hinge = sole.hinge_m();
    let rocker = toe_deg > 1e-3;
    let l_toe = b[4] - b[3];
    let inset = sole.edge_inset_m(f);
    let u_c = if rocker {
        (cop_x - x_back).clamp(hinge, sole.foot_len_m - inset.min(l_toe / 3.0))
    } else {
        let e = inset.min(sole.foot_len_m / 3.0);
        (cop_x - x_back).clamp(e, sole.foot_len_m - e)
    };
    let width = sole.foot_width_m.max(1e-3);
    let mut force = [0.0f32; 4];
    let mut peak = [0.0f32; 4];
    let mut loaded_len = [0.0f32; 4];
    let mut loaded = [b[4], b[0]];
    if !rocker {
        out.toe_held = u_c > hinge;
        let w = [1.0, 1.0, 1.0, if out.toe_held { 1.0 } else { sole.toe_share }];
        let (a, bb) = plate_line(&b, &w, u_c);
        for i in 0..4 {
            let (fi, _) = clipped_line(a, bb, b[i], b[i + 1]);
            force[i] = f * w[i] * fi;
            let d0 = (a + bb * b[i]).max(0.0);
            let d1 = (a + bb * b[i + 1]).max(0.0);
            peak[i] = f * w[i] * d0.max(d1) / width / 1000.0;
            let (v0, v1) = if bb.abs() < 1e-12 { (b[i], b[i + 1]) } else if bb > 0.0 { (b[i].max(-a / bb), b[i + 1]) } else { (b[i], b[i + 1].min(-a / bb)) };
            loaded_len[i] = (v1 - v0).max(0.0);
            if force[i] > 1e-6 {
                loaded[0] = loaded[0].min(v0);
                loaded[1] = loaded[1].max(v1);
            }
        }
        let (fi, mi) = clipped_line(a, bb, b[3], b[4]);
        out.toe_torque_nm = f * w[3] * (mi - hinge * fi);
        out.toe_deg = if out.toe_held { 0.0 } else { sole.static_toe_deg(out.toe_torque_nm) };
    } else {
        out.rocker = true;
        let x_tc = hinge + 0.5 * l_toe;
        let f_toe = f * ((u_c - hinge) / (x_tc - hinge)).clamp(0.0, 1.0);
        let f_meta = f - f_toe;
        out.toe_torque_nm = f * (u_c - hinge);
        out.toe_deg = toe_deg.min(TOE_DORSI_MAX_DEG);
        force[3] = f_toe;
        if u_c <= x_tc {
            peak[3] = f_toe / (l_toe * width) / 1000.0;
            loaded_len[3] = l_toe;
        } else {
            let tb = [b[3], b[4]];
            let (a, bb) = plate_line(&tb, &[1.0], u_c);
            let d0 = (a + bb * b[3]).max(0.0);
            let d1 = (a + bb * b[4]).max(0.0);
            peak[3] = f_toe * d0.max(d1) / width / 1000.0;
            let v0 = if bb > 1e-12 { b[3].max(-a / bb) } else { b[3] };
            loaded_len[3] = (b[4] - v0).max(0.0);
        }
        if f_meta > 1e-6 {
            let tan = out.toe_deg.to_radians().tan().max(1e-4);
            let band = (2.0 * f_meta / (sole.k_per_m * tan)).sqrt().max(2.0 * f_meta / (sole.k_per_m * SOLE_COMPRESS_M)).min(b[3] - b[2]);
            force[2] = f_meta;
            peak[2] = 2.0 * f_meta / (band * width) / 1000.0;
            loaded_len[2] = band;
        }
        loaded = [if f_meta > 1e-6 { hinge - loaded_len[2] } else { b[4] - loaded_len[3] }, b[4]];
    }
    let s_mag = (shear[0] * shear[0] + shear[1] * shear[1]).sqrt();
    let dir = if s_mag > 1e-9 { [shear[0] / s_mag, shear[1] / s_mag] } else { [1.0, 0.0] };
    let known = mu.is_finite() && mu > 0.0;
    let cap: Vec<f32> = (0..4).map(|i| if known { mu * force[i] } else { f32::INFINITY }).collect();
    let total_cap: f32 = if known { cap.iter().sum() } else { f32::INFINITY };
    let mut q = [0.0f32; 4];
    let mut slipping = [false; 4];
    if !known {
        let lsum: f32 = loaded_len.iter().sum::<f32>().max(1e-9);
        for i in 0..4 {
            q[i] = s_mag * loaded_len[i] / lsum;
        }
    } else if s_mag >= total_cap - 1e-9 {
        for i in 0..4 {
            q[i] = cap[i];
            slipping[i] = force[i] > 1e-6;
        }
    } else {
        let lsum: f32 = loaded_len.iter().sum::<f32>().max(1e-9);
        let (mut lo, mut hi) = (0.0f32, s_mag / lsum * 1e3 + 1.0);
        for _ in 0..40 {
            let u = 0.5 * (lo + hi);
            let s: f32 = (0..4).map(|i| (u * loaded_len[i]).min(cap[i])).sum();
            if s < s_mag {
                lo = u;
            } else {
                hi = u;
            }
        }
        let u = 0.5 * (lo + hi);
        for i in 0..4 {
            q[i] = (u * loaded_len[i]).min(cap[i]);
            slipping[i] = force[i] > 1e-6 && u * loaded_len[i] >= cap[i] - 1e-9;
        }
    }
    let mut stuck_len = 0.0;
    let mut use_max = 0.0f32;
    for i in 0..4 {
        let p = &mut out.patches[i];
        p.normal_n = force[i];
        p.shear = [q[i] * dir[0], q[i] * dir[1]];
        p.torsion_nm = torsion_nm * force[i] / f;
        p.peak_kpa = peak[i];
        p.slipping = slipping[i];
        if force[i] > 1e-6 {
            if known {
                let e = SQUARE_TORSION * (loaded_len[i].max(1e-3) * width).sqrt();
                let fmax = mu * force[i];
                p.cone_use = ((q[i] / fmax).powi(2) + (p.torsion_nm / (e * fmax)).powi(2)).sqrt();
                use_max = use_max.max(p.cone_use);
            }
            if !slipping[i] {
                stuck_len += loaded_len[i];
            }
        }
    }
    let loaded_total: f32 = loaded_len.iter().sum::<f32>().max(1e-9);
    out.stick_share = (stuck_len / loaded_total).clamp(0.0, 1.0);
    out.cop_x = x_back + u_c;
    out.loaded = [x_back + loaded[0].min(loaded[1]), x_back + loaded[1].max(loaded[0])];
    out.cone_use = if known { s_mag / (mu * f) } else { 0.0 };
    out.cone_use_max = use_max;
    out
}

pub fn rocker_ankle_shift(foot_len_m: f32, ankle_h_m: f32, toe_deg: f32) -> [f32; 2] {
    let th = toe_deg.clamp(0.0, TOE_DORSI_MAX_DEG).to_radians();
    let d = (TOE_HINGE_FL - ANKLE_FL) * foot_len_m;
    let (s, c) = th.sin_cos();
    [d * (1.0 - c) + ankle_h_m * s, d * s + ankle_h_m * (c - 1.0)]
}

pub fn friction_room_n(mu: f32, normal_n: f32, shear_x_n: f32) -> [f32; 2] {
    let cap = mu.max(0.0) * normal_n.max(0.0);
    [(cap - shear_x_n).max(0.0), (cap + shear_x_n).max(0.0)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::BaseKind;
    use crate::hand::FingerCount;
    use crate::parts::{BodyPlan, Tier};

    #[test]
    fn pad_friction_falls_with_force_and_holds_its_floor() {
        let mu = pad_mu(0.6, PAD_MU_REF_N, PadSurface::Dry);
        assert!((mu - 0.6).abs() < 1e-6);
        let drop = pad_mu(0.6, 20.0, PadSurface::Dry) / pad_mu(0.6, 1.0, PadSurface::Dry);
        assert!((drop - 0.54).abs() < 0.01, "{drop}");
        assert!(pad_mu(0.6, 1e4, PadSurface::Dry) >= 0.6 * PAD_MU_FLOOR - 1e-6);
        assert!(pad_mu(0.6, 10.0, PadSurface::Wet) < mu && pad_mu(0.6, 10.0, PadSurface::Dusty) < mu);
    }

    #[test]
    fn the_limit_surface_couples_force_and_twist() {
        let (mu, n) = (0.6, 10.0);
        assert!((limit_use(mu * n, 0.0, mu, n) - 1.0).abs() < 1e-5);
        let m_max = torsion_radius_m(n) * mu * n;
        assert!((limit_use(0.0, m_max, mu, n) - 1.0).abs() < 1e-5);
        assert!((tangential_capacity(0.5 * m_max, mu, n) / (mu * n) - 0.75f32.sqrt()).abs() < 1e-4);
        assert!((slip_direction(1.0, 0.0, mu, n)[0] - 1.0).abs() < 1e-6);
        assert!((slip_direction(0.0, 0.01, mu, n)[1] - 1.0).abs() < 1e-6);
        let s = 0.5f32.sqrt();
        let d = slip_direction(s * mu * n, s * m_max, mu, n);
        assert!((d[0] - s).abs() < 1e-5 && (d[1] - s).abs() < 1e-5, "{d:?}");
    }

    #[test]
    fn a_twist_past_one_pad_moves_to_the_others_and_past_all_of_them_turns_the_object() {
        let (light, heavy) = (1.0f32, 20.0f32);
        let m_light = torsion_radius_m(light) * 0.6 * light;
        let m_heavy = torsion_radius_m(heavy) * 0.6 * heavy;
        let tw = 0.5 * (m_light + m_heavy);
        let s = share_load(&[light, heavy], &[0.6, 0.6], 0.0, tw);
        assert!(!s.gross_slip, "{s:?}");
        assert!((s.pads[0].torsion_nm + s.pads[1].torsion_nm - tw).abs() < 1e-6, "the whole twist is held: {s:?}");
        assert!(s.pads.iter().zip([m_light, m_heavy]).all(|(p, m)| p.torsion_nm <= m + 1e-7));
        let neg = share_load(&[light, heavy], &[0.6, 0.6], 0.0, -tw);
        assert!(neg.pads.iter().all(|p| p.torsion_nm <= 0.0) && !neg.gross_slip);
        let over = share_load(&[light, heavy], &[0.6, 0.6], 0.0, 1.1 * (m_light + m_heavy));
        assert!(over.gross_slip && over.stick_min == 0.0, "{over:?}");
    }

    #[test]
    fn a_pad_at_its_torsion_limit_sticks_nowhere_and_the_sensors_feel_it() {
        let n = 10.0f32;
        let mus = [0.45f32, 0.75];
        let m = |mu: f32| torsion_radius_m(n) * mu * n;
        let tw = 0.5 * (2.0 * m(mus[0]) + m(mus[0]) + m(mus[1]));
        for load in [0.0f32, 1.0] {
            let s = share_load(&[n, n], &mus, load, tw);
            assert!(!s.gross_slip && (s.pads[0].torsion_nm - m(mus[0])).abs() < 1e-6, "{load}: {s:?}");
            assert!(s.pads[0].state == SlipState::Slip && s.pads[0].stick < 0.05, "{load}: {:?}", s.pads[0]);
            assert!(s.stick_min < STICK_ALARM, "{load}: {}", s.stick_min);
            for p in &s.pads {
                assert!((p.state == SlipState::Stick) == (p.stick >= STICK_ALARM && p.cone_use < 0.999), "{p:?}");
            }
        }
        let s = share_load(&[10.0], &[0.6], 3.0, 0.0);
        assert!((s.pads[0].stick - stick_ratio(3.0, 6.0)).abs() < 1e-5, "{:?}", s.pads[0]);
    }

    #[test]
    fn the_stick_zone_shrinks_to_nothing_at_slip_and_tells_the_friction() {
        assert_eq!(stick_ratio(0.0, 5.0), 1.0);
        assert!(stick_ratio(5.0, 5.0) < 1e-6);
        let phi = stick_ratio(3.0, 6.0);
        assert!((phi - 0.5f32.powf(2.0 / 3.0)).abs() < 1e-5);
        assert!((mu_from_stick(3.0, 10.0, phi) - 0.6).abs() < 1e-4);
    }

    #[test]
    fn a_load_shares_by_stiffness_and_the_overflow_moves_on() {
        let s = share_load(&[10.0, 10.0], &[0.6, 0.6], 6.0, 0.0);
        assert!(!s.gross_slip && (s.pads[0].ft - 3.0).abs() < 1e-4 && s.pads[0].state == SlipState::Stick);
        let s = share_load(&[2.0, 20.0], &[0.3, 0.6], 8.0, 0.0);
        assert!((s.pads[0].ft - 0.6).abs() < 1e-4 && s.pads[0].state == SlipState::Slip && !s.gross_slip, "{s:?}");
        assert!((s.pads[0].ft + s.pads[1].ft - 8.0).abs() < 1e-3);
        assert!((s.pads[0].slip_dir[0] - 1.0).abs() < 1e-5 && s.pads[0].slip_dir[1].abs() < 1e-5, "{:?}", s.pads[0].slip_dir);
        let s = share_load(&[5.0, 5.0], &[0.5, 0.5], 6.0, 0.0);
        assert!(s.gross_slip && s.stick_min == 0.0);
        let a = share_load(&[10.0, 10.0], &[0.6, 0.6], 0.0, 0.0).capacity_n;
        let b = share_load(&[10.0, 10.0], &[0.6, 0.6], 0.0, 0.02).capacity_n;
        assert!(b < a, "{b} {a}");
    }

    #[test]
    fn a_patch_creeps_before_it_slides() {
        let (mu, n) = (0.6, 2.0);
        assert_eq!(preslip_displacement_m(0.0, mu, n), 0.0);
        let full = preslip_displacement_m(mu * n, mu, n);
        assert!((full - 3.0 * mu * n / (16.0 * tangential_modulus_pa() * patch_radius_m(n))).abs() < 1e-7);
        let small = preslip_displacement_m(0.01 * mu * n, mu, n);
        assert!((small * tangential_stiffness(n) / (0.01 * mu * n) - 1.0).abs() < 0.02, "{small}");
        assert!(preslip_displacement_m(0.5 * mu * n, mu, n) < 0.5 * full, "the creep speeds up toward the slide");
        assert!(patch_area_m2(2.0) > patch_area_m2(1.0) && (patch_area_m2(8.0) / patch_area_m2(1.0) - 4.0).abs() < 0.01);
    }

    #[test]
    fn the_grip_loop_answers_a_pull_after_its_sensor_sees_it() {
        let hold = |rate: f32, plateau: f32| ramp_hold(0.6, PadSurface::Dry, 0.6, 2.0, PRELOAD_N, 1, rate, plateau, 0.5, 0.0875);
        let lat: Vec<f32> = [2.0, 4.0, 8.0, 32.0].iter().map(|&r| hold(r, 2.0).latency_s.unwrap()).collect();
        assert!(lat.iter().all(|l| *l >= detect_delay_s()), "{lat:?}");
        assert!(lat.windows(2).all(|w| w[1] <= w[0]) && lat[0] > lat[3], "{lat:?}");
        for p in [1.0, 2.0, 4.0] {
            let h = hold(4.0, p);
            assert!(!h.slipped && h.safety_margin > 0.3 && h.safety_margin < 0.9, "{h:?}");
            assert!(h.grip_n > PRELOAD_N + 2.0 * p / (PINCH_FACES * 0.6) - 0.05, "{h:?}");
        }
        let blind = ramp_hold(0.6, PadSurface::Dry, 0.6, 2.0, PRELOAD_N, 1, 4.0, 4.0, 0.5, 1e9);
        assert!(blind.slipped, "{blind:?}");
        let crawl = ramp_hold(0.6, PadSurface::Dry, 0.6, 2.0, PRELOAD_N, 1, 1e-9, 4.0, 0.5, 0.0875);
        assert!(!crawl.slipped && crawl.load_n < 1e-3 && crawl.latency_s.is_none(), "{crawl:?}");
    }

    #[test]
    fn finger_contacts_serialise_as_a_list() {
        let mut c = FingerContacts::default();
        c.push(FingerContact { finger: 1, fn_n: 3.0, ..FingerContact::default() });
        let j = serde_json::to_value(c).unwrap();
        assert_eq!(j.as_array().unwrap().len(), 1);
        assert_eq!(j[0]["finger"], 1);
    }

    fn legs() -> Body {
        Body::new(&BodyPlan::uniform(Tier::Mid), BaseKind::HumanFoot, FingerCount::Five)
    }

    #[test]
    fn a_flat_foot_loads_the_heel_behind_and_the_forefoot_ahead() {
        let b = legs();
        let s = SoleModel::of(&b);
        assert!(s.toe_kp > 1.0 && s.toe_kp < 10.0 && s.toe_share > 0.0 && s.toe_share < 0.5, "{s:?}");
        let fl = s.foot_len_m;
        let w = 300.0;
        let back = sole_contact(&s, 0.0, w, [0.0; 2], 0.0, 0.0, 0.0, 0.7);
        let sum: f32 = back.patches.iter().map(|p| p.normal_n).sum();
        assert!((sum - w).abs() < 0.5, "{sum}");
        assert!(!back.rocker && back.patches[0].normal_n > 0.5 * w, "{:?}", back.patches);
        let m: f32 = back.patches.iter().map(|p| p.normal_n * 0.5 * (p.x0 + p.x1)).sum::<f32>() / w;
        assert!(m.abs() < 0.25 * fl, "{m}");
        let fore = sole_contact(&s, 0.0, w, [0.0; 2], 0.0, 0.40 * fl, 0.0, 0.7);
        assert!(!fore.rocker && !fore.toe_held && fore.patches[0].normal_n < 1e-3 && fore.patches[2].normal_n > 0.5 * w, "{:?}", fore.patches);
        let push = sole_contact(&s, 0.0, w, [30.0, 0.0], 0.0, 0.60 * fl, 20.0, 0.7);
        assert!(push.rocker && push.patches[0].normal_n == 0.0 && push.patches[3].normal_n > 0.5 * w, "{push:?}");
        assert!(push.toe_deg > 0.0 && push.toe_torque_nm > 0.0);
        assert!(push.loaded[0] >= 0.30 * fl - 1e-4, "the support is the forefoot: {:?}", push.loaded);
        let held = sole_contact(&s, 0.0, w, [30.0, 0.0], 0.0, 0.60 * fl, 0.0, 0.7);
        assert!(!held.rocker && held.toe_held && held.toe_deg == 0.0 && held.patches[3].normal_n > 0.0, "{held:?}");
        assert!(push.cone_use < 1.0 && push.patches.iter().all(|p| !p.slipping));
        let slide = sole_contact(&s, 0.0, w, [250.0, 0.0], 0.0, 0.0, 0.0, 0.7);
        assert!(slide.cone_use > 1.0 && slide.patches.iter().filter(|p| p.normal_n > 0.0).all(|p| p.slipping));
        let blind = sole_contact(&s, 0.0, w, [250.0, 20.0], 0.0, 0.0, 0.0, 0.0);
        let sx: f32 = blind.patches.iter().map(|p| p.shear[0]).sum();
        assert!((sx - 250.0).abs() < 0.5 && blind.patches.iter().all(|p| !p.slipping) && blind.cone_use == 0.0, "{blind:?}");
    }

    #[test]
    fn the_rear_foot_turns_about_the_toe_hinge() {
        assert_eq!(rocker_ankle_shift(0.2, 0.08, 0.0), [0.0, 0.0]);
        let s = rocker_ankle_shift(0.2, 0.08, 30.0);
        assert!(s[1] > 0.035 && s[1] < 0.05 && s[0] > 0.0, "{s:?}");
    }

    #[test]
    #[ignore]
    fn print_soles() {
        for tier in [Tier::Budget, Tier::Mid, Tier::Pro] {
            let b = Body::new(&BodyPlan::uniform(tier), BaseKind::HumanFoot, FingerCount::Five);
            let s = SoleModel::of(&b);
            println!("[SOLE] {tier:?} {:.1} kg, CoM {:.3} m: Kp {:.2} N·m/rad, Kd/Kp {:.3} s, toe share {:.3}, k {:.3e} N/m/m", b.mass_kg(), b.com_height_m(), s.toe_kp, s.toe_kd / s.toe_kp, s.toe_share, s.k_per_m);
        }
    }

    #[test]
    fn the_toe_spring_settles_with_its_time_constant() {
        let s = SoleModel::of(&legs());
        let m = 2.0;
        let mut d = 0.0;
        for _ in 0..200 {
            d = s.toe_step_deg(d, m, 0.02);
        }
        assert!((d - s.static_toe_deg(m)).abs() < 0.05, "{d} vs {}", s.static_toe_deg(m));
        assert_eq!(s.toe_step_deg(10.0, -1.0, 0.02), 0.0, "the heel comes down");
    }
}
