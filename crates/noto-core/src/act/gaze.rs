//! Gaze before the hand. People look at what they are about to pick up before the hand moves, and keep looking at it
//! while they reach: the eyes land on the target about 0.5–1 s ahead of the hand and stay there until it is in hand
//! (Land & Hayhoe, Vision Res. 41, 2001; Pelz & Canosa, Vision Res. 41, 2001, "look-ahead fixations"). A robot
//! whose eyes are a head camera does the same with its neck: it pitches the head until the object sits in the middle of
//! the image (the waist takes what the neck cannot), holds still long enough to see it, and keeps the object in view
//! while the body bends down to it.

use crate::balance::Body;
use crate::kin::{self, KinLoads, KinPose};

use super::{min_jerk, ActEvent, ActKind, ActRollout, ActSample, Outcome, Part, MIN_JERK_PEAK, SAMPLE_S};

pub const HEAD_CAMERA_MOUNT_DEG: f32 = -12.0;
pub const HEAD_PEAK_DPS: f32 = 150.0;
pub const LOOK_MIN_S: f32 = 0.15;
pub const FIXATE_S: f32 = 0.35;
/// Off-the-ball scanning: a footballer without the ball turns the head away from the play every second or so to see
/// who is around before it matters, quick turns held only long enough to see. A robot on the move with nothing in its
/// hands does the same with its neck while its body goes on the way it goes: every `SCAN_PERIOD_S` the head turns
/// `SCAN_PAN_DEG` to one side (the other side the next time), holds a fixation and comes back [추정: the period is the
/// intermittent look cadence of the robot's vision, 1.5 s; the turn keeps the way ahead inside a 70° camera's view].
pub const SCAN_PERIOD_S: f32 = 1.5;
pub const SCAN_PAN_DEG: f32 = 25.0;

/// The neck's pan of an off-the-ball scan `t_s` into the schedule (deg, left positive): ahead most of the period, then
/// a min-jerk turn out at the neck's speed, a fixation and a turn back, ending with the period. Each period turns to the
/// other side. Zero for a head that cannot pan, or not fast enough to be back within a period.
pub fn scan_pan_deg(body: &Body, t_s: f32) -> f32 {
    let pan_max = body.head_range_deg().1;
    if pan_max <= 0.0 || t_s < 0.0 {
        return 0.0;
    }
    let pan = SCAN_PAN_DEG.min(pan_max);
    let peak = body.head.neck.speed_rad_s.to_degrees().min(HEAD_PEAK_DPS).max(30.0);
    let turn = (MIN_JERK_PEAK * pan / peak).max(LOOK_MIN_S);
    let glance = 2.0 * turn + FIXATE_S;
    if glance >= SCAN_PERIOD_S {
        return 0.0;
    }
    let k = (t_s / SCAN_PERIOD_S).floor();
    let tau = t_s - k * SCAN_PERIOD_S - (SCAN_PERIOD_S - glance);
    if tau < 0.0 {
        return 0.0;
    }
    let side = if (k as i64).rem_euclid(2) == 0 { 1.0 } else { -1.0 };
    let u = if tau < turn {
        min_jerk(tau / turn)
    } else if tau < turn + FIXATE_S {
        1.0
    } else {
        min_jerk((1.0 - (tau - turn - FIXATE_S) / turn).clamp(0.0, 1.0))
    };
    side * pan * u
}

pub fn can_aim(body: &Body) -> bool {
    body.head_range_deg().0 > 0.0
}

/// The neck angle that puts the point (`x` forward, `z` up, body frame) in the middle of the head camera's image.
pub fn aim_neck(body: &Body, pose: &KinPose, x: f32, z: f32) -> f32 {
    let max = body.head_range_deg().0;
    let trunk = if body.head_on_trunk() { pose.trunk_deg() } else { 0.0 };
    let mut neck = pose.neck_deg;
    for _ in 0..3 {
        let eye = kin::head_camera(body, &KinPose { neck_deg: neck, ..*pose }).pos;
        let below = (eye[2] - z).atan2(x - eye[0]).to_degrees();
        neck = (below + HEAD_CAMERA_MOUNT_DEG - trunk).clamp(-max, max);
    }
    neck
}

/// How far the aim falls short of the centre of the image when the neck is at its limit (deg, 0 when it reaches).
pub fn aim_shortfall_deg(body: &Body, pose: &KinPose, x: f32, z: f32) -> f32 {
    let trunk = if body.head_on_trunk() { pose.trunk_deg() } else { 0.0 };
    let eye = kin::head_camera(body, pose).pos;
    let below = (eye[2] - z).atan2(x - eye[0]).to_degrees();
    (below + HEAD_CAMERA_MOUNT_DEG - trunk - pose.neck_deg).max(0.0)
}

/// The pose that looks at the point from where `from` stands: the neck first, the waist for the rest.
pub fn look_pose(body: &Body, from: &KinPose, x: f32, z: f32) -> KinPose {
    let mut p = KinPose { neck_deg: aim_neck(body, from, x, z), ..*from };
    let short = aim_shortfall_deg(body, &p, x, z);
    if short > 1.0 {
        let rom = kin::Rom::of(body);
        if rom.waist_max > 0.0 {
            p.waist_deg = (p.waist_deg + short).min(rom.waist_max);
        } else if body.legged() && rom.hip_flex_max > 0.0 {
            p.pelvis_deg = (p.pelvis_deg + short).min(0.5 * rom.hip_flex_max);
        }
        p.neck_deg = aim_neck(body, &p, x, z);
    }
    p
}

pub fn shift_s(body: &Body, from: &KinPose, to: &KinPose) -> f32 {
    let peak = body.head.neck.speed_rad_s.to_degrees().min(HEAD_PEAK_DPS).max(30.0);
    let turn = (to.neck_deg - from.neck_deg).abs().max((to.trunk_deg() - from.trunk_deg()).abs() * 2.0);
    (MIN_JERK_PEAK * turn / peak).max(LOOK_MIN_S)
}

fn sample(body: &Body, t: f32, phase: &'static str, pose: KinPose, object: [f32; 3]) -> ActSample {
    let r = kin::solve_fast(body, &pose, &KinLoads::default());
    ActSample { t, phase, pose, loads: KinLoads::default(), margin: r.margin, zmp: r.zmp, com: r.com, peak_ratio: r.worst().map_or(0.0, |j| j.ratio), grip: None, object: Some(object) }
}

/// Head shift ("look") and the hold to see ("check"), appended after `t0`. Returns the pose that looks and the end time.
pub fn lead(body: &Body, from: &KinPose, object: [f32; 3], t0: f32, samples: &mut Vec<ActSample>, events: &mut Vec<ActEvent>) -> (KinPose, f32) {
    let to = look_pose(body, from, object[0], object[2]);
    let t_look = shift_s(body, from, &to);
    let n = (t_look / SAMPLE_S).ceil().max(1.0) as usize;
    for i in 0..=n {
        let u = min_jerk(i as f32 / n as f32);
        samples.push(sample(body, t0 + i as f32 * t_look / n as f32, "look", KinPose::lerp(from, &to, u), object));
    }
    let t_fix = t0 + t_look;
    let waist = (to.trunk_deg() - from.trunk_deg()).abs() > 0.5;
    let label = if waist {
        format!("시선 고정: 목 {:.0}° · 허리 {:.0}° 숙여 물건을 화면 가운데에", to.neck_deg, to.trunk_deg())
    } else {
        format!("시선 고정: 목 {:.0}° 숙여 물건을 화면 가운데에", to.neck_deg)
    };
    events.push(ActEvent::new(t_fix, "gaze_fix", label, &[Part::Neck], Some(to.neck_deg), true));
    let m = (FIXATE_S / SAMPLE_S).ceil().max(1.0) as usize;
    for i in 1..=m {
        samples.push(sample(body, t_fix + i as f32 * FIXATE_S / m as f32, "check", to, object));
    }
    let t_end = t_fix + FIXATE_S;
    events.push(ActEvent::new(t_end, "vision_check", format!("응시 {:.2} s: 머리 카메라로 물건을 확인", FIXATE_S), &[Part::Neck], Some(FIXATE_S), true));
    (to, t_end)
}

/// A look by itself: from `from`, at the point (`x` forward, `y` left, `z` up) — the act that a pick-up starts from.
pub fn look(body: &Body, from: &KinPose, object: [f32; 3]) -> ActRollout {
    if !can_aim(body) {
        let s = vec![sample(body, 0.0, "check", *from, object)];
        return ActRollout { kind: ActKind::Look, samples: s, events: vec![], outcome: Outcome::fail("no_neck", "목·카메라 틸트 관절이 없어 고개를 숙여 볼 수 없음"), duration_s: 0.0, metrics: vec![], notes: vec![] };
    }
    let mut from = *from;
    if object[1].abs() > 1e-3 {
        let pan_max = body.head_range_deg().1;
        from.neck_pan_deg = object[1].atan2(object[0].max(0.05)).to_degrees().clamp(-pan_max, pan_max);
    }
    let (mut samples, mut events) = (Vec::new(), Vec::new());
    let (to, t) = lead(body, &from, object, 0.0, &mut samples, &mut events);
    let metrics = vec![("neck_deg", to.neck_deg), ("trunk_deg", to.trunk_deg()), ("look_s", t - FIXATE_S), ("fixate_s", FIXATE_S), ("duration_s", t)];
    ActRollout { kind: ActKind::Look, samples, events, outcome: Outcome::ok(), duration_s: t, metrics, notes: vec![] }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::BaseKind;
    use crate::hand::FingerCount;
    use crate::parts::BodyPlan;

    fn body(base: BaseKind) -> Body {
        Body::new(&BodyPlan::default(), base, FingerCount::Five)
    }

    #[test]
    fn the_head_goes_down_until_the_object_is_in_the_middle_of_the_image() {
        let b = body(BaseKind::HumanFoot);
        let stand = b.stand(0.0);
        let p = look_pose(&b, &stand, 1.0, 0.075);
        let hc = kin::head_camera(&b, &p);
        let axis = HEAD_CAMERA_MOUNT_DEG - hc.pitch_down_deg;
        let to = -(hc.pos[2] - 0.075).atan2(1.0 - hc.pos[0]).to_degrees();
        assert!((axis - to).abs() < 1.0, "camera axis {axis:.1}° vs object {to:.1}°");
        assert!(p.neck_deg > 25.0 && p.neck_deg <= kin::NECK_PITCH_MAX_DEG + 1e-3, "{}", p.neck_deg);
        assert_eq!(p.waist_deg, stand.waist_deg, "a metre away the neck alone reaches");
        let close = look_pose(&b, &stand, 0.35, 0.075);
        assert!(close.trunk_deg() > stand.trunk_deg(), "at the feet the waist helps: {close:?}");
    }

    #[test]
    fn a_look_shifts_the_head_then_holds_to_see() {
        let b = body(BaseKind::HumanFoot);
        let r = look(&b, &b.stand(0.0), [0.9, 0.0, 0.075]);
        assert!(r.outcome.ok);
        let first_check = r.samples.iter().position(|s| s.phase == "check").unwrap();
        assert!(r.samples[..first_check].iter().all(|s| s.phase == "look"));
        assert!(r.samples.windows(2).all(|w| w[1].t >= w[0].t && w[1].pose.neck_deg >= w[0].pose.neck_deg - 1e-4), "the head goes one way");
        let look_s = r.metric("look_s").unwrap();
        assert!(look_s > 0.2 && look_s < 1.0, "a 30–40° head shift takes a few tenths of a second: {look_s}");
        assert!((r.duration_s - look_s - FIXATE_S).abs() < 1e-4);
        assert!(r.events.iter().any(|e| e.kind == "gaze_fix") && r.events.iter().any(|e| e.kind == "vision_check"));
        let side = look(&b, &b.stand(0.0), [0.6, 0.5, 0.075]);
        assert!(side.samples.last().unwrap().pose.neck_pan_deg > 20.0, "it turns the head to an object beside it");
    }

    #[test]
    fn an_off_the_ball_scan_glances_to_each_side_in_turn_and_mostly_looks_ahead() {
        let b = body(BaseKind::HumanFoot);
        let pans: Vec<f32> = (0..300).map(|i| scan_pan_deg(&b, i as f32 * 0.01)).collect();
        // Ahead at the start of a period, a glance to the left at the end of the first, to the right in the second.
        assert_eq!(pans[0], 0.0);
        let first = pans[..150].iter().copied().fold(f32::MIN, f32::max);
        let second = pans[150..].iter().copied().fold(f32::MAX, f32::min);
        assert!((first - SCAN_PAN_DEG).abs() < 1e-3 && (second + SCAN_PAN_DEG).abs() < 1e-3, "{first} {second}");
        assert!(pans[..150].iter().all(|p| *p >= 0.0) && pans[150..].iter().all(|p| *p <= 0.0));
        // The head is back ahead by the end of each period, and looks ahead for well over a third of it.
        assert!(scan_pan_deg(&b, SCAN_PERIOD_S - 1e-4).abs() < 0.5);
        let ahead = pans.iter().filter(|p| p.abs() < 1e-3).count() as f32 / pans.len() as f32;
        assert!(ahead > 0.35, "{ahead}");
        // It turns no faster than the neck turns a look.
        let rate = pans.windows(2).map(|w| (w[1] - w[0]).abs() / 0.01).fold(0.0f32, f32::max);
        assert!(rate <= 1.05 * HEAD_PEAK_DPS, "{rate}");
        // A head with no neck does not scan.
        let fixed = Body::new(&BodyPlan { morph: crate::morph::Morphology::named("wheel_noneck").unwrap(), ..BodyPlan::default() }, BaseKind::Wheel, FingerCount::Five);
        assert!((0..300).all(|i| scan_pan_deg(&fixed, i as f32 * 0.01) == 0.0));
    }
}
