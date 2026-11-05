//! Sub-workers move by the same code as the first worker: the same follower, the same pose checks, the same collisions,
//! the same load on the same physics. That code reads the robot from the world's own fields, so a sub-worker takes the
//! first worker's place in them for the length of its step, and hands them back.

use super::*;
use crate::crew::Worker;

/// How a sub-worker's step of a move ended, and what its wheels slipped on it (for its odometry).
pub(crate) struct MoveStep {
    pub done: Option<(bool, Option<String>)>,
    pub lost: bool,
    pub slip_over: f32,
}

impl World {
    /// Runs `f` with sub-worker `i` in the first worker's place.
    pub(crate) fn as_worker<R>(&mut self, i: usize, f: impl FnOnce(&mut World) -> R) -> R {
        let id = self.crew[i].id;
        self.turn_others = self.robot_outlines().into_iter().filter(|o| o.id != id && o.id != traffic::RIDER_ID + id).collect();
        let odom_over = self.odom_over;
        self.seat(i);
        self.crew_turn = Some(id);
        *self.limits.lock().unwrap_or_else(|e| e.into_inner()) = None;
        let r = f(self);
        *self.limits.lock().unwrap_or_else(|e| e.into_inner()) = None;
        self.crew_turn = None;
        self.turn_others.clear();
        self.odom_over = odom_over;
        self.seat(i);
        r
    }

    /// Swaps sub-worker `i` with whoever is in the first worker's place.
    fn seat(&mut self, i: usize) {
        let w: &mut Worker = &mut self.crew[i];
        let mut a = Agent { x: w.x, z: w.z, yaw: w.yaw, carried: std::mem::take(&mut w.carried), energy_used_j: w.energy_j, strikes: 0, speed_mps: w.speed_mps, grip_margin: None, terrain_margin: None, strike_acc: 0.0 };
        std::mem::swap(&mut self.agent, &mut a);
        (w.x, w.z, w.yaw, w.carried, w.energy_j, w.speed_mps) = (a.x, a.z, a.yaw, a.carried, a.energy_used_j, a.speed_mps);
        std::mem::swap(&mut self.body, &mut w.body);
        std::mem::swap(&mut self.hand.base, &mut w.spec);
        std::mem::swap(&mut self.walk, &mut w.walk);
        std::mem::swap(&mut self.ride, &mut w.ride);
        std::mem::swap(&mut self.stride, &mut w.stride);
        std::mem::swap(&mut self.fallen_until, &mut w.fallen_until);
    }

    /// A route for sub-worker `i` to within `reach_m` of (`x`, `z`), round the cells of `extra`, ready to follow.
    pub(crate) fn worker_move(&mut self, i: usize, x: f32, z: f32, reach_m: f32, extra: &HashSet<Cell>) -> Option<MoveState> {
        self.as_worker(i, |world| {
            let goal = Goal::Within { x, z, reach_m };
            let mut both = extra.clone();
            both.extend(world.loose_cells(goal));
            let plan = world.plan_route_avoiding(goal, &both).or_else(|| world.plan_route_avoiding(goal, extra))?;
            let plan = world.fit_plan(plan, goal, &both);
            let mut follow = world.start_follow(&plan.edges);
            follow.stop_within = Some(((x, z), reach_m));
            Some(MoveState::new(plan.edges, follow, goal))
        })
    }

    /// What the eye of robot `id` sees (1 is the first worker): each robot's own head camera, as `render` draws the
    /// first worker's.
    pub fn render_unit(&mut self, id: u32, cam: &crate::render::Camera, fork: bool) -> Option<crate::render::Frame> {
        let frame = |w: &mut World| {
            w.fork_cam = fork && noto_core::act::fork::applies(&w.body);
            let f = w.render(cam);
            w.fork_cam = false;
            f
        };
        if id == 1 {
            return Some(frame(self));
        }
        let i = self.crew.iter().position(|w| w.id == id)?;
        Some(self.as_worker(i, frame))
    }

    /// Puts robot `id` (1 is the first worker) in another body where it stands, without touching the rest of the world.
    /// Refused when the new body does not fit there.
    pub fn swap_body(&mut self, id: u32, plan: BodyPlan, base: noto_core::BaseKind) -> Result<(), String> {
        let fingers = self.cfg.noto.fingers;
        let body = Body::new(&plan, base, fingers);
        let fp = Footprint::of(&body);
        if id == 1 {
            let (x, z, yaw) = (self.agent.x, self.agent.z, self.agent.yaw);
            let room = fp.room(&self.clear_map(), x, z, yaw);
            if room < -1e-3 {
                return Err(format!("지금 자리에서는 새 몸이 주변에 {:.0} cm 닿습니다: 넓은 곳으로 옮긴 뒤 다시 적용하세요", -room * 100.0));
            }
            self.cancel();
            let mut cfg = self.cfg.noto;
            cfg.base = base;
            self.hand = cfg.build().map_err(|e| e.to_string())?.scaled(plan.k());
            self.cfg.noto = cfg;
            self.cfg.body = plan;
            self.body = body;
            self.grasps = ItemKind::ALL.iter().map(|&k| top_grasp_info(&self.hand, k, 1.0)).collect();
            *self.limits.lock().unwrap_or_else(|e| e.into_inner()) = None;
            self.walk = WalkState::default();
            self.ride = CargoRide::default();
            self.stride.crouch_m = 0.0;
            self.motion = None;
            return Ok(());
        }
        let i = self.crew.iter().position(|w| w.id == id).ok_or_else(|| format!("작업자 {id}가 없습니다"))?;
        let w = &self.crew[i];
        let room = fp.room(&self.clear_map(), w.x, w.z, w.yaw);
        if room < -1e-3 {
            return Err(format!("지금 자리에서는 새 몸이 주변에 {:.0} cm 닿습니다: 넓은 곳으로 옮긴 뒤 다시 적용하세요", -room * 100.0));
        }
        self.crew[i].rebody(plan, base, fingers);
        Ok(())
    }

    /// One tick of sub-worker `i` on the move, and of its load.
    pub(crate) fn worker_step(&mut self, i: usize, m: &mut MoveState, dt: f32, events: &mut Vec<TimedEvent>) -> MoveStep {
        let id = self.crew[i].id;
        let mut local = Vec::new();
        let step = self.as_worker(i, |world| {
            let (mut left, mut guard, mut done) = (dt, 0, None);
            while left > 1e-6 && done.is_none() && guard < 64 {
                guard += 1;
                let used = world.advance_follow(m, left, &mut local, &mut done);
                left -= used.max(1e-6);
            }
            let lost = world.ride_cargo(dt, true, id, &mut local);
            MoveStep { done, lost, slip_over: std::mem::take(&mut world.odom_over) }
        });
        // The first worker's driver listens to these events: of a sub-worker's, only what it did to the world.
        events.extend(local.into_iter().filter(|e| matches!(e.event, WorldEvent::Bump { .. } | WorldEvent::CargoSlip { .. } | WorldEvent::CargoLost { .. } | WorldEvent::StepSeen { .. } | WorldEvent::StepTaken { .. } | WorldEvent::Hold { .. })));
        step
    }
}
