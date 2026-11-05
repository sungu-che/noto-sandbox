// 모의 프레임 렌더러: 오프스크린 캔버스에 단순한 가짜 3D 장면(하늘 그라데이션, 지면 체크 무늬, 나무·바위·광석 블록)을
// 에이전트 기준으로 그리고 WebP 로 인코딩한다. 실제 world-sim 의 레이캐스터가 아니다.
// 탑뷰는 지도 전체를 요청 크기에 맞춰 그리고 로봇마다 프레임 좌표(units)를 돌려준다. unit_frame · depth_view 도 여기서 그린다.
import { DETECT_LABEL, procedureView, visibleObjects, world } from './mock-world.js';
import { mockFocus, mockOverlay, unitRows } from './mock-units.js';

// 1×1 무손실 WebP (캔버스를 쓸 수 없는 환경용 대체 프레임)
const BLANK_WEBP = 'UklGRhoAAABXRUJQVlA4TA0AAAAvAAAAEAcQERGIiP4HAA==';
let cv = null;

const clampInt = (v, lo, hi) => Math.max(lo, Math.min(hi, Math.round(Number(v) || lo)));
const hash = (s) => {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return (h >>> 0) / 4294967296;
};
const DROP_COLOR = { parcel: '#d49b54', wood: '#b98a4e', iron_ore: '#c4764a', coal: '#1c1c20', stone: '#9a9ca2' };

function detection(o, bbox, w, h, tick) {
  const [id, ko] = o.drop ? [`${o.item_id}_drop`, `${o.label_ko} (드롭)`] : DETECT_LABEL[o.kind];
  const base = 0.97 - o.dist * 0.03 + (hash(o.id) - 0.5) * 0.08 + Math.sin(tick / 9 + hash(o.id) * 6) * 0.02;
  const score = Math.max(0.12, Math.min(0.99, base));
  const accepted = score >= 0.55;
  const [x0, y0, x1, y1] = bbox;
  return {
    label_id: id, label_ko: ko, score: Number(score.toFixed(3)), accepted,
    reason: accepted ? `점수 ${score.toFixed(2)} ≥ 0.55` : `보류: 점수 ${score.toFixed(2)} < 0.55 (거리 ${o.dist.toFixed(1)} m)`,
    bbox: [Math.max(0, x0 / w), Math.max(0, y0 / h), Math.min(1, x1 / w), Math.min(1, y1 / h)].map((v) => Number(v.toFixed(4))),
  };
}

function drawObject(ctx, o, sx, gy, s) {
  const col = (c) => { ctx.fillStyle = c; };
  if (o.kind === 'tree') {
    col('#5b3d24'); ctx.fillRect(sx - 0.15 * s, gy - 1.4 * s, 0.3 * s, 1.4 * s);
    col('#2f7a3c'); ctx.beginPath(); ctx.arc(sx, gy - 2.2 * s, 1.0 * s, 0, Math.PI * 2); ctx.fill();
    col('#3e9650'); ctx.beginPath(); ctx.arc(sx - 0.25 * s, gy - 2.4 * s, 0.55 * s, 0, Math.PI * 2); ctx.fill();
    return [sx - 1.0 * s, gy - 3.2 * s, sx + 1.0 * s, gy];
  }
  if (o.kind === 'rock' || o.kind === 'iron_ore' || o.kind === 'coal_ore') {
    const body = { rock: '#7d7f85', iron_ore: '#6d6a68', coal_ore: '#3a3a40' }[o.kind];
    col(body); ctx.beginPath();
    ctx.moveTo(sx - 0.6 * s, gy); ctx.lineTo(sx - 0.45 * s, gy - 0.6 * s); ctx.lineTo(sx - 0.1 * s, gy - 0.85 * s);
    ctx.lineTo(sx + 0.4 * s, gy - 0.7 * s); ctx.lineTo(sx + 0.62 * s, gy); ctx.closePath(); ctx.fill();
    if (o.kind !== 'rock') {
      col(o.kind === 'iron_ore' ? '#c4764a' : '#0d0d10');
      for (let i = 0; i < 6; i++) ctx.fillRect(sx + (hash(o.id + i) - 0.5) * 0.9 * s, gy - (0.15 + hash(`${o.id}y${i}`) * 0.55) * s, 0.14 * s, 0.14 * s);
    }
    return [sx - 0.62 * s, gy - 0.85 * s, sx + 0.62 * s, gy];
  }
  if (o.kind === 'chest') {
    col('#8a5a2b'); ctx.fillRect(sx - 0.45 * s, gy - 0.5 * s, 0.9 * s, 0.5 * s);
    col('#d6b65e'); ctx.fillRect(sx - 0.45 * s, gy - 0.3 * s, 0.9 * s, 0.08 * s);
    return [sx - 0.45 * s, gy - 0.5 * s, sx + 0.45 * s, gy];
  }
  if (o.drop) {
    const r = 0.22 * s;
    col(DROP_COLOR[o.item_id] ?? '#ddd');
    ctx.beginPath(); ctx.moveTo(sx, gy - 2 * r); ctx.lineTo(sx + r, gy - r); ctx.lineTo(sx, gy); ctx.lineTo(sx - r, gy - r); ctx.closePath(); ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.85)'; ctx.lineWidth = Math.max(1, s * 0.03); ctx.stroke();
    return [sx - r, gy - 2 * r, sx + r, gy];
  }
  return null;
}

function drawAgentView(ctx, w, h, W, dets) {
  const f = w * 0.9;
  const camH = W.agent.eye_m ?? 1.0;
  const pitchRad = ((W.agent.pitch_deg ?? -12.0) * Math.PI) / 180;
  const horizon = Math.round(h * 0.5 + Math.tan(pitchRad) * f);
  const skyBottom = Math.max(0, Math.min(h, horizon));
  const [ax, ay] = W.agent.pos;
  const yaw = W.agent.yaw - ((W.agent.pan_deg ?? 0) * Math.PI) / 180;
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  if (skyBottom > 0) {
    const sky = ctx.createLinearGradient(0, 0, 0, skyBottom);
    sky.addColorStop(0, '#27456b');
    sky.addColorStop(1, '#a5bed8');
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, w, skyBottom);
    ctx.fillStyle = '#40604d';
    for (let x = 0; x < w; x += 4) {
      const ang = yaw + (x - w / 2) / f;
      const hh = 12 + 12 * Math.sin(ang * 3 + W.seed) + 7 * Math.sin(ang * 7.3);
      ctx.fillRect(x, horizon - hh, 4, hh);
    }
  }
  const colors = [[['#42733c', '#3a6836'], ['#6b5a45', '#62523f']], [['#3b6a37', '#34602f'], ['#5f503d', '#584a38']]];
  const startY = Math.max(0, horizon + 1);
  for (let y = startY; y < h; y += 2) {
    const d = (camH * f) / Math.max(1, y - horizon);
    const fog = Math.min(1, d / 40);
    for (let x = 0; x < w; x += 8) {
      const ro = ((x + 4 - w / 2) / f) * d;
      const wx = ax + cy * d + sy * ro;
      const wy = ay + sy * d - cy * ro;
      const outside = wx < 0 || wy < 0 || wx > W.bounds[0] || wy > W.bounds[1];
      const parity = (Math.floor(wx / 1.5) + Math.floor(wy / 1.5)) & 1;
      ctx.fillStyle = colors[fog > 0.5 ? 1 : 0][outside ? 1 : 0][parity];
      ctx.fillRect(x, y, 8, 2);
    }
  }
  if (horizon > -h * 0.12 && horizon < h) {
    const haze = ctx.createLinearGradient(0, horizon, 0, horizon + h * 0.12);
    haze.addColorStop(0, 'rgba(165,190,216,0.75)');
    haze.addColorStop(1, 'rgba(165,190,216,0)');
    ctx.fillStyle = haze;
    ctx.fillRect(0, Math.max(0, horizon), w, h * 0.12);
  }
  const objs = visibleObjects(28);
  for (const o of objs) {
    if (o.forward < 0.25) continue;
    const s = f / o.forward;
    const sx = w / 2 + (o.right / o.forward) * f;
    const objZ = o.pos[2] ?? 0;
    const gy = horizon + (camH - objZ) * s;
    if (sx < -3 * s || sx > w + 3 * s || gy < -2 * s || gy - 3 * s > h) continue;
    ctx.globalAlpha = Math.max(0.4, 1 - o.forward / 36);
    if (o.kind === 'waypoint') {
      ctx.fillStyle = '#e8e8ea';
      ctx.fillRect(sx - 0.04 * s, gy - 3 * s, 0.08 * s, 3 * s);
      ctx.fillStyle = '#4aa3df';
      ctx.beginPath();
      ctx.moveTo(sx, gy - 3 * s);
      ctx.lineTo(sx + 0.9 * s, gy - 2.7 * s);
      ctx.lineTo(sx, gy - 2.4 * s);
      ctx.fill();
      ctx.globalAlpha = 1;
      continue;
    }
    const bbox = drawObject(ctx, o, sx, gy, s);
    ctx.globalAlpha = 1;
    if (bbox && o.forward < 20 && bbox[2] - bbox[0] >= 5 && bbox[2] > 0 && bbox[0] < w && bbox[3] > 0 && bbox[1] < h) {
      dets.push({ o, bbox });
    }
  }
  // 손에 들려 올라오는 물건(사람 몸): 시선이 따라가므로 화면 가운데 근처에 보인다.
  const held = W.agent.held;
  if (held && held.forward > 0.1) {
    const s = f / held.forward;
    drawObject(ctx, { kind: 'drop', drop: true, item_id: held.item_id, id: 'held' }, w / 2, horizon + (camH - held.z) * s, s);
  }
  ctx.fillStyle = 'rgba(30,30,34,0.85)';
  ctx.fillRect(w / 2 - w * 0.05, h - h * 0.07, w * 0.1, h * 0.07);
}

// 지게차 포크 카메라: 포크 뒤꿈치(차체 앞 0.27 m), 포크 위 0.07 m 에서 25° 아래를 본다. 아래쪽에 두 포크가 보인다.
function drawForkView(ctx, w, h, W, dets) {
  const a = W.agent;
  const keep = { pos: a.pos, eye_m: a.eye_m, pitch_deg: a.pitch_deg, pan_deg: a.pan_deg };
  a.pos = [a.pos[0] + 0.27 * Math.cos(a.yaw), a.pos[1] + 0.27 * Math.sin(a.yaw), a.pos[2] ?? 0];
  a.eye_m = (a.fork_m ?? 0.05) + 0.07;
  a.pitch_deg = -25;
  a.pan_deg = 0;
  try {
    drawAgentView(ctx, w, h, W, dets);
  } finally {
    Object.assign(a, keep);
  }
  ctx.fillStyle = '#3e4046';
  for (const side of [-1, 1]) {
    ctx.beginPath();
    ctx.moveTo(w / 2 + side * w * 0.08, h * 0.42);
    ctx.lineTo(w / 2 + side * w * 0.13, h * 0.42);
    ctx.lineTo(w / 2 + side * w * 0.42, h);
    ctx.lineTo(w / 2 + side * w * 0.2, h);
    ctx.closePath();
    ctx.fill();
  }
  // 포크가 물건 밑으로 들어가면 물건의 앞면이 화면을 채운다(가림).
  const blocked = a.fork_blocked ?? 0;
  if (blocked > 0) {
    ctx.fillStyle = DROP_COLOR[a.fork_blocked_item] ?? '#b98a4e';
    const bh = h * Math.min(1, blocked * 1.1);
    const bw = w * Math.min(1, 0.35 + blocked);
    ctx.fillRect((w - bw) / 2, 0, bw, bh);
    ctx.fillStyle = 'rgba(0,0,0,0.18)';
    for (let y = 6; y < bh; y += 9) ctx.fillRect((w - bw) / 2, y, bw, 2);
  }
}

/** 탑뷰 좌표: 지도 전체를 w×h 안에 비율 유지로 넣는다(+y 가 위쪽, 미니맵과 같다). */
function topMap(w, h, W) {
  const s = Math.min(w / W.bounds[0], h / W.bounds[1]);
  const ox = (w - W.bounds[0] * s) / 2;
  const oy = (h - W.bounds[1] * s) / 2;
  return { s, X: (x) => ox + x * s, Y: (y) => h - oy - y * s };
}

function drawRobot(ctx, r, X, Y, s) {
  ctx.save();
  ctx.translate(X(r.pos[0]), Y(r.pos[1]));
  ctx.rotate(-r.yaw);
  if (r.form === 'forklift') {
    ctx.fillStyle = '#3c3c44';
    for (const y of [-0.38, 0.28]) ctx.fillRect(0.4 * s, y * s, 0.95 * s, 0.1 * s);
    ctx.fillStyle = '#e8c028';
    ctx.fillRect(-1.0 * s, -0.5 * s, 1.4 * s, 1.0 * s);
  } else {
    ctx.fillStyle = '#ec8034';
    ctx.beginPath();
    ctx.arc(0, 0, 0.32 * s, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#fafafa';
    ctx.fillRect(0.2 * s, -0.06 * s, 0.32 * s, 0.12 * s);
  }
  ctx.restore();
}

function drawTopView(ctx, w, h, W) {
  const { s, X, Y } = topMap(w, h, W);
  ctx.fillStyle = '#16181a';
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = '#35553a';
  ctx.fillRect(X(0), Y(W.bounds[1]), W.bounds[0] * s, W.bounds[1] * s);
  ctx.strokeStyle = 'rgba(255,255,255,0.07)';
  ctx.lineWidth = 1;
  for (let g = 0; g <= Math.max(W.bounds[0], W.bounds[1]); g += 2) {
    ctx.beginPath();
    ctx.moveTo(X(g), Y(0));
    ctx.lineTo(X(g), Y(W.bounds[1]));
    ctx.moveTo(X(0), Y(g));
    ctx.lineTo(X(W.bounds[0]), Y(g));
    ctx.stroke();
  }
  ctx.font = `${Math.max(10, Math.round(s * 0.5))}px sans-serif`;
  const r = { tree: 0.9, rock: 0.6, iron_ore: 0.55, coal_ore: 0.55, chest: 0.4, drop: 0.25, waypoint: 0.5 };
  const col = { tree: '#2f7a3c', rock: '#7d7f85', iron_ore: '#c4764a', coal_ore: '#25252a', chest: '#d6b65e', waypoint: '#4aa3df' };
  for (const o of visibleObjects(Infinity)) {
    const x = X(o.pos[0]);
    const y = Y(o.pos[1]);
    const rad = (r[o.kind] ?? 0.4) * s;
    ctx.fillStyle = o.drop ? (DROP_COLOR[o.item_id] ?? '#ddd') : col[o.kind];
    if (o.kind === 'waypoint') {
      ctx.fillRect(x - rad * 0.6, y - rad * 0.6, rad * 1.2, rad * 1.2);
      ctx.fillStyle = '#d4d4d8';
      ctx.fillText(o.label_ko, x + rad, y - rad);
    } else {
      ctx.beginPath();
      ctx.arc(x, y, rad, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  for (const u of unitRows()) drawRobot(ctx, u, X, Y, s);
}

/**
 * world_frame.units: 탑뷰 프레임 안의 로봇 자리와 눈의 시야(sight: 그림 위 방향 x 오른쪽 · y 아래, 반각, 사거리).
 * 이 탑뷰는 y 가 위로 그려지므로 그림 위 방향은 −yaw 이고, 서브 작업자는 Rust 처럼 이동 중 1.5 s 마다 번갈아 곁눈질한다.
 */
function topUnits(w, h, W) {
  const { X, Y } = topMap(w, h, W);
  return unitRows().map(({ pos, yaw, battery_ratio, rig_id, rig_name_ko, ...u }) => {
    const beat = ((W.tick ?? 0) * 0.1 + u.id * 0.61) % 3;
    const glance = !u.first && u.moving ? (beat > 1.1 && beat < 1.5 ? 25 : beat > 2.6 ? -25 : 0) : 0;
    // A turn to the body's left (+) is counter-clockwise on this y-up map: on the picture, minus the yaw and the turn.
    return { ...u, u: X(pos[0]) / w, v: Y(pos[1]) / h, yaw_deg: (yaw * 180) / Math.PI, sight: { dir_deg: -(yaw * 180) / Math.PI - glance, half_deg: 35, range_m: 6, glance_deg: glance }, noticed: [] };
  });
}

function canvasFor(w, h) {
  if (typeof document === 'undefined') return null;
  cv = cv ?? document.createElement('canvas');
  if (cv.width !== w) cv.width = w;
  if (cv.height !== h) cv.height = h;
  try {
    return cv.getContext('2d');
  } catch {
    return null;
  }
}

function encode(quality) {
  const t0 = performance.now();
  const url = cv.toDataURL('image/webp', Math.max(0.05, Math.min(1, (Number(quality) || 75) / 100)));
  return { b64: url.slice(url.indexOf(',') + 1), encode_ms: performance.now() - t0 };
}

/** 가짜 깊이 색: 그려진 장면에 위(멂 · 파랑)에서 아래(가까움 · 빨강)로 색만 입힌다. */
function tintDepth(ctx, w, h, shift = 0) {
  const g = ctx.createLinearGradient(0, 0, 0, h);
  g.addColorStop(0, '#30123b');
  g.addColorStop(0.45 + shift, '#28bceb');
  g.addColorStop(0.75 + shift / 2, '#edd03a');
  g.addColorStop(1, '#7a0403');
  ctx.globalCompositeOperation = 'color';
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, h);
  ctx.globalCompositeOperation = 'source-over';
}

/** 그 로봇의 자리·방향에서 그린다(에이전트 시점 그리기를 빌려 쓴다). */
function fromUnit(W, r, draw) {
  const a = W.agent;
  const keep = { pos: a.pos, yaw: a.yaw };
  a.pos = r.pos;
  a.yaw = r.yaw;
  try {
    draw();
  } finally {
    Object.assign(a, keep);
  }
}

export function renderUnit({ unit, width, height, quality, mode = 'rgb' }) {
  const W = world();
  if (!W) throw '월드가 없습니다. world_new 를 먼저 호출하세요';
  const r = unitRows().find((u) => u.id === unit);
  if (!r) throw `유닛 ${unit} 이 없습니다`;
  const w = clampInt(width, 32, 1024);
  const h = clampInt(height, 32, 1024);
  const ctx = canvasFor(w, h);
  let b64 = BLANK_WEBP;
  if (ctx) {
    fromUnit(W, r, () => {
      if (mode === 'fork') drawForkView(ctx, w, h, W, []);
      else drawAgentView(ctx, w, h, W, []);
    });
    if (mode === 'depth') tintDepth(ctx, w, h);
    ({ b64 } = encode(quality));
  }
  return { webp_base64: b64, width: w, height: h, unit, mode, label_ko: r.label_ko, depth_name: r.cameras === 2 ? 'SGBM' : 'TIPSv2', procedure: r.first ? procedureView() : null };
}

function shot(W, w, h, quality, depth) {
  const ctx = canvasFor(w, h);
  if (!ctx) return BLANK_WEBP;
  drawAgentView(ctx, w, h, W, []);
  if (depth != null) tintDepth(ctx, w, h, depth);
  return encode(quality).b64;
}

/** 깊이 · 카메라 탭과 CCTV 확대 창의 모의 값: 그림은 가짜 색칠이고 지표는 데모 숫자다. */
export function depthView({ quality = 80 } = {}) {
  const W = world();
  if (!W) throw '월드가 없습니다. world_new 를 먼저 호출하세요';
  const [w, h] = [480, 320];
  const a = W.agent;
  return {
    tick: W.tick, mode: '모의 깊이', device: 'cpu (모의)', infer_ms: 12, flip_tta: false, max_depth_m: 80, scale_m: [0.5, 30],
    rgb_webp_base64: shot(W, w, h, quality, null),
    pred_webp_base64: shot(W, w, h, quality, 0.06),
    truth_webp_base64: shot(W, w, h, quality, 0),
    metrics: { absrel: 0.042, delta1: 0.96, rmse_m: 0.31, median_scale: 1.02 }, metrics_raw: { absrel: 0.11, delta1: 0.81, rmse_m: 0.62 },
    align: { a: 0.98, b: 0.012, inliers: 142, candidates: 168, median_rel_err: 0.03, span_m: [1.2, 9.4] },
    camera: {
      eye_m: a.eye_m ?? 1, standing_eye_m: a.standing_eye_m ?? 1, mount_deg: a.mount_deg ?? -12, head_pitch_deg: a.head_pitch_deg ?? 0, pitch_deg: a.pitch_deg ?? -12, pan_deg: a.pan_deg ?? 0, roll_deg: 0, blur_px: 0.4,
      shake: { source: 'rolling', pitch_deg: 0.2, dz_mm: 1.2, rate_dps: [2, 4, 1] },
      sensor: { name: '모의 센서', rgb: '8비트', exposure_ms: 5, scene_lux: 20000, readout_ms: 8, gyro_arw_deg: 0.01 }, ground: '평지', est_err_deg: { pitch: 0.02, roll: 0.01 }, terrain_tilt_deg: [0, 0],
    },
  };
}

export function renderFrame({ width, height, quality, view }) {
  const W = world();
  if (!W) throw '월드가 없습니다. world_new 를 먼저 호출하세요';
  const w = clampInt(width, 32, 2048);
  const h = clampInt(height, 32, 2048);
  const hits = [];
  const ctx = canvasFor(w, h);
  let b64 = BLANK_WEBP;
  let encode_ms = 0;
  if (ctx) {
    if (view === 'top') drawTopView(ctx, w, h, W);
    else if (view === 'fork') drawForkView(ctx, w, h, W, hits);
    else if (view === 'eye_depth') {
      // 모의 SGBM: 머리 카메라 그림에 깊이 색을 입히고 가운데 물건 자리를 가깝게(따뜻하게) 칠한다.
      drawAgentView(ctx, w, h, W, hits);
      tintDepth(ctx, w, h, 0.05);
      ctx.fillStyle = 'rgba(250, 120, 40, 0.85)';
      ctx.fillRect(w * 0.38, h * 0.42, w * 0.24, h * 0.2);
    } else drawAgentView(ctx, w, h, W, hits);
    ({ b64, encode_ms } = encode(quality));
  }
  const pad = b64.endsWith('==') ? 2 : b64.endsWith('=') ? 1 : 0;
  const detections = hits
    .sort((a, b) => a.o.dist - b.o.dist).slice(0, 12)
    .map(({ o, bbox }) => detection(o, bbox, w, h, W.tick));
  const a = W.agent;
  const moving = (a.speed ?? 0) > 0.05;
  const shakePitch = moving ? Math.sin(W.tick * 0.9) * 0.35 : 0;
  const camera = view !== 'agent' ? null : {
    eye_m: a.eye_m ?? 1.0,
    standing_eye_m: a.standing_eye_m ?? 1.0,
    mount_deg: a.mount_deg ?? -12.0,
    head_pitch_deg: a.head_pitch_deg ?? 0.0,
    pitch_deg: (a.pitch_deg ?? -12.0) + shakePitch,
    pan_deg: a.pan_deg ?? 0.0,
    roll_deg: a.roll_deg ?? 0.0,
    blur_px: moving ? 0.4 : 0.0,
    shake: {
      source: moving ? (W.base === 'biped' || W.base === 'quadruped' ? 'sway' : 'rolling') : 'still',
      pitch_deg: shakePitch,
      dz_mm: moving ? 1.2 : 0.0,
      rate_dps: moving ? [2, 4, 1] : [0, 0, 0],
    },
  };
  // 줍기 절차와 비전 확인 대상의 상자 (1인칭에서만): HUD 가 조준점을 그 위에 둔다.
  const procedure = camera ? procedureView() : null;
  if (procedure?.check) {
    const hit = hits.find((x) => x.o.id === procedure.target);
    if (hit) procedure.check = { ...procedure.check, bbox: hit.bbox.map((v, i) => Number(Math.max(0, Math.min(1, v / (i % 2 ? h : w))).toFixed(4))) };
  }
  // 다리 몸이 낮추고 쥐고 드는 동안 1인칭 프레임에도 포커스를 싣는다(Rust 와 같은 단계).
  const handsBusy = procedure?.mechanism === 'hand' && ['bend', 'grip', 'lift'].includes(procedure.stage);
  const eyeDepth = view === 'eye_depth' ? { source: 'sgbm', valid: 0.81, scale_m: [0.4, 3.2], ms: 0, grid: [w, h], box: [0.36, 0.4, 0.64, 0.64], box_valid: 0.72, box_depth_m: 0.68, item: null } : null;
  return {
    webp_base64: b64, width: w, height: h, bytes: Math.floor((b64.length * 3) / 4) - pad, encode_ms, tick: W.tick, detections, camera, procedure,
    px_per_m: view === 'top' ? topMap(w, h, W).s : null,
    relay: null,
    units: view === 'top' ? topUnits(w, h, W) : [],
    overlay: view.startsWith('chase') ? mockOverlay(w, h) : eyeDepth,
    focus: view.startsWith('focus') || handsBusy ? mockFocus() : null,
  };
}
