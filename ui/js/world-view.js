// 월드 화면: world_frame(WebP) 폴링 → 전체 화면 캔버스, 탐지 박스, 탐지 목록(분석 모달), 표시 토글.
// 폴링은 약 6fps 이고, 이전 호출이 끝나지 않았거나 틱·옵션이 그대로면 건너뛴다.
// 프레임은 캔버스 비율대로 요청한다. 탑뷰는 런타임이 지도 전체를 그 비율 안에 맞춰 그리고 로봇마다 프레임 좌표(units)를 준다.
// 3인칭·포커스 시점의 overlay · focus 는 'unit-detail' 로 넘겨 유닛 HUD 가 그린다.
import { call } from './api.js';
import { bus, state } from './state.js';
import { $, clear, el, fmt, observeResize } from './util.js';
import { drawJointOverlay } from './joint-view.js';
import { drawEyeDepthOverlay, renderPickHud, wantsEyeDepth, wantsForkCam } from './pick-hud.js';
import { isOpen } from './overlays.js';

const POLL_MS = 160;
const RETRY_MS = 2500;
const TOP_MAX_W = 1280; // 탑뷰는 화소당 비용이 작아 화면 폭까지 받는다
let pending = false;
let lastKey = '';
let retryAt = 0;
let drewOnce = false;

const viewNow = () => $('view-select').value;

/** 요청할 프레임 크기: 폭·품질은 #frame-preset, 높이는 캔버스 비율을 따른다. */
function framePreset() {
  const [preset, quality] = $('frame-preset').value.split('x').map(Number);
  const wrap = $('world-wrap');
  const W = wrap.clientWidth;
  const H = wrap.clientHeight;
  const aspect = W && H ? H / W : 2 / 3;
  const width = viewNow() === 'top' ? Math.min(TOP_MAX_W, Math.max(preset, W)) : preset;
  return { width, height: 2 * Math.round((width * aspect) / 2), quality };
}

/** 화면의 프레임 사각형(world-wrap 기준 px). CCTV · 유닛 선택이 u, v 를 화면 좌표로 바꿀 때 쓴다. */
export function frameRect() {
  const box = $('frame-box');
  return { left: box.offsetLeft, top: box.offsetTop, width: box.offsetWidth, height: box.offsetHeight };
}

// ── 배치: 프레임을 비율 유지로 화면에 맞춘다 ─────────────────────
function layoutFrame() {
  const wrap = $('world-wrap');
  const W = wrap.clientWidth;
  const H = wrap.clientHeight;
  if (!W || !H) return;
  const fw = state.frame?.width ?? 3;
  const fh = state.frame?.height ?? 2;
  const s = Math.min(W / fw, H / fh);
  const w = fw * s;
  const h = fh * s;
  const box = $('frame-box');
  box.style.left = `${(W - w) / 2}px`;
  box.style.top = `${(H - h) / 2}px`;
  box.style.width = `${w}px`;
  box.style.height = `${h}px`;
}

// ── 디코드 ──────────────────────────────────────────────────────
async function decode(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const type = bin.startsWith('RIFF') ? 'image/webp' : bin.startsWith('\x89PNG') ? 'image/png' : 'image/jpeg';
  const blob = new Blob([bytes], { type });
  if (typeof createImageBitmap === 'function') return createImageBitmap(blob);
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('프레임 디코드 실패'));
    img.src = URL.createObjectURL(blob);
  });
}

async function drawFrame(f) {
  const img = await decode(f.webp_base64);
  const canvas = $('world-canvas');
  if (canvas.width !== f.width || canvas.height !== f.height) {
    canvas.width = f.width;
    canvas.height = f.height;
  }
  canvas.getContext('2d').drawImage(img, 0, 0, f.width, f.height);
  img.close?.();
  const sizeChanged = !state.frame || state.frame.width !== f.width || state.frame.height !== f.height;
  // px_per_m: the top view's scale (frame pixels per metre), for the robots' sight fans.
  state.frame = { width: f.width, height: f.height, bytes: f.bytes, encode_ms: f.encode_ms, tick: f.tick, px_per_m: f.px_per_m ?? null };
  state.detections = f.detections ?? [];
  state.units = f.units ?? [];
  // 물리 · SDS 되먹임(작업자 1): 손이 기억하는 마찰, 손안 짐 쏠림, 동료로 고친 위치.
  state.relay = f.relay ?? null;
  // 3인칭: 관절 물리 오버레이(Rust 가 픽셀로 투영한 관절·CoM·ZMP). 표와 부위 그림은 유닛 HUD 에서.
  const chase = !!f.overlay;
  $('joint-layer').hidden = !chase;
  drawJointOverlay(f.overlay ?? null, f.width, f.height);
  if (f.overlay || f.focus) bus.emit('unit-detail', { overlay: f.overlay ?? null, focus: f.focus ?? null });
  if (sizeChanged) layoutFrame();
  if (!drewOnce) {
    drewOnce = true;
    $('world-empty').hidden = true;
  }
  // 1인칭이면 머리 카메라의 높이·피치·팬·흔들림·블러를 함께 적는다.
  const c = f.camera;
  if (c) {
    state.camera = c;
    bus.emit('camera', c);
  }
  const cam = c ? ` · 눈 ${fmt.n(c.eye_m, 2)} m · 피치 ${fmt.n(c.pitch_deg, 1)}° · 롤 ${fmt.n(c.roll_deg, 1)}°${Math.abs(c.pan_deg) > 0.5 ? ` · 팬 ${fmt.n(c.pan_deg, 0)}°` : ''} · 흔들림 ${c.shake?.source === 'still' ? '없음' : `${fmt.n(c.shake?.pitch_deg, 2)}°`} · 블러 ${fmt.n(c.blur_px, 1)} px` : '';
  $('frame-info').textContent = `틱 ${f.tick} · ${f.width}×${f.height} · ${fmt.bytes(f.bytes)} · 인코딩 ${fmt.n(f.encode_ms, 1)} ms${cam}`;
  $('sb-frame').textContent = `${fmt.bytes(f.bytes)} · ${fmt.n(f.encode_ms, 1)} ms`;
  state.procedure = c ? f.procedure ?? null : null;
  // While a walker bends, grips and lifts, the runtime puts the focus frame on the first-person frame too: the pick
  // HUD draws its arm and its finger pads from it.
  state.pickFocus = c ? f.focus ?? null : null;
  renderDetections();
  bus.emit('frame', state.frame);
}

// ── 탐지 박스 · 목록 ────────────────────────────────────────────
function tagText(d) {
  const base = `${d.label_ko} ${Math.round(d.score * 100)}${d.accepted ? '' : ' 보류'}`;
  return state.toggles.oracle ? `${base} · ${d.label_id}` : base;
}

function renderDetections() {
  const layer = $('det-layer');
  clear(layer);
  const dets = state.detections;
  if (state.toggles.boxes) {
    for (const d of dets) {
      if (!Array.isArray(d.bbox) || d.bbox.length < 4) continue;
      const [x0, y0, x1, y1] = d.bbox;
      const box = el('div', { class: `det-box${d.accepted ? '' : ' hold'}${y0 < 0.07 ? ' edge' : ''}`, title: d.reason ?? '' },
        el('span', { class: 'det-tag', text: tagText(d) }));
      box.style.left = `${x0 * 100}%`;
      box.style.top = `${y0 * 100}%`;
      box.style.width = `${Math.max(0, x1 - x0) * 100}%`;
      box.style.height = `${Math.max(0, y1 - y0) * 100}%`;
      layer.append(box);
    }
  }
  renderPickHud(state.procedure, state.pickFocus);
  const list = $('lg-list');
  clear(list);
  const sorted = [...dets].sort((a, b) => b.score - a.score).slice(0, 10);
  $('lg-count').textContent = dets.length ? `${dets.filter((d) => d.accepted).length}/${dets.length} 수락` : '';
  if (!sorted.length) {
    list.append(el('li', { class: 'lg-empty', text: '탐지 없음' }));
    return;
  }
  for (const d of sorted) {
    list.append(el('li', { class: d.accepted ? '' : 'hold', title: d.reason ?? '' },
      el('span', { class: 'sw' }),
      el('span', { class: 'nm', text: state.toggles.oracle ? `${d.label_ko} [${d.label_id}]` : d.label_ko }),
      el('span', { class: 'sc', text: `${Math.round(d.score * 100)}%` }),
      el('span', { class: `choice-badge ${d.accepted ? 'done' : 'rule'}`, text: d.accepted ? '수락' : '보류' })));
  }
}

// ── 지게차: 줍기 HUD 안의 포크 카메라 작은 화면 ──────────────────────
async function drawForkCam(quality) {
  const f = await call('world_frame', { width: 240, height: 180, quality, view: 'fork' }, { quiet: true });
  const img = await decode(f.webp_base64);
  const canvas = $('ph-forkcam');
  if (canvas.width !== f.width || canvas.height !== f.height) {
    canvas.width = f.width;
    canvas.height = f.height;
  }
  canvas.getContext('2d').drawImage(img, 0, 0, f.width, f.height);
  img.close?.();
}

// ── 다리 몸: 줍기 HUD 안의 머리 카메라 SGBM 깊이 (스테레오 눈) ────────────
// 런타임이 그 순간의 스테레오 쌍에 SGBM 을 돌려 색 입힌 깊이와 대상 상자를 준다. 정합 한 번이 수십 ms 라
// 프레임마다가 아니라 EYE_DEPTH_MS 마다 다시 받는다.
const EYE_DEPTH_MS = 400;
let eyeDepthAt = 0;
async function drawEyeDepth(quality) {
  if (Date.now() - eyeDepthAt < EYE_DEPTH_MS) return;
  eyeDepthAt = Date.now();
  const f = await call('world_frame', { width: 240, height: 160, quality, view: 'eye_depth' }, { quiet: true });
  const img = await decode(f.webp_base64);
  const canvas = $('ph-eyedepth');
  if (canvas.width !== 240 || canvas.height !== 160) {
    canvas.width = 240;
    canvas.height = 160;
  }
  canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
  img.close?.();
  drawEyeDepthOverlay(canvas, f.overlay);
}

// ── 폴링 ────────────────────────────────────────────────────────
async function poll(force = false) {
  // 분석 모달은 월드를 덮으므로 그동안은 받지 않는다(닫으면 틱이 달라 다음 주기에 받는다).
  if (pending || document.hidden || !state.snapshot || isOpen('analysis-modal')) return;
  if (!force && Date.now() < retryAt) return;
  const p = framePreset();
  const sel = viewNow();
  const view = sel === 'chase' ? `chase:${$('orbit-range').value}` : sel;
  const key = `${state.snapshot.tick}|${p.width}x${p.height}q${p.quality}|${view}`;
  if (!force && key === lastKey) return;
  pending = true;
  try {
    const f = await call('world_frame', { ...p, view }, { quiet: true });
    lastKey = key;
    await drawFrame(f);
    if (view === 'agent' && wantsForkCam(state.procedure)) await drawForkCam(p.quality);
    if (view === 'agent' && wantsEyeDepth(state.procedure)) await drawEyeDepth(p.quality).catch(() => {});
  } catch (e) {
    retryAt = Date.now() + RETRY_MS;
    $('frame-info').textContent = `프레임 오류: ${e.message}`;
    if (!drewOnce) showEmpty(`프레임을 가져오지 못했습니다: ${e.message}`, true, () => { retryAt = 0; poll(true); });
  } finally {
    pending = false;
  }
}

function showEmpty(message, isError = false, onRetry = null) {
  const box = $('world-empty');
  clear(box);
  box.hidden = false;
  box.classList.toggle('err', isError);
  box.append(el('div', { text: message }));
  if (isError && onRetry) {
    box.append(el('button', { type: 'button', class: 'btn btn-sm btn-outline', text: '다시 시도', on: { click: onRetry } }));
  }
}

// ── 표시 토글 ───────────────────────────────────────────────────
/** 탑뷰는 지도 전체가 화면이므로 미니맵은 다른 시점에서만 보인다. */
function syncMinimap() {
  $('vp-minimap').hidden = !state.toggles.path || viewNow() === 'top';
}

const TOGGLES = [
  ['tg-cctv', 'cctv', () => bus.emit('frame', state.frame)],
  ['tg-boxes', 'boxes', () => renderDetections()],
  ['tg-path', 'path', syncMinimap],
  ['tg-oracle', 'oracle', () => renderDetections()],
];

function initToggles() {
  for (const [id, key, apply] of TOGGLES) {
    const btn = $(id);
    state.toggles[key] = btn.getAttribute('aria-pressed') === 'true';
    btn.addEventListener('click', () => {
      state.toggles[key] = !state.toggles[key];
      btn.setAttribute('aria-pressed', String(state.toggles[key]));
      btn.classList.toggle('active', state.toggles[key]);
      apply();
    });
  }
  syncMinimap();
}

export function initWorldView() {
  initToggles();
  observeResize($('world-wrap'), layoutFrame);
  layoutFrame();
  const syncView = () => {
    const chase = viewNow() === 'chase';
    $('orbit-range').hidden = !chase;
    $('orbit-label').hidden = !chase;
    syncMinimap();
  };
  $('view-select').addEventListener('change', () => {
    syncView();
    state.units = [];
    bus.emit('view', viewNow());
    poll(true);
  });
  $('orbit-range').addEventListener('input', () => poll(true));
  syncView();
  $('frame-preset').addEventListener('change', () => poll(true));
  bus.on('world-new', () => { lastKey = ''; retryAt = 0; poll(true); });
  // 틱은 그대로인데 화면이 바뀌는 일(작업자 소환·제거, 몸 교체): 받는 중이어도 다음 주기에 다시 받게 키를 지운다.
  bus.on('frame-stale', () => { lastKey = ''; poll(true); });
  bus.on('world-error', (m) => showEmpty(`월드를 만들지 못했습니다: ${m}`, true, () => $('world-new').click()));
  setInterval(() => poll(false), POLL_MS);
  renderDetections();
}
