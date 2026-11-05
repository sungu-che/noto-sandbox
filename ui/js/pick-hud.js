// 줍기 절차 HUD: 다리 몸은 사람처럼 "고개를 숙여 응시 → 비전 확인 → 자세 낮춤 → 쥐기 → 시선 따라 들어올림",
// 바퀴·메카넘·트랙 몸은 지게차처럼 두 눈으로 교차 검증한다: "탑승자 시선 내려 확인 → 포크 내림(시선 정면) →
// 포크 카메라 확인·교차 검증 → 포크 끼우기(포크 카메라 추적) → 포크 카메라 가림 → 탑승자 재확인 → 안착 → 들어올림".
// world_frame 의 procedure(Rust 런타임 또는 모의 월드)를 그대로 그리며, 1인칭(머리·탑승자) 화면의 비전 확인에는
// 대상에 조준점을 겹친다. 지게차 절차에서는 포크 카메라 화면을 작게 함께 보이고, 가려지면 그렇게 표시한다.
// 탑뷰에서는 같은 절차를 unit_frame 의 procedure 로 받아 로봇 위 CCTV 말풍선에 줄여 그린다(renderProcedureBrief).
// 다리 몸(카메라 2대)은 지게차의 포크 카메라 자리에 머리 카메라의 SGBM 깊이를 띄우고(응시 → 쥐기), 응시 확인이 그 깊이로
// 읽은 물건의 놓인 각도·크기를 적으며, 자세를 낮추고 쥐고 드는 동안 팔 관절과 손가락 패드 접촉을 함께 그린다.
import { $, clear, el, fmt } from './util.js';
import { drawFocus } from './focus-view.js';
import { state } from './state.js';

const STATE_KO = { done: '완료', active: '진행', pending: '대기', failed: '실패' };
const PADS_KO = { dry: '마른 패드', wet: '젖은 패드', dusty: '먼지 낀 패드' };
const signed = (v, d) => `${v >= 0 ? '+' : ''}${fmt.n(v, d)}`;

/**
 * 물리 · SDS 되먹임(world_frame.relay): 이 물건을 쥘 때 손이 기억하는 마찰(그 물건 · 패드 상태로 겪은 미세 미끄럼에서
 * 배운 사전값, 학습 전이면 표 그대로)과, 손에 든 짐이 손안에서 밀려 전신에 준 것(ZMP · 어깨 부하).
 */
function relayLines(p, r) {
  const out = [];
  const g = r?.grip;
  if (g && p?.mechanism === 'hand') {
    const pads = PADS_KO[g.pads] ?? g.pads;
    const felt = g.grasps ? ` · 미세 미끄럼 ${fmt.n(g.slip_rate * 100, 0)}%${g.mu_seen != null ? ` · 읽은 마찰 ${fmt.n(g.mu_seen, 2)}` : ''} · ${g.grasps}회` : '';
    out.push(g.mu < g.table_mu - 1e-3
      ? `손이 기억하는 마찰 · ${g.item_ko}(${pads}) 표 ${fmt.n(g.table_mu, 2)} → ${fmt.n(g.mu, 2)} 로 더 세게 쥠${felt}`
      : `손이 기억하는 마찰 · ${g.item_ko}(${pads}) 표 ${fmt.n(g.table_mu, 2)} 그대로${g.learning ? ` (학습 중 ${g.grasps}/${g.min_grasps}회)` : ''}${felt}`);
  }
  const s = r?.slide;
  if (s && Math.hypot(s.shift_m[0], s.shift_m[1]) >= 0.005) {
    out.push(`손안 짐 쏠림 ${signed(s.shift_m[0] * 100, 1)} cm → ZMP ${signed(s.zmp_shift_m * 1000, 1)} mm · 어깨 부하 ${fmt.n(s.shoulder_ratio_held * 100, 0)} → ${fmt.n(s.shoulder_ratio * 100, 0)}%`);
  }
  return out;
}

function stageRow(s) {
  return el('li', { class: `ph-stage ${s.state}`, title: s.note ?? '' },
    el('span', { class: 'ph-dot', aria: { hidden: 'true' } }),
    el('span', { class: 'ph-nm', text: s.label_ko }),
    el('span', { class: 'ph-st', text: STATE_KO[s.state] ?? s.state }));
}

// 그 눈의 깊이 절차: SGBM(카메라 2대, 스테레오 법칙의 ±mm, 상자 안 유효 화소) 또는 TIPSv2(카메라 1대, 지면 정렬의 ±cm).
function depthText(c) {
  if (c.depth_source === 'sgbm') {
    const sig = c.depth_sigma_m != null ? ` ±${fmt.n(c.depth_sigma_m * 1000, 0)} mm` : '';
    const valid = c.depth_valid != null ? ` · 유효 ${fmt.n(c.depth_valid * 100, 0)}%` : '';
    return c.depth_m != null ? ` · SGBM ${fmt.n(c.depth_m, 2)} m${sig}${valid}` : ` · SGBM 깊이 없음${valid}`;
  }
  if (c.depth_source === 'tipsv2+ground') {
    const sig = c.depth_sigma_m != null ? ` ±${fmt.n(c.depth_sigma_m * 100, 0)} cm` : '';
    return c.depth_m != null ? ` · TIPSv2 ${fmt.n(c.depth_m, 2)} m${sig}` : ' · TIPSv2 지면 정렬 실패';
  }
  return c.depth_m != null ? ` · 깊이 ${fmt.n(c.depth_m, 2)} m` : '';
}

function sightText(c) {
  return `${c.label_ko ?? ''} ${c.score != null ? Math.round(c.score * 100) : '—'} · ${fmt.n(c.range_m, 2)} m${depthText(c)}`;
}

function checkLine(p) {
  const c = p.check;
  if (p.mechanism === 'fork') return crossLines(p);
  if (!c) return el('div', { class: 'ph-check wait', text: '비전 확인 대기 — 고개를 숙여 대상을 화면 가운데에 둡니다' });
  if (!c.ok) return el('div', { class: 'ph-check fail', text: `비전 확인 실패 · ${c.reason ?? '대상이 보이지 않음'}` });
  return el('div', { class: 'ph-check ok', text: `비전 확인 ✓ ${sightText(c)}` });
}

const VERDICT = {
  agree: ['ok', '교차 검증 ✓ 두 시점이 같은 자리'],
  disagree: ['fail', '교차 검증 ✗ 두 시점이 어긋남 — 포크 카메라를 믿음'],
  fork_only: ['wait', '교차 검증 △ 포크 카메라만 봄'],
  rider_only: ['wait', '교차 검증 △ 탑승자만 봄'],
  none: ['fail', '교차 검증 ✗ 두 시점 모두 못 봄'],
};

// 지게차의 두 눈: ① 탑승자 시선(머리 카메라) ② 포크 카메라, 둘의 교차 검증, ③ 포크 카메라가 가려진 뒤 탑승자 재확인.
function crossLines(p) {
  const x = p.cross ?? {};
  const box = el('div', { class: 'ph-cross' });
  const line = (cls, text) => box.append(el('div', { class: `ph-check ${cls}`, text }));
  const sight = (n, who, s, wait) => {
    if (!s) return line('wait', `${n} ${who} 대기 — ${wait}`);
    if (!s.ok) return line('fail', `${n} ${who} 실패 · ${s.reason ?? '대상이 보이지 않음'}`);
    if (s.matched === false) return line('wait', `${n} ${who} △ ${s.reason ?? '대상이 아닌 이웃 것만 보임'}`);
    const neck = s.camera === 'head' && s.neck_deg != null ? ` · 고개 ${fmt.n(s.neck_deg, 0)}°` : '';
    return line('ok', `${n} ${who} ✓ ${sightText(s)}${neck}`);
  };
  sight('①', '탑승자 시선', x.rider, '고개를 숙여 물건을 봅니다');
  sight('②', '포크 카메라', x.fork, '포크를 내리고 포크 카메라로 봅니다 (탑승자는 정면)');
  if (x.verdict) {
    let [cls, text] = VERDICT[x.verdict] ?? ['wait', x.verdict];
    // 포크 카메라가 보긴 했지만 거리를 못 쟀거나(스테레오 매칭 없음) 이웃 것만 본 경우.
    if (x.verdict === 'rider_only' && x.fork?.ok) text = '교차 검증 △ 포크 카메라로는 확인 못 함 — 탑승자 시선으로';
    const gap = x.apart_m != null ? ` (${fmt.n(x.apart_m * 100, 0)} cm, 허용 ${fmt.n(x.gate_m * 100, 0)} cm)` : '';
    line(cls, `${text}${gap}`);
  }
  if (x.recheck || ['recheck', 'seat', 'raise'].includes(p.stage)) {
    const blocked = x.blocked_share != null ? `포크 카메라 ${fmt.n(x.blocked_share * 100, 0)}% 가림 → ` : '포크 카메라 가림 → ';
    if (!x.recheck) line('wait', `③ ${blocked}탑승자 재확인 대기`);
    else if (!x.recheck.ok) line('fail', `③ ${blocked}재확인 실패 · ${x.recheck.reason ?? '대상이 보이지 않음'}`);
    else if (x.recheck.matched === false) line('wait', `③ ${blocked}재확인 △ ${x.recheck.reason ?? '대상이 아닌 이웃 것만 보임'}`);
    else {
      const gap = x.recheck_apart_m != null ? ` · 포크 카메라와 ${fmt.n(x.recheck_apart_m * 100, 0)} cm` : '';
      line(x.seated === false ? 'fail' : 'ok', `③ ${blocked}탑승자 재확인 ${x.seated === false ? '✗' : '✓'} 포크 위 ${x.recheck.label_ko ?? ''}${gap}`);
    }
  }
  return box;
}

function pose(p) {
  const bits = [];
  if (p.depth_procedure) bits.push(p.depth_procedure === 'sgbm' ? '깊이: SGBM 스테레오 (카메라 2대)' : '깊이: TIPSv2 단안 (카메라 1대)');
  if (p.mechanism === 'fork') {
    if (p.fork_m != null) bits.push(`포크 ${fmt.n(p.fork_m, 2)} m`);
    const down = (p.neck_deg ?? 0) > 1 || (p.trunk_deg ?? 0) > 1;
    bits.push(down ? `탑승자 고개 ${fmt.n(p.neck_deg, 0)}°${(p.trunk_deg ?? 0) > 1 ? ` · 몸 ${fmt.n(p.trunk_deg, 0)}°` : ''} 숙임 (카메라 ${fmt.n(p.pitch_deg, 0)}°)` : `탑승자 시선 정면 (카메라 ${fmt.n(p.pitch_deg, 0)}°)`);
    bits.push(p.camera === 'fork' ? '보는 눈: 포크 카메라' : '보는 눈: 탑승자');
    if (p.tip_ratio != null) bits.push(`전도 여유 ${fmt.n(p.tip_ratio, 2)}`);
    return bits.join(' · ');
  }
  if (p.pitch_deg != null) bits.push(`카메라 ${fmt.n(p.pitch_deg, 0)}°`);
  if (p.neck_deg != null) bits.push(`목 ${fmt.n(p.neck_deg, 0)}°`);
  if (p.trunk_deg != null) bits.push(`몸통 ${fmt.n(p.trunk_deg, 0)}°`);
  if (p.eye_m != null) bits.push(`눈 ${fmt.n(p.eye_m, 2)} m${p.standing_eye_m ? ` (선 자세 ${fmt.n(p.standing_eye_m, 2)})` : ''}`);
  return bits.join(' · ');
}

function reticle(c) {
  const layer = $('det-layer');
  layer.querySelectorAll('.gaze-reticle').forEach((n) => n.remove());
  if (!c?.bbox || c.bbox.length < 4) return;
  const [x0, y0, x1, y1] = c.bbox;
  const r = el('div', { class: `gaze-reticle${c.ok ? ' ok' : ' fail'}`, aria: { hidden: 'true' } });
  r.style.left = `${((x0 + x1) / 2) * 100}%`;
  r.style.top = `${((y0 + y1) / 2) * 100}%`;
  layer.append(r);
}

function title(p) {
  if (p.mechanism !== 'fork') return `${p.act === 'look' ? '응시' : '줍기'} · ${p.label_ko ?? p.target ?? ''}`;
  const what = p.act === 'set_down' ? '내려놓기' : p.act === 'look' ? '포크 확인' : '들기';
  return `지게차 ${what} · ${p.label_ko ?? p.target ?? ''}`;
}

/** 지게차 절차에서 포크 카메라 작은 화면을 보일지. */
export function wantsForkCam(p) {
  return !!p && p.act != null && p.mechanism === 'fork' && p.act !== 'set_down';
}

/** 다리 몸(스테레오 눈)의 줍기에서 머리 카메라 SGBM 깊이 화면을 보일지: 응시부터 쥘 때까지. */
export function wantsEyeDepth(p) {
  return !!p && p.act != null && p.mechanism === 'hand' && p.depth_procedure === 'sgbm' && ['look', 'check', 'bend', 'grip'].includes(p.stage);
}

/** 팔·손가락 화면: 자세를 낮추고 쥐고 드는 동안 (런타임이 그때 world_frame 에 focus 를 싣는다). */
function wantsHandViews(p, focus) {
  return !!p && !!focus && p.mechanism === 'hand' && ['bend', 'grip', 'lift'].includes(p.stage);
}

const FORM_KO = { bar: '막대형 (원목처럼 긴 것)', block: '덩어리형' };

/** 응시 확인이 깊이로 읽은 물건: 몸 기준 놓인 각도, 길이·폭·높이, 윗면 기울기. */
function itemLine(item) {
  if (!item) return '';
  const yaw = item.elongation >= 0.25 ? `놓인 각도 ${item.rel_yaw_deg >= 0 ? '+' : ''}${fmt.n(item.rel_yaw_deg, 0)}° (몸 기준)` : '놓인 각도 — (거의 정사각)';
  return `${yaw} · ${fmt.n(item.length_m, 2)} × ${fmt.n(item.width_m, 2)} × 높이 ${fmt.n(item.height_m, 2)} m · 윗면 ${fmt.n(item.top_tilt_deg, 0)}° · ${FORM_KO[item.form] ?? item.form}`;
}

/** 깊이 화면 위의 겹침: 대상 상자, 그 안의 SGBM 유효 몫과 깊이. */
export function drawEyeDepthOverlay(canvas, o) {
  const ctx = canvas.getContext('2d');
  const box = o?.box;
  if (box && box.length === 4) {
    const [x0, y0, x1, y1] = box;
    ctx.strokeStyle = (o.box_valid ?? 0) >= 0.05 ? '#3ad29f' : '#ff6b6b';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x0 * canvas.width, y0 * canvas.height, (x1 - x0) * canvas.width, (y1 - y0) * canvas.height);
  }
  const cap = $('ph-eyedepth-box')?.querySelector('figcaption');
  if (cap && o) {
    const inBox = o.box_depth_m != null ? ` · 대상 ${fmt.n(o.box_depth_m, 2)} m (유효 ${fmt.n((o.box_valid ?? 0) * 100, 0)}%)` : '';
    cap.textContent = `SGBM 깊이 · 매칭 ${fmt.n((o.valid ?? 0) * 100, 0)}%${inBox}`;
  }
}

/** 1인칭(머리·탑승자) 화면에 겹칠 비전 확인: 사람은 응시·확인 동안의 확인, 지게차는 탑승자가 숙여 본 확인(재확인 우선). */
function headCheck(p) {
  if (p.mechanism !== 'fork') return ['look', 'check'].includes(p.stage) ? p.check : null;
  if (p.camera === 'fork') return null;
  const x = p.cross ?? {};
  if (['recheck', 'seat'].includes(p.stage)) return x.recheck ?? null;
  return p.stage === 'rider' ? x.rider ?? null : null;
}

/** 두 시점 교차 검증의 한 줄 요약: 판정, 어긋난 거리, 포크 카메라 가림. */
function crossBrief(x) {
  if (!x?.verdict) return ['wait', '교차 검증 대기'];
  const [cls, text] = VERDICT[x.verdict] ?? ['wait', x.verdict];
  const gap = x.apart_m != null ? ` (오차 ${fmt.n(x.apart_m * 100, 0)} cm)` : '';
  const blocked = (x.blocked_share ?? 0) > 0 ? ` · 가림 ${fmt.n(x.blocked_share * 100, 0)}%` : '';
  return [cls, `${text}${gap}${blocked}`];
}

/** 비전 확인의 한 줄 요약 (사람 몸). */
function checkBrief(p) {
  const c = p.check;
  if (!c) return ['wait', '비전 확인 대기'];
  return c.ok ? ['ok', `비전 확인 ✓ ${sightText(c)}`] : ['fail', `비전 확인 실패 · ${c.reason ?? '대상이 보이지 않음'}`];
}

/** CCTV 말풍선의 절차 표시: 제목, 단계 막대, 지금 단계, 교차 검증(또는 비전 확인). 절차가 없으면 false. */
export function renderProcedureBrief(box, p) {
  clear(box);
  if (!p || p.act == null) return false;
  const stages = p.stages ?? [];
  const at = stages.findIndex((s) => s.state === 'active' || s.state === 'failed');
  const [cls, text] = p.mechanism === 'fork' ? crossBrief(p.cross) : checkBrief(p);
  box.append(
    el('div', { class: 'cp-title', text: title(p) }),
    el('ol', { class: 'cp-steps', aria: { hidden: 'true' } }, ...stages.map((s) => el('li', { class: s.state, title: s.label_ko }))),
    el('div', { class: 'cp-now', text: at >= 0 ? `${at + 1}/${stages.length} ${stages[at].label_ko}` : '' }),
    el('div', { class: `cp-check ${cls}`, text }));
  return true;
}

/** world_frame 응답의 procedure 를 HUD 로 그린다(1인칭 시점). 없으면 숨긴다. focus 는 그 응답의 FocusFrame. */
export function renderPickHud(p, focus = null) {
  const hud = $('pick-hud');
  if (!hud) return;
  const show = !!p && p.act != null;
  hud.hidden = !show;
  reticle(show ? headCheck(p) : null);
  const eye = show && wantsEyeDepth(p);
  const hands = show && wantsHandViews(p, focus);
  const views = $('ph-views');
  hud.classList.toggle('wide', hands);
  if (views) {
    views.hidden = !(eye || hands);
    $('ph-eyedepth-box').hidden = !eye;
    $('ph-arm-box').hidden = !hands || !(focus.arms ?? []).length;
    $('ph-fingers-box').hidden = !hands;
    if (hands) {
      if ((focus.arms ?? []).length) drawFocus($('ph-arm'), focus, 'arm', { compact: false });
      drawFocus($('ph-fingers'), focus, 'hand', { compact: false });
    }
  }
  const itemBox = $('ph-item');
  if (itemBox) {
    const line = show && p.mechanism === 'hand' ? itemLine(p.check?.item) : '';
    itemBox.hidden = !line;
    itemBox.textContent = line ? `깊이로 읽은 물건 · ${line}` : '';
  }
  const relayBox = $('ph-relay');
  if (relayBox) {
    const rows = show ? relayLines(p, state.relay) : [];
    relayBox.hidden = !rows.length;
    relayBox.replaceChildren(...rows.map((t) => el('div', { text: t })));
  }
  const cam = $('ph-forkcam-box');
  if (cam) {
    cam.hidden = !wantsForkCam(p);
    const live = show && p.camera === 'fork';
    const blocked = show && (p.cross?.blocked_share ?? 0) > 0 && ['recheck', 'seat'].includes(p.stage);
    cam.classList.toggle('live', live);
    cam.classList.toggle('blocked', blocked);
    const cap = cam.querySelector('figcaption');
    if (cap) cap.textContent = blocked ? `포크 카메라 · ${fmt.n(p.cross.blocked_share * 100, 0)}% 가려짐` : live ? '포크 카메라 · 보는 중' : '포크 카메라';
  }
  if (!show) return;
  $('ph-title').textContent = title(p);
  $('ph-pose').textContent = pose(p);
  const list = clear($('ph-stages'));
  for (const s of p.stages ?? []) list.append(stageRow(s));
  const box = clear($('ph-check'));
  box.append(checkLine(p));
}
