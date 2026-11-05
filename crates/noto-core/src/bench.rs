use std::path::Path;

use serde::Serialize;

use crate::actbench::{self, AxResult};
use crate::balance::{Body, FeedForward};
use crate::base::BaseKind;
use crate::effects::{self, FxResult};
use crate::hand::FingerCount;
use crate::nn::{self, ScenarioResult};
use crate::parts::BodyPlan;

pub const ENGINE_REV: &str = "noto-1.0.0-wave1";

pub const HOLDOUT_SEED: u64 = 20261024;

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(tag = "rule", rename_all = "snake_case")]
pub enum Rule {
    Like { reference: f32, tolerance: f32 },
    AtLeast { reference: f32, tolerance: f32 },
    AtMost { limit: f32 },
    Raw,
    NoWorse { reference: f32, tolerance: f32 },
}

impl Rule {
    pub fn is_reference(&self) -> bool {
        matches!(self, Rule::Like { .. } | Rule::AtLeast { .. } | Rule::NoWorse { .. })
    }

    pub fn reference(&self) -> Option<(f32, f32)> {
        match *self {
            Rule::Like { reference, tolerance } | Rule::AtLeast { reference, tolerance } | Rule::NoWorse { reference, tolerance } => Some((reference, tolerance)),
            _ => None,
        }
    }
}

impl From<effects::Rule> for Rule {
    fn from(r: effects::Rule) -> Rule {
        match r {
            effects::Rule::Like { reference, tolerance } => Rule::Like { reference, tolerance },
            effects::Rule::AtLeast { reference, tolerance } => Rule::AtLeast { reference, tolerance },
            effects::Rule::AtMost { limit } => Rule::AtMost { limit },
            effects::Rule::Raw => Rule::Raw,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Verdict {
    pub z: Option<f32>,
    pub score: Option<f32>,
    pub broken: Option<(f32, f32)>,
}

pub fn likeness(z: f32) -> f32 {
    (100.0 - 10.0 * z.abs()).max(0.0)
}

pub fn judge(value: Option<f32>, rule: Rule) -> Verdict {
    let Some(v) = value else {
        return Verdict { score: (!matches!(rule, Rule::Raw)).then_some(0.0), ..Verdict::default() };
    };
    let scored = |z: f32| Verdict { z: Some(z), score: Some(likeness(z)), broken: None };
    match rule {
        Rule::Like { reference, tolerance } => scored((v - reference) / tolerance.max(1e-6)),
        Rule::AtLeast { reference, tolerance } => scored(((v - reference) / tolerance.max(1e-6)).min(0.0)),
        Rule::NoWorse { reference, tolerance } => scored(((v - reference) / tolerance.max(1e-6)).max(0.0)),
        Rule::AtMost { limit } => Verdict { broken: (v > limit).then_some((v, limit)), ..Verdict::default() },
        Rule::Raw => Verdict::default(),
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BenchCell {
    pub area: &'static str,
    pub scenario: String,
    pub layer: String,
    pub axis: String,
    pub part: Option<String>,
    pub form: String,
    pub value: Option<f32>,
    pub rule: Rule,
    pub cite: String,
    pub z: Option<f32>,
    pub score: Option<f32>,
    pub gate: Option<String>,
    pub holdout: bool,
}

impl BenchCell {
    #[allow(clippy::too_many_arguments)]
    pub fn new(area: &'static str, scenario: &str, layer: &str, axis: &str, part: Option<&str>, form: &str, value: Option<f32>, rule: Rule, cite: &str, holdout: bool) -> BenchCell {
        let v = judge(value, rule);
        BenchCell {
            area,
            scenario: scenario.into(),
            layer: layer.into(),
            axis: axis.into(),
            part: part.map(Into::into),
            form: form.into(),
            value,
            rule,
            cite: cite.into(),
            z: v.z,
            score: v.score,
            gate: v.broken.map(|(x, l)| format!("{axis} {x:.3} > {l:.3}")),
            holdout,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BenchScenario {
    pub area: &'static str,
    pub id: String,
    pub name_ko: String,
    pub form: String,
    pub holdout: bool,
    pub skipped: Option<String>,
    pub cells: Vec<BenchCell>,
    pub score: Option<f32>,
    pub gates: Vec<String>,
    pub passed: bool,
}

impl BenchScenario {
    pub fn from_cells(area: &'static str, id: &str, name_ko: &str, form: &str, holdout: bool, cells: Vec<BenchCell>) -> BenchScenario {
        let gates: Vec<String> = cells.iter().filter_map(|c| c.gate.clone()).collect();
        let scored: Vec<f32> = cells.iter().filter_map(|c| c.score).collect();
        let score = if !gates.is_empty() { Some(0.0) } else { mean(&scored) };
        let within = cells.iter().all(|c| c.score.is_none() || c.z.map_or(c.score.unwrap_or(0.0) > 0.0, |z| z.abs() <= 2.0));
        BenchScenario { area, id: id.into(), name_ko: name_ko.into(), form: form.into(), holdout, skipped: None, passed: gates.is_empty() && within && !scored.is_empty(), score, gates, cells }
    }
}

fn mean(xs: &[f32]) -> Option<f32> {
    (!xs.is_empty()).then(|| xs.iter().sum::<f32>() / xs.len() as f32)
}

pub fn form_id(body: &Body) -> String {
    format!("{}:{:?}:f{}", body.plan.id(), body.base, body.fingers.count()).to_lowercase()
}

pub fn from_ax(form: &str, r: &AxResult) -> BenchScenario {
    let cells = r
        .cells
        .iter()
        .map(|c| BenchCell {
            area: "AX",
            scenario: r.id.into(),
            layer: c.layer.id().into(),
            axis: c.axis.into(),
            part: c.part.map(Into::into),
            form: form.into(),
            value: c.value,
            rule: c.rule.into(),
            cite: c.cite.into(),
            z: c.z,
            score: c.score,
            gate: c.gate.clone(),
            holdout: r.holdout,
        })
        .collect();
    let score = if r.skipped.is_some() {
        None
    } else if !r.gates.is_empty() {
        Some(0.0)
    } else {
        r.layer_scores.iter().any(Option::is_some).then_some(r.score)
    };
    BenchScenario { area: "AX", id: r.id.into(), name_ko: r.name_ko.into(), form: form.into(), holdout: r.holdout, skipped: r.skipped.clone(), cells, score, gates: r.gates.clone(), passed: r.passed }
}

pub fn from_fx(form: &str, r: &FxResult) -> BenchScenario {
    let cells = r
        .cells
        .iter()
        .map(|c| BenchCell {
            area: "FX",
            scenario: r.id.into(),
            layer: c.effect.id().into(),
            axis: c.axis.into(),
            part: None,
            form: form.into(),
            value: c.value,
            rule: c.rule.into(),
            cite: c.cite.into(),
            z: c.z,
            score: c.score,
            gate: c.gate.clone(),
            holdout: r.holdout,
        })
        .collect();
    let score = if r.skipped.is_some() {
        None
    } else if !r.gates.is_empty() {
        Some(0.0)
    } else {
        r.effect_scores.iter().any(Option::is_some).then_some(r.score)
    };
    BenchScenario { area: "FX", id: r.id.into(), name_ko: r.name_ko.into(), form: form.into(), holdout: r.holdout, skipped: r.skipped.clone(), cells, score, gates: r.gates.clone(), passed: r.passed }
}

pub fn from_nn(form: &str, r: &ScenarioResult) -> BenchScenario {
    let mut cells = Vec::new();
    for p in &r.phases {
        for c in &p.cells {
            let rule = match (c.reference, c.tolerance) {
                (Some(reference), Some(tolerance)) => Rule::Like { reference, tolerance },
                _ => Rule::Raw,
            };
            cells.push(BenchCell {
                area: "NN",
                scenario: r.id.into(),
                layer: p.id.clone(),
                axis: c.axis.clone(),
                part: Some(format!("{}>{}", c.source.id(), c.target.id())),
                form: form.into(),
                value: Some(c.value),
                rule,
                cite: c.cite.unwrap_or("").into(),
                z: c.z,
                score: c.score,
                gate: None,
                holdout: r.holdout,
            });
        }
    }
    let score = if r.skipped.is_some() {
        None
    } else if r.gated {
        Some(0.0)
    } else {
        r.phases.iter().any(|p| p.seg_scores.iter().any(Option::is_some)).then_some(r.score)
    };
    BenchScenario { area: "NN", id: r.id.into(), name_ko: r.name_ko.into(), form: form.into(), holdout: r.holdout, skipped: r.skipped.clone(), cells, score, gates: r.gates.clone(), passed: r.passed }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Form {
    pub plan: &'static str,
    pub base: BaseKind,
    pub fingers: FingerCount,
}

impl Form {
    pub fn body(&self) -> Body {
        Body::new(&BodyPlan::named(self.plan).expect("a named body plan"), self.base, self.fingers)
    }

    pub fn id(&self) -> String {
        form_id(&self.body())
    }
}

pub const PLANS: [&str; 5] = ["budget", "mid", "pro", "pro_arms", "wheel_lift"];
pub const BASES: [BaseKind; 3] = [BaseKind::HumanFoot, BaseKind::Wheel, BaseKind::Track];
pub const HOLDOUT_PLANS: [&str; 2] = ["mid", "pro"];

pub fn forms() -> Vec<Form> {
    let mut v: Vec<Form> = PLANS.iter().flat_map(|p| BASES.iter().map(move |b| Form { plan: p, base: *b, fingers: FingerCount::Five })).collect();
    v.push(Form { plan: "mid", base: BaseKind::Mecanum, fingers: FingerCount::Five });
    v
}

fn counts_holdout(f: &Form) -> bool {
    HOLDOUT_PLANS.contains(&f.plan)
}

pub fn run_ax(forms: &[Form]) -> Vec<BenchScenario> {
    let mut out = Vec::new();
    for f in forms {
        let (body, id) = (f.body(), f.id());
        out.extend(actbench::library().iter().map(|s| from_ax(&id, &actbench::evaluate(&body, s))));
        if counts_holdout(f) {
            out.extend(actbench::holdout(HOLDOUT_SEED, 6).iter().map(|s| from_ax(&id, &actbench::evaluate(&body, s))));
        }
    }
    out
}

pub fn run_fx(forms: &[Form]) -> Vec<BenchScenario> {
    let mut out = Vec::new();
    for f in forms {
        let (body, id) = (f.body(), f.id());
        out.extend(effects::library().iter().map(|s| from_fx(&id, &effects::evaluate(&body, s))));
        if counts_holdout(f) {
            out.extend(effects::holdout(HOLDOUT_SEED, 8).iter().map(|s| from_fx(&id, &effects::evaluate(&body, s))));
        }
    }
    out
}

pub fn run_nn(forms: &[Form]) -> Vec<BenchScenario> {
    let ff = FeedForward::default();
    let mut out = Vec::new();
    for f in forms {
        let (body, id) = (f.body(), f.id());
        out.extend(nn::library(&body).iter().map(|s| from_nn(&id, &nn::evaluate(&body, s, &ff))));
        if counts_holdout(f) {
            out.extend(nn::holdout(&body, HOLDOUT_SEED, 8).iter().map(|s| from_nn(&id, &nn::evaluate(&body, s, &ff))));
        }
    }
    out
}

#[derive(Clone, Copy, Debug)]
pub struct Area {
    pub id: &'static str,
    pub name_ko: &'static str,
    pub run: fn(&[Form]) -> Vec<BenchScenario>,
}

pub fn core_areas() -> Vec<Area> {
    vec![
        Area { id: "AX", name_ko: "행동", run: run_ax },
        Area { id: "FX", name_ko: "환경 효과", run: run_fx },
        Area { id: "NN", name_ko: "복합 N:N", run: run_nn },
    ]
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FarCell {
    pub form: String,
    pub scenario: String,
    pub layer: String,
    pub axis: String,
    pub z: f32,
    pub value: f32,
    pub reference: f32,
    pub tolerance: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Summary {
    pub area: String,
    pub name_ko: String,
    pub scenarios: usize,
    pub record_only: usize,
    pub skipped: usize,
    pub gated: usize,
    pub cells: usize,
    pub n_ref: usize,
    pub unanswered: usize,
    pub gate_cells: usize,
    pub median_abs_z: Option<f32>,
    pub rms_z: Option<f32>,
    pub max_abs_z: Option<f32>,
    pub within1: Option<f32>,
    pub within2: Option<f32>,
    pub mean_score: Option<f32>,
    pub holdout_run: usize,
    pub holdout_passed: usize,
    pub worst: Vec<FarCell>,
}

pub fn median(xs: &[f32]) -> Option<f32> {
    if xs.is_empty() {
        return None;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    Some(if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) })
}

pub fn summarize(area: &str, name_ko: &str, scenarios: &[BenchScenario]) -> Summary {
    let dev: Vec<&BenchScenario> = scenarios.iter().filter(|s| !s.holdout).collect();
    let held: Vec<&BenchScenario> = scenarios.iter().filter(|s| s.holdout && s.skipped.is_none()).collect();
    let cells: Vec<&BenchCell> = dev.iter().flat_map(|s| s.cells.iter()).collect();
    let refs: Vec<&BenchCell> = cells.iter().copied().filter(|c| c.rule.is_reference() && c.value.is_some() && c.z.is_some()).collect();
    let az: Vec<f32> = refs.iter().filter_map(|c| c.z).map(f32::abs).collect();
    let n = az.len();
    let share = |lim: f32| (n > 0).then(|| az.iter().filter(|z| **z <= lim).count() as f32 / n as f32);
    let mut far: Vec<FarCell> = refs
        .iter()
        .filter_map(|c| {
            let (reference, tolerance) = c.rule.reference()?;
            Some(FarCell { form: c.form.clone(), scenario: c.scenario.clone(), layer: c.layer.clone(), axis: c.axis.clone(), z: c.z?, value: c.value?, reference, tolerance })
        })
        .filter(|f| f.z != 0.0)
        .collect();
    far.sort_by(|a, b| b.z.abs().total_cmp(&a.z.abs()));
    far.truncate(3);
    let scores: Vec<f32> = dev.iter().filter_map(|s| s.score).collect();
    Summary {
        area: area.into(),
        name_ko: name_ko.into(),
        scenarios: dev.len(),
        record_only: dev.iter().filter(|s| s.skipped.is_none() && s.score.is_none()).count(),
        skipped: dev.iter().filter(|s| s.skipped.is_some()).count(),
        gated: dev.iter().filter(|s| !s.gates.is_empty()).count(),
        cells: cells.len(),
        n_ref: n,
        unanswered: cells.iter().filter(|c| c.value.is_none() && !matches!(c.rule, Rule::Raw)).count(),
        gate_cells: cells.iter().filter(|c| c.gate.is_some()).count(),
        median_abs_z: median(&az),
        rms_z: (n > 0).then(|| (az.iter().map(|z| z * z).sum::<f32>() / n as f32).sqrt()),
        max_abs_z: az.iter().copied().reduce(f32::max),
        within1: share(1.0),
        within2: share(2.0),
        mean_score: mean(&scores),
        holdout_run: held.len(),
        holdout_passed: held.iter().filter(|s| s.passed).count(),
        worst: far,
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Spread {
    pub areas: usize,
    pub max_median: f32,
    pub worst: String,
    pub min_median: f32,
    pub best: String,
    pub cv: f32,
    pub max_rms: f32,
    pub min_rms: f32,
}

pub fn spread(summaries: &[Summary]) -> Option<Spread> {
    let m: Vec<(&Summary, f32, f32)> = summaries.iter().filter_map(|s| Some((s, s.median_abs_z?, s.rms_z?))).collect();
    let worst = m.iter().max_by(|a, b| a.1.total_cmp(&b.1))?;
    let best = m.iter().min_by(|a, b| a.1.total_cmp(&b.1))?;
    let mu = m.iter().map(|x| x.1).sum::<f32>() / m.len() as f32;
    let sd = (m.iter().map(|x| (x.1 - mu).powi(2)).sum::<f32>() / m.len() as f32).sqrt();
    Some(Spread {
        areas: m.len(),
        max_median: worst.1,
        worst: worst.0.area.clone(),
        min_median: best.1,
        best: best.0.area.clone(),
        cv: if mu > 0.0 { sd / mu } else { 0.0 },
        max_rms: m.iter().map(|x| x.2).fold(f32::NEG_INFINITY, f32::max),
        min_rms: m.iter().map(|x| x.2).fold(f32::INFINITY, f32::min),
    })
}

fn opt(v: Option<f32>, digits: usize) -> String {
    v.map_or("—".into(), |x| format!("{x:.digits$}"))
}

fn pct(v: Option<f32>) -> String {
    v.map_or("—".into(), |x| format!("{:.0}%", x * 100.0))
}

pub fn ux_lines(summaries: &[Summary]) -> Vec<String> {
    let mut out = Vec::new();
    for s in summaries {
        let refs = if s.n_ref == 0 {
            "기준 칸 없음 (기록만)".to_string()
        } else {
            format!(
                "기준 칸 {} · 중앙|z| {} · RMS z {} · 최대|z| {} · |z|≤1 {} · |z|≤2 {}",
                s.n_ref,
                opt(s.median_abs_z, 2),
                opt(s.rms_z, 2),
                opt(s.max_abs_z, 1),
                pct(s.within1),
                pct(s.within2)
            )
        };
        out.push(format!(
            "[UX] {} {} · 시나리오 {} (기록만 {} · 건너뜀 {} · 게이트 {}) · 칸 {} · {} · 미구현 {} · 평균 점수 {} · 홀드아웃 통과 {}/{}",
            s.area,
            s.name_ko,
            s.scenarios,
            s.record_only,
            s.skipped,
            s.gated,
            s.cells,
            refs,
            s.unanswered,
            opt(s.mean_score, 1),
            s.holdout_passed,
            s.holdout_run
        ));
        for c in &s.worst {
            let at = if c.layer.is_empty() { c.axis.clone() } else { format!("{}:{}", c.layer, c.axis) };
            out.push(format!("[UX-CELL] {} {} {}/{at} z {:+.1} (값 {:.3}, 기준 {:.3} ± {:.3})", s.area, c.form, c.scenario, c.z, c.value, c.reference, c.tolerance));
        }
    }
    match spread(summaries) {
        Some(sp) => {
            let name = summaries.iter().find(|s| s.area == sp.worst).map_or("", |s| s.name_ko.as_str());
            out.push(format!(
                "[UX-SUM] 영역 {}개 · 중앙|z| 최대 {:.2} ({}) / 최소 {:.2} ({}) · 변동계수 {:.2} · RMS z {:.2}–{:.2} · 다음 작업: {} {} (기준에서 가장 먼 영역; 상수를 맞추지 않고 모델로 고침)",
                sp.areas, sp.max_median, sp.worst, sp.min_median, sp.best, sp.cv, sp.min_rms, sp.max_rms, sp.worst, name
            ));
        }
        None => out.push("[UX-SUM] 기준 칸이 있는 영역 없음".into()),
    }
    out
}

pub const ENGINE_FILES: [&str; 13] = [
    "crates/noto-core/src/**/*.rs",
    "crates/world-sim/src/**/*.rs",
    "crates/noto-vision/src/**/*.rs",
    "crates/noto-runtime/src/driver.rs",
    "crates/noto-runtime/src/mldrive.rs",
    "crates/noto-runtime/src/vision.rs",
    "crates/noto-runtime/src/precision.rs",
    "crates/noto-runtime/src/runtime.rs",
    "crates/noto-runtime/src/record.rs",
    "crates/noto-runtime/src/cargo.rs",
    "crates/noto-runtime/src/chase.rs",
    "crates/noto-runtime/src/perception.rs",
    "crates/noto-runtime/src/relay.rs",
];

pub const NOT_ENGINE: [(&str, &str); 3] = [
    ("crates/noto-core/src/lib.rs", "crate root: the module list and its docs name the benchmark modules"),
    ("crates/noto-core/src/effects.rs", "the environment benchmark (references)"),
    ("crates/noto-core/src/nn.rs", "the composite N:N benchmark (references)"),
];

pub const REFERENCE_SECTIONS: [(&str, &str); 1] = [("crates/noto-core/src/balance.rs", "pub struct RefCtx {")];

pub const DECISION_FILES: [&str; 5] = ["crates/noto-runtime/src/driver.rs", "crates/noto-runtime/src/mldrive.rs", "crates/noto-runtime/src/vision.rs", "crates/noto-runtime/src/precision.rs", "crates/noto-runtime/src/relay.rs"];

pub const REFERENCE_READS: [&str; 13] = [
    "reference_report",
    "exact_reference",
    "area_report",
    ":dev\"",
    ":absdev",
    ":dev2",
    ":in_band",
    "actbench",
    "effects::",
    "nn::",
    "composite_reference",
    "reference(",
    "bench::",
];

#[derive(Clone, Copy, Debug)]
pub struct SourceNames {
    pub area: &'static str,
    pub names: &'static [&'static str],
}

pub const BENCH_SOURCES: [SourceNames; 10] = [
    SourceNames {
        area: "AX",
        names: &[
            "Guillery", "Johansson", "Westling", "Flanagan", "Menzer", "Reiser", "Bazrgari", "Kingma", "Plamondon", "Hwang", "von Arx", "Dolan", "NIOSH", "Waters", "Liberty Mutual", "Snook", "Steele", "Mahler", "Dex-Net", "QT-Opt", "Kalashnikov", "Levine", "ten Pas", "Morrison", "Berscheid", "Bauza", "Beschorner", "Wang 2017",
            "Jafarnezhadgero", "Zhang 2013", "Wallace", "Baines", "Lockhart", "Pai 2010", "Yang 2012", "Brady", "Stansfield", "Vaughan", "Nilsson", "Keller", "Holowka", "Warmerdam", "Agarwal", "Yoshida",
        ],
    },
    SourceNames { area: "FX", names: &["Hanson", "Beschorner", "Rodríguez-Martínez", "Lejeune", "Knapik", "Middleton", "Frey Law", "Frey-Law", "maxon"] },
    SourceNames {
        area: "NN",
        names: &[
            "Jbabdi", "de Kam", "Krašna", "Schlenstedt", "Kanekar", "Mirka", "Franchignoni", "Horak 2009", "Padgett", "Hignett", "McAtamney", "Corlett", "Marcondes", "Atsawakaewmongkhon", "Aruin & Latash", "Hudson 2018", "Pozzo", "BESTest", "REBA", "RULA", "DARPA",
        ],
    },
    SourceNames { area: "CAM", names: &["Milella", "Reina", "Moon, Leary", "Moon 2025", "Waite", "Menz", "Kavanagh"] },
    SourceNames {
        area: "VX",
        names: &[
            "Loomis", "Rieser", "Thomson", "Steenhuis", "Elliott", "Madalena", "Westwood", "Hesse", "Patla", "Hollands", "Laurent", "Assaiante", "TidyBot", "Dusty, King", "Deyle", "Ramanan", "Holbein", "Wysog", "King, Judge", "Toussaint", "van der Burg",
        ],
    },
    SourceNames { area: "CX", names: &["Choi et al", "Wakeling", "Kroonenberg", "Tan et al", "Hsiao", "Laing", "Kumar et al", "Karydis", "Wimer", "Llado", "Sentis"] },
    SourceNames {
        area: "TX",
        names: &["Delhaye", "Thonnard", "Hager-Ross", "Häger-Ross", "Luciw", "WAY-EEG-GAL", "Jacob", "Hughes", "Klenerman", "Koo et al", "Hessert", "Chou et al"],
    },
    SourceNames { area: "KX", names: &["DGUV", "IFA 4224", "Kluth", "Dennis & Barrett", "Dennis and Barrett", "Sharp et al", "Nindl", "Visser", "Loiu", "Lanini", "Sambhus"] },
    SourceNames { area: "MX", names: &["Wevolver", "Konica", "GIGAZINE", "iFixit", "Techzine"] },
    SourceNames { area: "CODE", names: &["actbench", "effects::", "noto_core::nn::", "crate::nn::", "composite_reference"] },
];

pub const SOURCE_EXCEPTIONS: [(&str, &str, &str); 1] = [(
    "crates/noto-runtime/src/vision.rs",
    "Johansson",
    "Johansson et al. 2001 (eye-hand coordination: gaze reaches the grasp site before the hand) sets when the camera looks; the grasp benchmark scores grip-load coupling against Guillery 2013, not gaze timing",
)];

pub fn names(text: &str, needle: &str) -> bool {
    let ident = |c: char| c.is_alphanumeric() || c == '_';
    let open_start = needle.chars().next().is_some_and(ident);
    let open_end = needle.chars().next_back().is_some_and(ident);
    text.match_indices(needle).any(|(i, _)| {
        let before = open_start && text[..i].chars().next_back().is_some_and(ident);
        let after = open_end && text[i + needle.len()..].chars().next().is_some_and(ident);
        !before && !after
    })
}

pub fn is_bench_module(rel: &str) -> bool {
    rel.starts_with("crates/noto-core/src/") && rel.ends_with("bench.rs")
}

fn excluded(rel: &str) -> bool {
    is_bench_module(rel) || NOT_ENGINE.iter().any(|(f, _)| *f == rel)
}

fn walk(root: &Path, rel_dir: &str, out: &mut Vec<String>) {
    let Ok(rd) = std::fs::read_dir(root.join(rel_dir)) else { return };
    let mut entries: Vec<(String, bool)> = rd.filter_map(|e| e.ok()).map(|e| (e.file_name().to_string_lossy().into_owned(), e.file_type().map_or(false, |t| t.is_dir()))).collect();
    entries.sort();
    for (name, dir) in entries {
        let rel = format!("{rel_dir}/{name}");
        if dir {
            walk(root, &rel, out);
        } else if name.ends_with(".rs") {
            out.push(rel);
        }
    }
}

pub fn engine_files(root: &Path) -> Vec<String> {
    let mut out = Vec::new();
    for p in ENGINE_FILES {
        match p.strip_suffix("/**/*.rs") {
            Some(dir) => walk(root, dir, &mut out),
            None => {
                if root.join(p).is_file() {
                    out.push(p.to_string());
                }
            }
        }
    }
    out.retain(|f| !excluded(f));
    out.sort();
    out.dedup();
    out
}

pub fn engine_text<'a>(rel: &str, text: &'a str) -> &'a str {
    match REFERENCE_SECTIONS.iter().find(|(f, _)| *f == rel) {
        Some((_, marker)) => text.split(*marker).next().unwrap_or(text),
        None => text,
    }
}

pub fn source_hits(rel: &str, text: &str) -> Vec<(&'static str, &'static str)> {
    let text = engine_text(rel, text);
    let mut out = Vec::new();
    for s in BENCH_SOURCES {
        for n in s.names {
            if names(text, n) && !SOURCE_EXCEPTIONS.iter().any(|(f, x, _)| *f == rel && x == n) {
                out.push((s.area, *n));
            }
        }
    }
    out
}

pub fn reference_reads(text: &str) -> Vec<&'static str> {
    REFERENCE_READS.iter().copied().filter(|n| names(text, n)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_judge_scores_likeness_one_sided_rules_and_gates() {
        let like = judge(Some(1.3), Rule::Like { reference: 1.0, tolerance: 0.1 });
        assert!((like.z.unwrap() - 3.0).abs() < 1e-4 && (like.score.unwrap() - 70.0).abs() < 1e-3);
        let above = judge(Some(4.0), Rule::AtLeast { reference: 1.0, tolerance: 0.5 });
        assert_eq!((above.z, above.score), (Some(0.0), Some(100.0)), "above the floor is full marks");
        let below = judge(Some(0.5), Rule::AtLeast { reference: 1.0, tolerance: 0.5 });
        assert_eq!((below.z, below.score), (Some(-1.0), Some(90.0)));
        let better = judge(Some(0.01), Rule::NoWorse { reference: 0.1, tolerance: 0.05 });
        assert_eq!((better.z, better.score), (Some(0.0), Some(100.0)), "a smaller error than the reference is full marks");
        let worse = judge(Some(0.2), Rule::NoWorse { reference: 0.1, tolerance: 0.05 });
        assert!((worse.z.unwrap() - 2.0).abs() < 1e-5 && (worse.score.unwrap() - 80.0).abs() < 1e-3);
        let gate = judge(Some(1.2), Rule::AtMost { limit: 1.0 });
        assert_eq!((gate.broken, gate.score, gate.z), (Some((1.2, 1.0)), None, None));
        assert_eq!(judge(Some(0.9), Rule::AtMost { limit: 1.0 }), Verdict::default());
        assert_eq!(judge(None, Rule::Like { reference: 1.0, tolerance: 0.1 }).score, Some(0.0), "a missing model is not a pass");
        assert_eq!(judge(None, Rule::AtMost { limit: 1.0 }).score, Some(0.0));
        assert_eq!(judge(None, Rule::Raw), Verdict::default());
        assert_eq!(judge(Some(5.0), Rule::Raw), Verdict::default());
        assert_eq!(likeness(12.0), 0.0, "the score saturates; the z does not");
    }

    #[test]
    fn the_effects_rule_converts_field_for_field() {
        assert_eq!(Rule::from(effects::Rule::Like { reference: 0.5, tolerance: 0.1 }), Rule::Like { reference: 0.5, tolerance: 0.1 });
        assert_eq!(Rule::from(effects::Rule::AtMost { limit: 1.0 }), Rule::AtMost { limit: 1.0 });
        assert_eq!(format!("{:?}", Rule::NoWorse { reference: 0.115, tolerance: 0.08 }), "NoWorse { reference: 0.115, tolerance: 0.08 }");
        assert_eq!(format!("{:?}", Rule::Like { reference: 0.05, tolerance: 0.0176 }), "Like { reference: 0.05, tolerance: 0.0176 }");
    }

    fn cell(scenario: &str, value: Option<f32>, rule: Rule, holdout: bool) -> BenchCell {
        BenchCell::new("T", scenario, "l", "x", None, "f", value, rule, "", holdout)
    }

    #[test]
    fn the_summary_counts_record_only_scenarios_as_such() {
        let scen = vec![
            BenchScenario::from_cells("T", "A", "", "f", false, vec![cell("A", Some(0.5), Rule::Like { reference: 0.0, tolerance: 1.0 }, false), cell("A", Some(-1.5), Rule::Like { reference: 0.0, tolerance: 1.0 }, false)]),
            BenchScenario::from_cells("T", "B", "", "f", false, vec![cell("B", Some(3.0), Rule::Like { reference: 0.0, tolerance: 1.0 }, false), cell("B", Some(2.0), Rule::AtMost { limit: 1.0 }, false)]),
            BenchScenario::from_cells("T", "C", "", "f", false, vec![cell("C", Some(7.0), Rule::Raw, false)]),
            BenchScenario::from_cells("T", "D", "", "f", false, vec![cell("D", None, Rule::Like { reference: 0.0, tolerance: 1.0 }, false)]),
            BenchScenario::from_cells("T", "H", "", "f", true, vec![cell("H", Some(9.0), Rule::Like { reference: 0.0, tolerance: 1.0 }, true)]),
            BenchScenario::from_cells("T", "H2", "", "f", true, vec![cell("H2", Some(0.1), Rule::Like { reference: 0.0, tolerance: 1.0 }, true)]),
        ];
        assert_eq!(scen[2].score, None, "record only");
        assert_eq!(scen[1].score, Some(0.0), "gated");
        let s = summarize("T", "시험", &scen);
        assert_eq!((s.scenarios, s.record_only, s.gated, s.skipped), (4, 1, 1, 0));
        assert_eq!((s.n_ref, s.unanswered, s.gate_cells), (3, 1, 1));
        assert!((s.median_abs_z.unwrap() - 1.5).abs() < 1e-6);
        let rms = ((0.25f32 + 2.25 + 9.0) / 3.0).sqrt();
        assert!((s.rms_z.unwrap() - rms).abs() < 1e-5);
        assert!((s.within1.unwrap() - 1.0 / 3.0).abs() < 1e-6 && (s.within2.unwrap() - 2.0 / 3.0).abs() < 1e-6);
        assert_eq!(s.max_abs_z, Some(3.0));
        assert_eq!((s.holdout_run, s.holdout_passed), (2, 1));
        assert_eq!(s.worst[0].scenario, "B");
        assert!((s.mean_score.unwrap() - (90.0 + 0.0 + 0.0) / 3.0).abs() < 1e-3, "{:?}", s.mean_score);
    }

    #[test]
    fn the_spread_names_the_farthest_area() {
        let mk = |area: &str, z: f32| summarize(area, area, &[BenchScenario::from_cells("T", "A", "", "f", false, vec![cell("A", Some(z), Rule::Like { reference: 0.0, tolerance: 1.0 }, false)])]);
        let s = vec![mk("P", 1.0), mk("Q", 3.0), summarize("R", "R", &[])];
        let sp = spread(&s).unwrap();
        assert_eq!((sp.areas, sp.worst.as_str(), sp.best.as_str()), (2, "Q", "P"));
        assert!((sp.cv - 0.5).abs() < 1e-6, "SD 1 over mean 2: {}", sp.cv);
        let lines = ux_lines(&s);
        assert!(lines.last().unwrap().starts_with("[UX-SUM]") && lines.last().unwrap().contains("다음 작업: Q"));
        assert!(lines.iter().any(|l| l.starts_with("[UX] R") && l.contains("기준 칸 없음")));
        assert!(lines.iter().any(|l| l.starts_with("[UX-CELL] Q f A/l:x z +3.0")), "{lines:?}");
        let bare = summarize("V", "V", &[BenchScenario::from_cells("V", "V1", "", "f", false, vec![BenchCell::new("V", "V1", "", "err", None, "f", Some(2.0), Rule::Like { reference: 0.0, tolerance: 1.0 }, "", false)])]);
        assert!(ux_lines(&[bare]).iter().any(|l| l.starts_with("[UX-CELL] V f V1/err z +2.0")));
        assert!(spread(&[summarize("R", "R", &[])]).is_none());
        let even = summarize("E", "E", &[BenchScenario::from_cells("E", "E1", "", "f", false, vec![
            cell("E1", Some(0.01), Rule::NoWorse { reference: 0.1, tolerance: 0.05 }, false),
            cell("E1", Some(1.0), Rule::Like { reference: 1.0, tolerance: 0.1 }, false),
            cell("E1", Some(1.15), Rule::Like { reference: 1.0, tolerance: 0.1 }, false),
        ])]);
        assert_eq!((even.n_ref, even.worst.len()), (3, 1));
        assert_eq!(even.median_abs_z, Some(0.0), "the zeros still count in the deviation");
    }

    #[test]
    fn names_respect_identifier_boundaries() {
        assert!(!names("use candle_nn::VarBuilder;", "nn::"));
        assert!(names("noto_core::nn::cells(", "nn::"));
        assert!(names("(Hanson 1999)", "Hanson"));
        assert!(!names("Johanson", "Hanson"));
        assert!(names("let r = reference(src);", "reference("));
        assert!(!names("fn calibrated_reference(", "reference("));
        assert!(!names("the Jacobian transpose", "Jacob"));
        assert!(names("(Jacob 1989)", "Jacob") && names("Jacob's thesis", "Jacob"));
        assert!(names("NIOSH's equation", "NIOSH") && !names("Hughesville", "Hughes"));
        assert!(names("x = 1; // Koo et al. 2019", "Koo et al"));
        assert!(names("sds.agreement_rate(&s, \"margin:in_band\")", ":in_band"));
        assert!(names("baseline(&s, \"preload_s:dev\")", ":dev\""));
        assert!(names("format!(\"{axis}:absdev\")", ":absdev") && !names("\"x:absdevs\"", ":absdev"));
        assert!(!names("\"x:dev_note\"", ":dev\""));
        assert_eq!(engine_text("crates/noto-core/src/balance.rs", "engine\npub struct RefCtx {\nfn reference("), "engine\n");
        assert_eq!(engine_text("x.rs", "a pub struct RefCtx { b"), "a pub struct RefCtx { b");
    }

    #[test]
    fn a_planted_source_is_caught_and_an_exception_is_specific() {
        assert_eq!(source_hits("crates/world-sim/src/world.rs", "// friction from Hanson 1999"), vec![("FX", "Hanson")]);
        assert!(source_hits("crates/noto-runtime/src/vision.rs", "(Johansson et al. 2001)").is_empty());
        assert_eq!(source_hits("crates/noto-runtime/src/driver.rs", "(Johansson et al. 2001)"), vec![("AX", "Johansson")]);
        assert_eq!(reference_reads("let r = sds.reference_report(&s, \"x\");"), vec!["reference_report"]);
        assert!(excluded("crates/noto-core/src/actbench.rs") && excluded("crates/noto-core/src/bench.rs") && excluded("crates/noto-core/src/nn.rs"));
        assert!(!excluded("crates/noto-core/src/kin.rs"));
        assert!(excluded("crates/noto-core/src/collbench.rs") && excluded("crates/noto-core/src/tactbench.rs"));
        assert!(!excluded("crates/world-sim/src/workbench.rs"), "a bench outside noto-core is engine");
        assert!(!excluded("crates/noto-core/src/benchmark_free.rs"));
    }
}
