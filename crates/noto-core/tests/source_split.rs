use std::path::PathBuf;

use noto_core::bench::{engine_files, engine_text, reference_reads, source_hits, BENCH_SOURCES, DECISION_FILES, NOT_ENGINE, SOURCE_EXCEPTIONS};

fn root() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
}

fn read(rel: &str) -> String {
    std::fs::read_to_string(root().join(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"))
}

#[test]
fn the_registry_finds_the_engine_and_leaves_the_benchmarks_out() {
    let files = engine_files(&root());
    for f in [
        "crates/noto-core/src/kin.rs",
        "crates/noto-core/src/balance.rs",
        "crates/noto-core/src/act/grasp.rs",
        "crates/noto-core/src/act/gait.rs",
        "crates/world-sim/src/world.rs",
        "crates/world-sim/src/world/acts.rs",
        "crates/world-sim/src/world/eye.rs",
        "crates/world-sim/src/render.rs",
        "crates/noto-vision/src/decide.rs",
        "crates/noto-runtime/src/driver.rs",
        "crates/noto-runtime/src/vision.rs",
        "crates/noto-runtime/src/relay.rs",
    ] {
        assert!(files.iter().any(|x| x == f), "{f} is engine: {files:?}");
    }
    // What the robot felt goes back into its grip there (`relay`): a decision file, so it is held to never reading a
    // reference axis.
    assert!(DECISION_FILES.contains(&"crates/noto-runtime/src/relay.rs"));
    for f in ["crates/noto-core/src/actbench.rs", "crates/noto-core/src/bench.rs", "crates/noto-core/src/effects.rs", "crates/noto-core/src/nn.rs", "crates/noto-core/src/lib.rs", "crates/noto-runtime/src/actsds.rs", "crates/noto-runtime/src/api.rs"] {
        assert!(!files.iter().any(|x| x == f), "{f} is not engine");
    }
    for (f, _) in NOT_ENGINE {
        assert!(root().join(f).is_file(), "{f}");
    }
    for (f, _, _) in SOURCE_EXCEPTIONS {
        assert!(root().join(f).is_file(), "{f}");
    }
    let balance = read("crates/noto-core/src/balance.rs");
    let engine = engine_text("crates/noto-core/src/balance.rs", &balance);
    assert!(engine.len() < balance.len() && engine.contains("impl Body") && !engine.contains("pub fn reference("));
    println!("[SRC] 엔진 파일 {}개 · 영역 {}개의 채점 출처 {}개", files.len(), BENCH_SOURCES.len(), BENCH_SOURCES.iter().map(|s| s.names.len()).sum::<usize>());
}

#[test]
fn the_engine_never_names_any_benchmark_source() {
    let mut hits = Vec::new();
    for f in engine_files(&root()) {
        let text = read(&f);
        for (area, name) in source_hits(&f, &text) {
            hits.push(format!("{f} names the {area} benchmark source `{name}`"));
        }
    }
    assert!(hits.is_empty(), "{}", hits.join("\n"));
}

#[test]
fn decisions_never_read_reference_axes() {
    for f in DECISION_FILES {
        let reads = reference_reads(&read(f));
        assert!(reads.is_empty(), "{f} reads {reads:?}");
    }
}

#[test]
fn every_area_lists_distinct_sources() {
    let mut areas: Vec<&str> = BENCH_SOURCES.iter().map(|s| s.area).collect();
    areas.sort();
    areas.dedup();
    assert_eq!(areas.len(), BENCH_SOURCES.len(), "one row per area");
    for s in BENCH_SOURCES {
        assert!(!s.names.is_empty(), "{}", s.area);
        let mut n = s.names.to_vec();
        n.sort();
        n.dedup();
        assert_eq!(n.len(), s.names.len(), "{} lists a name twice", s.area);
    }
}
