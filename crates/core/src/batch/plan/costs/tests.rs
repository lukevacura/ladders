use super::{
    AARCH64, Costs, Fit, MERGE, OTHER, START, TARGET, WALK_BELOW_RUN, X86_64, break_even,
    cold_route, steps_through, walks,
};
use crate::spill::{Counts, Shape, Width};

/// This target's own table, named by its own `cfg`.
fn own() -> Costs {
    if cfg!(target_arch = "aarch64") {
        AARCH64
    } else if cfg!(target_arch = "x86_64") {
        X86_64
    } else {
        OTHER
    }
}

/// This target reads its own table, and every constant and crossover the planner reads is that
/// table's.
#[test]
fn each_target_reads_its_own_table() {
    let own = own();
    assert_eq!(TARGET, own);
    assert_eq!(
        (MERGE.to_bits(), START.to_bits()),
        (own.merge.to_bits(), own.start.to_bits())
    );
    assert_eq!(WALK_BELOW_RUN, own.walk_below_run);
    for m in [1usize, 16, 47, 48, 49, 64, 256, 1024, 1 << 20] {
        assert_eq!(break_even(m), own.break_even(m), "m {m}");
    }
    let shape = Shape {
        width: Width::Narrow128,
        m: 10,
        walk: Counts {
            cells: 1000,
            chains: 5,
        },
        required: Counts::default(),
        wide_cells_today: 400,
    };
    let _guard = crate::route::forced_route(crate::route::RouteForce::default());
    assert_eq!(cold_route(&shape), own.cold.plan(&shape));
    assert_ne!(AARCH64, X86_64, "the tables differ, so a swap shows");
    assert_ne!(
        AARCH64.break_even(1024),
        X86_64.break_even(1024),
        "the catch-up fits differ, so a swap shows"
    );
}

/// Any other target takes the larger of the two measured costs, each, and the larger of the two
/// catch-up fits.
#[test]
fn another_target_takes_the_larger_cost() {
    let larger = |f: fn(&Costs) -> u64| f(&AARCH64).max(f(&X86_64));
    assert_eq!(OTHER.merge, AARCH64.merge.max(X86_64.merge));
    assert_eq!(OTHER.start, AARCH64.start.max(X86_64.start));
    assert_eq!(
        OTHER.walk_below_run,
        AARCH64.walk_below_run.max(X86_64.walk_below_run)
    );
    assert_eq!(OTHER.cold.cell_128, larger(|c| c.cold.cell_128));
    assert_eq!(OTHER.cold.cell_256, larger(|c| c.cold.cell_256));
    assert_eq!(OTHER.cold.dot_term_128, larger(|c| c.cold.dot_term_128));
    assert_eq!(OTHER.cold.dot_term_256, larger(|c| c.cold.dot_term_256));
    assert_eq!(
        OTHER.cold.wide_extra_cell,
        larger(|c| c.cold.wide_extra_cell)
    );
    for m in [1usize, 16, 47, 48, 49, 64, 256, 1024, 1 << 20] {
        assert_eq!(
            OTHER.break_even(m),
            AARCH64.break_even(m).max(X86_64.break_even(m)),
            "m {m}"
        );
    }
}

/// Each target's catch-up crossover is its own fit: NEON `m/4 + 4` (8, 20, 68, 260 at m 16, 64,
/// 256 and 1024) and x86 `3m/8` (6, 24, 96, 384); any other target takes the larger.
#[test]
fn each_target_takes_its_own_catch_up_fit() {
    let ms = [16usize, 64, 256, 1024];
    assert_eq!(ms.map(|m| AARCH64.break_even(m)), [8, 20, 68, 260]);
    assert_eq!(ms.map(|m| X86_64.break_even(m)), [6, 24, 96, 384]);
    assert_eq!(ms.map(|m| OTHER.break_even(m)), [8, 24, 96, 384]);
    assert_eq!(
        AARCH64.catch_up,
        &[Fit {
            num: 1,
            den: 4,
            plus: 4
        }]
    );
    assert_eq!(
        X86_64.catch_up,
        &[Fit {
            num: 3,
            den: 8,
            plus: 0
        }]
    );
    // `3m/8` rounds down whole, not per part: 7 gives 2 and 13 gives 4.
    assert_eq!([7usize, 13].map(|m| X86_64.break_even(m)), [2, 4]);
    assert_eq!(break_even(usize::MAX), TARGET.break_even(usize::MAX));
    assert!(X86_64.break_even(usize::MAX) > usize::MAX / 3);
}

/// The catch-up steps a run shorter than the crossover and resumes any other, and the
/// walk-or-tile split walks a run shorter than its threshold and tiles any other.
#[test]
fn the_crossovers_split_at_their_thresholds() {
    for m in [16usize, 64, 256, 1024] {
        let even = break_even(m);
        assert!(steps_through(even - 1, m), "m {m}");
        assert!(!steps_through(even, m), "m {m}");
    }
    const { assert!(WALK_BELOW_RUN > 1) };
    assert!(walks(1));
    assert!(walks(WALK_BELOW_RUN - 1));
    assert!(!walks(WALK_BELOW_RUN));
}

/// Each target's fit agrees with every session measured on it: within a quarter of the measured
/// `resume_ns / step_ns` at m 16, 64, 256 and 1024 (a fit swapped between targets misses by more).
#[test]
fn each_fit_agrees_with_its_targets_sessions() {
    let ms = [16usize, 64, 256, 1024];
    let neon: [[f64; 4]; 3] = [
        [7.5, 19.0, 63.0, 243.0],
        [8.6, 22.8, 75.9, 288.6],
        [8.8, 22.5, 73.1, 280.1],
    ];
    let x86: [[f64; 4]; 2] = [[5.8, 21.5, 85.0, 337.0], [5.6, 20.5, 106.0, 420.3]];
    let tables: [(Costs, &[[f64; 4]]); 2] = [(AARCH64, &neon), (X86_64, &x86)];
    for (costs, sessions) in tables {
        for session in sessions {
            for (m, measured) in ms.iter().zip(session) {
                let fit = costs.break_even(*m) as f64;
                assert!(
                    fit <= 1.25 * measured && *measured <= 1.25 * fit,
                    "m {m}: measured {measured}, the fit {fit}: {costs:?}"
                );
            }
        }
    }
}

/// Every table's costs are positive where the models divide or weigh by them, and the cold route's
/// wider tier costs more than its narrower one.
#[test]
fn every_table_is_usable() {
    for costs in [AARCH64, X86_64, OTHER] {
        assert!(costs.merge > 0.0 && costs.start > 0.0, "{costs:?}");
        assert!(costs.walk_below_run > 1, "{costs:?}");
        let cold = costs.cold;
        assert!(cold.cell_256 > cold.cell_128, "{costs:?}");
        assert!(cold.dot_term_256 > cold.dot_term_128, "{costs:?}");
        assert!(cold.wide_extra_cell > 0, "{costs:?}");
    }
}

/// The library files outside the planner, each with its text: every `.rs` file under `src`
/// except `batch/plan.rs`, `batch/plan/` and test files.
fn library_files_outside_the_planner() -> std::vec::Vec<(std::string::String, std::string::String)>
{
    use std::string::ToString;
    // timing-ok: a compile-time path, fixed per build
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut pending = std::vec![root.clone()];
    let mut files = std::vec::Vec::new();
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).expect("the source tree reads") {
            let path = entry.expect("an entry").path();
            let name = path
                .strip_prefix(&root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            let is_test = name.ends_with("tests.rs") || name.contains("tests/");
            if path.is_dir() {
                if name != "batch/plan" {
                    pending.push(path);
                }
            } else if name.ends_with(".rs") && name != "batch/plan.rs" && !is_test {
                let text = std::fs::read_to_string(&path).expect("a source file reads");
                files.push((name.to_string(), text));
            }
        }
    }
    files
}

/// One owner: no library file outside the planner names a cost constant or the cost tables, and
/// the crossovers are read only through the planner's own decisions.
#[test]
fn nothing_outside_the_planner_reads_a_cost_or_decides_a_crossover() {
    const NAMES: [&str; 8] = [
        "WALK_BELOW_RUN",
        "MERGE_COST",
        "START_COST",
        "MEASURED",
        "ColdCosts",
        "costs::",
        "fn break_even",
        "fn walks",
    ];
    let files = library_files_outside_the_planner();
    assert!(files.iter().any(|(name, _)| name == "cross_term.rs"));
    assert!(files.iter().any(|(name, _)| name == "batch/sparse.rs"));
    let mut found = std::vec::Vec::new();
    for (name, text) in &files {
        for (no, line) in text.lines().enumerate() {
            // The bench drivers' entry point names the crossover only to delegate to the planner.
            let delegates = name == "cross_term.rs"
                && matches!(
                    line.trim(),
                    "pub const fn break_even(m: usize) -> usize {" | "crate::batch::break_even(m)"
                );
            let names = NAMES.iter().any(|n| line.contains(n)) && !delegates;
            let reads = line.contains("break_even(") && !delegates;
            let splits = line.contains("runs.split(") && !line.contains("runs.split()");
            if names || reads || splits {
                found.push(std::format!("{name}:{}: {line}", no + 1));
            }
        }
    }
    assert!(
        found.is_empty(),
        "outside the planner:\n{}",
        found.join("\n")
    );
}

#[cfg(not(any(loom, shuttle)))]
mod cold {
    use super::super::ColdCosts;
    use crate::route::{RouteForce, SpillForce, forced_route};
    use crate::spill::{Counts, Shape, Width};

    fn costs() -> ColdCosts {
        ColdCosts {
            cell_128: 10,
            cell_256: 40,
            dot_term_128: 2,
            dot_term_256: 8,
            wide_extra_cell: 30,
        }
    }

    fn shape(width: Width, cells: u64, chains: u64, wide_cells_today: u64) -> Shape {
        Shape {
            width,
            m: 10,
            walk: Counts { cells, chains },
            required: Counts::default(),
            wide_cells_today,
        }
    }

    #[test]
    fn the_model_engages_when_the_cold_route_costs_no_more_than_the_i64_pass_it_replaces() {
        let _guard = forced_route(RouteForce::default());
        // 1000 cells · 10 + 5 chains · 10 · 2 = 10100 against 400 wide cells · 30 = 12000.
        let engaged = costs().plan(&shape(Width::Narrow128, 1000, 5, 400));
        assert_eq!((engaged.cold_cost, engaged.saved_cost), (10_100, 12_000));
        assert!(engaged.natural && engaged.engage);
        // The 256-bit tier at the same counts costs 40000 + 5·10·8 = 40400.
        let declined = costs().plan(&shape(Width::Wide256, 1000, 5, 400));
        assert_eq!(declined.cold_cost, 40_400);
        assert!(!declined.natural && !declined.engage);
        // Equal cost engages: 900 cells · 10 against 300 wide cells · 30; one more cell does not.
        let equal = costs().plan(&shape(Width::Narrow128, 900, 0, 300));
        assert_eq!(equal.cold_cost, equal.saved_cost);
        assert!(equal.natural);
        let one_over = costs().plan(&shape(Width::Narrow128, 901, 0, 300));
        assert!(!one_over.natural);
    }

    #[test]
    fn only_what_taking_every_window_adds_to_the_required_walk_is_priced() {
        let _guard = forced_route(RouteForce::default());
        let mut priced = shape(Width::Narrow128, 1000, 5, 400);
        // Every pair is required: taking the rest is free, so it engages whenever there is
        // anything to save, and the whole walk is not charged against it.
        priced.required = priced.walk;
        let plan = costs().plan(&priced);
        assert_eq!((plan.cold_cost, plan.saved_cost), (0, 12_000));
        assert!(plan.natural);
        priced.wide_cells_today = 0;
        assert!(!costs().plan(&priced).natural, "nothing to save");
        // Part of the walk is required: only the rest is charged.
        let mut partly = shape(Width::Narrow128, 1000, 5, 400);
        partly.required = Counts {
            cells: 100,
            chains: 1,
        };
        // 900 cells · 10 + 4 chains · 10 · 2 = 9080.
        assert_eq!(costs().plan(&partly).cold_cost, 9_080);
    }

    #[test]
    fn nothing_to_walk_never_engages_and_nothing_to_save_never_engages() {
        let _guard = forced_route(RouteForce::default());
        assert!(!costs().plan(&shape(Width::Narrow128, 0, 0, 1 << 40)).engage);
        assert!(!costs().plan(&shape(Width::Narrow128, 1, 0, 0)).engage);
    }

    #[test]
    fn the_model_saturates_instead_of_wrapping() {
        let _guard = forced_route(RouteForce::default());
        let plan = costs().plan(&shape(Width::Wide256, u64::MAX, u64::MAX, u64::MAX));
        assert_eq!(plan.cold_cost, u64::MAX);
        assert_eq!(plan.saved_cost, u64::MAX);
        let cells_only = costs().plan(&shape(Width::Wide256, u64::MAX, 0, 0));
        assert_eq!(cells_only.cold_cost, u64::MAX);
    }

    #[test]
    fn a_forced_route_overrides_the_model_and_is_counted() {
        let engaging = shape(Width::Narrow128, 1000, 5, 400);
        let declining = shape(Width::Wide256, 1000, 5, 400);
        let on = RouteForce {
            spill: SpillForce::On,
            ..RouteForce::default()
        };
        let off = RouteForce {
            spill: SpillForce::Off,
            ..RouteForce::default()
        };
        let guard = forced_route(on);
        assert!(costs().plan(&engaging).engage);
        assert_eq!(guard.fired().spill_forced_on, 0);
        let plan = costs().plan(&declining);
        assert!(plan.engage && !plan.natural);
        assert_eq!(guard.fired().spill_forced_on, 1);
        assert_eq!(guard.fired().spill_forced_off, 0);
        drop(guard);
        let guard = forced_route(off);
        assert!(!costs().plan(&declining).engage);
        assert_eq!(guard.fired().spill_forced_off, 0);
        let plan = costs().plan(&engaging);
        assert!(!plan.engage && plan.natural);
        assert_eq!(guard.fired().spill_forced_off, 1);
        assert_eq!(guard.fired().spill_forced_on, 0);
        drop(guard);
        let guard = forced_route(RouteForce::default());
        assert!(costs().plan(&engaging).engage);
        assert!(!costs().plan(&declining).engage);
        assert_eq!(guard.fired(), crate::route::Fired::default());
    }
}
