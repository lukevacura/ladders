//! The gate's rule, and the filter a release run installs: the gate's own choice for the run's
//! shape on every engine, on both sides of each threshold, with nothing of the filter built or
//! allocated where the gate leaves it off.

use super::{Engine, FromM, LEFT_FROM_M, RunShape, TOP_K_FROM_M, TWO_SIDED_FROM_M, gate};
use crate::batch::survivor::Filter;
use crate::batch::{
    ab_join_report, ab_join_topk, left_profile_report, left_topk_profile, two_sided_spill_report,
    two_sided_topk_profile,
};
use crate::entry::Entry;
use crate::quantize::Scale;
use crate::route::{Fired, RouteForce, SurvivorForce, forced_route};
use alloc::format;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

const THREADS: [usize; 3] = [1, 2, 8];

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

/// Each measured engine and its thresholds.
const MEASURED: [(Engine, FromM); 5] = [
    (Engine::Left, LEFT_FROM_M),
    (Engine::TwoSided, TWO_SIDED_FROM_M),
    (Engine::TopK, TOP_K_FROM_M),
    (Engine::Join, TWO_SIDED_FROM_M),
    (Engine::JoinTopK, TOP_K_FROM_M),
];

/// The rule turns PAA on at each engine's threshold and past it, and leaves the filter off below
/// it and on every run it does not gate, at any thread count: a gate mutated always on fails just
/// below each threshold, one mutated always off at it.
#[test]
fn the_rule_turns_paa_on_from_each_engines_threshold() {
    for (engine, from) in MEASURED {
        for threads in THREADS {
            let from = from.at(threads);
            for (m, expected) in [
                (1, Filter::Off),
                (from - 1, Filter::Off),
                (from, Filter::Paa),
                (from + 1, Filter::Paa),
                (4 * from, Filter::Paa),
            ] {
                let shape = RunShape { m, engine, threads };
                assert_eq!(gate(shape), expected, "{shape:?}");
            }
        }
    }
    for m in [1, 1024, 1 << 20] {
        for threads in THREADS {
            let shape = RunShape {
                m,
                engine: Engine::Ungated,
                threads,
            };
            assert_eq!(gate(shape), Filter::Off, "{shape:?}");
        }
    }
    // The measured thresholds, pinned: the last m each engine leaves off and the first it turns
    // PAA on, on one thread and on more.
    for (engine, one_thread, threads) in [
        (Engine::Left, 256, 512),
        (Engine::TwoSided, 512, 768),
        (Engine::TopK, 512, 768),
        (Engine::Join, 512, 768),
        (Engine::JoinTopK, 512, 768),
    ] {
        for (threads, from) in [(1, one_thread), (2, threads), (8, threads)] {
            for (m, expected) in [(from - 1, Filter::Off), (from, Filter::Paa)] {
                let shape = RunShape { m, engine, threads };
                assert_eq!(gate(shape), expected, "{shape:?}");
            }
        }
    }
    assert_eq!(Engine::self_join(false, 1), Engine::Left);
    assert_eq!(Engine::self_join(true, 1), Engine::TwoSided);
    assert_eq!(Engine::self_join(false, 2), Engine::TopK);
    assert_eq!(Engine::self_join(true, 16), Engine::TopK);
    assert_eq!(Engine::join(1), Engine::Join);
    assert_eq!(Engine::join(2), Engine::JoinTopK);
}

/// A best-pair run turns PAA on at every window length and thread count; no engine the gate
/// measured does.
#[test]
fn a_best_pair_run_always_asks() {
    for m in [1, 2, 16, 255, 256, 511, 512, 767, 768, 1 << 16] {
        for threads in THREADS {
            let shape = RunShape {
                m,
                engine: Engine::BestPair,
                threads,
            };
            assert_eq!(gate(shape), Filter::Paa, "{shape:?}");
        }
    }
}

/// A walk of `n` ticks with steps of −3 to +3.
fn walk(n: usize) -> Vec<f64> {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut level = 0i64;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            level += (state >> 33) as i64 % 7 - 3;
            level as f64
        })
        .collect()
}

type Bits = (u64, Option<u64>, u64);

fn bits(row: u64, e: &Entry) -> Bits {
    (row, e.index().get(), e.rho().to_bits())
}

/// A run of one entry point on `threads` threads: its bits.
type Run = fn(&[f64], NonZeroUsize, NonZeroUsize) -> Vec<Bits>;

const K: usize = 4;

fn scale() -> Scale {
    Scale::new(1.0).unwrap()
}

fn left(series: &[f64], m: NonZeroUsize, threads: NonZeroUsize) -> Vec<Bits> {
    let (profile, _) = left_profile_report(series, m, scale(), threads);
    profile.iter().map(|(row, e)| bits(*row, e)).collect()
}

fn two_sided(series: &[f64], m: NonZeroUsize, threads: NonZeroUsize) -> Vec<Bits> {
    let (profile, _, _) = two_sided_spill_report(series, m, scale(), threads);
    profile
        .iter()
        .flat_map(|(row, t)| [t.left, t.right, t.nearest].map(|e| bits(*row, &e)))
        .collect()
}

fn left_topk(series: &[f64], m: NonZeroUsize, threads: NonZeroUsize) -> Vec<Bits> {
    let rows = left_topk_profile(series, m, scale(), threads, nz(K));
    rows.iter()
        .flat_map(|(row, entries)| entries.iter().map(|e| bits(*row, e)))
        .collect()
}

fn two_sided_topk(series: &[f64], m: NonZeroUsize, threads: NonZeroUsize) -> Vec<Bits> {
    let rows = two_sided_topk_profile(series, m, scale(), threads, nz(K)).unwrap();
    rows.iter()
        .flat_map(|(row, t)| {
            [t.left, t.right]
                .into_iter()
                .chain(t.nearest.iter().copied())
                .map(|e| bits(*row, &e))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn ab(series: &[f64], m: NonZeroUsize, threads: NonZeroUsize) -> Vec<Bits> {
    let other: Vec<f64> = series.iter().rev().copied().collect();
    let (profile, _) = ab_join_report(series, &other[..], m, scale(), scale(), threads);
    profile.iter().map(|(row, e)| bits(*row, e)).collect()
}

fn ab_topk(series: &[f64], m: NonZeroUsize, threads: NonZeroUsize) -> Vec<Bits> {
    let other: Vec<f64> = series.iter().rev().copied().collect();
    let rows = ab_join_topk(series, &other[..], m, scale(), scale(), threads, nz(K)).unwrap();
    rows.iter()
        .flat_map(|(row, entries)| entries.iter().map(|e| bits(*row, e)))
        .collect()
}

const ENTRIES: [(&str, Engine, Run); 6] = [
    ("left", Engine::Left, left),
    ("two-sided", Engine::TwoSided, two_sided),
    ("left top-k", Engine::TopK, left_topk),
    ("two-sided top-k", Engine::TopK, two_sided_topk),
    ("ab-join", Engine::Join, ab),
    ("ab-join top-k", Engine::JoinTopK, ab_topk),
];

/// The threshold of `engine` at `threads`, or the left profile's for a run the gate does not gate
/// (any m is on the same side for it).
fn from_m(engine: Engine, threads: usize) -> usize {
    MEASURED
        .iter()
        .find(|(e, _)| *e == engine)
        .map_or(LEFT_FROM_M, |(_, from)| *from)
        .at(threads)
}

/// One gate: every release entry point installs exactly the filter the gate chooses for its
/// shape, just below and at its threshold and far below it, at 1, 2 and 8 threads. Where the gate
/// turns PAA on, the run builds its coefficients and the one-loop route prunes blocks; where it
/// leaves the filter off, the run builds no coefficients and allocates no table of shared floors:
/// the plain pass, at no cost. Both give the plain pass's bits.
#[test]
fn every_release_run_installs_the_gates_choice() {
    for (name, engine, run) in ENTRIES {
        for threads in THREADS {
            let from = from_m(engine, threads);
            for m in [64, from - 1, from] {
                let series = walk(m + 1200);
                let threads =
                    crate::batch::plan::threads_for_size(series.len(), nz(m), nz(threads));
                let expected = gate(RunShape {
                    m,
                    engine,
                    threads: threads.get(),
                });
                let context = format!("{name}, m {m}, {threads} threads");
                // A join decides the gate once for each of its two sides.
                let sides = if name.starts_with("ab-join") { 2 } else { 1 };
                let plain = {
                    let _g = forced_route(RouteForce {
                        survivor: SurvivorForce::Off,
                        ..RouteForce::default()
                    });
                    run(&series, nz(m), threads)
                };
                let g = forced_route(RouteForce::default());
                let gated = run(&series, nz(m), threads);
                let fired: Fired = g.fired();
                drop(g);
                assert_eq!(gated, plain, "{context}");
                match expected {
                    Filter::Off => {
                        // No PAA table, no shared floor table, no gate on.
                        assert_eq!(
                            fired,
                            Fired::default(),
                            "{context}: the filter off costs nothing"
                        );
                    }
                    Filter::Paa => {
                        assert_eq!(fired.gate_on, sides, "{context}: {fired:?}");
                        // The one-loop route runs: its table of shared floors is allocated.
                        assert!(
                            fired.paa_tables > 0 && fired.floor_tables > 0,
                            "{context}: PAA is installed: {fired:?}"
                        );
                    }
                }
            }
        }
    }
}

/// A forced filter overrides the gate whatever it decides, and the gate's decision is still
/// counted: forced off at the threshold builds nothing, forced on below it installs PAA.
#[test]
fn a_forced_filter_overrides_the_gate() {
    let m = LEFT_FROM_M.at(1);
    let series = walk(m + 1200);
    for (survivor, m, on) in [
        (SurvivorForce::Off, m, true),
        (SurvivorForce::Paa, m - 1, false),
    ] {
        let g = forced_route(RouteForce {
            survivor,
            ..RouteForce::default()
        });
        let _ = left(&series, nz(m), nz(1));
        let fired = g.fired();
        assert_eq!(fired.gate_on, usize::from(on), "{survivor:?}: {fired:?}");
        let installed = survivor == SurvivorForce::Paa;
        assert_eq!(
            (fired.paa_tables > 0, fired.floor_tables > 0),
            (installed, installed),
            "{survivor:?}: {fired:?}"
        );
    }
}

/// Where the gate chooses PAA but its coefficients cannot be built (forced to fail here, as when
/// there is no room for the table), the run installs no filter instead: it builds no coefficients
/// and no table of shared floors, and gives the plain pass's bits. A forced PAA filter whose table
/// fails stays installed with nothing to read, and the run is plain.
#[test]
fn a_gated_paa_that_cannot_be_built_falls_back_to_off() {
    for (name, engine, run) in ENTRIES {
        if engine == Engine::Ungated {
            continue;
        }
        for threads in THREADS {
            let m = from_m(engine, threads);
            let series = walk(m + 1200);
            let threads = crate::batch::plan::threads_for_size(series.len(), nz(m), nz(threads));
            assert_eq!(
                gate(RunShape {
                    m,
                    engine,
                    threads: threads.get(),
                }),
                Filter::Paa
            );
            let context = format!("{name}, m {m}, {threads} threads");
            let sides = if name.starts_with("ab-join") { 2 } else { 1 };
            let plain = {
                let _g = forced_route(RouteForce {
                    survivor: SurvivorForce::Off,
                    ..RouteForce::default()
                });
                run(&series, nz(m), threads)
            };
            let g = forced_route(RouteForce {
                paa_build_fails: true,
                ..RouteForce::default()
            });
            let gated = run(&series, nz(m), threads);
            let fired: Fired = g.fired();
            drop(g);
            assert_eq!(gated, plain, "{context}");
            assert_eq!(
                fired,
                Fired {
                    gate_on: sides,
                    gate_fallbacks: sides,
                    ..Fired::default()
                },
                "{context}: the fallback builds nothing"
            );
        }
    }
    let m = LEFT_FROM_M.at(1);
    let series = walk(m + 1200);
    let g = forced_route(RouteForce {
        survivor: SurvivorForce::Paa,
        paa_build_fails: true,
        ..RouteForce::default()
    });
    let _ = left(&series, nz(m), nz(1));
    let fired = g.fired();
    assert_eq!(fired.gate_fallbacks, 0, "{fired:?}");
    assert_eq!(fired.paa_tables, 0, "{fired:?}");
    assert_eq!(fired.floor_tables, 0, "{fired:?}");
}
