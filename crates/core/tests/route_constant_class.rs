#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! The constant class in the batch self-joins: the constant lanes a row does not admit, the lanes a
//! constant row does not offer, the rows a constant column does not admit, and the floors rows and
//! columns start from. On constant-heavy, tie-heavy and mixed series, with gaps at the ends and
//! between constant runs, under the caller's flags and at several `k`, the left, two-sided and
//! top-k profiles equal the exact reference with the class and forced without it, on the narrow,
//! wide, per-tile, re-anchoring and cold passes, at every tiling and thread count; each rule and
//! seed fires where it applies, and forcing the class off drops all of them.
//!
//! Constant blocks: the plain pass and the one-loop route's tiles leave out, before the kernel,
//! each block of rows against a part of columns whose every pair the class settles, and the
//! skipped diagonals catch up when a later block visits them. With constant runs around the
//! block, part, chunk and break-even lengths, gaps beside the runs and at part, tile and band
//! edges, the caller's flags and several `k`, the same profiles equal the exact reference with
//! the block skip and forced without it, on the plain pass and on the route, stepping and
//! resuming, at 1, 2 and 8 threads. The skip fires on ties in every self-join, at a short window
//! length (the plain pass) and at one that takes the route, and is silent when forced off and on a
//! series with no constant window.
//!
//! The AB-join asks the same of the class for the rows of `a` against the windows of `b`: with
//! constant windows in `a`, in `b`, in both and in neither, with gaps and under the callers'
//! flags, at several `k`, thread counts and tilings, on the plain pass and the route, it equals
//! the exact reference with the block skip and forced without it, and the skip fires and leaves
//! pairs out on ties.

#[path = "common/route.rs"]
mod route;

use ladde::batch::{
    ab_join, ab_join_report, ab_join_topk, ab_join_topk_tiled, left_profile_report,
    left_profile_report_tiled, left_topk_profile, left_topk_profile_tiled, two_sided_profile,
    two_sided_profile_tiled, two_sided_topk_profile, two_sided_topk_profile_tiled,
};
use ladde::entry::{Entry, TwoSided, TwoSidedTopk};
use ladde::naive::{
    Ticks, naive_ab_join_topk, naive_left_topk, naive_two_sided, naive_two_sided_topk,
};
use ladde::quantize::Scale;
use ladde::route::{CatchUpForce, Fired, OneLoop, RouteForce, SpillForce, Work, forced_route};
use ladde::stats::Series;
use proptest::prelude::*;
use route::*;

const KS: [usize; 4] = [1, 2, 3, 8];
const TILINGS: [(usize, usize, usize); 4] = [(1, 1, 1), (3, 2, 5), (8, 4, 4), (64, 8, 64)];

/// The `k`, thread counts and tilings a check runs. With `rotate`, each pass past the first two
/// runs a tiling at one of the thread counts, another one from tiling to tiling and pass to pass.
struct Grid {
    ks: &'static [usize],
    threads: &'static [usize],
    tilings: &'static [(usize, usize, usize)],
    rotate: bool,
}

const SMALL: Grid = Grid {
    ks: &KS,
    threads: &[1, 3],
    tilings: &TILINGS,
    rotate: false,
};

/// Bands, chunks and tiles around the gate's blocks of 8 rows and parts of 64 lanes: one row and
/// one lane, chunks narrower than a part, a chunk of one part, chunks and tiles that end inside
/// parts, and chunks of two parts in a tile wider than the series.
const BLOCK_TILINGS: [(usize, usize, usize); 6] = [
    (1, 1, 3),
    (4, 2, 16),
    (8, 64, 64),
    (64, 24, 120),
    (16, 128, 512),
    (3, 5, 70),
];

const BLOCKS: Grid = Grid {
    ks: &KS,
    threads: &[1, 2, 8],
    tilings: &BLOCK_TILINGS,
    rotate: true,
};

fn unit() -> Scale {
    Scale::new(1.0).unwrap()
}

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *seed >> 33
}

/// Random levels held for random runs of 1 to `longest` samples.
fn runs_between_levels(n: usize, seed: u64, levels: u64, longest: u64) -> Vec<f64> {
    let mut seed = seed;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let level = (lcg(&mut seed) % levels) as f64;
        let run = 1 + (lcg(&mut seed) % longest) as usize;
        out.extend(std::iter::repeat_n(level, run));
    }
    out.truncate(n);
    out
}

fn noise(n: usize, seed: u64) -> Vec<f64> {
    let mut seed = seed;
    (0..n).map(|_| (lcg(&mut seed) % 1000) as f64).collect()
}

/// Constant-heavy, tie-heavy and mixed series: constant runs alone, between levels, at the ends,
/// next to and between gaps, spaced about one exclusion zone apart, and none at all.
fn series_set() -> Vec<Vec<f64>> {
    let mut gapped = runs_between_levels(90, 3, 3, 24);
    for i in (0..2).chain(30..32).chain(33..35).chain(88..90) {
        gapped[i] = f64::NAN;
    }
    let mut gaps_between_flats = vec![4.0; 96];
    for i in (20..23).chain(47..48).chain(70..76) {
        gaps_between_flats[i] = f64::NAN;
    }
    let mut one_flat = noise(70, 11);
    for x in &mut one_flat[40..52] {
        *x = 7.0;
    }
    let mut flat_then_noise = vec![2.0; 40];
    flat_then_noise.extend(noise(40, 5));
    let mut noise_then_flat = noise(40, 6);
    noise_then_flat.extend(vec![2.0; 40]);
    // Flat runs a few samples longer than the window, a zone's width of noise between them.
    let mut spaced = Vec::new();
    let mut seed = 13u64;
    while spaced.len() < 110 {
        let level = (lcg(&mut seed) % 5) as f64;
        spaced.extend(std::iter::repeat_n(
            level,
            9 + (lcg(&mut seed) % 4) as usize,
        ));
        spaced.extend(noise(1 + (lcg(&mut seed) % 3) as usize, lcg(&mut seed)));
    }
    let segment = noise(15, 21);
    let repeats: Vec<f64> = (0..6)
        .flat_map(|r| segment.iter().map(move |x| x + 100.0 * r as f64))
        .chain(std::iter::repeat_n(3.0, 12))
        .collect();
    let mut seed = 17u64;
    let few_levels: Vec<f64> = (0..100).map(|_| (lcg(&mut seed) % 3) as f64).collect();
    vec![
        vec![5.0; 60],
        runs_between_levels(120, 1, 2, 24),
        runs_between_levels(120, 2, 4, 24),
        gapped,
        gaps_between_flats,
        one_flat,
        flat_then_noise,
        noise_then_flat,
        spaced,
        repeats,
        few_levels,
        (0..64).map(|i| ((i / 9) % 2) as f64).collect(),
        (0..70).map(|i| (i % 7) as f64).collect(),
    ]
}

fn entries_bits(entries: &[Entry]) -> Vec<Bits> {
    entries.iter().map(|&e| bits(0, e)).collect()
}

fn left_bits(profile: &[(u64, Vec<Entry>)]) -> Vec<(u64, Vec<Bits>)> {
    profile
        .iter()
        .map(|(row, es)| (*row, entries_bits(es)))
        .collect()
}

type SidesBits = (u64, [Bits; 2], Vec<Bits>);

fn sides_bits(profile: &[(u64, TwoSidedTopk)]) -> Vec<SidesBits> {
    profile
        .iter()
        .map(|(row, s)| {
            (
                *row,
                [s.left, s.right].map(|e| bits(0, e)),
                entries_bits(&s.nearest),
            )
        })
        .collect()
}

fn two_sided_bits(profile: &[(u64, TwoSided)]) -> Vec<(u64, [Bits; 3])> {
    profile
        .iter()
        .map(|(row, s)| (*row, [s.left, s.right, s.nearest].map(|e| bits(0, e))))
        .collect()
}

/// The passes, each with the class and forced without it.
fn passes() -> Vec<RouteForce> {
    let none = routed();
    let passes = [
        none,
        RouteForce { wide: true, ..none },
        RouteForce {
            per_tile: true,
            ..none
        },
        RouteForce {
            reanchor_cell: 3,
            ..none
        },
        RouteForce {
            constant_rows_by_lane: true,
            ..none
        },
    ];
    passes
        .into_iter()
        .flat_map(|force| {
            [
                force,
                RouteForce {
                    no_constant_class: true,
                    ..force
                },
            ]
        })
        .collect()
}

/// What every run of one `k` must give: the left and two-sided top-`k` profiles, and at `k` of
/// one the two-sided profile.
struct Expected {
    left: Vec<(u64, Vec<Bits>)>,
    sides: Vec<SidesBits>,
    one: Vec<(u64, [Bits; 3])>,
}

/// The naive reference under the caller's `flags`.
fn naive(series: &[f64], m: usize, flags: Option<&[bool]>, k: usize) -> Expected {
    let ticks = samples(series, unit());
    let reference = match flags {
        Some(flags) => Ticks::with_constant_windows(&ticks, flags),
        None => Ticks::from(&ticks),
    };
    Expected {
        left: left_bits(&naive_left_topk(reference, m, nz(k))),
        sides: sides_bits(&naive_two_sided_topk(reference, m, nz(k))),
        one: two_sided_bits(&naive_two_sided(reference, m)),
    }
}

/// The k = 1 left profile as top-k lists: a row with no neighbor keeps none.
fn as_lists(profile: Vec<(u64, Entry)>) -> Vec<(u64, Vec<Entry>)> {
    profile
        .into_iter()
        .map(|(row, e)| (row, e.index().get().map(|_| e).into_iter().collect()))
        .collect()
}

/// The batch self-joins under the caller's `flags` (none for the default rule) equal `expected`
/// at every `k`, pass, tiling and thread count of `grid`.
fn check_batch(
    series: &[f64],
    m: usize,
    flags: Option<&[bool]>,
    (forces, grid): (&[RouteForce], &Grid),
    expected: impl Fn(usize) -> Expected,
) {
    let input = || match flags {
        Some(flags) => Series::with_constant_windows(series, nz(m), flags).unwrap(),
        None => Series::from(series),
    };
    for (at_k, &k) in grid.ks.iter().enumerate() {
        let Expected { left, sides, one } = expected(k);
        for (at_force, &force) in forces.iter().enumerate() {
            let _forced = forced_route(force);
            let context = (force, k, m);
            for &threads in grid.threads {
                let got = left_topk_profile(input(), nz(m), unit(), nz(threads), nz(k));
                assert_eq!(left_bits(&got), left, "left {context:?} t{threads}");
                let got = two_sided_topk_profile(input(), nz(m), unit(), nz(threads), nz(k));
                assert_eq!(sides_bits(&got.unwrap()), sides, "two-sided {context:?}");
                if k == 1 {
                    let (got, _) = left_profile_report(input(), nz(m), unit(), nz(threads));
                    assert_eq!(left_bits(&as_lists(got)), left, "left k1 {context:?}");
                    let got = two_sided_profile(input(), nz(m), unit(), nz(threads));
                    assert_eq!(two_sided_bits(&got), one, "two-sided k1 {context:?}");
                }
            }
            for (at_tiling, &(band, chunk, tile)) in grid.tilings.iter().enumerate() {
                let turn = (at_k + at_force + at_tiling) % grid.threads.len();
                let threads = if grid.rotate && at_force >= 2 {
                    &grid.threads[turn..=turn]
                } else {
                    grid.threads
                };
                for &threads in threads {
                    let shape = (nz(threads), nz(band), nz(chunk), nz(tile));
                    let tiled = (context, threads, (band, chunk, tile));
                    let (got, _) = left_topk_profile_tiled(
                        input(),
                        nz(m),
                        unit(),
                        shape.0,
                        shape.1,
                        shape.2,
                        shape.3,
                        nz(k),
                    );
                    assert_eq!(left_bits(&got), left, "left {tiled:?}");
                    let (got, _) = two_sided_topk_profile_tiled(
                        input(),
                        nz(m),
                        unit(),
                        shape.0,
                        shape.1,
                        shape.2,
                        shape.3,
                        nz(k),
                    )
                    .unwrap();
                    assert_eq!(sides_bits(&got), sides, "two-sided {tiled:?}");
                    if k == 1 {
                        let (got, _) = left_profile_report_tiled(
                            input(),
                            nz(m),
                            unit(),
                            shape.0,
                            shape.1,
                            shape.2,
                            shape.3,
                        );
                        assert_eq!(left_bits(&as_lists(got)), left, "left k1 {tiled:?}");
                        let (got, _) = two_sided_profile_tiled(
                            input(),
                            nz(m),
                            unit(),
                            shape.0,
                            shape.1,
                            shape.2,
                            shape.3,
                        );
                        assert_eq!(two_sided_bits(&got), one, "two-sided k1 {tiled:?}");
                    }
                }
            }
        }
    }
}

#[test]
fn batch_self_joins_equal_the_reference_with_and_without_the_class() {
    let forces = passes();
    for series in series_set() {
        for m in [3, 4, 8] {
            check_batch(&series, m, None, (&forces, &SMALL), |k| {
                naive(&series, m, None, k)
            });
        }
    }
}

/// Flags that make windows of equal ticks valid and windows of unequal ticks constant, at random
/// and for none.
fn flag_sets(series: &[f64], m: usize) -> Vec<Vec<bool>> {
    let flat = |w: &[f64]| w.iter().all(|x| *x == w[0]);
    let mut seed = series.len() as u64 * 31 + m as u64;
    vec![
        vec![false; series.len() + 1 - m],
        series.windows(m).map(|w| !flat(w)).collect(),
        series
            .windows(m)
            .map(|_| lcg(&mut seed).is_multiple_of(3))
            .collect(),
        series
            .windows(m)
            .map(|w| flat(w) || lcg(&mut seed).is_multiple_of(5))
            .collect(),
    ]
}

#[test]
fn flagged_self_joins_equal_the_reference_with_and_without_the_class() {
    let none = routed();
    let forces = [
        none,
        RouteForce {
            no_constant_class: true,
            ..none
        },
        RouteForce { wide: true, ..none },
    ];
    for series in series_set() {
        let m = 4;
        for flags in flag_sets(&series, m) {
            let expected = |k| naive(&series, m, Some(&flags), k);
            check_batch(&series, m, Some(&flags), (&forces, &SMALL), expected);
        }
    }
}

/// `series` holds an outlier excursion that only the cold route ranks, so the hot pass runs on
/// stand-in ticks with the excursion's windows non-finite: under each of `forces`, every run
/// equals the run without the class, whose left profile equals the exact reference that ranks
/// every window.
fn check_cold(series: &[f64], m: usize, forces: &[RouteForce], grid: &Grid) {
    let without = RouteForce {
        spill: SpillForce::On,
        no_constant_class: true,
        ..routed()
    };
    let expected = |k: usize| {
        let _forced = forced_route(without);
        let left = left_topk_profile(series, nz(m), unit(), nz(1), nz(k));
        let sides = two_sided_topk_profile(series, nz(m), unit(), nz(1), nz(k)).unwrap();
        let one = two_sided_profile(series, nz(m), unit(), nz(1));
        if k == 1 {
            let (got, _) = left_profile_report(series, nz(m), unit(), nz(1));
            let exact = stream_profile_bits(&samples(series, unit()), m);
            assert_eq!(profile_bits(&got), exact, "the run without the class");
        }
        Expected {
            left: left_bits(&left),
            sides: sides_bits(&sides),
            one: two_sided_bits(&one),
        }
    };
    check_batch(series, m, None, (forces, grid), expected);
}

#[test]
fn the_cold_route_equals_the_run_without_the_class() {
    let mut series = runs_between_levels(140, 7, 3, 20);
    for x in &mut series[60..63] {
        *x += 4.0e11;
    }
    let none = routed();
    let cold = RouteForce {
        spill: SpillForce::On,
        ..none
    };
    let forces = [
        cold,
        RouteForce {
            no_constant_class: true,
            ..cold
        },
        RouteForce { wide: true, ..cold },
    ];
    check_cold(&series, 4, &forces, &SMALL);
}

fn tie_heavy() -> impl Strategy<Value = (Vec<f64>, usize, usize)> {
    (
        3usize..9,
        1usize..5,
        40usize..140,
        any::<u64>(),
        1u64..5,
        1u64..30,
    )
        .prop_map(|(m, k, n, seed, levels, longest)| {
            let mut series = runs_between_levels(n, seed, levels, longest);
            let mut seed = seed;
            for x in &mut series {
                match lcg(&mut seed) % 40 {
                    0 => *x = f64::NAN,
                    1 => *x += (lcg(&mut seed) % 100) as f64,
                    _ => {}
                }
            }
            (series, m, k)
        })
}

/// Runs between a few levels long enough to fill blocks and parts, with a gap or an off-level
/// sample now and then.
fn long_ties() -> impl Strategy<Value = (Vec<f64>, usize, usize)> {
    (
        3usize..10,
        1usize..5,
        150usize..420,
        any::<u64>(),
        1u64..4,
        prop::sample::select(vec![6u64, 30, 90]),
    )
        .prop_map(|(m, k, n, seed, levels, longest)| {
            let mut series = runs_between_levels(n, seed, levels, longest);
            let mut seed = seed;
            for x in &mut series {
                match lcg(&mut seed) % 60 {
                    0 => *x = f64::NAN,
                    1 => *x += (lcg(&mut seed) % 100) as f64,
                    _ => {}
                }
            }
            (series, m, k)
        })
}

proptest! {
    #![proptest_config(testkit::generators::seeded(48))]

    #[test]
    fn tie_heavy_self_joins_equal_the_reference((series, m, k) in tie_heavy()) {
        let ticks = samples(&series, unit());
        let reference = Ticks::from(&ticks);
        let left = left_bits(&naive_left_topk(reference, m, nz(k)));
        let sides = sides_bits(&naive_two_sided_topk(reference, m, nz(k)));
        for no_constant_class in [false, true] {
            let _forced = forced_route(RouteForce { no_constant_class, ..routed() });
            for threads in [1, 2] {
                let got = left_topk_profile(&series[..], nz(m), unit(), nz(threads), nz(k));
                prop_assert_eq!(left_bits(&got), left.clone());
                let (got, _) = two_sided_topk_profile_tiled(
                    &series[..], nz(m), unit(), nz(threads), nz(4), nz(2), nz(6), nz(k),
                ).unwrap();
                prop_assert_eq!(sides_bits(&got), sides.clone());
            }
        }
    }

    #[test]
    fn long_ties_equal_the_reference_on_every_tiling(
        (series, m, k) in long_ties(),
        (band, chunk, tile) in prop::sample::select(BLOCK_TILINGS.to_vec()),
        threads in prop::sample::select(vec![1usize, 2, 8]),
        catch_up in prop::sample::select(vec![
            CatchUpForce::Rule,
            CatchUpForce::Step,
            CatchUpForce::Resume,
        ]),
        wide in any::<bool>(),
    ) {
        let ticks = samples(&series, unit());
        let reference = Ticks::from(&ticks);
        let left = left_bits(&naive_left_topk(reference, m, nz(k)));
        let sides = sides_bits(&naive_two_sided_topk(reference, m, nz(k)));
        let one = two_sided_bits(&naive_two_sided(reference, m));
        let shape = (nz(threads), nz(band), nz(chunk), nz(tile));
        let _forced = forced_route(RouteForce {
            catch_up,
            wide,
            ..routed()
        });
        let got = left_topk_profile(&series[..], nz(m), unit(), shape.0, nz(k));
        prop_assert_eq!(left_bits(&got), left.clone());
        let got = two_sided_topk_profile(&series[..], nz(m), unit(), shape.0, nz(k)).unwrap();
        prop_assert_eq!(sides_bits(&got), sides.clone());
        let (got, _) = left_topk_profile_tiled(
            &series[..], nz(m), unit(), shape.0, shape.1, shape.2, shape.3, nz(k),
        );
        prop_assert_eq!(left_bits(&got), left.clone());
        let (got, _) = two_sided_topk_profile_tiled(
            &series[..], nz(m), unit(), shape.0, shape.1, shape.2, shape.3, nz(k),
        ).unwrap();
        prop_assert_eq!(sides_bits(&got), sides.clone());
        if k == 1 {
            let (got, _) = left_profile_report_tiled(
                &series[..], nz(m), unit(), shape.0, shape.1, shape.2, shape.3,
            );
            prop_assert_eq!(left_bits(&as_lists(got)), left.clone());
            let (got, _) = two_sided_profile_tiled(
                &series[..], nz(m), unit(), shape.0, shape.1, shape.2, shape.3,
            );
            prop_assert_eq!(two_sided_bits(&got), one.clone());
        }
    }
}

/// The class's own counters of `fired`, every other one zero.
fn class_part(fired: Fired) -> Fired {
    Fired {
        constant_classes_dropped: fired.constant_classes_dropped,
        class_row_lanes: fired.class_row_lanes,
        class_offer_lanes: fired.class_offer_lanes,
        class_column_pairs: fired.class_column_pairs,
        class_row_seeds: fired.class_row_seeds,
        class_column_seeds: fired.class_column_seeds,
        ..Fired::default()
    }
}

/// What the class's rules and seeds did in a left and a two-sided top-`k` run of `series`.
fn class_fired(series: &[f64], m: usize, k: usize, force: RouteForce) -> (Fired, Fired) {
    let run = |two_sided: bool| {
        let guard = forced_route(force);
        let shape = (nz(2), nz(8), nz(4), nz(16));
        if two_sided {
            let _ = two_sided_topk_profile_tiled(
                series,
                nz(m),
                unit(),
                shape.0,
                shape.1,
                shape.2,
                shape.3,
                nz(k),
            );
        } else {
            let _ = left_topk_profile(series, nz(m), unit(), nz(1), nz(k));
        }
        guard.fired()
    };
    (run(false), run(true))
}

#[test]
fn every_rule_and_seed_fires_and_forcing_the_class_off_drops_them() {
    // Constant runs between levels, then noise: the last constant columns have no constant past
    // their zone, so valid rows reach them.
    let mut series = runs_between_levels(160, 9, 3, 20);
    series.extend(noise(60, 4));
    let (m, k) = (4, 2);
    for wide in [false, true] {
        let force = RouteForce { wide, ..routed() };
        let (left, two) = class_fired(&series, m, k, force);
        assert!(left.class_row_lanes > 0, "row lanes, wide {wide}");
        assert!(left.class_row_seeds > 0, "row seeds, wide {wide}");
        assert_eq!(left.class_column_seeds, 0, "a left run has no columns");
        assert!(two.class_row_lanes > 0, "two-sided row lanes, wide {wide}");
        assert!(two.class_offer_lanes > 0, "offer lanes, wide {wide}");
        assert!(two.class_column_pairs > 0, "column pairs, wide {wide}");
        assert!(two.class_row_seeds > 0, "two-sided row seeds, wide {wide}");
        assert!(two.class_column_seeds > 0, "column seeds, wide {wide}");
        assert_eq!(
            left.constant_classes_dropped + two.constant_classes_dropped,
            0
        );

        let off = RouteForce {
            no_constant_class: true,
            ..force
        };
        for fired in [
            class_fired(&series, m, k, off).0,
            class_fired(&series, m, k, off).1,
        ] {
            assert!(fired.constant_classes_dropped > 0, "dropped, wide {wide}");
            let rules = Fired {
                constant_classes_dropped: 0,
                ..class_part(fired)
            };
            assert_eq!(rules, Fired::default(), "wide {wide}");
        }
    }
}

#[test]
fn a_series_without_constant_windows_builds_no_class() {
    let series = noise(200, 8);
    let off = RouteForce {
        no_constant_class: true,
        ..routed()
    };
    for force in [routed(), off] {
        let (left, two) = class_fired(&series, 4, 2, force);
        assert_eq!(class_part(left), Fired::default(), "{force:?}");
        assert_eq!(class_part(two), Fired::default(), "{force:?}");
    }
}

/// Constant runs that make `windows` constant windows each at window length `m`, three samples
/// off every level between them.
fn sized_runs(m: usize, windows: &[usize]) -> Vec<f64> {
    let mut seed = 29u64;
    let mut between = || noise(3, lcg(&mut seed)).into_iter().map(|x| x + 10.0);
    let mut out: Vec<f64> = between().collect();
    for (at, &count) in windows.iter().enumerate() {
        out.extend(std::iter::repeat_n((at % 3) as f64, count + m - 1));
        out.extend(between());
    }
    out
}

/// Three levels in runs of a few samples, with a long run now and then.
fn ties(n: usize, seed: u64) -> Vec<f64> {
    let mut seed = seed;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let level = (lcg(&mut seed) % 3) as f64;
        let run = if lcg(&mut seed).is_multiple_of(8) {
            20 + lcg(&mut seed) % 100
        } else {
            1 + lcg(&mut seed) % 3
        };
        out.extend(std::iter::repeat_n(level, run as usize));
    }
    out.truncate(n);
    out
}

/// The passes and catch-ups a skipped block meets, each with the block skip and forced without
/// it: the plain pass (short windows run it) and the one-loop route (forced on at a short
/// window), narrow, wide and per-tile, stepping and resuming.
fn block_passes() -> Vec<RouteForce> {
    let none = routed();
    let on = RouteForce {
        one_loop: OneLoop::On,
        ..none
    };
    let passes = [
        none,
        RouteForce { wide: true, ..none },
        RouteForce {
            per_tile: true,
            ..none
        },
        RouteForce {
            reanchor_cell: 3,
            ..none
        },
        RouteForce {
            constant_rows_by_lane: true,
            ..none
        },
        RouteForce {
            catch_up: CatchUpForce::Step,
            ..none
        },
        RouteForce {
            catch_up: CatchUpForce::Resume,
            wide: true,
            ..none
        },
        RouteForce {
            resume_every: 2,
            ..none
        },
        RouteForce {
            one_loop: OneLoop::Off,
            ..none
        },
        on,
        RouteForce { wide: true, ..on },
        RouteForce {
            per_tile: true,
            ..on
        },
        RouteForce {
            catch_up: CatchUpForce::Resume,
            ..on
        },
    ];
    passes
        .into_iter()
        .flat_map(|force| {
            [
                force,
                RouteForce {
                    no_constant_blocks: true,
                    ..force
                },
            ]
        })
        .collect()
}

fn check_blocks(series: &[f64], m: usize) {
    check_batch(series, m, None, (&block_passes(), &BLOCKS), |k| {
        naive(series, m, None, k)
    });
}

/// Constant windows alone, and constant runs one window under, at and over the block of rows,
/// the part and chunk of lanes, and the lengths from which a skipped run resumes on either
/// target's plan.
#[test]
fn constant_blocks_keep_every_bit_on_runs_of_every_size() {
    let m = 8;
    check_blocks(&[5.0; 300], m);
    check_blocks(&sized_runs(m, &[7, 8, 9, 19, 20, 21, 23, 24, 25]), m);
    check_blocks(&sized_runs(m, &[63, 64, 65, 24, 8]), m);
}

/// Long runs between two levels, shorter runs with stretches of noise between them, and ties.
#[test]
fn constant_blocks_keep_every_bit_on_constant_heavy_tie_heavy_and_mixed_series() {
    check_blocks(&runs_between_levels(380, 1, 2, 90), 8);
    let mut mixed = Vec::new();
    for piece in 0..4 {
        mixed.extend(runs_between_levels(70, 40 + piece, 3, 40));
        mixed.extend(noise(25, 50 + piece));
    }
    check_blocks(&mixed, 8);
    check_blocks(&ties(400, 3), 8);
    check_blocks(&ties(300, 4), 3);
}

/// Gaps at the ends, inside and beside constant runs, and at the edges of parts, chunks, tiles
/// and bands; a series of constant and non-finite windows only; and first constant and first
/// finite windows that come late.
#[test]
fn constant_blocks_keep_every_bit_beside_gaps_and_late_first_windows() {
    let mut gapped = runs_between_levels(380, 3, 3, 60);
    for i in (0..2)
        .chain(62..66)
        .chain(119..121)
        .chain(127..129)
        .chain(191..193)
        .chain(200..203)
        .chain(255..257)
        .chain(378..380)
    {
        gapped[i] = f64::NAN;
    }
    check_blocks(&gapped, 8);
    let mut flat = vec![4.0; 330];
    for i in (20..23)
        .chain(64..65)
        .chain(128..130)
        .chain(200..212)
        .chain(300..301)
    {
        flat[i] = f64::NAN;
    }
    check_blocks(&flat, 8);
    let mut late = vec![f64::NAN; 12];
    late.extend(noise(110, 6));
    late.extend(vec![2.0; 110]);
    late.extend(noise(30, 7));
    late.extend(vec![1.0; 90]);
    check_blocks(&late, 8);
}

/// Under the caller's flags the class is the flagged windows, whatever their ticks.
#[test]
fn constant_blocks_keep_every_bit_under_the_callers_flags() {
    let none = routed();
    let forces = [
        none,
        RouteForce {
            no_constant_blocks: true,
            ..none
        },
        RouteForce { wide: true, ..none },
        RouteForce {
            catch_up: CatchUpForce::Resume,
            ..none
        },
    ];
    let m = 8;
    let mut gapped = runs_between_levels(300, 5, 3, 50);
    for i in (0..1).chain(63..65).chain(150..153) {
        gapped[i] = f64::NAN;
    }
    for series in [ties(300, 9), gapped] {
        for flags in flag_sets(&series, m) {
            let expected = |k| naive(&series, m, Some(&flags), k);
            check_batch(&series, m, Some(&flags), (&forces, &BLOCKS), expected);
        }
    }
}

#[test]
fn constant_blocks_keep_every_bit_on_the_cold_route() {
    let mut series = runs_between_levels(380, 7, 3, 60);
    for x in &mut series[200..203] {
        *x += 4.0e11;
    }
    let cold = RouteForce {
        spill: SpillForce::On,
        ..routed()
    };
    let forces = [
        cold,
        RouteForce {
            no_constant_blocks: true,
            ..cold
        },
        RouteForce { wide: true, ..cold },
        RouteForce {
            catch_up: CatchUpForce::Step,
            ..cold
        },
    ];
    check_cold(&series, 8, &forces, &BLOCKS);
}

/// Every batch self-join of `series` under `force`, by name: what fired and what the passes
/// computed.
fn self_joins(
    series: &[f64],
    (m, k, threads): (usize, usize, usize),
    force: RouteForce,
) -> Vec<(&'static str, Fired, Work)> {
    let (m, k, threads) = (nz(m), nz(k), nz(threads));
    // A band of three blocks of rows: a tile's states start over at each band, so only a later
    // block of the same band catches a skipped diagonal up.
    let (band, chunk, tile) = (nz(24), nz(64), nz(192));
    let run = |name, join: &dyn Fn()| {
        let guard = forced_route(force);
        join();
        (name, guard.fired(), guard.work())
    };
    vec![
        run("left", &|| {
            let _ = left_profile_report(series, m, unit(), threads);
        }),
        run("left tiled", &|| {
            let _ = left_profile_report_tiled(series, m, unit(), threads, band, chunk, tile);
        }),
        run("left top-k", &|| {
            let _ = left_topk_profile(series, m, unit(), threads, k);
        }),
        run("left top-k tiled", &|| {
            let _ = left_topk_profile_tiled(series, m, unit(), threads, band, chunk, tile, k);
        }),
        run("two-sided", &|| {
            let _ = two_sided_profile(series, m, unit(), threads);
        }),
        run("two-sided tiled", &|| {
            let _ = two_sided_profile_tiled(series, m, unit(), threads, band, chunk, tile);
        }),
        run("two-sided top-k", &|| {
            let _ = two_sided_topk_profile(series, m, unit(), threads, k);
        }),
        run("two-sided top-k tiled", &|| {
            let _ = two_sided_topk_profile_tiled(series, m, unit(), threads, band, chunk, tile, k)
                .unwrap();
        }),
    ]
}

/// On ties every self-join leaves blocks out and catches their diagonals up, at `k` of one and
/// above, on the plain pass (a window length with no filter) and on the one-loop route (forced on
/// at that length), narrow and wide, and computes fewer pairs for it than forced without the
/// block skip. Forced off, no block is left out while the class's other rules stand.
#[test]
fn the_block_skip_fires_on_ties_in_every_self_join_and_is_silent_forced_off() {
    let (series, m) = (ties(500, 3), 8);
    for one_loop in [OneLoop::Auto, OneLoop::On] {
        for wide in [false, true] {
            for threads in [1, 2, 8] {
                for k in [1, 3] {
                    let force = RouteForce {
                        one_loop,
                        wide,
                        ..routed()
                    };
                    let off = RouteForce {
                        no_constant_blocks: true,
                        ..force
                    };
                    let context = (one_loop, wide, threads, k);
                    let on = self_joins(&series, (m, k, threads), force);
                    let without = self_joins(&series, (m, k, threads), off);
                    for ((name, fired, work), (_, silent, plain)) in on.iter().zip(&without) {
                        assert!(
                            fired.class_blocks > 0 && fired.class_block_pairs >= fired.class_blocks,
                            "{name} {context:?}: {fired:?}"
                        );
                        assert!(
                            work.pairs_computed < plain.pairs_computed,
                            "{name} {context:?}: {work:?} against {plain:?}"
                        );
                        // Where the plan picks the band it may be one block tall, and a block left
                        // out then ends its band: nothing is left to catch up.
                        if name.ends_with("tiled") {
                            assert!(
                                work.lanes_resumed + work.lanes_stepped > 0,
                                "{name} {context:?}: {work:?}"
                            );
                        }
                        assert_eq!(
                            (silent.class_blocks, silent.class_block_pairs),
                            (0, 0),
                            "{name} {context:?}"
                        );
                        assert!(silent.class_row_seeds > 0, "{name} {context:?}: {silent:?}");
                    }
                }
            }
        }
    }
}

/// Constant runs at a window length that takes the one-loop route by default (from 768 up):
/// the block skip fires in every self-join, the route runs (its survivors are computed), pairs
/// are left out against the run without the skip, and every profile equals the run without the
/// skip and the plain pass, at 1, 2 and 8 threads.
#[test]
fn the_block_skip_fires_at_a_window_length_that_takes_the_loop() {
    let m = 1024;
    let series = sized_runs(m, &[150, 600, 300, 80]);
    let plain = RouteForce {
        one_loop: OneLoop::Off,
        ..routed()
    };
    let off = RouteForce {
        no_constant_blocks: true,
        ..routed()
    };
    for threads in [1, 2, 8] {
        for k in [1, 2] {
            let on = self_joins(&series, (m, k, threads), routed());
            let without = self_joins(&series, (m, k, threads), off);
            for ((name, fired, work), (_, _, base)) in on.iter().zip(&without) {
                let context = (threads, k);
                assert!(
                    fired.class_blocks > 0 && fired.class_block_pairs >= fired.class_blocks,
                    "{name} {context:?}: {fired:?}"
                );
                assert!(
                    work.survivor_runs > 0,
                    "{name} {context:?}: the loop did not run"
                );
                assert!(
                    work.pairs_computed < base.pairs_computed,
                    "{name} {context:?}: {work:?} against {base:?}"
                );
            }
            let bits = |force: RouteForce| {
                let _forced = forced_route(force);
                let left = left_topk_profile(&series[..], nz(m), unit(), nz(threads), nz(k));
                let two = two_sided_topk_profile(&series[..], nz(m), unit(), nz(threads), nz(k));
                (left_bits(&left), sides_bits(&two.unwrap()))
            };
            let reference = bits(plain);
            assert_eq!(bits(routed()), reference, "threads {threads} k {k}");
            assert_eq!(bits(off), reference, "no skip, threads {threads} k {k}");
        }
    }
}

/// A series the class cannot help runs as it did: with no constant window, with gaps or without,
/// no self-join leaves a block out.
#[test]
fn the_block_skip_is_silent_where_there_is_no_class() {
    let m = 8;
    let mut gapped = noise(500, 9);
    for i in (0..3).chain(100..104).chain(250..251) {
        gapped[i] = f64::NAN;
    }
    for series in [noise(500, 8), gapped] {
        for one_loop in [OneLoop::Auto, OneLoop::On] {
            for threads in [1, 2, 8] {
                let force = RouteForce {
                    one_loop,
                    ..routed()
                };
                for (name, fired, _) in self_joins(&series, (m, 3, threads), force) {
                    assert_eq!(class_part(fired), Fired::default(), "{name}");
                }
            }
        }
    }
}

/// One side of a join as the reference reads it, under the caller's `flags` if any.
fn join_ticks<'a>(ticks: &'a [Option<i64>], flags: Option<&'a [bool]>) -> Ticks<'a> {
    match flags {
        Some(flags) => Ticks::with_constant_windows(ticks, flags),
        None => Ticks::from(ticks),
    }
}

/// One side of a join as the batch engine reads it, under the caller's `flags` if any.
fn join_series<'a>(series: &'a [f64], flags: Option<&'a [bool]>, m: usize) -> Series<'a> {
    match flags {
        Some(flags) => Series::with_constant_windows(series, nz(m), flags).unwrap(),
        None => Series::from(series),
    }
}

/// The AB-join of `a` against `b` (each with the caller's `flags`, if any) at `m`, as `k` lists
/// per row: the exact reference.
fn join_expected(
    (a, b): (&[f64], &[f64]),
    (flags_a, flags_b): (Option<&[bool]>, Option<&[bool]>),
    m: usize,
    k: usize,
) -> Vec<(u64, Vec<Bits>)> {
    let (ticks_a, ticks_b) = (samples(a, unit()), samples(b, unit()));
    let profile = naive_ab_join_topk(
        join_ticks(&ticks_a, flags_a),
        join_ticks(&ticks_b, flags_b),
        m,
        nz(k),
    );
    left_bits(&profile)
}

/// The AB-join of `a` against `b` equals the exact reference at every `k`, pass, tiling and
/// thread count of `grid`, with the block skip and forced without it.
fn check_join(
    (a, b): (&[f64], &[f64]),
    flags: (Option<&[bool]>, Option<&[bool]>),
    m: usize,
    (forces, grid): (&[RouteForce], &Grid),
) {
    let sides = || (join_series(a, flags.0, m), join_series(b, flags.1, m));
    for (at_k, &k) in grid.ks.iter().enumerate() {
        let expected = join_expected((a, b), flags, m, k);
        for (at_force, &force) in forces.iter().enumerate() {
            let _forced = forced_route(force);
            let context = (force, k, m);
            for &threads in grid.threads {
                let (sa, sb) = sides();
                let got = ab_join_topk(sa, sb, nz(m), unit(), unit(), nz(threads), nz(k)).unwrap();
                assert_eq!(left_bits(&got), expected, "join {context:?} t{threads}");
                if k == 1 {
                    let (sa, sb) = sides();
                    let got = ab_join(sa, sb, nz(m), unit(), unit(), nz(threads));
                    assert_eq!(left_bits(&as_lists(got)), expected, "join k1 {context:?}");
                }
            }
            for (at_tiling, &(band, chunk, tile)) in grid.tilings.iter().enumerate() {
                let turn = (at_k + at_force + at_tiling) % grid.threads.len();
                let threads = if grid.rotate && at_force >= 2 {
                    &grid.threads[turn..=turn]
                } else {
                    grid.threads
                };
                for &threads in threads {
                    let (sa, sb) = sides();
                    let (got, _) = ab_join_topk_tiled(
                        sa,
                        sb,
                        nz(m),
                        unit(),
                        unit(),
                        nz(threads),
                        nz(band),
                        nz(chunk),
                        nz(tile),
                        nz(k),
                    )
                    .unwrap();
                    let tiled = (context, threads, (band, chunk, tile));
                    assert_eq!(left_bits(&got), expected, "join {tiled:?}");
                }
            }
        }
    }
}

/// Constant windows in `a` only, in `b` only, in both and in neither, of different lengths, with
/// gaps, and ties.
fn join_pairs() -> Vec<(Vec<f64>, Vec<f64>)> {
    let mut gapped = runs_between_levels(300, 5, 3, 50);
    for i in (0..1).chain(63..65).chain(150..153) {
        gapped[i] = f64::NAN;
    }
    let mut flat_gapped = vec![4.0; 120];
    for i in (20..23).chain(64..65).chain(100..103) {
        flat_gapped[i] = f64::NAN;
    }
    let flat_then_noise: Vec<f64> = vec![2.0; 90].into_iter().chain(noise(60, 5)).collect();
    vec![
        (ties(300, 3), ties(260, 7)),
        (ties(200, 4), ties(330, 8)),
        (noise(200, 1), sized_runs(8, &[40, 90, 10])),
        (sized_runs(8, &[70, 9, 25]), noise(220, 2)),
        (noise(200, 3), noise(240, 4)),
        (gapped.clone(), ties(280, 6)),
        (flat_gapped, gapped),
        (flat_then_noise.clone(), flat_then_noise),
        (vec![5.0; 150], vec![5.0; 110]),
        (ties(120, 1)[..120].to_vec(), noise(40, 9)),
    ]
}

/// The AB-join under the block skip equals the exact reference on every pass the self-joins'
/// blocks meet: the plain pass and the one-loop route (forced on at a short window), narrow,
/// wide and per-tile, stepping and resuming.
#[test]
fn join_blocks_keep_every_bit_with_constant_windows_on_either_side() {
    let forces = block_passes();
    for (a, b) in join_pairs() {
        check_join((&a, &b), (None, None), 8, (&forces, &BLOCKS));
    }
}

/// Windows too few to fill a block, and a window length close to the series'.
#[test]
fn join_blocks_keep_every_bit_on_short_sides() {
    let forces = block_passes();
    for m in [3, 8, 20] {
        let a: Vec<f64> = sized_runs(m, &[2, 3]);
        let b: Vec<f64> = ties(m + 4, 2);
        check_join((&a, &b), (None, None), m, (&forces, &SMALL));
        check_join((&b, &a), (None, None), m, (&forces, &SMALL));
    }
}

/// Under the callers' flags the class is the flagged windows, whatever their ticks.
#[test]
fn join_blocks_keep_every_bit_under_the_callers_flags() {
    let none = routed();
    let forces = [
        none,
        RouteForce {
            no_constant_blocks: true,
            ..none
        },
        RouteForce { wide: true, ..none },
        RouteForce {
            one_loop: OneLoop::On,
            ..none
        },
    ];
    let m = 8;
    let (a, b) = (ties(220, 12), runs_between_levels(250, 5, 3, 50));
    for flags_a in flag_sets(&a, m) {
        for flags_b in flag_sets(&b, m) {
            let flags = (Some(&flags_a[..]), Some(&flags_b[..]));
            check_join((&a, &b), flags, m, (&forces, &SMALL));
        }
    }
}

/// On ties the AB-join leaves blocks out, at `k` of one and above, on the plain pass and the
/// one-loop route, narrow and wide, per tile, and computes fewer pairs for it than forced without
/// the block skip; forced off, no block is left out.
#[test]
fn the_join_block_skip_fires_on_ties_and_is_silent_forced_off() {
    let (m, a, b) = (8, ties(500, 3), ties(460, 5));
    for one_loop in [OneLoop::Off, OneLoop::On] {
        for wide in [false, true] {
            for threads in [1, 2, 8] {
                for k in [1, 3] {
                    let force = RouteForce {
                        one_loop,
                        wide,
                        ..routed()
                    };
                    let off = RouteForce {
                        no_constant_blocks: true,
                        ..force
                    };
                    let run = |force: RouteForce| {
                        let guard = forced_route(force);
                        let _ =
                            ab_join_topk(&a[..], &b[..], nz(m), unit(), unit(), nz(threads), nz(k))
                                .unwrap();
                        (guard.fired(), guard.work())
                    };
                    let (fired, work) = run(force);
                    let (silent, plain) = run(off);
                    let context = (one_loop, wide, threads, k);
                    assert!(
                        fired.class_blocks > 0 && fired.class_block_pairs >= fired.class_blocks,
                        "{context:?}: {fired:?}"
                    );
                    // The plain pass counts the pairs it computes (a run without the skip may
                    // take the one-pass engine, which does not); the route counts its areas.
                    let pairs = (a.len() + 1 - m) * (b.len() + 1 - m);
                    match one_loop {
                        OneLoop::On => {
                            assert!(work.survivor_runs > 0, "{context:?}: the loop did not run");
                            assert!(
                                work.computed_area < plain.computed_area,
                                "{context:?}: {work:?} against {plain:?}"
                            );
                        }
                        _ => {
                            assert!(
                                work.pairs_computed < pairs,
                                "{context:?}: {work:?} of {pairs}"
                            );
                            // Parts left out are caught up when a later block of the band computes
                            // them; the plan's bands at more threads may be one block tall.
                            assert!(
                                threads > 1 || work.lanes_resumed + work.lanes_stepped > 0,
                                "{context:?}: {work:?}"
                            );
                        }
                    }
                    assert_eq!(
                        (silent.class_blocks, silent.class_block_pairs),
                        (0, 0),
                        "{context:?}"
                    );
                }
            }
        }
    }
}

/// Constant windows on one side only still leave blocks out: a constant row needs only the first
/// `k` finite windows of `b`, an ordinary row only `b`'s ordinary windows and its first `k`
/// constant ones.
#[test]
fn the_join_block_skip_fires_with_constant_windows_on_one_side() {
    let m = 8;
    let (flat, other) = (sized_runs(m, &[200, 150]), noise(400, 6));
    for (a, b) in [(&flat, &other), (&other, &flat)] {
        for threads in [1, 2] {
            let run = |force: RouteForce| {
                let guard = forced_route(force);
                let _ = ab_join_topk(&a[..], &b[..], nz(m), unit(), unit(), nz(threads), nz(2))
                    .unwrap();
                (guard.fired(), guard.work())
            };
            let (fired, work) = run(routed());
            let pairs = (a.len() + 1 - m) * (b.len() + 1 - m);
            assert!(fired.class_blocks > 0, "threads {threads}: {fired:?}");
            assert!(
                work.pairs_computed < pairs,
                "threads {threads}: {work:?} of {pairs}"
            );
        }
    }
}

/// A join at a window length that takes the route by default: the skip fires with the route's
/// survivors computed, pairs are left out, and the profile equals the run without the skip.
#[test]
fn the_join_block_skip_fires_at_a_window_length_that_takes_the_loop() {
    let m = 1024;
    let a = sized_runs(m, &[150, 600, 300, 80]);
    let b = sized_runs(m, &[400, 100, 500]);
    let off = RouteForce {
        no_constant_blocks: true,
        ..routed()
    };
    for threads in [1, 2, 8] {
        for k in [1, 2] {
            let run = |force: RouteForce| {
                let guard = forced_route(force);
                let got = ab_join_topk(&a[..], &b[..], nz(m), unit(), unit(), nz(threads), nz(k))
                    .unwrap();
                (left_bits(&got), guard.fired(), guard.work())
            };
            let (bits_on, fired, work) = run(routed());
            let (bits_off, _, base) = run(off);
            assert_eq!(bits_on, bits_off, "threads {threads} k {k}");
            assert!(fired.class_blocks > 0, "threads {threads} k {k}: {fired:?}");
            assert!(work.survivor_runs > 0, "the loop did not run");
            assert!(
                work.pairs_computed < base.pairs_computed,
                "threads {threads} k {k}: {work:?} against {base:?}"
            );
        }
    }
}

/// An AB-join with no constant window on either side leaves no block out.
#[test]
fn the_join_block_skip_is_silent_where_there_is_no_class() {
    let m = 8;
    let (a, b) = (noise(300, 1), noise(260, 2));
    for one_loop in [OneLoop::Auto, OneLoop::On] {
        let guard = forced_route(RouteForce {
            one_loop,
            ..routed()
        });
        let _ = ab_join_report(&a[..], &b[..], nz(m), unit(), unit(), nz(2));
        assert_eq!(class_part(guard.fired()), Fired::default());
    }
}
