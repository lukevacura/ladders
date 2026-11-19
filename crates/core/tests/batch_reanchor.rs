#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! A batch tile whose rows or columns drift past what one anchor covers runs the narrow kernel
//! when each of its windows fits: the carried diagonals move their anchors by an exact correction.
//! On drifting walks, level shifts and shocks every engine equals the exact reference on bits at
//! every thread count and tiling, runs no tile on the exact i64 pass, and moves its anchors.

#[path = "common/route.rs"]
mod route;
#[path = "common/spill_run.rs"]
mod spill_run;

use ladde::batch::{
    BatchReport, ab_join_report_tiled, left_profile_report, left_profile_report_tiled,
    two_sided_profile_tiled, two_sided_topk_profile_tiled,
};
use ladde::entry::{TwoSided, TwoSidedTopk};
use ladde::quantize::Scale;
use ladde::route::{RouteForce, forced_route};
use route::spill_ref;
use route::{Bits, nz, profile_bits};

const M: usize = 4;
/// The most ticks one anchor holds either side of itself at `M`.
const BOUND: f64 = (i32::MAX as usize / M) as f64;
/// Per-sample drift that crosses two bounds in two hundred samples: far past one anchor over a
/// series, well inside one for a cell of a few dozen lanes and rows.
const DRIFT: f64 = BOUND / 100.0;
const N: usize = 600;

fn scale() -> Scale {
    Scale::new(1.0).unwrap()
}

/// A small deterministic noise in `0..bound`.
fn noise(k: u64, bound: u64) -> f64 {
    let mut x = k.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    (x % bound) as f64
}

/// A steady rise over small noise.
fn steady(n: usize, seed: u64, sign: f64) -> Vec<f64> {
    (0..n)
        .map(|k| sign * k as f64 * DRIFT + noise(k as u64 + seed, 1000))
        .collect()
}

/// A walk whose steps are uniform in `0..2 · DRIFT`: the drift of `steady` with a walk's wander.
fn walk(n: usize, seed: u64) -> Vec<f64> {
    let mut level = 0.0;
    (0..n)
        .map(|k| {
            level += noise(k as u64 + seed, 1 << 20) / f64::from(1 << 20) * 2.0 * DRIFT;
            level + noise(k as u64, 1000)
        })
        .collect()
}

/// Flat noise that shifts up by most of a bound every ninety samples.
fn shifts(n: usize, seed: u64) -> Vec<f64> {
    (0..n)
        .map(|k| (k / 90) as f64 * 0.7 * BOUND + noise(k as u64 + seed, 1000))
        .collect()
}

/// A walk with single-sample shocks of more than half a bound.
fn shocks(n: usize, seed: u64) -> Vec<f64> {
    let mut series = walk(n, seed);
    for k in (37..n).step_by(113) {
        series[k] += 0.6 * BOUND;
    }
    series
}

fn kinds() -> Vec<(&'static str, Vec<f64>, bool)> {
    vec![
        ("steady", steady(N, 1, 1.0), true),
        ("falling", steady(N, 2, -1.0), true),
        ("walk", walk(N, 3), true),
        ("shifts", shifts(N, 4), true),
        ("shocks", shocks(N, 5), false),
    ]
}

const TILINGS: [(usize, usize, usize); 4] = [
    (64, 128, 256),
    (200, 100, 300),
    (600, 64, 600),
    (37, 70, 140),
];
const THREADS: [usize; 2] = [1, 3];

type TwoSidedBits = Vec<(u64, [Bits; 3])>;
type TopkBits = Vec<(u64, [Bits; 2], Vec<Bits>)>;

fn unrowed(b: Bits) -> Bits {
    (0, b.1, b.2, b.3)
}

fn two_sided_bits(profile: &[(u64, TwoSided)]) -> TwoSidedBits {
    profile
        .iter()
        .map(|&(row, s)| (row, [s.left, s.right, s.nearest].map(bits_of)))
        .collect()
}

fn bits_of(e: ladde::entry::Entry) -> Bits {
    route::bits(0, e)
}

fn exact_two_sided_bits(series: &[f64]) -> TwoSidedBits {
    spill_run::exact_two_sided(series, scale(), M)
        .into_iter()
        .enumerate()
        .map(|(row, sides)| (row as u64, sides.map(unrowed)))
        .collect()
}

fn two_sided_topk_bits(profile: &[(u64, TwoSidedTopk)]) -> TopkBits {
    profile
        .iter()
        .map(|(row, s)| {
            (
                *row,
                [s.left, s.right].map(bits_of),
                s.nearest.iter().map(|&e| bits_of(e)).collect(),
            )
        })
        .collect()
}

fn exact_two_sided_topk_bits(series: &[f64], k: usize) -> TopkBits {
    spill_run::exact_two_sided_topk(series, scale(), M, k)
        .into_iter()
        .enumerate()
        .map(|(row, (left, right, nearest))| {
            (
                row as u64,
                [unrowed(left), unrowed(right)],
                nearest.into_iter().map(unrowed).collect(),
            )
        })
        .collect()
}

/// What one engine run reports: whether it equals `expected`, and its report.
fn assert_narrow(report: BatchReport, narrow_only: bool, what: &str) {
    assert!(report.tiles > 0, "{what}");
    if narrow_only {
        assert_eq!(report.wide_tiles, 0, "{what}: {report:?}");
    }
}

#[test]
fn the_left_profile_of_a_drifting_series_runs_narrow_and_equals_the_reference() {
    let guard = forced_route(RouteForce::default());
    for (name, series, narrow_only) in kinds() {
        let expected = spill_run::reference(&series, M, scale());
        for threads in THREADS {
            let (default, report) = left_profile_report(&series, nz(M), scale(), nz(threads));
            assert_eq!(profile_bits(&default), expected, "{name} default {threads}");
            assert_narrow(report, narrow_only, name);
            for (band, chunk, tile) in TILINGS {
                let (got, report) = left_profile_report_tiled(
                    &series,
                    nz(M),
                    scale(),
                    nz(threads),
                    nz(band),
                    nz(chunk),
                    nz(tile),
                );
                let what = format!("{name} threads {threads} tiling {:?}", (band, chunk, tile));
                assert_eq!(profile_bits(&got), expected, "{what}");
                assert_narrow(report, narrow_only, &what);
            }
        }
    }
    assert!(
        guard.fired().cell_reanchors > 0,
        "the drift moved some anchors"
    );
}

#[test]
fn the_two_sided_profile_of_a_drifting_series_runs_narrow_and_equals_the_reference() {
    let guard = forced_route(RouteForce::default());
    for (name, series, narrow_only) in kinds() {
        let expected = exact_two_sided_bits(&series);
        for threads in THREADS {
            for (band, chunk, tile) in TILINGS {
                let (got, report) = two_sided_profile_tiled(
                    &series,
                    nz(M),
                    scale(),
                    nz(threads),
                    nz(band),
                    nz(chunk),
                    nz(tile),
                );
                let what = format!("{name} threads {threads} tiling {:?}", (band, chunk, tile));
                assert_eq!(two_sided_bits(&got), expected, "{what}");
                assert_narrow(report, narrow_only, &what);
            }
        }
    }
    assert!(guard.fired().cell_reanchors > 0);
}

#[test]
fn the_top_four_of_a_drifting_series_run_narrow_and_equal_the_reference() {
    let guard = forced_route(RouteForce::default());
    for (name, series, narrow_only) in kinds() {
        let expected = exact_two_sided_topk_bits(&series, 4);
        for threads in THREADS {
            for (band, chunk, tile) in TILINGS {
                let (got, report) = two_sided_topk_profile_tiled(
                    &series,
                    nz(M),
                    scale(),
                    nz(threads),
                    nz(band),
                    nz(chunk),
                    nz(tile),
                    nz(4),
                )
                .unwrap();
                let what = format!("{name} threads {threads} tiling {:?}", (band, chunk, tile));
                assert_eq!(two_sided_topk_bits(&got), expected, "{what}");
                assert_narrow(report, narrow_only, &what);
            }
        }
    }
    assert!(guard.fired().cell_reanchors > 0);
}

#[test]
fn a_join_of_drifting_series_runs_narrow_and_equals_the_reference() {
    let guard = forced_route(RouteForce::default());
    let kinds = kinds();
    for (i, (name, a, narrow_a)) in kinds.iter().enumerate() {
        let (other, b, narrow_b) = &kinds[(i + 2) % kinds.len()];
        let b = &b[..N / 2 + 11];
        let expected = spill_run::exact_join(a, scale(), b, scale(), M);
        for threads in THREADS {
            for (band, chunk, tile) in TILINGS {
                let (got, report) = ab_join_report_tiled(
                    a,
                    b,
                    nz(M),
                    scale(),
                    scale(),
                    nz(threads),
                    nz(band),
                    nz(chunk),
                    nz(tile),
                );
                let what = format!(
                    "{name} with {other} threads {threads} tiling {:?}",
                    (band, chunk, tile)
                );
                assert_eq!(profile_bits(&got), expected, "{what}");
                assert_narrow(report, *narrow_a && *narrow_b, &what);
            }
        }
    }
    assert!(guard.fired().cell_reanchors > 0);
}

/// The series alternates 2.8 bounds apart every other sample: no window fits one anchor, so no
/// cut helps and every tile takes the exact pass.
#[test]
fn a_series_no_cell_of_which_fits_runs_the_exact_pass_and_equals_the_reference() {
    let series: Vec<f64> = (0..200)
        .map(|k| noise(k as u64, 1000) + if k % 2 == 0 { 0.0 } else { 2.8 * BOUND })
        .collect();
    let _guard = forced_route(RouteForce::default());
    let expected = spill_run::reference(&series, M, scale());
    let (got, report) =
        left_profile_report_tiled(&series, nz(M), scale(), nz(2), nz(16), nz(64), nz(128));
    assert_eq!(profile_bits(&got), expected);
    assert_eq!(report.wide_tiles, report.tiles);
}

/// Forcing a small cell size makes every tile re-anchor often, on a series one anchor covers.
#[test]
fn forced_cells_equal_the_reference_and_count_their_boundaries() {
    let series = walk(150, 9);
    let expected = spill_run::reference(&series, M, scale());
    for cell in [1, 2, 5, 64] {
        for anchor_seed in [0, 0x5EED] {
            let guard = forced_route(RouteForce {
                reanchor_cell: cell,
                anchor_seed,
                ..RouteForce::default()
            });
            for (band, chunk, tile) in [(1, 1, 1), (9, 3, 7), (40, 8, 64), (150, 150, 150)] {
                let (got, report) = left_profile_report_tiled(
                    &series,
                    nz(M),
                    scale(),
                    nz(2),
                    nz(band),
                    nz(chunk),
                    nz(tile),
                );
                assert_eq!(
                    profile_bits(&got),
                    expected,
                    "cell {cell} seed {anchor_seed} tiling {:?}",
                    (band, chunk, tile)
                );
                assert_eq!(report.wide_tiles, 0);
            }
            assert!(
                guard.fired().extra_reanchors > 0 || cell >= 64,
                "cell {cell}: the forced size ended some segment"
            );
        }
    }
}
