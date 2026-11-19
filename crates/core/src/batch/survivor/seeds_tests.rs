//! The seed pre-pass: its pairs are the exact ones, its floors never exceed the floor a window
//! ends with, and a floor is exactly the band below the `k`-th estimate.

use super::SharedFloors;
use super::seeds::{BEST_PARTS, GROUP_ROWS, SUB_SLOTS, SeedInput, raise, seed_floors, sub_block};
use crate::batch::plan::Engine;
use crate::batch::quantized::Quantized;
use crate::columns::valid_row_rank;
use crate::cross_term::CrossTerms;
use crate::estimate::{F64, Producer, band_floor};
use crate::narrow;
use crate::quantize::Scale;
use crate::route::{RouteForce, SeedForce, SurvivorForce, forced_route};
use crate::row::centered_cross_fast;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

/// A series with structure the PAA bound can use: two waves and a little noise, whole ticks.
fn series(n: usize) -> Vec<f64> {
    let mut state = 12345u64;
    (0..n)
        .map(|p| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let t = p as f64;
            (30.0 * (t / 11.0).sin() + 12.0 * (t / 3.7).cos() + ((state >> 60) as f64)).round()
        })
        .collect()
}

fn built(series: &[f64], m: usize) -> Quantized {
    Quantized::build_on(
        series,
        nz(m),
        Scale::new(1.0).unwrap(),
        nz(1),
        Engine::TwoSided,
    )
}

/// The exact `m·Σ aₖbₖ − S_a·S_b` of windows `i` and `j`.
fn exact_n(q: &Quantized, m: usize, i: usize, j: usize) -> i64 {
    let (a, b) = (&q.ticks[i..i + m], &q.ticks[j..j + m]);
    let dot: i128 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| i128::from(x) * i128::from(y))
        .sum();
    let (sa, sb): (i128, i128) = (
        a.iter().map(|&x| i128::from(x)).sum(),
        b.iter().map(|&x| i128::from(x)).sum(),
    );
    (m as i128 * dot - sa * sb) as i64
}

/// `ρ` of the pair as an `f64`, from exact integers.
fn rho(q: &Quantized, m: usize, i: usize, j: usize) -> f64 {
    let n = exact_n(q, m, i, j) as f64;
    n / (exact_n(q, m, i, i) as f64 * exact_n(q, m, j, j) as f64).sqrt()
}

fn ranks(q: &Quantized, i: usize) -> bool {
    q.r[i] != 0.0 && q.bias[i] != f64::NEG_INFINITY
}

/// Each pair of a sub-block, on the wide and the narrow sides, is the exact estimate: the bits
/// the kernel's own formula gives the exact cross term.
#[test]
fn a_sub_blocks_pairs_are_the_exact_estimates() {
    let (series, m) = (series(700), 16);
    let _g = forced_route(RouteForce {
        survivor: SurvivorForce::Paa,
        ..RouteForce::default()
    });
    let q = built(&series, m);
    let exclusion = crate::exclusion::zone(m);
    let wide = q.columns();
    let anchor = q
        .narrow_anchor(nz(m))
        .expect("one anchor covers the series");
    let (ticks, sums) = q.narrow_columns(anchor, nz(m));
    for (rows, slots) in [
        (200..208, 40..48),
        (300..303, 5..13),
        (150..158, 0..3),
        (409..417, 140..148),
    ] {
        let mut by_side = Vec::new();
        for narrow_side in [false, true] {
            let mut out = Vec::new();
            if narrow_side {
                let input = SeedInput {
                    pairs: CrossTerms::of(&ticks[..], m),
                    centered: &|i, j, qt| narrow::cross(qt, sums[i], sums[j]),
                    r: &q.r,
                    bias: &q.bias,
                    paa: q.paa.as_deref().expect("paa"),
                    exclusion,
                    k: 1,
                };
                sub_block(&input, &rows, &slots, &mut out);
            } else {
                let input = SeedInput {
                    pairs: CrossTerms::of(wide.ticks, m),
                    centered: &|i, j, qt| {
                        centered_cross_fast(qt, m as i64, wide.sum[i], wide.sum[j])
                    },
                    r: &q.r,
                    bias: &q.bias,
                    paa: q.paa.as_deref().expect("paa"),
                    exclusion,
                    k: 1,
                };
                sub_block(&input, &rows, &slots, &mut out);
            }
            by_side.push(out);
        }
        let mut expected = Vec::new();
        for row in rows.clone() {
            for slot in slots.clone() {
                let column = row - exclusion - 1 - slot;
                if ranks(&q, row) && ranks(&q, column) {
                    let n = exact_n(&q, m, row, column);
                    expected.push((
                        row,
                        column,
                        valid_row_rank(n, q.r[row], q.r[column], q.bias[column]),
                    ));
                }
            }
        }
        expected.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        for mut got in by_side {
            got.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
            assert_eq!(got.len(), expected.len(), "{rows:?} {slots:?}");
            for (g, e) in got.iter().zip(&expected) {
                assert_eq!(
                    (g.0, g.1, g.2.to_bits()),
                    (e.0, e.1, e.2.to_bits()),
                    "{rows:?} {slots:?}"
                );
            }
        }
    }
}

/// A seed floor is exactly the band below the `k`-th best estimate of the window's seeds on its
/// side, and no floor where it has fewer than `k`.
#[test]
fn a_floor_is_the_band_below_the_kth_estimate() {
    let table = SharedFloors::try_new(20, true).unwrap();
    let mut entries = alloc::vec![
        (5, 1, 0.5),
        (5, 2, 0.9),
        (5, 3, 0.7),
        (6, 1, 0.6),
        (6, 2, 0.8),
        (7, 2, 0.4),
    ];
    let mut one = entries.clone();
    raise(&table, 1, &mut one);
    assert_eq!(
        table.row(5, f64::NEG_INFINITY).to_bits(),
        band_floor(0.9, F64::DELTA).to_bits()
    );
    assert_eq!(
        table.column(2, f64::NEG_INFINITY).to_bits(),
        band_floor(0.9, F64::DELTA).to_bits()
    );
    assert_eq!(
        table.column(1, f64::NEG_INFINITY).to_bits(),
        band_floor(0.6, F64::DELTA).to_bits()
    );
    let table = SharedFloors::try_new(20, true).unwrap();
    raise(&table, 2, &mut entries);
    // Row 5 has 0.9, 0.7, 0.5: its second best is 0.7; row 6 has 0.8, 0.6; row 7 has one.
    assert_eq!(
        table.row(5, f64::NEG_INFINITY).to_bits(),
        band_floor(0.7, F64::DELTA).to_bits()
    );
    assert_eq!(
        table.row(6, f64::NEG_INFINITY).to_bits(),
        band_floor(0.6, F64::DELTA).to_bits()
    );
    assert_eq!(table.row(7, f64::NEG_INFINITY), f64::NEG_INFINITY);
    // Column 2 has 0.9, 0.8, 0.4, column 1 has 0.6, 0.5, column 3 has one.
    assert_eq!(
        table.column(2, f64::NEG_INFINITY).to_bits(),
        band_floor(0.8, F64::DELTA).to_bits()
    );
    assert_eq!(
        table.column(1, f64::NEG_INFINITY).to_bits(),
        band_floor(0.5, F64::DELTA).to_bits()
    );
    assert_eq!(table.column(3, f64::NEG_INFINITY), f64::NEG_INFINITY);
}

/// The exact `k`-th best correlation of window `i` among the windows on one side, `None` with
/// fewer than `k`.
fn kth(q: &Quantized, m: usize, i: usize, k: usize, left: bool) -> Option<f64> {
    let exclusion = crate::exclusion::zone(m);
    let windows = q.windows.len();
    let mut all: Vec<f64> = (0..windows)
        .filter(|&j| ranks(q, j) && j.abs_diff(i) > exclusion && (j < i) == left)
        .map(|j| rho(q, m, i, j))
        .collect();
    all.sort_by(|a, b| b.total_cmp(a));
    all.get(k - 1).copied()
}

/// Whatever the seeds raise, no window's floor is above what it ends with: below its `k`-th best
/// correlation on that side by the band, in every seed route, at every `k`, on one and several
/// threads.
#[test]
fn a_seed_never_raises_a_floor_above_the_final_floor() {
    let (series, m) = (series(500), 12);
    let windows = series.len() - m + 1;
    for seeds in [SeedForce::Auto] {
        for k in [1, 2, 3] {
            let guard = forced_route(RouteForce {
                survivor: SurvivorForce::Paa,
                seeds,
                ..RouteForce::default()
            });
            let q = built(&series, m);
            let wide = q.columns();
            let table = SharedFloors::try_new(windows, true).unwrap();
            let input = SeedInput {
                pairs: CrossTerms::of(wide.ticks, m),
                centered: &|i, j, qt| centered_cross_fast(qt, m as i64, wide.sum[i], wide.sum[j]),
                r: &q.r,
                bias: &q.bias,
                paa: q.paa.as_deref().expect("paa"),
                exclusion: crate::exclusion::zone(m),
                k,
            };
            for threads in [1, 3] {
                seed_floors(Some(&table), &input, (false, false), nz(threads));
            }
            let fired = guard.fired();
            assert!(
                fired.seed_pairs > 0 && fired.seed_raises > 0,
                "{seeds:?} k {k}: {fired:?}"
            );
            let mut raised = 0;
            for i in 0..windows {
                for (left, floor) in [
                    (true, table.row(i, f64::NEG_INFINITY)),
                    (false, table.column(i, f64::NEG_INFINITY)),
                ] {
                    if floor == f64::NEG_INFINITY {
                        continue;
                    }
                    raised += 1;
                    let kth = kth(&q, m, i, k, left).expect("a seed floor needs k real pairs");
                    assert!(
                        floor <= kth - F64::DELTA,
                        "{seeds:?} k {k}: window {i} {} floor {floor} above {kth}",
                        if left { "row" } else { "column" }
                    );
                }
            }
            assert!(raised > 0, "{seeds:?} k {k}");
        }
    }
}

/// A run that keeps a global floor, walks chosen diagonals, or has no shared floors takes no seeds.
#[test]
fn seeds_are_taken_only_by_a_run_whose_floors_are_the_whole_runs() {
    let (series, m) = (series(300), 12);
    let windows = series.len() - m + 1;
    let guard = forced_route(RouteForce {
        survivor: SurvivorForce::Paa,
        seeds: SeedForce::Auto,
        ..RouteForce::default()
    });
    let q = built(&series, m);
    let wide = q.columns();
    let table = SharedFloors::try_new(windows, true).unwrap();
    let input = SeedInput {
        pairs: CrossTerms::of(wide.ticks, m),
        centered: &|i, j, qt| centered_cross_fast(qt, m as i64, wide.sum[i], wide.sum[j]),
        r: &q.r,
        bias: &q.bias,
        paa: q.paa.as_deref().expect("paa"),
        exclusion: crate::exclusion::zone(m),
        k: 1,
    };
    seed_floors(Some(&table), &input, (true, false), nz(1));
    seed_floors(Some(&table), &input, (false, true), nz(1));
    seed_floors(None, &input, (false, false), nz(1));
    assert_eq!(guard.fired().seed_pairs, 0);
    seed_floors(Some(&table), &input, (false, false), nz(1));
    assert!(guard.fired().seed_pairs > 0);
}

/// Sub-blocks are 8 slots and groups 64 rows: the constants the seeds' pair counts follow.
#[test]
fn a_group_seeds_at_most_its_blocks_sub_blocks() {
    let (series, m) = (series(900), 16);
    let windows = series.len() - m + 1;
    let guard = forced_route(RouteForce {
        survivor: SurvivorForce::Paa,
        seeds: SeedForce::Auto,
        ..RouteForce::default()
    });
    let q = built(&series, m);
    let wide = q.columns();
    let table = SharedFloors::try_new(windows, true).unwrap();
    let input = SeedInput {
        pairs: CrossTerms::of(wide.ticks, m),
        centered: &|i, j, qt| centered_cross_fast(qt, m as i64, wide.sum[i], wide.sum[j]),
        r: &q.r,
        bias: &q.bias,
        paa: q.paa.as_deref().expect("paa"),
        exclusion: crate::exclusion::zone(m),
        k: 1,
    };
    seed_floors(Some(&table), &input, (false, false), nz(2));
    let groups = (windows - crate::exclusion::zone(m) - 1).div_ceil(GROUP_ROWS);
    let most = groups * (GROUP_ROWS / SUB_SLOTS) * BEST_PARTS * SUB_SLOTS * SUB_SLOTS;
    let pairs = guard.fired().seed_pairs;
    assert!(pairs > 0 && pairs <= most, "{pairs} of at most {most}");
}
