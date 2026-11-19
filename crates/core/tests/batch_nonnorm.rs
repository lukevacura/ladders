#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! The non-normalized batch profiles (left, two-sided, AB-join) equal the exact reference on bits
//! at every thread count and tile shape: on series of ties, gaps at every boundary, ticks around
//! the `u64` bound, and rows whose distances no `u128` holds.

#[path = "common/nonnorm.rs"]
mod nonnorm;

use ladde::batch::nonnorm::{
    ab_join, ab_join_report_tiled, left_profile, left_profile_report, left_profile_report_tiled,
    two_sided_profile, two_sided_profile_tiled,
};
use nonnorm::*;
use proptest::prelude::*;

const THREADS: [usize; 3] = [1, 2, 7];

fn assert_left(series: &[f64], m: usize) {
    let expected = naive_rows(series, unit(), m);
    for threads in THREADS {
        for (band, chunk, tile) in shapes(series.len()) {
            let (rows, _) = left_profile_report_tiled(
                series,
                nz(m),
                unit(),
                nz(threads),
                nz(band),
                nz(chunk),
                nz(tile),
            );
            assert_eq!(
                batch_rows(&rows),
                expected,
                "left, m {m}, threads {threads}, shape {:?}",
                (band, chunk, tile)
            );
        }
        let rows = left_profile(series, nz(m), unit(), nz(threads));
        assert_eq!(batch_rows(&rows), expected, "left, default tiling");
    }
}

fn assert_two_sided(series: &[f64], m: usize) {
    let expected = naive_sides(series, unit(), m);
    for threads in THREADS {
        for (band, chunk, tile) in shapes(series.len()) {
            let (rows, _) = two_sided_profile_tiled(
                series,
                nz(m),
                unit(),
                nz(threads),
                nz(band),
                nz(chunk),
                nz(tile),
            );
            assert_eq!(
                batch_sides(&rows),
                expected,
                "two-sided, m {m}, threads {threads}, shape {:?}",
                (band, chunk, tile)
            );
        }
        let rows = two_sided_profile(series, nz(m), unit(), nz(threads));
        assert_eq!(batch_sides(&rows), expected, "two-sided, default tiling");
    }
}

fn assert_join(a: &[f64], b: &[f64], m: usize) {
    let expected = naive_join_rows(a, b, unit(), m);
    for threads in THREADS {
        for (band, chunk, tile) in shapes(a.len().max(b.len())) {
            let (rows, _) = ab_join_report_tiled(
                a,
                b,
                nz(m),
                unit(),
                nz(threads),
                nz(band),
                nz(chunk),
                nz(tile),
            );
            assert_eq!(
                batch_rows(&rows),
                expected,
                "join, m {m}, threads {threads}, shape {:?}",
                (band, chunk, tile)
            );
        }
        let rows = ab_join(a, b, nz(m), unit(), nz(threads));
        assert_eq!(batch_rows(&rows), expected, "join, default tiling");
    }
}

fn assert_all(series: &[f64], m: usize) {
    assert_left(series, m);
    assert_two_sided(series, m);
    assert_join(series, series, m);
}

proptest! {
    #![proptest_config(testkit::generators::seeded(32))]

    #[test]
    fn random_series_equal_the_reference(series in series(), m in 1usize..=6) {
        assert_all(&series, m);
    }

    #[test]
    fn tie_series_equal_the_reference(series in tie_series(), m in 1usize..=6) {
        assert_all(&series, m);
    }

    #[test]
    fn two_series_join_equal_the_reference(
        a in prop_oneof![series(), tie_series()],
        b in prop_oneof![series(), tie_series()],
        m in 1usize..=6,
    ) {
        assert_join(&a, &b, m);
    }
}

#[test]
fn constant_runs_periods_and_few_levels_tie_to_the_smaller_index() {
    let constant = vec![7.0; 40];
    let runs: Vec<f64> = (0..60)
        .map(|k| {
            if (k / 7) % 2 == 0 {
                0.0
            } else {
                (1u64 << 33) as f64
            }
        })
        .collect();
    let few_levels: Vec<f64> = (0..60).map(|k| f64::from((k * 5 % 3) as u8)).collect();
    let mut all = vec![constant, runs, few_levels];
    for period in [2usize, 4, 7] {
        all.push(
            (0..70)
                .map(|k| [0.0, 3.0, 1.0, 4.0, 1.0, 5.0, 9.0][k % period])
                .collect(),
        );
    }
    let offset_repeats: Vec<f64> = (0..70)
        .map(|k| (k % 9) as f64 + if k >= 35 { 1e6 } else { 0.0 })
        .collect();
    all.push(offset_repeats);
    for series in &all {
        for m in [1, 3, 5, 8] {
            assert_all(series, m);
        }
    }
    let flat = left_profile(&all[0], nz(4), unit(), nz(1));
    assert_eq!(batch_rows(&flat)[20], (20, 0, 0));
}

#[test]
fn gaps_at_the_start_the_end_and_every_tile_boundary() {
    let walk = || -> Vec<f64> { (0..48).map(|k| f64::from((k * 7 % 11) as u8)).collect() };
    let gap_sets: [&[usize]; 5] = [
        &[0, 1, 2],
        &[45, 46, 47],
        &[0, 47],
        &[5, 6, 11, 12, 17, 18, 23, 24],
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        ],
    ];
    for gaps in gap_sets {
        let mut series = walk();
        for &g in gaps {
            series[g] = f64::NAN;
        }
        for m in [2, 4, 6] {
            assert_all(&series, m);
        }
    }
    let all_gaps = vec![f64::NAN; 30];
    assert_all(&all_gaps, 4);
}

#[test]
fn empty_short_and_oversized_windows_return_no_rows_or_no_neighbors() {
    let empty: [f64; 0] = [];
    for threads in THREADS {
        assert!(left_profile(&empty, nz(3), unit(), nz(threads)).is_empty());
        assert!(left_profile(&[1.0, 2.0], nz(3), unit(), nz(threads)).is_empty());
        assert!(two_sided_profile(&[1.0, 2.0], nz(3), unit(), nz(threads)).is_empty());
        assert!(left_profile(&[1.0; 5], nz(usize::MAX), unit(), nz(threads)).is_empty());
        assert!(ab_join(&[1.0, 2.0], &[1.0; 5], nz(3), unit(), nz(threads)).is_empty());
        let none = ab_join(&[1.0; 5], &[1.0, 2.0], nz(3), unit(), nz(threads));
        assert_eq!(none.len(), 3);
        assert!(none.iter().all(|(_, n)| n.index().get().is_none()));
    }
}

/// Two levels `level` apart, four ticks to a window: the pair `level` apart sits at `m·R² =
/// 4·level²`, below `2⁶⁴` exactly when `level < 2³¹`.
fn two_levels(level: i64) -> Vec<f64> {
    (0..24)
        .map(|k| if (k / 6) % 2 == 0 { 0.0 } else { level as f64 })
        .collect()
}

#[test]
fn ticks_straddling_the_u64_bound_pick_the_width_and_keep_the_bits() {
    let whole = |series: &[f64]| {
        let n = series.len();
        let (rows, report) =
            left_profile_report_tiled(series, nz(4), unit(), nz(1), nz(n), nz(n), nz(n));
        (batch_rows(&rows), report)
    };
    for (level, wide) in [
        ((1i64 << 31) - 1, 0),
        (1 << 31, 1),
        ((1 << 31) + 1, 1),
        (1 << 32, 1),
        (1 << 40, 1),
    ] {
        let series = two_levels(level);
        let (rows, report) = whole(&series);
        assert_eq!(report.tiles, 1, "level {level}");
        assert_eq!(report.wide_tiles, wide, "level {level}");
        assert_eq!(rows, naive_rows(&series, unit(), 4), "level {level}");
        assert_all(&series, 4);
    }
}

/// Calm ticks, then a shock past the bound, then calm again: only the tiles that see the shock
/// run in `u128`.
fn shocked_walk() -> Vec<f64> {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut price = 0i64;
    (0..140)
        .map(|k| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            price += ((state >> 33) % 7) as i64 - 3;
            if (96..104).contains(&k) {
                (price + (1i64 << 33)) as f64
            } else {
                price as f64
            }
        })
        .collect()
}

#[test]
fn each_tile_picks_its_width_from_its_own_range() {
    let series = shocked_walk();
    let m = 4;
    let expected = naive_rows(&series, unit(), m);
    for threads in THREADS {
        let (rows, report) =
            left_profile_report_tiled(&series, nz(m), unit(), nz(threads), nz(8), nz(8), nz(8));
        assert_eq!(batch_rows(&rows), expected, "threads {threads}");
        assert!(report.wide_tiles > 0, "{report:?}");
        assert!(report.wide_tiles < report.tiles, "{report:?}");
    }
    let calm: Vec<f64> = series[..90].to_vec();
    let (_, report) = left_profile_report(&calm, nz(m), unit(), nz(1));
    assert_eq!(report.wide_tiles, 0);
    assert_all(&series, m);
}

/// A join's tiles pick their width too: `b` calm, `a` shocked.
#[test]
fn join_tiles_pick_their_width_from_their_own_sides() {
    let a = shocked_walk();
    let b: Vec<f64> = a[..60].to_vec();
    let m = 4;
    let expected = naive_join_rows(&a, &b, unit(), m);
    let (rows, report) = ab_join_report_tiled(&a, &b, nz(m), unit(), nz(2), nz(8), nz(8), nz(8));
    assert_eq!(batch_rows(&rows), expected);
    assert!(
        report.wide_tiles > 0 && report.wide_tiles < report.tiles,
        "{report:?}"
    );
}

/// A deterministic pseudo-random walk.
fn walk(n: usize, spread: i64) -> Vec<f64> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut level = 0i64;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            level += ((state >> 56) as i64 % (2 * spread + 1)) - spread;
            level as f64
        })
        .collect()
}

/// The squared distance of two windows, by plain `i128`.
fn squared(series: &[f64], i: usize, j: usize, m: usize) -> i128 {
    (0..m)
        .map(|k| {
            let d = series[i + k] as i128 - series[j + k] as i128;
            d * d
        })
        .sum()
}

#[test]
fn a_long_series_is_the_same_for_every_thread_count_and_tile_shape() {
    let series = walk(3_300, 3);
    let m: usize = 24;
    let exclusion = ladde::exclusion::zone(m);
    let windows = series.len() + 1 - m;
    let baseline = left_profile(&series, nz(m), unit(), nz(1));
    for (i, &(row, neighbor)) in baseline.iter().enumerate().step_by(97) {
        assert_eq!(row, i as u64);
        let best = (0..i.saturating_sub(exclusion))
            .map(|j| (squared(&series, i, j, m), j))
            .min();
        match best {
            Some((distance, j)) => {
                assert_eq!(neighbor.squared_ticks() as i128, distance, "row {i}");
                assert_eq!(neighbor.index().get(), Some(j as u64), "row {i}");
            }
            None => assert!(neighbor.index().get().is_none()),
        }
    }
    let expected = batch_rows(&baseline);
    let two = batch_sides(&two_sided_profile(&series, nz(m), unit(), nz(1)));
    for threads in [2, 5, 16] {
        let rows = left_profile(&series, nz(m), unit(), nz(threads));
        assert_eq!(batch_rows(&rows), expected, "threads {threads}");
        let sides = two_sided_profile(&series, nz(m), unit(), nz(threads));
        assert_eq!(batch_sides(&sides), two, "two-sided, threads {threads}");
    }
    for (band, chunk, tile) in [(1, 1, 1), (7, 3, 9), (64, 16, 256), (windows, 64, windows)] {
        let (rows, _) =
            left_profile_report_tiled(&series, nz(m), unit(), nz(4), nz(band), nz(chunk), nz(tile));
        assert_eq!(
            batch_rows(&rows),
            expected,
            "shape {:?}",
            (band, chunk, tile)
        );
        let (sides, _) =
            two_sided_profile_tiled(&series, nz(m), unit(), nz(4), nz(band), nz(chunk), nz(tile));
        assert_eq!(
            batch_sides(&sides),
            two,
            "two-sided shape {:?}",
            (band, chunk, tile)
        );
    }
    let joined = ab_join(&series, &series[500..1900], nz(m), unit(), nz(1));
    for threads in [2, 7] {
        let rows = ab_join(&series, &series[500..1900], nz(m), unit(), nz(threads));
        assert_eq!(
            batch_rows(&rows),
            batch_rows(&joined),
            "join, threads {threads}"
        );
    }
}

/// The widest window the ticks allow before `m·R²` can pass `u128`: `m = 2²⁰` and a range of
/// `2⁵⁴`.
const HUGE_M: usize = 1 << 20;

/// A calm series of `HUGE_M + exclusion + 8` ticks (eight windows past the exclusion zone's
/// first), with `+2⁵³` at tick `p` and `−2⁵³` at `p + 1`: from the row whose window ends at
/// `p + 1` on, `m·R²` is `2¹²⁸`. Returns the series and that row.
fn huge_series() -> (Vec<f64>, usize) {
    let exclusion = ladde::exclusion::zone(HUGE_M);
    let n = HUGE_M + exclusion + 8;
    let p = HUGE_M + exclusion + 2;
    let mut series: Vec<f64> = (0..n).map(|k| ((k * 31) % 17) as f64).collect();
    series[p] = (1i64 << 53) as f64;
    series[p + 1] = -((1i64 << 53) as f64);
    (series, p + 2 - HUGE_M)
}

#[test]
fn a_row_no_width_holds_has_no_neighbor_and_nothing_panics() {
    let (series, first) = huge_series();
    let m = nz(HUGE_M);
    let exclusion = ladde::exclusion::zone(HUGE_M);
    assert_eq!(first, exclusion + 4);
    let expected = naive_rows(&series[..HUGE_M + exclusion + 3], unit(), HUGE_M);
    assert_eq!(expected.len(), first);
    for (threads, shape) in [(1, (1, 1, 1)), (3, (2, 2, 4)), (2, (64, 64, 64))] {
        let (rows, report) = left_profile_report_tiled(
            &series,
            m,
            unit(),
            nz(threads),
            nz(shape.0),
            nz(shape.1),
            nz(shape.2),
        );
        let rows = batch_rows(&rows);
        assert_eq!(rows.len(), exclusion + 9);
        assert_eq!(rows[..first], expected[..first], "threads {threads}");
        assert!(rows[first - 1].1 > u128::from(u64::MAX));
        for row in &rows[first..] {
            assert_eq!((row.1, row.2), (0, u64::MAX), "row {}", row.0);
        }
        assert!(report.wide_tiles <= report.tiles);
    }

    let sides = batch_sides(&two_sided_profile(&series, m, unit(), nz(2)));
    let left = batch_rows(&left_profile(&series, m, unit(), nz(2)));
    let samples = samples(&series, unit());
    let tick = |k: usize| samples[k].unwrap() as i128;
    let distance = |i: usize, j: usize| {
        (0..HUGE_M)
            .map(|k| (tick(i + k) - tick(j + k)).pow(2))
            .sum::<i128>()
    };
    let mut right = vec![None::<(i128, usize)>; sides.len()];
    for i in exclusion + 1..first {
        for (j, best) in right.iter_mut().enumerate().take(i - exclusion) {
            let candidate = (distance(i, j), i);
            if best.is_none_or(|current| candidate < current) {
                *best = Some(candidate);
            }
        }
    }
    for (row, (index, side)) in sides.iter().enumerate() {
        assert_eq!(*index, row as u64);
        assert_eq!(
            (index, side[0].0, side[0].1),
            (&left[row].0, left[row].1, left[row].2)
        );
        let want = right[row].map_or((0, u64::MAX), |(d, i)| (d as u128, i as u64));
        assert_eq!(side[1], want, "right of row {row}");
        let nearest = match (side[0].1 == u64::MAX, right[row].is_none()) {
            (true, true) => (0, u64::MAX),
            (true, false) => want,
            (false, true) => side[0],
            (false, false) => {
                if want.0 < side[0].0 {
                    want
                } else {
                    side[0]
                }
            }
        };
        assert_eq!(side[2], nearest, "nearest of row {row}");
    }

    // A join of eight rows: the shock enters the rows' prefix at row 4.
    let mut a: Vec<f64> = (0..HUGE_M + 7).map(|k| ((k * 31) % 17) as f64).collect();
    a[HUGE_M + 2] = (1i64 << 53) as f64;
    a[HUGE_M + 3] = -((1i64 << 53) as f64);
    let calm: Vec<f64> = (0..HUGE_M + 2).map(|k| ((k * 7) % 13) as f64).collect();
    let join = batch_rows(&ab_join(&a, &calm, m, unit(), nz(2)));
    let expected = naive_join_rows(&a[..HUGE_M + 3], &calm, unit(), HUGE_M);
    assert_eq!(join.len(), 8);
    assert_eq!(join[..4], expected[..4]);
    for row in &join[4..] {
        assert_eq!((row.1, row.2), (0, u64::MAX));
    }

    let extremes: Vec<f64> = (0..HUGE_M + 2)
        .map(|k| {
            if k % 2 == 0 {
                (1i64 << 53) as f64
            } else {
                -((1i64 << 53) as f64)
            }
        })
        .collect();
    let join = ab_join(&calm, &extremes, m, unit(), nz(1));
    assert_eq!(join.len(), 3);
    assert!(join.iter().all(|(_, n)| n.index().get().is_none()));
}
