extern crate std;

use super::*;
use crate::multidim::{NonFinitePolicy, checked_include};
use crate::naive::naive_multi_profile_nonnorm;

const THREADS: [usize; 3] = [1, 2, 7];

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

fn tick_series(samples: &[Option<i64>], m: usize) -> TickSeries {
    TickSeries::from_samples(samples, m)
}

/// Dimension 0 swings between `±2⁶²` from sample `wide_from` on (`m·R² ≥ 2¹²⁸` from there for
/// `m ≥ 4`), dimension 1 stays small throughout, dimension 2 is a small copy with a gap.
fn huge_dimensions(n: usize, wide_from: usize) -> Vec<Vec<Option<i64>>> {
    let swing = |t: usize| match t % 5 {
        0 | 3 => 1i64 << 62,
        1 => -(1i64 << 62),
        _ => (t % 3) as i64,
    };
    let small: Vec<Option<i64>> = (0..n).map(|t| Some((t * t % 7) as i64)).collect();
    let mut gapped = small.clone();
    gapped[n / 2] = None;
    vec![
        (0..n)
            .map(|t| {
                Some(if t >= wide_from {
                    swing(t)
                } else {
                    (t % 4) as i64
                })
            })
            .collect(),
        small,
        gapped,
    ]
}

fn assert_rule(n: usize, wide_from: usize, m: usize, include: &[usize], discords: bool) {
    assert_dimensions(&huge_dimensions(n, wide_from), m, include, discords);
}

fn assert_dimensions(dimensions: &[Vec<Option<i64>>], m: usize, include: &[usize], discords: bool) {
    let n = dimensions[0].len();
    let refs: Vec<&[Option<i64>]> = dimensions.iter().map(Vec::as_slice).collect();
    let expected =
        naive_multi_profile_nonnorm(&refs, m, include, discords, NonFinitePolicy::Missing).unwrap();
    let series: Vec<TickSeries> = dimensions
        .iter()
        .map(|samples| tick_series(samples, m))
        .collect();
    let order = checked_include(include, series.len()).unwrap();
    for threads in THREADS {
        for (band, chunk, tile) in [(1, 1, 1), (3, 4, 8), (n, 2, 4), (4, n, n)] {
            let plan = TilePlan::fixed(nz(band), nz(chunk), nz(tile));
            let (profile, report) =
                nonnorm_on(&series, nz(m), &order, discords, nz(threads), Some(plan));
            assert_eq!(
                profile,
                expected,
                "threads {threads}, shape {:?}",
                (band, chunk, tile)
            );
            assert!(report.wide_tiles > 0 && report.wide_tiles <= report.tiles);
        }
    }
}

#[test]
fn a_missing_leading_sample_is_not_a_tick_for_the_rule() {
    // Dimension 0 starts with a gap, then holds ticks near `2⁶²` that differ by a few: its
    // present range is tiny, though the fill before its first sample would span `2⁶²`.
    let (n, m) = (50, 16);
    let dimensions = vec![
        (0..n)
            .map(|t| (t > 0).then(|| (1i64 << 62) + (t * t % 7) as i64))
            .collect::<Vec<_>>(),
        (0..n).map(|t| Some((t * 5 % 11) as i64)).collect(),
    ];
    for include in [&[][..], &[1][..]] {
        for discords in [false, true] {
            assert_dimensions(&dimensions, m, include, discords);
        }
    }
    let series = tick_series(&dimensions[0], m);
    assert_eq!(
        first_unrepresentable_row(&series, None, m),
        series.windows()
    );
}

#[test]
fn pairs_no_width_holds_have_no_distance() {
    for (n, wide_from, m) in [(40, 18, 4), (36, 0, 5), (30, 25, 4)] {
        for include in [&[][..], &[2, 0][..]] {
            for discords in [false, true] {
                assert_rule(n, wide_from, m, include, discords);
            }
        }
    }
}

#[test]
fn the_rule_cuts_a_dimension_at_its_own_first_row() {
    let (n, m) = (40, 4);
    let dimensions = huge_dimensions(n, 18);
    let series = tick_series(&dimensions[0], m);
    let cut = first_unrepresentable_row(&series, None, m);
    assert!(cut > 10 && cut < 20, "{cut}");
    let others = tick_series(&dimensions[1], m);
    assert_eq!(
        first_unrepresentable_row(&others, None, m),
        others.windows()
    );
    let order = checked_include(&[], 3).unwrap();
    let all: Vec<TickSeries> = dimensions.iter().map(|d| tick_series(d, m)).collect();
    let (profile, _) = nonnorm_on(&all, nz(m), &order, false, nz(1), None);
    // With three dimensions the last k needs the unrepresentable one wherever it is missing, so
    // rows past the cut lose the full-dimensional neighbor while k = 0 keeps one.
    let late = n - m;
    assert!(profile.distance[2][late].is_missing());
    assert!(!profile.distance[0][late].is_missing());
}

#[test]
fn a_band_holds_a_slot_per_dimension_within_the_budget_of_one() {
    let (windows, m, threads) = (20_000, 16, nz(4));
    let one = multi_plan(windows, m, 1, threads, None);
    assert_eq!(
        one.band_rows,
        TilePlan::for_join(windows, windows, m, 4, default_chunk_lanes()).band_rows
    );
    assert!(one.band_rows >= 8, "{}", one.band_rows);
    for count in [2, 5, 8] {
        let plan = multi_plan(windows, m, count, threads, None);
        assert_eq!(plan.band_rows, one.band_rows / count, "d {count}");
        assert_eq!(plan.tile_lanes, one.tile_lanes, "d {count}");
    }
    // Never below one row, and a plan fixed by the caller is kept as given.
    assert_eq!(multi_plan(windows, m, 1 << 40, threads, None).band_rows, 1);
    let fixed = TilePlan::fixed(nz(6), nz(2), nz(4));
    assert_eq!(multi_plan(windows, m, 5, threads, Some(fixed)).band_rows, 6);
}
