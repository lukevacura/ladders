extern crate std;

use super::super::{Single, first_unrepresentable_row};
use super::*;
use crate::nonnorm::exact_squared;
use alloc::vec::Vec;

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

#[test]
fn a_full_list_keeps_the_smallest_pairs_in_order_and_ties_go_to_the_smaller_index() {
    let mut list = Vec::new();
    for candidate in [(5, 9), (1, 4), (5, 2), (3, 3), (5, 7), (0, 8), (5, 2)] {
        keep_nearest(&mut list, 4, candidate);
        assert!(list.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(list.len() <= 4);
    }
    assert_eq!(list, [(0, 8), (1, 4), (3, 3), (5, 2)]);
    let mut merged = list.clone();
    merge_nearest(&mut merged, 4, &[(0, 1), (2, 2), (9, 9)]);
    assert_eq!(merged, [(0, 1), (0, 8), (1, 4), (2, 2)]);
    merge_nearest(&mut merged, 4, &[(7, 0)]);
    assert_eq!(merged, [(0, 1), (0, 8), (1, 4), (2, 2)]);
}

#[test]
fn a_fitted_plan_holds_its_lists_within_the_budget_for_any_k() {
    let budget = PARTIALS_BYTES / core::mem::size_of::<Candidate>();
    let windows = 1_000_000;
    for k in [1, 2, 3, 8, 100, 4_096, 1_000_000, usize::MAX] {
        for (band, lanes) in [(1_024, 4_096), (64, 1_536), (131_072, 131_072)] {
            let plan = TilePlan {
                band_rows: band,
                chunk_lanes: 682,
                tile_lanes: lanes,
                k: NonZeroUsize::MIN,
                survivor: Default::default(),
            };
            let fitted = fitted_plan(plan, nz(k), windows);
            let (b, l) = (fitted.band_rows, fitted.tile_lanes);
            assert!(b >= 1 && l >= 1 && b <= band && l <= lanes);
            assert!(fitted.chunk_lanes <= l);
            assert_eq!(fitted.k.get(), k);
            let held = b * k.min(l) + (l + b) * k.min(b);
            assert!(held <= budget || (b == 1 && l == 1), "k {k}: {b} x {l}");
        }
    }
    let small = TilePlan {
        band_rows: 3,
        chunk_lanes: 2,
        tile_lanes: 4,
        k: NonZeroUsize::MIN,
        survivor: Default::default(),
    };
    let kept = fitted_plan(small, nz(1_000), 100);
    assert_eq!(
        (kept.band_rows, kept.chunk_lanes, kept.tile_lanes),
        (3, 2, 4)
    );
}

/// Each list is reserved for the most it can hold, the complete windows that can reach its
/// column, and a window with a gap holds nothing.
#[test]
fn the_right_side_reserves_what_each_column_can_hold_and_nothing_for_a_gap() {
    let samples: Vec<Option<i64>> = (0..40)
        .map(|t| (t % 9 != 4 && t != 30).then_some(t * 3 % 5))
        .collect();
    let series = TickSeries::from_samples(&samples, 3);
    let exclusion = 1;
    for k in [1, 4, 1_000] {
        let book = RightTops::new(&series, nz(k), exclusion).unwrap();
        let complete: Vec<usize> = (0..series.windows())
            .filter(|&i| series.complete[i])
            .collect();
        for column in 0..series.windows() {
            let reach = complete
                .iter()
                .filter(|&&window| window > column + exclusion)
                .count();
            let capacity = book
                .groups
                .with_column(column, |lists| lists[column % RIGHT_GROUP].capacity());
            if series.complete[column] {
                assert_eq!(capacity, k.min(reach), "column {column}, k {k}");
            } else {
                assert_eq!(capacity, 0, "column {column}");
            }
        }
    }
}

/// The left lists of `rows` against a brute force over every earlier complete window, with the
/// rows whose distances no `u128` holds keeping none.
#[test]
fn the_left_lists_equal_a_brute_force_and_rows_past_the_wide_bound_keep_none() {
    let m = 4;
    // Small ticks with gaps, then a shock to `±2⁶²` (`m·R² = 2¹²⁸` from there on).
    let samples: Vec<Option<i64>> = (0..60)
        .map(|t| match t {
            17 | 18 => None,
            40 => Some(-(1i64 << 62)),
            41 => Some(1 << 62),
            _ => Some(t * 7 % 3),
        })
        .collect();
    let series = TickSeries::from_samples(&samples, m);
    let from = first_unrepresentable_row(&series, None, m);
    assert!(from < series.windows());
    let exclusion = crate::exclusion::zone(m);
    for k in [1usize, 3, 100] {
        for threads in [1, 3] {
            let keep = TopK { k, right: None };
            let (rows, _) = left_rows(&series, nz(m), nz(threads), None, None, keep, Vec::new());
            let mut at = 0;
            for row in 0..series.windows() {
                let mut kept = Vec::new();
                while at < rows.len() && rows[at].0 == row as u64 {
                    if let Some(index) = rows[at].1.index().get() {
                        kept.push((rows[at].1.squared_ticks(), index));
                    }
                    at += 1;
                }
                let mut expected: Vec<Candidate> = Vec::new();
                if row < from && series.complete[row] {
                    for j in 0..row.saturating_sub(exclusion) {
                        if series.complete[j] {
                            let squared =
                                exact_squared(&series.ticks[row..row + m], &series.ticks[j..j + m])
                                    .expect("rows before the bound are representable");
                            expected.push((squared, j as u64));
                        }
                    }
                }
                expected.sort_unstable();
                expected.truncate(k);
                assert_eq!(kept, expected, "row {row}, k {k}, threads {threads}");
            }
            assert_eq!(at, rows.len());
        }
    }
    // The one-neighbor keeper agrees at the front of every list.
    let (single, _) = left_rows(
        &series,
        nz(m),
        nz(2),
        None,
        None,
        Single { right: None },
        Vec::new(),
    );
    let (three, _) = left_rows(
        &series,
        nz(m),
        nz(2),
        None,
        None,
        TopK { k: 3, right: None },
        Vec::new(),
    );
    for (row, neighbor) in single {
        let first = three.iter().find(|(r, _)| *r == row).unwrap().1;
        assert_eq!(first, neighbor);
    }
}
