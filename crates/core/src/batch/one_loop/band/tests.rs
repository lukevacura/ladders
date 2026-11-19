//! The band driver's pieces: the layout's shape, the cells and columns of a range, and the lane
//! catch-up.

#[cfg(not(any(loom, shuttle)))]
use super::catch_up;
use super::{LEAF, Layout, RANGE_SLOTS, columns_of, valid_cells};
#[cfg(not(any(loom, shuttle)))]
use crate::batch::one_loop::Counts;
#[cfg(not(any(loom, shuttle)))]
use crate::batch::one_loop::store::Carry;
#[cfg(not(any(loom, shuttle)))]
use crate::cross_term::CrossTerms;
#[cfg(not(any(loom, shuttle)))]
use crate::route::{CatchUpForce, RouteForce, forced_route};
#[cfg(not(any(loom, shuttle)))]
use alloc::vec::Vec;

#[test]
fn a_layout_is_a_power_of_two_band_in_whole_tiles_and_chunks() {
    for band in [1, 8, 16, 24, 48, 64, 100, 1000, 4096, 5000] {
        for (chunk, tile) in [(1, 1), (24, 48), (64, 256), (1000, 1000)] {
            for windows in [1, 9, 100, 5000] {
                let l = Layout::new((band, chunk, tile), windows);
                assert!(l.side.is_power_of_two() && l.side >= LEAF, "{l:?}");
                assert!(
                    l.side <= band.max(LEAF),
                    "{l:?}: no taller than the plan's band"
                );
                assert_eq!(l.tile_lanes % l.side, 0, "{l:?}");
                assert_eq!(l.chunk % LEAF, 0, "{l:?}");
                assert!(l.chunk <= l.tile_lanes, "{l:?}");
            }
        }
    }
}

/// The range is a whole number of the ladder's grains, so no range is widened.
#[test]
fn a_range_is_whole_grains_of_the_coarsest_layer() {
    use crate::batch::one_loop::dense::{GRAIN, SLOTS};
    assert_eq!(RANGE_SLOTS % (SLOTS[0] * GRAIN), 0);
}

/// `valid_cells` against a cell-by-cell count of the pairs (lag at most the row), and `columns_of`
/// holds the column of every one of them.
#[test]
fn the_valid_cells_and_columns_of_a_range_match_an_enumeration() {
    for exclusion in [0usize, 1, 3, 8] {
        let lag = exclusion + 1;
        for rows in [0..0, 3..9, 10..27, 20..21, 0..40] {
            for slots in [0..0, 0..5, 4..19, 18..40, 30..31] {
                let mut count = 0;
                let (mut lo, mut hi) = (usize::MAX, 0);
                for row in rows.clone() {
                    for slot in slots.clone() {
                        if row >= lag + slot {
                            count += 1;
                            let column = row - lag - slot;
                            lo = lo.min(column);
                            hi = hi.max(column + 1);
                        }
                    }
                }
                assert_eq!(
                    valid_cells(exclusion, &rows, &slots),
                    count,
                    "{rows:?} {slots:?}"
                );
                let columns = columns_of(exclusion, &rows, &slots, 1000);
                if count > 0 {
                    assert!(
                        columns.start <= lo && hi <= columns.end,
                        "{columns:?} {lo}..{hi}"
                    );
                }
                let clipped = columns_of(exclusion, &rows, &slots, 7);
                assert!(clipped.end <= 7 && clipped.start <= clipped.end);
            }
        }
    }
}

/// I3: a lane caught up through a gap (stepped or resumed, whichever the rule or the force picks)
/// holds the very state a fresh exact start at the same row holds, for lanes that never started,
/// lanes a row behind and lanes many rows behind.
#[cfg(not(any(loom, shuttle)))]
#[test]
fn a_caught_up_lane_equals_a_fresh_exact_start() {
    let mut x = 0x2545_f491_4f6c_dd1du64;
    let ticks: Vec<i32> = (0..96)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % 2001) as i32 - 1000
        })
        .collect();
    let (m, lag, qa, top) = (8usize, 3usize, 0usize, 30usize);
    let pairs = CrossTerms::of(&ticks[..], m);
    for force in [CatchUpForce::Rule, CatchUpForce::Step, CatchUpForce::Resume] {
        let _guard = forced_route(RouteForce {
            catch_up: force,
            ..RouteForce::default()
        });
        for (r0, r1) in [(10, 11), (12, 20), (16, 40), (33, 34), (20, 60)] {
            let mut counts = Counts::default();
            let mut fresh = Carry::try_new(top).unwrap();
            let live = |row: usize| 0..top.min(row - lag);
            catch_up(
                &pairs,
                &mut fresh,
                (qa, top, lag),
                (live(r1), r1),
                &mut counts,
            );
            let mut walked = Carry::try_new(top).unwrap();
            catch_up(
                &pairs,
                &mut walked,
                (qa, top, lag),
                (live(r0), r0),
                &mut counts,
            );
            catch_up(
                &pairs,
                &mut walked,
                (qa, top, lag),
                (live(r1), r1),
                &mut counts,
            );
            for s in live(r1) {
                assert_eq!(walked.through[s - qa], r1);
                assert_eq!(
                    walked.qt[top - 1 - s],
                    fresh.qt[top - 1 - s],
                    "{force:?} rows {r0}->{r1} slot {s}"
                );
            }
        }
    }
}
