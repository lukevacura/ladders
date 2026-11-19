//! The move of lanes between anchors: a moved lane holds the very state a fresh exact start in the
//! new anchors holds, and only a lane current through the band's first row is moved.

use super::{Frame, Lanes, move_lanes};
use crate::batch::one_loop::store::Carry;
use crate::batch::pass::Shape;
use crate::cross_term::UNSTARTED;
use alloc::vec::Vec;

/// `m·QT` of the windows at `row` and `column` of `ticks`, measured from the anchors `frame`
/// (wrapping, as the pass's states are).
fn state(ticks: &[i64], m: usize, (row, column): (usize, usize), frame: Frame) -> i64 {
    let qt = (0..m).fold(0i64, |acc, k| {
        acc.wrapping_add((ticks[row + k] - frame.row).wrapping_mul(ticks[column + k] - frame.col))
    });
    qt.wrapping_mul(m as i64)
}

fn sums(ticks: &[i64], m: usize) -> Vec<i64> {
    (0..=ticks.len() - m)
        .map(|w| ticks[w..w + m].iter().sum())
        .collect()
}

fn series(n: usize, seed: u64) -> Vec<i64> {
    let mut x = seed | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % 20_001) as i64 - 10_000 + 1_000_000
        })
        .collect()
}

/// A carry of the slots `qa..top` whose lanes hold the state of the pair they carry, measured from
/// `from`: through `first_row`, or (for the slots `behind` picks) through the row before it.
fn carry_in(
    ticks: &[i64],
    (m, lag, first_row, qa, top): (usize, usize, usize, usize, usize),
    from: Frame,
    behind: impl Fn(usize) -> bool,
) -> Carry {
    let mut carry = Carry::try_new(top - qa).unwrap();
    carry.frame = from;
    for slot in qa..top.min(first_row - lag) {
        let through = if behind(slot) {
            first_row - 1
        } else {
            first_row
        };
        let row = through - 1;
        carry.through[slot - qa] = through;
        carry.qt[top - 1 - slot] = state(ticks, m, (row, row - lag - slot), from);
    }
    carry
}

#[test]
fn a_moved_lane_holds_the_state_of_a_fresh_start_in_the_new_anchors() {
    let (m, lag) = (8usize, 3usize);
    let ticks = series(120, 0x9E37_79B9);
    let sums = sums(&ticks, m);
    let frames = [
        Frame { row: 0, col: 0 },
        Frame {
            row: 1_000_000,
            col: 1_000_000,
        },
        Frame {
            row: 1_004_321,
            col: 996_000,
        },
        Frame {
            row: -50_000_000,
            col: 77_000_000_000,
        },
    ];
    for &(first_row, qa, top) in &[(40usize, 0usize, 30usize), (33, 5, 37), (90, 20, 60)] {
        for &from in &frames {
            for &to in &frames {
                let slots = qa..top.min(first_row - lag);
                let mut carry =
                    carry_in(&ticks, (m, lag, first_row, qa, top), from, |s| s % 3 == 0);
                let before = carry.qt.clone();
                let at = Lanes {
                    m,
                    shape: Shape::Triangle { exclusion: lag - 1 },
                    first_row,
                    slots: slots.clone(),
                    span: (qa, top),
                    sums: Some((&sums, &sums)),
                };
                let moved = move_lanes(&mut carry, to, &at);
                assert_eq!(carry.frame, to);
                let current = slots.clone().filter(|s| s % 3 != 0).count();
                assert_eq!(moved, if from == to { 0 } else { current });
                for s in slots {
                    let index = top - 1 - s;
                    if s % 3 == 0 || from == to {
                        assert_eq!(carry.qt[index], before[index], "a lane left alone: {s}");
                    } else {
                        let row = first_row - 1;
                        assert_eq!(
                            carry.qt[index],
                            state(&ticks, m, (row, row - lag - s), to),
                            "{from:?} -> {to:?} slot {s} first row {first_row}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn lanes_with_no_sums_at_hand_start_over() {
    let (m, lag, first_row, qa, top) = (8usize, 3usize, 40usize, 0usize, 30usize);
    let ticks = series(120, 7);
    let from = Frame { row: 1, col: 2 };
    let to = Frame { row: 3, col: 4 };
    let mut carry = carry_in(&ticks, (m, lag, first_row, qa, top), from, |_| false);
    let held_sums = sums(&ticks, m);
    // No sums at all, and sums that stop short of the lanes' windows.
    for held in [None, Some(&held_sums[..10])] {
        let mut lost = Carry {
            qt: carry.qt.clone(),
            through: carry.through.clone(),
            frame: carry.frame,
            wide: carry.wide,
        };
        let at = Lanes {
            m,
            shape: Shape::Triangle { exclusion: lag - 1 },
            first_row,
            slots: qa..top.min(first_row - lag),
            span: (qa, top),
            sums: held.map(|held| (held, held)),
        };
        assert_eq!(move_lanes(&mut lost, to, &at), 0);
        assert!(lost.through.iter().all(|&t| t == UNSTARTED));
        assert_eq!(lost.frame, to);
    }
    // The same frame moves nothing, whatever the sums.
    let at = Lanes {
        m,
        shape: Shape::Triangle { exclusion: lag - 1 },
        first_row,
        slots: qa..top.min(first_row - lag),
        span: (qa, top),
        sums: None,
    };
    carry.frame = to;
    let before = (carry.qt.clone(), carry.through.clone());
    assert_eq!(move_lanes(&mut carry, to, &at), 0);
    assert_eq!((carry.qt, carry.through), before);
}
