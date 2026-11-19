extern crate std;

use super::{
    ColumnMins, Fit, Lanes, Nearest, Pass, ascending, ascending_scalar, descending,
    descending_scalar,
};
use crate::nonnorm::TickStep;
use proptest::prelude::*;
use std::vec::Vec;

const PASSES: [Pass; 3] = [Pass::Advance, Pass::Rank, Pass::AdvanceAndRank];

/// One row's inputs: the row's own two ticks, each lane's two ticks, completeness and residue,
/// and each lane's column's held residue and row.
#[derive(Clone, Debug)]
struct Row {
    step: TickStep,
    enter: Vec<i64>,
    leave: Vec<i64>,
    complete: Vec<bool>,
    residues: Vec<u64>,
    held: Vec<u64>,
    held_rows: Vec<u64>,
}

impl Row {
    fn lanes(&self) -> Lanes<'_> {
        Lanes {
            enter: &self.enter,
            leave: &self.leave,
            complete: &self.complete,
        }
    }
}

/// A tick `base + offset` with the offset at either end of `0..=span` or anywhere between.
fn tick(base: i64, span: i64) -> impl Strategy<Value = i64> {
    prop_oneof![Just(0), Just(span), 0..=span].prop_map(move |offset| base + offset)
}

/// A residue at the ends of `u64`, a few small values that tie, or anywhere.
fn residue() -> impl Strategy<Value = u64> {
    prop_oneof![Just(0), Just(u64::MAX - 1), 0u64..4, any::<u64>()]
}

/// Rows whose ticks all lie within `span` of a base anywhere in `i64`, with `span` up to the
/// widest [`Fit::I32`] allows.
fn rows(max_lanes: usize) -> impl Strategy<Value = Row> {
    let span = prop_oneof![
        Just(i64::from(i32::MAX)),
        Just(3i64),
        0..=i64::from(i32::MAX)
    ];
    let base = i64::MIN..=i64::MAX - i64::from(i32::MAX);
    (base, span, 0..=max_lanes).prop_flat_map(|(base, span, len)| {
        (
            (tick(base, span), tick(base, span)),
            prop::collection::vec(tick(base, span), len),
            prop::collection::vec(tick(base, span), len),
            prop::collection::vec(prop::bool::weighted(0.85), len),
            prop::collection::vec(residue(), len),
            prop::collection::vec(residue(), len),
            prop::collection::vec(any::<u64>(), len),
        )
            .prop_map(
                |((entering, leaving), enter, leave, complete, residues, held, held_rows)| Row {
                    step: TickStep { entering, leaving },
                    enter,
                    leave,
                    complete,
                    residues,
                    held,
                    held_rows,
                },
            )
    })
}

/// The residue `NonNormalized<u64>` steps to, written out from the recurrence.
fn stepped(residue: u64, row: TickStep, enter: i64, leave: i64) -> u64 {
    let square = |a: i64, b: i64| {
        let d = a.wrapping_sub(b) as u64;
        d.wrapping_mul(d)
    };
    residue
        .wrapping_sub(square(row.leaving, leave))
        .wrapping_add(square(row.entering, enter))
}

/// What a row must produce, lane `k` read at position `at(k)`: the stepped residues and the
/// nearest complete lane, ties to the lane `prefer` picks.
fn reference(
    row: &Row,
    pass: Pass,
    at: impl Fn(usize) -> usize,
    prefer: impl Fn(usize, usize) -> bool,
) -> (Vec<u64>, Nearest) {
    let advance = matches!(pass, Pass::Advance | Pass::AdvanceAndRank);
    let residues: Vec<u64> = (0..row.residues.len())
        .map(|k| {
            if advance {
                stepped(
                    row.residues[k],
                    row.step,
                    row.enter[at(k)],
                    row.leave[at(k)],
                )
            } else {
                row.residues[k]
            }
        })
        .collect();
    let mut best: Nearest = None;
    if matches!(pass, Pass::Rank | Pass::AdvanceAndRank) {
        for (k, &residue) in residues.iter().enumerate() {
            if !row.complete[at(k)] {
                continue;
            }
            best = match best {
                Some((held, lane)) if held < residue || (held == residue && prefer(lane, k)) => {
                    Some((held, lane))
                }
                _ => Some((residue, k)),
            };
        }
    }
    (residues, best)
}

fn run_ascending(row: &Row, pass: Pass, columns: bool, scalar: bool) -> (Row, Nearest) {
    let mut out = row.clone();
    let lanes = Lanes {
        enter: &row.enter,
        leave: &row.leave,
        complete: &row.complete,
    };
    let mut mins = columns.then(|| ColumnMins {
        squared: &mut out.held,
        rows: &mut out.held_rows,
        row: 7,
    });
    let found = if scalar {
        ascending_scalar(&mut out.residues, row.step, lanes, pass, mins.as_mut())
    } else {
        ascending(
            &mut out.residues,
            row.step,
            lanes,
            Fit::I32,
            pass,
            mins.as_mut(),
        )
    };
    (out, found)
}

fn run_descending(row: &Row, pass: Pass, high: u64, scalar: bool) -> (Vec<u128>, Nearest) {
    let mut residues: Vec<u128> = row
        .residues
        .iter()
        .map(|&r| (u128::from(high) << 64) | u128::from(r))
        .collect();
    let found = if scalar {
        descending_scalar(&mut residues, row.step, row.lanes(), pass)
    } else {
        descending(&mut residues, row.step, row.lanes(), Fit::I32, pass)
    };
    (residues, found)
}

/// The columns a ranking row leaves: a complete lane below its column's residue takes it.
fn columns_reference(row: &Row, residues: &[u64]) -> (Vec<u64>, Vec<u64>) {
    let (mut held, mut rows) = (row.held.clone(), row.held_rows.clone());
    for k in 0..residues.len() {
        if row.complete[k] && residues[k] < held[k] {
            held[k] = residues[k];
            rows[k] = 7;
        }
    }
    (held, rows)
}

fn check_ascending(row: &Row) -> Result<(), TestCaseError> {
    for pass in PASSES {
        let expected = reference(row, pass, |k| k, |held, lane| held < lane);
        for columns in [false, true] {
            for scalar in [false, true] {
                let (out, found) = run_ascending(row, pass, columns, scalar);
                prop_assert_eq!(&out.residues, &expected.0, "{:?} {}", pass, scalar);
                prop_assert_eq!(found, expected.1, "{:?} {}", pass, scalar);
                let held = if columns && pass != Pass::Advance {
                    columns_reference(row, &expected.0)
                } else {
                    (row.held.clone(), row.held_rows.clone())
                };
                prop_assert_eq!((out.held, out.held_rows), held, "{:?} {}", pass, scalar);
            }
        }
    }
    Ok(())
}

fn check_descending(row: &Row, high: u64) -> Result<(), TestCaseError> {
    let len = row.residues.len();
    for pass in PASSES {
        let expected = reference(row, pass, |k| len - 1 - k, |held, lane| held > lane);
        for scalar in [false, true] {
            let (residues, found) = run_descending(row, pass, high, scalar);
            let lows: Vec<u64> = residues.iter().map(|&r| r as u64).collect();
            prop_assert_eq!(&lows, &expected.0, "{:?} {}", pass, scalar);
            prop_assert_eq!(found, expected.1, "{:?} {}", pass, scalar);
            // A stepped diagonal clears its high half; one left alone keeps it.
            let high_kept = if pass == Pass::Rank { high } else { 0 };
            prop_assert!(residues.iter().all(|&r| (r >> 64) as u64 == high_kept));
        }
    }
    Ok(())
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]

    #[test]
    fn ascending_lanes_equal_the_recurrence_and_the_scalar_body(row in rows(23)) {
        check_ascending(&row)?;
    }

    #[test]
    fn descending_lanes_equal_the_recurrence_and_the_scalar_body(
        row in rows(23),
        high in prop_oneof![Just(0u64), any::<u64>()],
    ) {
        check_descending(&row, high)?;
    }

    /// Every lane at one residue: the nearest is the first lane ascending, the last descending.
    #[test]
    fn ties_go_to_the_neighbor_that_starts_first(
        len in 1usize..20,
        value in residue(),
        incomplete in prop::collection::vec(any::<bool>(), 20),
    ) {
        let complete: Vec<bool> = incomplete[..len].iter().map(|&i| !i).collect();
        let row = Row {
            step: TickStep { entering: 5, leaving: 5 },
            enter: std::vec![5; len],
            leave: std::vec![5; len],
            complete: complete.clone(),
            residues: std::vec![value; len],
            held: std::vec![u64::MAX; len],
            held_rows: std::vec![0; len],
        };
        let first = complete.iter().position(|&c| c);
        let (_, found) = run_ascending(&row, Pass::AdvanceAndRank, false, false);
        prop_assert_eq!(found, first.map(|lane| (value, lane)));
        let last = complete.iter().position(|&c| c).map(|at| len - 1 - at);
        let (_, found) = run_descending(&row, Pass::AdvanceAndRank, 0, false);
        prop_assert_eq!(found, last.map(|lane| (value, lane)));
    }
}

/// The widest differences [`Fit::I32`] allows, both signs, around both ends of `i64`.
#[test]
fn the_widest_differences_square_exactly() {
    let span = i64::from(i32::MAX);
    for base in [i64::MIN, -span, 0, i64::MAX - span] {
        let (lo, hi) = (base, base + span);
        let row = Row {
            step: TickStep {
                entering: lo,
                leaving: hi,
            },
            enter: [hi, lo, hi, lo, hi, hi, lo].to_vec(),
            leave: [lo, hi, lo, lo, hi, lo, hi].to_vec(),
            complete: [true; 7].to_vec(),
            residues: [0, u64::MAX, 1 << 63, 5, 0, 1, 2].to_vec(),
            held: [u64::MAX; 7].to_vec(),
            held_rows: [0; 7].to_vec(),
        };
        check_ascending(&row).unwrap();
        check_descending(&row, 1).unwrap();
        assert_eq!(Fit::of(lo, hi), Fit::I32);
    }
}

#[test]
fn no_complete_lane_has_no_nearest() {
    let row = Row {
        step: TickStep {
            entering: 1,
            leaving: 2,
        },
        enter: [3; 9].to_vec(),
        leave: [4; 9].to_vec(),
        complete: [false; 9].to_vec(),
        residues: [0; 9].to_vec(),
        held: [u64::MAX; 9].to_vec(),
        held_rows: [0; 9].to_vec(),
    };
    assert_eq!(
        run_ascending(&row, Pass::AdvanceAndRank, true, false).1,
        None
    );
    assert_eq!(run_descending(&row, Pass::AdvanceAndRank, 0, false).1, None);
    assert_eq!(run_ascending(&row, Pass::Advance, false, false).1, None);
}

#[test]
fn fit_holds_up_to_a_difference_of_two_to_the_31_minus_one() {
    let max = i64::from(i32::MAX);
    assert_eq!(Fit::of(0, max), Fit::I32);
    assert_eq!(Fit::of(-1, max), Fit::U64);
    assert_eq!(Fit::of(i64::MIN, i64::MIN + max), Fit::I32);
    assert_eq!(Fit::of(i64::MIN, i64::MIN + max + 1), Fit::U64);
    assert_eq!(Fit::of(i64::MIN, i64::MAX), Fit::U64);
    assert_eq!(Fit::of(i64::MAX, i64::MIN), Fit::I32);
    assert_eq!(Fit::of(7, 7), Fit::I32);
}

/// Differences past `i32` take the scalar body, which wraps exactly as the recurrence does.
#[test]
fn differences_past_i32_still_equal_the_recurrence() {
    let far = 1i64 << 40;
    let row = Row {
        step: TickStep {
            entering: -far,
            leaving: far,
        },
        enter: (0..9).map(|k| far - k).collect(),
        leave: (0..9).map(|k| k * far / 3).collect(),
        complete: [true, false, true, true, true, true, false, true, true].to_vec(),
        residues: (0..9).map(|k| k * 77).collect(),
        held: [u64::MAX; 9].to_vec(),
        held_rows: [0; 9].to_vec(),
    };
    for pass in PASSES {
        let expected = reference(&row, pass, |k| k, |held, lane| held < lane);
        let mut residues = row.residues.clone();
        let found = ascending(&mut residues, row.step, row.lanes(), Fit::U64, pass, None);
        assert_eq!((residues, found), expected, "{pass:?}");
        let expected = reference(&row, pass, |k| 8 - k, |held, lane| held > lane);
        let mut residues: Vec<u128> = row.residues.iter().map(|&r| u128::from(r)).collect();
        let found = descending(&mut residues, row.step, row.lanes(), Fit::U64, pass);
        let lows: Vec<u64> = residues.iter().map(|&r| r as u64).collect();
        assert_eq!((lows, found), expected, "{pass:?}");
    }
}

/// The AVX2 kernel called directly, against the scalar body, wherever the CPU has AVX2.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use crate::kernel_avx2::detected;

    fn here() -> bool {
        if !detected() {
            std::eprintln!("skipped: this CPU has no AVX2, so the AVX2 kernel is not compared");
        }
        detected()
    }

    fn ascending_avx2(row: &Row, pass: Pass, columns: bool) -> (Row, Nearest) {
        let mut out = row.clone();
        let mut mins = columns.then(|| ColumnMins {
            squared: &mut out.held,
            rows: &mut out.held_rows,
            row: 7,
        });
        // SAFETY: AVX2 detected by the caller; every slice holds one element per lane and the
        // rows' ticks span at most `i32::MAX`.
        let found = unsafe {
            super::super::avx2::row::<[u64], false>(
                &mut out.residues,
                row.step,
                row.lanes(),
                pass,
                mins.as_mut(),
            )
        };
        (out, found)
    }

    fn descending_avx2(row: &Row, pass: Pass, high: u64) -> (Vec<u128>, Nearest) {
        let mut residues: Vec<u128> = row
            .residues
            .iter()
            .map(|&r| (u128::from(high) << 64) | u128::from(r))
            .collect();
        // SAFETY: as `ascending_avx2`.
        let found = unsafe {
            super::super::avx2::row::<[u128], true>(
                &mut residues,
                row.step,
                row.lanes(),
                pass,
                None,
            )
        };
        (residues, found)
    }

    fn check(row: &Row, high: u64) -> Result<(), TestCaseError> {
        for pass in PASSES {
            for columns in [false, true] {
                let (vector, found) = ascending_avx2(row, pass, columns);
                let (scalar, expected) = run_ascending(row, pass, columns, true);
                prop_assert_eq!(&vector.residues, &scalar.residues, "{:?}", pass);
                prop_assert_eq!(found, expected, "{:?}", pass);
                prop_assert_eq!(&vector.held, &scalar.held, "{:?}", pass);
                prop_assert_eq!(&vector.held_rows, &scalar.held_rows, "{:?}", pass);
            }
            let (vector, found) = descending_avx2(row, pass, high);
            let (scalar, expected) = run_descending(row, pass, high, true);
            prop_assert_eq!(vector, scalar, "{:?}", pass);
            prop_assert_eq!(found, expected, "{:?}", pass);
        }
        Ok(())
    }

    proptest! {
        #![proptest_config(testkit::generators::seeded_default())]

        #[test]
        fn avx2_lanes_equal_the_scalar_body(
            row in rows(70),
            high in prop_oneof![Just(0u64), any::<u64>()],
        ) {
            if here() {
                check(&row, high)?;
            }
        }
    }

    /// Every lane count and every tie: all residues equal, so the winner is decided by lane
    /// order alone, with some lanes incomplete.
    #[test]
    fn ties_and_remainders_equal_the_scalar_body() {
        if !here() {
            return;
        }
        for len in 0..=40usize {
            for gap in 1..5 {
                let row = Row {
                    step: TickStep {
                        entering: 5,
                        leaving: 5,
                    },
                    enter: std::vec![5; len],
                    leave: std::vec![5; len],
                    complete: (0..len).map(|k| k % gap != 0 || k % 7 == 3).collect(),
                    residues: std::vec![9; len],
                    held: (0..len as u64).map(|k| k % 3 * 9).collect(),
                    held_rows: std::vec![1; len],
                };
                check(&row, 3).unwrap();
            }
        }
    }
}
