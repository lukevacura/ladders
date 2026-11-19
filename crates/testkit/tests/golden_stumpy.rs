//! The exact reference's left profile against STUMPY's own batch `stump` output, on every
//! golden fixture. STUMPY's kernel is JIT-compiled with fastmath (may contract to FMA), so
//! distance parity is tolerance-based, never bitwise; an index mismatch is accepted only when
//! the two neighbors are an exact tie, computed from the ticks in exact integer arithmetic,
//! since STUMPY's own pick among tied neighbors depends on its thread count. A near-miss
//! within the distance tolerance is not a tie, and still fails.
//!
//! STUMPY's own known defect (a NaN gap at a high price level corrupts its running covariance
//! for the rest of that diagonal; see the private write-up the orchestrator holds) needs both
//! a gap and a high offset together. No fixture here combines them (`nonfinite_gaps` has gaps
//! but no offset; `random_walk_offset_1e8` has the offset but no gaps), so every row below is
//! compared, with no fixture excepted.

use ladde::naive::naive_left_profile;
use ladde::quantize::{Scale, to_ticks};
use num_bigint::{BigInt, Sign};
use testkit::golden::{Fixture, load, names};

/// The finest power-of-two tick scale that still keeps the largest fixture offset (about 1e8,
/// in `random_walk_offset_1e8`) under `MAX_TICKS` (2^53) with headroom: quantizing at this
/// scale moves a sample by at most half a step, about 7.5e-9 absolute, far below either
/// tolerance below, so it adds no detectable error of its own.
const SCALE: f64 = 1.0 / (1u64 << 26) as f64;

/// STUMPY's own `left_P` error against an exact (mpmath) oracle, measured on these fixtures:
/// about 3.3e-7 absolute at offset 1e8 (2.3e-7 relative), and 1.3e-7 absolute at an
/// exact-zero pair in `flat_segments` (STUMPY's own cancellation near zero). Both constants
/// below hold a several-fold margin over those measurements, fixed before this test ran.
const ABSOLUTE_TOLERANCE: f64 = 1e-6;
const RELATIVE_TOLERANCE: f64 = 1e-6;

/// `true` if `ours` and `stumpy`'s distance agree within tolerance (relative at large
/// offsets, absolute near 0), or both are the "no neighbor" sentinel (`+inf`).
fn distances_agree(ours: f64, stumpy: f64) -> bool {
    match (ours.is_finite(), stumpy.is_finite()) {
        (false, false) => ours == stumpy,
        (true, true) => {
            (ours - stumpy).abs() <= ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * stumpy.abs()
        }
        _ => false,
    }
}

/// A window's exact classification from ticks, independent of `naive`'s own types: a missing
/// tick makes it a `Gap`, equal ticks make it `Constant`, otherwise its exact sum and exact
/// `m·Σx² − (Σx)²` as arbitrary-precision integers (no range limit: this is a tie check, not
/// the library's own exclusion rule).
enum ExactWindow {
    Gap,
    Constant,
    Valid {
        sum: BigInt,
        variance_numerator: BigInt,
    },
}

fn exact_window(ticks: &[Option<i64>], start: usize, m: usize) -> ExactWindow {
    let window = &ticks[start..start + m];
    if window.iter().any(Option::is_none) {
        return ExactWindow::Gap;
    }
    let values: Vec<BigInt> = window
        .iter()
        .map(|tick| BigInt::from(tick.expect("checked above")))
        .collect();
    if values.iter().min() == values.iter().max() {
        return ExactWindow::Constant;
    }
    let sum: BigInt = values.iter().sum();
    let sum_sq: BigInt = values.iter().map(|v| v * v).sum();
    let variance_numerator = BigInt::from(m) * sum_sq - &sum * &sum;
    ExactWindow::Valid {
        sum,
        variance_numerator,
    }
}

/// `Σ_k ticks[row+k]·ticks[neighbor+k]`, exact; both windows are already known to hold ticks
/// (checked by the caller through [`exact_window`]).
fn exact_cross_term(ticks: &[Option<i64>], row: usize, neighbor: usize, m: usize) -> BigInt {
    (0..m)
        .map(|k| {
            let a = ticks[row + k].expect("gap windows are filtered before this is called");
            let b = ticks[neighbor + k].expect("gap windows are filtered before this is called");
            BigInt::from(a) * BigInt::from(b)
        })
        .sum()
}

/// A pair's exact `ρ`, kept as a signed rational rather than rounded to `f64`: `One` and `Half`
/// are the fixed constant-pair values, `Signed` is `n / √denominator` for a valid-valid pair
/// (ties going through `n`'s sign, same as the library's own exact comparison).
enum ExactRho {
    One,
    Half,
    Signed { n: BigInt, denominator: BigInt },
}

/// The pair's exact `ρ` at row-start `row` and neighbor-start `neighbor`, or `None` if either
/// window has a missing tick.
fn exact_rho(ticks: &[Option<i64>], m: usize, row: usize, neighbor: usize) -> Option<ExactRho> {
    Some(
        match (
            exact_window(ticks, row, m),
            exact_window(ticks, neighbor, m),
        ) {
            (ExactWindow::Gap, _) | (_, ExactWindow::Gap) => return None,
            (ExactWindow::Constant, ExactWindow::Constant) => ExactRho::One,
            (ExactWindow::Constant, ExactWindow::Valid { .. })
            | (ExactWindow::Valid { .. }, ExactWindow::Constant) => ExactRho::Half,
            (
                ExactWindow::Valid {
                    sum: row_sum,
                    variance_numerator: row_variance,
                },
                ExactWindow::Valid {
                    sum: neighbor_sum,
                    variance_numerator: neighbor_variance,
                },
            ) => {
                let qt = exact_cross_term(ticks, row, neighbor, m);
                let n = BigInt::from(m) * qt - row_sum * neighbor_sum;
                ExactRho::Signed {
                    n,
                    denominator: row_variance * neighbor_variance,
                }
            }
        },
    )
}

/// `true` if two pairs' `ρ` are exactly equal. `Half` ties a `Signed` pair exactly when that
/// pair's own `ρ` is also `1/2`: positive `n`, and `4·n² == denominator`. Two `Signed` pairs
/// tie when they share `n`'s sign and `n²` cross-multiplied against the other's denominator
/// agrees — `ρ`'s own squared ratio, decided without ever taking the irrational square root.
fn exact_rho_eq(a: &ExactRho, b: &ExactRho) -> bool {
    match (a, b) {
        (ExactRho::One, ExactRho::One) | (ExactRho::Half, ExactRho::Half) => true,
        (ExactRho::Half, ExactRho::Signed { n, denominator })
        | (ExactRho::Signed { n, denominator }, ExactRho::Half) => {
            n.sign() == Sign::Plus && n * n * BigInt::from(4) == *denominator
        }
        (
            ExactRho::Signed {
                n: na,
                denominator: da,
            },
            ExactRho::Signed {
                n: nb,
                denominator: db,
            },
        ) => na.sign() == nb.sign() && na * na * db == nb * nb * da,
        // `One` is both windows constant, `Signed` is neither; a row's own kind is fixed for
        // both sides of one comparison, so the two can never arise against the same row.
        _ => false,
    }
}

/// A `Signed` pair's `ρ`, as `f64` — only for this file's own tests, to build a synthetic
/// STUMPY distance that agrees with the true one within tolerance without being an exact tie;
/// the exact comparison above never leaves integers.
fn signed_rho_f64(n: &BigInt, denominator: &BigInt) -> f64 {
    let n: f64 = n.to_string().parse().expect("fits in f64");
    let denominator: f64 = denominator.to_string().parse().expect("fits in f64");
    n / denominator.sqrt()
}

/// Every row's distance must agree; a row's index may differ from STUMPY's only when the two
/// neighbors are an exact tie (same `ρ`, computed from the ticks in exact integer arithmetic),
/// never merely because the two distances agree within tolerance. Returns the largest absolute
/// distance gap and the number of accepted ties, for the caller to report.
fn compare_left_profile(fixture: &Fixture) -> (f64, usize) {
    let scale = Scale::new(SCALE).expect("SCALE is positive and finite");
    let ticks: Vec<Option<i64>> = fixture
        .series
        .iter()
        .map(|&value| to_ticks(value, scale))
        .collect();
    let ours = naive_left_profile(&ticks, fixture.m);
    assert_eq!(
        ours.len(),
        fixture.left_p.len(),
        "{}: row count",
        fixture.name
    );

    let mut max_abs_delta = 0.0_f64;
    let mut ties = 0usize;
    for (k, &(row, entry)) in ours.iter().enumerate() {
        assert_eq!(
            row, k as u64,
            "{}: rows are returned in order",
            fixture.name
        );
        let our_p = entry.squared_distance().sqrt();
        let stumpy_p = fixture.left_p[k];
        assert!(
            distances_agree(our_p, stumpy_p),
            "{}[{k}]: our P {our_p} vs stumpy's {stumpy_p}, beyond tolerance",
            fixture.name
        );
        if our_p.is_finite() {
            max_abs_delta = max_abs_delta.max((our_p - stumpy_p).abs());
        }

        let our_index = entry.index().get();
        let stumpy_index = (fixture.left_i[k] != u64::MAX).then_some(fixture.left_i[k]);
        if our_index != stumpy_index {
            // A sentinel ever disagreeing with a real index would already have failed the
            // distance assertion above (`+inf` only agrees with `+inf`), so both are real here.
            let ours = our_index.expect("sentinel mismatches already failed distances_agree");
            let stumpys = stumpy_index.expect("sentinel mismatches already failed distances_agree");
            let our_rho = exact_rho(&ticks, fixture.m, k, ours as usize)
                .expect("our own chosen neighbor's window holds no gap under our own ticks");
            let stumpy_rho = exact_rho(&ticks, fixture.m, k, stumpys as usize);
            let is_exact_tie =
                stumpy_rho.is_some_and(|stumpy_rho| exact_rho_eq(&our_rho, &stumpy_rho));
            assert!(
                is_exact_tie,
                "{}[{k}]: index differs (ours {ours}, stumpy's {stumpys}) but isn't an exact tie",
                fixture.name
            );
            ties += 1;
        }
    }
    (max_abs_delta, ties)
}

#[test]
fn golden_stumpy() {
    let fixture_names = names();
    assert!(!fixture_names.is_empty(), "no golden fixtures committed");

    let mut checked = 0usize;
    for name in &fixture_names {
        let fixture = load(name);
        let (max_abs_delta, ties) = compare_left_profile(&fixture);
        // Diagnostic only, not the assertions above: the per-fixture numbers this node's
        // report back cites.
        eprintln!("{name}: max|dP| {max_abs_delta:e}, {ties} accepted tie(s)");
        checked += 1;
    }
    assert_eq!(
        checked,
        fixture_names.len(),
        "every committed fixture was compared, none skipped"
    );
}

/// `nb1` and `nb2` are both one-tick-off near matches of `row` (same series `naive`'s own
/// `ranking_tolerance_uses_the_documented_epsilon_not_a_tighter_one` test pins): close enough
/// that `nb2`'s distance to `row` sits within `ABSOLUTE_TOLERANCE`/`RELATIVE_TOLERANCE` of
/// `nb1`'s, yet `nb1` is the true nearer neighbor, not a tie. A fixture whose STUMPY columns
/// claim `nb2`'s index at that row, with `nb2`'s own (not `nb1`'s) distance, is exactly the
/// mutant this test must still catch end to end: `distances_agree` alone would accept it, so
/// only `compare_left_profile`'s exact-tie check, exercised here and not just its comparator
/// functions, protects against it.
#[test]
#[should_panic(expected = "isn't an exact tie")]
fn compare_left_profile_rejects_a_near_miss_disguised_as_stumpys_pick() {
    let nb1: [i64; 3] = [-802_959_335, 909_343_706, 273_972_096];
    let nb2: [i64; 3] = [-802_959_340, 909_343_706, 273_972_096];
    let row: [i64; 3] = [-802_959_337, 909_343_706, 273_972_096];
    let ticks: Vec<i64> = nb1.iter().chain(&nb2).chain(&row).copied().collect();
    let as_ticks: Vec<Option<i64>> = ticks.iter().copied().map(Some).collect();

    let true_rho = exact_rho(&as_ticks, 3, 6, 0).expect("nb1 and row are both valid windows");
    let near_miss_rho = exact_rho(&as_ticks, 3, 6, 3).expect("nb2 and row are both valid windows");
    assert!(
        !exact_rho_eq(&true_rho, &near_miss_rho),
        "nb2 is a near miss, not an exact tie, with nb1"
    );
    let ExactRho::Signed { n, denominator } = &near_miss_rho else {
        panic!("row and nb2 are both valid, non-constant windows");
    };
    let near_miss_distance = (2.0 * 3.0 * (1.0 - signed_rho_f64(n, denominator))).sqrt();

    // `ours`, not hand-picked, for every row: the only deliberate change below is row 6's
    // index and reported distance, so every other row trivially agrees with itself.
    let ours = naive_left_profile(&as_ticks, 3);
    let mut left_p: Vec<f64> = ours
        .iter()
        .map(|&(_, entry)| entry.squared_distance().sqrt())
        .collect();
    let mut left_i: Vec<u64> = ours
        .iter()
        .map(|&(_, entry)| entry.index().get().unwrap_or(u64::MAX))
        .collect();
    assert_eq!(
        left_i[6], 0,
        "nb1 (index 0) is the true winner before the perturbation"
    );
    left_p[6] = near_miss_distance;
    left_i[6] = 3;

    let fixture = Fixture {
        name: "synthetic_near_miss".to_owned(),
        m: 3,
        series: ticks.iter().map(|&tick| tick as f64 * SCALE).collect(),
        left_p: left_p.clone(),
        left_i: left_i.clone(),
        p: left_p,
        i: left_i,
    };
    compare_left_profile(&fixture);
}

/// A STUMPY index pointing at a window that itself has a missing tick is never a tie: there is
/// no `ρ` to compare at all. If `compare_left_profile` read `stumpy_rho.map_or(true, …)`
/// instead of `.is_some_and(…)`, nothing else ever drives `stumpy_rho` to `None` at a
/// mismatched row, so this would default to "tied" and pass every other test silently.
#[test]
#[should_panic(expected = "isn't an exact tie")]
fn compare_left_profile_rejects_a_stumpy_index_whose_window_has_a_gap() {
    let gap_at = 2;
    let ticks: [i64; 9] = [10, 20, 0, 40, 50, 60, 70, 80, 90]; // ticks[gap_at] is unused
    let as_ticks: Vec<Option<i64>> = ticks
        .iter()
        .enumerate()
        .map(|(i, &tick)| (i != gap_at).then_some(tick))
        .collect();

    // Row 6's window [70,80,90]: windows 0..=2 all touch the gap and are excluded from
    // candidacy, leaving 3 and 4 as the only real neighbors, so the true winner is never 0.
    let ours = naive_left_profile(&as_ticks, 3);
    let left_p: Vec<f64> = ours
        .iter()
        .map(|&(_, entry)| entry.squared_distance().sqrt())
        .collect();
    let mut left_i: Vec<u64> = ours
        .iter()
        .map(|&(_, entry)| entry.index().get().unwrap_or(u64::MAX))
        .collect();
    assert_ne!(
        left_i[6], 0,
        "window 0 touches the gap, so it's never the true winner"
    );
    left_i[6] = 0; // STUMPY's claim: the gap window, with our own (within-tolerance) distance.

    let series: Vec<f64> = ticks
        .iter()
        .enumerate()
        .map(|(i, &tick)| {
            if i == gap_at {
                f64::NAN
            } else {
                tick as f64 * SCALE
            }
        })
        .collect();
    let fixture = Fixture {
        name: "synthetic_gap_stumpy_pick".to_owned(),
        m: 3,
        series,
        left_p: left_p.clone(),
        left_i: left_i.clone(),
        p: left_p,
        i: left_i,
    };
    compare_left_profile(&fixture);
}

/// Dropping the sign check would let equal-magnitude, opposite-sign pairs read as a tie.
#[test]
fn exact_rho_eq_rejects_matching_magnitude_opposite_sign() {
    let positive = ExactRho::Signed {
        n: BigInt::from(1),
        denominator: BigInt::from(4),
    };
    let negative = ExactRho::Signed {
        n: BigInt::from(-1),
        denominator: BigInt::from(4),
    };
    assert!(
        !exact_rho_eq(&positive, &negative),
        "opposite signs are never an exact tie, however close the magnitude"
    );
}

/// Cross-multiplying `n` and `denominator` unsquared, instead of `n²` against `denominator`,
/// gets both cases below wrong: `(1,4)` vs `(2,16)` (both `ρ = 1/2`) unsquared to `1·16 ≠
/// 2·4`, a false non-tie; `(1,4)` vs `(2,8)` (`ρ = 1/2` vs `ρ = 1/√2`) unsquared to `1·8 =
/// 2·4`, a false tie.
#[test]
fn exact_rho_eq_compares_the_squared_ratio_not_the_raw_cross_multiply() {
    let half = ExactRho::Signed {
        n: BigInt::from(1),
        denominator: BigInt::from(4),
    };
    let same_ratio = ExactRho::Signed {
        n: BigInt::from(2),
        denominator: BigInt::from(16),
    };
    let different_ratio = ExactRho::Signed {
        n: BigInt::from(2),
        denominator: BigInt::from(8),
    };
    assert!(
        exact_rho_eq(&half, &same_ratio),
        "1/4 and 4/16 are the same rho-squared"
    );
    assert!(
        !exact_rho_eq(&half, &different_ratio),
        "1/4 and 4/8 are not the same rho-squared"
    );
}

/// `row = (-4,-4,-3)` against a constant neighbor `(9,9,9)` gives `Half` (`ρ = 1/2` fixed);
/// against the valid neighbor `(-4,-3,-3)` it gives `Signed { n: 1, denominator: 4 }`, whose
/// own `ρ` is also exactly `1/2` — the library's own comparator treats these as the same
/// distance (both `√m`), so the exact-tie check must too.
#[test]
fn exact_rho_eq_ties_a_one_constant_pair_with_a_signed_pair_at_rho_half() {
    let constant_neighbor: [i64; 3] = [9, 9, 9];
    let valid_neighbor: [i64; 3] = [-4, -3, -3];
    let row: [i64; 3] = [-4, -4, -3];
    let ticks: Vec<Option<i64>> = constant_neighbor
        .iter()
        .chain(&valid_neighbor)
        .chain(&row)
        .copied()
        .map(Some)
        .collect();

    let half = exact_rho(&ticks, 3, 6, 0).expect("row and the constant neighbor are both valid");
    let signed = exact_rho(&ticks, 3, 6, 3).expect("row and the valid neighbor are both valid");
    assert!(matches!(half, ExactRho::Half));
    let ExactRho::Signed { n, denominator } = &signed else {
        panic!("row and the valid neighbor are a valid-valid pair");
    };
    assert_eq!(*n, BigInt::from(1));
    assert_eq!(*denominator, BigInt::from(4));

    assert!(
        exact_rho_eq(&half, &signed),
        "both give distance sqrt(m): rho = 1/2 either way"
    );
    assert!(exact_rho_eq(&signed, &half), "equality is symmetric");
}

/// A window with a missing tick is never comparable, so it is never an exact tie either.
#[test]
fn exact_rho_returns_none_for_a_window_with_a_gap() {
    let mut ticks: Vec<Option<i64>> = [1i64, 2, 3, 4, 5, 6].into_iter().map(Some).collect();
    ticks[1] = None; // a gap inside the row's own window, ticks[0..3)
    assert!(exact_rho(&ticks, 3, 0, 3).is_none());
}
