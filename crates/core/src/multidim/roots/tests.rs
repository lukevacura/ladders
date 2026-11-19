use super::{Terms, vectors};
use crate::nearest::ExactCorrelation;
use core::cmp::Ordering;

fn order(sign: i8) -> Ordering {
    i32::from(sign).cmp(&0)
}

fn correlation(rho_negative: bool, cross: u64, denominator: u128) -> ExactCorrelation {
    ExactCorrelation::Valid {
        sign: match (cross, rho_negative) {
            (0, _) => 0,
            (_, true) => -1,
            (_, false) => 1,
        },
        cross_squared: u128::from(cross) * u128::from(cross),
        row_variance: 1,
        neighbor_variance: 1,
        denominator,
    }
}

#[test]
fn sums_of_integer_roots_have_the_exact_sign() {
    let mut terms = Terms::new();
    for (index, &(plus, minus, sign)) in vectors::PLAIN.iter().enumerate() {
        terms.clear();
        for &radicand in plus {
            terms.push_root(false, radicand);
        }
        for &radicand in minus {
            terms.push_root(true, radicand);
        }
        assert_eq!(
            terms.sign(),
            order(sign),
            "vector {index}: {plus:?} against {minus:?}"
        );
    }
}

#[test]
fn sums_of_z_normalized_distances_have_the_exact_sign() {
    let mut terms = Terms::new();
    for (index, &(plus, minus, sign)) in vectors::ZNORM.iter().enumerate() {
        terms.clear();
        for &(window, negative, cross, denominator) in plus {
            terms.push_znorm(false, window, correlation(negative, cross, denominator));
        }
        for &(window, negative, cross, denominator) in minus {
            terms.push_znorm(true, window, correlation(negative, cross, denominator));
        }
        assert_eq!(
            terms.sign(),
            order(sign),
            "vector {index}: {plus:?} against {minus:?}"
        );
    }
}

#[test]
fn the_sign_does_not_depend_on_the_side() {
    let mut terms = Terms::new();
    for &(plus, minus, sign) in vectors::PLAIN.iter().step_by(7) {
        terms.clear();
        for &radicand in plus {
            terms.push_root(true, radicand);
        }
        for &radicand in minus {
            terms.push_root(false, radicand);
        }
        assert_eq!(terms.sign(), order(-sign));
    }
}

#[test]
fn an_empty_sum_is_zero() {
    let mut terms = Terms::new();
    assert_eq!(terms.sign(), Ordering::Equal);
    terms.push_root(false, 0);
    terms.push_znorm(true, 8, ExactCorrelation::BothConstant);
    assert_eq!(terms.sign(), Ordering::Equal);
    terms.push_znorm(false, 9, ExactCorrelation::OneConstant);
    assert_eq!(terms.sign(), Ordering::Greater);
}

#[test]
fn a_distance_is_split_by_what_its_correlation_allows() {
    let split = |window: usize, correlation: ExactCorrelation| {
        let mut terms = Terms::new();
        terms.push_znorm(false, window, correlation);
        (terms.parts.len(), terms.nested.len())
    };
    // ρ = ±1: nothing, or 2√m.
    assert_eq!(split(8, correlation(false, 5, 25)), (0, 0));
    assert_eq!(split(8, correlation(true, 5, 25)), (1, 0));
    // ρ = 0 and a rational ρ: one plain root.
    assert_eq!(split(8, correlation(false, 0, 7)), (1, 0));
    assert_eq!(split(8, correlation(false, 3, 25)), (1, 0));
    // D = 18, N = 2: D(D − N²) = 18·14 is not a square.
    assert_eq!(split(8, correlation(false, 2, 18)), (0, 1));
    // D = 18, N = 4: D(D − N²) = 18·2 = 6², so the distance is a difference of two roots.
    assert_eq!(split(8, correlation(false, 4, 18)), (2, 0));
    assert_eq!(split(8, correlation(true, 4, 18)), (2, 0));
    assert_eq!(split(8, ExactCorrelation::OneConstant), (1, 0));
    assert_eq!(split(8, ExactCorrelation::BothConstant), (0, 0));
}

#[test]
fn zero_bound_grows_with_the_terms_and_saturates() {
    let mut terms = Terms::new();
    terms.push_znorm(false, 8, correlation(false, 3, 10));
    let one = terms.zero_bound();
    terms.push_znorm(true, 8, correlation(false, 3, 10));
    let two = terms.zero_bound();
    assert!(one > 0 && two > 4 * one);
    for _ in 0..20 {
        terms.push_znorm(true, 8, correlation(false, 3, 10));
    }
    assert_eq!(terms.zero_bound(), u64::MAX);
}

#[test]
fn a_tie_with_one_nested_root_on_each_side_ends_at_the_bound() {
    let mut terms = Terms::new();
    terms.push_znorm(false, 8, correlation(false, 3, 10));
    terms.push_znorm(true, 8, correlation(false, 3, 10));
    assert_eq!(terms.sign(), Ordering::Equal);
}

#[test]
fn nested_roots_near_one_correlation_are_ordered() {
    // ρ² = 1 − 1/D: the distance is tiny but not zero, so any sum with it has a sign.
    let d: u128 = 10u128.pow(24) + 1;
    let n = 10u64.pow(12);
    let mut terms = Terms::new();
    terms.push_znorm(false, 8, correlation(false, n, d));
    assert_eq!(terms.sign(), Ordering::Greater);
    terms.push_znorm(true, 8, correlation(false, n - 1, d));
    assert_eq!(terms.sign(), Ordering::Less);
}

#[test]
fn nested_roots_a_hair_apart_need_more_than_the_first_precision() {
    // Correlations 1/√D apart, with D past 2^120: the distances differ by about 1e-19 relative to
    // each other's size, below what 64 bits after the point show, so the precision must grow.
    let d: u128 = (1u128 << 124) + 12345;
    let n = ((1u64 << 62) as f64 * 0.99) as u64;
    for sign in [false, true] {
        let mut terms = Terms::new();
        terms.push_znorm(false, 8, correlation(sign, n, d));
        terms.push_znorm(true, 8, correlation(sign, n + 1, d));
        let want = if sign {
            Ordering::Less
        } else {
            Ordering::Greater
        };
        assert_eq!(terms.sign(), want, "{sign}");
    }
}

#[cfg(not(any(loom, shuttle)))]
#[test]
fn a_sum_of_many_nested_roots_that_cancels_ends_at_the_budget_as_a_tie() {
    let before = crate::multidim::budget_fallbacks();
    let terms_of = [(3u64, 10u128), (5, 29), (7, 61), (11, 131)];
    let mut terms = Terms::new();
    for &(n, d) in &terms_of {
        terms.push_znorm(false, 8, correlation(false, n, d));
    }
    for &(n, d) in terms_of.iter().rev() {
        terms.push_znorm(true, 8, correlation(false, n, d));
    }
    assert_eq!(terms.sign(), Ordering::Equal);
    assert!(terms.zero_bound() > super::PRECISION_BUDGET);
    assert!(crate::multidim::budget_fallbacks() > before);
}

#[test]
fn nested_root_sums_down_to_two_to_the_minus_two_hundred_have_the_exact_sign() {
    let mut terms = Terms::new();
    for (index, &(window, sign, plus, minus)) in super::nested_vectors::NESTED.iter().enumerate() {
        terms.clear();
        for &(negative, cross, denominator) in plus {
            terms.push_znorm(false, window, correlation(negative, cross, denominator));
        }
        for &(negative, cross, denominator) in minus {
            terms.push_znorm(true, window, correlation(negative, cross, denominator));
        }
        assert_eq!(terms.sign(), order(sign), "vector {index}");
    }
}

#[test]
fn one_nested_root_written_with_scaled_integers_is_equal_to_itself() {
    // (N, D) and (kN, k²D) are the same correlation, so the same distance, but every integer the
    // enclosures start from differs: only enclosures that always contain the true value can end
    // with a sum that straddles zero (rather than a wrong sign) every time.
    let mut terms = Terms::new();
    for window in [1, 2, 3, 8] {
        for sign in [false, true] {
            for scale in 2..40u64 {
                let scaled = correlation(sign, 3 * scale, 10 * u128::from(scale * scale));
                terms.clear();
                terms.push_znorm(false, window, correlation(sign, 3, 10));
                terms.push_znorm(true, window, scaled);
                assert_eq!(terms.sign(), Ordering::Equal, "{window} {sign} {scale}");
            }
        }
    }
}

/// The enclosure of a nested root at a precision contains the enclosure at a much higher one, so
/// it contains the root: its upper end is an upper end (the integer square root floors, so the
/// rounding outward cannot be left to the correlation's alone), and its lower end a lower end.
#[test]
fn the_enclosure_of_a_nested_root_contains_the_root() {
    use super::super::big::{Arith, Big};
    use super::{Nested, enclose_nested};
    use core::cmp::Ordering::Greater;

    let mut state = 0x2545_F491_4F6C_DD1D_u64;
    let mut next = |bound: u64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % bound
    };
    let mut arith = Arith::default();
    let (mut low, mut high, mut deep_low, mut deep_high) = (
        Big::default(),
        Big::default(),
        Big::default(),
        Big::default(),
    );
    for round in 0..1500 {
        let denominator = (u128::from(next(u64::MAX)) << (next(60) as u32)) | 3;
        let denominator = denominator.min((1u128 << 126) - 1);
        let root = denominator.isqrt();
        if root * root == denominator || root < 3 {
            continue;
        }
        let cross = (1 + next(u64::try_from(root - 1).unwrap_or(u64::MAX >> 1))).min(1 << 62);
        if u128::from(cross) * u128::from(cross) > denominator {
            continue;
        }
        let nested = Nested {
            negative: false,
            rho_negative: round % 2 == 0,
            window: [1, 2, 3, 5, 8, 100][next(6) as usize],
            cross: arith.get_u128(u128::from(cross)),
            denominator: arith.get_u128(denominator),
        };
        for precision in [64u64, 100] {
            enclose_nested(&mut arith, &nested, precision, &mut low, &mut high);
            enclose_nested(
                &mut arith,
                &nested,
                precision + 256,
                &mut deep_low,
                &mut deep_high,
            );
            low.shl(256);
            high.shl(256);
            assert_ne!(low.compare(&deep_low), Greater, "lower end, round {round}");
            assert_ne!(
                deep_high.compare(&high),
                Greater,
                "upper end, round {round}"
            );
        }
    }
}

#[test]
fn a_correlation_on_256_bits_is_the_same_distance_as_on_the_hot_form() {
    use crate::spill::Spilled;
    use crate::wide::Wide;
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = |bound: u64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % bound
    };
    let mut terms = Terms::new();
    for round in 0..400 {
        let (row, neighbor) = (1 + next(1 << 40), 1 + next(1 << 40));
        let denominator = u128::from(row) * u128::from(neighbor);
        let cross = next(denominator.isqrt() as u64 + 1);
        let negative = round % 3 == 0;
        let window = [1, 4, 50][round % 3];
        let n = if negative {
            -i128::from(cross)
        } else {
            i128::from(cross)
        };
        let hot = ExactCorrelation::Valid {
            sign: n.signum() as i8,
            cross_squared: u128::from(cross) * u128::from(cross),
            row_variance: row,
            neighbor_variance: neighbor,
            denominator,
        };
        let wide = Spilled::Valid {
            n: Wide::from_i128(n),
            row_variance: Wide::from_u64(row),
            neighbor_variance: Wide::from_u64(neighbor),
        };
        terms.clear();
        terms.push_znorm(false, window, hot);
        terms.push_znorm_wide(true, window, wide);
        assert_eq!(terms.sign(), Ordering::Equal, "round {round}");
        terms.clear();
        terms.push_znorm(false, window, hot);
        terms.push_root(true, 2 * window as u128 * 4);
        let expected = terms.sign();
        terms.clear();
        terms.push_znorm_wide(false, window, wide);
        terms.push_root(true, 2 * window as u128 * 4);
        assert_eq!(terms.sign(), expected, "round {round}");
    }
}
