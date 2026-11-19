extern crate std;

use super::*;
use crate::multidim::Terms;
use crate::nearest::{ExactCorrelation, variance_product};
use core::cmp::Ordering;
use std::vec;

/// `1/√v`, correctly rounded twice, as a window's statistics hold it.
fn inv_sqrt(v: u64) -> f64 {
    1.0 / libm::sqrt(v as f64)
}

/// Whether `bound` is at most the exact distance `terms` holds (pushed positive), exactly.
fn at_most(terms: &mut Terms, bound: f64) -> bool {
    if bound > 0.0 {
        terms.push_scaled(true, bound, 1, 1);
    }
    terms.sign() != Ordering::Less
}

/// Variances near both ends of what a window can have, and cross terms at, next to and far from
/// the Cauchy–Schwarz limit `±√(Vi·Vj)` (correlations near `±1` and `0`), where the rounding of
/// the estimate matters most.
fn adversarial() -> vec::Vec<(i64, u64, u64)> {
    let variances = [
        1u64,
        2,
        3,
        7,
        1 << 20,
        (1 << 31) + 11,
        (1 << 52) - 1,
        (1 << 62) + 12_345,
        (1 << 63) - 1,
        834_911_684_991_698,
        275_720_901_775_960,
    ];
    let mut cases = vec::Vec::new();
    for &vi in &variances {
        for &vj in &variances {
            let limit = variance_product(vi, vj).isqrt();
            let limit = i64::try_from(limit).unwrap_or(i64::MAX);
            for n in [
                limit,
                limit - 1,
                limit - 2,
                limit / 2,
                limit / 3 + 1,
                1,
                0,
                -1,
                -(limit / 2),
                -(limit - 1),
                -limit,
            ] {
                if u128::from(n.unsigned_abs()).pow(2) <= variance_product(vi, vj) {
                    cases.push((n, vi, vj));
                }
            }
        }
    }
    cases
}

#[test]
fn a_z_normalized_bound_is_at_most_the_exact_distance() {
    let mut checked = 0;
    let mut tight = 0;
    for m in [3usize, 64, 1000] {
        for (n, vi, vj) in adversarial() {
            let rho_hat = crate::estimate::rank_estimate(n, inv_sqrt(vi), inv_sqrt(vj));
            let bound = znorm_lower(rho_hat, 2.0 * m as f64);
            let magnitude = u128::from(n.unsigned_abs());
            let correlation = ExactCorrelation::Valid {
                sign: n.signum() as i8,
                cross_squared: magnitude * magnitude,
                row_variance: vi,
                neighbor_variance: vj,
                denominator: variance_product(vi, vj),
            };
            let mut terms = Terms::new();
            terms.push_znorm(false, m, correlation);
            assert!(at_most(&mut terms, bound), "{m} {n} {vi} {vj}: {bound}");
            // Away from a zero distance the bound is a few units in the last place below it.
            let mut terms = Terms::new();
            terms.push_znorm(false, m, correlation);
            if bound > 1e-3 && !at_most(&mut terms, bound * (1.0 + 1e-12)) {
                tight += 1;
            }
            checked += 1;
        }
    }
    assert!(checked > 3000, "{checked}");
    assert!(tight > checked / 2, "{tight} of {checked}");
}

/// The margin is the f64 producer's bound: an estimate exactly that far below one bounds the
/// distance by zero, and one twice as far by `√(2m·δ)` exactly (`2m·δ = 2⁻⁴⁸` at `m = 1`), so
/// one ULP of the margin either way moves a bound.
#[test]
fn the_margin_is_the_f64_producers_bound() {
    use crate::estimate::{F64, Producer};
    assert_eq!(znorm_lower(1.0 - F64::DELTA, 2.0), 0.0);
    assert!(znorm_lower((1.0 - F64::DELTA).next_down(), 2.0) > 0.0);
    let twice = znorm_lower(1.0 - 2.0 * F64::DELTA, 2.0);
    assert_eq!(twice.to_bits(), (SHRINK / (1u64 << 24) as f64).to_bits());
}

#[test]
fn a_fixed_correlation_bounds_its_exact_distance() {
    for m in [1usize, 3, 64, 1000] {
        let two_m = 2.0 * m as f64;
        // Two constant windows are at distance zero, one at `√m`; a gap has none.
        assert_eq!(znorm_lower(1.0, two_m), 0.0);
        let one = znorm_lower(0.5, two_m);
        let mut terms = Terms::new();
        terms.push_root(false, m as u128);
        assert!(at_most(&mut terms, one), "{m}");
        assert!(one > libm::sqrt(m as f64) * (1.0 - 1e-13), "{m}");
        assert_eq!(znorm_lower(f64::NEG_INFINITY, two_m), f64::INFINITY);
    }
}

#[test]
fn a_root_bound_is_at_most_the_root() {
    let mut values = vec![0u128, 1, 2, 3, 4, 15, 16, 17, u128::MAX, u128::MAX - 1];
    for shift in 1..127 {
        let power = 1u128 << shift;
        values.extend([power - 1, power, power + 1]);
        let root = power.isqrt();
        values.extend([root * root, root * root - 1, (root + 1) * (root + 1) - 1]);
    }
    let mut x = 0x9E37_79B9_7F4A_7C15_u128;
    for _ in 0..2000 {
        x = x.wrapping_mul(0x2545_F491_4F6C_DD1D_9E37_79B9_7F4A_7C15) ^ (x >> 29);
        values.push(x >> (x % 120));
    }
    for squared in values {
        let bound = root_lower(squared);
        let mut terms = Terms::new();
        terms.push_root(false, squared);
        assert!(at_most(&mut terms, bound), "{squared}: {bound}");
    }
}

#[test]
fn the_sum_scale_exceeds_the_count_by_the_rounding_only() {
    for k in 0..1000 {
        let scale = sum_scale(k);
        let count = (k + 1) as f64;
        assert!(scale > count, "{k}");
        // `k + 2` roundings of `2^-53`, four times over: no more.
        assert!(scale <= count * (1.0 + (k + 2) as f64 * 4.5 / (1u64 << 53) as f64));
    }
    assert_eq!(sum_scale(1 << 51), f64::INFINITY);
}

fn screen(count: usize, include: &[usize], discords: bool, ceilings: &[f64]) -> Screen {
    let mut screen = Screen::new(count, include, discords, 1);
    assert!(screen.begin(&(5..6)));
    for (k, &ceiling) in ceilings.iter().enumerate() {
        screen.lower_one(5, k, ceiling);
    }
    screen
}

#[test]
fn a_cell_settles_only_when_every_mean_is_past_its_ceiling() {
    // Nearest first: the means of the bounds 1, 2, 3 are 1, 1.5 and 2.
    let bounds = [3.0, 1.0, 2.0];
    assert!(screen(3, &[], false, &[0.9, 1.4, 1.9]).settles(5, &bounds));
    for k in 0..3 {
        let mut ceilings = [0.9, 1.4, 1.9];
        ceilings[k] = [1.0, 1.5, 2.0][k];
        assert!(!screen(3, &[], false, &ceilings).settles(5, &bounds), "{k}");
    }
    // Farthest first: 3, 2.5 and 2.
    assert!(screen(3, &[], true, &[2.9, 2.4, 1.9]).settles(5, &bounds));
    assert!(!screen(3, &[], true, &[2.9, 2.5, 1.9]).settles(5, &bounds));
    assert!(!screen(3, &[], false, &[0.9, 2.4, 1.9]).settles(5, &bounds));
    // `include` comes first, as given: 3, then 2 and 2 (the rest nearest first); with `[2, 0]`
    // and discords, 2, then 2.5 and 2.
    assert!(screen(3, &[0], false, &[1.9, 1.9, 1.9]).settles(5, &bounds));
    assert!(!screen(3, &[0], false, &[1.9, 2.0, 1.9]).settles(5, &bounds));
    assert!(screen(3, &[2, 0], true, &[1.9, 2.4, 1.9]).settles(5, &bounds));
    assert!(!screen(3, &[2, 0], true, &[2.0, 2.4, 1.9]).settles(5, &bounds));
    // No ceiling yet: nothing settles.
    assert!(!screen(3, &[], false, &[]).settles(5, &bounds));
}

#[test]
fn a_mean_with_a_missing_distance_is_past_any_ceiling() {
    let bounds = [f64::INFINITY, 1.0, 2.0];
    // The missing dimension is last in both orders, so only the mean of all three holds it.
    assert!(screen(3, &[], false, &[0.9, 1.4, 0.0]).settles(5, &bounds));
    assert!(!screen(3, &[], false, &[0.9, 1.5, 0.0]).settles(5, &bounds));
    assert!(screen(3, &[], true, &[1.9, 1.4, 0.0]).settles(5, &bounds));
    assert!(!screen(3, &[], true, &[2.0, 1.4, 0.0]).settles(5, &bounds));
    // Included first, it makes every mean missing.
    assert!(screen(3, &[0], false, &[0.0, 0.0, 0.0]).settles(5, &bounds));
    // A mean that is missing never settles against a slot that holds none either.
    assert!(!screen(3, &[0], false, &[]).settles(5, &bounds));
}

#[test]
fn a_cell_whose_mean_ties_its_ceiling_does_not_settle() {
    // Equal means go to the exact path, which breaks the tie by window start.
    for count in [1usize, 2, 3, 8, 17, 40] {
        let bounds = vec![1.5; count];
        let ceilings = vec![1.5; count];
        for discords in [false, true] {
            assert!(
                !screen(count, &[], discords, &ceilings).settles(5, &bounds),
                "{count}"
            );
            let below: vec::Vec<f64> = ceilings.iter().map(|c| c * (1.0 - 1e-12)).collect();
            assert!(
                screen(count, &[], discords, &below).settles(5, &bounds),
                "{count}"
            );
        }
    }
}

#[test]
fn ceilings_last_while_the_rows_do_and_only_fall() {
    let mut screen = Screen::new(2, &[], false, 4);
    assert!(screen.begin(&(10..14)));
    screen.lower(12, |k| [3.0, 4.0][k]);
    screen.lower_one(12, 1, 5.0);
    screen.lower_one(12, 0, 2.0);
    assert!(!screen.begin(&(10..14)));
    let bounds = [2.5, 2.5];
    assert!(!screen.settles(12, &bounds));
    let bounds = [2.5, 6.0];
    assert!(screen.settles(12, &bounds));
    assert!(!screen.settles(13, &bounds));
    // Other rows start with no ceiling.
    assert!(screen.begin(&(14..18)));
    screen.lower(14, |_| f64::INFINITY);
    assert!(!screen.settles(14, &bounds));
}

/// `keys` sorted by the network for their length, through an array padded with infinities.
fn network_sorted(keys: &[f64]) -> vec::Vec<f64> {
    fn padded<const N: usize>(keys: &[f64], pairs: &[(usize, usize)]) -> vec::Vec<f64> {
        let mut array = [f64::INFINITY; N];
        array[..keys.len()].copy_from_slice(keys);
        sort_network(&mut array, pairs);
        array[..keys.len()].to_vec()
    }
    if keys.len() <= 4 {
        padded::<4>(keys, &FOUR)
    } else {
        padded::<8>(keys, &EIGHT)
    }
}

/// A key a cell can hold: a missing distance, a negated (flipped) bound, or a small bound that
/// often ties another.
fn key(x: &mut u64) -> f64 {
    *x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
    match *x >> 61 {
        0 => f64::INFINITY,
        1 => -((*x >> 40) as f64),
        _ => ((*x >> 54) % 5) as f64,
    }
}

#[test]
fn keys_sort_ascending_at_every_length() {
    let mut x = 7u64;
    for len in 0usize..=8 {
        for _ in 0..50 {
            let keys: vec::Vec<f64> = (0..len).map(|_| key(&mut x)).collect();
            let mut want = keys.clone();
            want.sort_by(f64::total_cmp);
            assert_eq!(network_sorted(&keys), want, "{len}");
        }
    }
}

/// Every input of zeros and ones sorts, so each network sorts every input (the zero-one
/// principle).
#[test]
fn the_networks_sort_every_input() {
    for len in 2usize..=8 {
        for bits in 0u32..1 << len {
            let keys: vec::Vec<f64> = (0..len).map(|i| f64::from(bits >> i & 1)).collect();
            let keys = network_sorted(&keys);
            assert!(
                keys.windows(2).all(|pair| pair[0] <= pair[1]),
                "{len} {bits:b}"
            );
            let ones = keys.iter().filter(|&&key| key == 1.0).count();
            assert_eq!(ones, bits.count_ones() as usize);
        }
    }
}

/// What the screen settles, from its definition: the `include` bounds as given, then the rest
/// nearest first (farthest first for discords, a missing one still last), each running sum past
/// its ceiling times the sum scale.
fn settles_by_definition(
    bounds: &[f64],
    include: &[usize],
    discords: bool,
    ceilings: &[f64],
) -> bool {
    let mut rest: vec::Vec<f64> = (0..bounds.len())
        .filter(|a| !include.contains(a))
        .map(|a| bounds[a])
        .collect();
    rest.sort_by(|a, b| match (*a == f64::INFINITY, *b == f64::INFINITY) {
        (false, false) if discords => b.total_cmp(a),
        (false, false) => a.total_cmp(b),
        (x, y) => x.cmp(&y),
    });
    let keys = include.iter().map(|&a| bounds[a]).chain(rest);
    let mut sum = 0.0;
    keys.zip(ceilings).enumerate().all(|(k, (bound, ceiling))| {
        sum += bound;
        sum > ceiling * sum_scale(k)
    }) && ceilings.len() == bounds.len()
}

/// The fixed arrays of up to eight dimensions and the vector beyond settle exactly the cells the
/// definition does, whatever the `include` list and the order, with keys that tie and go missing.
#[test]
fn every_count_settles_as_defined() {
    let mut x = 11u64;
    let mut settled = [0usize; 2];
    for count in 1usize..=12 {
        for case in 0..400 {
            let bounds: vec::Vec<f64> = (0..count).map(|_| key(&mut x).abs()).collect();
            let fixed = case % (count + 1);
            let mut include: vec::Vec<usize> = (0..count).collect();
            for i in (1..count).rev() {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
                include.swap(i, (x >> 33) as usize % (i + 1));
            }
            include.truncate(fixed);
            let discords = case % 2 == 1;
            // Mean ceilings among the bounds' own values, so both outcomes come up.
            let mut ceilings: vec::Vec<f64> = (0..count)
                .map(|_| {
                    x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
                    [0.0, 0.5, 1.0, 2.0][(x >> 62) as usize]
                })
                .collect();
            if case % 7 == 0 {
                ceilings.iter_mut().for_each(|c| *c = 0.0);
            }
            let want = settles_by_definition(&bounds, &include, discords, &ceilings);
            let got = screen(count, &include, discords, &ceilings).settles(5, &bounds);
            assert_eq!(
                got, want,
                "{count} {include:?} {discords} {bounds:?} {ceilings:?}"
            );
            settled[usize::from(got)] += 1;
        }
    }
    assert!(settled[0] > 100 && settled[1] > 100, "{settled:?}");
}

/// An estimate as far from the exact correlation as its stated error allows (`2^-50`, on either
/// side) still bounds the distance from below: the margin is what makes the screen sound. Equal
/// variances of `2^40` make every `ρ = ±(1 − t·2^-40)` exact in `f64`, so `ρ ± 2^-50` is exact
/// too, from a zero distance's neighborhood (`t = 1`) to ordinary ones.
#[test]
fn an_estimate_at_the_edge_of_its_error_still_bounds_the_distance() {
    let variance = 1u64 << 40;
    let error = 1.0 / (1u64 << 50) as f64;
    let mut checked = 0;
    for m in [3usize, 64, 1000] {
        for t in [0u64, 1, 2, 1 << 10, 1 << 30, 1 << 39, (1 << 40) - 1] {
            for negative in [false, true] {
                let magnitude = variance - t;
                let rho = magnitude as f64 / variance as f64;
                let rho = if negative { -rho } else { rho };
                let correlation = ExactCorrelation::Valid {
                    sign: if magnitude == 0 {
                        0
                    } else if negative {
                        -1
                    } else {
                        1
                    },
                    cross_squared: u128::from(magnitude).pow(2),
                    row_variance: variance,
                    neighbor_variance: variance,
                    denominator: variance_product(variance, variance),
                };
                for rho_hat in [rho - error, rho + error] {
                    assert_eq!(rho_hat - rho, if rho_hat < rho { -error } else { error });
                    let bound = znorm_lower(rho_hat, 2.0 * m as f64);
                    let mut terms = Terms::new();
                    terms.push_znorm(false, m, correlation);
                    assert!(
                        at_most(&mut terms, bound),
                        "{m} {t} {negative} {rho_hat}: {bound}"
                    );
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 3 * 7 * 2 * 2);
}
