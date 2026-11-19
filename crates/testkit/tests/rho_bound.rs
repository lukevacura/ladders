//! `rank_estimate` ranks every candidate pair in O(1). This checks, against exact
//! arbitrary-precision rational arithmetic rather than a second floating-point computation,
//! that its result never strays more than `2⁻⁵⁰` from the pair's true correlation.

use ladde::estimate::rank_estimate;
use num_bigint::BigInt;
use num_rational::BigRational;
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::TestRunner;

/// `2⁻⁵⁰`, the bound `rank_estimate` promises.
fn delta() -> BigRational {
    BigRational::new(BigInt::from(1), BigInt::from(1) << 50u32)
}

fn zero() -> BigRational {
    BigRational::from_integer(BigInt::from(0))
}

/// The exact value of a finite, non-subnormal `f64`, as a rational: `mantissa · 2^exponent`,
/// signed.
fn exact_f64(x: f64) -> BigRational {
    if x == 0.0 {
        return zero();
    }
    assert!(
        x.is_finite(),
        "rank_estimate never produces a non-finite result"
    );
    let bits = x.to_bits();
    let negative = bits >> 63 == 1;
    let biased_exponent = ((bits >> 52) & 0x7ff) as i64;
    assert_ne!(biased_exponent, 0, "no input to rank_estimate is subnormal");
    let fraction = bits & 0x000f_ffff_ffff_ffff;
    let mantissa = BigInt::from((1u64 << 52) | fraction);
    let mantissa = if negative { -mantissa } else { mantissa };
    let exponent = biased_exponent - 1023 - 52;
    if exponent >= 0 {
        BigRational::from_integer(mantissa << exponent as u32)
    } else {
        BigRational::new(mantissa, BigInt::from(1) << (-exponent) as u32)
    }
}

/// `lhs ≤ coefficient · √p`, decided exactly without computing the irrational `√p`:
/// `coefficient`'s sign pins the sign of the right side, and squaring both sides (valid once
/// they agree in sign) decides the rest.
fn le_scaled_sqrt(lhs: &BigInt, coefficient: &BigRational, p: &BigInt) -> bool {
    let p = BigRational::from_integer(p.clone());
    let lhs_squared = BigRational::from_integer(lhs * lhs);
    let scaled = coefficient * coefficient * p;
    let zero_int = BigInt::from(0);
    if *coefficient < zero() {
        *lhs <= zero_int && lhs_squared >= scaled
    } else {
        *lhs <= zero_int || lhs_squared <= scaled
    }
}

/// `|rank_estimate(n, ri, rj) − n/√(vi·vj)| ≤ 2⁻⁵⁰`, checked exactly: `n` sits between
/// `(ρ̂ − δ)·√p` and `(ρ̂ + δ)·√p`, `p = vi·vj`, with no floating point on the exact side.
fn within_bound(n: i64, vi: u64, vj: u64) -> bool {
    let ri = 1.0 / f64::sqrt(vi as f64);
    let rj = 1.0 / f64::sqrt(vj as f64);
    let rho_hat = exact_f64(rank_estimate(n, ri, rj));
    let delta = delta();
    let p = BigInt::from(vi) * BigInt::from(vj);
    let n_big = BigInt::from(n);

    let upper = &rho_hat + &delta; // n ≤ (ρ̂ + δ)·√p
    let lower = &delta - &rho_hat; // (ρ̂ − δ)·√p ≤ n  ⟺  −n ≤ (δ − ρ̂)·√p
    le_scaled_sqrt(&n_big, &upper, &p) && le_scaled_sqrt(&-n_big, &lower, &p)
}

/// `n`'s exact bound for a pair with variances `vi`, `vj`: `⌊√(vi·vj)⌋`.
fn max_n(vi: u64, vj: u64) -> i64 {
    ((vi as u128) * (vj as u128)).isqrt() as i64
}

// ---------- pinned adversarial corners ----------

#[test]
fn smallest_variances_stay_within_bound() {
    for n in [-1i64, 0, 1] {
        assert!(within_bound(n, 1, 1));
    }
}

#[test]
fn largest_variances_stay_within_bound() {
    let v = (1u64 << 63) - 1;
    let bound = max_n(v, v);
    for n in [-bound, -1, 0, 1, bound] {
        assert!(within_bound(n, v, v), "n {n}, v {v}");
    }
}

#[test]
fn mismatched_variances_by_a_large_ratio_stay_within_bound() {
    let (vi, vj) = (1u64, (1u64 << 63) - 1);
    let bound = max_n(vi, vj);
    for n in [-bound, 0, bound] {
        assert!(within_bound(n, vi, vj), "n {n}, vi {vi}, vj {vj}");
    }
}

#[test]
fn nearly_collinear_windows_stay_within_bound() {
    for &(vi, vj) in &[(3u64, 5u64), (1_000_003u64, 999_999_937u64)] {
        let bound = max_n(vi, vj);
        for n in [bound, bound - 1, -bound, -(bound - 1)] {
            assert!(within_bound(n, vi, vj), "n {n}, vi {vi}, vj {vj}");
        }
    }
}

#[test]
fn small_prime_variances_stay_within_bound() {
    for &vi in &[2u64, 3, 5, 7, 11, 13] {
        for &vj in &[2u64, 3, 5, 7, 11, 13] {
            let bound = max_n(vi, vj);
            for n in [-bound, 0, bound] {
                assert!(within_bound(n, vi, vj), "n {n}, vi {vi}, vj {vj}");
            }
        }
    }
}

// ---------- property: random and adversarial-weighted (vi, vj, n), exact rational oracle ----------

/// `v` across the whole domain the library can produce (`0 < v < 2⁶³`), weighted toward tiny,
/// near-`2⁶³`, and uniformly random magnitudes.
fn v_strategy() -> impl Strategy<Value = u64> {
    prop_oneof![
        3 => 1u64..1_000,
        3 => ((1u64 << 63) - 1_000)..(1u64 << 63),
        4 => 1u64..(1u64 << 63),
    ]
}

/// `(n, vi, vj)`, `n` weighted toward `0`, `±1`, and the collinear boundary `±⌊√(vi·vj)⌋`, as
/// well as plain uniform.
fn case_strategy() -> impl Strategy<Value = (i64, u64, u64)> {
    (v_strategy(), v_strategy()).prop_flat_map(|(vi, vj)| {
        let bound = max_n(vi, vj);
        prop_oneof![
            1 => Just(0i64),
            1 => Just(1i64),
            1 => Just(-1i64),
            1 => Just(bound),
            1 => Just(-bound),
            1 => Just(bound - 1),
            1 => Just(-(bound - 1)),
            4 => -bound..=bound,
        ]
        .prop_map(move |n| (n, vi, vj))
    })
}

/// Cases for the property test below: enough to mix the weighted adversarial corners with
/// genuine randomness, few enough to stay in the fast tier.
const CASES: u32 = 8192;

#[test]
fn rank_estimate_matches_exact_correlation_within_bound() {
    let mut runner = TestRunner::deterministic();
    let strategy = case_strategy();
    let one_ulp = 1.0 / (1u64 << 53) as f64;
    let mut worst_error_ulps = 0.0f64;

    for _ in 0..CASES {
        let (n, vi, vj) = strategy
            .new_tree(&mut runner)
            .expect("strategy produces a value")
            .current();

        assert!(
            within_bound(n, vi, vj),
            "n {n}, vi {vi}, vj {vj} exceeds the 2⁻⁵⁰ bound"
        );

        // Diagnostic only, not the oracle above: an f64 reference close enough to report how
        // tight the measured worst case is, in units of 2⁻⁵³.
        let ri = 1.0 / f64::sqrt(vi as f64);
        let rj = 1.0 / f64::sqrt(vj as f64);
        let rho_hat = rank_estimate(n, ri, rj);
        let rho_reference = (n as f64) / ((vi as f64) * (vj as f64)).sqrt();
        let error_ulps = (rho_hat - rho_reference).abs() / one_ulp;
        worst_error_ulps = worst_error_ulps.max(error_ulps);
    }

    eprintln!(
        "rank_estimate: worst observed |rho_hat - rho| ~ {worst_error_ulps:.3} x 2^-53 over {CASES} cases (bound is 2^-50 = 8 x 2^-53)"
    );
}
