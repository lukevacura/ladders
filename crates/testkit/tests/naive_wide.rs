//! The oracle's 256-bit arithmetic, cross-checked against `num-bigint` on inputs whose
//! intermediates overflow i128, and against the engines at m = 4096 near the tick ceiling.

use core::num::NonZeroUsize;
use ladde::batch::left_profile;
use ladde::naive::{NonNormDistance, naive_left_profile, naive_left_profile_nonnorm};
use ladde::quantize::{MAX_TICKS, Scale, to_ticks};
use num_bigint::{BigInt, Sign};
use std::cmp::Ordering;
use testkit::rng::XorShift64Star;

fn big(window: &[i64]) -> Vec<BigInt> {
    window.iter().map(|&t| BigInt::from(t)).collect()
}

fn dot(a: &[BigInt], b: &[BigInt]) -> BigInt {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `(Σx, V = m·Σx² − (Σx)²)` in unbounded integers.
fn sum_and_v(w: &[BigInt]) -> (BigInt, BigInt) {
    let s: BigInt = w.iter().sum();
    let v = BigInt::from(w.len()) * dot(w, w) - &s * &s;
    (s, v)
}

/// Orders two exact correlations `(n, p)` = `(N, Vi·Vj)`: by sign, then by `N²/P`.
fn cmp_rho(a: &(BigInt, BigInt), b: &(BigInt, BigInt)) -> Ordering {
    let sign = |n: &BigInt| match n.sign() {
        Sign::Minus => -1i8,
        Sign::NoSign => 0,
        Sign::Plus => 1,
    };
    let (sa, sb) = (sign(&a.0), sign(&b.0));
    if sa != sb {
        return sa.cmp(&sb);
    }
    let magnitude = (&a.0 * &a.0 * &b.1).cmp(&(&b.0 * &b.0 * &a.1));
    if sa >= 0 {
        magnitude
    } else {
        magnitude.reverse()
    }
}

#[test]
fn znorm_neighbors_match_bigint_where_i128_would_overflow() {
    let mut rng = XorShift64Star(0x9E37_79B9_7F4A_7C15);
    let mut checked = 0;
    for _ in 0..40 {
        // Ticks near 2^62: Σx² over 64 of them is about 2^130, past i128.
        let base = (1i64 << 62) - 100_000 + rng.below(1000) as i64;
        let m = 64usize;
        let series: Vec<i64> = (0..160).map(|_| base + rng.below(16) as i64).collect();
        let samples: Vec<Option<i64>> = series.iter().map(|&t| Some(t)).collect();
        let e = ladde::exclusion::zone(m);
        for &(row, entry) in &naive_left_profile(&samples, m) {
            let i = row as usize;
            let row_window = big(&series[i..i + m]);
            let (si, vi) = sum_and_v(&row_window);
            let mut best: Option<(usize, (BigInt, BigInt))> = None;
            for j in 0..i.saturating_sub(e) {
                let w = big(&series[j..j + m]);
                let (sj, vj) = sum_and_v(&w);
                let n = BigInt::from(m) * dot(&row_window, &w) - &si * &sj;
                let key = (n, &vi * &vj);
                if best
                    .as_ref()
                    .is_none_or(|(_, b)| cmp_rho(&key, b) == Ordering::Greater)
                {
                    best = Some((j, key));
                }
            }
            assert_eq!(
                entry.index().get(),
                best.map(|(j, _)| j as u64),
                "row {row}"
            );
            checked += usize::from(entry.index().get().is_some());
        }
    }
    assert!(checked > 1000);
}

#[test]
fn nonnorm_wide_distances_match_bigint() {
    let mut rng = XorShift64Star(0xD1B5_4A32_D192_ED03);
    let mut wide = 0;
    for _ in 0..40 {
        let m = 8usize;
        let series: Vec<i64> = (0..40)
            .map(|_| (rng.next_u64() as i64) >> rng.below(3)) // full i64 range, squares past i128
            .collect();
        let samples: Vec<Option<i64>> = series.iter().map(|&t| Some(t)).collect();
        let e = ladde::exclusion::zone(m);
        for &(row, entry) in &naive_left_profile_nonnorm(&samples, m) {
            let i = row as usize;
            let a = big(&series[i..i + m]);
            let mut best: Option<(usize, BigInt)> = None;
            for j in 0..i.saturating_sub(e) {
                let b = big(&series[j..j + m]);
                let d: BigInt = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
                if best.as_ref().is_none_or(|(_, c)| d < *c) {
                    best = Some((j, d));
                }
            }
            match (entry.distance(), best) {
                (_, None) => assert!(entry.index().get().is_none()),
                (NonNormDistance::Wide(d), Some((j, expected))) => {
                    assert_eq!(entry.index().get(), Some(j as u64));
                    assert_eq!(d.to_string(), expected.to_string(), "row {row}");
                    wide += 1;
                }
                (NonNormDistance::Narrow(_), Some(_)) => panic!("range is far past the bound"),
            }
        }
    }
    assert!(wide > 100);
}

#[test]
fn m_4096_near_the_tick_ceiling_agrees_with_the_engines_on_bits() {
    let m = 4096usize;
    let mut rng = XorShift64Star(0xA076_1D64_78BD_642F);
    // Σx² is about 2^118 per window; m·Σx² about 2^130, past i128. Small spread keeps V < 2^63.
    let base = MAX_TICKS - 1000;
    let series: Vec<f64> = (0..m + 1050)
        .map(|_| (base + rng.below(16) as i64) as f64)
        .collect();
    let scale = Scale::new(1.0).unwrap();
    let samples: Vec<Option<i64>> = series.iter().map(|&x| to_ticks(x, scale)).collect();
    assert!(samples.iter().all(Option::is_some));
    let oracle = naive_left_profile(&samples, m);
    check_last_rows_against_bigint(&samples, m, &oracle);
    let engine = left_profile(&series, NonZeroUsize::new(m).unwrap(), scale);
    assert_eq!(oracle.len(), engine.len());
    let mut matched = 0;
    for (&(row, o), &(_, g)) in oracle.iter().zip(&engine) {
        assert_eq!(o.rho().to_bits(), g.rho().to_bits(), "row {row}");
        assert_eq!(
            o.squared_distance().to_bits(),
            g.squared_distance().to_bits(),
            "row {row}"
        );
        assert_eq!(o.index(), g.index(), "row {row}");
        matched += usize::from(o.index().get().is_some());
    }
    assert!(matched > 20);
}

/// The last few rows' neighbors against unbounded integers, as in the z-normalized test.
fn check_last_rows_against_bigint(
    samples: &[Option<i64>],
    m: usize,
    oracle: &[(u64, ladde::entry::Entry)],
) {
    let series: Vec<i64> = samples.iter().map(|t| t.unwrap()).collect();
    let e = ladde::exclusion::zone(m);
    for &(row, entry) in oracle.iter().rev().take(3) {
        let i = row as usize;
        let row_window = big(&series[i..i + m]);
        let (si, vi) = sum_and_v(&row_window);
        let mut best: Option<(usize, (BigInt, BigInt))> = None;
        for j in 0..i - e {
            let w = big(&series[j..j + m]);
            let (sj, vj) = sum_and_v(&w);
            let key = (
                BigInt::from(m) * dot(&row_window, &w) - &si * &sj,
                &vi * &vj,
            );
            if best
                .as_ref()
                .is_none_or(|(_, b)| cmp_rho(&key, b) == Ordering::Greater)
            {
                best = Some((j, key));
            }
        }
        assert_eq!(
            entry.index().get(),
            best.map(|(j, _)| j as u64),
            "row {row}"
        );
    }
}
