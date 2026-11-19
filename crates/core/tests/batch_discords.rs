//! The batch top-k discords equal an independent brute force over exact rational correlations
//! (greedy pick, same zone and tie rule) at every thread count, on series with exact repeats,
//! scaled copies, plateaus and NaN gaps.

use core::cmp::Ordering;
use core::num::NonZeroUsize;

use ladde::batch::{discords, two_sided_profile};
use ladde::quantize::{Scale, to_ticks};
use proptest::prelude::*;

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

/// A correlation `sign * num / den` with `den > 0`.
#[derive(Clone, Copy)]
struct Rho {
    sign: i8,
    num: u128,
    den: u128,
}

fn cmp_rho(a: Rho, b: Rho) -> Ordering {
    let signed = |r: Rho| if r.num == 0 { 0 } else { i32::from(r.sign) };
    signed(a).cmp(&signed(b)).then_with(|| {
        let by_magnitude = (a.num * b.den).cmp(&(b.num * a.den));
        if signed(a) < 0 {
            by_magnitude.reverse()
        } else {
            by_magnitude
        }
    })
}

fn moments(window: &[i64]) -> (i128, i128) {
    let m = window.len() as i128;
    let sum: i128 = window.iter().map(|&x| i128::from(x)).sum();
    let squares: i128 = window.iter().map(|&x| i128::from(x) * i128::from(x)).sum();
    (sum, m * squares - sum * sum)
}

fn rho(a: &[i64], b: &[i64]) -> Rho {
    let m = a.len() as i128;
    let ((sum_a, var_a), (sum_b, var_b)) = (moments(a), moments(b));
    match (var_a == 0, var_b == 0) {
        (true, true) => Rho {
            sign: 1,
            num: 1,
            den: 1,
        },
        (false, false) => {
            let dot: i128 = a
                .iter()
                .zip(b)
                .map(|(&x, &y)| i128::from(x) * i128::from(y))
                .sum();
            let n = m * dot - sum_a * sum_b;
            Rho {
                sign: n.signum() as i8,
                num: (n * n) as u128,
                den: (var_a * var_b) as u128,
            }
        }
        _ => Rho {
            sign: 1,
            num: 1,
            den: 4,
        },
    }
}

/// The k discords by brute force: each window's best exact correlation among the windows
/// outside its zone, then the greedy pick farthest (smallest correlation) first.
fn brute_force(series: &[f64], m: usize, scale: Scale, k: usize) -> Vec<usize> {
    let ticks: Vec<Option<i64>> = series.iter().map(|&x| to_ticks(x, scale)).collect();
    let windows = ticks.len().saturating_sub(m - 1);
    let valid = |i: usize| -> Option<Vec<i64>> { ticks[i..i + m].iter().copied().collect() };
    let zone = ladde::exclusion::zone(m);
    let mut scored: Vec<(usize, Rho)> = Vec::new();
    for i in 0..windows {
        let Some(row) = valid(i) else { continue };
        let mut best: Option<Rho> = None;
        for j in (0..windows).filter(|&j| i.abs_diff(j) > zone) {
            let Some(neighbor) = valid(j) else { continue };
            let candidate = rho(&row, &neighbor);
            if best.is_none_or(|b| cmp_rho(candidate, b) == Ordering::Greater) {
                best = Some(candidate);
            }
        }
        if let Some(best) = best {
            scored.push((i, best));
        }
    }
    scored.sort_by(|a, b| cmp_rho(a.1, b.1).then(a.0.cmp(&b.0)));
    let mut taken: Vec<usize> = Vec::new();
    for (i, _) in scored {
        if taken.len() < k && taken.iter().all(|&t| t.abs_diff(i) > zone) {
            taken.push(i);
        }
    }
    taken
}

fn assert_matches_brute_force(series: &[f64], m: usize, scale: Scale) {
    for k in [0, 1, 3, 8] {
        let expected = brute_force(series, m, scale, k);
        let profile = two_sided_profile(series, nz(m), scale, nz(1));
        let mut reference = None;
        for threads in [1, 2, 7] {
            let found = discords(series, nz(m), scale, nz(threads), k);
            let indices: Vec<usize> = found
                .iter()
                .map(|d| d.index().get().unwrap() as usize)
                .collect();
            assert_eq!(indices, expected, "k {k}, threads {threads}");
            for d in &found {
                let nearest = profile[d.index().get().unwrap() as usize].1.nearest;
                assert_eq!(
                    d.squared_distance().to_bits(),
                    nearest.squared_distance().to_bits()
                );
            }
            let bits: Vec<(u64, u64)> = found
                .iter()
                .map(|d| (d.index().get().unwrap(), d.squared_distance().to_bits()))
                .collect();
            assert_eq!(*reference.get_or_insert(bits.clone()), bits);
        }
    }
}

fn scale() -> Scale {
    Scale::new(1.0).unwrap()
}

fn pattern(n: usize, seed: u64, levels: i64) -> Vec<f64> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) as i64 % levels) as f64
        })
        .collect()
}

proptest! {
    #![proptest_config(testkit::generators::seeded(40))]

    #[test]
    fn random_series_equal_brute_force(
        n in 12usize..90, m in 3usize..9, seed in any::<u64>(), levels in 2i64..9,
    ) {
        assert_matches_brute_force(&pattern(n.max(m + 4), seed, levels), m, scale());
    }

    #[test]
    fn scaled_copies_and_gaps_equal_brute_force(
        seed in any::<u64>(), m in 3usize..8, factor in 2i64..4, gap in 0usize..40,
    ) {
        let base = pattern(30, seed, 7);
        let mut series: Vec<f64> = base.clone();
        series.extend(base.iter().map(|x| x * factor as f64));
        series.extend(pattern(10, seed ^ 1, 7));
        if gap < series.len() {
            series[gap] = f64::NAN;
        }
        assert_matches_brute_force(&series, m, scale());
    }
}

#[test]
fn plateaus_equal_brute_force() {
    let mut series = pattern(80, 5, 6);
    series[10..30].fill(2.0);
    series[50..60].fill(2.0);
    series[40] = f64::NAN;
    assert_matches_brute_force(&series, 5, scale());
}

#[test]
fn a_series_shorter_than_a_window_has_no_discords() {
    assert!(discords(&[1.0, 2.0], nz(4), scale(), nz(1), 3).is_empty());
}
