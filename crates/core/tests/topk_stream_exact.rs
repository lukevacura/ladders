#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! The streams that keep `k` neighbors per row against a reference that shares nothing with the
//! engines or with `naive_left_topk`: direct 128-bit sums of the ticks, the order of two
//! correlations by 256-bit cross-multiplication of their squares, and its own tie rule (equal
//! correlations go to the smaller index). The growing stream is checked against all earlier
//! windows, the bounded one against the windows inside its last `w` samples, under every forced
//! route.

#[path = "common/route.rs"]
mod route;

use core::cmp::Ordering;

use ethnum::I256;
use ladde::bounded::BoundedStream;
use ladde::entry::Entry;
use ladde::quantize::Scale;
use ladde::route::forced_route;
use ladde::stream::GrowingStream;
use route::*;

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Gap,
    Constant,
    Valid,
}

#[derive(Clone, Copy)]
struct Win {
    kind: Kind,
    sum: i128,
    /// `m·Σt² − (Σt)²`, the variance numerator.
    spread: i128,
}

fn windows(ticks: &[Option<i64>], m: usize) -> Vec<Win> {
    (0..(ticks.len() + 1).saturating_sub(m))
        .map(|i| {
            let window = &ticks[i..i + m];
            if window.iter().any(Option::is_none) {
                return Win {
                    kind: Kind::Gap,
                    sum: 0,
                    spread: 0,
                };
            }
            let t: Vec<i128> = window.iter().map(|t| i128::from(t.unwrap())).collect();
            let sum: i128 = t.iter().sum();
            let spread = m as i128 * t.iter().map(|t| t * t).sum::<i128>() - sum * sum;
            Win {
                kind: if spread == 0 {
                    Kind::Constant
                } else {
                    Kind::Valid
                },
                sum,
                spread,
            }
        })
        .collect()
}

/// A correlation as `sign · num / den`, `None` when either window is a gap.
#[derive(Clone, Copy)]
struct Rho {
    sign: i32,
    num: I256,
    den: I256,
}

fn rho(ticks: &[Option<i64>], wins: &[Win], i: usize, j: usize, m: usize) -> Option<Rho> {
    let (a, b) = (wins[i], wins[j]);
    let one = I256::from(1);
    match (a.kind, b.kind) {
        (Kind::Gap, _) | (_, Kind::Gap) => None,
        (Kind::Constant, Kind::Constant) => Some(Rho {
            sign: 1,
            num: one,
            den: one,
        }),
        (Kind::Constant, _) | (_, Kind::Constant) => Some(Rho {
            sign: 1,
            num: one,
            den: I256::from(4),
        }),
        _ => {
            let dot: i128 = (0..m)
                .map(|t| i128::from(ticks[i + t].unwrap()) * i128::from(ticks[j + t].unwrap()))
                .sum();
            let cross = m as i128 * dot - a.sum * b.sum;
            let cross = I256::from(cross);
            Some(Rho {
                sign: cross.signum().as_i32(),
                num: cross * cross,
                den: I256::from(a.spread) * I256::from(b.spread),
            })
        }
    }
}

/// `Less` when `a` is the nearer (larger correlation): compares `a.sign·a.num/a.den` with
/// `b.sign·b.num/b.den` by multiplying both sides by the positive `a.den·b.den`.
fn nearer(a: Rho, b: Rho) -> Ordering {
    let left = I256::from(a.sign) * a.num * b.den;
    let right = I256::from(b.sign) * b.num * a.den;
    right.cmp(&left)
}

/// The `k` nearest windows to `row` among `first..row` outside the exclusion zone, nearest first,
/// equal correlations to the smaller index; each with its correlation as an `f64`.
fn reference(
    ticks: &[Option<i64>],
    m: usize,
    k: usize,
    row: usize,
    first: usize,
) -> Vec<(usize, f64)> {
    let wins = windows(ticks, m);
    let zone = m / 4 + usize::from(!m.is_multiple_of(4));
    let mut candidates: Vec<(usize, Rho)> = (first..row)
        .filter(|j| row - j > zone)
        .filter_map(|j| rho(ticks, &wins, row, j, m).map(|r| (j, r)))
        .collect();
    candidates.sort_by(|a, b| nearer(a.1, b.1).then(a.0.cmp(&b.0)));
    candidates.truncate(k);
    candidates
        .into_iter()
        .map(|(j, r)| {
            (
                j,
                f64::from(r.sign) * (r.num.as_f64() / r.den.as_f64()).sqrt(),
            )
        })
        .collect()
}

fn series() -> Vec<(&'static str, Vec<f64>, usize)> {
    let mut state = 12345u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut out = Vec::new();
    for (name, n, m, shape) in [
        ("constant", 90usize, 4usize, 0u8),
        ("period 2", 100, 4, 1),
        ("period 4", 100, 5, 2),
        ("period 7", 110, 6, 3),
        ("few levels", 120, 4, 4),
        ("walk", 120, 5, 5),
        ("two levels far apart", 100, 4, 6),
        ("gaps", 130, 5, 7),
        ("constant runs", 120, 4, 8),
        ("repeats with offsets", 120, 6, 9),
    ] {
        let mut v: Vec<f64> = Vec::new();
        for t in 0..n {
            let r = next();
            v.push(match shape {
                0 => 5.0,
                1 => (t % 2) as f64,
                2 => (t % 4) as f64,
                3 => (t % 7) as f64,
                4 => (r % 3) as f64,
                5 => v.last().copied().unwrap_or(0.0) + (r % 7) as f64 - 3.0,
                6 if (t / 9) % 2 == 0 => (r % 5) as f64,
                6 => 4_000_000.0 + (r % 5) as f64,
                7 if t % 29 < 3 || t < 4 || t > n - 5 || t == 60 => f64::NAN,
                7 => (r % 4) as f64,
                8 if (t / 13) % 2 == 0 => 0.0,
                8 => (r % 2) as f64 * 1000.0,
                _ => ((t % 6) + 10 * (t / 30)) as f64,
            });
        }
        out.push((name, v, m));
    }
    out
}

fn ticks(series: &[f64]) -> Vec<Option<i64>> {
    series
        .iter()
        .map(|x| x.is_finite().then_some(*x as i64))
        .collect()
}

fn check(name: &str, row: usize, got: &[Entry], want: &[(usize, f64)], m: usize, how: &str) {
    assert_eq!(got.len(), want.len(), "{name} {how} row {row}: count");
    for (entry, &(j, rho)) in got.iter().zip(want) {
        assert_eq!(
            entry.index().get().unwrap() as usize,
            j,
            "{name} {how} row {row}"
        );
        assert!(
            (entry.rho() - rho).abs() < 1e-9,
            "{name} {how} row {row}: rho {} against {rho}",
            entry.rho()
        );
        assert!(
            (entry.squared_distance() - 2.0 * m as f64 * (1.0 - rho)).abs() < 1e-6 * m as f64,
            "{name} {how} row {row}: squared distance"
        );
    }
}

#[test]
fn both_streams_equal_the_independent_reference() {
    let scale = Scale::new(1.0).unwrap();
    let mut rows_with_several = 0usize;
    for (name, v, m) in series() {
        let ticks = ticks(&v);
        for k in [2usize, 3, 5, 16, 40] {
            for force in forces(0x9e37_79b9_7f4a_7c15) {
                let _forced = forced_route(force);
                let mut growing = GrowingStream::with_k(nz(m), scale, nz(k));
                for &x in &v {
                    if let Some((row, kept)) = growing.update_topk(x) {
                        let row = row.get().unwrap() as usize;
                        let want = reference(&ticks, m, k, row, 0);
                        check(
                            name,
                            row,
                            kept,
                            &want,
                            m,
                            &format!("growing k {k} {force:?}"),
                        );
                        rows_with_several += usize::from(want.len() >= 2);
                    }
                }
                for w in [m, m + 1, m + 7, m + 25] {
                    let mut bounded = BoundedStream::with_k(nz(m), nz(w), scale, nz(k)).unwrap();
                    for (t, &x) in v.iter().enumerate() {
                        if let Some((row, kept)) = bounded.update_topk(x) {
                            let row = row.get().unwrap() as usize;
                            assert_eq!(row + m, t + 1);
                            let first = (row + m).saturating_sub(w);
                            let want = reference(&ticks, m, k, row, first);
                            let how = format!("bounded w {w} k {k} {force:?}");
                            check(name, row, kept, &want, m, &how);
                        }
                    }
                }
            }
        }
    }
    assert!(rows_with_several > 1000);
}
