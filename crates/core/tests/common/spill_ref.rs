//! The exact reference for the cold route, with no width cutoff: every statistic and cross term
//! is an arbitrary-precision integer, candidates are ordered by exact rationals, and the reported
//! `ρ̂` and distance follow the entry formulas with each integer rounded to the nearest `f64`
//! through its decimal string (Rust's parser rounds correctly). It shares no code with the
//! library's wide integer or its ranking.
#![allow(dead_code)]

use num_bigint::{BigInt, Sign};
use std::cmp::Ordering;
use std::string::ToString;
use std::vec::Vec;

/// `(row, ρ̂ bits, squared-distance bits, neighbor index)`; `u64::MAX` for no neighbor.
pub type Bits = (u64, u64, u64, u64);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Missing,
    Constant,
    Valid,
}

struct Window {
    kind: Kind,
    sum: BigInt,
    variance: BigInt,
}

fn nearest_f64(value: &BigInt) -> f64 {
    value.to_string().parse().expect("a decimal integer parses")
}

fn windows(ticks: &[i64], had: &[bool], m: usize) -> Vec<Window> {
    if ticks.len() < m {
        return Vec::new();
    }
    (0..=ticks.len() - m)
        .map(|start| {
            let span = &ticks[start..start + m];
            let sum: BigInt = span.iter().map(|&t| BigInt::from(t)).sum();
            let sum_sq: BigInt = span
                .iter()
                .map(|&t| BigInt::from(t) * BigInt::from(t))
                .sum();
            let variance = BigInt::from(m) * sum_sq - &sum * &sum;
            let kind = if had[start..start + m].iter().any(|&h| !h) {
                Kind::Missing
            } else if variance.sign() == Sign::NoSign {
                Kind::Constant
            } else {
                Kind::Valid
            };
            Window {
                kind,
                sum,
                variance,
            }
        })
        .collect()
}

/// A candidate of one row: its neighbor and its exact correlation as `(sign, ρ² numerator,
/// ρ² denominator)`.
struct Pair {
    index: usize,
    sign: i8,
    numerator: BigInt,
    denominator: BigInt,
    cross: BigInt,
}

fn pair(row: &Window, neighbor: &Window, index: usize, cross: BigInt) -> Pair {
    match (row.kind, neighbor.kind) {
        (Kind::Constant, Kind::Constant) => Pair {
            index,
            sign: 1,
            numerator: 1.into(),
            denominator: 1.into(),
            cross,
        },
        (Kind::Constant, _) | (_, Kind::Constant) => Pair {
            index,
            sign: 1,
            numerator: 1.into(),
            denominator: 4.into(),
            cross,
        },
        _ => Pair {
            index,
            sign: match cross.sign() {
                Sign::Minus => -1,
                Sign::NoSign => 0,
                Sign::Plus => 1,
            },
            numerator: &cross * &cross,
            denominator: &row.variance * &neighbor.variance,
            cross,
        },
    }
}

/// `Greater` when `a` has the larger exact correlation.
fn compare(a: &Pair, b: &Pair) -> Ordering {
    if a.sign != b.sign {
        return a.sign.cmp(&b.sign);
    }
    if a.sign == 0 {
        return Ordering::Equal;
    }
    let magnitude = (&a.numerator * &b.denominator).cmp(&(&b.numerator * &a.denominator));
    if a.sign > 0 {
        magnitude
    } else {
        magnitude.reverse()
    }
}

fn rho_hat(row: &Window, neighbor: &Window, pair: &Pair) -> f64 {
    match (row.kind, neighbor.kind) {
        (Kind::Constant, Kind::Constant) => 1.0,
        (Kind::Constant, _) | (_, Kind::Constant) => 0.5,
        _ => {
            let inv_row = 1.0 / nearest_f64(&row.variance).sqrt();
            let inv_neighbor = 1.0 / nearest_f64(&neighbor.variance).sqrt();
            ((nearest_f64(&pair.cross) * inv_row) * inv_neighbor).clamp(-1.0, 1.0)
        }
    }
}

fn squared_distance(m: usize, row: &Window, neighbor: &Window, pair: &Pair, rho: f64) -> f64 {
    let valid = row.kind == Kind::Valid && neighbor.kind == Kind::Valid;
    if valid && rho > 0.0 {
        let gap = &row.variance * &neighbor.variance - &pair.cross * &pair.cross;
        let inv_row = 1.0 / nearest_f64(&row.variance).sqrt();
        let inv_neighbor = 1.0 / nearest_f64(&neighbor.variance).sqrt();
        let inverse_product = inv_row * inv_neighbor;
        (2.0 * m as f64) * nearest_f64(&gap) * inverse_product * inverse_product / (1.0 + rho)
    } else {
        (2.0 * m as f64) * (1.0 - rho)
    }
}

/// Every row's candidates against the left neighbors outside the exclusion zone, nearest first by
/// the exact order, ties to the smaller index.
fn ranked(ticks: &[i64], had: &[bool], m: usize) -> (Vec<Window>, Vec<Vec<Pair>>) {
    let windows = windows(ticks, had, m);
    let exclusion = ladde::exclusion::zone(m);
    let mut rows = Vec::with_capacity(windows.len());
    for i in 0..windows.len() {
        let mut candidates = Vec::new();
        if windows[i].kind != Kind::Missing {
            for j in 0..i.saturating_sub(exclusion) {
                if windows[j].kind == Kind::Missing {
                    continue;
                }
                let dot: BigInt = (0..m)
                    .map(|k| BigInt::from(ticks[i + k]) * BigInt::from(ticks[j + k]))
                    .sum();
                let cross = BigInt::from(m) * dot - &windows[i].sum * &windows[j].sum;
                candidates.push(pair(&windows[i], &windows[j], j, cross));
            }
        }
        candidates.sort_by(|a, b| compare(b, a).then(a.index.cmp(&b.index)));
        rows.push(candidates);
    }
    (windows, rows)
}

/// The exact left profile of `ticks` (`had[p]` false for a filled tick), as entry bits. Walks
/// each diagonal with the exact dot-product recurrence and keeps only each row's best, so it
/// stays usable at long windows.
pub fn left_profile(ticks: &[i64], had: &[bool], m: usize) -> Vec<Bits> {
    left_profile_where(ticks, had, m, &|_, _| true)
}

/// [`left_profile`] over only the pairs `(row, neighbor)` for which `keep` holds: what the hot
/// passes find when they rank the pairs among the windows they can hold.
pub fn left_profile_where(
    ticks: &[i64],
    had: &[bool],
    m: usize,
    keep: &dyn Fn(usize, usize) -> bool,
) -> Vec<Bits> {
    let windows = windows(ticks, had, m);
    let count = windows.len();
    let exclusion = ladde::exclusion::zone(m);
    let mut best: Vec<Option<Pair>> = (0..count).map(|_| None).collect();
    for diagonal in exclusion + 1..count {
        let mut dot: BigInt = (0..m)
            .map(|k| BigInt::from(ticks[diagonal + k]) * BigInt::from(ticks[k]))
            .sum();
        for i in diagonal..count {
            let j = i - diagonal;
            if windows[i].kind != Kind::Missing && windows[j].kind != Kind::Missing && keep(i, j) {
                let cross = BigInt::from(m) * &dot - &windows[i].sum * &windows[j].sum;
                let candidate = pair(&windows[i], &windows[j], j, cross);
                let replaces = best[i].as_ref().is_none_or(|current| {
                    compare(&candidate, current).then(current.index.cmp(&candidate.index))
                        == Ordering::Greater
                });
                if replaces {
                    best[i] = Some(candidate);
                }
            }
            if i + 1 < count {
                dot += BigInt::from(ticks[i + m]) * BigInt::from(ticks[j + m]);
                dot -= BigInt::from(ticks[i]) * BigInt::from(ticks[j]);
            }
        }
    }
    best.iter()
        .enumerate()
        .map(|(i, candidate)| match candidate {
            None => (
                i as u64,
                f64::NEG_INFINITY.to_bits(),
                f64::INFINITY.to_bits(),
                u64::MAX,
            ),
            Some(best) => {
                let neighbor = &windows[best.index];
                let rho = rho_hat(&windows[i], neighbor, best);
                let distance = squared_distance(m, &windows[i], neighbor, best, rho);
                (
                    i as u64,
                    rho.to_bits(),
                    distance.to_bits(),
                    best.index as u64,
                )
            }
        })
        .collect()
}

/// The `k` nearest left neighbors of every row, as entry bits, nearest first.
pub fn left_topk(ticks: &[i64], had: &[bool], m: usize, k: usize) -> Vec<Vec<Bits>> {
    let (windows, rows) = ranked(ticks, had, m);
    rows.iter()
        .enumerate()
        .map(|(i, candidates)| {
            candidates
                .iter()
                .take(k)
                .map(|candidate| {
                    let neighbor = &windows[candidate.index];
                    let rho = rho_hat(&windows[i], neighbor, candidate);
                    let distance = squared_distance(m, &windows[i], neighbor, candidate, rho);
                    (
                        i as u64,
                        rho.to_bits(),
                        distance.to_bits(),
                        candidate.index as u64,
                    )
                })
                .collect()
        })
        .collect()
}

/// How many rows have a candidate that is exactly worse than the winner but whose `ρ̂` is within
/// `2ε` of the winner's: the rows where only the exact order decides.
pub fn rows_decided_by_the_exact_order(ticks: &[i64], had: &[bool], m: usize) -> usize {
    let (windows, rows) = ranked(ticks, had, m);
    let band = 2.0 / (1u64 << 49) as f64;
    rows.iter()
        .enumerate()
        .filter(|(i, candidates)| {
            let Some(best) = candidates.first() else {
                return false;
            };
            let best_rho = rho_hat(&windows[*i], &windows[best.index], best);
            candidates[1..].iter().any(|rival| {
                let rival_rho = rho_hat(&windows[*i], &windows[rival.index], rival);
                (best_rho - rival_rho).abs() <= band
            })
        })
        .count()
}

/// How many rows have a runner-up exactly as near as the winner (ties go to the smaller index).
pub fn rows_with_an_exact_tie(ticks: &[i64], had: &[bool], m: usize) -> usize {
    let (_, rows) = ranked(ticks, had, m);
    rows.iter()
        .filter(|candidates| {
            candidates.len() > 1 && compare(&candidates[0], &candidates[1]) == Ordering::Equal
        })
        .count()
}

/// The entry bits for "no neighbor".
fn none_bits(row: usize) -> Bits {
    (
        row as u64,
        f64::NEG_INFINITY.to_bits(),
        f64::INFINITY.to_bits(),
        u64::MAX,
    )
}

/// One series' ticks and which of them were measured.
pub type Series<'a> = (&'a [i64], &'a [bool]);

/// Every row of `rows` against every window of `cols` that `eligible(row, col)` admits: each row's
/// candidates nearest first by the exact order, ties to the smaller index.
fn between(
    rows: Series,
    cols: Series,
    m: usize,
    eligible: &dyn Fn(usize, usize) -> bool,
) -> (Vec<Window>, Vec<Window>, Vec<Vec<Pair>>) {
    let (row_windows, col_windows) = (windows(rows.0, rows.1, m), windows(cols.0, cols.1, m));
    let mut ranked_rows = Vec::with_capacity(row_windows.len());
    for (i, row) in row_windows.iter().enumerate() {
        let mut candidates = Vec::new();
        if row.kind != Kind::Missing {
            for (j, col) in col_windows.iter().enumerate() {
                if col.kind == Kind::Missing || !eligible(i, j) {
                    continue;
                }
                let dot: BigInt = (0..m)
                    .map(|k| BigInt::from(rows.0[i + k]) * BigInt::from(cols.0[j + k]))
                    .sum();
                let cross = BigInt::from(m) * dot - &row.sum * &col.sum;
                candidates.push(pair(row, col, j, cross));
            }
        }
        candidates.sort_by(|a, b| compare(b, a).then(a.index.cmp(&b.index)));
        ranked_rows.push(candidates);
    }
    (row_windows, col_windows, ranked_rows)
}

/// `candidate`'s entry bits for row `i`.
fn candidate_bits(
    m: usize,
    (rows, cols): (&[Window], &[Window]),
    i: usize,
    candidate: &Pair,
) -> Bits {
    let neighbor = &cols[candidate.index];
    let rho = rho_hat(&rows[i], neighbor, candidate);
    let distance = squared_distance(m, &rows[i], neighbor, candidate, rho);
    (
        i as u64,
        rho.to_bits(),
        distance.to_bits(),
        candidate.index as u64,
    )
}

/// The first of `candidates` that `keep` admits, as entry bits for row `i`, or "no neighbor".
fn first_bits(
    m: usize,
    windows: (&[Window], &[Window]),
    i: usize,
    candidates: &[Pair],
    keep: impl Fn(&Pair) -> bool,
) -> Bits {
    candidates
        .iter()
        .find(|candidate| keep(candidate))
        .map_or_else(|| none_bits(i), |c| candidate_bits(m, windows, i, c))
}

/// Every row's up to `k` nearest among the pairs `eligible` admits, as entry bits, nearest first.
fn topk_between(
    rows: Series,
    cols: Series,
    m: usize,
    k: usize,
    eligible: &dyn Fn(usize, usize) -> bool,
) -> Vec<Vec<Bits>> {
    let (row_windows, col_windows, ranked_rows) = between(rows, cols, m, eligible);
    ranked_rows
        .iter()
        .enumerate()
        .map(|(i, candidates)| {
            candidates
                .iter()
                .take(k)
                .map(|c| candidate_bits(m, (&row_windows, &col_windows), i, c))
                .collect()
        })
        .collect()
}

/// The `k` nearest right neighbors of every row (the windows after its exclusion zone), nearest
/// first.
pub fn right_topk(ticks: &[i64], had: &[bool], m: usize, k: usize) -> Vec<Vec<Bits>> {
    let exclusion = ladde::exclusion::zone(m);
    topk_between((ticks, had), (ticks, had), m, k, &|i, j| j > i + exclusion)
}

/// Every row's left nearest, right nearest and nearer of the two, as entry bits.
pub fn two_sided(ticks: &[i64], had: &[bool], m: usize) -> Vec<[Bits; 3]> {
    let exclusion = ladde::exclusion::zone(m);
    let eligible = |i: usize, j: usize| i.abs_diff(j) > exclusion;
    let (windows_of, _, ranked_rows) = between((ticks, had), (ticks, had), m, &eligible);
    ranked_rows
        .iter()
        .enumerate()
        .map(|(i, candidates)| {
            let w = (windows_of.as_slice(), windows_of.as_slice());
            [
                first_bits(m, w, i, candidates, |c| c.index < i),
                first_bits(m, w, i, candidates, |c| c.index > i),
                first_bits(m, w, i, candidates, |_| true),
            ]
        })
        .collect()
}

/// Every row's left nearest, right nearest and up to `k` nearest over both sides.
pub fn two_sided_topk(
    ticks: &[i64],
    had: &[bool],
    m: usize,
    k: usize,
) -> Vec<(Bits, Bits, Vec<Bits>)> {
    two_sided_topk_where(ticks, had, m, k, &|_, _| true)
}

/// [`two_sided_topk`] over only the pairs `(row, neighbor)` outside the exclusion zone that `keep`
/// admits.
pub fn two_sided_topk_where(
    ticks: &[i64],
    had: &[bool],
    m: usize,
    k: usize,
    keep: &dyn Fn(usize, usize) -> bool,
) -> Vec<(Bits, Bits, Vec<Bits>)> {
    let exclusion = ladde::exclusion::zone(m);
    let eligible = |i: usize, j: usize| i.abs_diff(j) > exclusion && keep(i, j);
    let (windows_of, _, ranked_rows) = between((ticks, had), (ticks, had), m, &eligible);
    ranked_rows
        .iter()
        .enumerate()
        .map(|(i, candidates)| {
            let w = (windows_of.as_slice(), windows_of.as_slice());
            (
                first_bits(m, w, i, candidates, |c| c.index < i),
                first_bits(m, w, i, candidates, |c| c.index > i),
                candidates
                    .iter()
                    .take(k)
                    .map(|c| candidate_bits(m, w, i, c))
                    .collect(),
            )
        })
        .collect()
}

/// Every window of `a` against every window of `b`: each row's up to `k` nearest, as entry bits.
pub fn join_topk(a: Series, b: Series, m: usize, k: usize) -> Vec<Vec<Bits>> {
    join_topk_where(a, b, m, k, &|_, _| true)
}

/// [`join_topk`] over only the pairs `(row of a, window of b)` that `keep` admits.
pub fn join_topk_where(
    a: Series,
    b: Series,
    m: usize,
    k: usize,
    keep: &dyn Fn(usize, usize) -> bool,
) -> Vec<Vec<Bits>> {
    topk_between(a, b, m, k, keep)
}

/// Per window, whether it has every sample present and a variance numerator of at least `2⁶³`:
/// the windows with no distance in the hot passes.
pub fn no_distance(ticks: &[i64], had: &[bool], m: usize) -> Vec<bool> {
    let limit = BigInt::from(1u64 << 63);
    windows(ticks, had, m)
        .iter()
        .map(|window| window.kind == Kind::Valid && window.variance >= limit)
        .collect()
}

/// Every window of `a` against every window of `b`: each row's nearest, or "no neighbor".
pub fn join(a: Series, b: Series, m: usize) -> Vec<Bits> {
    join_topk(a, b, m, 1)
        .into_iter()
        .enumerate()
        .map(|(row, mut kept)| kept.pop().unwrap_or_else(|| none_bits(row)))
        .collect()
}
