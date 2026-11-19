//! STUMPY's MPdist stated plainly with exact rational orders, for the tests that check MPdist or
//! what is built on it.
#![allow(dead_code)]

use core::cmp::Ordering;

use super::exact_pair::{self, cmp_rho, cross_pair};
use ladde::naive::Ticks;

/// How a test names the position: STUMPY's `percentage` or its `k`.
#[derive(Clone, Copy, Debug)]
pub enum Pick {
    Percentage(f64),
    Rank(usize),
}

/// A window's nearest window of the other series: exact correlation and f64 squared distance,
/// `None` without one; then the join's side (0 for A to B) and the window.
pub type JoinedRow = (Option<(exact_pair::Rho, f64)>, usize, usize);

/// `P_ABBA`, the A side first.
pub fn joined_nearest(ta: Ticks<'_>, tb: Ticks<'_>, m: usize) -> Vec<JoinedRow> {
    let mut joined = Vec::new();
    for (side, rows, neighbors) in [(0, ta, tb), (1, tb, ta)] {
        for row in 0..rows.len() + 1 - m {
            let mut best: Option<(exact_pair::Rho, f64)> = None;
            for neighbor in 0..neighbors.len() + 1 - m {
                if let Some((rho, d2)) = cross_pair(rows, neighbors, m, row, neighbor)
                    && best.is_none_or(|(b, _)| cmp_rho(rho, b) == Ordering::Greater)
                {
                    best = Some((rho, d2));
                }
            }
            joined.push((best, side, row));
        }
    }
    joined
}

/// STUMPY's `_mpdist`, `_compute_P_ABBA` and `_select_P_ABBA_value`; `None` where STUMPY would
/// clamp a percentage outside `[0, 1]` or a rank past the last position.
pub fn brute<'a, 'b>(
    (series_a, ta): (&[f64], impl Into<Ticks<'a>>),
    (series_b, tb): (&[f64], impl Into<Ticks<'b>>),
    m: usize,
    pick: Pick,
) -> Option<f64> {
    let (ta, tb): (Ticks<'a>, Ticks<'b>) = (ta.into(), tb.into());
    if ta.len() < m || tb.len() < m {
        return None;
    }
    let mut joined = joined_nearest(ta, tb, m);
    let total = joined.len();
    let position = match pick {
        Pick::Percentage(p) if (0.0..=1.0).contains(&p) => {
            ((p * (series_a.len() + series_b.len()) as f64).ceil() as usize).min(total - 1)
        }
        Pick::Rank(k) if k < total => k,
        // STUMPY clamps these; this library refuses them.
        _ => return None,
    };
    joined.retain(|(best, _, _)| best.is_some());
    joined.sort_by(|x, y| {
        cmp_rho(y.0.unwrap().0, x.0.unwrap().0)
            .then(x.1.cmp(&y.1))
            .then(x.2.cmp(&y.2))
    });
    let mut sorted: Vec<f64> = joined
        .iter()
        .map(|(best, _, _)| libm::sqrt(best.unwrap().1))
        .collect();
    sorted.resize(total, f64::INFINITY);
    if sorted[position].is_finite() {
        return Some(sorted[position]);
    }
    let finite_before = sorted[..position].iter().filter(|d| d.is_finite()).count();
    Some(sorted[finite_before.saturating_sub(1)])
}
