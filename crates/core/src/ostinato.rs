//! The consensus motif of a set of series, following STUMPY 1.14.1's `ostinato` with the order
//! between distances decided exactly.
//!
//! The radius of a window is its largest distance to the nearest window of each other series;
//! the consensus motif is a window with the smallest radius.
//!
//! Search: for each series `j` in order, join it against the next series `h` (the last against
//! the first) and take its windows by ascending join distance. A window starts at its join
//! distance and is raised to its nearest distance in each series other than `j` and `h`. It
//! replaces the best window so far only if its radius is strictly smaller, so the first of equal
//! radii wins. Once a radius reaches the best one the window is dropped, and when the join
//! distance alone does, the rest of the series is.
//!
//! Central pass: the best window's nearest window in each series is a candidate when that
//! distance is close to the best radius. In series order, a candidate replaces the best window
//! when its own largest nearest distance is close to the best radius and its mean nearest
//! distance is smaller than the mean of the window it would replace. "Close" is NumPy's
//! `isclose` with its default tolerances, and the means are NumPy's, summed pairwise.
//!
//! Differences from STUMPY: the order of windows, the comparison of radii and the nearest window
//! of a series (ties to the smaller start) are exact rather than on rounded distances; the
//! closeness and the means stay on the f64 distances, as STUMPY has them. A set with no window
//! that meets every series gives radius infinity at series 0, window 0, as STUMPY does.

use alloc::vec::Vec;
use core::num::NonZeroUsize;

use core::cmp::Ordering;

use crate::nearest::Sharing;
use crate::nonnorm::NonNormalized;
use crate::pattern::{Measure, TickedSeries};
use crate::quantize::Scale;
use crate::stats::Series;
use crate::znorm::ZNorm;

/// The consensus motif: the window `window` of series `series`, whose distances to the nearest
/// window of every series are at most `radius`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Consensus {
    pub radius: f64,
    pub series: usize,
    pub window: usize,
}

/// NumPy's `isclose(a, b)` with its default tolerances, `b` the reference value.
fn is_close(a: f64, b: f64) -> bool {
    a == b || (a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-8 + 1e-5 * b.abs())
}

/// NumPy's pairwise sum of `values`.
pub(crate) fn pairwise_sum(values: &[f64]) -> f64 {
    const BLOCK: usize = 128;
    let n = values.len();
    if n < 8 {
        return values.iter().fold(0.0, |sum, &x| sum + x);
    }
    if n <= BLOCK {
        let mut lanes = [0.0; 8];
        lanes.copy_from_slice(&values[..8]);
        let body = n - n % 8;
        for chunk in values[8..body].chunks_exact(8) {
            for (lane, &x) in lanes.iter_mut().zip(chunk) {
                *lane += x;
            }
        }
        let head = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
            + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
        return values[body..].iter().fold(head, |sum, &x| sum + x);
    }
    let half = n / 2;
    let half = half - half % 8;
    pairwise_sum(&values[..half]) + pairwise_sum(&values[half..])
}

fn mean(values: &[f64]) -> f64 {
    pairwise_sum(values) / values.len() as f64
}

fn order<M: Measure>(a: &M::Exact, b: &M::Exact) -> Ordering {
    M::compare(a, b, Sharing::Nothing)
}

/// The distance and start of each series' nearest window to the window `window` of series
/// `series`; infinity at start 0 for a series with none, as `argmin` over infinities gives.
fn across_series_nearest<M: Measure>(
    set: &[TickedSeries<M>],
    series: usize,
    window: usize,
) -> (Vec<f64>, Vec<usize>) {
    set.iter()
        .map(|other| match other.nearest_to(&set[series], window) {
            Some(nearest) => (
                other.distance_of(&nearest),
                M::index(&M::hit(&nearest)).expect("a window has a start") as usize,
            ),
            None => (f64::INFINITY, 0),
        })
        .unzip()
}

/// The window with the smallest radius, `None` when no window has a finite one.
fn smallest_radius<M: Measure>(
    series: &[(Series<'_>, Scale)],
    set: &[TickedSeries<M>],
    m: NonZeroUsize,
    threads: NonZeroUsize,
) -> Option<(M::Exact, usize, usize)> {
    let count = series.len();
    let mut best: Option<(M::Exact, usize, usize)> = None;
    let reached = |best: &Option<(M::Exact, usize, usize)>, radius: &M::Exact| {
        best.as_ref()
            .is_some_and(|(b, _, _)| order::<M>(radius, b).is_ge())
    };
    for j in 0..count {
        let h = (j + 1) % count;
        let join = M::ab_join(
            series[j].0,
            series[h].0,
            m,
            series[j].1,
            series[h].1,
            threads,
        );
        let mut by_distance: Vec<(M::Exact, usize)> = join
            .iter()
            .enumerate()
            .filter_map(|(window, hit)| {
                let neighbor = M::index(hit)? as usize;
                let exact = set[j]
                    .exact_across(window, &set[h], neighbor)
                    .expect("a joined pair has windows with distances");
                Some((exact, window))
            })
            .collect();
        drop(join);
        by_distance.sort_by(|x, y| order::<M>(&x.0, &y.0).then(x.1.cmp(&y.1)));
        for (mut radius, window) in by_distance {
            if reached(&best, &radius) {
                break;
            }
            let mut finite = true;
            for i in (0..count).filter(|&i| i != j && i != h) {
                let Some(nearest) = set[i].nearest_to(&set[j], window) else {
                    finite = false;
                    break;
                };
                if order::<M>(&nearest, &radius).is_gt() {
                    radius = nearest;
                }
                if reached(&best, &radius) {
                    break;
                }
            }
            if finite && !reached(&best, &radius) {
                best = Some((radius, j, window));
            }
        }
    }
    best
}

/// The consensus motif of `series` (each with the tick size it is quantized at) at window
/// length `m`; `None` for an empty set or a series with no window (fewer than `m` samples). The
/// same for every `threads`.
pub fn ostinato<'a, S: Into<Series<'a>> + Copy>(
    series: &[(S, Scale)],
    m: NonZeroUsize,
    threads: NonZeroUsize,
) -> Option<Consensus> {
    let series: Vec<(Series<'a>, Scale)> =
        series.iter().map(|&(s, scale)| (s.into(), scale)).collect();
    consensus_in::<ZNorm>(&series, m, threads)
}

/// [`ostinato`] under the Euclidean distance on the raw values, as STUMPY's `aamp_ostinato`: every
/// series is quantized at `scale`, and the radius is in the caller's units. A window whose
/// distance needs more than 128 bits has none, as a window with a gap does.
pub fn aamp_ostinato(
    series: &[&[f64]],
    scale: Scale,
    m: NonZeroUsize,
    threads: NonZeroUsize,
) -> Option<Consensus> {
    let set: Vec<(Series<'_>, Scale)> = series.iter().map(|&s| (s.into(), scale)).collect();
    consensus_in::<NonNormalized>(&set, m, threads)
}

fn consensus_in<M: Measure>(
    series: &[(Series<'_>, Scale)],
    m: NonZeroUsize,
    threads: NonZeroUsize,
) -> Option<Consensus> {
    if series.is_empty() || series.iter().any(|(s, _)| s.len() < m.get()) {
        return None;
    }
    let set: Vec<TickedSeries<M>> = series
        .iter()
        .map(|&(s, scale)| TickedSeries::new(s, m, scale))
        .collect();
    let Some((best, mut best_series, mut best_window)) = smallest_radius(series, &set, m, threads)
    else {
        return Some(Consensus {
            radius: f64::INFINITY,
            series: 0,
            window: 0,
        });
    };
    let radius = set[best_series].distance_of(&best);

    let (nearest_radii, nearest_starts) = across_series_nearest(&set, best_series, best_window);
    let mut best_mean = mean(&nearest_radii);
    for candidate in 0..set.len() {
        if !is_close(nearest_radii[candidate], radius) {
            continue;
        }
        let start = nearest_starts[candidate];
        let (candidate_radii, _) = across_series_nearest(&set, candidate, start);
        let farthest = candidate_radii
            .iter()
            .fold(f64::NEG_INFINITY, |a, &x| a.max(x));
        if is_close(farthest, radius) && mean(&candidate_radii) < best_mean {
            best_series = candidate;
            best_window = start;
            best_mean = mean(&candidate_radii);
        }
    }
    Some(Consensus {
        radius,
        series: best_series,
        window: best_window,
    })
}

#[cfg(test)]
mod tests;
