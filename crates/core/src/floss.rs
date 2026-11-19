//! FLOSS: the corrected arc curve of a window of left nearest-neighbor indices.
//!
//! An *arc* joins a row `k` to its left neighbor `a < k`; the arc count at cut `c` is the
//! number of arcs with `a <= c < k`. An arc whose neighbor lies before the window is *clipped*:
//! it still exists and crosses every cut from the window's first (as if `a` were the row just
//! before the window). Only `NONE`, a self index and a future index make no arc. A regime
//! boundary has few arcs over it, so the count is divided by what a series without structure
//! would give (the *idealized* count) and clamped at 1: the corrected arc curve (CAC), low
//! where a regime changes.
//!
//! # The idealized count (our own closed form, not STUMPY's simulated one)
//!
//! STUMPY's one-directional curve is a beta density fitted to one random draw of indices and
//! scaled by least squares, so it depends on a seed and on libm. Ours is the exact expected
//! arc count under one stated model, computed with `+`, `*` and `/` only.
//!
//! *Model.* The window holds `W` rows `0..W` starting at global row `s` (the stream's age at
//! the window). `q = ceil(m/4) + 1` is the smallest eligible distance (a neighbor at distance
//! `<= ceil(m/4)` is never eligible) and `h` the lookback horizon: the largest distance a
//! stream searches (`w - m` for a `BoundedStream`; `usize::MAX` for a `GrowingStream`).
//! Row `j` (global `g = s + j`) has an eligible neighbor iff `g >= q`, and then its left
//! neighbor is uniform over the global rows `max(0, g-h) ..= g-q`. A neighbor before the
//! window counts as in the clipping rule above. `h < q` means no row has a neighbor.
//!
//! *Derivation.* Row `j > c` puts an arc over cut `c` exactly when its (clipped) neighbor is
//! `<= c` in window coordinates. Let `n` be the number of eligible neighbors of `j` and
//! `p(j, c)` that probability.
//! - `j` in `c+1 ..= c+q` (and `g >= q`): the newest eligible neighbor is `j-q <= c`, so every
//!   neighbor qualifies and `p = 1`. Call the number of such rows `n1(c)`.
//! - `j >= c+q+1` with `g <= h` (neighbors reach back to global row 0): `n = g-q+1`, and the
//!   qualifying ones are global rows `0 ..= s+c`, so `p = (s+c+1)/(s+j-q+1)`. Summing over
//!   `j` in `c+q+1 ..= min(W-1, h-s)` gives `A(c) = (s+c+1)·(H[s+jb-q+1] - H[s+c+1])` with
//!   `jb = min(W-1, h-s)` and `H[k] = 1 + 1/2 + ... + 1/k`, `H[0] = 0` (0 if the range is empty).
//! - `j >= c+q+1` with `g > h` (neighbors are `j-h ..= j-q`, `n = h-q+1`): the qualifying ones
//!   are `j-h ..= c`, so `p = (c+h+1-j)/(h-q+1)` for `j <= c+h` and 0 beyond. Summing over
//!   `j` in `max(c+q+1, h-s+1) ..= min(W-1, c+h)` is an integer arithmetic series `T`, so
//!   `B(c) = T/(h-q+1)`.
//!
//! By linearity of expectation, `IAC(c) = (n1(c) + A(c)) + B(c)`. A window older than the
//! horizon (`s >= h`) behaves as `s = h`, so `A` vanishes and no table is needed.
//!
//! *Determinism and cost.* Each cut needs one difference `H[b] - H[a]` with `b - a` at most the
//! window width, computed in closed form by `harmonic_difference` (pinned `libm`, no fused
//! multiply-add), so the curve has the same bits on every target, costs constant time and
//! memory per cut at any start row, and its relative error does not grow with the stream's age.
//!
//! *Where the minimum lands.* On a regime change the left-neighbor curve's minimum leads
//! STUMPY's right-neighbor `floss` by roughly `m/3` to `m/2` samples (measured: 11 to 26
//! positions at `m` of 40 and 50). That follows from causality, not from a defect: a left
//! neighbor already sees the change from the first window that straddles it, while a right
//! neighbor sees it only once the straddling windows have found a match. Time-reversed (every
//! neighbor in-window, as for STUMPY) our curve agrees with STUMPY's to within about 0.03 and
//! one position.
//!
//! *Validity.* The model assumes every row has an arc and that its neighbor is uniform. A row
//! whose entry is `NONE` (a constant or non-finite window) has no arc and lowers the curve.

use alloc::collections::VecDeque;
use alloc::vec;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

use crate::capacity::{clamp_count, reserve_for};
use crate::entry::Entry;
use crate::index::SampleIndex;

/// Rows below this are summed directly; from it on the asymptotic series is used.
const HARMONIC_CUTOFF: u128 = 128;

/// `H(b) - H(a) = 1/(a+1) + ... + 1/b` for `a <= b`, in constant time and memory.
///
/// Below [`HARMONIC_CUTOFF`] the terms are added in ascending size with a compensated sum. From
/// it on, with `p = 1/a`, `r = 1/b`, `u = p - r = (b-a)/(ab)`, the Euler-Maclaurin series
/// `H(n) = ln n + g + 1/(2n) - 1/(12n^2) + 1/(120n^4) - 1/(252n^6) + ...` gives
/// `ln(b/a) - u/2 + u(p+r)/12 - u(p+r)(p^2+r^2)/120 + u(p+r)(p^4+p^2r^2+r^4)/252`, each
/// difference written without cancellation and `ln(b/a)` as `log1p((b-a)/a)`. A range that
/// straddles the cutoff is the direct part up to it plus the closed form from it.
///
/// Error: the truncated next term is `1/(240n^8)`, at most `4e-19` relative to the result at the
/// cutoff and falling as `a^-8`; every other operation is correctly rounded with relative
/// error `2^-53` and none cancels, so the result is within 4 ulp for every `a <= b` (the
/// direct sum adds at most `HARMONIC_CUTOFF` positive terms, compensated). The bound holds at
/// any age: nothing accumulates across calls.
fn harmonic_difference(a: u128, b: u128) -> f64 {
    if b <= a {
        return 0.0;
    }
    if a >= HARMONIC_CUTOFF {
        return harmonic_series(a, b);
    }
    let direct_end = b.min(HARMONIC_CUTOFF);
    let mut sum = 0.0f64;
    let mut carry = 0.0f64;
    let mut k = direct_end;
    while k > a {
        let term = 1.0 / (k as f64);
        let next = sum + term;
        carry += if sum.abs() >= term.abs() {
            (sum - next) + term
        } else {
            (term - next) + sum
        };
        sum = next;
        k -= 1;
    }
    let direct = sum + carry;
    if b > HARMONIC_CUTOFF {
        direct + harmonic_series(HARMONIC_CUTOFF, b)
    } else {
        direct
    }
}

/// The closed form of [`harmonic_difference`] for `HARMONIC_CUTOFF <= a < b`.
fn harmonic_series(a: u128, b: u128) -> f64 {
    let (a, b, d) = (a as f64, b as f64, (b - a) as f64);
    let (p, r) = (1.0 / a, 1.0 / b);
    let u = d * p * r;
    let s1 = p + r;
    let (p2, r2) = (p * p, r * r);
    let t2 = u * s1 / 12.0;
    let t4 = u * s1 * (p2 + r2) / 120.0;
    let t6 = u * s1 * ((p2 * p2 + p2 * r2) + r2 * r2) / 252.0;
    libm::log1p(d / a) + ((t2 - u / 2.0) + (t6 - t4))
}

fn clamp_age(window_start: u64, horizon: usize) -> usize {
    usize::try_from(window_start).map_or(horizon, |s| s.min(horizon))
}

/// The cut-by-cut arithmetic behind [`ideal_arc_curve`].
fn ideal_with(width: usize, exclusion: usize, horizon: usize, window_start: u64) -> Vec<f64> {
    let q = exclusion + 1;
    if horizon < q {
        return vec![0.0; width];
    }
    let h = horizon;
    let s = clamp_age(window_start, h);
    (0..width)
        .map(|c| {
            let last = width - 1;
            let first_certain = (c + 1).max(q.saturating_sub(s));
            let last_certain = c.saturating_add(q).min(last);
            let certain = if last_certain >= first_certain {
                last_certain - first_certain + 1
            } else {
                0
            };
            let from = c + q + 1;
            let to_a = last.min(h - s);
            let a = if from <= to_a {
                let lo = (s as u128) + (c as u128) + 1;
                let hi = (s as u128) + (to_a as u128) - (q as u128) + 1;
                (lo as f64) * harmonic_difference(lo, hi)
            } else {
                0.0
            };
            let from_b = from.max((h - s).saturating_add(1));
            let to_b = last.min(c.saturating_add(h));
            let b = if from_b <= to_b {
                let top = (c as u128) + (h as u128) + 1;
                let t_lo = top - (to_b as u128);
                let t_hi = top - (from_b as u128);
                let series = (t_lo + t_hi) * ((to_b - from_b + 1) as u128) / 2;
                (series as f64) / ((h - q + 1) as f64)
            } else {
                0.0
            };
            ((certain as f64) + a) + b
        })
        .collect()
}

/// The expected arc count at every cut of a `width`-row window starting at global row
/// `window_start`, with exclusion zone `exclusion` (`ceil(m/4)`) and lookback `horizon`,
/// under the model in the module docs. Entry `c` is the expected number of arcs over the cut
/// between rows `c` and `c + 1`.
pub fn ideal_arc_curve(
    width: usize,
    exclusion: usize,
    horizon: usize,
    window_start: u64,
) -> Vec<f64> {
    ideal_with(width, exclusion, horizon, window_start)
}

/// Which neighbors make arcs.
#[derive(Clone, Copy)]
enum Neighbors {
    /// Left neighbors only; one before the window is clipped to its first row.
    Left,
    /// Neighbors on either side, all inside the window; one outside it makes no arc.
    Either,
}

/// Exact integer arc counts: entry `c` counts rows `k` whose neighbor `a` (in local
/// coordinates) satisfies `min(a, k) <= c < max(a, k)`. `NONE` and a self index contribute no
/// arc. With [`Neighbors::Left`] a future index makes no arc and an index before the window is
/// clipped, not dropped.
fn arc_counts(indices: &[SampleIndex], window_start: u64, neighbors: Neighbors) -> Vec<i64> {
    let mut delta = vec![0_i64; indices.len() + 1];
    for (k, neighbor) in indices.iter().enumerate() {
        let Some(global) = neighbor.get() else {
            continue;
        };
        let a = global.saturating_sub(window_start);
        let span = match neighbors {
            Neighbors::Left if global < window_start.saturating_add(k as u64) => {
                Some((a as usize, k))
            }
            Neighbors::Either if global >= window_start && a < indices.len() as u64 => {
                let a = a as usize;
                Some((a.min(k), a.max(k)))
            }
            _ => None,
        };
        if let Some((low, high)) = span {
            delta[low] += 1;
            delta[high] -= 1;
        }
    }
    let mut running = 0_i64;
    delta.truncate(indices.len());
    for d in &mut delta {
        running += *d;
        *d = running;
    }
    delta
}

/// `arcs / ideal` per cut, clamped at 1, 1 where `ideal` is 0, and 1 in the end zones.
fn correct(counts: &[i64], ideal: &[f64], l: usize, excl_factor: usize) -> Vec<f64> {
    let width = counts.len();
    let edge = l.saturating_mul(excl_factor);
    counts
        .iter()
        .zip(ideal)
        .enumerate()
        .map(|(c, (&arcs, &expected))| {
            if c < edge || c >= width - edge || expected == 0.0 {
                1.0
            } else {
                let ratio = (arcs as f64) / expected;
                if ratio > 1.0 { 1.0 } else { ratio }
            }
        })
        .collect()
}

/// The corrected arc curve of the window whose rows are `window_start ..` and whose left
/// indices are `left_indices` (global sample indices; width is the slice length).
///
/// Entry `c` is `min(1, arcs(c) / ideal(c))` with `ideal` from [`ideal_arc_curve`], and 1
/// where `ideal(c)` is 0 (no arc can cross there). The first and last `l · excl_factor`
/// entries (STUMPY's end convention, `l` the regime length) are set to 1; a window shorter
/// than twice that is all 1.
pub fn corrected_arc_curve(
    left_indices: &[SampleIndex],
    window_start: u64,
    exclusion: usize,
    horizon: usize,
    l: usize,
    excl_factor: usize,
) -> Vec<f64> {
    let counts = arc_counts(left_indices, window_start, Neighbors::Left);
    let ideal = ideal_arc_curve(left_indices.len(), exclusion, horizon, window_start);
    correct(&counts, &ideal, l, excl_factor)
}

/// One row's share of the two-sided ideal curve, as increments of its second difference.
struct SecondDifference {
    exact: Vec<i64>,
    inexact: Vec<f64>,
    shared_count: usize,
}

impl SecondDifference {
    /// Adds `numerator / count` at `at`; rows whose `count` is the common one stay integers.
    fn add(&mut self, at: usize, numerator: i64, count: usize) {
        if count == self.shared_count {
            self.exact[at] += numerator;
        } else {
            self.inexact[at] += (numerator as f64) / (count as f64);
        }
    }

    /// Both sums, twice, back to values: the exact part over the common count.
    fn curve(self, width: usize) -> Vec<f64> {
        let (mut slope_exact, mut value_exact) = (0_i64, 0_i64);
        let (mut slope_inexact, mut value_inexact) = (0.0_f64, 0.0_f64);
        let shared = self.shared_count as f64;
        (0..width)
            .map(|c| {
                slope_exact += self.exact[c];
                value_exact += slope_exact;
                slope_inexact += self.inexact[c];
                value_inexact += slope_inexact;
                if c + 1 == width {
                    0.0
                } else {
                    let exact = if self.shared_count == 0 {
                        0.0
                    } else {
                        (value_exact as f64) / shared
                    };
                    (exact + value_inexact).max(0.0)
                }
            })
            .collect()
    }
}

/// The expected arc count at every cut of a `width`-row window when each row's neighbor is
/// uniform over the rows at distance greater than `exclusion` on either side (the model of the
/// module docs with no horizon and neighbors in both directions). Entry `c` is the expected
/// number of arcs over the cut between rows `c` and `c + 1`.
///
/// Row `j` with `n` eligible neighbors puts an arc over cut `c` with probability
/// `(eligible neighbors on the other side of c) / n`, which is piecewise linear in `c`; every
/// row adds a few breakpoints to a second difference, so the whole curve costs `O(width)`.
/// Rows far from both ends share one `n`; their part is kept in integers and divided once.
pub fn two_sided_ideal_arc_curve(width: usize, exclusion: usize) -> Vec<f64> {
    let q = exclusion + 1;
    let eligible = |j: usize| (j + 1).saturating_sub(q) + width.saturating_sub(j + q);
    let shared_count = (width + 1).saturating_sub(2 * q);
    let mut diff = SecondDifference {
        exact: vec![0; width + 2],
        inexact: vec![0.0; width + 2],
        shared_count,
    };
    for j in 0..width {
        let n = eligible(j);
        if n == 0 {
            continue;
        }
        let below = (j + 1).saturating_sub(q);
        let above_from = j + q;
        if j == 0 {
            diff.add(0, n as i64, n);
            diff.add(1, -(n as i64), n);
        } else {
            if below > 0 {
                diff.add(0, 1, n);
                diff.add(1, -1, n);
            }
            if below > 1 {
                // Slope +1 over cuts 1 ..= below - 1.
                diff.add(1, 1, n);
                diff.add(below, -1, n);
            }
            let jump = (n as i64) - 2 * (below as i64);
            diff.add(j, jump, n);
            diff.add(j + 1, -jump, n);
        }
        if above_from < width {
            diff.add(above_from, -1, n);
        }
    }
    diff.curve(width)
}

/// The smallest ideal count a cut is divided by, as in STUMPY, so an empty ideal never divides
/// by zero.
const IDEAL_FLOOR: f64 = 1e-10;

/// The corrected arc curve of a whole series' two-sided indices (`indices[k]` is row `k`'s
/// nearest neighbor on either side, `NONE` for none), as STUMPY's `fluss` computes it from its
/// `I`: `min(1, arcs / ideal)` per cut, the first and last `l * excl_factor` cuts set to 1.
/// `ideal` is `custom_ideal` when given, else [`two_sided_ideal_arc_curve`] for `exclusion`
/// (`ceil(m/4)`); an ideal entry of 0 is read as 1e-10. `l` must be at least 1. `None` when
/// `custom_ideal` is not as long as `indices`.
pub fn two_sided_corrected_arc_curve(
    indices: &[SampleIndex],
    exclusion: usize,
    l: usize,
    excl_factor: usize,
    custom_ideal: Option<&[f64]>,
) -> Option<Vec<f64>> {
    let mut ideal = match custom_ideal {
        Some(ideal) if ideal.len() == indices.len() => ideal.to_vec(),
        Some(_) => return None,
        None => two_sided_ideal_arc_curve(indices.len(), exclusion),
    };
    for expected in &mut ideal {
        if *expected == 0.0 {
            *expected = IDEAL_FLOOR;
        }
    }
    let counts = arc_counts(indices, 0, Neighbors::Either);
    Some(correct(&counts, &ideal, l, excl_factor))
}

/// STUMPY's regime extraction: `n_regimes - 1` times take the first minimum of `curve`, then
/// set the `excl_factor * l` cuts on either side of it to 1 so the next pick lies elsewhere.
/// Returns at most one location per window (STUMPY would pad with repeats of an
/// already returned location up to `n_regimes - 1`).
pub fn regime_locations(
    curve: &[f64],
    n_regimes: usize,
    l: usize,
    excl_factor: usize,
) -> Vec<usize> {
    if curve.is_empty() {
        return Vec::new();
    }
    let reach = l.saturating_mul(excl_factor);
    let mut left = curve.to_vec();
    let picks = clamp_count(n_regimes, curve.len().saturating_add(1)).saturating_sub(1);
    let mut locations = Vec::with_capacity(picks);
    for _ in 0..picks {
        let mut at = 0;
        for (c, &value) in left.iter().enumerate() {
            if value < left[at] {
                at = c;
            }
        }
        locations.push(at);
        let from = at.saturating_sub(reach);
        let to = at.saturating_add(reach).min(left.len());
        left[from..to].fill(1.0);
    }
    locations
}

/// STUMPY's `fluss` from a two-sided index: the curve of
/// [`two_sided_corrected_arc_curve`] and the regime locations of [`regime_locations`] over it.
pub fn fluss(
    indices: &[SampleIndex],
    exclusion: usize,
    l: usize,
    n_regimes: usize,
    excl_factor: usize,
    custom_ideal: Option<&[f64]>,
) -> Option<(Vec<f64>, Vec<usize>)> {
    let curve = two_sided_corrected_arc_curve(indices, exclusion, l, excl_factor, custom_ideal)?;
    let locations = regime_locations(&curve, n_regimes, l, excl_factor);
    Some((curve, locations))
}

/// Streaming FLOSS: keeps the last `width` rows' left indices from a stream and reports the
/// corrected arc curve of that window. The curve after tick `t` is a pure function of
/// the left indices of rows `..=t`, so it equals [`corrected_arc_curve`] over the same
/// window and a future sample never changes a past curve.
///
/// Rows must arrive consecutively, as a stream returns them; a row that is not the successor
/// of the last one restarts the window at that row.
#[derive(Clone, Debug)]
pub struct Floss {
    exclusion: usize,
    horizon: usize,
    width: usize,
    l: usize,
    excl_factor: usize,
    window: VecDeque<SampleIndex>,
    /// The row of `window`'s newest entry; meaningless while `window` is empty.
    newest: u64,
}

impl Floss {
    /// `m` is the stream's window length (it fixes the exclusion zone `ceil(m/4)`), `width`
    /// the number of rows kept, `horizon` how far back the stream looks for a neighbor
    /// (`w - m` for a `BoundedStream`; `usize::MAX` for a `GrowingStream`), `l` the regime length and `excl_factor` the end zone
    /// factor (STUMPY's 5). Arcs to rows before the window are clipped (see the module docs).
    /// The window reserves `width` rows in full, up to a fixed limit of 2^24 past which it grows
    /// during warm-up only.
    pub fn new(
        m: NonZeroUsize,
        width: NonZeroUsize,
        horizon: usize,
        l: usize,
        excl_factor: usize,
    ) -> Self {
        Self {
            exclusion: crate::exclusion::zone(m.get()),
            horizon,
            width: width.get(),
            l,
            excl_factor,
            window: VecDeque::with_capacity(reserve_for(width.get(), width.get())),
            newest: 0,
        }
    }

    /// Folds in the stream's next `(row, entry)`; only its left index is used.
    pub fn update(&mut self, row: SampleIndex, entry: Entry) {
        let Some(row) = row.get() else { return };
        if !self.window.is_empty() && self.newest.checked_add(1) != Some(row) {
            self.window.clear();
        }
        if self.window.len() == self.width {
            self.window.pop_front();
        }
        self.window.push_back(entry.index());
        self.newest = row;
    }

    /// The row of the window's oldest entry; `None` before the first update.
    pub fn window_start(&self) -> Option<u64> {
        let len = self.window.len() as u64;
        if len == 0 {
            None
        } else {
            Some(self.newest - (len - 1))
        }
    }

    /// The corrected arc curve over the current window (empty before the first update).
    pub fn curve(&self) -> Vec<f64> {
        let Some(start) = self.window_start() else {
            return Vec::new();
        };
        let (front, back) = self.window.as_slices();
        let mut left = Vec::with_capacity(self.window.len());
        left.extend_from_slice(front);
        left.extend_from_slice(back);
        let counts = arc_counts(&left, start, Neighbors::Left);
        let ideal = ideal_with(left.len(), self.exclusion, self.horizon, start);
        correct(&counts, &ideal, self.l, self.excl_factor)
    }

    /// The lowest value of the curve outside the forced end zones, as `(local position,
    /// value)`; ties go to the smaller position. `None` when the window has no interior.
    pub fn min_interior(&self) -> Option<(usize, f64)> {
        let curve = self.curve();
        let edge = self.l.saturating_mul(self.excl_factor);
        let end = curve.len().saturating_sub(edge);
        let mut best: Option<(usize, f64)> = None;
        for (c, &value) in curve.iter().enumerate().take(end).skip(edge) {
            if best.is_none_or(|(_, v)| value < v) {
                best = Some((c, value));
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::{
        Floss, Neighbors, arc_counts, corrected_arc_curve, fluss, harmonic_difference,
        ideal_arc_curve, regime_locations, two_sided_corrected_arc_curve,
        two_sided_ideal_arc_curve,
    };
    use crate::bounded::BoundedStream;
    use crate::entry::Entry;
    use crate::index::SampleIndex;
    use crate::naive::naive_left_profile;
    use crate::quantize::{Scale, to_ticks};
    use crate::stream::GrowingStream;
    use crate::stream_api::Stream;
    use core::num::NonZeroUsize;
    use proptest::prelude::*;
    use std::vec::Vec;

    fn bits(curve: &[f64]) -> Vec<u64> {
        curve.iter().map(|v| v.to_bits()).collect()
    }

    fn idx(v: &[Option<u64>]) -> Vec<SampleIndex> {
        v.iter()
            .map(|x| x.map_or(SampleIndex::NONE, |i| SampleIndex::new(i).unwrap()))
            .collect()
    }

    /// The model's expectation by direct double loop: row `j` (global `g = s + j`) is uniform
    /// over the global rows `g-h ..= g-q`; count the neighbors whose clipped local position
    /// is `<= c`.
    fn ideal_by_definition(width: usize, e: usize, h: usize, s: usize) -> Vec<f64> {
        let q = e + 1;
        (0..width)
            .map(|c| {
                let mut total = 0.0;
                for j in (c + 1)..width {
                    let g = s + j;
                    if g < q {
                        continue;
                    }
                    let (lo, hi) = (g.saturating_sub(h), g - q);
                    if lo > hi {
                        continue;
                    }
                    let n = hi - lo + 1;
                    let covered = if s + c >= lo {
                        hi.min(s + c) - lo + 1
                    } else {
                        0
                    };
                    total += covered as f64 / n as f64;
                }
                total
            })
            .collect()
    }

    #[test]
    fn arc_counts_of_a_known_window() {
        // Arcs (0,1), (0,2), (1,3) cross cuts 0..=3 as 2, 2, 1, 0.
        let left = idx(&[None, Some(0), Some(0), Some(1)]);
        assert_eq!(arc_counts(&left, 0, Neighbors::Left), [2, 2, 1, 0]);
    }

    #[test]
    fn arcs_before_the_window_are_clipped_and_non_arcs_dropped() {
        // Window starts at 10. Row 1 points before it (clipped to cut 0), row 2 at itself and
        // row 3 at the future make no arc, row 4 points at the window's first row.
        let left = idx(&[None, Some(3), Some(12), Some(13), Some(10)]);
        assert_eq!(arc_counts(&left, 10, Neighbors::Left), [2, 1, 1, 1, 0]);
    }

    #[test]
    fn ideal_without_exclusion_is_the_harmonic_form() {
        let ideal = ideal_arc_curve(3, 0, usize::MAX, 0);
        assert_eq!(ideal, [1.5, 1.0, 0.0]);
    }

    #[test]
    fn ideal_with_exclusion_keeps_certain_arcs() {
        // Width 4, e = 1: row 2 has one eligible neighbor (row 0), row 3 has two (rows 0, 1).
        // Cut 0: row 2 certainly, row 3 half the time. Cut 1: both certainly. Cut 2: row 3.
        let ideal = ideal_arc_curve(4, 1, usize::MAX, 0);
        assert_eq!(ideal, [1.5, 2.0, 1.0, 0.0]);
    }

    #[test]
    fn ideal_with_a_horizon_is_a_band() {
        // e = 0, horizon 2: row j looks at j-2 and j-1. By hand: [1.5, 1.5, 1.5, 1, 0],
        // both inside a long-running stream and from the stream's start.
        let want = [1.5, 1.5, 1.5, 1.0, 0.0];
        assert_eq!(ideal_arc_curve(5, 0, 2, 10), want);
        assert_eq!(ideal_arc_curve(5, 0, 2, 0), want);
        // A horizon shorter than the exclusion zone leaves no arc at all.
        assert_eq!(ideal_arc_curve(3, 2, 2, 5), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn corrected_curve_is_ratio_clamped_with_ends_set() {
        let left = idx(&[None, Some(0), Some(0), Some(1)]);
        let ideal = ideal_arc_curve(4, 0, usize::MAX, 0);
        let cac = corrected_arc_curve(&left, 0, 0, usize::MAX, 0, 5);
        for c in 0..4 {
            let want = if ideal[c] == 0.0 {
                1.0
            } else {
                (([2.0, 2.0, 1.0, 0.0][c]) / ideal[c]).min(1.0)
            };
            assert_eq!(cac[c].to_bits(), want.to_bits());
        }
        // With l = 1, excl_factor = 1 the first and last entries are forced to 1.
        let ends = corrected_arc_curve(&left, 0, 0, usize::MAX, 1, 1);
        assert_eq!((ends[0], ends[3]), (1.0, 1.0));
        // A window shorter than twice the edge zone is all ones.
        assert!(
            corrected_arc_curve(&left, 0, 0, usize::MAX, 2, 5)
                .iter()
                .all(|&v| v == 1.0)
        );
    }

    #[test]
    fn only_the_edge_zone_is_forced_to_one() {
        // No arcs at all: every cut outside the two-cut edge zones is 0.
        let left = idx(&[None; 12]);
        let cac = corrected_arc_curve(&left, 0, 1, usize::MAX, 1, 2);
        let want = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
        assert_eq!(cac, want);
    }

    #[test]
    fn empty_window_has_an_empty_curve() {
        assert!(corrected_arc_curve(&[], 0, 1, usize::MAX, 1, 5).is_empty());
    }

    /// The two-sided model by direct double loop: row `j` is uniform over the rows at distance
    /// at least `e + 1`; count its neighbors on the other side of cut `c`.
    fn two_sided_ideal_by_definition(width: usize, e: usize) -> Vec<f64> {
        let q = e + 1;
        (0..width)
            .map(|c| {
                let mut total = 0.0;
                for j in 0..width {
                    let eligible: Vec<usize> = (0..width).filter(|&i| i.abs_diff(j) >= q).collect();
                    if eligible.is_empty() {
                        continue;
                    }
                    let across = eligible.iter().filter(|&&i| (i <= c) != (j <= c)).count();
                    total += across as f64 / eligible.len() as f64;
                }
                total
            })
            .collect()
    }

    /// Arcs by definition: every row with a neighbor in the window crosses the cuts from the
    /// smaller to the larger of the two.
    fn two_sided_counts_by_definition(indices: &[Option<usize>]) -> Vec<i64> {
        (0..indices.len())
            .map(|c| {
                indices
                    .iter()
                    .enumerate()
                    .filter(|&(k, i)| i.is_some_and(|i| i.min(k) <= c && c < i.max(k)))
                    .count() as i64
            })
            .collect()
    }

    fn sided(indices: &[Option<usize>]) -> Vec<SampleIndex> {
        indices
            .iter()
            .map(|i| i.map_or(SampleIndex::NONE, |i| SampleIndex::new(i as u64).unwrap()))
            .collect()
    }

    #[test]
    fn two_sided_arcs_cross_from_the_smaller_to_the_larger_row() {
        // Arcs 0-2, 1-3, and 2-0 again; row 3 has none and row 4 points at itself.
        let indices = sided(&[Some(2), Some(3), Some(0), None, Some(4)]);
        assert_eq!(arc_counts(&indices, 0, Neighbors::Either), [2, 3, 1, 0, 0]);
    }

    #[test]
    fn two_sided_arcs_ignore_indices_outside_the_window() {
        let indices = sided(&[Some(9), Some(0)]);
        assert_eq!(arc_counts(&indices, 0, Neighbors::Either), [1, 0]);
    }

    #[test]
    fn two_sided_ideal_of_a_tiny_window_by_hand() {
        // Width 3, every other row eligible: cut 0 is crossed by row 0 for sure and by rows 1
        // and 2 half the time each; cut 1 likewise by row 2 and rows 0 and 1; cut 2 by none.
        assert_eq!(two_sided_ideal_arc_curve(3, 0), [2.0, 2.0, 0.0]);
        assert_eq!(two_sided_ideal_arc_curve(0, 0), Vec::<f64>::new());
        assert_eq!(two_sided_ideal_arc_curve(5, 9), [0.0; 5]);
    }

    #[test]
    fn regimes_are_the_repeated_minimum_with_an_exclusion_each_side() {
        let mut curve = [1.0; 30];
        curve[10] = 0.2;
        curve[14] = 0.1;
        curve[17] = 0.3;
        curve[25] = 0.4;
        // Reach 3 hides cuts 11..=16 around 14: 10 and 17 survive, in order of value.
        assert_eq!(regime_locations(&curve, 4, 1, 3), [14, 10, 17]);
        // Reach 4 hides 10..=17.
        assert_eq!(regime_locations(&curve, 3, 1, 4), [14, 25]);
        assert_eq!(regime_locations(&curve, 1, 1, 4), Vec::<usize>::new());
        // The first of equal minima wins.
        assert_eq!(regime_locations(&[1.0, 0.5, 0.5], 2, 1, 0), [1]);
    }

    #[test]
    fn end_zones_are_exactly_l_times_excl_factor_cuts() {
        // With no arcs at all every interior cut is 0, so the 1s are exactly the zones.
        let width = 40;
        let indices = [None; 40];
        let curve = two_sided_corrected_arc_curve(&sided(&indices), 1, 3, 4, None).unwrap();
        for (c, &v) in curve.iter().enumerate() {
            let in_zone = c < 12 || c >= width - 12;
            assert_eq!(v == 1.0, in_zone, "cut {c}: {v}");
        }
    }

    #[test]
    fn a_custom_ideal_replaces_the_default_and_zero_reads_as_tiny() {
        let indices = sided(&[Some(2), Some(3), Some(0), Some(1), None]);
        // Arcs: 0-2, 1-3, 2-0, 3-1 cross cuts as 2, 4, 2, 0, 0.
        let ideal = [4.0, 4.0, 0.0, 0.0, 0.0];
        let curve = two_sided_corrected_arc_curve(&indices, 0, 1, 0, Some(&ideal)).unwrap();
        assert_eq!(curve, [0.5, 1.0, 1.0, 0.0, 0.0]);
        assert!(two_sided_corrected_arc_curve(&indices, 0, 1, 0, Some(&ideal[1..])).is_none());
    }

    #[test]
    fn fluss_returns_the_curve_and_its_regimes() {
        let indices = sided(&[Some(2), Some(3), Some(0), Some(1), None, None, None, None]);
        let (curve, locations) = fluss(&indices, 0, 1, 2, 0, None).unwrap();
        assert_eq!(locations, regime_locations(&curve, 2, 1, 0));
        assert_eq!(curve.len(), 8);
    }

    proptest! {
        #![proptest_config(testkit::generators::seeded_default())]
        #[test]
        fn two_sided_ideal_matches_its_definition(width in 0usize..40, e in 0usize..8) {
            let got = two_sided_ideal_arc_curve(width, e);
            let want = two_sided_ideal_by_definition(width, e);
            prop_assert_eq!(got.len(), want.len());
            for (g, w) in got.iter().zip(&want) {
                prop_assert!((g - w).abs() <= 1e-9 * (1.0 + w.abs()), "{g} vs {w}");
            }
        }

        #[test]
        fn two_sided_counts_match_their_definition(
            rows in prop::collection::vec(prop::option::of(0usize..50), 0..40),
        ) {
            let got = arc_counts(&sided(&rows), 0, Neighbors::Either);
            let in_window: Vec<Option<usize>> =
                rows.iter().map(|i| i.filter(|&i| i < rows.len())).collect();
            prop_assert_eq!(got, two_sided_counts_by_definition(&in_window));
        }

        #[test]
        fn two_sided_curve_stays_in_zero_one(
            rows in prop::collection::vec(prop::option::of(0usize..50), 0..40),
            e in 0usize..6,
            l in 1usize..5,
            excl_factor in 0usize..4,
        ) {
            let curve = two_sided_corrected_arc_curve(&sided(&rows), e, l, excl_factor, None)
                .unwrap();
            prop_assert_eq!(curve.len(), rows.len());
            prop_assert!(curve.iter().all(|v| (0.0..=1.0).contains(v)));
        }
    }

    proptest! {
        #![proptest_config(testkit::generators::seeded_default())]
        #[test]
        fn ideal_matches_its_definition(
            width in 0usize..40,
            e in 0usize..6,
            h in prop_oneof![0usize..30, Just(usize::MAX)],
            s in 0usize..60,
        ) {
            let got = ideal_arc_curve(width, e, h, s as u64);
            let want = ideal_by_definition(width, e, h, s);
            prop_assert_eq!(got.len(), want.len());
            for (g, w) in got.iter().zip(&want) {
                prop_assert!((g - w).abs() <= 1e-9 * (1.0 + w.abs()), "{g} vs {w}");
            }
        }

        /// Non-arcs and clipped arcs are interchangeable with their kind: swapping an index
        /// that is NONE, a self or a future index for any other such value, or an index
        /// before the window for any other index before it, leaves the curve's bits. Moving
        /// the window and every index by the same amount changes nothing once the window is
        /// older than the horizon.
        #[test]
        fn irrelevant_inputs_and_shifts_do_not_matter(
            rows in prop::collection::vec((0u64..140, any::<u8>()), 1..40),
            start in 4u64..100,
            shift in 0u64..1000,
            e in 0usize..4,
            l in 0usize..4,
        ) {
            const HORIZON: usize = 4;
            let left: Vec<SampleIndex> =
                rows.iter().map(|&(a, _)| SampleIndex::new(a).unwrap()).collect();
            let base = corrected_arc_curve(&left, start, e, HORIZON, l, 2);

            // Every value is in [0, 1]; the edges are 1.
            prop_assert!(base.iter().all(|&v| (0.0..=1.0).contains(&v)));
            let edge = (l * 2).min(base.len());
            prop_assert!(base[..edge].iter().all(|&v| v == 1.0));
            prop_assert!(base[base.len() - edge..].iter().all(|&v| v == 1.0));

            let mut swapped = left.clone();
            for (k, (slot, &(_, pick))) in swapped.iter_mut().zip(&rows).enumerate() {
                let global = slot.get().unwrap();
                let row = start + k as u64;
                if global >= row {
                    *slot = match pick % 3 {
                        0 => SampleIndex::NONE,
                        1 => SampleIndex::new(row).unwrap(),
                        _ => SampleIndex::new(row + 7).unwrap(),
                    };
                } else if global < start {
                    *slot = SampleIndex::new(u64::from(pick) % start).unwrap();
                }
            }
            let again = corrected_arc_curve(&swapped, start, e, HORIZON, l, 2);
            prop_assert_eq!(bits(&base), bits(&again));

            let moved: Vec<SampleIndex> =
                left.iter().map(|i| SampleIndex::new(i.get().unwrap() + shift).unwrap()).collect();
            let shifted = corrected_arc_curve(&moved, start + shift, e, HORIZON, l, 2);
            prop_assert_eq!(bits(&base), bits(&shifted));
        }
    }

    const TICK: f64 = 0.01;

    fn nz(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).unwrap()
    }

    fn walk(steps: &[(i8, bool)]) -> Vec<f64> {
        let mut level = 0i64;
        steps
            .iter()
            .map(|&(step, gap)| {
                level += i64::from(step % 3);
                if gap && step % 5 == 0 {
                    f64::NAN
                } else {
                    level as f64 * TICK
                }
            })
            .collect()
    }

    /// The curve after every row of `stream` fed `series`.
    fn curves(stream: &mut impl Stream, floss: &mut Floss, series: &[f64]) -> Vec<Vec<u64>> {
        let mut out = Vec::new();
        for &x in series {
            if let Some((row, entry)) = stream.update(x) {
                floss.update(row, entry);
                out.push(bits(&floss.curve()));
            }
        }
        out
    }

    #[test]
    fn update_keeps_the_last_width_rows_and_restarts_on_a_gap() {
        let at = |i: u64| SampleIndex::new(i).unwrap();
        let mut floss = Floss::new(nz(4), nz(3), 100, 0, 5);
        assert_eq!(floss.window_start(), None);
        assert!(floss.curve().is_empty());
        for row in 0..5 {
            floss.update(at(row), Entry::NONE);
        }
        assert_eq!(floss.window_start(), Some(2));
        assert_eq!(floss.curve().len(), 3);
        floss.update(at(9), Entry::NONE);
        assert_eq!(floss.window_start(), Some(9));
        assert_eq!(floss.curve().len(), 1);
        floss.update(SampleIndex::NONE, Entry::NONE);
        assert_eq!(floss.window_start(), Some(9));
    }

    #[test]
    fn min_interior_takes_the_smallest_position_among_ties() {
        let at = |i: u64| SampleIndex::new(i).unwrap();
        // No arcs: the interior is all zeros, so the minimum is the first interior cut.
        let mut floss = Floss::new(nz(4), nz(12), 100, 1, 2);
        for row in 0..12 {
            floss.update(at(row), Entry::NONE);
        }
        assert_eq!(floss.min_interior(), Some((2, 0.0)));
        // Everything is end zone: no interior.
        let mut tiny = Floss::new(nz(4), nz(3), 100, 1, 2);
        for row in 0..3 {
            tiny.update(at(row), Entry::NONE);
        }
        assert_eq!(tiny.min_interior(), None);
    }

    #[test]
    fn min_interior_is_the_lowest_value_not_the_highest() {
        let at = |i: u64| SampleIndex::new(i).unwrap();
        let entry = |i: u64| Entry::new(1.0, 0.0, at(i)).unwrap();
        // Every row points back to row 0, except rows 5..9, which have no neighbor: the
        // curve dips where the missing arcs would have crossed.
        let mut floss = Floss::new(nz(4), nz(12), 100, 0, 5);
        for row in 0..12u64 {
            let back = if (5..9).contains(&row) { u64::MAX } else { 0 };
            let entry = if back == u64::MAX {
                Entry::NONE
            } else {
                entry(back)
            };
            floss.update(at(row), entry);
        }
        let curve = floss.curve();
        let low = curve.iter().copied().fold(f64::INFINITY, f64::min);
        let high = curve.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(low < high);
        let first = curve.iter().position(|&v| v == low).unwrap();
        assert_eq!(floss.min_interior(), Some((first, low)));
    }

    proptest! {
        #![proptest_config(testkit::generators::seeded_default())]
        /// Appending future samples never changes a past curve, on either stream.
        #[test]
        fn past_curves_ignore_the_future(
            steps in prop::collection::vec((any::<i8>(), any::<bool>()), 8..90),
            m in 2usize..=5,
            width in 3usize..=18,
            cut in 0usize..90,
            l in 0usize..3,
        ) {
            let series = walk(&steps);
            let cut = cut.min(series.len());
            let scale = Scale::new(TICK).unwrap();

            let mut g1 = GrowingStream::new(nz(m), scale);
            let mut f1 = Floss::new(nz(m), nz(width), usize::MAX, l, 2);
            let prefix = curves(&mut g1, &mut f1, &series[..cut]);
            let mut g2 = GrowingStream::new(nz(m), scale);
            let mut f2 = Floss::new(nz(m), nz(width), usize::MAX, l, 2);
            let full = curves(&mut g2, &mut f2, &series);
            prop_assert_eq!(&full[..prefix.len()], &prefix[..]);

            let w = nz(width + m - 1);
            let mut b1 = BoundedStream::new(nz(m), w, scale).unwrap();
            let mut f3 = Floss::new(nz(m), nz(width), width - 1, l, 2);
            let prefix = curves(&mut b1, &mut f3, &series[..cut]);
            let mut b2 = BoundedStream::new(nz(m), w, scale).unwrap();
            let mut f4 = Floss::new(nz(m), nz(width), width - 1, l, 2);
            let full = curves(&mut b2, &mut f4, &series);
            prop_assert_eq!(&full[..prefix.len()], &prefix[..]);
        }
    }

    proptest! {
        #![proptest_config(testkit::generators::seeded_default())]
        /// Streaming FLOSS equals `corrected_arc_curve` over `naive`'s left indices of the
        /// same window, bit for bit, at every tick, gaps included.
        #[test]
        fn streaming_equals_the_curve_over_naive_left_indices(
            steps in prop::collection::vec((any::<i8>(), any::<bool>()), 6..80),
            m in 2usize..=5,
            width in 3usize..=18,
            l in 0usize..3,
            factor in 0usize..4,
        ) {
            let series = walk(&steps);
            let scale = Scale::new(TICK).unwrap();
            let samples: Vec<Option<i64>> =
                series.iter().map(|&x| to_ticks(x, scale)).collect();
            let naive = naive_left_profile(&samples, m);

            let mut stream = GrowingStream::new(nz(m), scale);
            let mut floss = Floss::new(nz(m), nz(width), usize::MAX, l, factor);
            let mut tick = 0;
            for &x in &series {
                let Some((row, entry)) = stream.update(x) else { continue };
                floss.update(row, entry);
                let newest = row.get().unwrap() as usize;
                let first = (newest + 1).saturating_sub(width);
                let left: Vec<SampleIndex> =
                    naive[first..=newest].iter().map(|(_, e)| e.index()).collect();
                let want = corrected_arc_curve(&left, first as u64, crate::exclusion::zone(m), usize::MAX, l, factor);
                prop_assert_eq!(bits(&floss.curve()), bits(&want), "tick {}", tick);
                tick += 1;
            }
            prop_assert_eq!(tick, naive.len());
        }
    }

    proptest! {
        #![proptest_config(testkit::generators::seeded_default())]
        /// Bounded streaming FLOSS, with horizon `w - m`, equals `corrected_arc_curve` over the
        /// bounded stream's own published left indices of the same window, bit for bit at
        /// every tick, gaps included.
        #[test]
        fn bounded_streaming_equals_the_curve_over_its_own_left_indices(
            steps in prop::collection::vec((any::<i8>(), any::<bool>()), 6..90),
            m in 2usize..=5,
            extra in 0usize..=12,
            width in 3usize..=18,
            l in 0usize..3,
            factor in 0usize..4,
        ) {
            let series = walk(&steps);
            let scale = Scale::new(TICK).unwrap();
            let w = m + extra;
            let mut stream = BoundedStream::new(nz(m), nz(w), scale).unwrap();
            let mut floss = Floss::new(nz(m), nz(width), extra, l, factor);
            let mut published: Vec<SampleIndex> = Vec::new();
            for &x in &series {
                let Some((row, entry)) = stream.update(x) else { continue };
                floss.update(row, entry);
                published.push(entry.index());
                let first = published.len().saturating_sub(width);
                let want = corrected_arc_curve(
                    &published[first..],
                    first as u64,
                    crate::exclusion::zone(m),
                    extra,
                    l,
                    factor,
                );
                prop_assert_eq!(bits(&floss.curve()), bits(&want), "row {}", published.len() - 1);
            }
        }
    }

    fn ulps_apart(got: f64, want: f64) -> u64 {
        got.to_bits().abs_diff(want.to_bits())
    }

    const HARMONIC_CASES: [(u128, u128, u64); 34] = [
        (36, 97, 4607024911970301045),
        (35, 95, 4607088429233086840),
        (36, 98, 4607116822166777994),
        (38, 104, 4607168948880231378),
        (84, 108, 4598175087775050656),
        (0, 1, 4607182418800017408),
        (0, 5, 4612324028374598724),
        (1, 128, 4616677298325953703),
        (100, 127, 4597741327443119233),
        (100, 128, 4598022802419829889),
        (100, 129, 4598238657482626656),
        (127, 128, 4575657221408423936),
        (127, 129, 4580125909410776056),
        (128, 129, 4575587398158387184),
        (128, 300, 4605826986522430872),
        (255, 256, 4571153621781053440),
        (1000, 1100, 4591528968604158440),
        (127, 5000, 4615444664715954172),
        (5000, 5100, 4581364375002501461),
        (9391422, 9392422, 4547484853822789218),
        (1071321661075906, 1071321661076906, 4427156938098543183),
        (95, 96, 4577158421284214101),
        (354646831598904, 354646831598905, 4389151242875562693),
        (270, 277, 4583021163525806029),
        (504702, 504802, 4551441247501227357),
        (526122838479200, 526122838479264, 4413842874096122934),
        (77070, 77134, 4560795211653835965),
        (187358672084442, 187358672084506, 4420293537816125458),
        (161, 162, 4573766821564836320),
        (1810713, 1810813, 4543274644030435616),
        (34, 35, 4583892375012758557),
        (240, 1240, 4610067195129107734),
        (322, 386, 4595690263338508861),
        (
            4611686018427387904,
            4611686018427388004,
            4357514114457665536,
        ),
    ];
    /// `(width, exclusion, horizon, window_start, bits of each cut's exact value)`.
    type Curve = (usize, usize, usize, u64, &'static [u64]);
    const IDEAL_CURVES: [Curve; 8] = [
        (
            6,
            1,
            usize::MAX,
            4611686018427387904,
            &[
                4617315517961601024,
                4616189618054758400,
                4613937818241073152,
                4611686018427387904,
                4607182418800017408,
                0,
            ],
        ),
        (
            6,
            1,
            usize::MAX,
            1125899906842624,
            &[
                4617315517961601018,
                4616189618054758394,
                4613937818241073150,
                4611686018427387904,
                4607182418800017408,
                0,
            ],
        ),
        (
            9,
            2,
            usize::MAX,
            125,
            &[
                4620562960070779693,
                4619480704941102655,
                4618389584457828432,
                4617289667881532001,
                4616172428743203551,
                4613937818241073152,
                4611686018427387904,
                4607182418800017408,
                0,
            ],
        ),
        (
            9,
            0,
            usize::MAX,
            120,
            &[
                4620442970027444863,
                4619380136663521366,
                4618308074611836864,
                4617226858902373153,
                4616083508655143668,
                4613884903161789948,
                4611650834055299072,
                4607182418800017408,
                0,
            ],
        ),
        (
            12,
            3,
            usize::MAX,
            10000000,
            &[
                4622382065966133908,
                4621819116406777367,
                4621256166791125841,
                4620693216556229777,
                4619567317099746823,
                4618441417530673935,
                4617315517849011123,
                4616189618054758400,
                4613937818241073152,
                4611686018427387904,
                4607182418800017408,
                0,
            ],
        ),
        (
            40,
            4,
            usize::MAX,
            100,
            &[
                4630007227159926485,
                4629907150289478587,
                4629805693639733088,
                4629702870606605499,
                4629496971719925251,
                4629285938449445481,
                4629072249754657120,
                4628855930452609782,
                4628637004900778085,
                4628415497009710511,
                4628191430255218294,
                4627964827690125081,
                4627735711955596950,
                4627504105292071381,
                4627270029549802736,
                4627033506199040954,
                4626794556339859253,
                4626553200711645837,
                4626309459702273856,
                4626063353356963130,
                4625814901386846482,
                4625564123177252886,
                4625311037795719040,
                4624914510689981389,
                4624399223179044887,
                4623879432068481014,
                4623355173101143957,
                4622826481457008306,
                4622293391766361552,
                4621755938122587500,
                4621214154094556361,
                4620642927795144125,
                4619542235538555615,
                4618433077869133703,
                4617315517961601024,
                4616189618054758400,
                4613937818241073152,
                4611686018427387904,
                4607182418800017408,
                0,
            ],
        ),
        (
            7,
            1,
            usize::MAX,
            1099511627776,
            &[
                4618441417868433408,
                4617315517961594880,
                4616189618054752256,
                4613937818241071104,
                4611686018427387904,
                4607182418800017408,
                0,
            ],
        ),
        (
            30,
            2,
            50,
            8589934592,
            &[
                4626797706239541248,
                4626668696875215531,
                4626533823448875008,
                4626393085960519680,
                4626246484410149547,
                4626094018797764608,
                4625935689123364864,
                4625771495386950315,
                4625601437588520960,
                4625425515728076800,
                4625243729805617835,
                4624915342332788736,
                4624528314239811584,
                4624129558022804821,
                4623719073681768448,
                4623296861216702464,
                4622862920627606869,
                4622417251914481664,
                4621959855077326848,
                4621490730116142421,
                4621009877030928384,
                4620341373961240576,
                4619332755294694059,
                4618300680380088320,
                4617245149217423360,
                4616142705558639957,
                4613937818241073152,
                4611686018427387904,
                4607182418800017408,
                0,
            ],
        ),
    ];

    #[test]
    fn harmonic_differences_are_within_four_ulp_of_the_exact_value() {
        for (a, b, want) in HARMONIC_CASES {
            let got = harmonic_difference(a, b);
            let want = f64::from_bits(want);
            assert!(
                ulps_apart(got, want) <= 4,
                "H({b}) - H({a}): {got} vs {want}"
            );
        }
        assert_eq!(harmonic_difference(5, 5), 0.0);
        assert_eq!(harmonic_difference(9, 5), 0.0);
    }

    #[test]
    fn the_ideal_curve_is_within_eight_ulp_of_the_exact_rational_curve_at_any_start_row() {
        for (width, e, h, s, want) in IDEAL_CURVES {
            let got = ideal_arc_curve(width, e, h, s);
            assert_eq!(got.len(), want.len());
            for (c, (g, w)) in got.iter().zip(want).enumerate() {
                let w = f64::from_bits(*w);
                assert!(ulps_apart(*g, w) <= 8, "start {s} cut {c}: {g} vs {w}");
            }
        }
    }

    #[test]
    fn a_stream_far_from_its_start_still_matches_the_curve_function() {
        let at = |i: u64| SampleIndex::new(i).unwrap();
        let start = (1u64 << 62) + 3;
        let mut floss = Floss::new(nz(4), nz(20), usize::MAX, 2, 2);
        let mut left = Vec::new();
        for k in 0..20u64 {
            let neighbor = at(start + k.saturating_sub(1 + k % 5));
            floss.update(at(start + k), Entry::new(1.0, 0.0, neighbor).unwrap());
            left.push(neighbor);
        }
        let want = corrected_arc_curve(&left, start, 1, usize::MAX, 2, 2);
        assert_eq!(bits(&floss.curve()), bits(&want));
    }
}
