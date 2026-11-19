//! The exact reference: every row's left entry computed directly from `S`, `Q`, `V` and `N`
//! as exact 256-bit sums over its own window, never a recurrence. Every other engine is checked
//! against this, bit for bit, through the same final steps from `N` to an [`Entry`].

use alloc::vec::Vec;
use core::num::NonZeroUsize;

// Deliberately a second implementation, not the shared one: the oracle's arithmetic is
// independent of the engines' 192-bit helpers, so a bug in one cannot hide in the other.
use ethnum::I256;

use crate::entry::Entry;
use crate::index::SampleIndex;
use crate::nearest::{
    Candidate, ExactCorrelation, Nearest, nearer, variance_product, winner_entry,
};
use crate::stats::{Window as StatsWindow, WindowKind};
// Re-exported into this module's namespace so the test module's `use super::{...}` (unchanged
// since before the comparator moved into `nearest`) still resolves them.
#[cfg(test)]
use crate::estimate::EPSILON;
#[cfg(test)]
use crate::nearest::{Sharing, cmp_exact, cmp_wide_products};

const OVERFLOW_MESSAGE: &str =
    "naive reference value exceeds i128, the width of the engines' comparator";
const WIDE_OVERFLOW_MESSAGE: &str = "naive reference input exceeds 256 bits";

/// A window's exact classification: absent ticks beat flatness, which beats an out-of-range
/// variance (the same precedence `stats` uses on ticks).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Window {
    NonFinite,
    Constant,
    Valid { sum: i128, variance_numerator: i128 },
}

/// A window that can stand in a pair; `NonFinite` windows never reach this far.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Comparable {
    Constant,
    Valid { sum: i128, variance_numerator: i128 },
}

fn comparable(window: Window) -> Option<Comparable> {
    match window {
        Window::NonFinite => None,
        Window::Constant => Some(Comparable::Constant),
        Window::Valid {
            sum,
            variance_numerator,
        } => Some(Comparable::Valid {
            sum,
            variance_numerator,
        }),
    }
}

/// Direct 256-bit sums over `samples[start..start + m]`.
fn window_stats(samples: &[Option<i64>], start: usize, m: usize) -> Window {
    let ticks = &samples[start..start + m];
    if ticks.iter().any(Option::is_none) {
        return Window::NonFinite;
    }

    let mut sum = I256::ZERO;
    let mut sum_sq = I256::ZERO;
    let mut min = i64::MAX;
    let mut max = i64::MIN;
    for tick in ticks.iter().map(|tick| tick.expect("checked above")) {
        let tick_wide = I256::from(tick);
        sum = sum.checked_add(tick_wide).expect(WIDE_OVERFLOW_MESSAGE);
        let squared = tick_wide
            .checked_mul(tick_wide)
            .expect(WIDE_OVERFLOW_MESSAGE);
        sum_sq = sum_sq.checked_add(squared).expect(WIDE_OVERFLOW_MESSAGE);
        min = min.min(tick);
        max = max.max(tick);
    }

    if max == min {
        return Window::Constant;
    }

    let m_sum_sq = I256::from(m as u64)
        .checked_mul(sum_sq)
        .expect(WIDE_OVERFLOW_MESSAGE);
    let sum_squared = sum.checked_mul(sum).expect(WIDE_OVERFLOW_MESSAGE);
    let variance_numerator = m_sum_sq
        .checked_sub(sum_squared)
        .expect(WIDE_OVERFLOW_MESSAGE);

    if (I256::ONE..I256::new(1i128 << 63)).contains(&variance_numerator) {
        // In range, so it fits; the sum is at most `m·2^63`, which fits i128 for any real `m`.
        Window::Valid {
            sum: i128::try_from(sum).expect(OVERFLOW_MESSAGE),
            variance_numerator: i128::try_from(variance_numerator).expect(OVERFLOW_MESSAGE),
        }
    } else {
        Window::NonFinite
    }
}

/// A series of ticks, and optionally the caller's flag for each window being constant (one per
/// window): the reference's own reading of a constant-window predicate, directly on the ticks.
#[derive(Clone, Copy, Debug)]
pub struct Ticks<'a> {
    samples: &'a [Option<i64>],
    constant: Option<&'a [bool]>,
}

impl<'a> Ticks<'a> {
    /// `samples` with `flags[i]` saying whether the window at `i` is constant.
    pub fn with_constant_windows(samples: &'a [Option<i64>], flags: &'a [bool]) -> Self {
        Self {
            samples,
            constant: Some(flags),
        }
    }

    /// Every window of `m` ticks, classified directly: a window with a missing tick is
    /// non-finite whatever its flag; a flagged one is constant; an unflagged window of equal
    /// ticks has variance zero; the rest are as [`window_stats`] finds them.
    pub fn samples(&self) -> &'a [Option<i64>] {
        self.samples
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// The caller's flags, `None` for the default rule.
    pub fn constant_windows(&self) -> Option<&'a [bool]> {
        self.constant
    }

    fn windows(&self, m: usize) -> Vec<Window> {
        if m == 0 || m > self.samples.len() {
            return Vec::new();
        }
        let count = self.samples.len() - m + 1;
        if let Some(flags) = self.constant {
            assert_eq!(flags.len(), count, "one constant flag per window");
        }
        (0..count)
            .map(|start| {
                let window = window_stats(self.samples, start, m);
                let Some(flags) = self.constant else {
                    return window;
                };
                match (window, flags[start]) {
                    (Window::NonFinite, _) => Window::NonFinite,
                    (_, true) => Window::Constant,
                    (Window::Constant, false) => Window::Valid {
                        sum: self.samples[start..start + m]
                            .iter()
                            .map(|tick| i128::from(tick.expect("a constant window has ticks")))
                            .sum(),
                        variance_numerator: 0,
                    },
                    (valid, false) => valid,
                }
            })
            .collect()
    }
}

impl<'a> From<&'a [Option<i64>]> for Ticks<'a> {
    fn from(samples: &'a [Option<i64>]) -> Self {
        Self {
            samples,
            constant: None,
        }
    }
}

impl<'a> From<&'a Vec<Option<i64>>> for Ticks<'a> {
    fn from(samples: &'a Vec<Option<i64>>) -> Self {
        samples.as_slice().into()
    }
}

impl<'a, const N: usize> From<&'a [Option<i64>; N]> for Ticks<'a> {
    fn from(samples: &'a [Option<i64>; N]) -> Self {
        samples.as_slice().into()
    }
}

/// `Σ_k a[i+k]·b[j+k]`, exact in 256 bits; both windows are already known to hold ticks (checked
/// by the caller through [`comparable`]). `a` and `b` may be the same slice (a self-join
/// pair) or different ones (an AB-join pair): the sum is computed the same direct way
/// either way, never derived from one side by transforming the other.
fn cross_term_pair(a: &[Option<i64>], i: usize, b: &[Option<i64>], j: usize, m: usize) -> I256 {
    let mut total = I256::ZERO;
    for k in 0..m {
        let x = a[i + k].expect("non-finite windows are filtered before pairing");
        let y = b[j + k].expect("non-finite windows are filtered before pairing");
        let product = I256::from(x)
            .checked_mul(I256::from(y))
            .expect(WIDE_OVERFLOW_MESSAGE);
        total = total.checked_add(product).expect(WIDE_OVERFLOW_MESSAGE);
    }
    total
}

/// `Σ_k samples[i+k]·samples[j+k]`, the self-join case of [`cross_term_pair`] with both
/// windows drawn from the same series.
fn cross_term(samples: &[Option<i64>], i: usize, j: usize, m: usize) -> I256 {
    cross_term_pair(samples, i, samples, j, m)
}

/// `N = m·Σxᵢxⱼ − SᵢSⱼ`, exact in 256 bits.
fn centered_cross(m: usize, qt: I256, sum_i: i128, sum_j: i128) -> I256 {
    let scaled = I256::from(m as u64)
        .checked_mul(qt)
        .expect(WIDE_OVERFLOW_MESSAGE);
    let sum_product = I256::from(sum_i)
        .checked_mul(I256::from(sum_j))
        .expect(WIDE_OVERFLOW_MESSAGE);
    scaled
        .checked_sub(sum_product)
        .expect(WIDE_OVERFLOW_MESSAGE)
}

fn exact_correlation(
    row: Comparable,
    neighbor: Comparable,
    cross: impl Into<I256>,
) -> ExactCorrelation {
    let cross: I256 = cross.into();
    match (row, neighbor) {
        (Comparable::Constant, Comparable::Constant) => ExactCorrelation::BothConstant,
        (Comparable::Constant, Comparable::Valid { .. })
        | (Comparable::Valid { .. }, Comparable::Constant) => ExactCorrelation::OneConstant,
        (
            Comparable::Valid {
                variance_numerator: row_variance,
                ..
            },
            Comparable::Valid {
                variance_numerator: neighbor_variance,
                ..
            },
        ) => {
            let squared = cross.checked_mul(cross).expect(WIDE_OVERFLOW_MESSAGE);
            // The engines' comparator carries `N²` as u128; past that there is nothing to
            // compare against, so say so rather than cap it.
            let cross_squared = u128::try_from(squared).expect(OVERFLOW_MESSAGE);
            let row_variance = row_variance as u64;
            let neighbor_variance = neighbor_variance as u64;
            ExactCorrelation::Valid {
                sign: i8::try_from(cross.signum()).expect("a signum is -1, 0 or 1"),
                cross_squared,
                row_variance,
                neighbor_variance,
                denominator: variance_product(row_variance, neighbor_variance),
            }
        }
    }
}

fn candidate(m: usize, row: Comparable, neighbor: Comparable, qt: I256, index: u64) -> Candidate {
    let (cross, rho_hat) = match (row, neighbor) {
        (Comparable::Constant, Comparable::Constant) => (I256::ZERO, 1.0),
        (Comparable::Constant, Comparable::Valid { .. })
        | (Comparable::Valid { .. }, Comparable::Constant) => (I256::ZERO, 0.5),
        (
            Comparable::Valid {
                sum: row_sum,
                variance_numerator: row_variance,
            },
            Comparable::Valid {
                sum: neighbor_sum,
                variance_numerator: neighbor_variance,
            },
        ) => {
            let cross = centered_cross(m, qt, row_sum, neighbor_sum);
            // A window of variance zero (equal ticks the caller did not flag constant) has a
            // zero cross term with everything, so its correlation is zero.
            let rho_hat = if row_variance == 0 || neighbor_variance == 0 {
                0.0
            } else {
                let row_inv = 1.0 / libm::sqrt(row_variance as f64);
                let neighbor_inv = 1.0 / libm::sqrt(neighbor_variance as f64);
                // |N| < 2^63 for two valid windows (Cauchy-Schwarz), so this is the same
                // conversion the engines make.
                let cross_f64 =
                    i64::try_from(cross).expect("|N| < 2^63 for two valid windows") as f64;
                (cross_f64 * row_inv * neighbor_inv).clamp(-1.0, 1.0)
            };
            (cross, rho_hat)
        }
    };
    Candidate {
        index,
        rho_hat,
        correlation: exact_correlation(row, neighbor, cross),
    }
}

/// `samples[k]` is `Some(tick)` or `None` (no tick); returns one entry per window start
/// `i` in `0..=n−m`, considering only `j ≤ i − e − 1` with `e = ceil(m/4)`.
pub fn naive_left_profile<'a>(samples: impl Into<Ticks<'a>>, m: usize) -> Vec<(u64, Entry)> {
    let ticks: Ticks<'a> = samples.into();
    let samples = ticks.samples;
    let n = samples.len();
    if m == 0 || m > n {
        return Vec::new();
    }

    let num_windows = n - m + 1;
    let windows: Vec<Window> = ticks.windows(m);
    let e = crate::exclusion::zone(m);

    let mut results = Vec::with_capacity(num_windows);
    for i in 0..num_windows {
        let entry = match comparable(windows[i]) {
            None => Entry::NONE,
            Some(row) => {
                let mut best: Option<Candidate> = None;
                if i > e {
                    for (j, &neighbor_window) in windows[..i - e].iter().enumerate() {
                        if let Some(neighbor) = comparable(neighbor_window) {
                            let qt = cross_term(samples, i, j, m);
                            let offered = candidate(m, row, neighbor, qt, j as u64);
                            if best.is_none_or(|current| nearer(offered, current)) {
                                best = Some(offered);
                            }
                        }
                    }
                }
                best.map_or(Entry::NONE, |winner| winner_entry(m, winner))
            }
        };
        results.push((i as u64, entry));
    }
    results
}

/// `row`/`neighbor`'s naive classification, carried into the engine's bounded window
/// representation so it can feed [`Nearest`]: exact, since [`comparable`] already enforced the
/// same `1..2^63` variance range `stats::classify` uses.
fn stats_window(window: Comparable) -> StatsWindow {
    match window {
        Comparable::Constant => StatsWindow {
            sum: 0,
            kind: WindowKind::Constant,
        },
        Comparable::Valid {
            sum,
            variance_numerator,
        } => {
            let variance_numerator = variance_numerator as u64;
            StatsWindow {
                sum: sum as i64,
                kind: WindowKind::Valid {
                    variance_numerator,
                    inv_sqrt: if variance_numerator == 0 {
                        1.0
                    } else {
                        1.0 / libm::sqrt(variance_numerator as f64)
                    },
                },
            }
        }
    }
}

/// The pair's exact centered cross term, narrowed to `i64`: Cauchy-Schwarz bounds every
/// valid-valid pair's `|N|` below `sqrt(Vi·Vj) < 2^63` (each variance numerator is itself below
/// `2^63`), so the cast never truncates. A pair touching a constant window reports `0`: unused by
/// [`Nearest::offer`], which decides those pairs from the windows' kind alone.
fn exact_cross(m: usize, row: Comparable, neighbor: Comparable, qt: I256) -> i64 {
    match (row, neighbor) {
        (
            Comparable::Valid { sum: row_sum, .. },
            Comparable::Valid {
                sum: neighbor_sum, ..
            },
        ) => i64::try_from(centered_cross(m, qt, row_sum, neighbor_sum))
            .expect("Cauchy-Schwarz keeps |N| below 2^63 for two valid windows"),
        _ => 0,
    }
}

/// `samples[k]` is `Some(tick)` or `None` (no tick); returns, per window start `i` in `0..=n−m`,
/// up to `k` entries nearest first, considering only `j ≤ i − e − 1` with `e = ceil(m/4)`.
pub fn naive_left_topk<'a>(
    samples: impl Into<Ticks<'a>>,
    m: usize,
    k: NonZeroUsize,
) -> Vec<(u64, Vec<Entry>)> {
    let ticks: Ticks<'a> = samples.into();
    let samples = ticks.samples;
    let n = samples.len();
    if m == 0 || m > n {
        return Vec::new();
    }

    let num_windows = n - m + 1;
    let windows: Vec<Window> = ticks.windows(m);
    let e = crate::exclusion::zone(m);

    let mut results = Vec::with_capacity(num_windows);
    for i in 0..num_windows {
        let entries = match comparable(windows[i]) {
            None => Vec::new(),
            Some(row) => {
                let row_window = stats_window(row);
                let mut nearest = Nearest::with_k(&row_window, m, k, usize::MAX);
                if i > e {
                    for (j, &neighbor_window) in windows[..i - e].iter().enumerate() {
                        if let Some(neighbor) = comparable(neighbor_window) {
                            let qt = cross_term(samples, i, j, m);
                            let cross = exact_cross(m, row, neighbor, qt);
                            let index = SampleIndex::new(j as u64)
                                .expect("window starts stay below SampleIndex::NONE's sentinel");
                            nearest.offer(cross, &stats_window(neighbor), index);
                        }
                    }
                }
                nearest.finish_k()
            }
        };
        results.push((i as u64, entries));
    }
    results
}

pub use crate::entry::{TwoSided, TwoSidedTopk};

mod diagonals;
pub use diagonals::{
    naive_ab_join_nonnorm_on, naive_ab_join_topk_on, naive_two_sided_nonnorm_on,
    naive_two_sided_topk_on,
};
mod multi;
pub use multi::{naive_multi_profile, naive_multi_profile_nonnorm};
mod mdl;
pub use mdl::{naive_log2, naive_maamp_mdl, naive_mdl};
mod subspace;
pub use subspace::{
    naive_choose, naive_edge_below, naive_linear_bins, naive_linear_range, naive_maamp_subspace,
    naive_subspace, naive_z_bins,
};

/// `left`'s own winning candidate, recomputed directly from the samples rather than from its
/// already-rounded `rho`/`squared_distance`: the same per-pair computation this module's main
/// loop uses to build every candidate, so comparing it against a fresh right-side candidate
/// with [`nearer`] decides between them by the exact order and tie rule that chose each
/// one in the first place. `neighbor_index`/`neighbor_window` must already be known comparable
/// (the caller passes only an index a real candidate was already built from).
fn exact_candidate_for(
    samples: &[Option<i64>],
    m: usize,
    row_index: usize,
    row: Comparable,
    neighbor_index: usize,
    neighbor_window: Window,
) -> Candidate {
    let neighbor = comparable(neighbor_window)
        .expect("neighbor_index came from an already-resolved candidate: its window is comparable");
    let qt = cross_term(samples, row_index, neighbor_index, m);
    candidate(m, row, neighbor, qt, neighbor_index as u64)
}

/// `samples[k]` is `Some(tick)` or `None` (no tick); returns, per window start `i` in
/// `0..=n−m`: the exact left neighbor (`j ≤ i − e − 1`, exactly [`naive_left_profile`]'s own
/// entry), the exact right neighbor (`j > i + e`), and the nearer of the two under the exact
/// comparator (ties to the smaller index), with `e = ceil(m/4)`. Looking right trades on
/// future information: the name says so.
///
/// `right` is a direct `j > i + e` scan, the mirror image of `naive_left_profile`'s own `j ≤ i
/// − e − 1` loop, rather than a reversed-series lookup: reversing the series preserves every
/// pair's `rho`/`squared_distance` (an exact integer function of the tick values, independent
/// of summation order) but *inverts* which of two exactly-tied candidates has the smaller
/// index, since reversing a sequence negates every index around its midpoint — so a tie that
/// must resolve to the smaller original index would instead resolve to the larger one. Scanning
/// `j > i + e` directly, with every candidate's true original index, has no such inversion.
pub fn naive_two_sided<'a>(samples: impl Into<Ticks<'a>>, m: usize) -> Vec<(u64, TwoSided)> {
    let ticks: Ticks<'a> = samples.into();
    let samples = ticks.samples;
    let left_profile = naive_left_profile(ticks, m);
    if left_profile.is_empty() {
        return Vec::new();
    }

    let num_windows = left_profile.len();
    let windows: Vec<Window> = ticks.windows(m);
    let e = crate::exclusion::zone(m);

    let mut results = Vec::with_capacity(num_windows);
    for (row, left) in left_profile {
        let i = row as usize;
        let two_sided = match comparable(windows[i]) {
            None => TwoSided {
                left,
                right: Entry::NONE,
                nearest: Entry::NONE,
            },
            Some(row_window) => {
                let mut right_best: Option<Candidate> = None;
                for (j, &neighbor_window) in windows.iter().enumerate().skip(i + e + 1) {
                    if let Some(neighbor) = comparable(neighbor_window) {
                        let qt = cross_term(samples, i, j, m);
                        let offered = candidate(m, row_window, neighbor, qt, j as u64);
                        if right_best.is_none_or(|current| nearer(offered, current)) {
                            right_best = Some(offered);
                        }
                    }
                }
                let right = right_best.map_or(Entry::NONE, |winner| winner_entry(m, winner));

                let left_candidate = left.index().get().map(|j| {
                    exact_candidate_for(samples, m, i, row_window, j as usize, windows[j as usize])
                });
                let nearest = match (left_candidate, right_best) {
                    (None, None) => Entry::NONE,
                    (Some(_), None) => left,
                    (None, Some(_)) => right,
                    (Some(l), Some(r)) => {
                        if nearer(l, r) {
                            left
                        } else {
                            right
                        }
                    }
                };
                TwoSided {
                    left,
                    right,
                    nearest,
                }
            }
        };
        results.push((row, two_sided));
    }
    results
}

/// `a[k]`/`b[k]` is `Some(tick)` or `None` (no tick); returns one entry per window start `i`
/// in `0..=n_a−m`, searching every window of `b` directly, with no exclusion zone: `i`'s own
/// position in `a` carries no special meaning for `b`, so even `b = a` may let a row match
/// itself. Each window is centered and normalized by its own mean and variance, so `a` and
/// `b` may come from different instruments or price levels. Ties go to the smaller `b` index
/// and a non-finite `b` window is never chosen — the same tie rule and non-finite handling
/// [`naive_left_profile`] uses — and a non-finite `a` window gives [`Entry::NONE`]. `b`
/// shorter than `m` leaves every row at [`Entry::NONE`] too.
pub fn naive_ab_join<'a>(
    a: impl Into<Ticks<'a>>,
    b: impl Into<Ticks<'a>>,
    m: usize,
) -> Vec<(u64, Entry)> {
    let (a_ticks, b_ticks): (Ticks<'a>, Ticks<'a>) = (a.into(), b.into());
    let (a, b) = (a_ticks.samples, b_ticks.samples);
    let n_a = a.len();
    if m == 0 || m > n_a {
        return Vec::new();
    }

    let num_windows_a = n_a - m + 1;
    let a_windows: Vec<Window> = a_ticks.windows(m);
    let b_windows: Vec<Window> = b_ticks.windows(m);

    let mut results = Vec::with_capacity(num_windows_a);
    for (i, &a_window) in a_windows.iter().enumerate() {
        let entry = match comparable(a_window) {
            None => Entry::NONE,
            Some(row) => {
                let mut best: Option<Candidate> = None;
                for (j, &neighbor_window) in b_windows.iter().enumerate() {
                    if let Some(neighbor) = comparable(neighbor_window) {
                        let qt = cross_term_pair(a, i, b, j, m);
                        let offered = candidate(m, row, neighbor, qt, j as u64);
                        if best.is_none_or(|current| nearer(offered, current)) {
                            best = Some(offered);
                        }
                    }
                }
                best.map_or(Entry::NONE, |winner| winner_entry(m, winner))
            }
        };
        results.push((i as u64, entry));
    }
    results
}

/// The `k` nearest of `candidates` under the exact comparator, nearest first, ties to the smaller
/// index: a full sort, not a running reduction.
fn nearest_k(m: usize, mut candidates: Vec<Candidate>, k: NonZeroUsize) -> Vec<Entry> {
    candidates.sort_by(|&a, &b| {
        if nearer(a, b) {
            core::cmp::Ordering::Less
        } else if nearer(b, a) {
            core::cmp::Ordering::Greater
        } else {
            core::cmp::Ordering::Equal
        }
    });
    candidates
        .into_iter()
        .take(k.get())
        .map(|winner| winner_entry(m, winner))
        .collect()
}

/// [`naive_two_sided`] with the `k` nearest neighbors over both sides (`j ≤ i − e − 1` or
/// `j > i + e`): `left` and `right` are [`naive_two_sided`]'s own, `nearest` holds up to `k`
/// entries, nearest first, ties to the smaller index; none for a window with no neighbor.
pub fn naive_two_sided_topk<'a>(
    samples: impl Into<Ticks<'a>>,
    m: usize,
    k: NonZeroUsize,
) -> Vec<(u64, TwoSidedTopk)> {
    let ticks: Ticks<'a> = samples.into();
    let samples = ticks.samples;
    let sides = naive_two_sided(ticks, m);
    let windows: Vec<Window> = ticks.windows(m);
    let e = crate::exclusion::zone(m);
    sides
        .into_iter()
        .map(|(row, sides)| {
            let i = row as usize;
            let nearest = comparable(windows[i]).map_or_else(Vec::new, |row_window| {
                let candidates = windows
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j + e < i || j > i + e)
                    .filter_map(|(j, &window)| {
                        let neighbor = comparable(window)?;
                        let qt = cross_term(samples, i, j, m);
                        Some(candidate(m, row_window, neighbor, qt, j as u64))
                    })
                    .collect();
                nearest_k(m, candidates, k)
            });
            (
                row,
                TwoSidedTopk {
                    left: sides.left,
                    right: sides.right,
                    nearest,
                },
            )
        })
        .collect()
}

/// [`naive_ab_join`] with the `k` nearest `b` windows per `a` window, nearest first, ties to the
/// smaller `b` index; none for a non-finite `a` window or when `b` has no finite window.
pub fn naive_ab_join_topk<'a>(
    a: impl Into<Ticks<'a>>,
    b: impl Into<Ticks<'a>>,
    m: usize,
    k: NonZeroUsize,
) -> Vec<(u64, Vec<Entry>)> {
    let (a_ticks, b_ticks): (Ticks<'a>, Ticks<'a>) = (a.into(), b.into());
    let (a, b) = (a_ticks.samples, b_ticks.samples);
    if m == 0 || m > a.len() {
        return Vec::new();
    }
    let (a_windows, b_windows) = (a_ticks.windows(m), b_ticks.windows(m));
    a_windows
        .iter()
        .enumerate()
        .map(|(i, &a_window)| {
            let entries = comparable(a_window).map_or_else(Vec::new, |row| {
                let candidates = b_windows
                    .iter()
                    .enumerate()
                    .filter_map(|(j, &window)| {
                        let neighbor = comparable(window)?;
                        let qt = cross_term_pair(a, i, b, j, m);
                        Some(candidate(m, row, neighbor, qt, j as u64))
                    })
                    .collect();
                nearest_k(m, candidates, k)
            });
            (i as u64, entries)
        })
        .collect()
}

/// The exact squared Euclidean distance of a non-normalized pair, carried as `u64` when
/// `m·R² < 2⁶⁴` holds for `R`, the range of every present tick in the input (so no pair's
/// distance can exceed `u64`), and as a 256-bit integer otherwise. The oracle always computes it in
/// 256 bits; only the carrier differs, and one call uses one carrier for every entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NonNormDistance {
    Narrow(u64),
    Wide(I256),
}

/// A non-normalized neighbor: its exact squared distance and its index. No neighbor is
/// [`SampleIndex::NONE`], with the carrier's largest distance.
#[derive(Clone, Copy, Debug)]
pub struct NonNormEntry {
    distance: NonNormDistance,
    index: SampleIndex,
}

impl NonNormEntry {
    fn none(narrow: bool) -> Self {
        let distance = if narrow {
            NonNormDistance::Narrow(u64::MAX)
        } else {
            NonNormDistance::Wide(I256::MAX)
        };
        Self {
            distance,
            index: SampleIndex::NONE,
        }
    }

    fn winner(index: u64, distance: I256, narrow: bool) -> Self {
        let distance = if narrow {
            NonNormDistance::Narrow(
                u64::try_from(distance).expect("the range bound keeps every distance below 2^64"),
            )
        } else {
            NonNormDistance::Wide(distance)
        };
        Self {
            distance,
            index: SampleIndex::new(index)
                .expect("window starts stay below SampleIndex::NONE's sentinel"),
        }
    }

    pub const fn distance(self) -> NonNormDistance {
        self.distance
    }

    pub const fn index(self) -> SampleIndex {
        self.index
    }
}

/// `true` once `m·R² < 2⁶⁴`, with `R` the max − min over every present tick: each pair's
/// squared distance is at most `m·R²`, so it then fits `u64`. No ticks means nothing is ever
/// compared.
fn fits_narrow_bound(samples: &[Option<i64>], m: usize) -> bool {
    let mut min = i64::MAX;
    let mut max = i64::MIN;
    for &tick in samples.iter().flatten() {
        min = min.min(tick);
        max = max.max(tick);
    }
    if max < min {
        return true;
    }
    let range = (max as i128 - min as i128) as u128;
    range
        .checked_mul(range)
        .and_then(|squared| squared.checked_mul(m as u128))
        .is_some_and(|scaled| scaled < 1u128 << 64)
}

/// `Σ_k (a[i+k] − b[j+k])²` on raw ticks, exact in 256 bits and computed per pair, never from the
/// bound above; both windows are already known to hold ticks.
fn squared_distance_pair(
    a: &[Option<i64>],
    i: usize,
    b: &[Option<i64>],
    j: usize,
    m: usize,
) -> I256 {
    let mut total = I256::ZERO;
    for k in 0..m {
        let x = a[i + k].expect("windows with a gap are filtered before pairing");
        let y = b[j + k].expect("windows with a gap are filtered before pairing");
        let diff = I256::from(x) - I256::from(y);
        let squared = diff.checked_mul(diff).expect(WIDE_OVERFLOW_MESSAGE);
        total = total.checked_add(squared).expect(WIDE_OVERFLOW_MESSAGE);
    }
    total
}

/// `samples[k]` is `Some(tick)` or `None` (no tick); returns one entry per window start `i`
/// in `0..=n−m`, considering only `j ≤ i − e − 1` with `e = ceil(m/4)`, ties to the smaller
/// index (the same zone and tie rule as [`naive_left_profile`]). The distance is the squared
/// Euclidean distance between the raw tick windows, never mean-centered. A window with any
/// gap is non-finite: its row has no neighbor and it is never chosen.
pub fn naive_left_profile_nonnorm(samples: &[Option<i64>], m: usize) -> Vec<(u64, NonNormEntry)> {
    let n = samples.len();
    if m == 0 || m > n {
        return Vec::new();
    }

    let num_windows = n - m + 1;
    let has_gap = nonnorm_gap_windows(samples, m);
    let e = crate::exclusion::zone(m);
    let narrow = fits_narrow_bound(samples, m);

    let mut results = Vec::with_capacity(num_windows);
    for (i, &row_has_gap) in has_gap.iter().enumerate() {
        let mut best: Option<(u64, I256)> = None;
        if !row_has_gap && i > e {
            for (j, &neighbor_has_gap) in has_gap[..i - e].iter().enumerate() {
                if !neighbor_has_gap {
                    let distance = squared_distance_pair(samples, i, samples, j, m);
                    if best.is_none_or(|(_, current)| distance < current) {
                        best = Some((j as u64, distance));
                    }
                }
            }
        }
        let entry = best.map_or(NonNormEntry::none(narrow), |(index, distance)| {
            NonNormEntry::winner(index, distance, narrow)
        });
        results.push((i as u64, entry));
    }
    results
}

/// A window's left, right and nearer non-normalized neighbor.
#[derive(Clone, Copy, Debug)]
pub struct NonNormSides {
    pub left: NonNormEntry,
    pub right: NonNormEntry,
    pub nearest: NonNormEntry,
}

/// Whether each window of `m` samples of `samples` has a gap.
fn nonnorm_gap_windows(samples: &[Option<i64>], m: usize) -> Vec<bool> {
    let mut gaps = samples[..m].iter().filter(|s| s.is_none()).count();
    let mut windows = Vec::with_capacity(samples.len() - m + 1);
    windows.push(gaps > 0);
    for k in m..samples.len() {
        gaps += usize::from(samples[k].is_none());
        gaps -= usize::from(samples[k - m].is_none());
        windows.push(gaps > 0);
    }
    windows
}

/// The nearest `b` window among `columns` (all of them complete) to `a`'s window `i`, ties to the
/// smaller index.
fn nonnorm_nearest(
    (a, i): (&[Option<i64>], usize),
    b: &[Option<i64>],
    columns: impl Iterator<Item = usize>,
    m: usize,
) -> Option<(u64, I256)> {
    let mut best: Option<(u64, I256)> = None;
    for j in columns {
        let distance = squared_distance_pair(a, i, b, j, m);
        if best.is_none_or(|(_, current)| distance < current) {
            best = Some((j as u64, distance));
        }
    }
    best
}

fn nonnorm_entry(best: Option<(u64, I256)>, narrow: bool) -> NonNormEntry {
    best.map_or(NonNormEntry::none(narrow), |(index, distance)| {
        NonNormEntry::winner(index, distance, narrow)
    })
}

/// [`naive_left_profile_nonnorm`]'s zone and tie rule, with each window's nearest later window
/// too (`j ≥ i + e + 1`, ties to the smaller index) and the nearer of the two, the left one on
/// equal distances. A window with a gap has no neighbor on either side and is never one.
pub fn naive_two_sided_nonnorm(samples: &[Option<i64>], m: usize) -> Vec<(u64, NonNormSides)> {
    if m == 0 || m > samples.len() {
        return Vec::new();
    }
    let has_gap = nonnorm_gap_windows(samples, m);
    let windows = has_gap.len();
    let e = crate::exclusion::zone(m);
    let narrow = fits_narrow_bound(samples, m);
    let complete = |range: core::ops::Range<usize>| {
        range
            .filter(|&j| !has_gap[j])
            .collect::<Vec<_>>()
            .into_iter()
    };
    (0..windows)
        .map(|i| {
            let (left, right) = if has_gap[i] {
                (None, None)
            } else {
                let left_columns = if i > e { 0..i - e } else { 0..0 };
                (
                    nonnorm_nearest((samples, i), samples, complete(left_columns), m),
                    nonnorm_nearest((samples, i), samples, complete(i + e + 1..windows), m),
                )
            };
            let nearest = match (left, right) {
                (Some(l), Some(r)) => Some(if r.1 < l.1 { r } else { l }),
                (l, r) => l.or(r),
            };
            let sides = NonNormSides {
                left: nonnorm_entry(left, narrow),
                right: nonnorm_entry(right, narrow),
                nearest: nonnorm_entry(nearest, narrow),
            };
            (i as u64, sides)
        })
        .collect()
}

/// Every window of `a` against every window of `b`, no exclusion zone: per `a` window, its
/// nearest `b` window by squared Euclidean distance on raw ticks, ties to the smaller index; none
/// for a window with a gap, or when `b` has no complete window. The distance carrier follows the
/// range of every present tick of both inputs.
pub fn naive_ab_join_nonnorm(
    a: &[Option<i64>],
    b: &[Option<i64>],
    m: usize,
) -> Vec<(u64, NonNormEntry)> {
    if m == 0 || m > a.len() {
        return Vec::new();
    }
    let a_gaps = nonnorm_gap_windows(a, m);
    let b_gaps = if m <= b.len() {
        nonnorm_gap_windows(b, m)
    } else {
        Vec::new()
    };
    let both: Vec<Option<i64>> = a.iter().chain(b).copied().collect();
    let narrow = fits_narrow_bound(&both, m);
    let columns: Vec<usize> = (0..b_gaps.len()).filter(|&j| !b_gaps[j]).collect();
    a_gaps
        .iter()
        .enumerate()
        .map(|(i, &gap)| {
            let best = if gap {
                None
            } else {
                nonnorm_nearest((a, i), b, columns.iter().copied(), m)
            };
            (i as u64, nonnorm_entry(best, narrow))
        })
        .collect()
}

/// The non-normalized entry of `row` when only the last `w` samples up to the end of its window
/// are searched: [`naive_left_profile_nonnorm`] on those samples, the neighbor index in
/// whole-series terms. The distance carrier follows the range of the samples searched.
///
/// # Panics
/// If `w < m`, or `row` is not a window start of `samples`.
pub fn windowed_nonnorm_entry(
    samples: &[Option<i64>],
    m: usize,
    w: usize,
    row: usize,
) -> NonNormEntry {
    assert!(w >= m, "the search space must hold at least one window");
    let start = (row + m).saturating_sub(w);
    let (_, local) = *naive_left_profile_nonnorm(&samples[start..row + m], m)
        .last()
        .expect("the slice holds row's own window");
    match local.index.get() {
        None => local,
        Some(local_j) => NonNormEntry {
            index: SampleIndex::new(local_j + start as u64).expect("a global index is below NONE"),
            ..local
        },
    }
}

/// The exact entry of `row` when only the last `w` samples up to the end of its window are
/// searched: the naive search of that row's window against every eligible earlier window that
/// starts inside the last `w` samples, with the neighbor index in whole-series terms.
///
/// # Panics
/// If `w < m`: a search space shorter than one window is not a bounded stream.
pub fn windowed_entry(samples: &[Option<i64>], m: usize, w: usize, row: usize) -> Entry {
    assert!(w >= m, "the search space must hold at least one window");
    let exclusion = crate::exclusion::zone(m);
    if row <= exclusion {
        return Entry::NONE;
    }
    let start = (row + m).saturating_sub(w);
    let end = row - exclusion - 1 + m;
    if end < start + m {
        return Entry::NONE;
    }
    let candidates = &samples[start..end];
    let (_, local) = naive_ab_join(&samples[row..row + m], candidates, m)[0];
    match local.index().get() {
        None => Entry::NONE,
        Some(local_j) => Entry::new(
            local.rho(),
            local.squared_distance(),
            SampleIndex::new(local_j + start as u64).expect("a global index is below NONE"),
        )
        .expect("rho and squared distance pass through unchanged"),
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::{
        Candidate, Comparable, EPSILON, ExactCorrelation, I256, NonNormDistance, NonNormEntry,
        Sharing, TwoSided, cmp_exact, cmp_wide_products, exact_correlation, naive_ab_join,
        naive_left_profile, naive_left_profile_nonnorm, naive_left_topk, naive_two_sided, nearer,
    };
    use crate::index::SampleIndex;
    use alloc::vec::Vec;
    use core::cmp::Ordering;
    use core::num::NonZeroUsize;
    use proptest::prelude::*;
    use proptest::strategy::ValueTree;
    use proptest::test_runner::TestRunner;

    fn ticks(values: &[i64]) -> Vec<Option<i64>> {
        values.iter().copied().map(Some).collect()
    }

    fn entry_of(profile: &[(u64, crate::entry::Entry)], row: u64) -> crate::entry::Entry {
        profile
            .iter()
            .find(|&&(r, _)| r == row)
            .expect("row exists")
            .1
    }

    #[test]
    fn zero_or_oversized_window_has_no_rows() {
        let samples = ticks(&[1, 2, 3, 4]);
        assert!(naive_left_profile(&samples, 0).is_empty());
        assert!(naive_left_profile(&samples, 5).is_empty());
    }

    #[test]
    #[should_panic(expected = "exceeds i128")]
    fn cross_squared_rejects_overflow_instead_of_saturating() {
        // Past the domain `window_stats` enforces (`|N| < 2^63`), squaring this `cross`
        // overflows u128; a silently capped product would be a wrong comparison, not merely
        // a slow one.
        let window = Comparable::Valid {
            sum: 0,
            variance_numerator: 1,
        };
        exact_correlation(window, window, 1i128 << 65);
    }

    #[test]
    fn exact_repeat_recomputes_to_zero_distance() {
        // The ranking ρ for this self-match rounds to just under 1 (two independent
        // reciprocal square roots, not a single division), and the entry keeps that ranking
        // ρ; only the distance is recomputed exactly, from D = Vi·Vj − N² = 0.
        let samples = ticks(&[8, 4, -6, 100, 200, 8, 4, -6]);
        let profile = naive_left_profile(&samples, 3);
        let entry = entry_of(&profile, 5);
        assert_eq!(entry.index(), SampleIndex::new(0).unwrap());
        assert_eq!(entry.rho(), 0.9999999999999998);
        assert_eq!(entry.squared_distance(), 0.0);
    }

    #[test]
    fn both_constant_windows_have_zero_distance() {
        let samples = ticks(&[5, 5, 5, 1, 2, 9, 9, 9]);
        let profile = naive_left_profile(&samples, 3);
        let entry = entry_of(&profile, 5);
        assert_eq!(entry.index(), SampleIndex::new(0).unwrap());
        assert_eq!(entry.rho(), 1.0);
        assert_eq!(entry.squared_distance(), 0.0);
    }

    #[test]
    fn one_constant_window_has_distance_m() {
        let samples = ticks(&[1, 2, 7, 7, 7]);
        let profile = naive_left_profile(&samples, 3);
        let entry = entry_of(&profile, 2);
        assert_eq!(entry.index(), SampleIndex::new(0).unwrap());
        assert_eq!(entry.rho(), 0.5);
        assert_eq!(entry.squared_distance(), 3.0);
    }

    #[test]
    fn none_inside_window_is_nonfinite_and_never_a_neighbor() {
        let mut samples = ticks(&[1, 2, 3, 0, 5, 6, 7, 1, 2, 3]);
        samples[3] = None;
        let profile = naive_left_profile(&samples, 3);

        for row in [1, 2, 3] {
            assert_eq!(entry_of(&profile, row).index(), SampleIndex::NONE);
            assert_eq!(entry_of(&profile, row).squared_distance(), f64::INFINITY);
        }

        let repeat = entry_of(&profile, 7);
        assert_eq!(repeat.index(), SampleIndex::new(0).unwrap());
        assert_eq!(repeat.rho(), 1.0);
        assert_eq!(repeat.squared_distance(), 0.0);
    }

    #[test]
    fn exclusion_boundary_excludes_e_and_allows_e_plus_1() {
        // e = ceil(4/4) = 1: row 2's window [7,7,7,7] is constant and identical to row 1's
        // window [7,7,7,7], which would win outright (ρ = 1) if the boundary let it through.
        // The only legal candidate is row 0's non-constant window, so the correct answer is
        // the mixed-constancy pair (ρ = 0.5), not the excluded perfect match.
        let samples = ticks(&[1, 7, 7, 7, 7, 7]);
        let profile = naive_left_profile(&samples, 4);
        let entry = entry_of(&profile, 2);
        assert_eq!(entry.index(), SampleIndex::new(0).unwrap());
        assert_eq!(entry.rho(), 0.5);
        assert_eq!(entry.squared_distance(), 4.0);
    }

    #[test]
    fn exclusion_half_width_rounds_up() {
        // m = 5: ceil(5/4) = 2, floor(5/4) = 1. Row 2 has no legal candidate under the
        // correct half-width (2 ≤ e), but would gain one under the floored half-width.
        let samples = ticks(&[1, 2, 3, 4, 5, 6, 7]);
        let profile = naive_left_profile(&samples, 5);
        let entry = entry_of(&profile, 2);
        assert_eq!(entry.index(), SampleIndex::NONE);
        assert_eq!(entry.squared_distance(), f64::INFINITY);
    }

    #[test]
    fn winning_near_match_recomputes_a_nonzero_exact_distance() {
        // Row 4's window differs from row 0's by one tick in one position; the raw ranking ρ
        // rounds to just over 1.0 (two independent reciprocal square roots, not a single
        // division), so clamping keeps the entry's ρ at exactly 1 rather than letting it
        // leave [-1, 1]. This is not a true exact match (D = Vi·Vj − N² > 0 strictly), so,
        // unlike a real exact match, the winner's distance is recomputed to a small positive
        // value rather than 0.
        let samples = ticks(&[
            235_732_534,
            -190_400_685,
            248_938_659,
            -501_918_519,
            235_732_534,
            -190_400_685,
            248_938_660,
            -501_918_519,
        ]);
        let profile = naive_left_profile(&samples, 4);
        let entry = entry_of(&profile, 4);
        assert_eq!(entry.index(), SampleIndex::new(0).unwrap());
        assert_eq!(entry.rho(), 1.0);
        assert_eq!(entry.squared_distance(), 5.274993140419151e-18);
    }

    #[test]
    fn negative_correlation_keeps_the_ranking_distance() {
        // Row 2's window [5,1] is the exact mirror of row 0's window [1,2]: m = 2 makes any
        // two non-constant windows perfectly correlated one way or the other, and N² = Vi·Vj
        // holds here with N negative, ρ = −1. The recomputation only applies for ρ > 0, so
        // this entry keeps the ordinary ranking distance, not a recomputed D = 0.
        let samples = ticks(&[1, 2, 5, 1]);
        let profile = naive_left_profile(&samples, 2);
        let entry = entry_of(&profile, 2);
        assert_eq!(entry.index(), SampleIndex::new(0).unwrap());
        assert_eq!(entry.rho(), -1.0);
        assert_eq!(entry.squared_distance(), 8.0);
    }

    /// A generic row with several ordinary (non-constant, non-exact-match) candidates, none
    /// at an exclusion boundary. Shared by the pinned-bit tests below.
    const GENERIC_SERIES: [i64; 8] = [1, 8, 2, 50, 60, 10, 3, 7];

    #[test]
    fn generic_valid_pair_pins_rho_and_distance_bits() {
        let samples = ticks(&GENERIC_SERIES);
        let profile = naive_left_profile(&samples, 3);

        let row2 = entry_of(&profile, 2);
        assert_eq!(row2.index(), SampleIndex::new(0).unwrap());
        assert_eq!(row2.rho().to_bits(), 0x3fde59b497f4fdc1);
        assert_eq!(row2.squared_distance().to_bits(), 0x40093cb88e0841af);

        let row3 = entry_of(&profile, 3);
        assert_eq!(row3.index(), SampleIndex::new(0).unwrap());
        assert_eq!(row3.rho().to_bits(), 0x3fe1921c00da2997);
        assert_eq!(row3.squared_distance().to_bits(), 0x4005a4d5feb8c19d);
    }

    #[test]
    fn ranking_rho_multiplies_by_one_reciprocal_square_root_at_a_time() {
        // Row 2's pair (same series as the pinned-bit test above) is sensitive to
        // multiplication order: `(N * r_i) * r_j` and `N * (r_i * r_j)` round to different
        // last bits here, so this pins the left-to-right order specifically, not just a
        // value a reassociated computation might also happen to produce.
        let samples = ticks(&GENERIC_SERIES);
        let profile = naive_left_profile(&samples, 3);
        let row2 = entry_of(&profile, 2);
        assert_eq!(row2.rho().to_bits(), 0x3fde59b497f4fdc1);
        assert_ne!(row2.rho().to_bits(), 0x3fde59b497f4fdc0);
    }

    #[test]
    fn exact_repeat_beats_a_near_miss_even_when_ranking_ties() {
        // Row 8's window repeats row 4's exactly, and row 0's window is a one-tick near
        // miss of both. Their ranking ρ̂ both clamp to exactly 1 (within 2ε of each other),
        // so ranking alone can't distinguish them and the ranking formula would even favor
        // row 0 (smaller index) — but the exact comparison is decisive: row 4 is a true
        // exact match (ρ² = 1/1 exactly) and row 0 is not, so row 4 must win, with d² = 0.
        const A: [i64; 4] = [235_732_534, -190_400_685, 248_938_659, -501_918_519];
        const B: [i64; 4] = [235_732_534, -190_400_685, 248_938_660, -501_918_519];
        let mut samples = ticks(&A);
        samples.extend(ticks(&B));
        samples.extend(ticks(&B));

        let profile = naive_left_profile(&samples, 4);
        let entry = entry_of(&profile, 8);
        assert_eq!(entry.index(), SampleIndex::new(4).unwrap());
        assert_eq!(entry.rho(), 1.0);
        assert_eq!(entry.squared_distance(), 0.0);
    }

    fn valid_correlation(
        sign: i8,
        cross_squared: u128,
        row_variance: u64,
        neighbor_variance: u64,
    ) -> ExactCorrelation {
        ExactCorrelation::Valid {
            sign,
            cross_squared,
            row_variance,
            neighbor_variance,
            denominator: u128::from(row_variance) * u128::from(neighbor_variance),
        }
    }

    #[test]
    fn ranking_tie_falls_back_to_exact_correlation_not_index() {
        // Equal ranking ρ̂, so the selection can only come from the exact ratio: 100/100 (an
        // exact match) must beat 81/100, no matter that the exact match sits at the larger
        // index.
        let weaker = Candidate {
            index: 1,
            rho_hat: 0.9,
            correlation: valid_correlation(1, 81, 10, 10),
        };
        let exact_match = Candidate {
            index: 9,
            rho_hat: 0.9,
            correlation: valid_correlation(1, 100, 10, 10),
        };
        assert!(nearer(exact_match, weaker));
        assert!(!nearer(weaker, exact_match));
    }

    #[test]
    fn ranking_tolerance_is_symmetric_and_falls_back_within_it() {
        let tiny = f64::EPSILON;
        let band = 2.0 * EPSILON;
        let base = Candidate {
            index: 5,
            rho_hat: 0.5,
            correlation: ExactCorrelation::BothConstant,
        };
        let candidate_at = |rho_hat: f64| Candidate {
            index: 7,
            rho_hat,
            correlation: ExactCorrelation::BothConstant,
        };

        // Clearly outside the band: ρ̂ alone decides, despite the larger index.
        assert!(nearer(candidate_at(0.5 + band + tiny), base));
        assert!(!nearer(base, candidate_at(0.5 + band + tiny)));
        assert!(!nearer(candidate_at(0.5 - band - tiny), base));
        assert!(nearer(base, candidate_at(0.5 - band - tiny)));

        // Inside the band on either side: falls back to the exact tie (equal here), where
        // the smaller index (base, 5) wins over the larger (7).
        assert!(!nearer(candidate_at(0.5 + band - tiny), base));
        assert!(nearer(base, candidate_at(0.5 + band - tiny)));
        assert!(!nearer(candidate_at(0.5 - band + tiny), base));
        assert!(nearer(base, candidate_at(0.5 - band + tiny)));
    }

    #[test]
    fn ranking_tolerance_uses_the_documented_epsilon_not_a_tighter_one() {
        // Row `nb1` and `nb2` are both one-tick-off near matches of `row`; their ranking ρ̂
        // differ by about 3.33e-16, strictly between `2·2⁻⁵³` and the documented `2·2⁻⁴⁹`.
        // Ranking alone favors `nb2` (its ρ̂ rounds to exactly 1.0, `nb1`'s to just under),
        // but the true nearer neighbor is `nb1` (found by an independent, exact search), so
        // the documented ε must be wide enough to route this pair to the exact comparison.
        // A tighter ε = 2⁻⁵³ would put the gap outside the band and let ranking's wrong
        // answer stand.
        let nb1: [i64; 3] = [-802_959_335, 909_343_706, 273_972_096];
        let nb2: [i64; 3] = [-802_959_340, 909_343_706, 273_972_096];
        let row: [i64; 3] = [-802_959_337, 909_343_706, 273_972_096];
        let mut samples = ticks(&nb1);
        samples.extend(ticks(&nb2));
        samples.extend(ticks(&row));

        let profile = naive_left_profile(&samples, 3);
        let entry = entry_of(&profile, 6);
        assert_eq!(entry.index(), SampleIndex::new(0).unwrap());
        assert_eq!(entry.rho().to_bits(), 0x3fef_ffff_ffff_fffd);
    }

    #[test]
    fn exact_comparison_checks_sign_before_magnitude() {
        // Same |ρ|² scale, opposite sign: positive must win regardless of magnitude.
        let positive_small = valid_correlation(1, 1, 10, 10);
        let negative_large = valid_correlation(-1, 81, 10, 10);
        assert_eq!(
            cmp_exact(positive_small, negative_large, Sharing::Row),
            Ordering::Greater
        );
        assert_eq!(
            cmp_exact(negative_large, positive_small, Sharing::Row),
            Ordering::Less
        );
    }

    #[test]
    fn equal_exact_correlation_keeps_the_smaller_index() {
        let tied = valid_correlation(1, 4, 2, 8); // ρ² = 4/16 = 0.25, same on both sides
        let smaller = Candidate {
            index: 2,
            rho_hat: 0.5,
            correlation: tied,
        };
        let larger = Candidate {
            index: 9,
            rho_hat: 0.5,
            correlation: tied,
        };
        assert!(nearer(smaller, larger));
        assert!(!nearer(larger, smaller));
    }

    #[test]
    fn one_constant_ratio_is_one_quarter_not_one_half() {
        // ρ² = 0.3 sits strictly between 1/4 and 1/2: it must beat a one-constant pair
        // (ρ² = 1/4), which it would not if the fixed ratio were mistakenly 1/2.
        let one_constant = ExactCorrelation::OneConstant;
        let valid = valid_correlation(1, 3, 1, 10); // ρ² = 3/10
        assert_eq!(
            cmp_exact(valid, one_constant, Sharing::Row),
            Ordering::Greater
        );
        assert_eq!(cmp_exact(one_constant, valid, Sharing::Row), Ordering::Less);
    }

    #[test]
    fn wide_product_compares_the_carried_limb_not_the_value_mod_two_pow_128() {
        // Three pairs near the `2^126 · 2^63` ceiling where `a·b` truly exceeds `c·d` only
        // because of the limb carried past `2^128`; comparing just the low 128 bits (as
        // dropping the top limb amounts to) gives the opposite answer for every one of them.
        let witnesses: [(u128, u128, u128, u128); 3] = [
            (
                69_578_391_373_808_748_640_188_136_967_499_002_357,
                7_306_870_091_484_026_235,
                11_360_641_757_789_959_836_646_488_788_177_435_588,
                4_643_923_907_442_759_843,
            ),
            (
                85_070_591_297_777_832_805_047_007_575_311_650_259,
                9_223_372_034_090_138_926,
                85_070_591_012_045_208_564_917_258_585_242_011_175,
                9_223_372_030_187_670_072,
            ),
            (
                85_070_591_683_515_808_973_076_982_338_823_578_872,
                9_223_372_028_206_307_030,
                85_070_590_830_675_396_535_354_108_674_782_103_547,
                9_223_372_033_964_830_740,
            ),
        ];
        for (a, b, c, d) in witnesses {
            assert_eq!(cmp_wide_products(a, b, c, d), Ordering::Greater);
            assert_eq!(cmp_wide_products(c, d, a, b), Ordering::Less);
        }
    }

    fn bits(entry: crate::entry::Entry) -> (u64, u64, u64) {
        (
            entry.rho().to_bits(),
            entry.squared_distance().to_bits(),
            entry.index().get().unwrap_or(u64::MAX),
        )
    }

    const MAX_TICK_MAGNITUDE: i64 = 500;

    fn tick_strategy() -> impl Strategy<Value = i64> {
        -MAX_TICK_MAGNITUDE..=MAX_TICK_MAGNITUDE
    }

    fn sample_strategy() -> impl Strategy<Value = Option<i64>> {
        prop_oneof![9 => tick_strategy().prop_map(Some), 1 => Just(None)]
    }

    fn series_and_m() -> impl Strategy<Value = (Vec<Option<i64>>, usize)> {
        (2usize..=5).prop_flat_map(|m| (prop::collection::vec(sample_strategy(), m..=10), Just(m)))
    }

    /// Two independent series sharing one window size, for the AB-join oracle: lengths are
    /// generated independently of each other, so most cases exercise `a` and `b` at
    /// different lengths.
    fn two_series_and_m() -> impl Strategy<Value = (Vec<Option<i64>>, Vec<Option<i64>>, usize)> {
        (2usize..=5).prop_flat_map(|m| {
            (
                prop::collection::vec(sample_strategy(), m..=8),
                prop::collection::vec(sample_strategy(), m..=8),
                Just(m),
            )
        })
    }

    proptest! {
        #![proptest_config(testkit::generators::seeded_default())]
        #[test]
        fn offset_invariance(
            (series, m) in series_and_m(),
            offset in -MAX_TICK_MAGNITUDE..=MAX_TICK_MAGNITUDE,
        ) {
            let shifted: Vec<Option<i64>> =
                series.iter().map(|tick| tick.map(|v| v + offset)).collect();
            let before = naive_left_profile(&series, m);
            let after = naive_left_profile(&shifted, m);
            prop_assert_eq!(before.len(), after.len());
            for ((_, a), (_, b)) in before.iter().zip(&after) {
                prop_assert_eq!(bits(*a), bits(*b));
            }
        }
    }

    /// An exact rational ρ, independent of `naive`'s own floating-point steps: `One` and
    /// `Half` are the fixed constant-pair correlations, `Ratio` is `n / sqrt(vi·vj)` for a
    /// valid-valid pair, compared by cross-multiplying its squared, signed rational form.
    #[derive(Clone, Copy)]
    enum Rho {
        One,
        Half,
        Ratio { n: i128, vi: i128, vj: i128 },
    }

    /// `(sign, numerator, denominator)` with `rho == sign * sqrt(numerator / denominator)`.
    fn canonical(rho: Rho) -> (i8, u128, u128) {
        match rho {
            Rho::One => (1, 1, 1),
            Rho::Half => (1, 1, 4),
            Rho::Ratio { n, vi, vj } => {
                let magnitude = n.unsigned_abs();
                (
                    n.signum() as i8,
                    magnitude * magnitude,
                    (vi as u128) * (vj as u128),
                )
            }
        }
    }

    fn cmp_rho(a: Rho, b: Rho) -> Ordering {
        let (sign_a, num_a, den_a) = canonical(a);
        let (sign_b, num_b, den_b) = canonical(b);
        if sign_a != sign_b {
            return sign_a.cmp(&sign_b);
        }
        if sign_a == 0 {
            return Ordering::Equal;
        }
        let magnitude_order = (num_a * den_b).cmp(&(num_b * den_a));
        if sign_a > 0 {
            magnitude_order
        } else {
            magnitude_order.reverse()
        }
    }

    /// `None` for a window with any missing tick; `(is_constant, sum, variance_numerator)`
    /// otherwise, computed independently of `window_stats`.
    fn oracle_window(
        samples: &[Option<i64>],
        start: usize,
        m: usize,
    ) -> Option<(bool, i128, i128)> {
        let window: Option<Vec<i64>> = samples[start..start + m].iter().copied().collect();
        let window = window?;
        let min = *window.iter().min().expect("m > 0");
        let max = *window.iter().max().expect("m > 0");
        if min == max {
            return Some((true, window.iter().map(|&t| t as i128).sum(), 0));
        }
        let sum: i128 = window.iter().map(|&t| t as i128).sum();
        let sum_sq: i128 = window.iter().map(|&t| (t as i128) * (t as i128)).sum();
        let variance_numerator = m as i128 * sum_sq - sum * sum;
        (1..1i128 << 63)
            .contains(&variance_numerator)
            .then_some((false, sum, variance_numerator))
    }

    fn oracle_cross_term_pair(
        a: &[Option<i64>],
        i: usize,
        b: &[Option<i64>],
        j: usize,
        m: usize,
    ) -> i128 {
        (0..m)
            .map(|k| {
                let x = a[i + k].expect("non-finite windows are filtered by the caller");
                let y = b[j + k].expect("non-finite windows are filtered by the caller");
                x as i128 * y as i128
            })
            .sum()
    }

    /// The exact ρ of the pair `(a[i], b[j])`, or `None` if either window has a missing tick;
    /// `a` and `b` may be the same slice or different ones.
    fn rho_of_pair_ab(
        a: &[Option<i64>],
        m: usize,
        i: usize,
        b: &[Option<i64>],
        j: usize,
    ) -> Option<Rho> {
        let (i_constant, i_sum, i_variance) = oracle_window(a, i, m)?;
        let (j_constant, j_sum, j_variance) = oracle_window(b, j, m)?;
        Some(match (i_constant, j_constant) {
            (true, true) => Rho::One,
            (true, false) | (false, true) => Rho::Half,
            (false, false) => {
                let qt = oracle_cross_term_pair(a, i, b, j, m);
                let n = m as i128 * qt - i_sum * j_sum;
                Rho::Ratio {
                    n,
                    vi: i_variance,
                    vj: j_variance,
                }
            }
        })
    }

    /// The exact ρ of pair `(i, j)` within one series, or `None` if either window has a
    /// missing tick.
    fn rho_of_pair(samples: &[Option<i64>], m: usize, i: usize, j: usize) -> Option<Rho> {
        rho_of_pair_ab(samples, m, i, samples, j)
    }

    /// Row `i`'s best neighbor among `j ≤ i − e − 1`, exact ρ with ties going to the smaller
    /// index; `None` if `i`'s own window has a missing tick or no candidate exists.
    fn best_neighbor(samples: &[Option<i64>], m: usize, i: usize, e: usize) -> Option<(u64, Rho)> {
        oracle_window(samples, i, m)?;
        if i <= e {
            return None;
        }
        let mut best: Option<(u64, Rho)> = None;
        for j in 0..=i - e - 1 {
            let Some(rho) = rho_of_pair(samples, m, i, j) else {
                continue;
            };
            let better = match best {
                None => true,
                Some((best_j, best_rho)) => match cmp_rho(rho, best_rho) {
                    Ordering::Greater => true,
                    Ordering::Equal => (j as u64) < best_j,
                    Ordering::Less => false,
                },
            };
            if better {
                best = Some((j as u64, rho));
            }
        }
        best
    }

    /// Row `i`'s `k` nearest neighbors among `j ≤ i − e − 1`, exact ρ nearest first, ties to
    /// the smaller index; fewer than `k` if fewer candidates are eligible, empty if `i`'s own
    /// window has a missing tick. Built on the same `rho_of_pair`/`cmp_rho` this module's
    /// single-best oracle (`best_neighbor`) uses, just keeping every eligible candidate instead
    /// of only the winner.
    fn best_k_neighbors(
        samples: &[Option<i64>],
        m: usize,
        i: usize,
        e: usize,
        k: usize,
    ) -> Vec<(u64, Rho)> {
        if oracle_window(samples, i, m).is_none() || i <= e {
            return Vec::new();
        }
        let mut candidates: Vec<(u64, Rho)> = (0..=i - e - 1)
            .filter_map(|j| rho_of_pair(samples, m, i, j).map(|rho| (j as u64, rho)))
            .collect();
        candidates.sort_by(
            |&(index_a, rho_a), &(index_b, rho_b)| match cmp_rho(rho_a, rho_b) {
                Ordering::Greater => Ordering::Less,
                Ordering::Less => Ordering::Greater,
                Ordering::Equal => index_a.cmp(&index_b),
            },
        );
        candidates.truncate(k);
        candidates
    }

    /// Row `i`'s best RIGHT neighbor among `j > i + e`, exact ρ with ties going to the smaller
    /// index; the opposite-direction mirror of `best_neighbor`, built from the same
    /// `rho_of_pair`/`cmp_rho`.
    fn best_right_neighbor(
        samples: &[Option<i64>],
        m: usize,
        i: usize,
        e: usize,
        num_windows: usize,
    ) -> Option<(u64, Rho)> {
        oracle_window(samples, i, m)?;
        let mut best: Option<(u64, Rho)> = None;
        for j in (i + e + 1)..num_windows {
            let Some(rho) = rho_of_pair(samples, m, i, j) else {
                continue;
            };
            let better = match best {
                None => true,
                Some((best_j, best_rho)) => match cmp_rho(rho, best_rho) {
                    Ordering::Greater => true,
                    Ordering::Equal => (j as u64) < best_j,
                    Ordering::Less => false,
                },
            };
            if better {
                best = Some((j as u64, rho));
            }
        }
        best
    }

    /// The better of `best_neighbor` and `best_right_neighbor`, ties going to the smaller
    /// index; independent of which side the index came from, since a left index is always
    /// smaller than any right index for the same row.
    fn best_two_sided_neighbor(
        samples: &[Option<i64>],
        m: usize,
        i: usize,
        e: usize,
        num_windows: usize,
    ) -> Option<(u64, Rho)> {
        let left = best_neighbor(samples, m, i, e);
        let right = best_right_neighbor(samples, m, i, e, num_windows);
        match (left, right) {
            (None, None) => None,
            (Some(l), None) => Some(l),
            (None, Some(r)) => Some(r),
            (Some((lj, lrho)), Some((rj, rrho))) => match cmp_rho(lrho, rrho) {
                Ordering::Greater => Some((lj, lrho)),
                Ordering::Less => Some((rj, rrho)),
                Ordering::Equal if lj < rj => Some((lj, lrho)),
                Ordering::Equal => Some((rj, rrho)),
            },
        }
    }

    /// Row `i` of `a`'s best neighbor among **every** window of `b` (no exclusion zone),
    /// exact ρ with ties going to the smaller `b` index; `None` if `i`'s own window has a
    /// missing tick or `b` has no eligible window at all. The AB-join mirror of
    /// `best_neighbor`, built from the same `rho_of_pair_ab`/`cmp_rho`.
    fn best_ab_neighbor(
        a: &[Option<i64>],
        m: usize,
        i: usize,
        b: &[Option<i64>],
        num_windows_b: usize,
    ) -> Option<(u64, Rho)> {
        oracle_window(a, i, m)?;
        let mut best: Option<(u64, Rho)> = None;
        for j in 0..num_windows_b {
            let Some(rho) = rho_of_pair_ab(a, m, i, b, j) else {
                continue;
            };
            let better = match best {
                None => true,
                Some((best_j, best_rho)) => match cmp_rho(rho, best_rho) {
                    Ordering::Greater => true,
                    Ordering::Equal => (j as u64) < best_j,
                    Ordering::Less => false,
                },
            };
            if better {
                best = Some((j as u64, rho));
            }
        }
        best
    }

    /// Cross-checks `naive` against an independently coded, exact-rational computation of ρ:
    /// the chosen neighbor must agree in every case, with zero tolerance. Prints the
    /// disagreement count, which must be `0`.
    #[test]
    fn matches_rational_oracle() {
        let mut runner = TestRunner::deterministic();
        let strategy = series_and_m();
        let cases = 4096;
        let mut disagreements = 0usize;

        for case in 0..cases {
            let (series, m) = strategy
                .new_tree(&mut runner)
                .expect("strategy produces a value")
                .current();
            let e = crate::exclusion::zone(m);
            let profile = naive_left_profile(&series, m);

            for &(row, entry) in &profile {
                let naive_index = entry.index().get();
                let oracle_index = best_neighbor(&series, m, row as usize, e).map(|(j, _)| j);
                if naive_index != oracle_index {
                    disagreements += 1;
                    std::eprintln!(
                        "case {case} row {row}: naive picked {naive_index:?}, \
                         oracle picked {oracle_index:?}"
                    );
                }
            }
        }

        std::println!(
            "matches_rational_oracle: {disagreements} disagreements across {cases} cases"
        );
        assert_eq!(disagreements, 0);
    }

    // ---------- naive_left_topk ----------

    /// `k = 1` reproduces `naive_left_profile`'s single winner, bit for bit, on every row;
    /// `finish_k`'s empty case stands in for `naive_left_profile`'s own [`Entry::NONE`].
    #[test]
    fn topk_k_one_equals_left_profile_on_bits() {
        let mut runner = TestRunner::deterministic();
        let strategy = series_and_m();
        let cases = 512;

        for _ in 0..cases {
            let (series, m) = strategy
                .new_tree(&mut runner)
                .expect("strategy produces a value")
                .current();
            let profile = naive_left_profile(&series, m);
            let topk = naive_left_topk(&series, m, NonZeroUsize::new(1).unwrap());

            for (&(row, expected), (topk_row, entries)) in profile.iter().zip(&topk) {
                assert_eq!(row, *topk_row);
                assert!(entries.len() <= 1);
                let actual = entries
                    .first()
                    .copied()
                    .unwrap_or(crate::entry::Entry::NONE);
                assert_eq!(
                    bits(actual),
                    bits(expected),
                    "series {series:?}, m {m}, row {row}"
                );
            }
        }
    }

    /// Cross-checks `naive_left_topk` against the same independent exact-rational oracle
    /// `matches_rational_oracle` uses above (this module's own `Rho`/`cmp_rho`, not a
    /// `num-rational` dependency: one already lives here), extended by `best_k_neighbors` to
    /// keep every eligible candidate instead of only the winner, across every `k` the spec
    /// names. Compares the chosen *sequence* of indices, so a wrong order or a wrong tie
    /// resolution fails this just as a wrong single winner would. Prints the disagreement
    /// count, which must be `0`.
    #[test]
    fn matches_rational_oracle_topk() {
        let mut runner = TestRunner::deterministic();
        let strategy = series_and_m();
        let cases = 1024;
        let mut disagreements = 0usize;
        let mut nontrivial = 0usize;
        let mut total = 0usize;

        for case in 0..cases {
            let (series, m) = strategy
                .new_tree(&mut runner)
                .expect("strategy produces a value")
                .current();
            let e = crate::exclusion::zone(m);

            for &k in &[1usize, 2, 3, 7] {
                let profile = naive_left_topk(&series, m, NonZeroUsize::new(k).unwrap());
                for (row, entries) in &profile {
                    total += 1;
                    let actual_indices: Vec<u64> = entries
                        .iter()
                        .map(|entry| entry.index().get().expect("a kept entry is never NONE"))
                        .collect();
                    let oracle = best_k_neighbors(&series, m, *row as usize, e, k);
                    let oracle_indices: Vec<u64> = oracle.iter().map(|&(j, _)| j).collect();
                    if actual_indices != oracle_indices {
                        disagreements += 1;
                        std::eprintln!(
                            "case {case} row {row} k {k}: naive picked {actual_indices:?}, \
                             oracle picked {oracle_indices:?}"
                        );
                    }
                    if !actual_indices.is_empty() {
                        nontrivial += 1;
                    }
                }
            }
        }

        std::println!(
            "matches_rational_oracle_topk: {disagreements} disagreements across {total} \
             (case, row, k) combinations"
        );
        assert_eq!(disagreements, 0);
        assert!(
            nontrivial * 10 > total,
            "only {nontrivial} of {total} (row, k) combinations had any eligible candidate; \
             the generator is too degenerate to exercise top-k at all"
        );
    }

    /// Two exact repeats of the row at different indices (a true tie: both `ρ = 1`, `d² = 0`)
    /// and nothing else in range can reach `ρ = 1`, so `k = 2` must keep exactly the repeats,
    /// smaller index first; `k = 3` must then reach past the tie to a real third entry. A
    /// mutant that ignores `k` (always keeping only one) truncates every case below to a
    /// single entry.
    #[test]
    fn topk_exact_repeats_tie_and_a_real_third_entry_follows() {
        #[rustfmt::skip]
        let samples = ticks(&[
            1, 2, 9, // index 0: exact repeat of the row
            1, 2, 9, // index 3: exact repeat of the row
            0, 0, 0, // filler, no exact match possible
            1, 2, 9, // row (index 9)
        ]);

        let k2 = naive_left_topk(&samples, 3, NonZeroUsize::new(2).unwrap());
        let (row, entries) = &k2[9];
        assert_eq!(*row, 9);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].index(), SampleIndex::new(0).unwrap());
        assert_eq!(entries[0].rho(), 1.0);
        assert_eq!(entries[0].squared_distance(), 0.0);
        assert_eq!(entries[1].index(), SampleIndex::new(3).unwrap());
        assert_eq!(entries[1].rho(), 1.0);
        assert_eq!(entries[1].squared_distance(), 0.0);

        let k3 = naive_left_topk(&samples, 3, NonZeroUsize::new(3).unwrap());
        let (row3, entries3) = &k3[9];
        assert_eq!(*row3, 9);
        assert_eq!(entries3.len(), 3);
        for (a, b) in entries3[..2].iter().zip(&entries[..2]) {
            assert_eq!(bits(*a), bits(*b));
        }
        let third_index = entries3[2]
            .index()
            .get()
            .expect("a kept entry is never NONE");
        assert!(![0u64, 3u64].contains(&third_index));
        assert!(entries3[2].squared_distance() > 0.0);
        assert!(entries3[2].squared_distance().is_finite());
    }

    /// A row with a missing tick never produces candidates, for any `k`; neighbors with a
    /// missing tick (`j` = 1, 2, 3, each touching the gap at position 3) are skipped, never
    /// chosen, no matter how large `k` is — only `j` = 0, 4, 5 are eligible, and `j` = 0 is an
    /// exact repeat of the row (same series
    /// `none_inside_window_is_nonfinite_and_never_a_neighbor` above uses for
    /// `naive_left_profile`).
    #[test]
    fn topk_a_gap_empties_the_row_for_every_k_and_is_never_offered_as_a_neighbor() {
        let mut samples = ticks(&[1, 2, 3, 0, 5, 6, 7, 1, 2, 3]);
        samples[3] = None;

        for &k in &[1usize, 3, 5] {
            let profile = naive_left_topk(&samples, 3, NonZeroUsize::new(k).unwrap());
            for row in [1u64, 2, 3] {
                let entries = &profile.iter().find(|&&(r, _)| r == row).unwrap().1;
                assert!(entries.is_empty(), "row {row}, k {k}: {entries:?}");
            }
            let (row, entries) = &profile[7];
            assert_eq!(*row, 7);
            assert_eq!(entries.len(), k.min(3), "k {k}: {entries:?}");
            assert_eq!(entries[0].index(), SampleIndex::new(0).unwrap());
            assert_eq!(entries[0].rho(), 1.0);
            assert_eq!(entries[0].squared_distance(), 0.0);
            for entry in entries {
                let neighbor = entry.index().get().expect("a kept entry is never NONE");
                assert!(
                    ![1u64, 2, 3].contains(&neighbor),
                    "gap-touching neighbor {neighbor} was kept"
                );
            }
        }
    }

    /// A constant row ranks a constant neighbor (`ρ = 1`) above every valid neighbor (fixed
    /// `ρ = 1/2`, an exact tie among themselves, smaller index first).
    #[test]
    fn topk_constant_row_ranks_constant_above_tied_valid_neighbors() {
        let samples = ticks(&[5, 5, 5, 1, 2, 9, 9, 9]);
        let profile = naive_left_topk(&samples, 3, NonZeroUsize::new(4).unwrap());
        let (row, entries) = &profile[5];
        assert_eq!(*row, 5);
        assert_eq!(entries.len(), 4);

        assert_eq!(entries[0].index(), SampleIndex::new(0).unwrap());
        assert_eq!(entries[0].rho(), 1.0);
        assert_eq!(entries[0].squared_distance(), 0.0);

        let tied_indices: Vec<u64> = entries[1..]
            .iter()
            .map(|entry| entry.index().get().expect("a kept entry is never NONE"))
            .collect();
        assert_eq!(tied_indices, [1u64, 2, 3]); // exact ρ ties resolve to the smaller index
        for entry in &entries[1..] {
            assert_eq!(entry.rho(), 0.5);
            assert_eq!(entry.squared_distance(), 3.0);
        }
    }

    /// Fewer eligible candidates than `k`: `finish_k` returns only what is real, never padded.
    #[test]
    fn topk_k_exceeding_the_candidate_count_returns_only_the_real_ones() {
        let samples = ticks(&[1, 2, 7, 7, 7]);
        let profile = naive_left_topk(&samples, 3, NonZeroUsize::new(10).unwrap());
        let (row, entries) = &profile[2];
        assert_eq!(*row, 2);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].index(), SampleIndex::new(0).unwrap());
        assert!(entries[0].squared_distance().is_finite());
    }

    /// `m == 0` and `m > n` both return no rows at all, for `naive_left_topk` exactly as for
    /// `naive_left_profile`.
    #[test]
    fn topk_zero_or_oversized_window_has_no_rows() {
        let samples = ticks(&[1, 2, 3, 4]);
        assert!(naive_left_topk(&samples, 0, NonZeroUsize::new(1).unwrap()).is_empty());
        assert!(naive_left_topk(&samples, 5, NonZeroUsize::new(1).unwrap()).is_empty());
    }

    // ---------- naive_two_sided ----------

    fn two_sided_of(profile: &[(u64, TwoSided)], row: u64) -> TwoSided {
        profile
            .iter()
            .find(|&&(r, _)| r == row)
            .expect("row exists")
            .1
    }

    /// `left` reproduces `naive_left_profile`'s entry, bit for bit, on every row.
    #[test]
    fn two_sided_left_equals_left_profile_on_bits() {
        let mut runner = TestRunner::deterministic();
        let strategy = series_and_m();
        let cases = 512;

        for _ in 0..cases {
            let (series, m) = strategy
                .new_tree(&mut runner)
                .expect("strategy produces a value")
                .current();
            let left_profile = naive_left_profile(&series, m);
            let two_sided = naive_two_sided(&series, m);

            for (&(row, expected), &(two_row, ts)) in left_profile.iter().zip(&two_sided) {
                assert_eq!(row, two_row);
                assert_eq!(
                    bits(ts.left),
                    bits(expected),
                    "series {series:?}, m {m}, row {row}"
                );
            }
        }
    }

    /// A minimal, hand-worked mirror: with only one eligible right candidate per row (no tie
    /// is possible), `right`'s `rho`/`squared_distance`/`index` equal the entry
    /// `naive_left_profile` finds for the corresponding row of the series reversed end to
    /// end, mapped back through `(num_windows - 1) - reversed_index`. (A tie among several
    /// right candidates breaks this equality — see
    /// `two_sided_right_tie_resolves_to_the_smaller_index` below for why.)
    #[test]
    fn two_sided_right_matches_the_reversed_left_profile_when_only_one_candidate_is_eligible() {
        let series = ticks(&[3, 1, 4, 1, 5]);
        let m = 3;
        let two_sided = naive_two_sided(&series, m);
        let num_windows = two_sided.len();
        let reversed: Vec<Option<i64>> = series.iter().rev().copied().collect();
        let reversed_profile = naive_left_profile(&reversed, m);

        // e = ceil(3/4) = 1, num_windows = 3: row 0's only right candidate is j = 2 (j = 1 is
        // excluded, and there is no j = 3), so there is nothing to tie with.
        let ts = two_sided_of(&two_sided, 0);
        let mirrored_row = (num_windows - 1) as u64;
        let (_, reversed_entry) = reversed_profile[mirrored_row as usize];
        assert_eq!(ts.right.rho(), reversed_entry.rho());
        assert_eq!(
            ts.right.squared_distance(),
            reversed_entry.squared_distance()
        );
        let reversed_j = reversed_entry
            .index()
            .get()
            .expect("row 0 has exactly one eligible right candidate");
        assert_eq!(
            ts.right.index().get(),
            Some((num_windows - 1) as u64 - reversed_j)
        );
    }

    /// e = ceil(4/4) = 1: row 0's right-side mirror of row 2's window [7,7,7,7] is identical
    /// and sits at `j = i + e`, which must stay excluded; the only legal right candidate is
    /// the non-constant window at `j = i + e + 1`, so the right answer is the mixed-constancy
    /// pair (ρ = 1/2), not the excluded perfect match (reverses
    /// `exclusion_boundary_excludes_e_and_allows_e_plus_1` above, which pins the same
    /// boundary on the left).
    #[test]
    fn two_sided_right_exclusion_excludes_e_and_allows_e_plus_1() {
        let samples = ticks(&[7, 7, 7, 7, 7, 1]);
        let two_sided = naive_two_sided(&samples, 4);
        let ts = two_sided_of(&two_sided, 0);
        assert_eq!(ts.right.index(), SampleIndex::new(2).unwrap());
        assert_eq!(ts.right.rho(), 0.5);
        assert_eq!(ts.right.squared_distance(), 4.0);
    }

    /// m = 5: ceil(5/4) = 2, floor(5/4) = 1. Row 0 has no legal right candidate under the
    /// correct half-width (only two window starts follow it, both within e), but would gain
    /// one under the floored half-width (reverses `exclusion_half_width_rounds_up` above).
    #[test]
    fn two_sided_right_half_width_rounds_up() {
        let samples = ticks(&[7, 6, 5, 4, 3, 2, 1]);
        let two_sided = naive_two_sided(&samples, 5);
        let ts = two_sided_of(&two_sided, 0);
        assert_eq!(ts.right.index(), SampleIndex::NONE);
        assert_eq!(ts.right.squared_distance(), f64::INFINITY);
    }

    /// Row's window has an exact repeat on each side — index 0 on the left, index 9 on the
    /// right — both true exact matches (ρ = 1, d² = 0): an exact tie under the comparator, so
    /// `nearest` must take the smaller index, the left one, not default to whichever side a
    /// careless tie-break favors.
    #[test]
    fn two_sided_nearest_prefers_the_smaller_index_on_an_exact_tie() {
        #[rustfmt::skip]
        let samples = ticks(&[
            1, 2, 9, // index 0: exact repeat (left)
            0, 0, 0, // filler
            1, 2, 9, // row (index 6)
            1, 2, 9, // index 9: exact repeat (right)
        ]);
        let two_sided = naive_two_sided(&samples, 3);
        let ts = two_sided_of(&two_sided, 6);

        assert_eq!(ts.left.index(), SampleIndex::new(0).unwrap());
        assert_eq!(ts.left.rho(), 1.0);
        assert_eq!(ts.left.squared_distance(), 0.0);

        assert_eq!(ts.right.index(), SampleIndex::new(9).unwrap());
        assert_eq!(ts.right.rho(), 1.0);
        assert_eq!(ts.right.squared_distance(), 0.0);

        assert_eq!(ts.nearest.index(), SampleIndex::new(0).unwrap());
        assert_eq!(ts.nearest.rho(), 1.0);
        assert_eq!(ts.nearest.squared_distance(), 0.0);
    }

    /// `m = 2`: any two non-constant windows are perfectly correlated one way or the other
    /// (the exact comparator agrees exactly, not just within the ranking's tolerance). Right
    /// candidates at index 2 and index 4 share the row's own decreasing direction — an exact
    /// tie at ρ = 1 — while the window between them (index 3, mixing the end of one candidate
    /// with the start of the other) goes the other way and cannot compete. The tie must
    /// resolve to the smaller index, 2: a reversed-series implementation gets this backwards
    /// (see this module's doc comment on `naive_two_sided`), which is why this test exists.
    #[test]
    fn two_sided_right_tie_resolves_to_the_smaller_index() {
        let samples = ticks(&[100, 50, 9, 5, 30, 10]);
        let two_sided = naive_two_sided(&samples, 2);
        let ts = two_sided_of(&two_sided, 0);
        assert_eq!(ts.right.index(), SampleIndex::new(2).unwrap());
        assert_eq!(ts.right.rho(), 1.0);
    }

    /// `left` is unaffected by samples appended after it was computed: recomputing over
    /// an extended series leaves every original row's `left` unchanged, bit for bit. `right`
    /// is not asserted here — it is meant to look ahead, so it may legitimately change as
    /// more future data arrives.
    #[test]
    fn two_sided_left_is_unaffected_by_samples_appended_later() {
        let prefix = ticks(&[3, 1, 4, 1, 5, 9, 2, 6]);
        let m = 3;
        let before = naive_two_sided(&prefix, m);

        let mut extended = prefix.clone();
        extended.extend(ticks(&[5, 3, 5, 8, 9, 7]));
        let after = naive_two_sided(&extended, m);

        for (&(row, before_ts), &(after_row, after_ts)) in before.iter().zip(&after) {
            assert_eq!(row, after_row);
            assert_eq!(
                bits(after_ts.left),
                bits(before_ts.left),
                "row {row}: left changed after appending future samples"
            );
        }
    }

    /// Cross-checks `naive_two_sided`'s `left`, `right` and `nearest` against the independent
    /// exact-rational oracle (`best_neighbor`, `best_right_neighbor`, `best_two_sided_neighbor`
    /// above), which enumerates each side's candidate range directly rather than through this
    /// module's own `e`/mirroring arithmetic — so an off-by-one in either exclusion boundary,
    /// in the reversed-series index mapping, or in the left/right tie-break shows up as a
    /// disagreement here. Prints the disagreement count, which must be `0`.
    #[test]
    fn matches_rational_oracle_two_sided() {
        let mut runner = TestRunner::deterministic();
        let strategy = series_and_m();
        let cases = 2048;
        let mut disagreements = 0usize;
        let mut nontrivial = 0usize;
        let mut total = 0usize;

        for case in 0..cases {
            let (series, m) = strategy
                .new_tree(&mut runner)
                .expect("strategy produces a value")
                .current();
            let e = crate::exclusion::zone(m);
            let two_sided = naive_two_sided(&series, m);
            let num_windows = two_sided.len();

            for &(row, ts) in &two_sided {
                total += 3;
                let i = row as usize;
                let left_expected = best_neighbor(&series, m, i, e).map(|(j, _)| j);
                let right_expected =
                    best_right_neighbor(&series, m, i, e, num_windows).map(|(j, _)| j);
                let nearest_expected =
                    best_two_sided_neighbor(&series, m, i, e, num_windows).map(|(j, _)| j);

                let left_actual = ts.left.index().get();
                let right_actual = ts.right.index().get();
                let nearest_actual = ts.nearest.index().get();

                for (label, actual, expected) in [
                    ("left", left_actual, left_expected),
                    ("right", right_actual, right_expected),
                    ("nearest", nearest_actual, nearest_expected),
                ] {
                    if actual != expected {
                        disagreements += 1;
                        std::eprintln!(
                            "case {case} row {row} {label}: naive picked {actual:?}, \
                             oracle picked {expected:?}"
                        );
                    } else if actual.is_some() {
                        nontrivial += 1;
                    }
                }
            }
        }

        std::println!(
            "matches_rational_oracle_two_sided: {disagreements} disagreements across {total} \
             (case, row, side) combinations"
        );
        assert_eq!(disagreements, 0);
        assert!(
            nontrivial * 5 > total,
            "only {nontrivial} of {total} (row, side) combinations had any eligible candidate; \
             the generator is too degenerate to exercise two_sided at all"
        );
    }

    // ---------- naive_ab_join ----------

    /// Every row of `a = b` may pick itself: with no exclusion zone, row `i`'s own window is
    /// always an eligible, exact-matching candidate in `b`, so (absent an earlier exact
    /// repeat with a smaller index) the winner is always `i` itself at distance 0 — the
    /// opposite of `naive_left_profile`, whose exclusion zone forbids this exact pairing.
    #[test]
    fn ab_join_of_a_series_against_itself_each_row_matches_itself_at_zero_distance() {
        let samples = ticks(&GENERIC_SERIES);
        let profile = naive_ab_join(&samples, &samples, 3);
        for &(row, entry) in &profile {
            assert_eq!(entry.index(), SampleIndex::new(row).unwrap());
            assert_eq!(entry.rho(), 1.0);
            assert_eq!(entry.squared_distance(), 0.0);
        }
    }

    /// `a` and `b` have different lengths and therefore different window counts; the result
    /// has one entry per window of `a` alone, and a window of `b` that exactly repeats `a`'s
    /// row is found regardless of where it falls in `b`.
    #[test]
    fn different_lengths_of_a_and_b_each_use_their_own_window_count() {
        let a = ticks(&[1, 2, 3, 4, 5]);
        let b = ticks(&[9, 9, 9, 1, 2, 3, 7, 7]);
        let profile = naive_ab_join(&a, &b, 3);
        assert_eq!(profile.len(), 3); // 5 - 3 + 1 windows of `a`
        let entry = entry_of(&profile, 0);
        assert_eq!(entry.index(), SampleIndex::new(3).unwrap());
        assert_eq!(entry.rho(), 1.0);
        assert_eq!(entry.squared_distance(), 0.0);
    }

    /// `b` shorter than `m` has no windows at all: every row of `a` is `Entry::NONE`, never
    /// an empty result (that is reserved for `a` itself being too short or `m == 0`).
    #[test]
    fn b_shorter_than_m_leaves_every_row_of_a_none() {
        let a = ticks(&[1, 2, 3, 4, 5]);
        let b = ticks(&[9, 9]);
        let profile = naive_ab_join(&a, &b, 3);
        assert_eq!(profile.len(), 3);
        for &(_, entry) in &profile {
            assert_eq!(entry.index(), SampleIndex::NONE);
            assert_eq!(entry.squared_distance(), f64::INFINITY);
        }
    }

    /// Two exact repeats of `a`'s row sit in `b`, at different indices: the tie must resolve
    /// to the smaller one, exactly as [`super::nearer`]'s tie rule does for a self-join.
    #[test]
    fn ab_join_tie_between_two_exact_repeats_in_b_goes_to_the_smaller_index() {
        let a = ticks(&[1, 2, 9]);
        #[rustfmt::skip]
        let b = ticks(&[
            1, 2, 9, // index 0: exact repeat
            0, 0, 0, // index 3: filler, no match
            1, 2, 9, // index 6: exact repeat too
        ]);
        let profile = naive_ab_join(&a, &b, 3);
        let entry = entry_of(&profile, 0);
        assert_eq!(entry.index(), SampleIndex::new(0).unwrap());
        assert_eq!(entry.rho(), 1.0);
        assert_eq!(entry.squared_distance(), 0.0);
    }

    /// `b`'s only window would be an exact match of `a`'s row, but a gap inside it makes it
    /// non-finite: it is never chosen, even as the sole candidate, leaving the row at
    /// `Entry::NONE` rather than a false exact match.
    #[test]
    fn ab_join_a_lone_gap_window_in_b_is_never_chosen_leaving_the_row_none() {
        let a = ticks(&[1, 2, 9]);
        let mut b = ticks(&[1, 2, 9]);
        b[1] = None;
        let profile = naive_ab_join(&a, &b, 3);
        let entry = entry_of(&profile, 0);
        assert_eq!(entry.index(), SampleIndex::NONE);
        assert_eq!(entry.squared_distance(), f64::INFINITY);
    }

    /// Cross-checks `naive_ab_join` against the module's own exact-rational oracle, extended
    /// to two independent series (`best_ab_neighbor`, built on `rho_of_pair_ab`): every row's
    /// chosen `b` index must agree, with zero tolerance. Prints the disagreement count, which
    /// must be `0`.
    #[test]
    fn matches_rational_oracle_ab_join() {
        let mut runner = TestRunner::deterministic();
        let strategy = two_series_and_m();
        let cases = 2048;
        let mut disagreements = 0usize;
        let mut nontrivial = 0usize;
        let mut total = 0usize;

        for case in 0..cases {
            let (a, b, m) = strategy
                .new_tree(&mut runner)
                .expect("strategy produces a value")
                .current();
            let profile = naive_ab_join(&a, &b, m);
            let num_windows_b = b.len().checked_sub(m).map_or(0, |rest| rest + 1);

            for &(row, entry) in &profile {
                total += 1;
                let actual = entry.index().get();
                let expected =
                    best_ab_neighbor(&a, m, row as usize, &b, num_windows_b).map(|(j, _)| j);
                if actual != expected {
                    disagreements += 1;
                    std::eprintln!(
                        "case {case} row {row}: naive picked {actual:?}, \
                         oracle picked {expected:?}"
                    );
                } else if actual.is_some() {
                    nontrivial += 1;
                }
            }
        }

        std::println!(
            "matches_rational_oracle_ab_join: {disagreements} disagreements across {total} \
             (case, row) combinations"
        );
        assert_eq!(disagreements, 0);
        assert!(
            nontrivial * 5 > total,
            "only {nontrivial} of {total} (case, row) combinations had any eligible candidate; \
             the generator is too degenerate to exercise ab_join at all"
        );
    }

    // ---------- naive_left_profile_nonnorm ----------

    fn nonnorm_of(profile: &[(u64, NonNormEntry)], row: u64) -> NonNormEntry {
        profile
            .iter()
            .find(|&&(r, _)| r == row)
            .expect("row exists")
            .1
    }

    /// Row `i`'s nearest among `j ≤ i − e − 1` by plain i128 `Σ(a−b)²` on `i64` windows,
    /// strictly smaller distance only (so ties keep the smaller index); `None` if `i`'s
    /// window has a gap or no candidate exists. Shares nothing with the oracle's own code.
    fn best_nonnorm(samples: &[Option<i64>], m: usize, i: usize, e: usize) -> Option<(u64, i128)> {
        let row: Option<Vec<i64>> = samples[i..i + m].iter().copied().collect();
        let row = row?;
        let mut best: Option<(u64, i128)> = None;
        for j in 0..i.saturating_sub(e) {
            let neighbor: Option<Vec<i64>> = samples[j..j + m].iter().copied().collect();
            let Some(neighbor) = neighbor else { continue };
            let distance: i128 = row
                .iter()
                .zip(&neighbor)
                .map(|(&x, &y)| (i128::from(x) - i128::from(y)).pow(2))
                .sum();
            if best.is_none_or(|(_, current)| distance < current) {
                best = Some((j as u64, distance));
            }
        }
        best
    }

    #[test]
    fn nonnorm_matches_plain_i128_distances() {
        let mut runner = TestRunner::deterministic();
        let strategy = series_and_m();
        let mut disagreements = 0usize;
        let mut nontrivial = 0usize;
        let mut total = 0usize;

        for case in 0..2048 {
            let (series, m) = strategy
                .new_tree(&mut runner)
                .expect("strategy produces a value")
                .current();
            let e = crate::exclusion::zone(m);
            for &(row, entry) in &naive_left_profile_nonnorm(&series, m) {
                total += 1;
                let expected = best_nonnorm(&series, m, row as usize, e);
                let actual = entry.index().get().map(|j| (j, entry.distance()));
                let expected_pair = expected.map(|(j, d)| (j, NonNormDistance::Narrow(d as u64)));
                if actual != expected_pair {
                    disagreements += 1;
                    std::eprintln!("case {case} row {row}: {actual:?} vs {expected_pair:?}");
                } else if actual.is_some() {
                    nontrivial += 1;
                }
            }
        }
        assert_eq!(disagreements, 0);
        assert!(
            nontrivial * 5 > total,
            "{nontrivial} of {total} non-trivial"
        );
    }

    #[test]
    fn nonnorm_distance_is_u64_just_below_the_bound_and_i128_at_it() {
        // m = 4 and global range R: m·R² = 2^64 - 2^34 + 4 for R = 2^31 - 1 (fits u64), and
        // exactly 2^64 for R = 2^31 (does not).
        for (range, narrow) in [((1i64 << 31) - 1, true), (1i64 << 31, false)] {
            let samples = ticks(&[0, 0, 0, 0, range, 0, 0, 0]);
            let profile = naive_left_profile_nonnorm(&samples, 4);
            let entry = nonnorm_of(&profile, 4);
            let squared = i128::from(range) * i128::from(range);
            let expected = if narrow {
                NonNormDistance::Narrow(squared as u64)
            } else {
                NonNormDistance::Wide(I256::from(squared))
            };
            assert_eq!(entry.index(), SampleIndex::new(0).unwrap());
            assert_eq!(entry.distance(), expected);
        }
    }

    #[test]
    fn nonnorm_equal_flat_windows_have_zero_distance() {
        let samples = ticks(&[5, 5, 5, 1, 2, 3, 5, 5, 5]);
        let profile = naive_left_profile_nonnorm(&samples, 3);
        let entry = nonnorm_of(&profile, 6);
        assert_eq!(entry.index(), SampleIndex::new(0).unwrap());
        assert_eq!(entry.distance(), NonNormDistance::Narrow(0));
    }

    #[test]
    fn nonnorm_a_gap_empties_its_row_and_is_never_a_neighbor() {
        let mut samples = ticks(&[1, 2, 3, 0, 5, 6, 7, 1, 2, 3]);
        samples[3] = None;
        let profile = naive_left_profile_nonnorm(&samples, 3);
        for row in [1, 2, 3] {
            assert_eq!(nonnorm_of(&profile, row).index(), SampleIndex::NONE);
        }
        let repeat = nonnorm_of(&profile, 7);
        assert_eq!(repeat.index(), SampleIndex::new(0).unwrap());
        assert_eq!(repeat.distance(), NonNormDistance::Narrow(0));
    }

    #[test]
    fn nonnorm_exclusion_boundary_and_half_width() {
        // Same series as the z-normalized boundary tests: row 2's only legal candidate is 0.
        let entry = nonnorm_of(
            &naive_left_profile_nonnorm(&ticks(&[1, 7, 7, 7, 7, 7]), 4),
            2,
        );
        assert_eq!(entry.index(), SampleIndex::new(0).unwrap());
        assert_eq!(entry.distance(), NonNormDistance::Narrow(36));

        let none = naive_left_profile_nonnorm(&ticks(&[1, 2, 3, 4, 5, 6, 7]), 5);
        assert_eq!(nonnorm_of(&none, 2).index(), SampleIndex::NONE);
    }

    #[test]
    fn nonnorm_tie_goes_to_the_smaller_index() {
        let samples = ticks(&[1, 2, 9, 1, 2, 9, 0, 0, 0, 1, 2, 9]);
        let entry = nonnorm_of(&naive_left_profile_nonnorm(&samples, 3), 9);
        assert_eq!(entry.index(), SampleIndex::new(0).unwrap());
        assert_eq!(entry.distance(), NonNormDistance::Narrow(0));
    }

    #[test]
    fn nonnorm_distance_is_on_raw_ticks_not_mean_centered() {
        // Window 0 is the row's shape shifted by 100: mean-centered it would match exactly
        // (distance 0) and win; on raw ticks it is far, and window 4 (distance 9) wins.
        let samples = ticks(&[101, 102, 103, 0, 0, 0, 1, 2, 3]);
        let entry = nonnorm_of(&naive_left_profile_nonnorm(&samples, 3), 6);
        assert_eq!(entry.index(), SampleIndex::new(4).unwrap());
        assert_eq!(entry.distance(), NonNormDistance::Narrow(9));
    }

    /// A fixed pseudo-random pattern of small ticks (xorshift64*).
    fn pattern(seed: u64, len: usize, spread: u64) -> Vec<i64> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                (state.wrapping_mul(0x2545_F491_4F6C_DD1D) % spread) as i64
            })
            .collect()
    }

    fn bits_of(profile: &[(u64, crate::entry::Entry)]) -> Vec<(u64, u64, Option<u64>)> {
        profile
            .iter()
            .map(|&(_, e)| {
                (
                    e.rho().to_bits(),
                    e.squared_distance().to_bits(),
                    e.index().get(),
                )
            })
            .collect()
    }

    #[test]
    fn znorm_is_exact_where_i128_sums_would_overflow() {
        // `N` and `V` are shift-invariant integers, so a pattern shifted to ticks near 2^62
        // (where m·Σx² reaches 2^130) must give bit-identical entries to the unshifted one.
        let m = 64;
        for seed in 1..6u64 {
            let small = pattern(seed, 140, 16);
            let shifted: Vec<i64> = small.iter().map(|&t| t + (1i64 << 62)).collect();
            let a = naive_left_profile(&ticks(&small), m);
            let b = naive_left_profile(&ticks(&shifted), m);
            assert_eq!(bits_of(&a), bits_of(&b));
            assert!(a.iter().filter(|(_, e)| e.index().get().is_some()).count() > 30);
        }
    }

    #[test]
    fn nonnorm_wide_distance_is_exact_past_i128() {
        // Pattern in {-1, 0, 1} times 2^62: every distance is the small pattern's times 2^124,
        // and m·(2^63)² overflows i128.
        let m = 8;
        for seed in 1..6u64 {
            let small: Vec<i64> = pattern(seed, 40, 3).iter().map(|&t| t - 1).collect();
            let scaled: Vec<i64> = small.iter().map(|&t| t << 62).collect();
            let a = naive_left_profile_nonnorm(&ticks(&small), m);
            let b = naive_left_profile_nonnorm(&ticks(&scaled), m);
            for ((_, small_entry), (_, wide_entry)) in a.iter().zip(&b) {
                assert_eq!(small_entry.index(), wide_entry.index());
                if small_entry.index().get().is_none() {
                    continue;
                }
                let (NonNormDistance::Narrow(d), NonNormDistance::Wide(w)) =
                    (small_entry.distance(), wide_entry.distance())
                else {
                    panic!("small pattern is narrow, scaled one is wide");
                };
                assert_eq!(w, I256::from(d) << 124);
            }
        }
    }

    #[test]
    fn a_tiny_negative_correlation_loses_to_exactly_zero() {
        // ρ̂ cannot separate these (both within 2ε of 0), so the exact sign decides. With
        // x = [A, −A, 1, −1], window 0 [B, B, 0, 1] has N = −4 and window 5 [B, B, 0, 0] has
        // N = 0; zero is nearer. Gaps keep every other window non-finite.
        let a = 1i64 << 29;
        let b = 1i64 << 29;
        let mut samples = ticks(&[b, b, 0, 1]);
        samples.push(None);
        samples.extend(ticks(&[b, b, 0, 0]));
        samples.push(None);
        samples.extend(ticks(&[a, -a, 1, -1]));
        let profile = naive_left_profile(&samples, 4);
        assert_eq!(entry_of(&profile, 10).index(), SampleIndex::new(5).unwrap());
    }
}
