//! MPdist, following STUMPY 1.14.1's `mpdist` with the order between distances decided exactly.
//!
//! The joined profile `P_ABBA` is the nearest `b` window of every `a` window (the AB-join)
//! followed by the nearest `a` window of every `b` window (the BA-join); a window with no
//! distance contributes infinity. The reported distance is the value at sorted position `k` of
//! it, with `k` given directly or as `ceil(percentage * (n_a + n_b))` over the two series' sample
//! counts. STUMPY clamps `percentage` to `[0, 1]` and `k` to the last position; here a percentage
//! outside `[0, 1]` is no [`Selection`] at all and a rank past the last position gives no distance,
//! so neither is changed silently. A percentage inside `[0, 1]` whose position lies past the last
//! one selects the last, as in STUMPY: that is how STUMPY defines the position. When the
//! value at `k` is infinite, STUMPY reports the largest finite one among the `k` smallest, or
//! infinity when there is none; so does this.
//!
//! Difference from STUMPY: the sort is the exact order of distances rather than of their
//! rounded values, and exactly equal distances are taken AB-join first, then by window.
//!
//! The procedure is written once over `M: Measure`: [`mpdist`] is it under the z-normalized
//! distance, each series at its own scale, and [`aampdist`] (STUMPY's `aampdist`) under the
//! Euclidean distance on the raw values, both series at one shared scale, so the distances in
//! `P_ABBA` compare in one unit. STUMPY's two procedures are the same one.

use alloc::vec::Vec;
use core::num::NonZeroUsize;

use crate::nearest::Sharing;
use crate::nonnorm::NonNormalized;
use crate::pattern::{Measure, TickedSeries};
use crate::quantize::Scale;
use crate::stats::Series;
use crate::znorm::ZNorm;

/// Which value of the joined profile MPdist reports: a position, or a share of the samples.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Selection(Choice);

#[derive(Clone, Copy, Debug, PartialEq)]
enum Choice {
    Percentage(f64),
    Rank(usize),
}

impl Selection {
    /// `ceil(percentage * (n_a + n_b))`; `None` for NaN or a share outside `[0, 1]`.
    pub fn percentage(percentage: f64) -> Option<Self> {
        (0.0..=1.0)
            .contains(&percentage)
            .then_some(Self(Choice::Percentage(percentage)))
    }

    /// The value at sorted position `k`.
    pub const fn rank(k: usize) -> Self {
        Self(Choice::Rank(k))
    }

    /// Whether a list of `positions` values holds the position this selects: a rank must lie below
    /// it, and a share always does (its position is at most the last one).
    pub fn fits(self, positions: usize) -> bool {
        match self.0 {
            Choice::Percentage(_) => true,
            Choice::Rank(k) => k < positions,
        }
    }

    /// The sorted position of a share of `samples`, or the rank.
    pub(crate) fn position(self, samples: usize) -> usize {
        match self.0 {
            Choice::Percentage(share) => libm::ceil(share * samples as f64) as usize,
            Choice::Rank(k) => k,
        }
    }
}

/// STUMPY's default, five percent.
impl Default for Selection {
    fn default() -> Self {
        Self(Choice::Percentage(0.05))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Direction {
    AToB,
    BToA,
}

pub(crate) struct Joined<M: Measure = ZNorm> {
    pub(crate) distance: M::Exact,
    pub(crate) direction: Direction,
    pub(crate) window: usize,
}

impl<M: Measure> Clone for Joined<M> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<M: Measure> Copy for Joined<M> {}

impl<M: Measure> Joined<M> {
    fn order(&self, other: &Self) -> core::cmp::Ordering {
        M::compare(&self.distance, &other.distance, Sharing::Nothing)
            .then(self.direction.cmp(&other.direction))
            .then(self.window.cmp(&other.window))
    }
}

fn collect_join<M: Measure>(
    join: &[M::Hit],
    rows: &TickedSeries<M>,
    neighbors: &TickedSeries<M>,
    direction: Direction,
    into: &mut Vec<Joined<M>>,
) {
    for (window, hit) in join.iter().enumerate() {
        let Some(neighbor) = M::index(hit) else {
            continue;
        };
        let distance = rows
            .exact_across(window, neighbors, neighbor as usize)
            .expect("a joined pair has windows with distances");
        into.push(Joined {
            distance,
            direction,
            window,
        });
    }
}

/// The MPdist between `a` and `b` at window length `m`, each series quantized at its own scale;
/// `None` when either has no window (fewer than `m` samples) or a rank is past the last of the
/// `windows(a) + windows(b)` joined positions ([`Selection::fits`]). The same for every `threads`.
pub fn mpdist<'a, 'b>(
    a: impl Into<Series<'a>>,
    b: impl Into<Series<'b>>,
    m: NonZeroUsize,
    scale_a: Scale,
    scale_b: Scale,
    threads: NonZeroUsize,
    selection: Selection,
) -> Option<f64> {
    distance_in::<ZNorm>(
        a.into(),
        b.into(),
        m,
        (scale_a, scale_b),
        threads,
        selection,
    )
}

/// [`mpdist`] under the Euclidean distance on the raw values, as STUMPY's `aampdist`: both series
/// are quantized at `scale`, and the distance is in the caller's units. A window whose distance
/// needs more than 128 bits has none, as a window with a gap does.
pub fn aampdist(
    a: &[f64],
    b: &[f64],
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
    selection: Selection,
) -> Option<f64> {
    distance_in::<NonNormalized>(a.into(), b.into(), m, (scale, scale), threads, selection)
}

fn distance_in<M: Measure>(
    a: Series<'_>,
    b: Series<'_>,
    m: NonZeroUsize,
    (scale_a, scale_b): (Scale, Scale),
    threads: NonZeroUsize,
    selection: Selection,
) -> Option<f64> {
    let windows = |len: usize| (len + 1).saturating_sub(m.get());
    let (windows_a, windows_b) = (windows(a.len()), windows(b.len()));
    if windows_a == 0 || windows_b == 0 || !selection.fits(windows_a + windows_b) {
        return None;
    }
    let (ticked_a, ticked_b) = (
        TickedSeries::<M>::new(a, m, scale_a),
        TickedSeries::<M>::new(b, m, scale_b),
    );
    let mut joined = Vec::with_capacity(windows_a + windows_b);
    let join = M::ab_join(a, b, m, scale_a, scale_b, threads);
    collect_join(&join, &ticked_a, &ticked_b, Direction::AToB, &mut joined);
    let join = M::ab_join(b, a, m, scale_b, scale_a, threads);
    collect_join(&join, &ticked_b, &ticked_a, Direction::BToA, &mut joined);

    let picked = selected(&mut joined, selection.position(a.len() + b.len()));
    Some(picked.map_or(f64::INFINITY, |exact| ticked_a.distance_of(&exact)))
}

/// The distance at sorted `position` of `joined`, in whatever order it comes; when `position` is
/// at or past its last entry, the largest. `None` for an empty `joined`.
pub(crate) fn selected<M: Measure>(joined: &mut [Joined<M>], position: usize) -> Option<M::Exact> {
    let picked = if position < joined.len() {
        Some(*joined.select_nth_unstable_by(position, Joined::order).1)
    } else {
        joined.iter().copied().max_by(Joined::order)
    };
    picked.map(|p| p.distance)
}
