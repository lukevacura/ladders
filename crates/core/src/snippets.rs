//! Snippets of a series, following STUMPY 1.14.1's `snippets` with every MPdist order decided
//! exactly.
//!
//! The series is cut into `⌊n / m⌋` non-overlapping segments of length `m` (a shorter tail is
//! dropped from the segments but stays in the series every profile runs over). Each segment's
//! profile holds, for every window `T[p..p + m]`, the MPdist between the segment and that window
//! at inner window length `s` (`s = m` unless shortened by `percentage` or `s`): the value at
//! sorted position `ceil(percentage * 2m)` (or the given rank) of the `2(m − s + 1)` nearest
//! distances, each of the segment's `s`-windows to its nearest `s`-window inside the window and
//! each of the window's `s`-windows to its nearest `s`-window of the segment. That is
//! [`crate::mpdist`] of the segment and the window, with its selection and its fallback for a
//! non-finite value.
//!
//! Greedy pick, `k` times: the segment whose profile, minimized against the running minimum of
//! the profiles already picked, has the smallest area (the first of equal areas) is taken, and
//! the running minimum is lowered by its profile. Afterwards each snippet's fraction is the share
//! of positions where its profile is at most the remaining minimum (which drops by one at every
//! position a snippet claims, so later snippets claim only what is left), and its regimes are the
//! runs of those positions. `k` may exceed the number of segments: a segment is then picked again.
//!
//! Differences from STUMPY: every distance order inside a profile is exact, and the nearest window
//! of a window is the exactly nearest, ties to the smaller start; areas, the running minimum,
//! fractions and the claim test stay f64 as STUMPY has them, areas summed pairwise as NumPy does.
//! The shorter tail is never padded: padded windows only ever fall outside every position's
//! range. Windows with a gap or no distance contribute infinity. An inner window length below 3
//! is an error: z-normalized windows that short are degenerate (length 1 is always constant) and
//! STUMPY spins for minutes there.
//!
//! The metric is a type parameter, `Measure`, chosen once at the entry point: [`snippets`] is
//! z-normalized and [`snippets_nonnorm`] is STUMPY's `aampdist_snippets`, the same procedure over
//! the non-normalized Euclidean distance (profiles in the units of the series). Everything but
//! the arithmetic of one pair is shared; the procedure differences STUMPY has are the segments
//! taken (the aamp version drops the last segment when `m` divides the series length) and the
//! non-normalized arithmetic itself, which runs in `u64` or `u128` by the series' whole tick range
//! (a series whose range needs more than 128 bits for a window has no distance at all).
//!
//! Memory beyond the profiles: with `s` far below `m` on tie-heavy data the per-row sliding
//! deques keep every equal value (the earliest column wins ties), so they hold up to `j²` entries
//! per segment per thread, `j = m − s + 1`; per-thread peaks add.

use alloc::collections::VecDeque;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

use crate::batch::nonnorm::TickSeries;
use crate::capacity::{CapacityError, reserved_vec};
use crate::mpdist::{Direction, Joined, Selection, selected};
use crate::nearest::Sharing;
use crate::nonnorm::{NonNormalized, RowWidth, routed_row_width};
use crate::ostinato::pairwise_sum;
use crate::pattern::{Measure, TickedSeries};
use crate::quantize::Scale;
use crate::stats::Series;
use crate::znorm::ZNorm;

/// How long the inner windows are, as a share of the segment length or as a length.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InnerLength {
    /// `ceil(share * m)`, `share` in `[0, 1]`.
    Share(f64),
    /// `s`, at most `m`.
    Length(usize),
}

/// STUMPY's default: the whole segment.
impl Default for InnerLength {
    fn default() -> Self {
        Self::Share(1.0)
    }
}

impl InnerLength {
    /// The inner window length for segments of `m`; an error for an empty one, a NaN share, a
    /// share outside `[0, 1]` or a length above `m` (STUMPY clamps the last two).
    pub fn resolve(self, m: usize) -> Result<NonZeroUsize, SnippetsError> {
        let s = match self {
            Self::Share(share) if share.is_nan() => return Err(SnippetsError::NotANumber),
            Self::Share(share) if !(0.0..=1.0).contains(&share) => {
                return Err(SnippetsError::ShareOutOfRange);
            }
            Self::Share(share) => libm::ceil(share * m as f64) as usize,
            Self::Length(s) if s > m => return Err(SnippetsError::InnerWindowAboveSegment),
            Self::Length(s) => s,
        };
        match s {
            0 => Err(SnippetsError::EmptyInnerWindow),
            1 | 2 => Err(SnippetsError::InnerWindowTooShort),
            s => Ok(NonZeroUsize::new(s).expect("s is at least 3")),
        }
    }
}

/// Which segments of the series are candidate snippets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Segments {
    /// All `⌊n / m⌋` of them, as STUMPY's `snippets`.
    All,
    /// STUMPY's `aampdist_snippets`, which pads the series to a multiple of `m` and takes one
    /// segment fewer than the padded series holds: the last whole segment is lost when `m`
    /// divides `n`.
    AllButPaddedLast,
}

impl Segments {
    fn count(self, len: usize, m: usize) -> usize {
        match self {
            Self::All => len / m,
            Self::AllButPaddedLast => len.div_ceil(m) - 1,
        }
    }
}

/// Why no snippets were computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SnippetsError {
    /// The series is shorter than `2m`: no two segments fit.
    SeriesTooShort,
    /// The inner window length came out as zero.
    EmptyInnerWindow,
    /// The inner window length came out as 1 or 2.
    InnerWindowTooShort,
    /// A share was NaN.
    NotANumber,
    /// A share was outside `[0, 1]`.
    ShareOutOfRange,
    /// The inner window length `s` was above the segment length `m`.
    InnerWindowAboveSegment,
    /// The MPdist rank was past the last of the `2(m - s + 1)` joined positions.
    RankPastLast,
    /// The `k` snippets and their profiles are too large to allocate.
    Capacity,
}

impl From<CapacityError> for SnippetsError {
    fn from(_: CapacityError) -> Self {
        Self::Capacity
    }
}

/// The `k` snippets, in STUMPY's shapes (matrices row-major).
#[derive(Clone, Debug, PartialEq)]
pub struct Snippets {
    /// Window length of each snippet and length of each profile.
    pub m: usize,
    pub profile_len: usize,
    /// The `k` snippets, `m` samples each.
    pub snippets: Vec<f64>,
    /// Where each snippet starts in the series.
    pub indices: Vec<usize>,
    /// The `k` MPdist profiles, `profile_len` values each.
    pub profiles: Vec<f64>,
    /// The share of positions each snippet claims.
    pub fractions: Vec<f64>,
    /// The area each snippet adds under the running minimum.
    pub areas: Vec<f64>,
    /// `(snippet, start, stop)`: the runs of positions each snippet claims, `stop` exclusive.
    pub regimes: Vec<[usize; 3]>,
}

impl Snippets {
    /// No snippets yet, with room for `k` of them, or [`SnippetsError::Capacity`] before any work
    /// when that cannot be allocated.
    fn reserved(m: usize, profile_len: usize, k: usize) -> Result<Self, SnippetsError> {
        let (wide_k, wide_m, wide_len) = (k as u128, m as u128, profile_len as u128);
        Ok(Self {
            m,
            profile_len,
            snippets: reserved_vec(wide_k * wide_m)?,
            indices: reserved_vec(wide_k)?,
            profiles: reserved_vec(wide_k * wide_len)?,
            fractions: reserved_vec(wide_k)?,
            areas: reserved_vec(wide_k)?,
            regimes: Vec::new(),
        })
    }
}

/// Everything one segment's profile needs.
struct Plan<'a, M: Measure> {
    ticked: &'a TickedSeries<M>,
    m: usize,
    s: usize,
    position: usize,
}

/// The MPdist profile of the segment starting at `start` against every window of the series.
///
/// Columns (the series' `s`-windows) are swept in order, each cross term carried down the
/// diagonal from the row above. A row's nearest column inside the sliding span of `j` columns is
/// kept in a monotone deque (equal distances keep the earlier column), a column's nearest row in
/// a ring of the last `j` columns (equal distances keep the earlier row); when a span is complete
/// the `2j` nearest distances are selected from.
fn segment_profile<M: Measure>(plan: &Plan<M>, start: usize) -> Vec<f64> {
    let Plan {
        ticked,
        m,
        s,
        position,
    } = *plan;
    let (ticks, windows) = (ticked.ticks(), ticked.windows());
    let j = m - s + 1;
    let columns = windows.len();
    let mut profile = Vec::with_capacity(columns + 1 - j);
    let mut cross: Vec<M::Diagonal> = Vec::with_capacity(j);
    let mut nearest_columns: Vec<VecDeque<(usize, M::Exact)>> =
        (0..j).map(|_| VecDeque::new()).collect();
    let mut nearest_rows: Vec<Option<M::Exact>> = alloc::vec![None; j];
    let mut joined: Vec<Joined<M>> = Vec::with_capacity(2 * j);

    for c in 0..columns {
        let column = &windows[c];
        let column_ticks = &ticks[c..c + s];
        if c == 0 {
            let first = M::step(column, 0, 0);
            cross.extend((0..j).map(|a| {
                M::start(
                    &ticks[start + a..start + a + s],
                    &M::step(&windows[start + a], 0, 0),
                    column_ticks,
                    &first,
                )
            }));
        } else {
            let leaving = ticks[c - 1];
            let entering = ticks[c + s - 1];
            let column_step = M::step(column, entering, leaving);
            for a in (1..j).rev() {
                let r = start + a;
                cross[a] = M::advance(
                    cross[a - 1],
                    &M::step(&windows[r], ticks[r + s - 1], ticks[r - 1]),
                    &column_step,
                );
            }
            cross[0] = M::start(
                &ticks[start..start + s],
                &M::step(&windows[start], 0, 0),
                column_ticks,
                &M::step(column, 0, 0),
            );
        }

        nearest_rows[c % j] = M::scan_column(
            &cross,
            &windows[start..start + j],
            column,
            s,
            c as u64,
            |a, distance| {
                let deque = &mut nearest_columns[a];
                while deque
                    .back()
                    .is_some_and(|(_, held)| M::compare(held, &distance, Sharing::Nothing).is_gt())
                {
                    deque.pop_back();
                }
                deque.push_back((c, distance));
            },
        );

        if c + 1 < j {
            continue;
        }
        let p = c + 1 - j;
        joined.clear();
        for (a, deque) in nearest_columns.iter_mut().enumerate() {
            while deque.front().is_some_and(|&(column, _)| column < p) {
                deque.pop_front();
            }
            if let Some(&(_, distance)) = deque.front() {
                joined.push(Joined {
                    distance,
                    direction: Direction::AToB,
                    window: a,
                });
            }
        }
        for t in 0..j {
            if let Some(distance) = nearest_rows[(p + t) % j] {
                joined.push(Joined {
                    distance,
                    direction: Direction::BToA,
                    window: t,
                });
            }
        }
        let picked = selected(&mut joined, position);
        profile.push(picked.map_or(f64::INFINITY, |exact| ticked.distance_of(&exact)));
    }
    profile
}

#[cfg(feature = "parallel")]
fn profiles_of<M: Measure>(
    plan: &Plan<M>,
    segments: usize,
    threads: NonZeroUsize,
) -> Vec<Vec<f64>> {
    crate::par::for_each_chunk(threads, segments, |i| segment_profile(plan, i * plan.m))
}

#[cfg(not(feature = "parallel"))]
fn profiles_of<M: Measure>(
    plan: &Plan<M>,
    segments: usize,
    _threads: NonZeroUsize,
) -> Vec<Vec<f64>> {
    (0..segments)
        .map(|i| segment_profile(plan, i * plan.m))
        .collect()
}

/// The area under `profile` lowered by `running`, which is left in `lowered`.
fn area_under_minimum(profile: &[f64], running: &[f64], lowered: &mut [f64]) -> f64 {
    for ((slot, &x), &q) in lowered.iter_mut().zip(profile).zip(running) {
        *slot = x.min(q);
    }
    pairwise_sum(lowered)
}

/// The runs of `true` in `mask`, as `[start, stop)`.
fn runs(mask: &[bool]) -> impl Iterator<Item = (usize, usize)> + '_ {
    let mut at = 0;
    core::iter::from_fn(move || {
        let start = at + mask[at..].iter().position(|&hit| hit)?;
        let stop = start
            + mask[start..]
                .iter()
                .position(|&hit| !hit)
                .unwrap_or(mask.len() - start);
        at = stop;
        Some((start, stop))
    })
}

/// The `k` snippets of `t` for segment length `m` (STUMPY's `snippets`). `inner` picks the inner
/// window length and `selection` the value of each segment-window MPdist; `t` is quantized at
/// `scale`. The same for every `threads`. `Err` when `t` holds fewer than `2m` samples, the inner
/// window comes out empty or a share is NaN. The windows of `t` the caller flags constant (STUMPY's
/// `mpdist_T_subseq_isconstant`) are those of the inner length [`InnerLength::resolve`] gives, one
/// per start in `t`.
pub fn snippets<'a>(
    t: impl Into<Series<'a>>,
    m: NonZeroUsize,
    k: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
    inner: InnerLength,
    selection: Selection,
) -> Result<Snippets, SnippetsError> {
    let t: Series<'a> = t.into();
    let shape = Shape::new(t.values(), m, inner, selection, Segments::All)?;
    let out = Snippets::reserved(m.get(), t.len() - m.get() + 1, k.get())?;
    let ticked = TickedSeries::<ZNorm>::new(t, shape.s, scale);
    let profiles = profiles_of(&shape.plan(&ticked), shape.segments, threads);
    Ok(pick(out, t.values(), m.get(), k.get(), &profiles))
}

/// [`snippets`] under the non-normalized metric, as STUMPY's `aampdist_snippets`: the profiles
/// are MPdist under the Euclidean distance on the raw values, in the units of `t`, which is
/// quantized at `scale`. STUMPY drops the last segment when `m` divides the length of `t`, and so
/// does this. A window with a gap, or one the series' tick range makes need more than 128 bits,
/// has no distance.
pub fn snippets_nonnorm(
    t: &[f64],
    m: NonZeroUsize,
    k: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
    inner: InnerLength,
    selection: Selection,
) -> Result<Snippets, SnippetsError> {
    let shape = Shape::new(t, m, inner, selection, Segments::AllButPaddedLast)?;
    let out = Snippets::reserved(m.get(), t.len() - m.get() + 1, k.get())?;
    let profiles = match tick_width(t, shape.s, scale, shape.s.get()) {
        RowWidth::Narrow => {
            let ticked = TickedSeries::<NonNormalized<u64>>::new(t, shape.s, scale);
            profiles_of(&shape.plan(&ticked), shape.segments, threads)
        }
        RowWidth::Wide => {
            let ticked = TickedSeries::<NonNormalized<u128>>::new(t, shape.s, scale);
            profiles_of(&shape.plan(&ticked), shape.segments, threads)
        }
        RowWidth::Unrepresentable => {
            alloc::vec![alloc::vec![f64::INFINITY; t.len() - m.get() + 1]; shape.segments]
        }
    };
    Ok(pick(out, t, m.get(), k.get(), &profiles))
}

/// The width the exact squared distance of two `s`-windows needs anywhere in `t`, by the range of
/// the ticks present (a gap's fill is not data).
fn tick_width(t: &[f64], m: NonZeroUsize, scale: Scale, s: usize) -> RowWidth {
    let (low, high) = TickSeries::build(t, m, scale)
        .present_range()
        .unwrap_or((i64::MAX, i64::MIN));
    routed_row_width(low, high, s)
}

/// What the metric-independent setup of a call decided.
struct Shape {
    m: usize,
    s: NonZeroUsize,
    position: usize,
    segments: usize,
}

impl Shape {
    fn new(
        t: &[f64],
        m: NonZeroUsize,
        inner: InnerLength,
        selection: Selection,
        segments: Segments,
    ) -> Result<Self, SnippetsError> {
        let m = m.get();
        if t.len() < 2 * m {
            return Err(SnippetsError::SeriesTooShort);
        }
        let s = inner.resolve(m)?;
        if !selection.fits(2 * (m - s.get() + 1)) {
            return Err(SnippetsError::RankPastLast);
        }
        Ok(Self {
            m,
            s,
            position: selection.position(2 * m),
            segments: segments.count(t.len(), m),
        })
    }

    fn plan<'a, M: Measure>(&self, ticked: &'a TickedSeries<M>) -> Plan<'a, M> {
        Plan {
            ticked,
            m: self.m,
            s: self.s.get(),
            position: self.position,
        }
    }
}

/// The greedy pick of `k` snippets from the segments' `profiles`, and their fractions and regimes.
fn pick(mut out: Snippets, t: &[f64], m: usize, k: usize, profiles: &[Vec<f64>]) -> Snippets {
    let profile_len = t.len() - m + 1;
    let mut running = alloc::vec![f64::INFINITY; profile_len];
    let mut lowered = alloc::vec![0.0; profile_len];
    for _ in 0..k {
        let mut best = (0, f64::INFINITY);
        for (segment, profile) in profiles.iter().enumerate() {
            let area = area_under_minimum(profile, &running, &mut lowered);
            if segment == 0 || area < best.1 {
                best = (segment, area);
            }
        }
        let (segment, area) = best;
        area_under_minimum(&profiles[segment], &running, &mut lowered);
        running.copy_from_slice(&lowered);
        out.snippets
            .extend_from_slice(&t[segment * m..(segment + 1) * m]);
        out.indices.push(segment * m);
        out.profiles.extend_from_slice(&profiles[segment]);
        out.areas.push(area);
    }

    let mut remaining = alloc::vec![f64::INFINITY; profile_len];
    for profile in out.profiles.chunks_exact(profile_len) {
        for (low, &x) in remaining.iter_mut().zip(profile) {
            *low = low.min(x);
        }
    }
    for (snippet, profile) in out.profiles.chunks_exact(profile_len).enumerate() {
        let mask: Vec<bool> = profile
            .iter()
            .zip(&remaining)
            .map(|(x, r)| x <= r)
            .collect();
        let claimed = mask.iter().filter(|&&hit| hit).count();
        out.fractions.push(claimed as f64 / profile_len as f64);
        for (low, _) in remaining.iter_mut().zip(&mask).filter(|(_, hit)| **hit) {
            *low -= 1.0;
        }
        out.regimes
            .extend(runs(&mask).map(|(start, stop)| [snippet, start, stop]));
    }
    out
}

#[cfg(test)]
mod tests;
