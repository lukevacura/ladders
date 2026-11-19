//! Narrowed covariance arithmetic: the same centered cross term and variance numerator as
//! [`crate::znorm`], computed from i32 offsets to an anchor instead of full i64 ticks, so the
//! per-lag products widen from i32 instead of needing a 64-bit multiply. A stream keeps an
//! anchor near its current ticks and re-anchors when a tick would otherwise fall outside
//! [`Narrow::fits`]'s bound.

use alloc::vec::Vec;
use core::num::NonZeroUsize;

use crate::route;

/// The tick every offset in a narrowed diagonal is measured from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Anchor {
    pub tick: i64,
}

impl Anchor {
    /// An anchor centered on `ticks`' own range, or `None` when that range is wider than any one
    /// anchor covers (twice [`Narrow::fits`]'s bound) or `ticks` is empty. The same centering a
    /// stream uses, so a batch run narrows a whole series, a tile's rows or a tile's columns the
    /// one way.
    pub(crate) fn covering(ticks: &[i64], m: NonZeroUsize) -> Option<Anchor> {
        let &lo = ticks.iter().min()?;
        let &hi = ticks.iter().max()?;
        Self::spanning(lo, hi, m, 0)
    }

    /// Whether one anchor can cover `ticks` by the natural rule alone: no route is forced and none
    /// is counted, so it can ask where [`Self::covering`] must not.
    pub(crate) fn can_cover(ticks: &[i64], m: NonZeroUsize) -> bool {
        let (low, high) = ticks
            .iter()
            .fold((i64::MAX, i64::MIN), |(low, high), &tick| {
                (low.min(tick), high.max(tick))
            });
        low > high || high - low <= 2 * (i64::from(i32::MAX) / m.get() as i64)
    }

    /// An anchor centered on `lo..=hi`, or `None` when that span is wider than any one anchor
    /// covers (twice [`Narrow::fits`]'s bound). `salt` only varies where a forced route places it.
    pub(crate) fn spanning(lo: i64, hi: i64, m: NonZeroUsize, salt: u64) -> Option<Anchor> {
        let bound = i64::from(i32::MAX) / m.get() as i64;
        if route::wide(hi - lo > 2 * bound) {
            return None;
        }
        Self::fitting(lo, hi, m, salt)
    }

    /// [`Self::spanning`] for a span the caller has already decided to narrow, whatever a forced
    /// route says: `None` only when `lo..=hi` is wider than one anchor covers.
    pub(crate) fn fitting(lo: i64, hi: i64, m: NonZeroUsize, salt: u64) -> Option<Anchor> {
        let bound = i64::from(i32::MAX) / m.get() as i64;
        if hi - lo > 2 * bound {
            return None;
        }
        let centered = Anchor {
            tick: lo + (hi - lo) / 2,
        };
        Some(route::place(centered, lo, hi, m, salt))
    }

    /// [`Self::spanning`], but only when `lo..=hi` is no wider than one bound, so the anchor keeps
    /// half the span it can cover free: a stream returning to the narrow path takes this one, so
    /// a span just inside the limit does not bounce it straight back out.
    pub(crate) fn roomy_spanning(lo: i64, hi: i64, m: NonZeroUsize, salt: u64) -> Option<Anchor> {
        let bound = i64::from(i32::MAX) / m.get() as i64;
        if hi - lo > bound {
            return None;
        }
        Self::spanning(lo, hi, m, salt)
    }

    /// `tick − self.tick` as an i32, or `None` when that offset falls outside
    /// [`Narrow::fits`]'s bound for `m`.
    pub fn offset(&self, tick: i64, m: NonZeroUsize) -> Option<i32> {
        let offset = tick - self.tick;
        if Narrow::fits(offset, m) {
            i32::try_from(offset).ok()
        } else {
            None
        }
    }
}

/// The bound an offset must respect for `m·offset²`, summed over `m` lags, to stay within i64:
/// every offset at most `⌊(2³¹−1)/m⌋` keeps both a single m-scaled offset and the sum of `m`
/// raw offsets within i32, and `m·QT` within i64 however the m lags combine.
pub struct Narrow;

impl Narrow {
    /// Whether `offset` is within `±⌊(2³¹−1)/m⌋`, the bound above.
    pub fn fits(offset: i64, m: NonZeroUsize) -> bool {
        offset.unsigned_abs() <= i32::MAX as u64 / m.get() as u64
    }
}

/// `tick`'s offset from `anchor`, trusted to be in bound rather than re-checked: a caller
/// checks every stored or live tick exactly once, the one time it arrives ([`Anchor::offset`]'s
/// own [`Narrow::fits`] check), not again here on every later row some diagonal reads it back
/// on — re-deriving that check on every one of those reads would put it back on the hot
/// per-row path it does not belong on. `debug_assert` still catches a violation in
/// tests and debug builds; release pays nothing for it.
pub(crate) fn tick_offset(anchor: Anchor, m: NonZeroUsize, tick: i64) -> i32 {
    let offset = tick - anchor.tick;
    debug_assert!(
        Narrow::fits(offset, m),
        "every stored or live tick fits the anchor (checked once per tick)"
    );
    offset as i32
}

/// `ticks` narrowed to `anchor` into `out` (replacing its contents, keeping its capacity); every
/// tick must be within `anchor`'s reach, as [`Anchor::covering`] guarantees for its own input.
pub(crate) fn narrow_ticks_into(
    anchor: Anchor,
    m: NonZeroUsize,
    ticks: &[i64],
    out: &mut Vec<i32>,
) {
    out.clear();
    out.extend(ticks.iter().map(|&tick| tick_offset(anchor, m, tick)));
}

/// Window `sums` narrowed to `anchor` into `out`, like [`narrow_ticks_into`]; every window's
/// ticks must be within `anchor`'s reach.
pub(crate) fn narrow_sums_into(anchor: Anchor, m: NonZeroUsize, sums: &[i64], out: &mut Vec<i32>) {
    out.clear();
    out.extend(sums.iter().map(|&sum| sum_offset(anchor, m, sum)));
}

/// A window's own sum, re-expressed as an offset from `anchor`: `S − m·anchor`. In bound
/// whenever every one of that window's own `m` ticks is (`Narrow`'s own argument: the sum of
/// `m` in-bound offsets still fits i32), so this never needs its own `fits` check; trusted the
/// same way [`tick_offset`] is, for the same reason.
pub(crate) fn sum_offset(anchor: Anchor, m: NonZeroUsize, sum: i64) -> i32 {
    // Wrapping, because `m·anchor` alone can pass i64 at large `m` and large ticks while the
    // true `S − m·anchor` is in i32: every step is a ring operation mod 2⁶⁴, so the wrapped
    // result is exactly the true one.
    let raw = sum.wrapping_sub((m.get() as i64).wrapping_mul(anchor.tick));
    debug_assert!(
        i32::try_from(raw).is_ok(),
        "a window's own sum offset fits i32 whenever each of its ticks does"
    );
    raw as i32
}

/// This row's own entering or leaving tick, offset from `anchor` and scaled by `m`: the
/// `me`/`ml` [`advance`] and [`crate::kernel_neon::RowScalars`] both take, always the
/// m-scaled side. Trusted the same way [`tick_offset`] is.
pub(crate) fn row_scaled_offset(anchor: Anchor, m: NonZeroUsize, tick: i64) -> i32 {
    let raw = tick_offset(anchor, m, tick);
    let scaled = i64::from(raw) * m.get() as i64;
    debug_assert!(
        i32::try_from(scaled).is_ok(),
        "an in-bound offset scales by m within i32 (Narrow's own bound)"
    );
    scaled as i32
}

/// `m·QT`, advanced one lag: `me`/`ml` are the entering and leaving offset of one window,
/// already scaled by `m`; `te`/`tl` are the other window's entering and leaving offset,
/// unscaled. Exactly one scale by `m` is needed per product, so folding it into one side up
/// front leaves this step with none. Wrapping i64; exact whenever the true `m·QT` at the next
/// lag fits i64, which holds for any anchor whose offsets are all within [`Narrow::fits`].
pub fn advance(qt_m: i64, me: i32, ml: i32, te: i32, tl: i32) -> i64 {
    qt_m.wrapping_sub(i64::from(ml).wrapping_mul(i64::from(tl)))
        .wrapping_add(i64::from(me).wrapping_mul(i64::from(te)))
}

/// `N = m·QT − Sᵢ·Sⱼ`, from the m-scaled diagonal state and the two windows' own (unscaled)
/// offset sums. Wrapping i64, by the same exact-recovery argument as [`advance`]: shift
/// invariance makes this equal the i64 path's `N` for any in-bound anchor, variance included
/// when `si` and `sj` are the same window's own sum.
pub fn cross(qt_m: i64, si: i32, sj: i32) -> i64 {
    qt_m.wrapping_sub(i64::from(si).wrapping_mul(i64::from(sj)))
}

/// Re-anchors one diagonal's state from its current anchor to `delta` past it: the exact maps
/// `S → S − m·delta` for each window's own offset sum, and
/// `m·QT → m·QT − (m·delta)·(Sₐ + S_b) + (m·delta)²`. Wrapping i64; exact whenever the true
/// state at both anchors fits i64. A stream applies this to every diagonal it keeps live.
pub fn reanchor(delta: i64, m: NonZeroUsize, qt_m: i64, sa: i64, sb: i64) -> (i64, i64, i64) {
    let m_delta = (m.get() as i64).wrapping_mul(delta);
    let sa_next = sa.wrapping_sub(m_delta);
    let sb_next = sb.wrapping_sub(m_delta);
    let qt_m_next = qt_m
        .wrapping_sub(m_delta.wrapping_mul(sa.wrapping_add(sb)))
        .wrapping_add(m_delta.wrapping_mul(m_delta));
    (qt_m_next, sa_next, sb_next)
}

/// One diagonal's `m·QT` with its row side's anchor moved `delta_row` ticks and its column side's
/// `delta_col` ticks, from the sums of the pair's two windows over offsets from the old anchors:
/// `m·QT → m·QT − m·delta_col·Sᵣ − m·delta_row·S_c + m²·delta_row·delta_col`. The two sides move
/// independently because `N` and `V` are unchanged by a whole-tick shift of either window.
/// Wrapping i64, exact whenever the true `m·QT` at the new anchors fits i64, which holds when
/// both windows' ticks are within [`Narrow::fits`]'s bound of their new anchors.
pub(crate) fn reanchor_pair(
    m: NonZeroUsize,
    delta_row: i64,
    delta_col: i64,
    qt_m: i64,
    row_sum: i64,
    col_sum: i64,
) -> i64 {
    let m = m.get() as i64;
    qt_m.wrapping_sub(m.wrapping_mul(delta_col).wrapping_mul(row_sum))
        .wrapping_sub(m.wrapping_mul(delta_row).wrapping_mul(col_sum))
        .wrapping_add(
            m.wrapping_mul(m)
                .wrapping_mul(delta_row)
                .wrapping_mul(delta_col),
        )
}

/// The absolute `QT = Σ aₖ·bₖ` (wrapping) of two windows whose sums are `sum_a` and `sum_b`, from
/// their `m·QT` over offsets from `anchor`: `QT = QT_off + A·(Sₐ + S_b) − m·A²`. The `m·QT`
/// divides by `m` exactly because its true value, not only its residue, is in i64.
pub(crate) fn absolute_qt(
    anchor: Anchor,
    m: NonZeroUsize,
    qt_m: i64,
    sum_a: i64,
    sum_b: i64,
) -> i64 {
    let a = anchor.tick;
    (qt_m / m.get() as i64)
        .wrapping_add(a.wrapping_mul(sum_a.wrapping_add(sum_b)))
        .wrapping_sub((m.get() as i64).wrapping_mul(a.wrapping_mul(a)))
}

/// [`absolute_qt`]'s inverse: the `m·QT` over offsets from `anchor` of an absolute `QT`.
pub(crate) fn narrow_qt_m(anchor: Anchor, m: NonZeroUsize, qt: i64, sum_a: i64, sum_b: i64) -> i64 {
    let a = anchor.tick;
    qt.wrapping_sub(a.wrapping_mul(sum_a.wrapping_add(sum_b)))
        .wrapping_add((m.get() as i64).wrapping_mul(a.wrapping_mul(a)))
        .wrapping_mul(m.get() as i64)
}

/// Which arithmetic produced a row: narrowed to an anchor's i32 offsets, or the wide i64 path
/// a stream falls back to when a tick would not fit any reachable anchor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Path {
    Narrow,
    Wide,
}

/// How many updates a stream ran on each route and how often it changed route. Informational
/// only: no entry depends on it, a checkpoint does not hold it, and a restored stream starts
/// again from zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RouteCounts {
    /// Updates (samples fed) that ran on the narrow arithmetic.
    pub narrow_updates: u64,
    /// Updates that ran on the exact i64 arithmetic.
    pub wide_updates: u64,
    /// Times the narrow anchor moved to cover a wider span.
    pub reanchors: u64,
    /// Times the stream switched from narrow to i64 arithmetic.
    pub switches_to_wide: u64,
    /// Times a bounded stream on the i64 arithmetic returned to the narrow one.
    pub switches_to_narrow: u64,
    /// Ticks withheld from the hot arithmetic, their windows ranked by the exact cold route.
    pub withheld_ticks: u64,
    /// Rows that ranked at least one pair by the exact cold route.
    pub cold_rows: u64,
}

impl RouteCounts {
    pub(crate) fn count_update(&mut self, path: Path) {
        match path {
            Path::Narrow => self.narrow_updates += 1,
            Path::Wide => self.wide_updates += 1,
        }
    }
}

#[cfg(test)]
mod tests;
