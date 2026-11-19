//! The one place a window's kind (valid / constant / non-finite) is turned into numbers a
//! kernel can read: `r_bias` encodes it, [`window_of`] decodes it back, and `encode` builds
//! the triple every per-window columns struct stores. The narrowed per-window columns
//! [`crate::kernel_neon::rank_row`] reads (one lag's own i32 offsets and `(r, bias)` pair), and
//! the small pure pieces that turn that same `(r, bias)` pair back into a ranking value for a
//! row that is not Valid, live here too, and so does `CrossRange`, the bound on every cross
//! term a kernel converts. Shared by any stream driving the narrow fast path
//! (bounded, growing, batch): a stream fills these fresh every row from its own exact, absolute
//! per-window data, so nothing here ever needs its own re-anchor logic.

use core::num::NonZeroUsize;

use crate::estimate::rank_estimate;
use crate::kernel_neon::Conversion;
use crate::stats::{Window, WindowKind, WindowStats};

/// One lag's `(r, bias)` pair: [`crate::kernel_neon::rank_row`]'s formula folds a branch on a
/// neighbor's kind into one multiply-add by reading `1/√V` and `0.0` for a Valid window, `0.0`
/// and `0.5` for a Constant one, or `0.0` and `−∞` for a non-finite one.
pub(crate) fn r_bias(window: &Window) -> (f64, f64) {
    match window.kind {
        WindowKind::Valid { inv_sqrt, .. } => (inv_sqrt, 0.0),
        WindowKind::Constant => (0.0, 0.5),
        WindowKind::NonFinite => (0.0, f64::NEG_INFINITY),
    }
}

/// `window`'s own `(r, bias)` pair, plus the exact variance numerator [`Nearest`]'s fallback
/// needs (meaningless, and never read, unless `window` is Valid): every per-window columns
/// struct that stores a window's kind stores exactly this triple, built here and nowhere else.
///
/// [`Nearest`]: crate::nearest::Nearest
pub(crate) fn encode(window: &Window) -> (f64, f64, u64) {
    let (r, bias) = r_bias(window);
    (r, bias, variance(window))
}

/// `window`'s exact variance numerator: `0` unless it is Valid.
pub(crate) fn variance(window: &Window) -> u64 {
    match window.kind {
        WindowKind::Valid {
            variance_numerator, ..
        } => variance_numerator,
        WindowKind::Constant | WindowKind::NonFinite => 0,
    }
}

/// `encode`'s exact inverse: the `Window` one lag's `(r, bias, variance)` triple (plus its
/// own wrapped sum) describes. `r` and `bias` only ever hold the three exact pairs `r_bias`
/// produces, so reading them back is exact, never a value that could land near one of them by
/// chance.
pub fn window_of(sum: i64, r: f64, bias: f64, variance: u64) -> Window {
    let kind = if bias == f64::NEG_INFINITY {
        WindowKind::NonFinite
    } else if r == 0.0 {
        WindowKind::Constant
    } else {
        WindowKind::Valid {
            variance_numerator: variance,
            inv_sqrt: r,
        }
    };
    Window { sum, kind }
}

/// Whether the `(r, bias)` pair [`r_bias`] encoded is a constant window's.
#[inline(always)]
pub(crate) fn is_constant(r: f64, bias: f64) -> bool {
    r == 0.0 && bias != f64::NEG_INFINITY
}

/// A constant row's own fixed ranking against a neighbor read off its `(r, bias)` pair: `1.0`
/// for a constant neighbor, `0.5` for a valid one, `bias` itself (`−∞`, excluded) for a
/// non-finite one. The narrow path's own copy of `row::rank_constant_row`'s mapping (that
/// function is private to `row`'s module), read from the narrowed encoding instead of a
/// `Window`'s kind byte: `bias` and `r` are only ever exactly `-inf`/`0.5`/`0.0`/positive, never
/// a computed value that could land near one of these by chance, so the comparisons below are
/// exact.
#[inline(always)]
pub(crate) fn constant_row_rank(r: f64, bias: f64) -> f64 {
    if bias == f64::NEG_INFINITY {
        bias
    } else if r == 0.0 {
        1.0
    } else {
        0.5
    }
}

/// A Valid row's own ranking against one lag, given that lag's exact centered cross term `n`:
/// [`rank_estimate`] plus `bias`, the same chain [`crate::kernel_neon::rank_row`]'s lane
/// formula uses, for the one lag (a brand new diagonal) a stream computes outside that
/// kernel's own batched loop. Sound for every neighbor kind, not only Valid: `bias` is `0.0`
/// there, and `r` is exactly `0.0` for a Constant or non-finite neighbor, so `rank_estimate`'s
/// product is exactly `0.0` (whatever sign) and `bias` alone decides — `0.5` or `−∞`.
#[inline(always)]
pub(crate) fn valid_row_rank(n: i64, row_ri: f64, r: f64, bias: f64) -> f64 {
    rank_estimate(n, row_ri, r) + bias
}

/// The largest variance numerator `V = m·Q − S²` over every window of a series' ticks, as the
/// hot pass reads them (gaps filled, outliers stood in), whatever each window's kind: it bounds
/// every centered cross term a row kernel computes between two of those windows.
///
/// Proof. The kernel converts, for each lane it ranks, `N = m·Σₖ aₖbₖ − Sₐ·S_b` of the row's
/// window `a` and the lane's neighbor `b`, both windows of these ticks, whether `b` is valid,
/// constant or non-finite (then `r = 0`), stood in or not. `N = m·Σₖ (aₖ − ā)(bₖ − b̄)`, so by
/// Cauchy–Schwarz `N² ≤ Vₐ·V_b ≤ widest²`, and `N` is an integer: `|N| ≤ widest`, reached when
/// `b` repeats `a`. Every anchor and re-anchoring computes this same `N` (it is unchanged by a
/// whole-tick shift of either window), and its wrapping `i64` arithmetic gives the true `N`
/// whenever `|N| < 2⁶³`, which a bound below that ensures. For two series (a join) the bound is
/// the larger of theirs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CrossRange {
    /// `u64::MAX` when some window's `V` is `2⁶³` or more.
    widest: u64,
}

impl CrossRange {
    /// A bound not taken yet: it fits nothing.
    pub(crate) const UNKNOWN: Self = Self { widest: u64::MAX };

    /// The fast `i64 → f64` conversion is exact on `−2⁵¹ ≤ N < 2⁵¹`.
    const FAST_CONVERSION_LIMIT: u64 = 1 << 51;

    /// The bound over every length-`m` window of `ticks`.
    pub(crate) fn over(ticks: &[i64], m: NonZeroUsize) -> Self {
        let mut stats = WindowStats::new(m);
        let widest = ticks
            .iter()
            .filter_map(|&tick| {
                stats.push(tick, true)?;
                Some(stats.numerator().unwrap_or(u64::MAX))
            })
            .fold(0, u64::max);
        Self { widest }
    }

    /// The bound over the cross terms between this series' windows and `other`'s.
    pub(crate) fn joint(self, other: Self) -> Self {
        Self {
            widest: self.widest.max(other.widest),
        }
    }

    /// Whether every cross term is within the fast conversion's range: `widest < 2⁵¹` gives
    /// `|N| ≤ 2⁵¹ − 1`.
    pub(crate) fn fits_fast_conversion(self) -> bool {
        self.widest < Self::FAST_CONVERSION_LIMIT
    }

    /// The conversion a row kernel may take on these cross terms: fast exactly when they fit,
    /// the full-range one otherwise.
    pub(crate) fn conversion(self) -> Conversion {
        if crate::route::fast_conversion(self.fits_fast_conversion()) {
            Conversion::FAST
        } else {
            Conversion::FULL_RANGE
        }
    }
}

#[cfg(test)]
mod tests;
