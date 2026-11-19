//! The PAA upper bound: an upper-bound producer (one-sided producers, ladder amendment) that bounds
//! every exact correlation in a block of pairs from each window's piecewise means over its own
//! segments, so the survivor filter can skip a block whose bound is below every floor in it.
//!
//! **Segments.** A window of `m` ticks is cut into `k` segments, segment `s` the offsets
//! `⌊s·m/k⌋ .. ⌊(s+1)·m/k⌋`: widths `w_s` of `⌊m/k⌋` or `⌈m/k⌉`, each at least one when `k ≤ m`.
//! A remainder is spread this way, never dropped. The indicator of segment `s`, divided by `√w_s`,
//! is a unit vector, and the `k` of them have disjoint supports, so they are orthonormal.
//!
//! **Coefficients.** With `S` the window's sum, `V = m·Q − S²` its variance numerator and `A_s` the
//! sum of segment `s`, the z-normalized window `z_t = (m·x_t − S)/√V` has `‖z‖² = m`, and its
//! correlation with another window is `ρ = ⟨zᵢ, zⱼ⟩/m`, so `‖zᵢ − zⱼ‖² = 2m(1 − ρ)`. Its coefficient
//! on segment `s` is `e_s = ⟨z, 1_s/√w_s⟩ = N_s/(√w_s·√V)` with the exact integer numerator
//! `N_s = m·A_s − w_s·S`. By Bessel's inequality `‖zᵢ − zⱼ‖² ≥ Σ_s (e_s(i) − e_s(j))²`.
//!
//! **Block value.** Over a block, the rows' coefficients lie in a box `[lo_r, hi_r]` and the
//! columns' in `[lo_c, hi_c]`, per segment. For every pair, `|e_s(i) − e_s(j)|` is at least the gap
//! `g_s = max(lo_c − hi_r, lo_r − hi_c, 0)` between the two intervals, so
//! `ρ ≤ U = 1 − Σ_s g_s²/(2m)`. [`Coefficients::value`] evaluates it with margins, proved there.
//!
//! **What it never bounds.** Only valid windows carry coefficients; a constant or non-finite
//! window's are NaN and never enter a box. A block holding a constant window is visited by the
//! survivor filter whatever its value (the constant-row route owns it), and a non-finite window
//! ranks nothing. A series whose numerators the wrapping arithmetic cannot hold (`m·w > 2⁶³`), or
//! with `m` below the segment count, gets no coefficients, and a filter forced to this producer
//! there is refused.

use alloc::vec::Vec;
use core::ops::Range;

use crate::estimate::{Producer, Sides};

/// `2⁻⁴⁴`: the slack the block value declares, at least twice its proved evaluation error
/// (`2⁻⁴⁷` for `k ≤ 16`, `2⁻⁴⁵` for `k ≤ 64`; see [`Coefficients::value`]).
pub(crate) const DELTA: f64 = 1.0 / (1u64 << 44) as f64;

/// The PAA producer over `K` segments: its value is [`Coefficients::value`] over a block's row
/// and column boxes, and `value + DELTA` is at least every exact correlation in the block.
pub(crate) struct Paa<const K: usize>;

impl<const K: usize> Producer for Paa<K> {
    const DELTA: f64 = DELTA;
    const SIDES: Sides = Sides::Above;
}

/// How many segments a window is cut into: chosen once per column set from `m`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Segments {
    Four,
    Eight,
    Sixteen,
}

impl Segments {
    /// The count for windows of `m` ticks: 4 up to `m` 256, 8 below 512 and 16 from 512. On the
    /// engine's block of 8 rows by 64 slots, at n 65536 on grid, float, messy and BTC series,
    /// 16 segments skip as many pairs as 8 at `m` 512 on the synthetic series and 2 to 6 points
    /// more on BTC, and 1 to 3 points more at `m` 1024; 4 skip 3 to 7 points fewer than either at
    /// `m` 512.
    pub(crate) fn for_window(m: usize) -> Self {
        if m <= 256 {
            Self::Four
        } else if m < 512 {
            Self::Eight
        } else {
            Self::Sixteen
        }
    }

    pub(crate) const fn count(self) -> usize {
        match self {
            Self::Four => 4,
            Self::Eight => 8,
            Self::Sixteen => 16,
        }
    }
}

/// Consecutive windows whose boxes are kept together: a block's column box is the union of the
/// group boxes its columns fall in, a superset of its columns, which only loosens the bound.
const COLUMN_GROUP: usize = 8;

/// Consecutive groups whose union box is kept too ([`Coefficients::columns_span`]): a part of the
/// engine's block (8 rows by 64 slots, 71 columns) spans 9 or 10 groups, so it reads two of these
/// unions, which overlap, instead of joining ten group boxes.
const UNION_GROUPS: usize = 8;

/// Every window's PAA coefficients over one column set, built once from the column builder's ticks,
/// window sums and reciprocal roots, each group of [`COLUMN_GROUP`] windows' box and each run of
/// [`UNION_GROUPS`] groups' union box, and a running count of the constant windows.
pub(crate) struct Coefficients {
    segments: Segments,
    /// Window `i`'s coefficients at `i·k..(i+1)·k`, rounded to the nearest `f32`; NaN for a window
    /// that is not valid.
    values: Vec<f32>,
    /// Group `g`'s box at `g·2k..(g+1)·2k`: its `k` lows, then its `k` negated highs, over its
    /// valid windows (`+∞` both when it has none). Never NaN.
    groups: Vec<f32>,
    /// The union of groups `g..g + UNION_GROUPS` (fewer at the end), laid out as `groups`.
    unions: Vec<f32>,
    /// `constants[i]` is how many of windows `0..i` are constant (`columns::is_constant`).
    constants: Vec<u32>,
    /// The gap margin, `2⁻²²·√m` rounded as [`Coefficients::value`] proves it.
    margin: f64,
    /// `1/(2m)`, rounded.
    half_inv_m: f64,
}

/// One side's box over a block: per segment, the lowest coefficient and the negated highest, so
/// that the union of two boxes is one minimum over both halves (`+∞` both when empty).
#[derive(Clone, Copy)]
pub(crate) struct Span<const K: usize> {
    pub(crate) lo: [f32; K],
    pub(crate) neg_hi: [f32; K],
}

/// `a` if it is below `b`, else `b`: the minimum, as one compare and select (`minps` on x86, the
/// same compare on NEON), never reading a NaN (no box holds one).
#[inline(always)]
fn min(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

impl<const K: usize> Span<K> {
    const EMPTY: Self = Self {
        lo: [f32::INFINITY; K],
        neg_hi: [f32::INFINITY; K],
    };

    /// Widens the box by `values`' `K` coefficients; a NaN (a window that is not valid) widens
    /// nothing.
    #[inline(always)]
    fn take(&mut self, values: &[f32]) {
        for ((lo, neg_hi), &v) in self.lo.iter_mut().zip(&mut self.neg_hi).zip(&values[..K]) {
            *lo = if v < *lo { v } else { *lo };
            *neg_hi = if -v < *neg_hi { -v } else { *neg_hi };
        }
    }

    /// Widens the box by a stored box (`K` lows, then `K` negated highs): one minimum per lane.
    #[inline(always)]
    fn join(&mut self, other: &[f32]) {
        let (lo, neg_hi) = other[..2 * K].split_at(K);
        for (a, &b) in self.lo.iter_mut().zip(lo) {
            *a = min(*a, b);
        }
        for (a, &b) in self.neg_hi.iter_mut().zip(neg_hi) {
            *a = min(*a, b);
        }
    }

    /// The box's highest coefficient on segment `s`.
    #[cfg(test)]
    pub(crate) fn hi(&self, s: usize) -> f32 {
        -self.neg_hi[s]
    }
}

/// The coefficients a block's rows and columns are read from: the same column set in a
/// self-join, each side's own in an AB-join. Both are built at the same `m`.
#[derive(Clone, Copy)]
pub(crate) struct PaaSides<'a> {
    pub(crate) rows: &'a Coefficients,
    pub(crate) cols: &'a Coefficients,
}

impl<'a> PaaSides<'a> {
    /// The rows and columns of one column set.
    pub(crate) fn of(both: &'a Coefficients) -> Self {
        Self {
            rows: both,
            cols: both,
        }
    }

    /// Each side's own, when both have coefficients.
    pub(crate) fn join(
        rows: Option<&'a Coefficients>,
        cols: Option<&'a Coefficients>,
    ) -> Option<Self> {
        Some(Self {
            rows: rows?,
            cols: cols?,
        })
    }

    pub(crate) fn segments(self) -> Segments {
        self.cols.segments
    }
}

/// The union boxes of every run of `width` consecutive boxes of `boxes` (`2k` values each), fewer
/// at the end, laid out as `boxes`; `None` when there is no room.
fn unions(boxes: &[f32], k: usize, width: usize) -> Option<Vec<f32>> {
    let mut out = Vec::new();
    out.try_reserve_exact(boxes.len()).ok()?;
    let count = boxes.len() / (2 * k);
    for g in 0..count {
        let mut union = alloc::vec![f32::INFINITY; 2 * k];
        for other in boxes[g * 2 * k..(g + width).min(count) * 2 * k].chunks_exact(2 * k) {
            for (a, &b) in union.iter_mut().zip(other) {
                *a = min(*a, b);
            }
        }
        out.extend_from_slice(&union);
    }
    Some(out)
}

impl Coefficients {
    /// The coefficients of every window of `ticks` (the ticks the hot pass reads: gaps filled,
    /// outliers stood in), from the column builder's own window sums `sum` (exact modulo `2⁶⁴`)
    /// and kinds `(r, bias)`, at `segments` segments; `None` where the proof does not cover the
    /// series: `m` below the segment count, `m·w > 2⁶³` for a segment's width `w`, or no room for
    /// the table (or more windows than a `u32` counts).
    pub(crate) fn over(
        ticks: &[i64],
        (sum, r, bias): (&[i64], &[f64], &[f64]),
        m: usize,
        segments: Segments,
    ) -> Option<Self> {
        let k = segments.count();
        if m < k {
            return None;
        }
        let widest = m.div_ceil(k);
        if (m as u128) * (widest as u128) > 1u128 << 63 {
            return None;
        }
        let windows = sum.len();
        u32::try_from(windows).ok()?;
        // Ticks `a..b` sum to `prefix[b] − prefix[a]`, modulo `2⁶⁴`.
        let mut prefix = Vec::new();
        prefix.try_reserve_exact(ticks.len() + 1).ok()?;
        prefix.push(0i64);
        let mut running = 0i64;
        for &tick in ticks {
            running = running.wrapping_add(tick);
            prefix.push(running);
        }
        let bounds: Vec<(usize, usize)> = (0..k).map(|s| (s * m / k, (s + 1) * m / k)).collect();
        let roots: Vec<f64> = bounds
            .iter()
            .map(|&(a, b)| libm::sqrt((b - a) as f64))
            .collect();
        let mut values = Vec::new();
        values.try_reserve_exact(windows.checked_mul(k)?).ok()?;
        for i in 0..windows {
            let valid = r[i] != 0.0 && bias[i] != f64::NEG_INFINITY;
            for (&(a, b), &root) in bounds.iter().zip(&roots) {
                if !valid {
                    values.push(f32::NAN);
                    continue;
                }
                let segment = prefix[i + b].wrapping_sub(prefix[i + a]);
                let n = (m as i64)
                    .wrapping_mul(segment)
                    .wrapping_sub(((b - a) as i64).wrapping_mul(sum[i]));
                values.push(((n as f64 * r[i]) / root) as f32);
            }
        }
        let mut groups = Vec::new();
        groups
            .try_reserve_exact(windows.div_ceil(COLUMN_GROUP).checked_mul(2 * k)?)
            .ok()?;
        for group in values.chunks(COLUMN_GROUP * k) {
            let (mut lo, mut neg_hi) = (Vec::new(), Vec::new());
            lo.resize(k, f32::INFINITY);
            neg_hi.resize(k, f32::INFINITY);
            for window in group.chunks_exact(k) {
                for s in 0..k {
                    let v = window[s];
                    lo[s] = if v < lo[s] { v } else { lo[s] };
                    neg_hi[s] = if -v < neg_hi[s] { -v } else { neg_hi[s] };
                }
            }
            groups.extend_from_slice(&lo);
            groups.extend_from_slice(&neg_hi);
        }
        let unions = unions(&groups, k, UNION_GROUPS)?;
        let mut constants = Vec::new();
        constants.try_reserve_exact(windows + 1).ok()?;
        constants.push(0u32);
        let mut count = 0u32;
        for (&r, &bias) in r.iter().zip(bias) {
            count += u32::from(crate::columns::is_constant(r, bias));
            constants.push(count);
        }
        Some(Self {
            segments,
            values,
            groups,
            unions,
            constants,
            margin: libm::sqrt(m as f64) * MARGIN_PER_ROOT,
            half_inv_m: 0.5 / m as f64,
        })
    }

    /// How many segments the windows are cut into.
    pub(crate) fn segments(&self) -> Segments {
        self.segments
    }

    /// How many windows have coefficients (valid or not).
    pub(crate) fn windows(&self) -> usize {
        self.values.len() / self.segments.count()
    }

    /// Window `i`'s coefficients, NaN for a window that is not valid.
    #[cfg(test)]
    pub(crate) fn window(&self, i: usize) -> &[f32] {
        let k = self.segments.count();
        &self.values[i * k..(i + 1) * k]
    }

    /// Whether any of `columns` is a constant window: two reads of the running count.
    #[inline]
    pub(crate) fn holds_constant(&self, columns: Range<usize>) -> bool {
        self.constants[columns.end] != self.constants[columns.start]
    }

    /// The box of the valid windows among `rows`, `None` when none is valid.
    #[inline]
    pub(crate) fn rows_span<const K: usize>(&self, rows: Range<usize>) -> Option<Span<K>> {
        debug_assert_eq!(K, self.segments.count());
        let mut span = Span::EMPTY;
        for values in self.values[rows.start * K..rows.end * K].chunks_exact(K) {
            span.take(values);
        }
        (span.lo[0] < f32::INFINITY).then_some(span)
    }

    /// The box of exactly the valid windows among `columns`, read window by window: tighter than
    /// [`Self::columns_span`], which rounds out to whole groups, and costing `K` reads a window.
    /// It holds every valid window among `columns`, which is all [`Self::value`]'s proof asks of
    /// a box.
    #[inline]
    pub(crate) fn window_span<const K: usize>(&self, columns: Range<usize>) -> Span<K> {
        debug_assert_eq!(K, self.segments.count());
        let mut span = Span::EMPTY;
        for values in self.values[columns.start * K..columns.end * K].chunks_exact(K) {
            span.take(values);
        }
        span
    }

    /// A box holding every valid window among `columns` (non-empty): the union of the boxes of
    /// the groups they fall in. From [`UNION_GROUPS`] groups on, it joins the union boxes that
    /// start at the first group and at every [`UNION_GROUPS`]-th past it, and the one that ends
    /// at the last group: they overlap, and together cover exactly those groups (a union of
    /// boxes is a box over every window in any of them). Fewer groups are joined one by one.
    #[inline]
    pub(crate) fn columns_span<const K: usize>(&self, columns: Range<usize>) -> Span<K> {
        debug_assert_eq!(K, self.segments.count());
        let mut span = Span::EMPTY;
        let (first, end) = (
            columns.start / COLUMN_GROUP,
            (columns.end - 1) / COLUMN_GROUP + 1,
        );
        let at = |g: usize| g * 2 * K..(g + 1) * 2 * K;
        if end - first < UNION_GROUPS {
            for g in first..end {
                span.join(&self.groups[at(g)]);
            }
            return span;
        }
        let mut g = first;
        while g + UNION_GROUPS < end {
            span.join(&self.unions[at(g)]);
            g += UNION_GROUPS;
        }
        span.join(&self.unions[at(end - UNION_GROUPS)]);
        span
    }

    /// The block value `U` of the pairs between windows boxed by `rows` (non-empty) and by `cols`:
    /// `U ≥ ρ − 2⁻⁴⁷` for every such pair when `K ≤ 16` (`ρ − 2⁻⁴⁵` when `K ≤ 64`), so
    /// `U + DELTA`, rounded, is at least `ρ`. Every operation is a separate IEEE-754 operation in
    /// round-to-nearest; no fused multiply-add, so every target gives the same bits.
    ///
    /// **Proof.** Write `u = 2⁻⁵³` and `v = 2⁻²⁴`, the unit roundoffs of `f64` and `f32`, and let
    /// `i` be any valid row window and `j` any valid column window in the boxes. `|e| ≤ ‖z‖ = √m`
    /// for every coefficient.
    /// 0. *The boxes.* A box is, per segment, the minimum of stored coefficients and the minimum
    ///    of their negations (the negated maximum), each exact: a minimum selects one of its
    ///    operands, and negation is exact. The column box [`Self::columns_span`] gives a block is
    ///    the union of group boxes, or of union boxes of [`UNION_GROUPS`] groups (part
    ///    granularity): a union of boxes holds every coefficient any of them holds, so every
    ///    valid column window of the block is in it, whichever unions cover its groups. A larger
    ///    box only lowers each gap below. So the steps below hold for every pair in the block.
    /// 1. *A stored coefficient.* `N` is exact: `|N_s| ≤ √(m·w_s·V) < 2⁶³` (Cauchy–Schwarz on the
    ///    segment, `V < 2⁶³`, `m·w_s ≤ 2⁶³`), so its wrapping `i64` value is the integer itself.
    ///    `N as f64` rounds by `u`; the column builder's `r = 1/√V` is within `3u` relative
    ///    (`V as f64`, the square root and the division round by `u` each); the product, `√w_s`
    ///    and the division round by `u` each: `7u` and `O(u²)` in all, below `8u`. The `f32`
    ///    rounding adds `v` relative (no subnormal: a nonzero `|e| ≥ 2⁻³¹·⁵/√m`). So the stored
    ///    `ê` has `|ê − e| ≤ ε = (v + 8u)√m`.
    /// 2. *A gap.* Box bounds are stored values, exact. `D = fl(lo_c − hi_r)`, formed as
    ///    `fl(lo_c + (−hi_r))`, the same operation, has
    ///    `|D − (lo_c − hi_r)| ≤ u·2(1 + 2v)√m ≤ 2.0001u√m`, and by 1, `e_s(j) − e_s(i) ≥
    ///    lo_c − hi_r − 2ε`, so `e_s(j) − e_s(i) ≥ D − (2v + 18.0001u)√m`. The margin is
    ///    `M = fl(√fl(m))·2⁻²² ≥ 4v√m(1 − 2u)`. With `g = fl(max(D, D') − M) > 0` (`D'` the other
    ///    side, alike), `g ≤ (D − M) + u(2.0001√m + M) ≤ (D − M) + 2.01u√m` for the side that
    ///    gives it, so `e_s(j) − e_s(i) ≥ g + (4v(1 − 2u) − 2v − 20.02u)√m ≥ g > 0` (`2v ≫ 21u`).
    ///    Hence `(e_s(i) − e_s(j))² ≥ g_s²` for every segment kept, and the real
    ///    `G = Σ g_s² ≤ ‖zᵢ − zⱼ‖² = 2m(1 − ρ) ≤ 4m`.
    /// 3. *The sum.* A segment that keeps no gap adds an exact `0`. `K` products and, on any
    ///    term's path through the sum (four interleaved partial sums, then two pairs, in any
    ///    order), at most `K − 1` additions of non-negative terms give `Ĝ ≤ G(1 + u)^K`, so `G ≥ Ĝ(1 − Ku)` and `Ĝ/(2m) ≤ 2.0001`; then
    ///    `ρ ≤ 1 − G/(2m) ≤ 1 − Ĝ/(2m) + 2.0001·K·u`.
    /// 4. *The value.* `half_inv_m = fl(0.5/fl(m))` is `1/(2m)` within `2.0001u` relative, and
    ///    `H = fl(Ĝ·half_inv_m)` is `Ĝ/(2m)` within `3.0001u` relative, so within `6.01u`
    ///    absolute; `U = fl(1 − H)` rounds by at most `1.0002u` (`|1 − H| ≤ 1.0002`). So
    ///    `ρ ≤ U + (2.0001·K + 7.02)·u`: `≤ U + 39.1u < U + 2⁻⁴⁷` for `K ≤ 16`, `< U + 2⁻⁴⁵` for
    ///    `K ≤ 64`.
    /// 5. *The consumer's sum.* `fl(U + DELTA)` with `DELTA = 2⁻⁴⁴` loses at most `2u = 2⁻⁵²`
    ///    (`|U + DELTA| ≤ 2`), so it is at least `ρ + 2⁻⁴⁵ − 2⁻⁵² > ρ`.
    ///
    /// Infinities: an empty column box (no valid column) gives gaps of `+∞` and `U = −∞`, which
    /// bounds the empty set of valid pairs. No NaN arises: a row box is never empty, and every
    /// stored bound in a non-empty box is finite.
    #[inline]
    pub(crate) fn value<const K: usize>(&self, rows: &Span<K>, cols: &Span<K>) -> f64 {
        const { assert!(K <= 64, "the proof's slack covers at most 64 segments") };
        // The gaps, a lane per segment with no branch, then their squares summed in four
        // interleaved partial sums: the proof holds for any order of the additions.
        let mut squares = [0.0f64; K];
        for (s, square) in squares.iter_mut().enumerate() {
            // `lo_c − hi_r` and `lo_r − hi_c`, as additions of the stored negated highs: the
            // same operation on the same values, so the same bits.
            let above = f64::from(cols.lo[s]) + f64::from(rows.neg_hi[s]);
            let below = f64::from(rows.lo[s]) + f64::from(cols.neg_hi[s]);
            let gap = if above > below { above } else { below } - self.margin;
            let kept = if gap > 0.0 { gap } else { 0.0 };
            *square = kept * kept;
        }
        let mut partial = [0.0f64; 4];
        for (s, &square) in squares.iter().enumerate() {
            partial[s % 4] += square;
        }
        let sum = (partial[0] + partial[1]) + (partial[2] + partial[3]);
        1.0 - sum * self.half_inv_m
    }
}

/// `2⁻²²`: the gap margin per `√m`, four times the `f32` unit roundoff.
const MARGIN_PER_ROOT: f64 = 1.0 / (1u64 << 22) as f64;

#[cfg(test)]
mod tests;
