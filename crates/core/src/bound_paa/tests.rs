extern crate std;

use super::{COLUMN_GROUP, Coefficients, DELTA, Segments, Span, UNION_GROUPS};
use crate::batch::Quantized;
use crate::quantize::Scale;
use alloc::vec::Vec;
use core::num::NonZeroUsize;
use num_bigint::BigInt;
use proptest::prelude::*;

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

/// What the tests read of a column set.
struct Set {
    ticks: Vec<i64>,
    r: Vec<f64>,
    bias: Vec<f64>,
}

impl Set {
    fn windows(&self) -> usize {
        self.r.len()
    }
}

/// The column set of `ticks` (as whole ticks at scale 1) and its coefficients at `m`.
fn built(ticks: &[f64], m: usize) -> (Set, Option<Coefficients>) {
    let q = Quantized::build(ticks, nz(m), Scale::new(1.0).unwrap());
    let (ticks, sum, r, bias) = q.parts();
    let c = Coefficients::over(ticks, (sum, r, bias), m, Segments::for_window(m));
    let set = Set {
        ticks: ticks.to_vec(),
        r: r.to_vec(),
        bias: bias.to_vec(),
    };
    (set, c)
}

fn valid(q: &Set, i: usize) -> bool {
    q.r[i] != 0.0 && q.bias[i] != f64::NEG_INFINITY
}

/// `(N, Va, Vb)` of windows `i` and `j` of `ticks`, exact: wrapping `i128` sums give the true
/// values, every one of them below `2⁶³` in magnitude for valid windows.
fn exact(ticks: &[i64], m: usize, i: usize, j: usize) -> (i128, i128, i128) {
    let (a, b) = (&ticks[i..i + m], &ticks[j..j + m]);
    let sum = |w: &[i64]| w.iter().fold(0i128, |s, &t| s.wrapping_add(i128::from(t)));
    let dot = |x: &[i64], y: &[i64]| {
        x.iter().zip(y).fold(0i128, |s, (&p, &q)| {
            s.wrapping_add(i128::from(p).wrapping_mul(i128::from(q)))
        })
    };
    let mm = m as i128;
    let cross = |x: &[i64], y: &[i64]| {
        mm.wrapping_mul(dot(x, y))
            .wrapping_sub(sum(x).wrapping_mul(sum(y)))
    };
    (cross(a, b), cross(a, a), cross(b, b))
}

/// Whether `u ≥ ρ = n/√(va·vb)` exactly (`va`, `vb > 0`).
fn at_least(u: f64, (n, va, vb): (i128, i128, i128)) -> bool {
    if u.is_infinite() {
        return u > 0.0;
    }
    // u = mantissa · 2^exp, exactly.
    let bits = u.to_bits();
    let exp_field = ((bits >> 52) & 0x7ff) as i64;
    let frac = bits & ((1u64 << 52) - 1);
    let (mantissa, exp) = if exp_field == 0 {
        (frac, -1074)
    } else {
        (frac | (1u64 << 52), exp_field - 1075)
    };
    let (u_neg, n_neg) = (u < 0.0, n < 0);
    if u >= 0.0 && n <= 0 {
        return true;
    }
    if u < 0.0 && n >= 0 {
        return false;
    }
    // Same signs: compare u²·va·vb with n², scaled to integers.
    let mut lhs = BigInt::from(mantissa).pow(2) * BigInt::from(va) * BigInt::from(vb);
    let mut rhs = BigInt::from(n).pow(2);
    if exp < 0 {
        rhs <<= (-2 * exp) as usize;
    } else {
        lhs <<= (2 * exp) as usize;
    }
    debug_assert_eq!(u_neg, n_neg);
    if u_neg { lhs <= rhs } else { lhs >= rhs }
}

/// The upper bound the consumer reads, `value + DELTA` rounded as `Bound::upper` rounds it.
fn upper<const K: usize>(c: &Coefficients, rows: &Span<K>, cols: &Span<K>) -> f64 {
    c.value(rows, cols) + DELTA
}

/// Checks every valid pair between `rows` and `cols` against the value of their boxes, both as
/// a column range (group boxes) and as exact windows; returns how many pairs it checked.
fn check_block<const K: usize>(
    q: &Set,
    c: &Coefficients,
    m: usize,
    rows: core::ops::Range<usize>,
    cols: core::ops::Range<usize>,
) -> Result<usize, TestCaseError> {
    let Some(row_span) = c.rows_span::<K>(rows.clone()) else {
        prop_assert!(rows.clone().all(|i| !valid(q, i)));
        return Ok(0);
    };
    let grouped = upper(c, &row_span, &c.columns_span::<K>(cols.clone()));
    let exact_cols = c.rows_span::<K>(cols.clone());
    let tight = exact_cols.as_ref().map(|span| upper(c, &row_span, span));
    let mut checked = 0;
    for i in rows.clone().filter(|&i| valid(q, i)) {
        for j in cols.clone().filter(|&j| valid(q, j)) {
            let pair = exact(&q.ticks, m, i, j);
            prop_assert!(
                at_least(grouped, pair),
                "grouped {grouped} below ρ of ({i}, {j})"
            );
            let tight = tight.expect("a valid column");
            prop_assert!(
                at_least(tight, pair),
                "tight {tight} below ρ of ({i}, {j}) {pair:?}"
            );
            checked += 1;
        }
    }
    Ok(checked)
}

/// Checks every valid pair between `rows` and `cols` against the value of the rows' box and the
/// box of exactly the columns' windows ([`Coefficients::window_span`], a sub-block's), and of the
/// group boxes of the same columns: the exact box is the tighter, and both bound every pair.
/// Returns how many pairs it checked.
fn check_fine<const K: usize>(
    q: &Set,
    c: &Coefficients,
    m: usize,
    rows: core::ops::Range<usize>,
    cols: core::ops::Range<usize>,
) -> Result<usize, TestCaseError> {
    let Some(row_span) = c.rows_span::<K>(rows.clone()) else {
        return Ok(0);
    };
    let fine = upper(c, &row_span, &c.window_span::<K>(cols.clone()));
    let grouped = upper(c, &row_span, &c.columns_span::<K>(cols.clone()));
    prop_assert!(
        fine <= grouped,
        "the window box {fine} is looser than the group box {grouped}"
    );
    let mut checked = 0;
    for i in rows.clone().filter(|&i| valid(q, i)) {
        for j in cols.clone().filter(|&j| valid(q, j)) {
            prop_assert!(
                at_least(fine, exact(&q.ticks, m, i, j)),
                "fine {fine} below ρ of ({i}, {j})"
            );
            checked += 1;
        }
    }
    Ok(checked)
}

fn check_all(
    q: &Set,
    c: &Coefficients,
    m: usize,
    block: usize,
    lanes: usize,
) -> Result<usize, TestCaseError> {
    let n = q.windows();
    let mut checked = 0;
    for r0 in (0..n).step_by(block.max(1)) {
        let rows = r0..(r0 + block).min(n);
        for c0 in (0..n).step_by(lanes.max(1)) {
            let cols = c0..(c0 + lanes).min(n);
            checked += match c.segments {
                Segments::Four => check_block::<4>(q, c, m, rows.clone(), cols)?,
                Segments::Eight => check_block::<8>(q, c, m, rows.clone(), cols)?,
                Segments::Sixteen => check_block::<16>(q, c, m, rows.clone(), cols)?,
            };
        }
    }
    Ok(checked)
}

/// Steps of `width` ticks: every window starting at a multiple of `width` is constant on each of
/// its segments when `width` is a segment's width, so Bessel's inequality is an equality for
/// pairs of them and the bound is as tight as it gets.
fn steps(levels: &[i64], width: usize) -> Vec<f64> {
    levels
        .iter()
        .flat_map(|&level| core::iter::repeat_n(level as f64, width))
        .collect()
}

proptest! {
    #![proptest_config(testkit::generators::seeded(64))]

    /// Every valid pair's exact correlation is at most the value of any block holding it, on
    /// random ticks, with flat runs and gaps.
    #[test]
    fn the_value_bounds_every_pair_in_a_block(
        ticks in prop::collection::vec(-40i64..40, 40..160),
        m in 4usize..24,
        flat in 0usize..30,
        block in 1usize..9,
        lanes in 1usize..40,
    ) {
        let mut series: Vec<f64> = ticks.iter().map(|&t| t as f64).collect();
        let at = flat.min(series.len() - 1);
        let level = series[at];
        for x in &mut series[at..(at + 2 * m).min(ticks.len())] {
            *x = level;
        }
        if flat % 3 == 0 {
            let len = series.len();
            series[flat % len] = f64::NAN;
        }
        let (q, c) = built(&series, m);
        if q.windows() == 0 {
            return Ok(());
        }
        let c = c.expect("m covers the segments");
        check_all(&q, &c, m, block, lanes)?;
    }

    /// The same where the bound is tight: windows constant on their segments, one row against one
    /// column, at every segment count.
    #[test]
    fn the_value_bounds_every_pair_where_bessel_is_tight(
        levels in prop::collection::vec(-1000i64..1000, 12..40),
        m in prop::sample::select(alloc::vec![4usize, 8, 16, 32, 64, 384, 512, 1024]),
    ) {
        let k = Segments::for_window(m).count();
        let width = m / k;
        let series = steps(&levels, width);
        let (q, c) = built(&series, m);
        prop_assume!(q.windows() != 0);
        let c = c.expect("m covers the segments");
        let aligned: Vec<usize> = (0..q.windows()).step_by(width).collect();
        for &i in &aligned {
            for &j in &aligned {
                match c.segments {
                    Segments::Four => check_block::<4>(&q, &c, m, i..i + 1, j..j + 1)?,
                    Segments::Eight => check_block::<8>(&q, &c, m, i..i + 1, j..j + 1)?,
                    Segments::Sixteen => check_block::<16>(&q, &c, m, i..i + 1, j..j + 1)?,
                };
            }
        }
    }

    /// Ticks near the largest the column builder keeps valid: wide levels with a small wiggle.
    #[test]
    fn the_value_bounds_every_pair_near_the_range_limit(
        wiggle in prop::collection::vec(0i64..3, 60..120),
        top in (1i64 << 40)..(1i64 << 52),
        m in 4usize..12,
    ) {
        let series: Vec<f64> = wiggle
            .iter()
            .enumerate()
            .map(|(p, &w)| (if (p / 17) % 2 == 0 { top } else { -top } + w) as f64)
            .collect();
        let (q, c) = built(&series, m);
        let c = c.expect("m covers the segments");
        check_all(&q, &c, m, 8, 64)?;
    }

    /// The same over parts of the engine's block (8 rows against up to 71 columns and more, so
    /// a part reads union boxes of [`UNION_GROUPS`] groups), on longer random walks with flat
    /// runs: every valid pair is at most the value, in exact arithmetic.
    #[test]
    fn the_part_boxes_bound_every_pair_in_a_part(
        steps in prop::collection::vec(-6i64..7, 400..900),
        m in prop::sample::select(alloc::vec![8usize, 13, 16, 32, 40]),
        flat in 0usize..300,
        lanes in 57usize..200,
        offset in 0usize..8,
    ) {
        let mut level = 0i64;
        let mut series: Vec<f64> = steps
            .iter()
            .map(|&d| {
                level += d;
                level as f64
            })
            .collect();
        let at = flat.min(series.len() - 1);
        let held = series[at];
        for x in &mut series[at..(at + m).min(steps.len())] {
            *x = held;
        }
        let (q, c) = built(&series, m);
        prop_assume!(q.windows() > lanes + offset + 8);
        let c = c.expect("m covers the segments");
        let n = q.windows();
        let mut checked = 0;
        for r0 in (0..n).step_by(8) {
            let rows = r0..(r0 + 8).min(n);
            for c0 in (offset..n).step_by(lanes) {
                let cols = c0..(c0 + lanes).min(n);
                checked += match c.segments {
                    Segments::Four => check_block::<4>(&q, &c, m, rows.clone(), cols)?,
                    Segments::Eight => check_block::<8>(&q, &c, m, rows.clone(), cols)?,
                    Segments::Sixteen => check_block::<16>(&q, &c, m, rows.clone(), cols)?,
                };
            }
        }
        prop_assert!(checked > 0);
    }

    /// A sub-block's box (8 rows against the 15 windows of 8 slots) and a group's (64 rows against
    /// the 127 windows of one part) hold every valid pair under their own value, in exact
    /// arithmetic, on walks with flat runs.
    #[test]
    fn the_sub_block_and_group_boxes_bound_every_pair_in_them(
        steps in prop::collection::vec(-6i64..7, 300..700),
        m in prop::sample::select(alloc::vec![8usize, 13, 16, 40]),
        flat in 0usize..300,
        offset in 0usize..16,
    ) {
        let mut level = 0i64;
        let mut series: Vec<f64> = steps
            .iter()
            .map(|&d| {
                level += d;
                level as f64
            })
            .collect();
        let at = flat.min(series.len() - 1);
        let held = series[at];
        for x in &mut series[at..(at + m).min(steps.len())] {
            *x = held;
        }
        let (q, c) = built(&series, m);
        let c = c.expect("m covers the segments");
        let n = q.windows();
        prop_assume!(n > 200 + offset);
        let mut checked = 0;
        for (rows, width) in [(8usize, 15usize), (64, 127)] {
            for r0 in (0..n).step_by(rows) {
                let rows = r0..(r0 + rows).min(n);
                for c0 in (offset..n).step_by(width.div_ceil(2)) {
                    let cols = c0..(c0 + width).min(n);
                    checked += match c.segments {
                        Segments::Four => check_fine::<4>(&q, &c, m, rows.clone(), cols)?,
                        Segments::Eight => check_fine::<8>(&q, &c, m, rows.clone(), cols)?,
                        Segments::Sixteen => check_fine::<16>(&q, &c, m, rows.clone(), cols)?,
                    };
                }
            }
        }
        prop_assert!(checked > 0);
    }

    /// Every stored coefficient is within the proof's `(2⁻²⁴ + 8·2⁻⁵³)·√m` of `N/√(w·V)`.
    #[test]
    fn a_coefficient_is_its_exact_value_rounded(
        ticks in prop::collection::vec(-1000i64..1000, 30..120),
        m in 4usize..30,
    ) {
        let series: Vec<f64> = ticks.iter().map(|&t| t as f64).collect();
        let (q, c) = built(&series, m);
        let c = c.expect("m covers the segments");
        let k = c.segments.count();
        for i in (0..q.windows()).filter(|&i| valid(&q, i)) {
            let window = &q.ticks[i..i + m];
            let total: i64 = window.iter().sum();
            let (_, v, _) = exact(&q.ticks, m, i, i);
            for s in 0..k {
                let (a, b) = (s * m / k, (s + 1) * m / k);
                let segment: i64 = window[a..b].iter().sum();
                let n = m as i64 * segment - (b - a) as i64 * total;
                let e = n as f64 / libm::sqrt((b - a) as f64 * v as f64);
                let stored = f64::from(c.values[i * k + s]);
                let eps = (1.0 / (1u64 << 24) as f64 + 12.0 / (1u64 << 53) as f64)
                    * libm::sqrt(m as f64);
                prop_assert!((stored - e).abs() <= eps, "window {i} segment {s}: {stored} against {e}");
            }
        }
    }
}

/// `x` as `mantissa · 2^exp`, exactly (`x` finite).
fn dyadic(x: f64) -> (BigInt, i64) {
    let bits = x.to_bits();
    let exp_field = ((bits >> 52) & 0x7ff) as i64;
    let frac = bits & ((1u64 << 52) - 1);
    let (mantissa, exp) = if exp_field == 0 {
        (frac, -1074)
    } else {
        (frac | (1u64 << 52), exp_field - 1075)
    };
    let mantissa = BigInt::from(mantissa);
    (if x < 0.0 { -mantissa } else { mantissa }, exp)
}

/// Steps 3 to 5 of the value's proof, in exact arithmetic: with the gaps `g_s` the evaluation
/// keeps (each a float, formed as `value` forms it), `value + DELTA`, rounded, is at least
/// `1 − Σ g_s²/(2m)` computed exactly.
fn check_rounding<const K: usize>(
    m: usize,
    rows: &Span<K>,
    cols: &Span<K>,
) -> Result<(), TestCaseError> {
    let c = Coefficients::over(&[], (&[], &[], &[]), m, Segments::for_window(m)).unwrap();
    let up = c.value(rows, cols) + DELTA;
    let gaps: Vec<f64> = (0..K)
        .map(|s| {
            let above = f64::from(cols.lo[s]) - f64::from(rows.hi(s));
            let below = f64::from(rows.lo[s]) - f64::from(cols.hi(s));
            above.max(below) - c.margin
        })
        .filter(|&gap| gap > 0.0)
        .collect();
    // 2m·(1 − up) ≤ Σ g², every term scaled to the lowest exponent.
    let mut terms: Vec<(BigInt, i64)> = gaps
        .iter()
        .map(|&g| {
            let (a, e) = dyadic(g);
            (&a * &a, 2 * e)
        })
        .collect();
    let (a, e) = dyadic(up);
    let two_m = BigInt::from(2 * m as u64);
    let low = terms.iter().map(|t| t.1).chain([e, 0]).min().unwrap();
    let scale = |(v, exp): &(BigInt, i64)| v.clone() << (exp - low) as usize;
    let sum: BigInt = terms.drain(..).map(|t| scale(&t)).sum();
    let lhs = &two_m * (scale(&(BigInt::from(1), 0)) - scale(&(a, e)));
    prop_assert!(
        lhs <= sum,
        "value + DELTA = {up} below 1 − Σg²/(2m) at m {m}: gaps {gaps:?}"
    );
    Ok(())
}

fn span<const K: usize>(lo: &[f32], width: &[f32]) -> Span<K> {
    let mut span = Span::<K>::EMPTY;
    for s in 0..K {
        span.lo[s] = lo[s];
        span.neg_hi[s] = -(lo[s] + width[s]);
    }
    span
}

proptest! {
    #![proptest_config(testkit::generators::seeded(512))]

    /// The float evaluation of the value never undershoots its exact gaps' bound by more than
    /// `DELTA` covers, at 4 and 16 segments, over boxes anywhere in `±√m`.
    #[test]
    fn the_value_rounds_no_lower_than_its_gaps_allow(
        m in prop::sample::select(alloc::vec![4usize, 7, 64, 100, 256, 1024, 3000]),
        lo in prop::collection::vec(-1.0f32..1.0, 32),
        width in prop::collection::vec(0.0f32..0.5, 32),
    ) {
        let root = libm::sqrtf(m as f32);
        let lo: Vec<f32> = lo.iter().map(|x| x * root).collect();
        let width: Vec<f32> = width.iter().map(|x| x * root).collect();
        if m <= 256 {
            check_rounding::<4>(m, &span(&lo[..4], &width[..4]), &span(&lo[4..8], &width[4..8]))?;
        } else {
            check_rounding::<16>(m, &span(&lo[..16], &width[..16]), &span(&lo[16..], &width[16..]))?;
        }
    }
}

/// Segments of `⌊m/k⌋` and `⌈m/k⌉` cover the whole window: the coefficients weighted by `√w`
/// sum to `Σ z = 0`, and at one tick a segment they hold all of `‖z‖² = m`.
#[test]
fn segments_cover_the_window_with_a_remainder() {
    let series: Vec<f64> = (0..200).map(|p| ((p * 37) % 23) as f64).collect();
    for m in [4, 5, 7, 10, 13, 255, 257, 600, 1030] {
        let series: Vec<f64> = (0..m + 40)
            .map(|p| series[p % 200] + (p % 7) as f64)
            .collect();
        let (q, c) = built(&series, m);
        let c = c.unwrap();
        let k = c.segments.count();
        for i in (0..q.windows()).filter(|&i| valid(&q, i)) {
            let weighted: f64 = (0..k)
                .map(|s| {
                    f64::from(c.values[i * k + s])
                        * libm::sqrt(((s + 1) * m / k - s * m / k) as f64)
                })
                .sum();
            assert!(
                weighted.abs() < 1e-4 * m as f64,
                "m {m} window {i}: {weighted}"
            );
        }
    }
    let ticks: Vec<f64> = [3.0, -1.0, 4.0, 1.0, -5.0, 9.0, 2.0, 6.0]
        .into_iter()
        .cycle()
        .take(40)
        .collect();
    let (q, c) = built(&ticks, 4);
    let c = c.unwrap();
    for i in (0..q.windows()).filter(|&i| valid(&q, i)) {
        let energy: f64 = c.values[i * 4..i * 4 + 4]
            .iter()
            .map(|&e| f64::from(e) * f64::from(e))
            .sum();
        assert!((energy - 4.0).abs() < 1e-5, "window {i}: {energy}");
    }
}

/// Constant and non-finite windows carry no coefficients and enter no box; a block of rows with
/// none valid has no box, and columns with none valid bound nothing (`−∞`).
#[test]
fn windows_that_are_not_valid_are_never_bounded() {
    let mut series: Vec<f64> = (0..60).map(|p| ((p * 13) % 11) as f64).collect();
    series[10..30].fill(5.0);
    series[45] = f64::NAN;
    let m = 8;
    let (q, c) = built(&series, m);
    let c = c.unwrap();
    for i in 0..q.windows() {
        let nan = c.values[i * 4..i * 4 + 4].iter().all(|v| v.is_nan());
        assert_eq!(nan, !valid(&q, i), "window {i}");
    }
    let constant = 12..18;
    assert!(constant.clone().all(|i| !valid(&q, i)));
    assert!(c.rows_span::<4>(constant.clone()).is_none());
    let rows = c.rows_span::<4>(0..1).unwrap();
    // Groups of eight are 8..16 and 16..24: windows 8..10 and 18..24 are valid.
    assert!(c.value(&rows, &c.columns_span::<4>(constant)) > f64::NEG_INFINITY);
    let none = Span::<4>::EMPTY;
    assert_eq!(c.value(&rows, &none), f64::NEG_INFINITY);
}

/// No coefficients where the proof does not cover the series: `m` below the segment count, or
/// `m·w` past `2⁶³` (checked before any tick is read).
#[test]
fn no_coefficients_where_the_proof_does_not_cover() {
    let (_, c) = built(&[1.0, 2.0, 4.0, 3.0, 5.0, 0.0], 3);
    assert!(c.is_none());
    let huge = 1usize << 40;
    assert!(Coefficients::over(&[], (&[], &[], &[]), huge, Segments::for_window(huge)).is_none());
    let fits = 1usize << 32;
    assert!(Coefficients::over(&[], (&[], &[], &[]), fits, Segments::for_window(fits)).is_some());
}

#[test]
fn the_segment_count_follows_m() {
    assert_eq!(Segments::for_window(64).count(), 4);
    assert_eq!(Segments::for_window(256).count(), 4);
    assert_eq!(Segments::for_window(257).count(), 8);
    assert_eq!(Segments::for_window(511).count(), 8);
    assert_eq!(Segments::for_window(512).count(), 16);
    assert_eq!(Segments::for_window(1024).count(), 16);
}

/// A box's lows and highs, in order.
fn bounds<const K: usize>(span: &Span<K>) -> (Vec<f32>, Vec<f32>) {
    (span.lo.to_vec(), (0..K).map(|s| span.hi(s)).collect())
}

/// The column box of any range of columns is exactly the union of its groups' boxes, joined one
/// group at a time: the union boxes neither drop a covered group (the bound would skip pairs it
/// does not bound) nor add one (looser than the groups).
#[test]
fn a_column_box_is_the_union_of_its_groups() {
    let series: Vec<f64> = (0..1400)
        .map(|p| ((p * 7919) % 211) as f64 + (p / 50) as f64 * 3.0)
        .collect();
    let mut series = series;
    series[300..330].fill(9.0);
    series[700] = f64::NAN;
    let m = 16;
    let (q, c) = built(&series, m);
    let c = c.unwrap();
    let n = q.windows();
    let by_groups = |cols: core::ops::Range<usize>| {
        let mut span = Span::<4>::EMPTY;
        for g in cols.start / COLUMN_GROUP..=(cols.end - 1) / COLUMN_GROUP {
            span.join(&c.groups[g * 8..(g + 1) * 8]);
        }
        span
    };
    let mut unions = 0;
    for start in (0..n).step_by(3) {
        for len in [
            1, 7, 8, 9, 57, 63, 64, 65, 71, 72, 80, 127, 128, 129, 200, 513,
        ] {
            let cols = start..(start + len).min(n);
            if cols.is_empty() {
                continue;
            }
            let groups = (cols.end - 1) / COLUMN_GROUP - cols.start / COLUMN_GROUP + 1;
            unions += usize::from(groups >= UNION_GROUPS);
            assert_eq!(
                bounds(&c.columns_span::<4>(cols.clone())),
                bounds(&by_groups(cols.clone())),
                "columns {cols:?}"
            );
        }
    }
    assert!(unions > 1000, "the union boxes are read: {unions}");
}

/// The running count of constant windows answers, for every range, whether one of its windows
/// is constant, as reading every window's kind does.
#[test]
fn the_constant_count_finds_every_constant_window() {
    let mut series: Vec<f64> = (0..300).map(|p| ((p * 31) % 17) as f64).collect();
    series[40..52].fill(2.0);
    series[200..230].fill(-1.0);
    series[120] = f64::NAN;
    let m = 8;
    let (q, c) = built(&series, m);
    let c = c.unwrap();
    let constant = |i: usize| crate::columns::is_constant(q.r[i], q.bias[i]);
    assert!((0..q.windows()).any(constant));
    for a in 0..q.windows() {
        for b in a + 1..=q.windows() {
            assert_eq!(c.holds_constant(a..b), (a..b).any(constant), "{a}..{b}");
        }
    }
}
