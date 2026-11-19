//! The dense ladder: the rule's constants and threshold, the kernel against an independent
//! oracle, the boxes against every pair they must hold, the masks against a cell-by-cell
//! enumeration, and the pass against the exact `f64` block value and the true correlations.

extern crate std;

use super::{
    Band, Boxes, FloorBlocks, Floors, GRAIN, LAYERS, Ladder, LayerBoxes, MARGIN_PER_ROOT, Rule,
    SLACK, SLOTS, SurvivorRuns, fill_bits, filled, gaps, get_bits, put_bits,
};
use crate::batch::Quantized;
use crate::bound_paa::{Coefficients, DELTA, PaaSides, Segments};
use crate::columns::is_constant;
use crate::quantize::Scale;
use alloc::vec;
use alloc::vec::Vec;
use core::num::NonZeroUsize;
use core::ops::Range;
use proptest::prelude::*;

fn bit(words: &[u64], i: usize) -> bool {
    get_bits(words, i, 1) == 1
}

// ---- the rule ----

#[test]
fn the_threshold_of_the_special_floors() {
    let rule = Rule::new(100, 16);
    assert_eq!(rule.threshold(f64::NAN), f32::INFINITY, "nothing skips");
    assert_eq!(rule.threshold(f64::NEG_INFINITY), f32::INFINITY);
    assert_eq!(rule.threshold(f64::INFINITY), -1.0, "everything skips");
    assert_eq!(rule.threshold(2.0), -1.0);
    assert_eq!(rule.threshold(1.0 + 2.0 * SLACK), -1.0);
    // At one plus the slack the bound is exactly zero: only a positive sum skips.
    assert_eq!(rule.threshold(1.0 + SLACK), f32::from_bits(1));
    assert!(rule.threshold(1.0) > 0.0);
    assert!(rule.threshold(-1.0).is_finite());
}

proptest! {
    #![proptest_config(testkit::generators::seeded(512))]

    /// A higher floor never raises the threshold.
    #[test]
    fn the_threshold_never_rises_with_the_floor(
        a in -3.0f64..3.0,
        b in -3.0f64..3.0,
        m in 16usize..100_000,
    ) {
        let rule = Rule::new(m, 16);
        let (low, high) = if a <= b { (a, b) } else { (b, a) };
        prop_assert!(rule.threshold(low) >= rule.threshold(high));
    }

    /// The threshold is at least `2m(1 − F + slack)` raised by the sum's rounding bound
    /// `1 + (K + 3)·2⁻²⁴` (the proof's requirement), for every `K` the rule covers.
    #[test]
    fn the_threshold_covers_the_sums_rounding(
        floor in -2.0f64..1.000_001,
        m in 16usize..1_000_000,
        k in prop::sample::select(vec![4usize, 8, 16, 32, 64]),
    ) {
        let rule = Rule::new(m, k);
        let exact = ((1.0 - floor) + SLACK) * 2.0 * m as f64;
        let t = f64::from(rule.threshold(floor));
        if exact > 0.0 {
            let v = 1.0 / (1u64 << 24) as f64;
            prop_assert!(t >= exact * (1.0 + (k as f64 + 3.0) * v), "{t} vs {exact}");
        } else {
            prop_assert!(t == -1.0 || t > 0.0);
        }
    }

    /// The margin is the rounded-up `2⁻²¹·√m`: strictly above the `f64` product.
    #[test]
    fn the_margin_is_above_its_product(m in 4usize..4_000_000_000) {
        let margin = f64::from(Rule::new(m, 4).margin());
        prop_assert!(margin > MARGIN_PER_ROOT * libm::sqrt(m as f64));
        prop_assert!(margin >= 1.0 / (1u64 << 21) as f64);
    }
}

// ---- the kernel against an oracle ----

/// A column table of `len` boxes of `K` segments from `values` (low, width, empty), as the layer
/// stores them.
fn table<const K: usize>(len: usize, q: usize, values: &[(f32, f32, bool)]) -> LayerBoxes<K> {
    let mut lo = vec![f32::INFINITY; K * len];
    let mut neg_hi = vec![f32::INFINITY; K * len];
    for (i, &(low, width, empty)) in values.iter().enumerate().take(K * len) {
        if !empty {
            lo[i] = low;
            neg_hi[i] = -(low + width);
        }
    }
    LayerBoxes {
        slots: 16 * q,
        rows_per: 16,
        q,
        shift: 0,
        rows: Vec::new(),
        len,
        lo,
        neg_hi,
        cols_n: 0,
        shape: crate::batch::pass::Shape::Triangle { exclusion: 0 },
    }
}

/// `G` of the entries `v0 + e·q`, written out straight from the rule, segment by segment.
fn oracle<const K: usize>(
    table: &LayerBoxes<K>,
    row: &([f32; K], [f32; K]),
    v0: usize,
    margin: f32,
) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for (e, g) in out.iter_mut().enumerate() {
        let v = v0 + e * table.q;
        for k in 0..K {
            let (lo_c, neg_hi_c) = if v < table.len {
                (table.lo[k * table.len + v], table.neg_hi[k * table.len + v])
            } else {
                (f32::INFINITY, f32::INFINITY)
            };
            let above = lo_c + row.1[k];
            let below = row.0[k] + neg_hi_c;
            let kept = (above.max(below) - margin).max(0.0);
            *g += kept * kept;
        }
    }
    out
}

fn check_kernel<const K: usize>(
    q: usize,
    v0: usize,
    values: &[(f32, f32, bool)],
    row: &[(f32, f32, bool)],
) -> Result<(), TestCaseError> {
    let len = 40;
    let t = table::<K>(len, q, values);
    let mut r = ([f32::INFINITY; K], [f32::INFINITY; K]);
    for (k, &(low, width, empty)) in row.iter().enumerate().take(K) {
        if !empty {
            r.0[k] = low;
            r.1[k] = -(low + width);
        }
    }
    let margin = Rule::new(100, K).margin();
    let got = gaps(&r, &t, v0, margin);
    let want = oracle(&t, &r, v0, margin);
    for e in 0..16 {
        prop_assert!(!got[e].is_nan());
        prop_assert_eq!(got[e].to_bits(), want[e].to_bits(), "entry {}", e);
    }
    Ok(())
}

proptest! {
    #![proptest_config(testkit::generators::seeded(256))]

    /// The pass's gaps are the rule's, bit for bit, on random boxes with empty ones, from any
    /// start and at any stride, at the three segment counts; never NaN.
    #[test]
    fn the_kernel_equals_the_rule_written_out(
        values in prop::collection::vec((-4.0f32..4.0, 0.0f32..3.0, prop::bool::weighted(0.15)), 16 * 40),
        row in prop::collection::vec((-4.0f32..4.0, 0.0f32..3.0, prop::bool::weighted(0.1)), 16),
        q in prop::sample::select(vec![1usize, 2, 4, 16]),
        v0 in 0usize..60,
    ) {
        check_kernel::<4>(q, v0, &values, &row)?;
        check_kernel::<8>(q, v0, &values, &row)?;
        check_kernel::<16>(q, v0, &values, &row)?;
    }
}

// ---- bit helpers and runs ----

proptest! {
    #![proptest_config(testkit::generators::seeded(256))]

    /// Reads, writes and range fills agree with a vector of booleans.
    #[test]
    fn the_bit_helpers_agree_with_a_vector_of_bools(
        seed in prop::collection::vec(any::<u64>(), 4),
        ops in prop::collection::vec((0usize..256, 0usize..17, any::<u32>(), 0usize..300, any::<bool>()), 1..12),
    ) {
        let mut words = seed.clone();
        let mut model: Vec<bool> = (0..256).map(|i| seed[i / 64] >> (i % 64) & 1 == 1).collect();
        for (pos, len, value, end, on) in ops {
            put_bits(&mut words, pos, len, value);
            for j in 0..len {
                if pos + j < 256 {
                    model[pos + j] = value >> j & 1 == 1;
                }
            }
            fill_bits(&mut words, pos..end, on);
            if pos < end.min(256) {
                model[pos..end.min(256)].fill(on);
            }
            let at = pos / 2;
            let want = (0..len).fold(0u32, |a, j| a | u32::from(model.get(at + j).copied().unwrap_or(false)) << j);
            prop_assert_eq!(get_bits(&words, at, len), want);
        }
        for (i, &b) in model.iter().enumerate() {
            prop_assert_eq!(bit(&words, i), b);
        }
    }

    /// The runs are exactly the maximal runs of clear bits below `bits`, whatever is above.
    #[test]
    fn the_runs_are_the_maximal_runs_of_zero_bits(
        words in prop::collection::vec(prop::sample::select(vec![0u64, !0, 0x00FF_00FF_00FF_00FF, 1, 1 << 63]), 1..6),
        noise in prop::collection::vec(any::<u64>(), 0..6),
        bits in 0usize..330,
    ) {
        let words: Vec<u64> = words.iter().zip(noise.iter().chain(core::iter::repeat(&0))).map(|(&w, &n)| w ^ n & n.rotate_left(7)).collect();
        let bits = bits.min(words.len() * 64);
        let got: Vec<Range<usize>> = SurvivorRuns::new(&words, bits).collect();
        let mut want = Vec::new();
        let mut i = 0;
        while i < bits {
            if bit(&words, i) {
                i += 1;
                continue;
            }
            let start = i;
            while i < bits && !bit(&words, i) {
                i += 1;
            }
            want.push(start..i);
        }
        prop_assert_eq!(got, want);
    }
}

/// One run over every bit, none over no bits.
fn whole(bits: usize) -> Vec<Range<usize>> {
    (bits > 0).then_some(0..bits).into_iter().collect()
}

#[test]
fn runs_of_all_zero_and_all_one_masks() {
    for bits in [0usize, 1, 63, 64, 65, 127, 128, 130] {
        let words = vec![0u64; 3];
        let runs: Vec<_> = SurvivorRuns::new(&words, bits).collect();
        assert_eq!(runs, whole(bits));
        let ones = vec![!0u64; 3];
        assert_eq!(SurvivorRuns::new(&ones, bits).count(), 0);
        // Padding above `bits` never makes a run or ends one.
        let mut padded = vec![0u64; 3];
        fill_bits(&mut padded, bits..192, true);
        let runs: Vec<_> = SurvivorRuns::new(&padded, bits).collect();
        assert_eq!(runs, whole(bits));
    }
}

#[test]
fn a_run_may_cross_a_word_and_a_one_splits_it() {
    let mut words = vec![0u64; 3];
    fill_bits(&mut words, 70..71, true);
    fill_bits(&mut words, 64..65, true);
    let runs: Vec<_> = SurvivorRuns::new(&words, 150).collect();
    assert_eq!(runs, vec![0..64, 65..70, 71..150]);
}

#[test]
fn a_floor_block_is_nan_when_anything_it_reads_is() {
    let blocks = [0.5, f64::NAN, 0.25, 0.75];
    let floors = FloorBlocks {
        first: 8,
        floors: &blocks,
    };
    assert_eq!(floors.lowest(8..16), 0.5);
    assert!(floors.lowest(8..17).is_nan(), "a NaN block makes it NaN");
    assert!(floors.lowest(16..24).is_nan());
    assert_eq!(floors.lowest(24..40), 0.25);
    assert!(floors.lowest(0..8).is_nan(), "before the first block");
    assert!(floors.lowest(32..48).is_nan(), "past the last block");
    assert!(floors.lowest(9..9).is_nan(), "empty");
    let ties = FloorBlocks {
        first: 0,
        floors: &[0.5, 0.5, 0.25, 0.25],
    };
    assert_eq!(ties.lowest(0..32), 0.25);
}

#[test]
fn a_size_that_cannot_be_allocated_is_none() {
    assert!(filled::<u64>(usize::MAX / 2, 0).is_none());
    assert!(filled::<u64>(4, 7).is_some_and(|v| v == [7; 4]));
}

// ---- fixtures ----

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

struct Fixture<const K: usize> {
    c: Coefficients,
    series: Vec<f64>,
    r: Vec<f64>,
    bias: Vec<f64>,
    n: usize,
    m: usize,
    lag: usize,
}

impl<const K: usize> Fixture<K> {
    fn valid(&self, i: usize) -> bool {
        self.r[i] != 0.0 && self.bias[i] != f64::NEG_INFINITY
    }

    fn constant(&self, i: usize) -> bool {
        is_constant(self.r[i], self.bias[i])
    }
}

fn fixture<const K: usize>(series: Vec<f64>, m: usize) -> Fixture<K> {
    let q = Quantized::build(&series[..], nz(m), Scale::new(1.0).unwrap());
    let (ticks, sum, r, bias) = q.parts();
    let segments = match K {
        4 => Segments::Four,
        8 => Segments::Eight,
        _ => Segments::Sixteen,
    };
    let c = Coefficients::over(ticks, (sum, r, bias), m, segments).expect("m covers the segments");
    Fixture {
        n: r.len(),
        r: r.to_vec(),
        bias: bias.to_vec(),
        c,
        series,
        m,
        lag: crate::exclusion::zone(m) + 1,
    }
}

#[derive(Clone, Debug)]
struct Shape {
    walk: Vec<i64>,
    m: usize,
    flat: (usize, usize),
    gap: (usize, usize),
    repeat: usize,
    picks: (usize, usize, usize, usize, usize, usize),
    floors: Vec<(u8, f64, u8)>,
}

fn shapes() -> impl Strategy<Value = Shape> {
    (
        prop::collection::vec(-6i64..7, 260..520),
        16usize..40,
        (0usize..400, 0usize..120),
        (0usize..400, 0usize..80),
        0usize..300,
        (
            0usize..5,
            0usize..64,
            0usize..8,
            0usize..400,
            0usize..90,
            0usize..3,
        ),
        prop::collection::vec((0u8..20, -1.0f64..1.0, 0u8..16), 64),
    )
        .prop_map(|(walk, m, flat, gap, repeat, picks, floors)| Shape {
            walk,
            m,
            flat,
            gap,
            repeat,
            picks,
            floors,
        })
}

fn series_of(s: &Shape) -> Vec<f64> {
    let mut level = 0i64;
    let mut series: Vec<f64> = s
        .walk
        .iter()
        .map(|&step| {
            level += step;
            level as f64
        })
        .collect();
    let len = series.len();
    if s.repeat > 0 {
        let at = s.repeat.min(len - 41);
        let head: Vec<f64> = series[..40].to_vec();
        series[at..at + 40].copy_from_slice(&head);
    }
    let a = s.flat.0.min(len - 1);
    let b = (a + s.flat.1).min(len);
    let level = series[a];
    series[a..b].fill(level);
    let a = s.gap.0.min(len - 1);
    let b = (a + s.gap.1).min(len);
    series[a..b].fill(f64::NAN);
    series
}

struct Case<const K: usize> {
    fix: Fixture<K>,
    band: Band,
    boxes: Boxes<K>,
    ladder: Ladder<K>,
}

/// The case of `s` at `K` segments, `None` when its picks leave no band.
fn case<const K: usize>(s: &Shape) -> Option<Case<K>> {
    let fix = fixture::<K>(series_of(s), s.m);
    let (n, lag) = (fix.n, fix.lag);
    let (side_pick, band_pick, off_pick, qa_pick, len_pick, _) = s.picks;
    let side = 16usize << side_pick;
    let row0 = (band_pick % n.div_ceil(side)) * side;
    let mut start = row0.max(lag);
    if off_pick % 4 == 0 {
        start = (row0 + off_pick * side / 8).max(lag);
    }
    let end = (row0 + side).min(n);
    if start >= end || n < lag + 16 {
        return None;
    }
    let qa = (qa_pick % ((n - lag) / 16 + 1)) * 16;
    let qb = (qa + 16 * (1 + len_pick)).min(n);
    if qa >= qb {
        return None;
    }
    let band = Band {
        rows: start..end,
        slots: qa..qb,
    };
    let boxes = Boxes::try_new(PaaSides::of(&fix.c), fix.m, (n, lag, side)).expect("boxes");
    let ladder = Ladder::try_new(&boxes, &band).expect("ladder");
    Some(Case {
        fix,
        band,
        boxes,
        ladder,
    })
}

fn k_of(s: &Shape) -> usize {
    [4, 8, 16][s.picks.5 % 3]
}

/// A floor per block of eight windows, as `floors` and the pool of block values describe.
fn floors_from(floors: &[(u8, f64, u8)], pool: &[f64], blocks: usize, salt: usize) -> Vec<f64> {
    const OFFSETS: [f64; 10] = [-4e-6, -1e-6, -1e-7, 0.0, 1e-7, 5e-7, 1e-6, 2e-6, 4e-6, 9e-6];
    (0..blocks)
        .map(|b| {
            let (kind, v, p) = floors[(b + salt) % floors.len()];
            match kind {
                0 => f64::NAN,
                1 => 2.0,
                2 => f64::NEG_INFINITY,
                3 => 1.0,
                4..=7 => v,
                _ if pool.is_empty() => v,
                _ => {
                    pool[(usize::from(p) * 7 + b) % pool.len()]
                        + OFFSETS[usize::from(kind + p) % 10]
                }
            }
        })
        .collect()
}

fn run<const K: usize>(case: &mut Case<K>, rows: &[f64], cols: &[f64]) -> super::Work {
    run_with(case, rows, Some(cols))
}

/// [`run`] with no column floors at all when `cols` is `None`: a one-sided run.
fn run_with<const K: usize>(case: &mut Case<K>, rows: &[f64], cols: Option<&[f64]>) -> super::Work {
    let floors = Floors {
        rows: FloorBlocks {
            first: 0,
            floors: rows,
        },
        cols: cols.map(|cols| FloorBlocks {
            first: 0,
            floors: cols,
        }),
    };
    let all = 0..case.ladder.bits;
    case.ladder
        .refresh(&case.boxes, PaaSides::of(&case.fix.c), &floors, all.clone());
    case.ladder.pass(&case.boxes, all)
}

/// What a block holds, by enumeration: its pair cells as (row, column), and whether every cell
/// of it is a pair.
struct Entry {
    cells: Vec<(usize, usize)>,
    all: bool,
    rows: Range<usize>,
    cols: Range<usize>,
}

fn entry<const K: usize>(case: &Case<K>, l: usize, r: usize, e: usize) -> Entry {
    let layer = &case.ladder.layers[l];
    let rows = (layer.a0 + r) * layer.rows_per..(layer.a0 + r + 1) * layer.rows_per;
    let slots = (layer.base + e) * layer.slots..(layer.base + e + 1) * layer.slots;
    let (mut cells, mut total) = (Vec::new(), 0);
    for i in rows.clone() {
        for s in slots.clone() {
            total += 1;
            if case.band.rows.contains(&i) && case.band.slots.contains(&s) && s + case.fix.lag <= i
            {
                cells.push((i, i - case.fix.lag - s));
            }
        }
    }
    let span = |f: fn(&(usize, usize)) -> usize| {
        let lo = cells.iter().map(f).min().unwrap_or(0);
        let hi = cells.iter().map(f).max().unwrap_or(0);
        lo..hi + 1
    };
    Entry {
        all: cells.len() == total,
        rows: span(|c| c.0),
        cols: span(|c| c.1),
        cells,
    }
}

fn entries<const K: usize>(case: &Case<K>) -> Vec<(usize, usize, usize)> {
    let mut all = Vec::new();
    for (l, layer) in case.ladder.layers.iter().enumerate() {
        for r in 0..layer.nrows {
            for e in 0..layer.bits {
                all.push((l, r, e));
            }
        }
    }
    all
}

fn skipped<const K: usize>(case: &Case<K>, (l, r, e): (usize, usize, usize)) -> bool {
    let layer = &case.ladder.layers[l];
    bit(layer.row(&layer.skip, r), e)
}

// ---- the layout ----

fn check_masks<const K: usize>(s: &Shape) -> Result<(), TestCaseError> {
    let Some(case) = case::<K>(s) else {
        return Ok(());
    };
    for &(l, r, e) in &entries(&case) {
        let layer = &case.ladder.layers[l];
        let held = entry(&case, l, r, e);
        let pairs = !held.cells.is_empty();
        prop_assert_eq!(
            bit(layer.row(&layer.void, r), e),
            !pairs,
            "void {:?}",
            (l, r, e)
        );
        prop_assert_eq!(
            bit(layer.row(&layer.edge, r), e),
            pairs && !held.all,
            "edge {:?}",
            (l, r, e)
        );
        prop_assert_eq!(skipped(&case, (l, r, e)), !pairs, "fresh {:?}", (l, r, e));
    }
    Ok(())
}

fn check_boxes<const K: usize>(s: &Shape) -> Result<(), TestCaseError> {
    let fix = fixture::<K>(series_of(s), s.m);
    let (n, lag) = (fix.n, fix.lag);
    let side = 16usize << s.picks.0;
    let boxes = Boxes::<K>::try_new(PaaSides::of(&fix.c), fix.m, (n, lag, side)).expect("boxes");
    for table in &boxes.layers {
        for i in lag..n {
            for slot in 0..=i - lag {
                let (a, b, j) = (i / table.rows_per, slot / table.slots, i - lag - slot);
                let v = table.column_index(a, b);
                prop_assert!(v < table.len, "a pair's column box exists");
                for (window, lo, neg_hi) in [
                    (i, table.rows[a].0.to_vec(), table.rows[a].1.to_vec()),
                    (
                        j,
                        (0..K).map(|k| table.lo[k * table.len + v]).collect(),
                        (0..K).map(|k| table.neg_hi[k * table.len + v]).collect(),
                    ),
                ] {
                    let values = fix.c.window(window);
                    if values[0].is_nan() {
                        continue;
                    }
                    for k in 0..K {
                        prop_assert!(
                            lo[k] <= values[k] && values[k] <= -neg_hi[k],
                            "window {window} segment {k}"
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

proptest! {
    #![proptest_config(testkit::generators::seeded(40))]

    /// The void and edge masks are what a cell-by-cell enumeration of every block says, and a
    /// fresh ladder holds exactly its void blocks.
    #[test]
    fn the_masks_match_a_cell_by_cell_enumeration(s in shapes()) {
        match k_of(&s) {
            4 => check_masks::<4>(&s)?,
            8 => check_masks::<8>(&s)?,
            _ => check_masks::<16>(&s)?,
        }
    }

    /// Every pair's row and column box, at every layer, holds the pair's two windows: the column
    /// index and the ranges behind it put each pair in the box that bounds it.
    #[test]
    fn every_pair_is_inside_the_boxes_that_bound_it(s in shapes()) {
        match k_of(&s) {
            4 => check_boxes::<4>(&s)?,
            8 => check_boxes::<8>(&s)?,
            _ => check_boxes::<16>(&s)?,
        }
    }
}

// ---- the pass ----

/// Each true correlation of every valid pair, by definition, or `None` where a window ranks
/// nothing.
fn correlations<const K: usize>(fix: &Fixture<K>) -> Vec<Option<Vec<f64>>> {
    (0..fix.n)
        .map(|i| {
            let w = &fix.series[i..i + fix.m];
            if !fix.valid(i) || fix.constant(i) || w.iter().any(|x| !x.is_finite()) {
                return None;
            }
            let mean = w.iter().sum::<f64>() / fix.m as f64;
            let sd = (w.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / fix.m as f64).sqrt();
            Some(w.iter().map(|x| (x - mean) / sd).collect())
        })
        .collect()
}

fn rho(z: &[Option<Vec<f64>>], i: usize, j: usize) -> Option<f64> {
    let (a, b) = (z[i].as_ref()?, z[j].as_ref()?);
    Some(a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>() / a.len() as f64)
}

fn check_nesting<const K: usize>(s: &Shape) -> Result<(), TestCaseError> {
    let Some(mut case) = case::<K>(s) else {
        return Ok(());
    };
    let n = case.fix.n;
    let blocks = n.div_ceil(8) + 1;
    // The pool of floors that sit at the `f64` rule's own boundary.
    let mut pool = Vec::new();
    let all = entries(&case);
    for &(l, r, e) in all.iter().filter(|c| c.0 == LAYERS - 1).step_by(5) {
        let held = entry(&case, l, r, e);
        if held.cells.is_empty() {
            continue;
        }
        if let Some(rows) = case.fix.c.rows_span::<K>(held.rows.clone()) {
            let cols = case.fix.c.window_span::<K>(held.cols.clone());
            if cols.lo[0] < f32::INFINITY {
                pool.push(case.fix.c.value(&rows, &cols));
            }
        }
    }
    let rows = floors_from(&s.floors, &pool, blocks, 0);
    let cols = floors_from(&s.floors, &pool, blocks, 31);
    run(&mut case, &rows, &cols);
    let z = correlations(&case.fix);
    for &(l, r, e) in &all {
        if !skipped(&case, (l, r, e)) {
            continue;
        }
        let held = entry(&case, l, r, e);
        if held.cells.is_empty() {
            continue;
        }
        let rows_box = case.fix.c.rows_span::<K>(held.rows.clone());
        let cols_box = case.fix.c.window_span::<K>(held.cols.clone());
        let empty = rows_box.is_none() || cols_box.lo[0] == f32::INFINITY;
        let floors = held
            .rows
            .clone()
            .map(|i| rows[i / 8])
            .chain(held.cols.clone().map(|j| cols[j / 8]));
        for floor in floors {
            if empty {
                prop_assert!(!floor.is_nan(), "{:?} skipped under a NaN floor", (l, r, e));
            } else {
                let value = case.fix.c.value(rows_box.as_ref().unwrap(), &cols_box);
                prop_assert!(
                    value + DELTA < floor,
                    "{:?}: the f64 rule keeps it ({value} vs {floor})",
                    (l, r, e)
                );
            }
        }
        for &(i, j) in &held.cells {
            let floor = rows[i / 8].min(cols[j / 8]);
            if let Some(rho) = rho(&z, i, j) {
                prop_assert!(
                    rho <= floor + 1e-10,
                    "pair {:?} has rho {rho} over floor {floor}",
                    (i, j)
                );
            }
        }
    }
    Ok(())
}

proptest! {
    #![proptest_config(testkit::generators::seeded(60))]

    /// Every block the dense rule skips is skipped by the `f64` rule on the exact windows of its
    /// pairs, for every floor it reads, at every layer and segment count, with floors on the
    /// boundary, NaN, infinite and above one, flat runs and NaN gaps in the series, and every
    /// pair in it has a correlation below its floors.
    #[test]
    fn a_dense_skip_is_nested_in_the_f64_rule_and_sound(s in shapes()) {
        match k_of(&s) {
            4 => check_nesting::<4>(&s)?,
            8 => check_nesting::<8>(&s)?,
            _ => check_nesting::<16>(&s)?,
        }
    }
}

/// A one-sided run (no column floors): every block skipped is sound against the floors of its rows
/// alone (the f64 rule, and every pair's correlation), and every block skipped with column floors
/// is skipped without them, whatever those floors were.
fn check_one_sided<const K: usize>(s: &Shape) -> Result<(), TestCaseError> {
    let Some(mut case) = case::<K>(s) else {
        return Ok(());
    };
    let blocks = case.fix.n.div_ceil(8) + 1;
    let mut pool = Vec::new();
    let all = entries(&case);
    for &(l, r, e) in all.iter().filter(|c| c.0 == LAYERS - 1).step_by(5) {
        let held = entry(&case, l, r, e);
        if held.cells.is_empty() {
            continue;
        }
        if let Some(rows) = case.fix.c.rows_span::<K>(held.rows.clone()) {
            let cols = case.fix.c.window_span::<K>(held.cols.clone());
            if cols.lo[0] < f32::INFINITY {
                pool.push(case.fix.c.value(&rows, &cols));
            }
        }
    }
    let rows = floors_from(&s.floors, &pool, blocks, 0);
    let cols = floors_from(&s.floors, &pool, blocks, 31);
    run(&mut case, &rows, &cols);
    let two_sided: Vec<bool> = all.iter().map(|&c| skipped(&case, c)).collect();
    run_with(&mut case, &rows, None);
    let z = correlations(&case.fix);
    for (&(l, r, e), &was) in all.iter().zip(&two_sided) {
        let now = skipped(&case, (l, r, e));
        prop_assert!(
            now || !was,
            "{:?} skipped with column floors only",
            (l, r, e)
        );
        if !now {
            continue;
        }
        let held = entry(&case, l, r, e);
        if held.cells.is_empty() {
            continue;
        }
        let rows_box = case.fix.c.rows_span::<K>(held.rows.clone());
        let cols_box = case.fix.c.window_span::<K>(held.cols.clone());
        let empty = rows_box.is_none() || cols_box.lo[0] == f32::INFINITY;
        for floor in held.rows.clone().map(|i| rows[i / 8]) {
            if empty {
                // A block with a constant or non-finite window is kept under a NaN row floor.
                prop_assert!(!floor.is_nan(), "{:?} skipped under a NaN floor", (l, r, e));
            } else {
                let value = case.fix.c.value(rows_box.as_ref().unwrap(), &cols_box);
                prop_assert!(
                    value + DELTA < floor,
                    "{:?}: the f64 rule keeps it ({value} vs {floor})",
                    (l, r, e)
                );
            }
        }
        for &(i, j) in &held.cells {
            if let Some(rho) = rho(&z, i, j) {
                prop_assert!(
                    rho <= rows[i / 8] + 1e-10,
                    "pair {:?} has rho {rho} over its row floor {}",
                    (i, j),
                    rows[i / 8]
                );
            }
        }
    }
    Ok(())
}

proptest! {
    #![proptest_config(testkit::generators::seeded(60))]

    /// With no column floors (a one-sided run) a skip rests on the rows' floors alone and is sound
    /// against them, and it keeps every skip the column floors would have made.
    #[test]
    fn a_one_sided_skip_rests_on_the_row_floors_alone(s in shapes()) {
        match k_of(&s) {
            4 => check_one_sided::<4>(&s)?,
            8 => check_one_sided::<8>(&s)?,
            _ => check_one_sided::<16>(&s)?,
        }
    }
}

#[allow(clippy::needless_range_loop)]
fn check_ladder<const K: usize>(s: &Shape) -> Result<(), TestCaseError> {
    let Some(mut case) = case::<K>(s) else {
        return Ok(());
    };
    let blocks = case.fix.n.div_ceil(8) + 1;
    let random = (
        floors_from(&s.floors, &[], blocks, 0),
        floors_from(&s.floors, &[], blocks, 17),
    );
    // Random floors, then floors above one, where whole parents are proven and skip their children.
    for (rows, cols) in [random, (vec![2.0; blocks], vec![2.0; blocks])] {
        let work = run(&mut case, &rows, &cols);
        let mut expected = [0usize; LAYERS];
        for l in 0..LAYERS {
            let layer = &case.ladder.layers[l];
            for r in 0..layer.nrows {
                if layer.a0 + r >= case.boxes.layers[l].rows.len() {
                    continue;
                }
                for e in (0..layer.bits).step_by(16) {
                    let family = if l == 0 {
                        0
                    } else {
                        let parent = &case.ladder.layers[l - 1];
                        let pr = r / (parent.rows_per / layer.rows_per);
                        get_bits(parent.row(&parent.skip, pr), e / 4, 4)
                    };
                    if l > 0 {
                        for child in e..e + 16 {
                            let parent = &case.ladder.layers[l - 1];
                            let pr = r / (parent.rows_per / layer.rows_per);
                            if bit(parent.row(&parent.skip, pr), child / 4) {
                                prop_assert!(
                                    skipped(&case, (l, r, child)),
                                    "a child of a skipped parent is skipped"
                                );
                            }
                        }
                    }
                    if family != 0xF {
                        expected[l] += 16;
                    }
                }
            }
        }
        prop_assert_eq!(
            work.evaluated,
            expected,
            "only the children of surviving parents are read"
        );
    }
    for r in 0..case.ladder.finest_rows() {
        let layer = &case.ladder.layers[LAYERS - 1];
        let mut want = Vec::new();
        let mut e = 0;
        while e < layer.bits {
            if bit(layer.row(&layer.skip, r), e) {
                e += 1;
                continue;
            }
            let start = e;
            while e < layer.bits && !bit(layer.row(&layer.skip, r), e) {
                e += 1;
            }
            want.push(start..e);
        }
        let got: Vec<_> = case.ladder.survivors(r).collect();
        prop_assert_eq!(got, want);
    }
    Ok(())
}

proptest! {
    #![proptest_config(testkit::generators::seeded(60))]

    /// A child is read only under a surviving parent (the count of entries read is exactly the
    /// chunks with one), a skipped parent's children are skipped, and the survivors are the
    /// finest layer's runs of zero bits.
    #[test]
    fn the_ladder_reads_children_only_under_surviving_parents(s in shapes()) {
        match k_of(&s) {
            4 => check_ladder::<4>(&s)?,
            8 => check_ladder::<8>(&s)?,
            _ => check_ladder::<16>(&s)?,
        }
    }
}

fn check_constants<const K: usize>(s: &Shape) -> Result<(), TestCaseError> {
    let Some(mut case) = case::<K>(s) else {
        return Ok(());
    };
    let blocks = case.fix.n.div_ceil(8) + 1;
    // Every floor above one: the rule skips everything it is let to.
    run(&mut case, &vec![2.0; blocks], &vec![2.0; blocks]);
    let any_constant = (0..case.fix.n).any(|i| case.fix.constant(i));
    for &(l, r, e) in &entries(&case) {
        let held = entry(&case, l, r, e);
        let layer = &case.ladder.layers[l];
        let edge = bit(layer.row(&layer.edge, r), e);
        if held.cells.is_empty() {
            prop_assert!(skipped(&case, (l, r, e)));
            continue;
        }
        let holds = held
            .cells
            .iter()
            .any(|&(i, j)| case.fix.constant(i) || case.fix.constant(j));
        if skipped(&case, (l, r, e)) {
            prop_assert!(
                !holds,
                "{:?} holds a constant window and is skipped",
                (l, r, e)
            );
            prop_assert!(!edge, "a block with a cell that is no pair is skipped");
        } else if !any_constant {
            prop_assert!(edge, "{:?}: nothing keeps it", (l, r, e));
        }
    }
    // NaN floors: nothing is skipped but the blocks without a pair.
    run(&mut case, &vec![f64::NAN; blocks], &vec![f64::NAN; blocks]);
    for &(l, r, e) in &entries(&case) {
        let layer = &case.ladder.layers[l];
        prop_assert_eq!(skipped(&case, (l, r, e)), bit(layer.row(&layer.void, r), e));
    }
    Ok(())
}

proptest! {
    #![proptest_config(testkit::generators::seeded(60))]

    /// A block holding a constant window is never skipped, whatever its bound (floors above one
    /// skip everything else), and a NaN floor skips nothing.
    #[test]
    fn constants_and_nan_floors_are_never_skipped(s in shapes()) {
        match k_of(&s) {
            4 => check_constants::<4>(&s)?,
            8 => check_constants::<8>(&s)?,
            _ => check_constants::<16>(&s)?,
        }
    }

    /// Higher floors skip a superset, and a pass over parts of the slots equals one over all.
    #[test]
    #[allow(clippy::needless_range_loop)]
    fn higher_floors_skip_more_and_parts_equal_the_whole(s in shapes(), cut in 0usize..40, bump in 0.0f64..0.3) {
        let Some(mut whole) = case::<8>(&s) else { return Ok(()); };
        let blocks = whole.fix.n.div_ceil(8) + 1;
        let rows = floors_from(&s.floors, &[], blocks, 0);
        let cols = floors_from(&s.floors, &[], blocks, 9);
        run(&mut whole, &rows, &cols);
        let before: Vec<Vec<u64>> = whole.ladder.layers.iter().map(|l| l.skip.clone()).collect();
        // Parts, at a cut that is a multiple of the grain.
        let mut parts = case::<8>(&s).unwrap();
        let top = parts.ladder.bits;
        let at = (cut * GRAIN).min(top);
        let floors = Floors {
            rows: FloorBlocks { first: 0, floors: &rows },
            cols: Some(FloorBlocks { first: 0, floors: &cols }),
        };
        let sides = PaaSides::of(&parts.fix.c);
        for range in [0..at, at..top] {
            parts.ladder.refresh(&parts.boxes, sides, &floors, range.clone());
            parts.ladder.pass(&parts.boxes, range);
        }
        for (l, skip) in before.iter().enumerate() {
            prop_assert_eq!(&parts.ladder.layers[l].skip, skip, "layer {}", l);
        }
        // Rising floors only add skips.
        let rise = |f: &[f64]| -> Vec<f64> { f.iter().map(|&x| if x.is_nan() { x } else { x + bump }).collect() };
        run(&mut whole, &rise(&rows), &rise(&cols));
        for (l, skip) in before.iter().enumerate() {
            for (w, (&old, &new)) in skip.iter().zip(&whole.ladder.layers[l].skip).enumerate() {
                prop_assert_eq!(old & !new, 0, "layer {} word {} lost a skip", l, w);
            }
        }
        // A pass over the first grain leaves every later block as it was made.
        let mut few = case::<8>(&s).unwrap();
        let fresh: Vec<Vec<u64>> = few.ladder.layers.iter().map(|l| l.skip.clone()).collect();
        let sides = PaaSides::of(&few.fix.c);
        few.ladder.refresh(&few.boxes, sides, &floors, 0..GRAIN);
        few.ladder.pass(&few.boxes, 0..GRAIN);
        for l in 0..LAYERS {
            let layer = &few.ladder.layers[l];
            let scale = SLOTS[0] / layer.slots;
            for r in 0..layer.nrows {
                for e in GRAIN * scale..layer.bits {
                    prop_assert_eq!(bit(layer.row(&layer.skip, r), e), bit(layer.row(&fresh[l], r), e));
                }
            }
        }
    }
}

// ---- constants, empty boxes and non-finite windows by construction ----

fn walk(len: usize, seed: i64) -> Vec<f64> {
    let mut level = 0i64;
    (0..len)
        .map(|t| {
            level += (t as i64 * 7 + seed) % 5 - 2;
            level as f64
        })
        .collect()
}

/// 100 ticks of a walk, 200 of `middle`, 100 of a walk: m 16, so windows 100..=284 are all
/// inside the middle.
fn with_middle(middle: f64) -> Fixture<4> {
    let mut series = walk(100, 1);
    series.extend(core::iter::repeat_n(middle, 200));
    series.extend(walk(100, 3));
    fixture::<4>(series, 16)
}

/// A ladder over rows 208..224 and slots 0..16 of `fix`.
fn middle_case(fix: Fixture<4>) -> Case<4> {
    let band = Band {
        rows: 208..224,
        slots: 0..16,
    };
    let boxes = Boxes::try_new(PaaSides::of(&fix.c), fix.m, (fix.n, fix.lag, 16)).unwrap();
    let ladder = Ladder::try_new(&boxes, &band).unwrap();
    Case {
        fix,
        band,
        boxes,
        ladder,
    }
}

#[test]
fn a_block_of_only_constant_windows_has_an_empty_box_and_is_never_skipped() {
    let mut case = middle_case(with_middle(5.0));
    assert!((208..224).all(|i| case.fix.constant(i)));
    let blocks = case.fix.n.div_ceil(8) + 1;
    run(&mut case, &vec![2.0; blocks], &vec![2.0; blocks]);
    let finest = &case.ladder.layers[LAYERS - 1];
    // The rows' box and the columns' box are both empty: the proof alone would skip it.
    let table = &case.boxes.layers[LAYERS - 1];
    let v = table.column_index(finest.a0, finest.base);
    assert_eq!(table.rows[finest.a0].0, [f32::INFINITY; 4]);
    assert_eq!(table.lo[v], f32::INFINITY);
    let g = gaps(&table.rows[finest.a0], table, v, case.boxes.rule().margin());
    assert_eq!(g[0], f32::INFINITY, "the empty box has an infinite bound");
    assert!(!skipped(&case, (LAYERS - 1, 0, 0)), "and is kept");
    assert_eq!(finest.t_rows[0], f32::INFINITY, "by its threshold");
    assert!(!bit(finest.row(&finest.void, 0), 0));
    assert!(!bit(finest.row(&finest.edge, 0), 0));
}

#[test]
fn windows_that_are_not_finite_rank_nothing_and_are_skipped() {
    let mut case = middle_case(with_middle(f64::NAN));
    assert!((208..224).all(|i| !case.fix.valid(i) && !case.fix.constant(i)));
    let blocks = case.fix.n.div_ceil(8) + 1;
    run(&mut case, &vec![0.5; blocks], &vec![0.5; blocks]);
    assert!(
        skipped(&case, (LAYERS - 1, 0, 0)),
        "an empty box over windows that rank nothing"
    );
    // A floor that says nothing keeps it.
    run(&mut case, &vec![f64::NAN; blocks], &vec![f64::NAN; blocks]);
    assert!(!skipped(&case, (LAYERS - 1, 0, 0)));
    run(
        &mut case,
        &vec![f64::NEG_INFINITY; blocks],
        &vec![f64::NEG_INFINITY; blocks],
    );
    assert!(!skipped(&case, (LAYERS - 1, 0, 0)));
}

// ---- API edges ----

#[test]
fn the_boxes_and_the_ladder_refuse_what_they_do_not_cover() {
    let fix = fixture::<8>(walk(300, 2), 24);
    let sides = PaaSides::of(&fix.c);
    let (n, lag) = (fix.n, fix.lag);
    assert!(
        Boxes::<4>::try_new(sides, 24, (n, lag, 16)).is_none(),
        "the wrong segment count"
    );
    assert!(
        Boxes::<8>::try_new(sides, 24, (n + 1, lag, 16)).is_none(),
        "the wrong window count"
    );
    assert!(
        Boxes::<8>::try_new(sides, 24, (n, lag, 48)).is_none(),
        "a side that is not a power of two"
    );
    assert!(
        Boxes::<8>::try_new(sides, 24, (n, 0, 16)).is_none(),
        "no lag"
    );
    let boxes = Boxes::<8>::try_new(sides, 24, (n, lag, 32)).unwrap();
    let band = |rows: Range<usize>, slots: Range<usize>| Band { rows, slots };
    assert!(Ladder::try_new(&boxes, &band(40..64, 0..32)).is_some());
    assert!(
        Ladder::try_new(&boxes, &band(40..40, 0..32)).is_none(),
        "no rows"
    );
    assert!(
        Ladder::try_new(&boxes, &band(40..64, 5..5)).is_none(),
        "no slots"
    );
    assert!(
        Ladder::try_new(&boxes, &band(40..n + 1, 0..32)).is_none(),
        "rows past the series"
    );
    assert!(
        Ladder::try_new(&boxes, &band(40..64, 0..n + 1)).is_none(),
        "slots past the series"
    );
    assert!(
        Ladder::try_new(&boxes, &band(30..70, 0..32)).is_none(),
        "rows past the unit"
    );
}

#[test]
fn the_geometry_helpers_name_rows_and_slots() {
    let fix = fixture::<4>(walk(400, 4), 16);
    let boxes = Boxes::<4>::try_new(PaaSides::of(&fix.c), 16, (fix.n, fix.lag, 64)).unwrap();
    let band = Band {
        rows: 64..128,
        slots: 32..fix.n,
    };
    let ladder = Ladder::try_new(&boxes, &band).unwrap();
    assert_eq!(ladder.finest_rows(), 4);
    assert_eq!(ladder.rows_of(0), 64..80);
    assert_eq!(ladder.rows_of(3), 112..128);
    // The coarse layer starts at slot block 0 (slot 0), so finest bit 2 is slots 32..48.
    assert_eq!(ladder.slots_of(2..5), 32..80);
    assert_eq!(ladder.layers.len(), LAYERS);
    // The coarse bits: blocks of 256 slots from slot 0, padded to whole grains.
    assert_eq!(ladder.coarse_bits() % GRAIN, 0);
    assert!(ladder.coarse_bits() * SLOTS[0] >= fix.n);
    assert_eq!(ladder.coarse_slots(1..3), 256..768);
    // The finest bits that cover slots: a cut bit is included, and the result stays inside.
    assert_eq!(ladder.fine_bits(32..80), 2..5);
    assert_eq!(ladder.fine_bits(33..81), 2..6);
    assert_eq!(ladder.fine_bits(0..0), 0..0);
    let all = ladder.fine_bits(0..10_000);
    assert_eq!(
        all.end,
        ladder.coarse_bits() * (SLOTS[0] / SLOTS[LAYERS - 1])
    );
}

/// The survivors of a row block among a range of bits are the whole row's survivors cut at the
/// range's ends.
#[test]
fn survivors_in_a_range_are_the_rows_survivors_cut_at_its_ends() {
    let fix = fixture::<4>(walk(1100, 6), 16);
    let band = Band {
        rows: 832..896,
        slots: 0..700,
    };
    let boxes = Boxes::<4>::try_new(PaaSides::of(&fix.c), 16, (fix.n, fix.lag, 64)).unwrap();
    let mut ladder = Ladder::try_new(&boxes, &band).unwrap();
    let blocks = fix.n.div_ceil(8) + 1;
    let floors = vec![0.97; blocks];
    let floors = Floors {
        rows: FloorBlocks {
            first: 0,
            floors: &floors,
        },
        cols: Some(FloorBlocks {
            first: 0,
            floors: &floors,
        }),
    };
    let all = 0..ladder.coarse_bits();
    ladder.refresh(&boxes, PaaSides::of(&fix.c), &floors, all.clone());
    ladder.pass(&boxes, all);
    let fine = ladder.fine_bits(0..10_000).end;
    for r in 0..ladder.finest_rows() {
        let whole: Vec<_> = ladder.survivors(r).collect();
        for (from, to) in [(0, fine), (0, 7), (5, 30), (13, 14), (20, fine)] {
            let cut: Vec<_> = ladder.survivors_in(r, from..to).collect();
            let want: Vec<_> = whole
                .iter()
                .filter_map(|w| {
                    let (a, b) = (w.start.max(from), w.end.min(to));
                    (a < b).then_some(a..b)
                })
                .collect();
            assert_eq!(cut, want, "row block {r}, bits {from}..{to}");
        }
    }
}

#[test]
fn the_table_of_a_layer_serves_blocks_whose_rows_are_narrower_than_their_slots() {
    // At side 16 the layer of 256 slots has 16-row blocks: sixteen slot-block steps per box step.
    let fix = fixture::<4>(walk(400, 4), 16);
    let boxes = Boxes::<4>::try_new(PaaSides::of(&fix.c), 16, (fix.n, fix.lag, 16)).unwrap();
    let coarse = &boxes.layers[0];
    assert_eq!((coarse.rows_per, coarse.q), (16, 16));
    assert_eq!(coarse.column_index(3, 1) - coarse.column_index(3, 0), 16);
    let fine = &boxes.layers[LAYERS - 1];
    assert_eq!((fine.rows_per, fine.q), (16, 1));
    assert_eq!(fine.column_index(3, 1) - fine.column_index(3, 0), 1);
}

#[test]
fn a_proven_coarse_block_sets_every_block_under_it_and_nothing_below_is_read() {
    let fix = fixture::<4>(walk(1100, 5), 16);
    assert!((0..fix.n).all(|i| !fix.constant(i)));
    let band = Band {
        rows: 832..896,
        slots: 0..512,
    };
    let boxes = Boxes::<4>::try_new(PaaSides::of(&fix.c), 16, (fix.n, fix.lag, 64)).unwrap();
    let ladder = Ladder::try_new(&boxes, &band).unwrap();
    let mut case = Case {
        fix,
        band,
        boxes,
        ladder,
    };
    let blocks = case.fix.n.div_ceil(8) + 1;
    let work = run(&mut case, &vec![2.0; blocks], &vec![2.0; blocks]);
    assert_eq!(work.evaluated[0], 16, "the two coarse blocks of one row");
    assert_eq!(
        work.evaluated[1..],
        [0, 0],
        "nothing under a proven block is read"
    );
    for r in 0..case.ladder.finest_rows() {
        assert_eq!(case.ladder.survivors(r).count(), 0);
        let layer = &case.ladder.layers[LAYERS - 1];
        assert!((0..512 / 16).all(|e| bit(layer.row(&layer.skip, r), e)));
    }
}

// ---- the AB-join's geometry ----

/// A band of an AB-join of two series: `a`'s rows by slots `s` meeting column `i + n_b - 1 - s` of
/// `b`.
struct JoinCase<const K: usize> {
    fa: Fixture<K>,
    fb: Fixture<K>,
    band: Band,
    boxes: Boxes<K>,
    ladder: Ladder<K>,
}

/// The join case of two shapes at `K` segments (`t`'s series at `s`'s window length), `None` when
/// the picks leave no band.
fn join_case<const K: usize>(s: &Shape, t: &Shape) -> Option<JoinCase<K>> {
    let fa = fixture::<K>(series_of(s), s.m);
    let fb = fixture::<K>(
        series_of(&Shape {
            m: s.m,
            ..t.clone()
        }),
        s.m,
    );
    let (n_a, n_b) = (fa.n, fb.n);
    let slots = n_a + n_b - 1;
    let (side_pick, band_pick, off_pick, qa_pick, len_pick, _) = s.picks;
    let side = 16usize << side_pick;
    let row0 = (band_pick % n_a.div_ceil(side)) * side;
    let mut start = row0;
    if off_pick % 4 == 0 {
        start = row0 + off_pick * side / 8;
    }
    let end = (row0 + side).min(n_a);
    if start >= end {
        return None;
    }
    let qa = (qa_pick * 7 % (slots / 16 + 1)) * 16;
    let qb = (qa + 16 * (1 + len_pick * 3)).min(slots);
    if qa >= qb {
        return None;
    }
    let band = Band {
        rows: start..end,
        slots: qa..qb,
    };
    let sides = PaaSides {
        rows: &fa.c,
        cols: &fb.c,
    };
    let boxes = Boxes::try_new_join(sides, s.m, ((n_a, n_b), side)).expect("boxes");
    let ladder = Ladder::try_new(&boxes, &band).expect("ladder");
    Some(JoinCase {
        fa,
        fb,
        band,
        boxes,
        ladder,
    })
}

/// What a block holds, by enumeration over the join's cells.
fn join_entry<const K: usize>(case: &JoinCase<K>, l: usize, r: usize, e: usize) -> Entry {
    let layer = &case.ladder.layers[l];
    let n_b = case.fb.n;
    let rows = (layer.a0 + r) * layer.rows_per..(layer.a0 + r + 1) * layer.rows_per;
    let slots = (layer.base + e) * layer.slots..(layer.base + e + 1) * layer.slots;
    let (mut cells, mut total) = (Vec::new(), 0);
    for i in rows.clone() {
        for s in slots.clone() {
            total += 1;
            if case.band.rows.contains(&i) && case.band.slots.contains(&s) && i <= s && s < i + n_b
            {
                cells.push((i, i + n_b - 1 - s));
            }
        }
    }
    let span = |f: fn(&(usize, usize)) -> usize| {
        let lo = cells.iter().map(f).min().unwrap_or(0);
        let hi = cells.iter().map(f).max().unwrap_or(0);
        lo..hi + 1
    };
    Entry {
        all: cells.len() == total,
        rows: span(|c| c.0),
        cols: span(|c| c.1),
        cells,
    }
}

fn join_entries<const K: usize>(case: &JoinCase<K>) -> Vec<(usize, usize, usize)> {
    let mut all = Vec::new();
    for (l, layer) in case.ladder.layers.iter().enumerate() {
        for r in 0..layer.nrows {
            for e in 0..layer.bits {
                all.push((l, r, e));
            }
        }
    }
    all
}

fn join_skipped<const K: usize>(case: &JoinCase<K>, (l, r, e): (usize, usize, usize)) -> bool {
    let layer = &case.ladder.layers[l];
    bit(layer.row(&layer.skip, r), e)
}

/// The join's void and edge masks are what a cell-by-cell enumeration says, and a fresh ladder
/// holds exactly its void blocks.
fn check_join_masks<const K: usize>(s: &Shape, t: &Shape) -> Result<(), TestCaseError> {
    let Some(case) = join_case::<K>(s, t) else {
        return Ok(());
    };
    for (l, r, e) in join_entries(&case) {
        let layer = &case.ladder.layers[l];
        let held = join_entry(&case, l, r, e);
        let pairs = !held.cells.is_empty();
        prop_assert_eq!(
            bit(layer.row(&layer.void, r), e),
            !pairs,
            "void {:?}",
            (l, r, e)
        );
        prop_assert_eq!(
            bit(layer.row(&layer.edge, r), e),
            pairs && !held.all,
            "edge {:?}",
            (l, r, e)
        );
        prop_assert_eq!(
            join_skipped(&case, (l, r, e)),
            !pairs,
            "fresh {:?}",
            (l, r, e)
        );
    }
    Ok(())
}

/// Every pair of the join, at every layer, lies inside the row box and the column box that bound
/// it: the column index of its block and the ranges behind it.
fn check_join_boxes<const K: usize>(s: &Shape, t: &Shape) -> Result<(), TestCaseError> {
    let fa = fixture::<K>(series_of(s), s.m);
    let fb = fixture::<K>(
        series_of(&Shape {
            m: s.m,
            ..t.clone()
        }),
        s.m,
    );
    let (n_a, n_b) = (fa.n, fb.n);
    let side = 16usize << s.picks.0;
    let sides = PaaSides {
        rows: &fa.c,
        cols: &fb.c,
    };
    let boxes = Boxes::<K>::try_new_join(sides, s.m, ((n_a, n_b), side)).expect("boxes");
    for table in &boxes.layers {
        for i in 0..n_a {
            for j in 0..n_b {
                let slot = i + n_b - 1 - j;
                let (a, b) = (i / table.rows_per, slot / table.slots);
                let v = table.column_index(a, b);
                prop_assert!(v < table.len, "a pair's column box exists");
                for (c, window, lo, neg_hi) in [
                    (&fa.c, i, table.rows[a].0.to_vec(), table.rows[a].1.to_vec()),
                    (
                        &fb.c,
                        j,
                        (0..K).map(|k| table.lo[k * table.len + v]).collect(),
                        (0..K).map(|k| table.neg_hi[k * table.len + v]).collect(),
                    ),
                ] {
                    let values = c.window(window);
                    if values[0].is_nan() {
                        continue;
                    }
                    for k in 0..K {
                        prop_assert!(
                            lo[k] <= values[k] && values[k] <= -neg_hi[k],
                            "window {window} segment {k}"
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

/// Every block the join's ladder skips under row floors alone is skipped by the `f64` rule on
/// the exact windows of its pairs, and every pair in it has a correlation below its row's floor;
/// a block with a constant window is kept.
fn check_join_nesting<const K: usize>(s: &Shape, t: &Shape) -> Result<(), TestCaseError> {
    let Some(mut case) = join_case::<K>(s, t) else {
        return Ok(());
    };
    let blocks = case.fa.n.div_ceil(8) + 1;
    let all = join_entries(&case);
    let mut pool = Vec::new();
    for &(l, r, e) in all.iter().filter(|c| c.0 == LAYERS - 1).step_by(5) {
        let held = join_entry(&case, l, r, e);
        if held.cells.is_empty() {
            continue;
        }
        if let Some(rows) = case.fa.c.rows_span::<K>(held.rows.clone()) {
            let cols = case.fb.c.window_span::<K>(held.cols.clone());
            if cols.lo[0] < f32::INFINITY {
                pool.push(case.fa.c.value(&rows, &cols));
            }
        }
    }
    let rows = floors_from(&s.floors, &pool, blocks, 0);
    {
        let floors = Floors {
            rows: FloorBlocks {
                first: 0,
                floors: &rows,
            },
            cols: None,
        };
        let sides = PaaSides {
            rows: &case.fa.c,
            cols: &case.fb.c,
        };
        let every = 0..case.ladder.bits;
        case.ladder
            .refresh(&case.boxes, sides, &floors, every.clone());
        case.ladder.pass(&case.boxes, every);
    }
    let (za, zb) = (correlations(&case.fa), correlations(&case.fb));
    let rho = |i: usize, j: usize| {
        let (a, b) = (za[i].as_ref()?, zb[j].as_ref()?);
        Some(a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>() / a.len() as f64)
    };
    for &(l, r, e) in &all {
        if !join_skipped(&case, (l, r, e)) {
            continue;
        }
        let held = join_entry(&case, l, r, e);
        if held.cells.is_empty() {
            continue;
        }
        for &(i, j) in &held.cells {
            prop_assert!(
                !case.fa.constant(i) && !case.fb.constant(j),
                "{:?} skipped with a constant window in pair {:?}",
                (l, r, e),
                (i, j)
            );
        }
        let rows_box = case.fa.c.rows_span::<K>(held.rows.clone());
        let cols_box = case.fb.c.window_span::<K>(held.cols.clone());
        let empty = rows_box.is_none() || cols_box.lo[0] == f32::INFINITY;
        for floor in held.rows.clone().map(|i| rows[i / 8]) {
            if empty {
                prop_assert!(!floor.is_nan(), "{:?} skipped under a NaN floor", (l, r, e));
            } else {
                let value = case.fa.c.value(rows_box.as_ref().unwrap(), &cols_box);
                prop_assert!(
                    value + DELTA < floor,
                    "{:?}: the f64 rule keeps it ({value} vs {floor})",
                    (l, r, e)
                );
            }
        }
        for &(i, j) in &held.cells {
            if let Some(rho) = rho(i, j) {
                prop_assert!(
                    rho <= rows[i / 8] + 1e-10,
                    "pair {:?} has rho {rho} over its row floor {}",
                    (i, j),
                    rows[i / 8]
                );
            }
        }
    }
    Ok(())
}

proptest! {
    #![proptest_config(testkit::generators::seeded(40))]

    /// The void and edge masks of an AB-join's band are what a cell-by-cell enumeration of every
    /// block says, which is the self-join's rule with a row's slots `i..i + n_b`.
    #[test]
    fn the_joins_masks_match_a_cell_by_cell_enumeration(s in shapes(), t in shapes()) {
        match k_of(&s) {
            4 => check_join_masks::<4>(&s, &t)?,
            8 => check_join_masks::<8>(&s, &t)?,
            _ => check_join_masks::<16>(&s, &t)?,
        }
    }

    /// Every pair of an AB-join lies in the boxes that bound it, at every layer: one table of
    /// column boxes per layer, indexed by `q·(b + 1) + shift − a`, serves every band.
    #[test]
    fn every_pair_of_the_join_is_inside_the_boxes_that_bound_it(s in shapes(), t in shapes()) {
        match k_of(&s) {
            4 => check_join_boxes::<4>(&s, &t)?,
            8 => check_join_boxes::<8>(&s, &t)?,
            _ => check_join_boxes::<16>(&s, &t)?,
        }
    }
}

proptest! {
    #![proptest_config(testkit::generators::seeded(60))]

    /// An AB-join's skip rests on its rows' floors alone and is nested in the `f64` rule and
    /// sound against every pair's correlation, with floors on the boundary, NaN, infinite and above
    /// one, flat runs and NaN gaps on both sides; no block with a constant window is skipped.
    #[test]
    fn a_joins_skip_is_nested_in_the_f64_rule_and_sound(s in shapes(), t in shapes()) {
        match k_of(&s) {
            4 => check_join_nesting::<4>(&s, &t)?,
            8 => check_join_nesting::<8>(&s, &t)?,
            _ => check_join_nesting::<16>(&s, &t)?,
        }
    }
}

#[test]
fn the_join_boxes_refuse_what_they_do_not_cover() {
    let (fa, fb) = (
        fixture::<8>(walk(300, 2), 24),
        fixture::<8>(walk(340, 3), 24),
    );
    let sides = PaaSides {
        rows: &fa.c,
        cols: &fb.c,
    };
    assert!(Boxes::<8>::try_new_join(sides, 24, ((fa.n, fb.n), 16)).is_some());
    assert!(
        Boxes::<4>::try_new_join(sides, 24, ((fa.n, fb.n), 16)).is_none(),
        "the wrong segment count"
    );
    assert!(
        Boxes::<8>::try_new_join(sides, 24, ((fa.n, fb.n + 1), 16)).is_none(),
        "the wrong column count"
    );
    assert!(
        Boxes::<8>::try_new_join(sides, 24, ((fa.n + 1, fb.n), 16)).is_none(),
        "the wrong row count"
    );
    assert!(
        Boxes::<8>::try_new_join(sides, 24, ((fa.n, fb.n), 48)).is_none(),
        "a side that is not a power of two"
    );
    assert!(
        Boxes::<8>::try_new_join(sides, 24, ((fa.n, 0), 16)).is_none(),
        "no columns"
    );
    let boxes = Boxes::<8>::try_new_join(sides, 24, ((fa.n, fb.n), 32)).unwrap();
    let band = |rows: Range<usize>, slots: Range<usize>| Band { rows, slots };
    let slots = fa.n + fb.n - 1;
    assert!(Ladder::try_new(&boxes, &band(0..32, 0..slots)).is_some());
    assert!(
        Ladder::try_new(&boxes, &band(0..32, 0..slots + 1)).is_none(),
        "slots past the rectangle"
    );
    assert!(
        Ladder::try_new(&boxes, &band(fa.n - 8..fa.n + 1, 0..32)).is_none(),
        "rows past the series"
    );
}

/// `valid_cells_over` against a count of the cells of each shape, row by row.
#[test]
fn the_valid_cells_of_a_shape_match_an_enumeration() {
    use crate::batch::pass::Shape as Run;
    for shape in [
        Run::Triangle { exclusion: 3 },
        Run::Join { n_b: 1 },
        Run::Join { n_b: 5 },
        Run::Join { n_b: 40 },
    ] {
        for rows in [0..0, 0..7, 5..40, 30..31] {
            for slots in [0..0, 0..1, 3..9, 0..100, 20..60] {
                let counted = rows
                    .clone()
                    .map(|i| {
                        slots
                            .clone()
                            .filter(|&s| match shape {
                                Run::Triangle { exclusion } => s + exclusion < i,
                                Run::Join { n_b } => i <= s && s < i + n_b,
                            })
                            .count()
                    })
                    .sum::<usize>();
                assert_eq!(
                    super::valid_cells_over(shape, &rows, &slots),
                    counted,
                    "{shape:?} rows {rows:?} slots {slots:?}"
                );
            }
        }
    }
}
