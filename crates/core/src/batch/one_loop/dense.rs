//! The dense ladder: a bulk pass that proves blocks of a band's pairs skippable, layer by layer
//! from large blocks to small, into one bitmask per layer. A set bit means no pair of the block
//! needs computing (a proof against the floors, or no pair in it at all); the zero bits of the
//! finest layer are the survivors, as runs ([`SurvivorRuns`]).
//!
//! **Layers.** A layer is blocks of `S` slots by `min(S, side)` rows, `S` in [`SLOTS`] (256, 64,
//! 16) and `side` the band's rows, so a block never spans two bands, and a block of one layer is
//! four blocks of the next in slots (and one, two or four in rows). The finest rows are one leaf's
//! when the leaf is 16, so a run of surviving blocks of one row block is one call of the leaf
//! compute. Blocks are aligned: row block `a` is the rows `a·rb..(a+1)·rb`, slot block `b` the
//! slots `b·S..(b+1)·S`; pair `(i, s)` meets column `i − lag − s`. An AB-join band has the same
//! layers over its own geometry (`Shape::Join`): row `i` has the slots `i..i + n_b` and pair
//! `(i, s)` meets column `i + n_b − 1 − s` of the other series, which is the same table with the
//! signed lag `1 − n_b` ([`Boxes::try_new_join`]).
//!
//! **Boxes** ([`Boxes`]) are built once per run, not per band. The columns of a block are
//! `c0..c0 + rb + S − 1` with `c0 = rb·(a − q·(b + 1)) + 1 − lag` and `q = S/rb`, which depends on
//! `a − q·(b + 1)` alone, so one table of column boxes per layer serves every block of every
//! band, indexed by `v = q·(b + 1) + shift − a`. It grows by `q` with the slot block, so a row's
//! blocks read consecutive boxes (`q = 1`) or every `q`-th (the coarse layers of a narrow band,
//! few entries). A column box is the union of the PAA group boxes its range covers (a superset of
//! its valid windows), clipped to the series.
//!
//! **The rule.** Per segment `D = max(lo_c + nh_r, lo_r + nh_c)` and `kept = max(D − M, 0)`, all in
//! `f32`; a block's `G` is the sum of the `kept²` in any order (here the segments in order, a
//! lane per block). The block is skipped iff `G > T`, `T` the threshold of the lowest floor of its
//! rows and columns ([`Rule::threshold`]). `M` is `2⁻²¹·√m` in `f64` rounded to `f32` and one ulp
//! up: it pays for the `f32` storage of the coefficients and for the rounding of `D`, so every
//! kept gap is a true separation (`|e_i − e_j| ≥ g/(1 + 2⁻²⁴)`). The squares neither underflow
//! nor overflow (a kept gap is a multiple of `2⁻⁴⁴` and at most `3√m`), the sum of `K − 1`
//! additions in any order loses at most `(1 + 2⁻²⁴)^(K − 1)`, and Bessel's inequality gives
//! `2m(1 − ρ) ≥ G/(1 + (K + 3)·2⁻²⁴)`. `T` is `2m(1 − F + 2⁻⁴³)` raised by `1 + 2⁻¹⁹` (`K ≤ 16`,
//! `1 + 2⁻¹⁷` up to 64), rounded to `f32` and one ulp up, so a skip means every correlation of
//! the block is below `F`, and also that the `f64` value of
//! [`crate::bound_paa::Coefficients::value`] skips it (the slack and the margin make the `f32`
//! set a subset). An empty box has `G = +∞`, a NaN floor gives `T = +∞`, which nothing exceeds,
//! and a floor above one gives `T = −1`: every block skipped.
//!
//! **What is never skipped.** A block whose rows or columns hold a constant window has `T = +∞`
//! (constant windows are the constant route's), so even an empty box, every window constant,
//! stays. A block with a cell that is not a pair is kept too (the hypotenuse leaves hold fresh
//! lanes), and a block with no pair at all is set. Both are fixed by the band, not the floors.
//!
//! **The pass** reads 16 entries of one row block at once, over the `K` segments, only where one
//! of the four parents of the 16 survives; a skipped parent sets its four children unread.

use crate::batch::pass::Shape;
use crate::bound_paa::PaaSides;
use crate::capacity::reserved_vec;
use crate::kbest::lower_floor;
use alloc::vec::Vec;
use core::ops::Range;

/// Slots per block of each layer, coarse to fine; each is four times the next.
pub(super) const SLOTS: [usize; LAYERS] = [256, 64, 16];
pub(super) const LAYERS: usize = 3;
const RATIO: usize = 4;
/// Entries evaluated at once.
const CHUNK: usize = 16;
/// The coarse bits a refresh or a pass covers at a time: the children of four coarse blocks are
/// one chunk of the next layer.
pub(super) const GRAIN: usize = RATIO;
/// Windows per block of the floors a caller supplies.
const FLOOR_BLOCK: usize = 8;

/// `2⁻²¹`, the gap margin per `√m`.
const MARGIN_PER_ROOT: f64 = 1.0 / (1u64 << 21) as f64;
/// `2⁻⁴³`, added to `1 − F` before the threshold's inflation.
const SLACK: f64 = 1.0 / (1u64 << 43) as f64;

/// The margin and threshold constants of a run: functions of `m` and `K`.
#[derive(Clone, Copy, Debug)]
pub(super) struct Rule {
    margin: f32,
    two_m: f64,
    tau: f64,
}

impl Rule {
    /// The rule for windows of `m` ticks cut into `k` segments (`4 ≤ k ≤ 64`, `k ≤ m`).
    pub(super) fn new(m: usize, k: usize) -> Self {
        let tau = if k <= 16 {
            1.0 + 1.0 / (1u64 << 19) as f64
        } else {
            1.0 + 1.0 / (1u64 << 17) as f64
        };
        Self {
            margin: ((MARGIN_PER_ROOT * libm::sqrt(m as f64)) as f32).next_up(),
            two_m: 2.0 * m as f64,
            tau,
        }
    }

    /// The gap margin `M`.
    pub(super) fn margin(&self) -> f32 {
        self.margin
    }

    /// The threshold `T` of floor `floor`: a block with `G > T` has every correlation below the
    /// floor. `+∞` for a NaN floor, `−1` for a floor above one.
    pub(super) fn threshold(&self, floor: f64) -> f32 {
        if floor.is_nan() {
            return f32::INFINITY;
        }
        let t = ((1.0 - floor) + SLACK) * self.two_m;
        if t.is_nan() {
            return f32::INFINITY;
        }
        if t < 0.0 {
            return -1.0;
        }
        ((t * self.tau) as f32).next_up()
    }
}

/// The threshold of a side with no floor to beat (a one-sided run's columns): `-1`, the threshold
/// of a floor above one, which every gap exceeds.
const NO_FLOOR: f32 = -1.0;

/// Floors of one side per block of eight windows, as the band's snapshot holds them: block `b`
/// is the windows `first + 8b..first + 8b + 8`, `first` a multiple of eight.
#[derive(Clone, Copy, Debug)]
pub(super) struct FloorBlocks<'a> {
    pub(super) first: usize,
    pub(super) floors: &'a [f64],
}

impl FloorBlocks<'_> {
    /// The lowest floor over `windows`; NaN when a block is NaN, the range is empty or any of it
    /// lies outside the blocks held.
    pub(super) fn lowest(&self, windows: Range<usize>) -> f64 {
        if windows.start >= windows.end || windows.start < self.first {
            return f64::NAN;
        }
        let from = (windows.start - self.first) / FLOOR_BLOCK;
        let to = (windows.end - 1 - self.first) / FLOOR_BLOCK + 1;
        match self.floors.get(from..to) {
            Some(blocks) => blocks.iter().copied().fold(f64::INFINITY, lower_floor),
            None => f64::NAN,
        }
    }
}

/// The floors of a band's rows and, in a run that offers pairs to its columns, of its columns.
///
/// **One-sided runs.** A one-sided run (the left profile) offers the pair of row `i` and column
/// `j < i` to row `i` alone: no partial of column `j` ever takes it, so a column has no floor a
/// pair must beat, and a block is skippable iff every correlation in it is below the floors of its
/// *rows* alone. `cols` is `None` then, and a column's threshold is [`NO_FLOOR`] (`-1`, below
/// every gap, so a column never keeps a block), not `+inf` (which keeps every block) and not the
/// floor of a column table that holds none (`-inf`, which reads as the lowest floor and keeps
/// every block too). A column that holds a constant window is still kept: its pairs have no PAA
/// bound, so no block holding one is skipped.
#[derive(Clone, Copy, Debug)]
pub(super) struct Floors<'a> {
    pub(super) rows: FloorBlocks<'a>,
    pub(super) cols: Option<FloorBlocks<'a>>,
}

/// A band: its rows (a tile's, inside one unit of `side` rows) and its slots.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Band {
    pub(super) rows: Range<usize>,
    pub(super) slots: Range<usize>,
}

/// A vector of `len` copies of `value`, `None` when it cannot be allocated.
fn filled<T: Copy>(len: usize, value: T) -> Option<Vec<T>> {
    let mut v = reserved_vec(len as u128).ok()?;
    v.resize(len, value);
    Some(v)
}

/// One layer's boxes over a run: a row block's box for each row block, and a column box for each
/// column index `v`.
struct LayerBoxes<const K: usize> {
    slots: usize,
    rows_per: usize,
    q: usize,
    /// Added to the column index so it never goes below one.
    shift: usize,
    /// Row block `a`'s lows and negated highs (`+∞` both when none of its rows is valid).
    rows: Vec<([f32; K], [f32; K])>,
    /// Column boxes, segment-major: segment `k` of box `v` at `k·len + v`.
    len: usize,
    lo: Vec<f32>,
    neg_hi: Vec<f32>,
    /// The columns' windows (the series in a self-join, `b` in an AB-join).
    cols_n: usize,
    /// The run's shape, whose slot `s` of row `i` meets column `i − s − lag`.
    shape: Shape,
}

/// `lag` of a shape: slot `s` of row `i` meets column `i − s − lag`. The self-join's is its first
/// lag, `exclusion + 1`; the AB-join's is `1 − n_b` (negative: column `i + n_b − 1 − s`).
fn lag_of(shape: Shape) -> i128 {
    match shape {
        Shape::Triangle { exclusion } => exclusion as i128 + 1,
        Shape::Join { n_b } => 1 - n_b as i128,
    }
}

/// The slots row `row` has pairs in, as the pass lays them out.
pub(super) fn row_slots(shape: Shape, row: usize) -> Range<usize> {
    match shape {
        Shape::Triangle { exclusion } => 0..row.saturating_sub(exclusion),
        Shape::Join { n_b } => row..row.saturating_add(n_b),
    }
}

/// The pairs of `rows` by `slots` that exist in a run of `shape`.
pub(super) fn valid_cells_over(shape: Shape, rows: &Range<usize>, slots: &Range<usize>) -> usize {
    if slots.start >= slots.end {
        return 0;
    }
    rows.clone()
        .map(|row| {
            let held = row_slots(shape, row);
            held.end
                .min(slots.end)
                .saturating_sub(held.start.max(slots.start))
        })
        .sum()
}

impl<const K: usize> LayerBoxes<K> {
    fn build(
        sides: PaaSides<'_>,
        ((rows_n, cols_n), shape, side): ((usize, usize), Shape, usize),
        slots: usize,
    ) -> Option<Self> {
        let rows_per = slots.min(side);
        let row_blocks = rows_n.div_ceil(rows_per);
        let mut this = Self {
            slots,
            rows_per,
            q: slots / rows_per,
            shift: row_blocks.checked_add(side / rows_per)?,
            rows: reserved_vec(row_blocks as u128).ok()?,
            len: 0,
            lo: Vec::new(),
            neg_hi: Vec::new(),
            cols_n,
            shape,
        };
        for a in 0..row_blocks {
            let rows = a * rows_per..((a + 1) * rows_per).min(rows_n);
            this.rows.push(match sides.rows.rows_span::<K>(rows) {
                Some(span) => (span.lo, span.neg_hi),
                None => ([f32::INFINITY; K], [f32::INFINITY; K]),
            });
        }
        while this.start(this.len) + this.width() as i128 > 0 {
            this.len += 1;
        }
        let cells = this.len.checked_mul(K)?;
        this.lo = filled(cells, f32::INFINITY)?;
        this.neg_hi = filled(cells, f32::INFINITY)?;
        for v in 0..this.len {
            let range = this.column_range(v);
            if range.is_empty() {
                continue;
            }
            let span = sides.cols.columns_span::<K>(range);
            for k in 0..K {
                this.lo[k * this.len + v] = span.lo[k];
                this.neg_hi[k * this.len + v] = span.neg_hi[k];
            }
        }
        Some(this)
    }

    /// The windows a block's columns span: rows plus slots less one.
    fn width(&self) -> usize {
        self.rows_per + self.slots - 1
    }

    /// The first column of column index `v`, unclipped.
    fn start(&self, v: usize) -> i128 {
        self.rows_per as i128 * (self.shift as i128 - v as i128) + 1 - lag_of(self.shape)
    }

    /// The column index of row block `a` and slot block `b`.
    fn column_index(&self, a: usize, b: usize) -> usize {
        (self.q * (b + 1) + self.shift).saturating_sub(a)
    }

    /// The columns column index `v` covers, clipped to the series.
    fn column_range(&self, v: usize) -> Range<usize> {
        let clip = |x: i128| x.clamp(0, self.cols_n as i128) as usize;
        let start = self.start(v);
        clip(start)..clip(start + self.width() as i128)
    }
}

/// The boxes of every layer over a run, read-only once built.
pub(super) struct Boxes<const K: usize> {
    layers: Vec<LayerBoxes<K>>,
    rule: Rule,
    /// The rows of the run and its slots: the most a band may hold.
    rows: usize,
    slots: usize,
    side: usize,
}

impl<const K: usize> Boxes<K> {
    /// The boxes of a self-join of `windows` windows of `m` ticks with exclusion lag `lag`, for
    /// bands of `side` rows (a power of two). `None` when the coefficients do not match `K` or
    /// `windows`, the sizes are unusable, or the tables cannot be allocated.
    pub(super) fn try_new(
        sides: PaaSides<'_>,
        m: usize,
        (windows, lag, side): (usize, usize, usize),
    ) -> Option<Self> {
        if lag == 0 {
            return None;
        }
        let shape = Shape::Triangle { exclusion: lag - 1 };
        Self::over(sides, m, ((windows, windows), shape, side), windows)
    }

    /// The boxes of an AB-join of `n_a` rows by `n_b` columns (`sides.rows` over `a`'s windows,
    /// `sides.cols` over `b`'s), for bands of `side` rows: slot `s` of row `i` meets column
    /// `i + n_b - 1 - s`, so a block's columns depend on `a - q*(b + 1)` as in the self-join (its
    /// lag is `1 - n_b`) and one table per layer serves every band. `None` as [`Self::try_new`].
    pub(super) fn try_new_join(
        sides: PaaSides<'_>,
        m: usize,
        ((n_a, n_b), side): ((usize, usize), usize),
    ) -> Option<Self> {
        if n_b == 0 {
            return None;
        }
        let slots = n_a.checked_add(n_b - 1)?;
        Self::over(sides, m, ((n_a, n_b), Shape::Join { n_b }, side), slots)
    }

    fn over(
        sides: PaaSides<'_>,
        m: usize,
        (counts, shape, side): ((usize, usize), Shape, usize),
        slots: usize,
    ) -> Option<Self> {
        const { assert!(K >= 4 && K <= 64, "the rule covers 4 to 64 segments") };
        let (rows, cols) = counts;
        if sides.rows.segments().count() != K
            || sides.cols.segments().count() != K
            || sides.rows.windows() != rows
            || sides.cols.windows() != cols
            || !side.is_power_of_two()
            || m < K
        {
            return None;
        }
        let mut layers = Vec::new();
        layers.try_reserve_exact(LAYERS).ok()?;
        for layer_slots in SLOTS {
            layers.push(LayerBoxes::build(
                sides,
                (counts, shape, side),
                layer_slots,
            )?);
        }
        Some(Self {
            layers,
            rule: Rule::new(m, K),
            rows,
            slots,
            side,
        })
    }

    /// The rule the boxes' thresholds follow.
    #[cfg(test)]
    pub(super) fn rule(&self) -> Rule {
        self.rule
    }
}

/// `get_bits`: `len` (at most 16) bits of `words` from bit `pos`, zero past the end.
fn get_bits(words: &[u64], pos: usize, len: usize) -> u32 {
    let (w, sh) = (pos / 64, pos % 64);
    let mut v = words.get(w).copied().unwrap_or(0) >> sh;
    if sh + len > 64 {
        v |= words.get(w + 1).copied().unwrap_or(0) << (64 - sh);
    }
    (v & ((1u64 << len) - 1)) as u32
}

/// Writes the low `len` (at most 16) bits of `value` at bit `pos`; bits past the end are dropped.
fn put_bits(words: &mut [u64], pos: usize, len: usize, value: u32) {
    let (w, sh) = (pos / 64, pos % 64);
    let mask = (1u64 << len) - 1;
    let value = u64::from(value) & mask;
    if let Some(word) = words.get_mut(w) {
        *word = (*word & !(mask << sh)) | (value << sh);
    }
    if sh + len > 64
        && let Some(word) = words.get_mut(w + 1)
    {
        let up = 64 - sh;
        *word = (*word & !(mask >> up)) | (value >> up);
    }
}

/// Sets or clears the bits in `range`.
fn fill_bits(words: &mut [u64], range: Range<usize>, on: bool) {
    let mut pos = range.start;
    while pos < range.end {
        let (w, sh) = (pos / 64, pos % 64);
        let n = (64 - sh).min(range.end - pos);
        let mask = if n == 64 { !0 } else { ((1u64 << n) - 1) << sh };
        if let Some(word) = words.get_mut(w) {
            if on {
                *word |= mask;
            } else {
                *word &= !mask;
            }
        }
        pos += n;
    }
}

/// The first bit at or after `from`, below `bits`, that is `one` (set) or clear.
fn find_bit(words: &[u64], bits: usize, from: usize, one: bool) -> Option<usize> {
    if from >= bits {
        return None;
    }
    let flip = if one { 0 } else { !0 };
    let mut w = from / 64;
    let mut word = (words.get(w)? ^ flip) & (!0u64 << (from % 64));
    loop {
        if word != 0 {
            let at = w * 64 + word.trailing_zeros() as usize;
            return (at < bits).then_some(at);
        }
        w += 1;
        if w * 64 >= bits {
            return None;
        }
        word = words.get(w)? ^ flip;
    }
}

/// The maximal runs of clear bits of a mask of `bits` bits, as bit ranges in ascending order:
/// the survivors of a row block.
#[derive(Clone, Debug)]
pub(super) struct SurvivorRuns<'a> {
    words: &'a [u64],
    bits: usize,
    next: usize,
}

impl<'a> SurvivorRuns<'a> {
    /// The runs of clear bits among the first `bits` bits of `words`.
    #[cfg(test)]
    pub(super) fn new(words: &'a [u64], bits: usize) -> Self {
        Self::between(words, 0..bits)
    }

    /// The runs of clear bits among the bits `bits` of `words`, each cut at the range's ends.
    pub(super) fn between(words: &'a [u64], bits: Range<usize>) -> Self {
        Self {
            words,
            bits: bits.end,
            next: bits.start,
        }
    }
}

impl Iterator for SurvivorRuns<'_> {
    type Item = Range<usize>;

    fn next(&mut self) -> Option<Range<usize>> {
        let start = find_bit(self.words, self.bits, self.next, false)?;
        let end = find_bit(self.words, self.bits, start, true).unwrap_or(self.bits);
        self.next = end;
        Some(start..end)
    }
}

/// `EXPAND[n]`: the 16 child bits of the four parent bits `n`, each parent bit four times.
const EXPAND: [u32; 16] = {
    let mut table = [0u32; 16];
    let mut n = 0;
    while n < 16 {
        let mut bit = 0;
        while bit < 4 {
            if n >> bit & 1 == 1 {
                table[n] |= 0xF << (4 * bit);
            }
            bit += 1;
        }
        n += 1;
    }
    table
};

/// One chunk of a layer's table: `CHUNK` entries starting at index `v0`, `q` apart; an entry past
/// the table reads `+∞` (an empty box, or a threshold nothing exceeds).
#[inline(always)]
fn load(table: &[f32], offset: usize, len: usize, v0: usize, q: usize) -> [f32; CHUNK] {
    if q == 1
        && v0.saturating_add(CHUNK) <= len
        && let Some(src) = table
            .get(offset.saturating_add(v0)..offset.saturating_add(v0 + CHUNK))
            .and_then(|s| <&[f32; CHUNK]>::try_from(s).ok())
    {
        return *src;
    }
    load_edge(table, offset, len, v0, q)
}

/// [`load`] for a strided chunk or one that runs past the table.
#[inline(never)]
fn load_edge(table: &[f32], offset: usize, len: usize, v0: usize, q: usize) -> [f32; CHUNK] {
    let mut out = [f32::INFINITY; CHUNK];
    for (j, slot) in out.iter_mut().enumerate() {
        let v = v0.saturating_add(j.saturating_mul(q));
        if v < len
            && let Some(&x) = table.get(offset.saturating_add(v))
        {
            *slot = x;
        }
    }
    out
}

/// One segment's `kept²` added into `acc`, for `CHUNK` columns against one row block.
#[inline(always)]
fn step(
    acc: &mut [f32; CHUNK],
    cols: (&[f32; CHUNK], &[f32; CHUNK]),
    rows: (f32, f32),
    margin: f32,
) {
    let (lo_c, neg_hi_c) = cols;
    let (lo_r, neg_hi_r) = rows;
    for e in 0..CHUNK {
        let above = lo_c[e] + neg_hi_r;
        let below = lo_r + neg_hi_c[e];
        let d = if above > below { above } else { below };
        let g = d - margin;
        let kept = if g > 0.0 { g } else { 0.0 };
        acc[e] += kept * kept;
    }
}

/// `G` of row block box `row` against `CHUNK` column boxes from index `v0`, `q` apart.
#[inline(always)]
fn gaps<const K: usize>(
    row: &([f32; K], [f32; K]),
    table: &LayerBoxes<K>,
    v0: usize,
    margin: f32,
) -> [f32; CHUNK] {
    let mut acc = [0.0f32; CHUNK];
    for k in 0..K {
        let lo = load(&table.lo, k * table.len, table.len, v0, table.q);
        let neg_hi = load(&table.neg_hi, k * table.len, table.len, v0, table.q);
        step(&mut acc, (&lo, &neg_hi), (row.0[k], row.1[k]), margin);
    }
    acc
}

/// The entries of `g` above their thresholds (the larger of the row's and the column's), as bits.
#[inline(always)]
fn proven(g: &[f32; CHUNK], t_row: f32, t_cols: &[f32; CHUNK]) -> u32 {
    let mut bits = 0;
    for e in 0..CHUNK {
        let t = if t_row > t_cols[e] { t_row } else { t_cols[e] };
        bits |= u32::from(g[e] > t) << e;
    }
    bits
}

/// The slot blocks (absolute) of row block `r0..r0 + rb` of `s` slots that hold a pair, and the
/// ones whose every cell is a pair, for rows `rows` and slots `slots` of a run of `shape`. A row's
/// slots are an interval that moves up with the row, so a block's rows have one interval between
/// them (from the first row's start to the last row's end) and every row has the slots from the
/// last row's start to the first row's end.
fn pair_blocks(
    (r0, rb, s): (usize, usize, usize),
    rows: &Range<usize>,
    slots: &Range<usize>,
    shape: Shape,
) -> (Range<usize>, Range<usize>) {
    let last = (r0 + rb).min(rows.end);
    let first = r0.max(rows.start);
    let some = if first < last {
        let lo = row_slots(shape, first).start.max(slots.start);
        let hi = row_slots(shape, last - 1).end.min(slots.end);
        if lo < hi {
            lo / s..hi.div_ceil(s)
        } else {
            0..0
        }
    } else {
        0..0
    };
    let all = if r0 >= rows.start && r0 + rb <= rows.end {
        let lo = row_slots(shape, r0 + rb - 1).start.max(slots.start);
        let hi = row_slots(shape, r0).end.min(slots.end);
        if lo < hi {
            let bottom = lo.div_ceil(s);
            bottom..(hi / s).max(bottom)
        } else {
            0..0
        }
    } else {
        0..0
    };
    let bottom = all.start.max(some.start);
    (some.clone(), bottom..all.end.min(some.end).max(bottom))
}

/// What the pass evaluated, in entries per layer (counted in chunks).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Work {
    pub(super) evaluated: [usize; LAYERS],
}

/// One layer of a band: its masks and thresholds.
struct Layer {
    slots: usize,
    rows_per: usize,
    q: usize,
    /// The first row block (absolute) and how many a band holds.
    a0: usize,
    nrows: usize,
    /// The first slot block (absolute) and how many bits a row has, in words of 64.
    base: usize,
    bits: usize,
    words: usize,
    /// The first column index the thresholds hold.
    v_min: usize,
    /// Row `r`'s bits at `r·words`: set where the block needs no compute.
    skip: Vec<u64>,
    /// Set where the block has no pair, and where it has a cell that is not a pair.
    void: Vec<u64>,
    edge: Vec<u64>,
    t_rows: Vec<f32>,
    t_cols: Vec<f32>,
}

impl Layer {
    fn new<const K: usize>(
        boxes: &Boxes<K>,
        table: &LayerBoxes<K>,
        band: &Band,
        (base0, bits0): (usize, usize),
    ) -> Option<Self> {
        let scale = SLOTS[0] / table.slots;
        let (base, bits) = (base0 * scale, bits0 * scale);
        let words = bits.div_ceil(64);
        let nrows = boxes.side / table.rows_per;
        let a0 = band.rows.start / boxes.side * boxes.side / table.rows_per;
        let cells = nrows.checked_mul(words)?;
        let mut void = filled(cells, !0u64)?;
        let mut edge = filled(cells, 0u64)?;
        for r in 0..nrows {
            let (some, all) = pair_blocks(
                ((a0 + r) * table.rows_per, table.rows_per, table.slots),
                &band.rows,
                &band.slots,
                table.shape,
            );
            let bits_of = |blocks: Range<usize>| {
                blocks.start.saturating_sub(base).min(bits)
                    ..blocks.end.saturating_sub(base).min(bits)
            };
            let row = r * words..(r + 1) * words;
            fill_bits(void.get_mut(row.clone())?, bits_of(some.clone()), false);
            let edge_row = edge.get_mut(row)?;
            fill_bits(edge_row, bits_of(some), true);
            fill_bits(edge_row, bits_of(all), false);
        }
        let v_min = table.column_index(a0 + nrows - 1, base);
        let v_max = table.column_index(a0, base + bits - 1);
        let mut skip = reserved_vec(cells as u128).ok()?;
        skip.extend_from_slice(&void);
        Some(Self {
            slots: table.slots,
            rows_per: table.rows_per,
            q: table.q,
            a0,
            nrows,
            base,
            bits,
            words,
            v_min,
            skip,
            void,
            edge,
            t_rows: filled(nrows, f32::INFINITY)?,
            t_cols: filled(v_max.checked_sub(v_min)? + 1, f32::INFINITY)?,
        })
    }

    fn row<'a>(&self, masks: &'a [u64], r: usize) -> &'a [u64] {
        masks
            .get(r * self.words..(r + 1) * self.words)
            .unwrap_or(&[])
    }

    /// Recomputes the thresholds of the row blocks and of the column boxes the slot blocks `bits`
    /// read, from `floors`.
    fn refresh<const K: usize>(
        &mut self,
        table: &LayerBoxes<K>,
        rule: Rule,
        (sides, floors, rows): (PaaSides<'_>, &Floors<'_>, &Range<usize>),
        bits: Range<usize>,
    ) {
        for r in 0..self.nrows {
            let a = self.a0 + r;
            let range =
                (a * self.rows_per).max(rows.start)..((a + 1) * self.rows_per).min(rows.end);
            let t = if range.is_empty() || sides.rows.holds_constant(range.clone()) {
                f32::INFINITY
            } else {
                rule.threshold(floors.rows.lowest(range))
            };
            if let Some(slot) = self.t_rows.get_mut(r) {
                *slot = t;
            }
        }
        let low = table.column_index(self.a0 + self.nrows - 1, self.base + bits.start);
        let high = table.column_index(self.a0, self.base + bits.end - 1);
        for v in low..=high {
            let range = table.column_range(v);
            let t = if range.is_empty() || sides.cols.holds_constant(range.clone()) {
                f32::INFINITY
            } else {
                floors
                    .cols
                    .map_or(NO_FLOOR, |cols| rule.threshold(cols.lowest(range)))
            };
            if let Some(slot) = v
                .checked_sub(self.v_min)
                .and_then(|i| self.t_cols.get_mut(i))
            {
                *slot = t;
            }
        }
    }

    /// Evaluates the slot blocks `bits` of every row block of the layer under the parent layer's
    /// masks (none for the coarsest); returns the entries evaluated.
    fn pass<const K: usize>(
        &mut self,
        table: &LayerBoxes<K>,
        margin: f32,
        parent: Option<&Layer>,
        bits: Range<usize>,
    ) -> usize {
        let mut evaluated = 0;
        for r in 0..self.nrows {
            let a = self.a0 + r;
            let Some(row) = table.rows.get(a) else {
                continue;
            };
            let t_row = self.t_rows.get(r).copied().unwrap_or(f32::INFINITY);
            let parent_row = parent.map(|p| p.row(&p.skip, r / (p.rows_per / self.rows_per)));
            let mut e = bits.start;
            while e < bits.end {
                let n = CHUNK.min(bits.end - e);
                let family = parent_row.map_or(0, |p| get_bits(p, e / RATIO, RATIO));
                let mut set = EXPAND[family as usize];
                if family != 0xF {
                    let v0 = table.column_index(a, self.base + e);
                    let g = gaps(row, table, v0, margin);
                    let t_cols = load(
                        &self.t_cols,
                        0,
                        self.t_cols.len(),
                        v0.saturating_sub(self.v_min),
                        self.q,
                    );
                    set |= proven(&g, t_row, &t_cols);
                    evaluated += CHUNK;
                }
                let at = r * self.words;
                let keep = get_bits(self.row(&self.edge, r), e, n);
                let none = get_bits(self.row(&self.void, r), e, n);
                if let Some(words) = self.skip.get_mut(at..at + self.words) {
                    put_bits(words, e, n, (set & !keep) | none);
                }
                e += n;
            }
        }
        evaluated
    }
}

/// A band's ladder: the masks and thresholds of every layer.
pub(super) struct Ladder<const K: usize> {
    layers: Vec<Layer>,
    rows: Range<usize>,
    /// Coarse bits per row (a multiple of [`GRAIN`]).
    bits: usize,
}

impl<const K: usize> Ladder<K> {
    /// The ladder of `band`, nothing evaluated: every block with a pair reads as a survivor.
    /// `None` when the band is not inside one unit of the boxes' `side` rows and the series, or
    /// when its memory cannot be allocated.
    pub(super) fn try_new(boxes: &Boxes<K>, band: &Band) -> Option<Self> {
        let side = boxes.side;
        if band.rows.is_empty()
            || band.slots.is_empty()
            || band.rows.end > boxes.rows
            || band.slots.end > boxes.slots
            || band.rows.end - band.rows.start / side * side > side
        {
            return None;
        }
        let top = SLOTS[0];
        let base = band.slots.start / top;
        let bits = (band.slots.end.div_ceil(top) - base).next_multiple_of(GRAIN);
        let mut layers = Vec::new();
        layers.try_reserve_exact(LAYERS).ok()?;
        for table in &boxes.layers {
            layers.push(Layer::new(boxes, table, band, (base, bits))?);
        }
        Some(Self {
            layers,
            rows: band.rows.clone(),
            bits,
        })
    }

    /// `coarse` (coarse-layer bits) widened to whole multiples of [`GRAIN`] inside the band.
    fn aligned(&self, coarse: Range<usize>) -> Range<usize> {
        let end = coarse
            .end
            .min(self.bits)
            .next_multiple_of(GRAIN)
            .min(self.bits);
        let start = coarse.start.min(end) / GRAIN * GRAIN;
        start..end
    }

    /// Recomputes the thresholds of the slot blocks `coarse` of the coarsest layer and of
    /// everything below them from `floors` (widened to whole multiples of [`GRAIN`]).
    pub(super) fn refresh(
        &mut self,
        boxes: &Boxes<K>,
        sides: PaaSides<'_>,
        floors: &Floors<'_>,
        coarse: Range<usize>,
    ) {
        let range = self.aligned(coarse);
        if range.is_empty() {
            return;
        }
        for (layer, table) in self.layers.iter_mut().zip(&boxes.layers) {
            let scale = SLOTS[0] / layer.slots;
            let bits = range.start * scale..range.end * scale;
            layer.refresh(table, boxes.rule, (sides, floors, &self.rows), bits);
        }
    }

    /// Evaluates the slot blocks `coarse` (widened as [`Self::refresh`]'s are), coarsest layer
    /// first, a block's children only under a surviving parent, and sets the masks.
    pub(super) fn pass(&mut self, boxes: &Boxes<K>, coarse: Range<usize>) -> Work {
        let mut work = Work::default();
        let range = self.aligned(coarse);
        if range.is_empty() {
            return work;
        }
        for (l, table) in boxes.layers.iter().enumerate() {
            let (done, rest) = self.layers.split_at_mut(l);
            let Some(layer) = rest.first_mut() else {
                break;
            };
            let scale = SLOTS[0] / layer.slots;
            let bits = range.start * scale..range.end * scale;
            if let Some(count) = work.evaluated.get_mut(l) {
                *count = layer.pass(table, boxes.rule.margin(), done.last(), bits);
            }
        }
        work
    }

    /// The finest layer's row blocks, and how many slot blocks of 16 a row has.
    pub(super) fn finest_rows(&self) -> usize {
        self.layers.last().map_or(0, |l| l.nrows)
    }

    /// The rows of finest row block `r`.
    pub(super) fn rows_of(&self, r: usize) -> Range<usize> {
        self.layers.last().map_or(0..0, |l| {
            let start = (l.a0 + r) * l.rows_per;
            start..start + l.rows_per
        })
    }

    /// The slots of finest bits `bits`.
    pub(super) fn slots_of(&self, bits: Range<usize>) -> Range<usize> {
        self.layers.last().map_or(0..0, |l| {
            (l.base + bits.start) * l.slots..(l.base + bits.end) * l.slots
        })
    }

    /// The coarse bits per row: the blocks of the coarsest layer the band's slots touch, in whole
    /// multiples of [`GRAIN`].
    pub(super) fn coarse_bits(&self) -> usize {
        self.bits
    }

    /// The slots of the coarse bits `bits`.
    pub(super) fn coarse_slots(&self, bits: Range<usize>) -> Range<usize> {
        self.layers.first().map_or(0..0, |l| {
            (l.base + bits.start) * l.slots..(l.base + bits.end) * l.slots
        })
    }

    /// The finest bits that cover the slots `slots`, a bit that `slots` cuts included, kept inside
    /// the band's bits.
    pub(super) fn fine_bits(&self, slots: Range<usize>) -> Range<usize> {
        self.layers.last().map_or(0..0, |l| {
            let first = l.base * l.slots;
            let bit = |slot: usize| slot.saturating_sub(first);
            let start = (bit(slots.start) / l.slots).min(l.bits);
            let end = bit(slots.end).div_ceil(l.slots).min(l.bits);
            start..end.max(start)
        })
    }

    /// The survivors of finest row block `r` among the finest bits `bits`: runs of zero bits.
    pub(super) fn survivors_in(&self, r: usize, bits: Range<usize>) -> SurvivorRuns<'_> {
        match self.layers.last() {
            Some(l) => SurvivorRuns::between(l.row(&l.skip, r), bits.start..bits.end.min(l.bits)),
            None => SurvivorRuns::between(&[], 0..0),
        }
    }

    /// The survivors of finest row block `r`: runs of zero bits.
    #[cfg(test)]
    pub(super) fn survivors(&self, r: usize) -> SurvivorRuns<'_> {
        self.survivors_in(r, 0..self.layers.last().map_or(0, |l| l.bits))
    }
}

#[cfg(test)]
mod tests;
