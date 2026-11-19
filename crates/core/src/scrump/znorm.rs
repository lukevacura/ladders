//! `scrump`: the z-normalized anytime profile.

use super::{Anytime, DiagonalSource, Engine, Geometry, ScrumpError};
use crate::batch::{
    Cold, JoinSide, KeptList, Pairs, Quantized, SlotRuns, ab_join_over, group_by_row, has_neighbor,
    merge_kept, merged_two_sided, no_distance_cold, no_distance_cold_join, two_sided_sides_over,
};
use crate::capacity::reserved_vec;
use crate::entry::{Entry, TwoSidedTopk};
use crate::quantize::Scale;
use crate::spill::Reach;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

/// The sides' kept neighbors as the batch engines lay them out: per window, in order, a run of up
/// to `k` entries nearest first, or one empty entry.
type Sides = Vec<(u64, Entry)>;

/// The lists the anytime profile starts from for `reaches`, and the `k` they are kept at: one
/// empty entry per window of the `rows` windows, or with `cold`, every pair it ranks, ranked up
/// front. `cols` is the neighbors' series.
fn starting_lists<const N: usize>(
    cold: Option<&Cold>,
    reaches: [Reach; N],
    (rows, cols): (usize, &Quantized),
    (m, k): (NonZeroUsize, NonZeroUsize),
    threads: NonZeroUsize,
) -> Result<([KeptList; N], NonZeroUsize), ScrumpError> {
    let mut lists = [const { Vec::new() }; N];
    let Some(cold) = cold else {
        for list in &mut lists {
            *list = reserved_vec(rows as u128)?;
            list.extend((0..rows as u64).map(|row| (row, Entry::NONE)));
        }
        return Ok((lists, cols.clamp_k(k)));
    };
    let mut kept = k;
    for (list, reach) in lists.iter_mut().zip(reaches) {
        (*list, kept) = cold.ranked_alone(reach, (m, k), threads)?;
    }
    Ok((lists, kept))
}

/// What a z-normalized self-join keeps: both sides' up to `k` nearest neighbors per window over
/// the diagonals visited, and over every pair the cold route ranks.
struct ZSelf {
    quantized: Quantized,
    /// The windows only the cold route ranks, ranked when the profile is made.
    cold: Option<Cold>,
    m: NonZeroUsize,
    threads: NonZeroUsize,
    k: NonZeroUsize,
    left: Sides,
    right: Sides,
}

impl ZSelf {
    fn new(
        series: &[f64],
        m: NonZeroUsize,
        scale: Scale,
        threads: NonZeroUsize,
        k: NonZeroUsize,
    ) -> Result<Self, ScrumpError> {
        let mut quantized = Quantized::build(series, m, scale);
        let cold = no_distance_cold(series.into(), scale, &mut quantized, m)?;
        let reaches = [Reach::Left, Reach::Right];
        let windows = (series.len() + 1).saturating_sub(m.get());
        let pair = (windows, &quantized);
        let ([left, right], k) = starting_lists(cold.as_ref(), reaches, pair, (m, k), threads)?;
        Ok(Self {
            k,
            left,
            right,
            quantized,
            cold,
            m,
            threads,
        })
    }
}

impl Engine for ZSelf {
    fn visit(&mut self, runs: &SlotRuns) -> Result<(), ScrumpError> {
        let (new_left, new_right) =
            two_sided_sides_over(&self.quantized, self.m, self.threads, self.k, runs)?;
        let pairs = Pairs {
            rows: &self.quantized,
            cols: &self.quantized,
        };
        let merging = (pairs, self.cold.as_ref());
        let left = merge_kept(merging, self.m, self.k, &self.left, &new_left)?;
        let right = merge_kept(merging, self.m, self.k, &self.right, &new_right)?;
        (self.left, self.right) = (left, right);
        Ok(())
    }
}

/// The z-normalized anytime profile of one series against itself, STUMPY's `scrump`: after any
/// number of [`update`](Self::update)s, [`profile`](Self::profile) is the exact two-sided profile
/// ([`crate::batch::two_sided_topk_profile`]'s rows) over the diagonals visited so far and the
/// pairs ranked when it was made, on bits, and after the last one it is that profile whole. See
/// the [module](super) for the procedure, which pairs are ranked up front, and its differences from
/// STUMPY.
pub struct Scrump(Anytime<ZSelf>);

impl Scrump {
    /// The first diagonal and how many there are for a series of `len` samples at window length
    /// `m`, which a [`DiagonalSource`] orders; [`ScrumpError::NoDiagonals`] when there is none.
    pub fn diagonals(len: usize, m: NonZeroUsize) -> Result<(i64, usize), ScrumpError> {
        Geometry::self_join(len, m.get()).range()
    }

    /// Prepares `series` at window length `m`, quantized at `scale`, keeping each window's `k`
    /// nearest neighbors (`k` is limited to the number of finite windows), on up to `threads`
    /// threads. Visits the diagonals in the order `source` gives, in chunks of `percentage` of
    /// the distances each; nothing is visited until the first update. The pairs of a window with
    /// no distance in the hot passes are ranked here, before it returns (see the [module](super)).
    ///
    /// `percentage` must be in `(0, 1]`, and the series must have a diagonal to visit.
    pub fn new(
        series: &[f64],
        m: NonZeroUsize,
        scale: Scale,
        threads: NonZeroUsize,
        k: NonZeroUsize,
        percentage: f64,
        source: impl DiagonalSource,
    ) -> Result<Self, ScrumpError> {
        let geometry = Geometry::self_join(series.len(), m.get());
        geometry.range()?;
        let engine = ZSelf::new(series, m, scale, threads, k)?;
        Anytime::new(engine, geometry, percentage, source).map(Self)
    }

    /// Visits the next chunk of diagonals; nothing once [`is_complete`](Self::is_complete). An
    /// error leaves the profile as it was.
    pub fn update(&mut self) -> Result<(), ScrumpError> {
        self.0.update()
    }

    /// Whether every diagonal has been visited.
    pub fn is_complete(&self) -> bool {
        self.0.is_complete()
    }

    /// How many updates have run, at most [`chunks`](Self::chunks).
    pub fn updates(&self) -> usize {
        self.0.updates()
    }

    /// How many updates visit every diagonal.
    pub fn chunks(&self) -> usize {
        self.0.chunks()
    }

    /// The diagonals visited so far, in visiting order.
    pub fn visited(&self) -> &[i64] {
        self.0.visited()
    }

    /// The profile over the diagonals visited so far and the pairs ranked up front: per window
    /// start, in order, the nearest on each side and up to `k` nearest over both, nearest first,
    /// ties to the smaller index; a window with none of those pairs has none. `Err` when the rows
    /// are too large to allocate.
    pub fn profile(&self) -> Result<Vec<(u64, TwoSidedTopk)>, ScrumpError> {
        let engine = &self.0.engine;
        Ok(merged_two_sided(
            &engine.quantized,
            engine.cold.as_ref(),
            engine.m,
            engine.k,
            &engine.left,
            &engine.right,
        )?)
    }
}

/// What a z-normalized join keeps: each `a` window's up to `k` nearest `b` windows over the
/// diagonals visited, and over every pair the cold route ranks.
struct ZJoin {
    a: Quantized,
    b: Quantized,
    /// The windows of either series only the cold route ranks, ranked when the join is made.
    cold: Option<Cold>,
    m: NonZeroUsize,
    threads: NonZeroUsize,
    k: NonZeroUsize,
    rows: usize,
    kept: KeptList,
}

impl Engine for ZJoin {
    fn visit(&mut self, runs: &SlotRuns) -> Result<(), ScrumpError> {
        let new = ab_join_over(&self.a, &self.b, self.m, self.threads, self.k, runs)?;
        let pairs = Pairs {
            rows: &self.a,
            cols: &self.b,
        };
        let merging = (pairs, self.cold.as_ref());
        self.kept = merge_kept(merging, self.m, self.k, &self.kept, &new)?;
        Ok(())
    }
}

/// The z-normalized anytime join of one series against another, STUMPY's `scrump` with a `T_B`:
/// after any number of [`update`](Self::update)s, [`profile`](Self::profile) is the exact
/// top-`k` join ([`crate::batch::ab_join_topk`]'s rows) over the diagonals visited so far and
/// the pairs ranked when it was made (those of a window of either series with no distance in the
/// hot passes), on bits, and after the last one it is that join whole. A diagonal is a window of
/// `b` minus a window of `a`; there is no exclusion zone. See the [module](super) for the procedure
/// and its differences from STUMPY.
pub struct ScrumpJoin(Anytime<ZJoin>);

impl ScrumpJoin {
    /// The first diagonal and how many there are for series of `len_a` and `len_b` samples at
    /// window length `m`, which a [`DiagonalSource`] orders; [`ScrumpError::NoDiagonals`] when
    /// either has no window.
    pub fn diagonals(
        len_a: usize,
        len_b: usize,
        m: NonZeroUsize,
    ) -> Result<(i64, usize), ScrumpError> {
        Geometry::join(len_a, len_b, m.get()).range()
    }

    /// Prepares `a` and `b` at window length `m`, each quantized at its own scale, keeping each
    /// `a` window's `k` nearest `b` windows (`k` is limited to the finite windows of `b`).
    /// Everything else is as [`Scrump::new`].
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        a: &[f64],
        b: &[f64],
        m: NonZeroUsize,
        (scale_a, scale_b): (Scale, Scale),
        threads: NonZeroUsize,
        k: NonZeroUsize,
        percentage: f64,
        source: impl DiagonalSource,
    ) -> Result<Self, ScrumpError> {
        let geometry = Geometry::join(a.len(), b.len(), m.get());
        geometry.range()?;
        let rows = (a.len() + 1).saturating_sub(m.get());
        let (mut qa, mut qb) = (
            Quantized::build(a, m, scale_a),
            Quantized::build(b, m, scale_b),
        );
        let cold = no_distance_cold_join(
            JoinSide {
                series: a.into(),
                scale: scale_a,
                quantized: &mut qa,
            },
            JoinSide {
                series: b.into(),
                scale: scale_b,
                quantized: &mut qb,
            },
            m,
        )?;
        let pair = (rows, &qb);
        let ([kept], k) = starting_lists(cold.as_ref(), [Reach::Join], pair, (m, k), threads)?;
        let engine = ZJoin {
            a: qa,
            b: qb,
            cold,
            m,
            threads,
            k,
            rows,
            kept,
        };
        Anytime::new(engine, geometry, percentage, source).map(Self)
    }

    /// [`Scrump::update`].
    pub fn update(&mut self) -> Result<(), ScrumpError> {
        self.0.update()
    }

    /// [`Scrump::is_complete`].
    pub fn is_complete(&self) -> bool {
        self.0.is_complete()
    }

    /// [`Scrump::updates`].
    pub fn updates(&self) -> usize {
        self.0.updates()
    }

    /// [`Scrump::chunks`].
    pub fn chunks(&self) -> usize {
        self.0.chunks()
    }

    /// [`Scrump::visited`].
    pub fn visited(&self) -> &[i64] {
        self.0.visited()
    }

    /// The join over the diagonals visited so far and the pairs ranked up front: per window of
    /// `a`, in order, up to `k` entries nearest first, ties to the smaller index; none for a
    /// window with none of those pairs. `Err` when the rows are too large to allocate.
    pub fn profile(&self) -> Result<Vec<(u64, Vec<Entry>)>, ScrumpError> {
        let engine = &self.0.engine;
        let mut flat: KeptList = reserved_vec(engine.kept.len() as u128)?;
        flat.extend_from_slice(&engine.kept);
        Ok(group_by_row(flat, engine.k, engine.rows, has_neighbor))
    }
}
