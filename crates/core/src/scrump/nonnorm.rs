//! `scraamp`: the non-normalized anytime profile.

use super::{Anytime, DiagonalSource, Engine, Geometry, ScrumpError};
use crate::batch::SlotRuns;
use crate::batch::nonnorm::{TickSeries, ab_join_over, two_sided_over};
use crate::capacity::reserved_vec;
use crate::nonnorm::{NonNormNeighbor, NonNormTwoSided};
use crate::quantize::Scale;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

/// What a non-normalized self-join keeps: each window's nearest neighbor on each side over the
/// diagonals visited.
struct NSelf {
    ticks: TickSeries,
    m: NonZeroUsize,
    threads: NonZeroUsize,
    left: Vec<NonNormNeighbor>,
    right: Vec<NonNormNeighbor>,
}

impl NSelf {
    fn new(
        series: &[f64],
        m: NonZeroUsize,
        scale: Scale,
        threads: NonZeroUsize,
    ) -> Result<Self, ScrumpError> {
        let windows = (series.len() + 1).saturating_sub(m.get());
        let nothing = || -> Result<Vec<NonNormNeighbor>, ScrumpError> {
            let mut sides: Vec<NonNormNeighbor> = reserved_vec(windows as u128)?;
            sides.resize(windows, NonNormNeighbor::NONE);
            Ok(sides)
        };
        Ok(Self {
            left: nothing()?,
            right: nothing()?,
            ticks: TickSeries::build(series, m, scale),
            m,
            threads,
        })
    }
}

impl Engine for NSelf {
    fn visit(&mut self, runs: &SlotRuns) -> Result<(), ScrumpError> {
        let (new_left, new_right) = two_sided_over(&self.ticks, self.m, self.threads, runs);
        for (kept, new) in [(&mut self.left, new_left), (&mut self.right, new_right)] {
            for (kept, new) in kept.iter_mut().zip(new) {
                *kept = kept.nearer(new);
            }
        }
        Ok(())
    }
}

/// The non-normalized anytime profile of one series against itself, STUMPY's `scraamp` at
/// `p = 2`: after any number of [`update`](Self::update)s, [`profile`](Self::profile) is the exact
/// non-normalized two-sided profile ([`crate::batch::nonnorm::two_sided_profile`]'s rows) over the
/// diagonals visited so far, and after the last one it is that profile whole. Only the nearest
/// neighbor is kept (`k` of one). See the [module](super) for the procedure and its differences
/// from STUMPY.
pub struct Scraamp(Anytime<NSelf>);

impl Scraamp {
    /// [`Scrump::diagonals`](super::Scrump::diagonals).
    pub fn diagonals(len: usize, m: NonZeroUsize) -> Result<(i64, usize), ScrumpError> {
        Geometry::self_join(len, m.get()).range()
    }

    /// [`Scrump::new`](super::Scrump::new) for squared Euclidean distance on ticks of `scale`,
    /// keeping each window's one nearest neighbor.
    pub fn new(
        series: &[f64],
        m: NonZeroUsize,
        scale: Scale,
        threads: NonZeroUsize,
        percentage: f64,
        source: impl DiagonalSource,
    ) -> Result<Self, ScrumpError> {
        let geometry = Geometry::self_join(series.len(), m.get());
        geometry.range()?;
        let engine = NSelf::new(series, m, scale, threads)?;
        Anytime::new(engine, geometry, percentage, source).map(Self)
    }

    /// [`Scrump::update`](super::Scrump::update).
    pub fn update(&mut self) -> Result<(), ScrumpError> {
        self.0.update()
    }

    /// [`Scrump::is_complete`](super::Scrump::is_complete).
    pub fn is_complete(&self) -> bool {
        self.0.is_complete()
    }

    /// [`Scrump::updates`](super::Scrump::updates).
    pub fn updates(&self) -> usize {
        self.0.updates()
    }

    /// [`Scrump::chunks`](super::Scrump::chunks).
    pub fn chunks(&self) -> usize {
        self.0.chunks()
    }

    /// [`Scrump::visited`](super::Scrump::visited).
    pub fn visited(&self) -> &[i64] {
        self.0.visited()
    }

    /// The profile over the diagonals visited so far: per window start, in order, the nearest on
    /// each side and the nearer of the two (the left one on equal distances); a window with no
    /// visited pair has none. `Err` when the rows are too large to allocate.
    pub fn profile(&self) -> Result<Vec<(u64, NonNormTwoSided)>, ScrumpError> {
        let engine = &self.0.engine;
        let mut rows = reserved_vec(engine.left.len() as u128)?;
        for (row, (&left, &right)) in engine.left.iter().zip(&engine.right).enumerate() {
            let nearest = left.nearer(right);
            let sides = NonNormTwoSided {
                left,
                right,
                nearest,
            };
            rows.push((row as u64, sides));
        }
        Ok(rows)
    }
}

/// What a non-normalized join keeps: each `a` window's nearest `b` window over the diagonals
/// visited.
struct NJoin {
    a: TickSeries,
    b: TickSeries,
    m: NonZeroUsize,
    threads: NonZeroUsize,
    nearest: Vec<NonNormNeighbor>,
}

impl Engine for NJoin {
    fn visit(&mut self, runs: &SlotRuns) -> Result<(), ScrumpError> {
        let new = ab_join_over(&self.a, &self.b, self.m, self.threads, runs);
        for (kept, new) in self.nearest.iter_mut().zip(new) {
            *kept = kept.nearer(new);
        }
        Ok(())
    }
}

/// The non-normalized anytime join of one series against another, STUMPY's `scraamp` with a `T_B`
/// at `p = 2`: after any number of [`update`](Self::update)s, [`profile`](Self::profile) is the
/// exact non-normalized join ([`crate::batch::nonnorm::ab_join`]'s rows) over the diagonals
/// visited so far, and after the last one it is that join whole. Both series are quantized at
/// one `scale`. See the [module](super) for the procedure and its differences from STUMPY.
pub struct ScraampJoin(Anytime<NJoin>);

impl ScraampJoin {
    /// [`ScrumpJoin::diagonals`](super::ScrumpJoin::diagonals).
    pub fn diagonals(
        len_a: usize,
        len_b: usize,
        m: NonZeroUsize,
    ) -> Result<(i64, usize), ScrumpError> {
        Geometry::join(len_a, len_b, m.get()).range()
    }

    /// [`ScrumpJoin::new`](super::ScrumpJoin::new) for squared Euclidean distance on ticks of
    /// `scale`, keeping each window's one nearest neighbor.
    pub fn new(
        a: &[f64],
        b: &[f64],
        m: NonZeroUsize,
        scale: Scale,
        threads: NonZeroUsize,
        percentage: f64,
        source: impl DiagonalSource,
    ) -> Result<Self, ScrumpError> {
        let geometry = Geometry::join(a.len(), b.len(), m.get());
        geometry.range()?;
        let rows = (a.len() + 1).saturating_sub(m.get());
        let mut nearest: Vec<NonNormNeighbor> = reserved_vec(rows as u128)?;
        nearest.resize(rows, NonNormNeighbor::NONE);
        let engine = NJoin {
            a: TickSeries::build(a, m, scale),
            b: TickSeries::build(b, m, scale),
            m,
            threads,
            nearest,
        };
        Anytime::new(engine, geometry, percentage, source).map(Self)
    }

    /// [`Scrump::update`](super::Scrump::update).
    pub fn update(&mut self) -> Result<(), ScrumpError> {
        self.0.update()
    }

    /// [`Scrump::is_complete`](super::Scrump::is_complete).
    pub fn is_complete(&self) -> bool {
        self.0.is_complete()
    }

    /// [`Scrump::updates`](super::Scrump::updates).
    pub fn updates(&self) -> usize {
        self.0.updates()
    }

    /// [`Scrump::chunks`](super::Scrump::chunks).
    pub fn chunks(&self) -> usize {
        self.0.chunks()
    }

    /// [`Scrump::visited`](super::Scrump::visited).
    pub fn visited(&self) -> &[i64] {
        self.0.visited()
    }

    /// The join over the diagonals visited so far: per window of `a`, in order, its nearest
    /// window of `b`; none for a window with no visited pair. `Err` when the rows are too large
    /// to allocate.
    pub fn profile(&self) -> Result<Vec<(u64, NonNormNeighbor)>, ScrumpError> {
        let nearest = &self.0.engine.nearest;
        let mut rows = reserved_vec(nearest.len() as u128)?;
        rows.extend(nearest.iter().enumerate().map(|(row, &n)| (row as u64, n)));
        Ok(rows)
    }
}
