//! The pan matrix profile, following STUMPY 1.14.1's `stimp` at `percentage = 1`: the exact
//! two-sided matrix profile at every window size of a range, visited in STUMPY's breadth-first
//! order, and STUMPY's transform of the rows computed so far into one image.
//!
//! The window sizes are STUMPY's list. With `max_m` given, the ends are `min_m` and `max_m`, which
//! must be in order, `min_m` at least 3 and `max_m` at most `max_window_size(len)`: STUMPY sorts
//! them, raises the smaller to 3 and lowers the larger to that size, and each of those is an error
//! here. Without it the ends are `min_m` and `max(min_m + 1, max_window_size(len))`, unclamped, as
//! in STUMPY. Every `step`-th size from the first is
//! used. The sizes are visited as the levels of the balanced binary search tree over them, so a
//! prefix of the visits already spans the whole range. Each visit is exactly
//! [`crate::batch::two_sided_profile`] at that size on bits.
//!
//! Differences from STUMPY: every row is exact (STUMPY's default samples the profile); a step of
//! zero, a list with no size, and a list holding a size below 3 or past the series length are
//! errors at construction, where STUMPY builds the object and fails later in `update`; and with
//! no scale given, each size infers its own scale from the series and that size. A negative step
//! is an error in the Python class (STUMPY accepts it and builds a descending range, which only
//! yields sizes on tiny series); here the step is unsigned.
//!
//! [`PanProfile::new_nonnorm`] is STUMPY's `aamp_stimp` (at `percentage = 1`): the same procedure
//! over the non-normalized metric, each size's row being the two-sided profile under
//! `NonNormalized`, in the units of the series. Only the image's normalization differs: a
//! distance is divided by `span * sqrt(window)`, where `span` is 2 for the z-normalized metric
//! and `|max - min|` over the finite samples for the non-normalized one, as STUMPY has it. A
//! series with no finite sample has no span (NaN), where STUMPY's constructor raises.
//!
//! Memory: one distance (`f64`) and one neighbor index (`u64`) per window per computed size, at
//! most `sizes * len` of each; the transform builds a `sizes * len` image of `f64`.

use alloc::vec::Vec;
use core::cmp::Ordering;
use core::num::NonZeroUsize;

use crate::index::SampleIndex;
use crate::nonnorm::NonNormalized;
use crate::pattern::Measure;
use crate::quantize::{Scale, ScaleChoice};
use crate::stats::{ConstantFlagsLength, Series};
use crate::znorm::ZNorm;

/// Why a pan profile could not be set up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PanError {
    /// The step between window sizes was zero.
    ZeroStep,
    /// The list holds a size below 3.
    WindowBelowThree,
    /// The list holds a size longer than the series (only reachable without `max_m`).
    WindowBeyondSeries,
    /// `min_m` is above `max_m`.
    MinAboveMax,
    /// `max_m` is above [`max_window_size`] of the series, `largest`.
    MaxAboveLargest { largest: usize },
}

/// The largest window size STUMPY allows for a self-join over `len` samples.
pub const fn max_window_size(len: usize) -> usize {
    len.saturating_sub(len.saturating_add(3) / 5)
        .saturating_sub(1)
}

/// STUMPY's breadth-first (level order) indices of the balanced binary search tree over `0..n`.
pub fn bfs_indices(n: usize) -> Vec<usize> {
    let mut order = Vec::with_capacity(n);
    let mut level: Vec<(usize, usize)> = Vec::new();
    if n > 0 {
        level.push((0, n));
    }
    while !level.is_empty() {
        let mut next = Vec::with_capacity(level.len() * 2);
        for &(low, high) in &level {
            let middle = low + (high - low) / 2;
            order.push(middle);
            if low < middle {
                next.push((low, middle));
            }
            if middle + 1 < high {
                next.push((middle + 1, high));
            }
        }
        level = next;
    }
    order
}

/// One computed window size: the distance and neighbor of every window, as the two-sided matrix
/// profile reports them (infinity and no index for a window with no neighbor). `scale` is the
/// tick size used, `None` when none could be inferred and the row holds no neighbors.
#[derive(Clone, Debug)]
pub struct PanRow {
    pub window: usize,
    pub scale: Option<Scale>,
    pub distances: Vec<f64>,
    pub indices: Vec<SampleIndex>,
}

/// How [`PanProfile::pan`] transforms the computed rows, STUMPY's defaults being
/// [`PanTransform::default`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PanTransform {
    pub threshold: f64,
    pub normalize: bool,
    pub contrast: bool,
    pub binary: bool,
    pub clip: bool,
}

impl Default for PanTransform {
    fn default() -> Self {
        Self {
            threshold: 0.2,
            normalize: true,
            contrast: true,
            binary: true,
            clip: true,
        }
    }
}

/// The pan matrix profile of a series over a range of window sizes, computed one size per
/// [`Self::update`].
#[derive(Clone, Debug)]
pub struct PanProfile {
    series: Vec<f64>,
    scale: ScaleChoice,
    threads: NonZeroUsize,
    /// Window sizes, ascending.
    sizes: Vec<usize>,
    /// Position in `sizes` of each visit, in visiting order.
    visits: Vec<usize>,
    rows: Vec<PanRow>,
    /// The metric's profile of one size.
    profile: ProfileFn,
    /// What normalization divides a distance by, over `sqrt(window)`.
    span: f64,
}

/// One size's nearest-neighbor distances (in the caller's units) and indices over `series`.
type ProfileFn = fn(Series<'_>, NonZeroUsize, Scale, NonZeroUsize) -> (Vec<f64>, Vec<SampleIndex>);

fn profile_in<M: Measure>(
    series: Series<'_>,
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
) -> (Vec<f64>, Vec<SampleIndex>) {
    M::nearest(series, m, scale, threads)
        .iter()
        .map(|hit| {
            let index = M::index(hit).map_or(SampleIndex::NONE, |start| {
                SampleIndex::new(start).expect("a window start is below NONE's sentinel")
            });
            (M::distance(hit, scale), index)
        })
        .unzip()
}

impl PanProfile {
    /// Nothing is computed yet. A given `scale` is used for every size; otherwise each size infers
    /// its own by the rule `scale` names (`None` is the default rule).
    pub fn new<S: Into<ScaleChoice>>(
        series: &[f64],
        min_m: usize,
        max_m: Option<usize>,
        step: usize,
        scale: S,
        threads: NonZeroUsize,
    ) -> Result<Self, PanError> {
        Self::build(
            series,
            (min_m, max_m, step),
            scale.into(),
            threads,
            profile_in::<ZNorm>,
            2.0,
        )
    }

    /// [`Self::new`] under the non-normalized metric, as STUMPY's `aamp_stimp`: each row is the
    /// two-sided profile of `NonNormalized`, distances in the units of `series`.
    pub fn new_nonnorm<S: Into<ScaleChoice>>(
        series: &[f64],
        min_m: usize,
        max_m: Option<usize>,
        step: usize,
        scale: S,
        threads: NonZeroUsize,
    ) -> Result<Self, PanError> {
        let finite = || series.iter().copied().filter(|x| x.is_finite());
        let span = finite()
            .reduce(f64::max)
            .zip(finite().reduce(f64::min))
            .map_or(f64::NAN, |(high, low)| libm::fabs(high - low));
        Self::build(
            series,
            (min_m, max_m, step),
            scale.into(),
            threads,
            profile_in::<NonNormalized>,
            span,
        )
    }

    fn build(
        series: &[f64],
        (min_m, max_m, step): (usize, Option<usize>, usize),
        scale: ScaleChoice,
        threads: NonZeroUsize,
        profile: ProfileFn,
        span: f64,
    ) -> Result<Self, PanError> {
        if step == 0 {
            return Err(PanError::ZeroStep);
        }
        let largest = max_window_size(series.len());
        let (first, end) = match max_m {
            Some(max_m) if min_m > max_m => return Err(PanError::MinAboveMax),
            Some(max_m) if max_m > largest => return Err(PanError::MaxAboveLargest { largest }),
            Some(max_m) => (min_m, max_m),
            None => (min_m, min_m.saturating_add(1).max(largest)),
        };
        if first < 3 {
            return Err(PanError::WindowBelowThree);
        }
        if first + (end - first) / step * step > series.len() {
            return Err(PanError::WindowBeyondSeries);
        }
        let sizes: Vec<usize> = (first..=end).step_by(step).collect();
        let visits = bfs_indices(sizes.len());
        Ok(Self {
            series: series.to_vec(),
            scale,
            threads,
            sizes,
            visits,
            rows: Vec::new(),
            profile,
            span,
        })
    }

    /// Every window size, in visiting order.
    pub fn window_sizes(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.visits.iter().map(|&position| self.sizes[position])
    }

    /// How many sizes in all.
    pub fn sizes(&self) -> usize {
        self.sizes.len()
    }

    /// How many sizes have been computed.
    pub fn processed(&self) -> usize {
        self.rows.len()
    }

    /// The series length, which is the width of [`Self::pan`]'s image.
    pub fn len(&self) -> usize {
        self.series.len()
    }

    pub fn is_empty(&self) -> bool {
        self.series.is_empty()
    }

    /// The computed rows, in visiting order.
    pub fn rows(&self) -> &[PanRow] {
        &self.rows
    }

    /// The next size in visiting order, `None` when all are done.
    pub fn next_window_size(&self) -> Option<usize> {
        self.visits
            .get(self.rows.len())
            .map(|&position| self.sizes[position])
    }

    /// Computes the next size in visiting order; `false` when all are done.
    pub fn update(&mut self) -> bool {
        self.update_with(None).unwrap_or(false)
    }

    /// [`Self::update`] with the caller's flag for each window of the next size being constant
    /// (see [`Series::with_constant_windows`]); `None` is the default rule. The flags are for
    /// one size, so a caller with a predicate evaluates it at [`Self::next_window_size`]. The
    /// non-normalized metric has no constant windows and ignores them.
    pub fn update_with(&mut self, constant: Option<&[bool]>) -> Result<bool, ConstantFlagsLength> {
        let Some(&position) = self.visits.get(self.rows.len()) else {
            return Ok(false);
        };
        let window = self.sizes[position];
        let m = NonZeroUsize::new(window).expect("window sizes are at least 3");
        let scale = self.scale.resolve(&self.series, m);
        let windows = self.series.len() + 1 - window;
        let series = match constant {
            Some(flags) => Series::with_constant_windows(&self.series, m, flags)?,
            None => Series::from(&self.series),
        };
        let (distances, indices) = match scale {
            Some(scale) => (self.profile)(series, m, scale, self.threads),
            None => (
                alloc::vec![f64::INFINITY; windows],
                alloc::vec![SampleIndex::NONE; windows],
            ),
        };
        self.rows.push(PanRow {
            window,
            scale,
            distances,
            indices,
        });
        Ok(true)
    }

    /// STUMPY's `pan`: the computed rows, one image row per window size in ascending order and
    /// one column per sample, row-major (`sizes() * len()` values). Each row is normalized by
    /// its window size, contrasted around the threshold by rank, binarized and clipped as
    /// asked; every row then replaces the not-yet-computed rows above it, and every NaN left
    /// takes the largest value of the image. All NaN when nothing is computed.
    pub fn pan(&self, transform: PanTransform) -> Vec<f64> {
        let width = self.series.len();
        let mut image = alloc::vec![f64::NAN; self.sizes.len() * width];
        let mut computed = alloc::vec![false; self.sizes.len()];
        for (row, &position) in self.rows.iter().zip(&self.visits) {
            computed[position] = true;
            let cells = &mut image[position * width..][..width];
            for (cell, &distance) in cells.iter_mut().zip(&row.distances) {
                if distance != f64::INFINITY {
                    *cell = distance;
                }
            }
        }
        let visited = &self.visits[..self.rows.len()];

        if transform.normalize {
            for (row, &position) in self.rows.iter().zip(visited) {
                let norm = 1.0 / (self.span * libm::sqrt(row.window as f64));
                for cell in &mut image[position * width..][..width] {
                    let scaled = *cell * norm;
                    *cell = if scaled.is_nan() {
                        scaled
                    } else {
                        scaled.min(1.0)
                    };
                }
            }
        }
        if transform.contrast {
            contrast(&mut image, width, visited, transform.threshold);
        }
        if transform.binary {
            for &position in visited {
                for cell in &mut image[position * width..][..width] {
                    *cell = if *cell <= transform.threshold {
                        0.0
                    } else {
                        1.0
                    };
                }
            }
        }
        if transform.clip {
            for &position in visited {
                for cell in &mut image[position * width..][..width] {
                    if !cell.is_nan() {
                        *cell = cell.clamp(0.0, 1.0);
                    }
                }
            }
        }

        let mut source = None;
        for position in (0..self.sizes.len()).rev() {
            if computed[position] {
                source = Some(position);
            }
            if let Some(source) = source.filter(|&source| source != position) {
                image.copy_within(source * width..(source + 1) * width, position * width);
            }
        }
        let largest = image
            .iter()
            .copied()
            .filter(|cell| !cell.is_nan())
            .fold(None, |best: Option<f64>, cell| {
                Some(best.map_or(cell, |best| best.max(cell)))
            });
        if let Some(largest) = largest {
            for cell in &mut image {
                if cell.is_nan() {
                    *cell = largest;
                }
            }
        }
        image
    }
}

/// STUMPY's contrast: every cell of the visited rows, read row by row in visiting order, is
/// replaced by the logistic of its rank's share (ties by reading order, NaN last) around
/// `threshold`.
fn contrast(image: &mut [f64], width: usize, visited: &[usize], threshold: f64) {
    let count = visited.len() * width;
    let cell_at = |read: usize| visited[read / width] * width + read % width;
    let mut order: Vec<usize> = (0..count).collect();
    order.sort_by(|&a, &b| {
        let (a, b) = (image[cell_at(a)], image[cell_at(b)]);
        match (a.is_nan(), b.is_nan()) {
            (false, false) => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
            (nan_a, nan_b) => nan_a.cmp(&nan_b),
        }
    });
    let step = 1.0 / (count as f64 - 1.0);
    for (rank, &read) in order.iter().enumerate() {
        let share = if rank + 1 == count && count > 1 {
            1.0
        } else if count > 1 {
            rank as f64 * step
        } else {
            0.0
        };
        image[cell_at(read)] = 1.0 / (1.0 + libm::exp(-10.0 * (share - threshold)));
    }
}
