use super::plan::{default_chunk_lanes, threads_for_size};
use super::two_sided::{nearer_side, two_sided_profile, two_sided_sides};
use crate::capacity::clamp_count;
use crate::discord::{Discord, DiscordTracker};
use crate::index::SampleIndex;
use crate::quantize::Scale;
use crate::stats::Series;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

/// Whole-series segmentation (STUMPY's FLUSS) over `series`: the corrected arc curve of the
/// [`two_sided_profile`] indices and the `n_regimes - 1` regime locations picked from it by
/// [`crate::floss::regime_locations`]. `l` (at least 1) is the regime length and `excl_factor`
/// scales it into the end zones and the exclusion around each pick; `custom_ideal`, one value
/// per window, replaces the idealized arc curve. Returns at most one regime location per window
/// (STUMPY would pad with repeats of an already returned location). Looks ahead, like
/// [`two_sided_profile`]. `None` when `custom_ideal` has the wrong length. The thread count never
/// changes a bit.
#[allow(clippy::too_many_arguments)]
pub fn fluss<'a>(
    series: impl Into<Series<'a>>,
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
    l: usize,
    n_regimes: usize,
    excl_factor: usize,
    custom_ideal: Option<&[f64]>,
) -> Option<(Vec<f64>, Vec<usize>)> {
    let series: Series<'a> = series.into();
    let indices: Vec<SampleIndex> = two_sided_profile(series, m, scale, threads)
        .iter()
        .map(|(_, sides)| sides.nearest.index())
        .collect();
    crate::floss::fluss(
        &indices,
        crate::exclusion::zone(m.get()),
        l,
        n_regimes,
        excl_factor,
        custom_ideal,
    )
}

/// The top `k` discords of `series`: the windows whose exact nearest neighbor, on either side,
/// is farthest, picked farthest first and pairwise outside the exclusion zone, ties to the
/// smaller index, by the same picking rule as [`DiscordTracker`]. A window with no neighbor is
/// never a discord. Equal on bits whatever `threads` is.
pub fn discords<'a>(
    series: impl Into<Series<'a>>,
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
    k: usize,
) -> Vec<Discord> {
    let series: Series<'a> = series.into();
    let threads = threads_for_size(series.len(), m, threads);
    let (quantized, left, right, _) = two_sided_sides(
        series,
        m,
        scale,
        threads,
        default_chunk_lanes(),
        None,
        NonZeroUsize::MIN,
    );
    let scores = left
        .into_iter()
        .zip(right)
        .map(|((row, left), (_, right))| {
            let score = nearer_side(&quantized, m.get(), row as usize, left, right);
            (SampleIndex::new(row).unwrap_or(SampleIndex::NONE), score)
        });
    DiscordTracker::pick_all(m, clamp_count(k, quantized.windows.len()), scores)
}
