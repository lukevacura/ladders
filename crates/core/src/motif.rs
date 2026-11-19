//! Batch motifs, following STUMPY 1.14.1's `motifs` with every order decided exactly, z-normalized
//! ([`motifs`]) or not ([`motifs_nonnorm`], STUMPY's `aamp_motifs`): one procedure over the
//! metric (see [`crate::pattern`]), so the two cannot drift.
//!
//! Candidates are the windows by ascending distance to their nearest neighbor (the two-sided
//! profile), ties to the smaller start. Each candidate's matches are those of
//! `TickedSeries::matches_of_window` (the pattern matcher, the candidate itself first at
//! distance zero), and the candidate is a motif when it has more than `min_neighbors` of them.
//! Every window of every candidate's matches, motif or not, then leaves the candidate pool
//! together with the `ceil(m/4)` starts on each side of it, so motifs are pairwise outside the
//! exclusion zone. The walk stops at `max_motifs` motifs, or at the first remaining candidate
//! whose distance exceeds `cutoff` (or `max_distance`, when it is a value).
//!
//! Differences from STUMPY: candidate and match orders are the exact order of distances rather
//! than of their rounded values; thresholds are compared on the f64 distances as
//! [`find_matches`](crate::pattern::find_matches) does, with no tolerance unless
//! `match_rules.atol` asks for STUMPY's; the
//! default `cutoff` and `max_distance` are summed in index order, not pairwise; and the default
//! `cutoff` is at least the distance of the exactly nearest window, which two exactly equal
//! distances can round to a hair above the smallest one.
//!
//! STUMPY's two procedures differ in one step, and each is kept as STUMPY has it: `motifs` asks
//! for at most `max_matches` matches of a candidate, so only those leave the candidate pool and
//! only they count towards `min_neighbors`; `aamp_motifs` asks for all of them, so every match
//! leaves the pool and counts, and only then are the first `max_matches` kept.
//!
//! [`motif_pairs`] is the best pair alone: the pair of highest exact correlation, found by a run
//! that never finishes the profile ([`pick_pair`] is the rule it shares with a check over the full
//! profile).
//!
//! The multi-dimensional motifs ([`mmotifs`], [`mmotifs_nonnorm`]) are the same procedure over a
//! multi-dimensional profile the caller supplies, with a subspace chosen for each candidate (see
//! [`crate::mdl`]): the matches are the windows nearest in the mean of the subspace's dimensions'
//! distances, selected by the same greedy rule through the same metric concept.

use alloc::vec::Vec;
use core::num::NonZeroUsize;

use crate::entry::{Entry, TwoSided};
use crate::index::SampleIndex;
use crate::mdl::{Description, maamp_mdl, mdl};
use crate::multidim::{
    DimensionDistance, MeanDistance, MeanOrder, MeanWork, MultiError, checked_include,
};
use crate::nearest::ExactDistance;
use crate::nearest::Sharing;
use crate::nonnorm::{NonNormNeighbor, NonNormalized};
use crate::pattern::{
    MatchRules, MaxDistance, Measure, Selection, TickedSeries, default_threshold, select_starts,
};
use crate::quantize::Scale;
use crate::stats::Series;
use crate::subspace::{Options, window_count};
use crate::znorm::ZNorm;
use core::cell::RefCell;

/// What makes a window a motif and how many matches it keeps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotifParams {
    /// A candidate is a motif when it has more than this many matches, itself included.
    pub min_neighbors: usize,
    pub max_distance: MaxDistance,
    /// The largest nearest-neighbor distance a candidate may have; `None` is STUMPY's default,
    /// `max(mean(P) - 2 * std(P), min(P))` over the finite distances.
    pub cutoff: Option<f64>,
    /// `None` keeps every match.
    pub max_matches: Option<usize>,
    pub max_motifs: usize,
    /// STUMPY's `atol`, added to the threshold of every candidate's matches as it is for
    /// [`find_matches_with`](crate::pattern::find_matches_with). Its `query_idx` is not read: each
    /// candidate is its own query.
    pub match_rules: MatchRules,
}

impl Default for MotifParams {
    fn default() -> Self {
        Self {
            min_neighbors: 1,
            max_distance: MaxDistance::Default,
            cutoff: None,
            max_matches: Some(10),
            max_motifs: 1,
            match_rules: MatchRules {
                atol: 0.0,
                query_idx: None,
            },
        }
    }
}

/// The motifs of `series`, best first; each is its matches as `(start, entry)`, the motif's own
/// window first at distance zero, then nearest first. `threads` only shares the profile's work:
/// the result is the same for every count.
pub fn motifs<'a>(
    series: impl Into<Series<'a>>,
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
    params: &MotifParams,
) -> Vec<Vec<(u64, Entry)>> {
    motifs_in::<ZNorm>(series.into(), m, scale, threads, params, Asked::AtMostMax)
}

/// [`motifs`] under the non-normalized metric, as STUMPY's `aamp_motifs`: distances are Euclidean
/// on the raw values quantized at `scale` and compared (`cutoff`, `max_distance`) in the caller's
/// units; every order is the exact squared distance on ticks, ties to the smaller start.
pub fn motifs_nonnorm(
    series: &[f64],
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
    params: &MotifParams,
) -> Vec<Vec<(u64, NonNormNeighbor)>> {
    motifs_in::<NonNormalized>(series.into(), m, scale, threads, params, Asked::All)
}

/// The best pair of a series: the pair of two windows outside each other's exclusion zone with
/// the highest exact correlation, ties to the smaller `(a, b)`; `a < b`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotifPair {
    /// The earlier window of the pair.
    pub a: usize,
    /// The later window of the pair.
    pub b: usize,
    /// The z-normalized Euclidean distance between the two windows, as window `a`'s profile entry
    /// reports it.
    pub distance: f64,
}

/// Why [`motif_pairs`] gave no answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MotifPairsError {
    /// The run's tables could not be allocated.
    Capacity,
}

/// The best pair of `series` under the z-normalized metric: the pair of highest exact
/// correlation, ties to the smaller `(a, b)`, with `a < b`. An empty answer is a series with no
/// window that has a finite distance to another. Exact: the pair is the one [`pick_pair`] takes
/// from the full two-sided profile, on bits and for every `threads`; the run only skips work
/// that pair cannot need, by keeping a global floor, so it is faster than the profile on series
/// that have a pair clearly nearer than most.
///
/// # Errors
/// [`MotifPairsError::Capacity`] when the run cannot allocate its tables.
pub fn motif_pairs<'a>(
    series: impl Into<Series<'a>>,
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
) -> Result<Vec<MotifPair>, MotifPairsError> {
    let series: Series<'a> = series.into();
    let profile = crate::batch::best_pair_profile(series, m, scale, threads)
        .map_err(|_| MotifPairsError::Capacity)?;
    Ok(pick_pair(series, m, scale, &profile).into_iter().collect())
}

/// The pair of `series` a two-sided `profile` of it holds: the pair of highest exact correlation,
/// ties to the smaller `(a, b)`, `a` the earlier window. The one rule [`motif_pairs`] and its
/// checks over a full profile share.
///
/// Orientation can move a pair's two entries a rounding apart (window `a`'s distance to `b` and
/// window `b`'s to `a`), so the entries only select candidates and the exact comparator
/// ([`ExactDistance`]) ranks them. A window with a neighbor and a finite squared distance is a
/// candidate when that distance is within `2^-40` (relative) of the least one plus `2m·2^-40`
/// (absolute): a squared distance is `2m(1 - ρ)` of a correlation the kernel gets within a few
/// units of `2^-52`, so the band is some thousand times wider than any orientation difference,
/// and wide only costs one exact pair per extra candidate. The best pair `(a, b)` has both its
/// windows within the band, since each window's best correlation is the pair's. Window `a`'s
/// nearest neighbor is `b`: a partner of `a` before `a` would be a pair earlier than `(a, b)`,
/// and the nearest among the later ones is the smallest, which `b` is. In a best-pair run
/// window `a` keeps its exact entry, since the global floor stays below the best correlation by
/// more than the near-tie band. The distance is that of window `a`'s entry.
#[doc(hidden)]
pub fn pick_pair<'a>(
    series: impl Into<Series<'a>>,
    m: NonZeroUsize,
    scale: Scale,
    profile: &[(u64, TwoSided)],
) -> Option<MotifPair> {
    let neighbor_of = |&(start, sides): &(u64, TwoSided)| {
        let squared = sides.nearest.squared_distance();
        let neighbor = sides.nearest.index().get()?;
        squared
            .is_finite()
            .then_some((start as usize, neighbor as usize, squared))
    };
    let least = profile
        .iter()
        .filter_map(neighbor_of)
        .map(|(_, _, squared)| squared)
        .min_by(f64::total_cmp)?;
    let within = least * (1.0 + BAND) + 2.0 * m.get() as f64 * BAND;
    let ticked = TickedSeries::<ZNorm>::new(series, m, scale);
    let mut best: Option<((usize, usize), ExactDistance)> = None;
    for (start, neighbor, squared) in profile.iter().filter_map(neighbor_of) {
        if squared > within {
            continue;
        }
        let pair = (start.min(neighbor), start.max(neighbor));
        let Some(exact) = ticked.exact_pair(pair.0, pair.1) else {
            continue;
        };
        let better = best.as_ref().is_none_or(|(held_pair, held)| {
            exact.cmp(held).then_with(|| pair.cmp(held_pair)).is_lt()
        });
        if better {
            best = Some((pair, exact));
        }
    }
    let ((a, b), _) = best?;
    let (row, sides) = profile.get(a)?;
    debug_assert_eq!(*row as usize, a, "the profile has one row per window");
    debug_assert_eq!(
        sides.nearest.index().get(),
        Some(b as u64),
        "window a's nearest is b"
    );
    Some(MotifPair {
        a,
        b,
        distance: libm::sqrt(sides.nearest.squared_distance()),
    })
}

/// `2^-40`: the band of [`pick_pair`]'s candidates.
const BAND: f64 = 1.0 / (1u64 << 40) as f64;

/// How many matches of a candidate the motif procedure asks the matcher for. STUMPY's two
/// procedures differ here, so each public function supplies its own.
#[derive(Clone, Copy)]
enum Asked {
    /// At most `max_matches` (STUMPY's `motifs`): only those leave the candidate pool and count
    /// towards `min_neighbors`.
    AtMostMax,
    /// All of them (STUMPY's `aamp_motifs`, which slices to `max_matches` only afterwards).
    All,
}

fn motifs_in<M: Measure>(
    series: Series<'_>,
    m: NonZeroUsize,
    scale: Scale,
    threads: NonZeroUsize,
    params: &MotifParams,
    asked: Asked,
) -> Vec<Vec<(u64, M::Hit)>> {
    let nearest = M::nearest(series, m, scale, threads);
    let distances: Vec<f64> = nearest.iter().map(|hit| M::distance(hit, scale)).collect();
    let mut found = Vec::new();
    let ticked = TickedSeries::<M>::new(series, m, scale);
    let mut order: Vec<(usize, M::Exact)> = nearest
        .iter()
        .enumerate()
        .filter_map(|(row, hit)| {
            let neighbor = M::index(hit)? as usize;
            Some((row, ticked.exact_pair(row, neighbor)?))
        })
        .collect();
    order.sort_by(|a, b| M::compare(&a.1, &b.1, Sharing::Nothing).then(a.0.cmp(&b.0)));
    let Some(&(first, _)) = order.first() else {
        return found;
    };
    let cutoff = match params.cutoff {
        Some(cutoff) => cutoff,
        // Every distance overflowed to infinity (a huge scale): no candidate can qualify.
        None => match default_threshold(&distances) {
            Some(threshold) => threshold.max(distances[first]),
            None => return found,
        },
    };

    let zone = crate::exclusion::zone(m.get());
    let mut excluded = alloc::vec![false; nearest.len()];
    for (row, _) in order {
        if found.len() >= params.max_motifs {
            break;
        }
        if excluded[row] {
            continue;
        }
        let distance = distances[row];
        if distance > cutoff || !distance.is_finite() {
            break;
        }
        if matches!(params.max_distance, MaxDistance::Value(v) if distance > v) {
            break;
        }
        let mut matches = ticked.matches_of_window(
            row,
            params.max_distance,
            match asked {
                Asked::AtMostMax => params.max_matches,
                Asked::All => None,
            },
            params.match_rules.atol,
        );
        let blocked_starts = if matches.is_empty() {
            alloc::vec![row as u64]
        } else {
            matches.iter().map(|&(start, _)| start).collect()
        };
        for start in blocked_starts {
            let start = start as usize;
            let to = (start + zone + 1).min(excluded.len());
            excluded[start.saturating_sub(zone)..to].fill(true);
        }
        if matches.len() > params.min_neighbors {
            matches.truncate(params.max_matches.unwrap_or(usize::MAX));
            found.push(matches);
        }
    }
    found
}

/// The largest nearest-neighbor distance a candidate of each row of the multi-dimensional profile
/// may have.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Cutoffs<'a> {
    /// STUMPY's default for every row: `max(mean(P) - 2 * std(P), min(P))` over the row's finite
    /// distances; a row without one has no candidate.
    Default,
    /// The same for every row.
    All(f64),
    /// One for each row, so one per dimension.
    PerRow(&'a [f64]),
}

/// What makes a window a multi-dimensional motif, how many matches it keeps and how the
/// subspace is chosen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MultiMotifParams<'a> {
    /// A candidate is a motif when it has more than this many matches, itself included.
    pub min_neighbors: usize,
    /// How far a window may be from the candidate, in the mean over the subspace; a value also
    /// bounds the candidate's own profile distance.
    pub max_distance: MaxDistance,
    pub cutoffs: Cutoffs<'a>,
    /// `None` keeps every match.
    pub max_matches: Option<usize>,
    pub max_motifs: usize,
    /// Added to the threshold a window's distance is compared with.
    pub atol: f64,
    /// The subspace has `k + 1` dimensions; `None` takes the `k` of the smallest description
    /// length, for each motif.
    pub k: Option<usize>,
    /// Dimensions that must be in every subspace.
    pub include: &'a [usize],
}

impl Default for MultiMotifParams<'_> {
    fn default() -> Self {
        Self {
            min_neighbors: 1,
            max_distance: MaxDistance::Default,
            cutoffs: Cutoffs::Default,
            max_matches: Some(10),
            max_motifs: 1,
            atol: 1e-8,
            k: None,
            include: &[],
        }
    }
}

/// A multi-dimensional motif: its matches as `(start, distance)`, the motif's own window first
/// at distance zero then nearest first, the distance the mean over the subspace's dimensions of
/// their distances; the subspace; and the description length at every `k`.
#[derive(Clone, Debug, PartialEq)]
pub struct MultiMotif {
    pub matches: Vec<(u64, f64)>,
    pub subspace: Vec<usize>,
    pub bit_sizes: Vec<f64>,
}

/// The multi-dimensional motifs of z-normalized `set` (each series quantized at its own scale)
/// from its profile (`distance[k][i]` and `index[k][i]`, as [`crate::batch::multi::mstump`]
/// returns them), following STUMPY 1.14.1's `mmotifs`.
///
/// Each round takes, for every `k`, the window of the row's smallest remaining distance (the
/// smaller start on equal distances) and its neighbor, picks `k` (the one given, or the smallest
/// description length, the smaller `k` on equal ones) and with it the subspace of that pair, and
/// stops at the first candidate whose distance exceeds its row's cutoff or `max_distance`, or
/// that has no distance or neighbor. The candidate's matches are the windows of the series
/// nearest to it in the mean of the subspace's dimensions' distances, as
/// [`crate::pattern::find_matches`] selects them; it is a motif when it has more than
/// `min_neighbors` of them. Whether or not it is, every matched window then leaves every row of
/// the profile together with the `ceil(m/4)` starts on each side of it.
///
/// Differences from STUMPY: every order is that of exact distances, not of their rounded values;
/// a window with a gap in a subspace dimension has no distance, and a candidate whose own window
/// has a gap in its subspace matches only itself; a row whose candidate has no neighbor has no
/// description (an infinite size); a candidate with no matches at all (`max_matches` zero or a
/// negative `max_distance`) still leaves the pool, where STUMPY would never stop; the default
/// cutoffs are summed in index order, not pairwise. `Err` for what [`crate::mdl::mdl`] refuses,
/// a profile that is not one row of one entry per window for each dimension, an index that is
/// not a window start or none, a `k` that is not a dimension, or cutoffs that are not one per
/// dimension.
pub fn mmotifs<'s, S: Into<Series<'s>> + Copy>(
    set: &[(S, Scale)],
    m: NonZeroUsize,
    distance: &[Vec<f64>],
    index: &[Vec<SampleIndex>],
    params: &MultiMotifParams<'_>,
) -> Result<Vec<MultiMotif>, MultiError> {
    let series: Vec<Series<'s>> = set.iter().map(|&(series, _)| series.into()).collect();
    let ticked: Vec<TickedSeries<ZNorm>> = series
        .iter()
        .zip(set)
        .map(|(&series, &(_, scale))| TickedSeries::new(series, m, scale))
        .collect();
    let options = Options {
        include: params.include,
        ..Options::default()
    };
    mmotifs_in(
        series.iter().map(Series::len),
        &ticked,
        m,
        distance,
        index,
        params,
        |subseq_idx, nn_idx| mdl(set, m, subseq_idx, nn_idx, &options),
    )
}

/// [`mmotifs`] under the non-normalized metric, as STUMPY's `aamp_mmotifs` with `p = 2`: every
/// series quantized at the one `scale`, a dimension's distance the Euclidean distance on its raw
/// values, in the caller's units, and the subspace chosen as [`crate::mdl::maamp_mdl`] does. The
/// distance of a dimension whose window pair needs more than 128 bits is none.
pub fn mmotifs_nonnorm(
    set: &[&[f64]],
    m: NonZeroUsize,
    scale: Scale,
    distance: &[Vec<f64>],
    index: &[Vec<SampleIndex>],
    params: &MultiMotifParams<'_>,
) -> Result<Vec<MultiMotif>, MultiError> {
    let ticked: Vec<TickedSeries<NonNormalized>> = set
        .iter()
        .map(|series| TickedSeries::new(*series, m, scale))
        .collect();
    let options = Options {
        include: params.include,
        ..Options::default()
    };
    mmotifs_in(
        set.iter().map(|series| series.len()),
        &ticked,
        m,
        distance,
        index,
        params,
        |subseq_idx, nn_idx| maamp_mdl(set, m, subseq_idx, nn_idx, &options),
    )
}

/// The first position of the smallest value, as NumPy's `argmin`: a NaN is the smallest.
fn argmin(values: &[f64]) -> usize {
    let mut best = 0;
    for (i, &value) in values.iter().enumerate() {
        if value.is_nan() {
            return i;
        }
        if value < values[best] {
            best = i;
        }
    }
    best
}

fn row_cutoffs(rows: &[Vec<f64>], cutoffs: Cutoffs<'_>) -> Result<Vec<Option<f64>>, MultiError> {
    match cutoffs {
        Cutoffs::Default => Ok(rows.iter().map(|row| default_threshold(row)).collect()),
        Cutoffs::All(cutoff) => Ok(alloc::vec![Some(cutoff); rows.len()]),
        Cutoffs::PerRow(cutoffs) if cutoffs.len() == rows.len() => {
            Ok(cutoffs.iter().map(|&cutoff| Some(cutoff)).collect())
        }
        Cutoffs::PerRow(_) => Err(MultiError::CutoffCount),
    }
}

/// A window's mean over the subspace and where its distances start in the flat list of them.
struct Matched<D> {
    mean: MeanDistance<D>,
    at: usize,
}

impl<D> Clone for Matched<D> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<D> Copy for Matched<D> {}

/// The multi-dimensional matches of the window at `start`: its mean distance over `subspace` to
/// every window, selected greedily, as `(start, distance)`.
fn subspace_matches<M: Measure>(
    ticked: &[TickedSeries<M>],
    subspace: &[usize],
    start: usize,
    params: &MultiMotifParams<'_>,
) -> Vec<(u64, f64)> {
    let queries: Vec<_> = subspace
        .iter()
        .map(|&a| ticked[a].query_at(start))
        .collect();
    let windows = ticked[0].windows().len();
    let width = subspace.len();
    let mut parts: Vec<Option<M::Dimension>> = Vec::with_capacity(width * windows);
    let full: Vec<Option<Matched<M::Dimension>>> = (0..windows)
        .map(|window| {
            let at = parts.len();
            parts.extend(subspace.iter().zip(&queries).map(|(&a, query)| {
                ticked[a]
                    .exact_at(query, window)
                    .map(|exact| M::dimension(&exact))
            }));
            let mean = MeanDistance::of(&parts[at..]);
            Some(Matched { mean, at }).filter(|matched| !matched.mean.is_missing())
        })
        .collect();
    let distance_of =
        |matched: &Matched<M::Dimension>| ticked[0].mean_in_caller_units(matched.mean.to_f64());
    // Matches go by the exact order of their means, ties to the smaller start.
    let ranking = MeanOrder::new(
        width,
        <M::Dimension as DimensionDistance>::ROOT_RELATIVE_ERROR,
    );
    let work = RefCell::new(MeanWork::new());
    let compare = |a: &Matched<M::Dimension>, b: &Matched<M::Dimension>| {
        ranking.compare(
            &mut work.borrow_mut(),
            &a.mean,
            &parts[a.at..a.at + width],
            &b.mean,
            &parts[b.at..b.at + width],
        )
    };
    let selection = Selection {
        zone: ticked[0].zone(),
        rules: MatchRules {
            atol: params.atol,
            query_idx: Some(start),
        },
        max_distance: params.max_distance,
        max_matches: params.max_matches,
    };
    select_starts(&full, &selection, distance_of, compare)
        .into_iter()
        .map(|i| {
            let distance = if i == start {
                0.0
            } else {
                full[i].as_ref().map_or(f64::INFINITY, &distance_of)
            };
            (i as u64, distance)
        })
        .collect()
}

fn mmotifs_in<M: Measure>(
    lengths: impl Iterator<Item = usize>,
    ticked: &[TickedSeries<M>],
    m: NonZeroUsize,
    distance: &[Vec<f64>],
    index: &[Vec<SampleIndex>],
    params: &MultiMotifParams<'_>,
    describe: impl Fn(&[usize], &[usize]) -> Result<Description, MultiError>,
) -> Result<Vec<MultiMotif>, MultiError> {
    let windows = window_count(lengths, m)?;
    let dimensions = ticked.len();
    checked_include(params.include, dimensions)?;
    if params.k.is_some_and(|k| k >= dimensions) {
        return Err(MultiError::KOutOfRange);
    }
    if distance.len() != dimensions
        || index.len() != dimensions
        || distance.iter().any(|row| row.len() != windows)
        || index.iter().any(|row| row.len() != windows)
        || index
            .iter()
            .flatten()
            .any(|neighbor| neighbor.get().is_some_and(|start| start >= windows as u64))
    {
        return Err(MultiError::ProfileShape);
    }
    let cutoffs = row_cutoffs(distance, params.cutoffs)?;
    let zone = crate::exclusion::zone(m.get());
    let mut profile = distance.to_vec();
    let mut found = Vec::new();
    while found.len() < params.max_motifs {
        let candidates: Vec<usize> = profile.iter().map(|row| argmin(row)).collect();
        let neighbors: Vec<Option<usize>> = index
            .iter()
            .zip(&candidates)
            .map(|(row, &candidate)| row[candidate].get().map(|start| start as usize))
            .collect();
        let pairs: Vec<usize> = neighbors
            .iter()
            .zip(&candidates)
            .map(|(neighbor, &candidate)| neighbor.unwrap_or(candidate))
            .collect();
        let mut description = describe(&candidates, &pairs)?;
        for (size, neighbor) in description.bit_sizes.iter_mut().zip(&neighbors) {
            if neighbor.is_none() {
                *size = f64::INFINITY;
            }
        }
        let k = params.k.unwrap_or_else(|| argmin(&description.bit_sizes));
        let start = candidates[k];
        let value = profile[k][start];
        if cutoffs[k].is_some_and(|cutoff| value > cutoff)
            || !value.is_finite()
            || neighbors[k].is_none()
            || matches!(params.max_distance, MaxDistance::Value(v) if value > v)
        {
            break;
        }
        let subspace = description.subspaces.swap_remove(k).dimensions;
        let matches = subspace_matches(ticked, &subspace, start, params);
        // The candidate's own window leaves with its matches, so every round makes progress.
        let blocked = matches
            .iter()
            .map(|&(start, _)| start as usize)
            .chain(core::iter::once(start));
        for blocked in blocked {
            let to = (blocked + zone + 1).min(windows);
            for row in &mut profile {
                row[blocked.saturating_sub(zone)..to].fill(f64::INFINITY);
            }
        }
        if matches.len() > params.min_neighbors {
            found.push(MultiMotif {
                matches,
                subspace,
                bit_sizes: description.bit_sizes,
            });
        }
    }
    Ok(found)
}
