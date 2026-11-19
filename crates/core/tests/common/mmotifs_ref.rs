//! The reference the multi-dimensional motif tests share: STUMPY's `mmotifs` procedure stated
//! plainly over a metric's per-dimension distances, whichever metric supplies them, with the
//! description-length reference choosing `k` and the subspace.
#![allow(dead_code)]

use ladde::index::SampleIndex;
use ladde::motif::{Cutoffs, MultiMotif, MultiMotifParams};
use ladde::multidim::{DimensionDistance, MultiError, MultiProfile};
use ladde::pattern::MaxDistance;
use ladde::subspace::Subspace;

/// STUMPY's default threshold over the finite values: `max(mean - 2 std, min)`.
pub fn default_threshold(values: &[f64]) -> Option<f64> {
    let finite: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if finite.is_empty() {
        return None;
    }
    let n = finite.len() as f64;
    let mean = finite.iter().fold(0.0, |a, x| a + x) / n;
    let var = finite.iter().fold(0.0, |a, x| a + (x - mean) * (x - mean)) / n;
    let min = finite.iter().copied().fold(f64::INFINITY, f64::min);
    Some((mean - 2.0 * libm::sqrt(var)).max(min))
}

/// What a motif is, as plain values: its matches' starts and distance bits, subspace, sizes' bits.
pub type Plain = (Vec<(u64, u64)>, Vec<usize>, Vec<u64>);

pub fn plain(motif: &MultiMotif) -> Plain {
    (
        motif
            .matches
            .iter()
            .map(|&(start, distance)| (start, distance.to_bits()))
            .collect(),
        motif.subspace.clone(),
        motif.bit_sizes.iter().map(|x| x.to_bits()).collect(),
    )
}

/// What the reference needs of a metric.
pub trait RefMetric {
    /// Orders the exact distances of one dimension, nearer smaller.
    type Key: Ord + Copy;

    fn dimensions(&self) -> usize;
    fn samples(&self) -> usize;
    fn m(&self) -> usize;

    /// Dimension `a`'s distance between the windows at `window` and `start`: its root in the
    /// unit the means are formed in, and its exact order; `None` for no distance.
    fn part(&self, a: usize, window: usize, start: usize) -> Option<(f64, Self::Key)>;

    /// A mean in the caller's units.
    fn caller_units(&self, mean: f64) -> f64;

    /// The description lengths and subspaces of the pairs.
    fn describe(
        &self,
        starts: &[usize],
        neighbors: &[usize],
        include: &[usize],
    ) -> Result<(Vec<f64>, Vec<Subspace>), MultiError>;
}

/// A window's distances in the subspace's dimensions, `None` if any has none.
type Parts<K> = Option<Vec<(f64, K)>>;

fn matches<R: RefMetric>(
    r: &R,
    subspace: &[usize],
    start: usize,
    params: &MultiMotifParams<'_>,
) -> Vec<(u64, u64)> {
    let windows = r.samples() - r.m() + 1;
    let zone = ladde::exclusion::zone(r.m());
    let parts: Vec<Parts<R::Key>> = (0..windows)
        .map(|j| subspace.iter().map(|&a| r.part(a, j, start)).collect())
        .collect();
    let mean: Vec<f64> = parts
        .iter()
        .map(|parts| {
            parts.as_ref().map_or(f64::INFINITY, |parts| {
                let mut sum = 0.0;
                for &(root, _) in parts {
                    sum += root;
                }
                if parts.len() == 1 {
                    sum
                } else {
                    sum / parts.len() as f64
                }
            })
        })
        .collect();
    let mut d: Vec<f64> = mean.iter().map(|&x| r.caller_units(x)).collect();
    d[start] = 0.0;
    let threshold = match params.max_distance {
        MaxDistance::Value(v) => Some(v),
        MaxDistance::Default => default_threshold(&d),
    };
    let mut out = Vec::new();
    let Some(threshold) = threshold else {
        return out;
    };
    let threshold = threshold + params.atol;
    let mut alive: Vec<bool> = d.iter().map(|&x| x.is_finite() && x <= threshold).collect();
    // One dimension orders by the exact distance alone; more by the mean, except that two
    // windows whose distances are the same exact ones, in any order, tie whatever their f64 means
    // round to (the library orders by the exact means, and these are the exact ties).
    let keys = |w: usize| {
        let mut keys: Vec<R::Key> = parts[w].as_ref().unwrap().iter().map(|p| p.1).collect();
        keys.sort();
        keys
    };
    let nearer = |a: usize, b: usize| -> bool {
        if subspace.len() == 1 {
            let key = |w: usize| parts[w].as_ref().unwrap()[0].1;
            key(a) < key(b)
        } else {
            mean[a] < mean[b] && keys(a) != keys(b)
        }
    };
    let mut next = Some(start).filter(|_| alive[start]);
    while out.len() < params.max_matches.unwrap_or(usize::MAX) {
        let best = next.take().or_else(|| {
            let mut best: Option<usize> = None;
            for i in (0..windows).filter(|&i| alive[i]) {
                if best.is_none_or(|b| nearer(i, b)) {
                    best = Some(i);
                }
            }
            best
        });
        let Some(best) = best else { break };
        out.push((best as u64, d[best].to_bits()));
        for (i, a) in alive.iter_mut().enumerate() {
            if i.abs_diff(best) <= zone {
                *a = false;
            }
        }
    }
    out
}

/// The first position of the smallest value.
fn first_smallest(values: &[f64]) -> usize {
    let mut best = 0;
    for (i, &x) in values.iter().enumerate() {
        if x < values[best] {
            best = i;
        }
    }
    best
}

/// STUMPY's loop over the profile `distance`, `index`.
pub fn reference<R: RefMetric>(
    r: &R,
    distance: &[Vec<f64>],
    index: &[Vec<SampleIndex>],
    params: &MultiMotifParams<'_>,
) -> Result<Vec<Plain>, MultiError> {
    let d = r.dimensions();
    let windows = r.samples() - r.m() + 1;
    let zone = ladde::exclusion::zone(r.m());
    let cutoffs: Vec<Option<f64>> = (0..d)
        .map(|k| match params.cutoffs {
            Cutoffs::Default => default_threshold(&distance[k]),
            Cutoffs::All(c) => Some(c),
            Cutoffs::PerRow(c) => Some(c[k]),
        })
        .collect();
    let mut profile = distance.to_vec();
    let mut found = Vec::new();
    while found.len() < params.max_motifs {
        let candidates: Vec<usize> = profile.iter().map(|row| first_smallest(row)).collect();
        let neighbors: Vec<Option<usize>> = (0..d)
            .map(|k| index[k][candidates[k]].get().map(|s| s as usize))
            .collect();
        let pairs: Vec<usize> = (0..d)
            .map(|k| neighbors[k].unwrap_or(candidates[k]))
            .collect();
        let (mut sizes, mut subspaces) = r.describe(&candidates, &pairs, params.include)?;
        for (size, neighbor) in sizes.iter_mut().zip(&neighbors) {
            if neighbor.is_none() {
                *size = f64::INFINITY;
            }
        }
        let k = params.k.unwrap_or_else(|| first_smallest(&sizes));
        let (start, value) = (candidates[k], profile[k][candidates[k]]);
        if cutoffs[k].is_some_and(|c| value > c)
            || !value.is_finite()
            || neighbors[k].is_none()
            || matches!(params.max_distance, MaxDistance::Value(v) if value > v)
        {
            break;
        }
        let subspace = subspaces.swap_remove(k).dimensions;
        let found_matches = matches(r, &subspace, start, params);
        let blocked: Vec<usize> = if found_matches.is_empty() {
            vec![start]
        } else {
            found_matches.iter().map(|&(i, _)| i as usize).collect()
        };
        for at in blocked {
            for row in &mut profile {
                for (i, x) in row.iter_mut().enumerate().take(windows) {
                    if i.abs_diff(at) <= zone {
                        *x = f64::INFINITY;
                    }
                }
            }
        }
        if found_matches.len() > params.min_neighbors {
            found.push((
                found_matches,
                subspace,
                sizes.iter().map(|x| x.to_bits()).collect(),
            ));
        }
    }
    Ok(found)
}

/// A profile as the plain rows `mmotifs` takes.
pub fn rows<D: DimensionDistance>(
    profile: &MultiProfile<D>,
) -> (Vec<Vec<f64>>, Vec<Vec<SampleIndex>>) {
    (
        profile
            .distance
            .iter()
            .map(|row| row.iter().map(|d| d.to_f64()).collect())
            .collect(),
        profile.index.clone(),
    )
}

/// Parameter sets that between them reach every branch: the defaults, many motifs, every limit at
/// zero and at infinity, each way to choose `k`, and thresholds at and past the distances.
pub fn grid<'a>(
    include: &'a [usize],
    reach: f64,
    d: usize,
    cutoffs: &'a [f64],
) -> Vec<MultiMotifParams<'a>> {
    use Cutoffs::{All, Default as Auto, PerRow};
    use MaxDistance::{Default as Mean, Value};
    type Set<'a> = (
        usize,
        Option<usize>,
        usize,
        MaxDistance,
        Cutoffs<'a>,
        Option<usize>,
        f64,
    );
    let inf = f64::INFINITY;
    let sets: [Set<'a>; 13] = [
        (1, Some(10), 1, Mean, Auto, None, 1e-8),
        (1, Some(10), 3, Mean, Auto, None, 1e-8),
        (0, None, 100, Value(inf), All(inf), None, 0.0),
        (0, Some(4), 100, Mean, All(inf), None, 1e-8),
        (2, Some(3), 4, Mean, All(0.6 * reach), Some(0), 0.25),
        (0, Some(1), 100, Mean, All(inf), Some(d - 1), 0.0),
        (1, Some(0), 2, Mean, All(inf), None, 1e-8),
        (
            0,
            Some(10),
            5,
            Value(0.8 * reach),
            PerRow(cutoffs),
            None,
            1e-8,
        ),
        (0, Some(10), 5, Value(0.0), All(inf), None, 1e-8),
        (1, Some(10), 3, Mean, All(0.0), None, 1e-8),
        (1, Some(10), 3, Mean, Auto, Some(0), 0.25),
        (0, None, 100, Mean, All(inf), Some(d.min(2) - 1), 1e-8),
        (
            0,
            Some(6),
            100,
            Value(0.5 * reach),
            All(inf),
            Some(d - 1),
            0.0,
        ),
    ];
    sets.into_iter()
        .map(
            |(min_neighbors, max_matches, max_motifs, max_distance, cutoffs, k, atol)| {
                MultiMotifParams {
                    min_neighbors,
                    max_distance,
                    cutoffs,
                    max_matches,
                    max_motifs,
                    atol,
                    k,
                    include,
                }
            },
        )
        .collect()
}
