//! A plain statement of STUMPY's `aamp_match`, `aamp_motifs`, `aampdist` and `aamp_ostinato` on
//! exact integer ticks: squared
//! distances are sums of squared tick differences, every order is on those integers (ties to the
//! smaller start), and only the thresholds are f64 distances in the caller's units.
#![allow(dead_code)]

use ladde::motif::MotifParams;
use ladde::pattern::MaxDistance;
use ladde::quantize::{Scale, to_ticks};

pub type Ticks = Vec<Option<i64>>;

/// A match as `(start, squared distance in ticks)`.
pub type Group = Vec<(u64, u128)>;

pub fn ticks(series: &[f64], scale: Scale) -> Ticks {
    series.iter().map(|&x| to_ticks(x, scale)).collect()
}

pub fn window(t: &Ticks, start: usize, m: usize) -> Option<Vec<i64>> {
    t[start..start + m].iter().copied().collect()
}

pub fn windows(t: &Ticks, m: usize) -> usize {
    (t.len() + 1).saturating_sub(m)
}

pub fn squared(a: &[i64], b: &[i64]) -> u128 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = i128::from(x) - i128::from(y);
            (d * d) as u128
        })
        .sum()
}

/// The Euclidean distance in the caller's units, as a reader of the squared ticks computes it.
pub fn distance(squared_ticks: u128, scale: Scale) -> f64 {
    libm::sqrt(squared_ticks as f64) * scale.get()
}

/// The squared distance from `query` to every window of `t`, `None` for a window with a gap.
pub fn profile(query: &[i64], t: &Ticks) -> Vec<Option<u128>> {
    (0..windows(t, query.len()))
        .map(|i| window(t, i, query.len()).map(|w| squared(query, &w)))
        .collect()
}

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

/// STUMPY's `match` loop on a squared-distance profile: with an `anchor`, that window first at
/// distance zero; then the exactly nearest remaining window within the threshold, blanking the
/// zone around each taken one, until `limit`.
pub fn brute_matches(
    profile: &[Option<u128>],
    m: usize,
    scale: Scale,
    max_distance: MaxDistance,
    limit: Option<usize>,
    anchor: Option<usize>,
) -> Group {
    let zone = ladde::exclusion::zone(m);
    let mut d: Vec<f64> = profile
        .iter()
        .map(|p| p.map_or(f64::INFINITY, |s| distance(s, scale)))
        .collect();
    if let Some(anchor) = anchor {
        d[anchor] = 0.0;
    }
    let threshold = match max_distance {
        MaxDistance::Value(v) => Some(v),
        MaxDistance::Default => default_threshold(&d),
    };
    let mut out = Vec::new();
    let Some(threshold) = threshold else {
        return out;
    };
    let mut alive: Vec<bool> = d.iter().map(|&x| x.is_finite() && x <= threshold).collect();
    let mut next = anchor.filter(|&a| alive[a]);
    while out.len() < limit.unwrap_or(usize::MAX) {
        let best = next.take().or_else(|| {
            (0..alive.len())
                .filter(|&i| alive[i])
                .min_by_key(|&i| (profile[i].expect("a live window has a distance"), i))
        });
        let Some(best) = best else { break };
        let squared = if Some(best) == anchor {
            0
        } else {
            profile[best].unwrap()
        };
        out.push((best as u64, squared));
        for (i, flag) in alive.iter_mut().enumerate() {
            if i.abs_diff(best) <= zone {
                *flag = false;
            }
        }
    }
    out
}

/// Each window's exact nearest neighbor outside its zone, on either side: `(squared, start)`.
pub fn nearest(t: &Ticks, m: usize) -> Vec<Option<(u128, usize)>> {
    let zone = ladde::exclusion::zone(m);
    let count = windows(t, m);
    (0..count)
        .map(|i| {
            let own = window(t, i, m)?;
            let mut best: Option<(u128, usize)> = None;
            for j in (0..count).filter(|&j| i.abs_diff(j) > zone) {
                if let Some(other) = window(t, j, m) {
                    let candidate = (squared(&own, &other), j);
                    if best.is_none_or(|b| candidate < b) {
                        best = Some(candidate);
                    }
                }
            }
            best
        })
        .collect()
}

/// STUMPY's `aamp_motifs` loop: candidates by exact nearest distance (ties to the smaller start),
/// each matched with [`brute_matches`] without a limit, every matched window blanking its zone
/// for later candidates, and the first `max_matches` of them kept.
pub fn brute_motifs(series: &[f64], m: usize, scale: Scale, params: &MotifParams) -> Vec<Group> {
    let t = ticks(series, scale);
    let zone = ladde::exclusion::zone(m);
    let near = nearest(&t, m);
    let distances: Vec<f64> = near
        .iter()
        .map(|n| n.map_or(f64::INFINITY, |(s, _)| distance(s, scale)))
        .collect();
    let pick = |excluded: &[bool]| {
        (0..near.len())
            .filter(|&i| !excluded[i] && near[i].is_some())
            .min_by_key(|&i| (near[i].unwrap().0, i))
    };
    let mut found = Vec::new();
    let mut excluded = vec![false; near.len()];
    let Some(first) = pick(&excluded) else {
        return found;
    };
    // The default cutoff is never below the exactly nearest window's distance.
    let cutoff = params
        .cutoff
        .unwrap_or_else(|| default_threshold(&distances).unwrap().max(distances[first]));
    while found.len() < params.max_motifs {
        let Some(candidate) = pick(&excluded) else {
            break;
        };
        let d = distances[candidate];
        if d > cutoff || !d.is_finite() {
            break;
        }
        if matches!(params.max_distance, MaxDistance::Value(v) if d > v) {
            break;
        }
        let own = window(&t, candidate, m).expect("a candidate has a window");
        let matches = brute_matches(
            &profile(&own, &t),
            m,
            scale,
            params.max_distance,
            None,
            Some(candidate),
        );
        let blocked: Vec<usize> = if matches.is_empty() {
            vec![candidate]
        } else {
            matches.iter().map(|&(i, _)| i as usize).collect()
        };
        for at in blocked {
            for (i, flag) in excluded.iter_mut().enumerate() {
                if i.abs_diff(at) <= zone {
                    *flag = true;
                }
            }
        }
        if matches.len() > params.min_neighbors {
            let kept = params.max_matches.unwrap_or(usize::MAX).min(matches.len());
            found.push(matches[..kept].to_vec());
        }
    }
    found
}

/// How a test names the position of `aampdist`: STUMPY's `percentage` or its `k`.
#[derive(Clone, Copy, Debug)]
pub enum Pick {
    Percentage(f64),
    Rank(usize),
}

/// Each window of `rows`' exactly nearest window of `neighbors`: `(squared distance, start)`,
/// ties to the smaller start, `None` for a window with a gap or without a window to meet.
pub fn join(rows: &Ticks, neighbors: &Ticks, m: usize) -> Vec<Option<(u128, usize)>> {
    (0..windows(rows, m))
        .map(|i| {
            let own = window(rows, i, m)?;
            (0..windows(neighbors, m))
                .filter_map(|j| Some((squared(&own, &window(neighbors, j, m)?), j)))
                .min()
        })
        .collect()
}

/// STUMPY's `aampdist`: the AB-join then the BA-join distances (a window with none is infinity),
/// sorted by exact squared distance (equal ones AB-join first, then by window), the value at the
/// position, and STUMPY's fallback to the largest finite one before it when that is infinite;
/// `None` where STUMPY would clamp a percentage outside `[0, 1]` or a rank past the last position.
pub fn brute_aampdist(a: &[f64], b: &[f64], m: usize, scale: Scale, pick: Pick) -> Option<f64> {
    let (ta, tb) = (ticks(a, scale), ticks(b, scale));
    if windows(&ta, m) == 0 || windows(&tb, m) == 0 {
        return None;
    }
    let mut joined: Vec<(u128, usize, usize)> = Vec::new();
    for (side, rows, neighbors) in [(0, &ta, &tb), (1, &tb, &ta)] {
        for (i, nearest) in join(rows, neighbors, m).into_iter().enumerate() {
            if let Some((sq, _)) = nearest {
                joined.push((sq, side, i));
            }
        }
    }
    let total = windows(&ta, m) + windows(&tb, m);
    let position = match pick {
        Pick::Percentage(p) if (0.0..=1.0).contains(&p) => {
            ((p * (a.len() + b.len()) as f64).ceil() as usize).min(total - 1)
        }
        Pick::Rank(k) if k < total => k,
        // STUMPY clamps these; this library refuses them.
        _ => return None,
    };
    joined.sort();
    let mut sorted: Vec<f64> = joined
        .iter()
        .map(|&(sq, _, _)| distance(sq, scale))
        .collect();
    sorted.resize(total, f64::INFINITY);
    if sorted[position].is_finite() {
        return Some(sorted[position]);
    }
    let finite_before = sorted[..position].iter().filter(|d| d.is_finite()).count();
    Some(sorted[finite_before.saturating_sub(1)])
}

/// NumPy's `isclose(a, b)` with its default tolerances.
pub fn is_close(a: f64, b: f64) -> bool {
    a == b || (a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-8 + 1e-5 * b.abs())
}

/// The exactly nearest window of `target` to the window of `from` at `at`, `None` without one.
fn nearest_of(from: &Ticks, at: usize, target: &Ticks, m: usize) -> Option<(u128, usize)> {
    let query = window(from, at, m)?;
    (0..windows(target, m))
        .filter_map(|j| Some((squared(&query, &window(target, j, m)?), j)))
        .min()
}

/// Each series' nearest distance to the window `at` of `set[from]` and its start; infinity at
/// start 0 for a series with none, as `argmin` over infinities gives.
fn across(set: &[Ticks], from: usize, at: usize, m: usize, scale: Scale) -> (Vec<f64>, Vec<usize>) {
    set.iter()
        .map(|target| match nearest_of(&set[from], at, target, m) {
            Some((sq, start)) => (distance(sq, scale), start),
            None => (f64::INFINITY, 0),
        })
        .unzip()
}

/// STUMPY's `aamp_ostinato` with the order between distances exact: for each series, its windows
/// by ascending join distance to the next series (the last to the first), each raised to its
/// nearest distance in every other series, replacing the best only when strictly smaller; then
/// the central pass over the nearest windows whose distances are close to the radius. Means are
/// plain sums, which equal NumPy's below eight series.
pub fn brute_aamp_ostinato(
    series: &[Vec<f64>],
    m: usize,
    scale: Scale,
) -> Option<(f64, usize, usize)> {
    if series.is_empty() || series.iter().any(|s| s.len() < m) {
        return None;
    }
    let set: Vec<Ticks> = series.iter().map(|s| ticks(s, scale)).collect();
    let k = set.len();
    let mut best: Option<(u128, usize, usize)> = None;
    for j in 0..k {
        let h = (j + 1) % k;
        let mut order: Vec<(u128, usize)> = join(&set[j], &set[h], m)
            .into_iter()
            .enumerate()
            .filter_map(|(q, nearest)| Some((nearest?.0, q)))
            .collect();
        order.sort();
        for (mut radius, q) in order {
            if best.is_some_and(|(b, _, _)| radius >= b) {
                break;
            }
            let mut finite = true;
            for i in (0..k).filter(|&i| i != j && i != h) {
                let Some((sq, _)) = nearest_of(&set[j], q, &set[i], m) else {
                    finite = false;
                    break;
                };
                radius = radius.max(sq);
                if best.is_some_and(|(b, _, _)| radius >= b) {
                    break;
                }
            }
            if finite && best.is_none_or(|(b, _, _)| radius < b) {
                best = Some((radius, j, q));
            }
        }
    }
    let Some((sq, mut series_at, mut window_at)) = best else {
        return Some((f64::INFINITY, 0, 0));
    };
    let radius = distance(sq, scale);
    let mean = |values: &[f64]| values.iter().fold(0.0, |s, &x| s + x) / values.len() as f64;
    let (radii, starts) = across(&set, series_at, window_at, m, scale);
    let mut best_mean = mean(&radii);
    for candidate in 0..k {
        if !is_close(radii[candidate], radius) {
            continue;
        }
        let (candidate_radii, _) = across(&set, candidate, starts[candidate], m, scale);
        let farthest = candidate_radii
            .iter()
            .fold(f64::NEG_INFINITY, |a, &x| a.max(x));
        if is_close(farthest, radius) && mean(&candidate_radii) < best_mean {
            series_at = candidate;
            window_at = starts[candidate];
            best_mean = mean(&candidate_radii);
        }
    }
    Some((radius, series_at, window_at))
}
