#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! A constant row's own admission, and the same rows admitted lane by lane, equal the exact
//! reference on bits in every engine, on tie-heavy series whose constant windows fall at tile,
//! chunk and stream boundaries and next to gaps, under the caller's flags too; and forcing the
//! lane-by-lane admission fires in every engine.

#[path = "common/route.rs"]
mod route;

use ladde::batch::{left_topk_profile_tiled, two_sided_topk_profile_tiled};
use ladde::bounded::BoundedStream;
use ladde::entry::Entry;
use ladde::naive::{Ticks, naive_left_topk, naive_two_sided_topk};
use ladde::quantize::Scale;
use ladde::route::{RouteForce, forced_route};
use ladde::stats::Series;
use ladde::stream::GrowingStream;
use route::*;

const KS: [usize; 4] = [1, 2, 3, 8];
const TILINGS: [(usize, usize, usize); 4] = [(1, 1, 1), (3, 2, 5), (8, 4, 4), (64, 8, 64)];

fn unit() -> Scale {
    Scale::new(1.0).unwrap()
}

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *seed >> 33
}

/// Random levels held for random runs, some shorter than `m` and some much longer.
fn runs_between_levels(n: usize, seed: u64, levels: u64) -> Vec<f64> {
    let mut seed = seed;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let level = (lcg(&mut seed) % levels) as f64;
        let run = 1 + (lcg(&mut seed) % 24) as usize;
        out.extend(std::iter::repeat_n(level, run));
    }
    out.truncate(n);
    out
}

/// Tie-heavy series whose constant windows land on every boundary the tilings and stream
/// chunks cut at, next to gaps, alone, and filling the whole series.
fn series_set() -> Vec<Vec<f64>> {
    let mut gapped = runs_between_levels(90, 3, 3);
    for i in (0..2).chain(30..32).chain(33..35).chain(88..90) {
        gapped[i] = f64::NAN;
    }
    let mut one_flat = runs_between_levels(70, 11, 1000);
    for x in &mut one_flat[40..52] {
        *x = 7.0;
    }
    let mut seed = 5u64;
    let noisy: Vec<f64> = (0..80).map(|_| (lcg(&mut seed) % 1000) as f64).collect();
    let mut flat_then_noise = vec![2.0; 40];
    flat_then_noise.extend(&noisy[..40]);
    let mut noise_then_flat = noisy[..40].to_vec();
    noise_then_flat.extend(vec![2.0; 40]);
    vec![
        vec![5.0; 60],
        runs_between_levels(120, 1, 2),
        runs_between_levels(120, 2, 4),
        gapped,
        one_flat,
        flat_then_noise,
        noise_then_flat,
        (0..64).map(|i| ((i / 9) % 2) as f64).collect(),
    ]
}

fn entries_bits(entries: &[Entry]) -> Vec<Bits> {
    entries.iter().map(|&e| bits(0, e)).collect()
}

fn topk_profile_bits(profile: &[(u64, Vec<Entry>)]) -> Vec<(u64, Vec<Bits>)> {
    profile
        .iter()
        .map(|(row, es)| (*row, entries_bits(es)))
        .collect()
}

/// The admission forced each way, on the narrow and the wide pass.
fn admissions() -> [RouteForce; 4] {
    let none = RouteForce::default();
    [
        none,
        RouteForce {
            constant_rows_by_lane: true,
            ..none
        },
        RouteForce { wide: true, ..none },
        RouteForce {
            constant_rows_by_lane: true,
            wide: true,
            ..none
        },
    ]
}

/// Batch left and two-sided top-k profiles under the caller's `flags` (none for the default
/// rule) equal the naive reference under every admission, tiling and thread count.
fn check_batch(series: &[f64], m: usize, flags: Option<&[bool]>) {
    let ticks = samples(series, unit());
    let reference = match flags {
        Some(flags) => Ticks::with_constant_windows(&ticks, flags),
        None => Ticks::from(&ticks),
    };
    let input = || match flags {
        Some(flags) => Series::with_constant_windows(series, nz(m), flags).unwrap(),
        None => Series::from(series),
    };
    for k in KS {
        let left = topk_profile_bits(&naive_left_topk(reference, m, nz(k)));
        let sides: Vec<_> = naive_two_sided_topk(reference, m, nz(k))
            .into_iter()
            .map(|(row, s)| {
                (
                    row,
                    [s.left, s.right].map(|e| bits(0, e)),
                    entries_bits(&s.nearest),
                )
            })
            .collect();
        for force in admissions() {
            let _forced = forced_route(force);
            for threads in [1, 3] {
                for (band, chunk, tile) in TILINGS {
                    let shape = (nz(threads), nz(band), nz(chunk), nz(tile));
                    let (got, _) = left_topk_profile_tiled(
                        input(),
                        nz(m),
                        unit(),
                        shape.0,
                        shape.1,
                        shape.2,
                        shape.3,
                        nz(k),
                    );
                    let context = (force, k, m, threads, (band, chunk, tile));
                    assert_eq!(topk_profile_bits(&got), left, "left {context:?}");
                    let (got, _) = two_sided_topk_profile_tiled(
                        input(),
                        nz(m),
                        unit(),
                        shape.0,
                        shape.1,
                        shape.2,
                        shape.3,
                        nz(k),
                    )
                    .unwrap();
                    let got: Vec<_> = got
                        .iter()
                        .map(|(row, s)| {
                            (
                                *row,
                                [s.left, s.right].map(|e| bits(0, e)),
                                entries_bits(&s.nearest),
                            )
                        })
                        .collect();
                    assert_eq!(got, sides, "two-sided {context:?}");
                }
            }
        }
    }
}

#[test]
fn batch_constant_rows_equal_the_reference_under_either_admission() {
    for series in series_set() {
        for m in [3, 4, 8] {
            check_batch(&series, m, None);
        }
    }
}

/// Flags that make windows of equal ticks valid (uncorrelated with everything) and windows of
/// unequal ticks constant.
fn flag_sets(series: &[f64], m: usize) -> Vec<Vec<bool>> {
    let flat = |w: &[f64]| w.iter().all(|x| *x == w[0]);
    let mut seed = series.len() as u64 * 31 + m as u64;
    vec![
        vec![false; series.len() + 1 - m],
        series.windows(m).map(|w| !flat(w)).collect(),
        series
            .windows(m)
            .map(|_| lcg(&mut seed).is_multiple_of(3))
            .collect(),
    ]
}

#[test]
fn flagged_constant_rows_equal_the_reference_under_either_admission() {
    for series in series_set() {
        let m = 4;
        let ticks = samples(&series, unit());
        for flags in flag_sets(&series, m) {
            check_batch(&series, m, Some(&flags));
            let reference = Ticks::with_constant_windows(&ticks, &flags);
            for k in KS {
                let expected = topk_profile_bits(&naive_left_topk(reference, m, nz(k)));
                for force in admissions() {
                    let _forced = forced_route(force);
                    let mut stream = GrowingStream::with_k(nz(m), unit(), nz(k));
                    let mut got = Vec::new();
                    for (i, &x) in series.iter().enumerate() {
                        let flag = i.checked_sub(m - 1).map(|row| flags[row]);
                        if let Some((row, es)) = stream.update_topk_flagged(x, flag) {
                            got.push((row.get().unwrap(), entries_bits(es)));
                        }
                    }
                    assert_eq!(got, expected, "{force:?} k {k}");
                }
            }
        }
    }
}

#[test]
fn stream_constant_rows_equal_the_reference_under_either_admission() {
    for series in series_set() {
        let ticks = samples(&series, unit());
        for m in [3, 8] {
            for k in KS {
                let growing = stream_topk_rows(&ticks, m, k);
                for force in admissions() {
                    let _forced = forced_route(force);
                    assert_eq!(
                        growing_topk_rows(&series, m, unit(), k),
                        growing,
                        "{force:?} m {m} k {k}"
                    );
                    for w in [m + 1, m + 13, 64] {
                        for (row, got) in bounded_topk_rows(&series, m, w, unit(), k) {
                            assert_eq!(
                                got,
                                stream_window_topk_bits(&ticks, m, w, row as usize, k),
                                "{force:?} m {m} w {w} k {k}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn lane_by_lane_constant_rows_fire_in_every_engine() {
    let series = runs_between_levels(200, 9, 3);
    let (m, k) = (4, 2);
    for wide in [false, true] {
        let force = RouteForce {
            constant_rows_by_lane: true,
            wide,
            ..RouteForce::default()
        };
        let fired = |run: &dyn Fn()| {
            let guard = forced_route(force);
            run();
            guard.fired().constant_row_spans_by_lane
        };
        let batch = || {
            let _ = left_topk_profile_tiled(
                &series[..],
                nz(m),
                unit(),
                nz(2),
                nz(8),
                nz(4),
                nz(16),
                nz(k),
            );
        };
        let growing = || {
            let mut stream = GrowingStream::with_k(nz(m), unit(), nz(k));
            for &x in &series {
                let _ = stream.update_topk(x);
            }
        };
        let bounded = || {
            let mut stream = BoundedStream::with_k(nz(m), nz(40), unit(), nz(k)).unwrap();
            for &x in &series {
                let _ = stream.update_topk(x);
            }
        };
        assert!(fired(&batch) > 0, "batch wide {wide}");
        assert!(fired(&growing) > 0, "growing wide {wide}");
        assert!(fired(&bounded) > 0, "bounded wide {wide}");
    }
    let guard = forced_route(RouteForce::default());
    let mut stream = GrowingStream::with_k(nz(m), unit(), nz(k));
    for &x in &series {
        let _ = stream.update_topk(x);
    }
    assert_eq!(guard.fired().constant_row_spans_by_lane, 0);
}
