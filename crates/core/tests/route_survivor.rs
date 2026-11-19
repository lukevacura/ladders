#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! The survivor filter forced off, left to the production gate, and forced on with PAA at every
//! window length leave every left, top-k, two-sided and AB-join result equal on bits to the plain
//! pass, at 1, 2 and 8 threads, on both sides of the gate's thresholds, on every tile shape and at
//! k of 1, 4 and 16.

#[path = "common/route.rs"]
mod route;

use ladde::batch::{
    ab_join_report, ab_join_report_tiled, ab_join_topk_tiled, left_profile_report,
    left_profile_report_tiled, left_topk_profile_tiled, two_sided_profile_tiled,
    two_sided_topk_profile_tiled,
};
use ladde::entry::Entry;
use ladde::quantize::Scale;
use ladde::route::{RouteForce, SpillForce, SurvivorForce, forced_route};
use route::*;
use testkit::generators::MARKET_TICK;

/// Bands, chunks and tiles: one row and one lane, chunks narrower and wider than a part, a band
/// past one block of rows, and a tile of many chunks.
const SHAPES: [(usize, usize, usize); 4] = [(1, 1, 3), (4, 2, 16), (8, 64, 64), (64, 24, 120)];

const THREADS: [usize; 3] = [1, 2, 8];

const KS: [usize; 3] = [1, 4, 16];

/// What one entry point returns, as bits.
type Ran = Vec<Bits>;

type Family = fn(&[f64], usize, Scale, usize, usize) -> Ran;

fn topk_rows(rows: &[(u64, Vec<Entry>)]) -> Vec<Bits> {
    rows.iter()
        .flat_map(|(row, entries)| topk_bits(*row, entries))
        .collect()
}

/// The series against itself shifted by a third, so the two sides differ.
fn other_side(series: &[f64]) -> Vec<f64> {
    let cut = series.len() / 3;
    series[cut..]
        .iter()
        .chain(&series[..cut])
        .copied()
        .collect()
}

fn left_whole(series: &[f64], m: usize, scale: Scale, threads: usize, _k: usize) -> Ran {
    let (profile, _) = left_profile_report(series, nz(m), scale, nz(threads));
    profile_bits(&profile)
}

fn left_tiled(series: &[f64], m: usize, scale: Scale, threads: usize, _k: usize) -> Ran {
    let runs: Vec<_> = SHAPES
        .iter()
        .map(|&(band, chunk, tile)| {
            left_profile_report_tiled(
                series,
                nz(m),
                scale,
                nz(threads),
                nz(band),
                nz(chunk),
                nz(tile),
            )
        })
        .collect();
    runs.iter().flat_map(|(p, _)| profile_bits(p)).collect()
}

fn left_topk(series: &[f64], m: usize, scale: Scale, threads: usize, k: usize) -> Ran {
    let runs: Vec<_> = SHAPES
        .iter()
        .map(|&(band, chunk, tile)| {
            left_topk_profile_tiled(
                series,
                nz(m),
                scale,
                nz(threads),
                nz(band),
                nz(chunk),
                nz(tile),
                nz(k),
            )
        })
        .collect();
    runs.iter().flat_map(|(p, _)| topk_rows(p)).collect()
}

fn two_sided(series: &[f64], m: usize, scale: Scale, threads: usize, _k: usize) -> Ran {
    let runs: Vec<_> = SHAPES
        .iter()
        .map(|&(band, chunk, tile)| {
            two_sided_profile_tiled(
                series,
                nz(m),
                scale,
                nz(threads),
                nz(band),
                nz(chunk),
                nz(tile),
            )
        })
        .collect();
    runs.iter()
        .flat_map(|(p, _)| {
            p.iter()
                .flat_map(|&(row, t)| [t.left, t.right, t.nearest].map(|e| bits(row, e)))
        })
        .collect()
}

fn two_sided_topk(series: &[f64], m: usize, scale: Scale, threads: usize, k: usize) -> Ran {
    let runs: Vec<_> = SHAPES
        .iter()
        .map(|&(band, chunk, tile)| {
            two_sided_topk_profile_tiled(
                series,
                nz(m),
                scale,
                nz(threads),
                nz(band),
                nz(chunk),
                nz(tile),
                nz(k),
            )
            .expect("small series fit")
        })
        .collect();
    runs.iter()
        .flat_map(|(p, _)| {
            p.iter().flat_map(|(row, t)| {
                let entries: Vec<Entry> = [t.left, t.right]
                    .into_iter()
                    .chain(t.nearest.clone())
                    .collect();
                topk_bits(*row, &entries)
            })
        })
        .collect()
}

fn ab_whole(series: &[f64], m: usize, scale: Scale, threads: usize, _k: usize) -> Ran {
    let b = other_side(series);
    let (profile, _) = ab_join_report(series, &b[..], nz(m), scale, scale, nz(threads));
    profile_bits(&profile)
}

fn ab_tiled(series: &[f64], m: usize, scale: Scale, threads: usize, _k: usize) -> Ran {
    let b = other_side(series);
    let runs: Vec<_> = SHAPES
        .iter()
        .map(|&(band, chunk, tile)| {
            ab_join_report_tiled(
                series,
                &b[..],
                nz(m),
                scale,
                scale,
                nz(threads),
                nz(band),
                nz(chunk),
                nz(tile),
            )
        })
        .collect();
    runs.iter().flat_map(|(p, _)| profile_bits(p)).collect()
}

fn ab_topk(series: &[f64], m: usize, scale: Scale, threads: usize, k: usize) -> Ran {
    let b = other_side(series);
    let runs: Vec<_> = SHAPES
        .iter()
        .map(|&(band, chunk, tile)| {
            ab_join_topk_tiled(
                series,
                &b[..],
                nz(m),
                scale,
                scale,
                nz(threads),
                nz(band),
                nz(chunk),
                nz(tile),
                nz(k),
            )
            .expect("small series fit")
        })
        .collect();
    runs.iter().flat_map(|(p, _)| topk_rows(p)).collect()
}

/// Every entry point, and whether it keeps `k` neighbors.
const FAMILIES: [(&str, Family, bool); 8] = [
    ("left, whole series", left_whole, false),
    ("left tiled", left_tiled, false),
    ("left top-k", left_topk, true),
    ("two-sided", two_sided, false),
    ("two-sided top-k", two_sided_topk, true),
    ("ab-join, whole series", ab_whole, false),
    ("ab-join tiled", ab_tiled, false),
    ("ab-join top-k", ab_topk, true),
];

fn with(base: RouteForce, survivor: SurvivorForce) -> RouteForce {
    RouteForce { survivor, ..base }
}

/// Each entry point at each thread count and `k`, under `base`: the filter off, the production
/// gate's choice and PAA forced on at every window length all give the plain pass's bits.
fn check(series: &[f64], m: usize, scale: Scale, base: RouteForce) {
    check_families(&FAMILIES, series, m, scale, base)
}

/// [`check`] over `families` only.
fn check_families(
    families: &[(&'static str, Family, bool)],
    series: &[f64],
    m: usize,
    scale: Scale,
    base: RouteForce,
) {
    for &(name, family, keeps_k) in families {
        let ks: &[usize] = if keeps_k { &KS } else { &[1] };
        for &threads in &THREADS {
            for &k in ks {
                let run = |survivor| {
                    let _g = forced_route(with(base, survivor));
                    family(series, m, scale, threads, k)
                };
                let plain = run(SurvivorForce::Off);
                assert_eq!(
                    run(SurvivorForce::Gate),
                    plain,
                    "{name} gate, {threads} threads, k {k}"
                );
                assert_eq!(
                    run(SurvivorForce::Paa),
                    plain,
                    "{name} paa, {threads} threads, k {k}"
                );
            }
        }
    }
}

#[test]
fn the_filter_keeps_every_bit_in_every_engine() {
    let scale = Scale::new(MARKET_TICK).unwrap();
    check(&walk(700), 8, scale, routed());
}

#[test]
fn the_filter_keeps_every_bit_on_every_route() {
    let (series, m, scale) = (walk(500), 8, Scale::new(MARKET_TICK).unwrap());
    for base in [
        RouteForce {
            wide: true,
            ..routed()
        },
        RouteForce {
            reanchor_cell: 3,
            ..routed()
        },
        RouteForce {
            per_tile: true,
            anchor_seed: 9,
            ..routed()
        },
        RouteForce {
            resume_every: 2,
            ..routed()
        },
    ] {
        check(&series, m, scale, base);
    }
}

/// Exact repeats and near-ties: a period that repeats exactly, and one that repeats up to a tick.
#[test]
fn the_filter_keeps_every_bit_on_ties() {
    let scale = Scale::new(1.0).unwrap();
    let exact: Vec<f64> = (0..150)
        .map(|k| f64::from([3, 7, 1, 9, 4, 4, 2][k % 7]))
        .collect();
    let near: Vec<f64> = (0..150)
        .map(|k| f64::from([30, 70, 10, 90, 40][k % 5]) + f64::from(u8::from(k % 11 == 0)))
        .collect();
    for series in [exact, near] {
        check(&series, 6, scale, routed());
    }
}

/// Flat runs between levels (constant windows, which the filter never skips), gaps at the very
/// start, across tiles and at the end, and a print far off the level that the cold route stands in.
#[test]
fn the_filter_keeps_every_bit_on_flat_dirty_and_stood_in_windows() {
    let (walk, m, scale) = long_walk();
    let mut series = walk[..150].to_vec();
    for flat in [20..40, 90..104] {
        let level = series[flat.start];
        series[flat].fill(level);
    }
    for at in [0, 1, 17, 64, 65, 148, 149] {
        series[at] = f64::NAN;
    }
    check(&series, m, scale, routed());
    series[120] += 1.0e9;
    check(
        &series,
        m,
        scale,
        RouteForce {
            spill: SpillForce::On,
            ..routed()
        },
    );
}

/// Ticks near the largest exact tick count, on two levels no anchor covers together: the pass runs
/// the i64 arithmetic, and the exact producer reads cross terms that wrap.
#[test]
fn the_filter_keeps_every_bit_near_the_integer_limits() {
    let top = (1i64 << 53) as f64 - 4096.0;
    let series: Vec<f64> = (0..120)
        .map(|k| {
            let level = if (k / 23) % 2 == 0 { top } else { -top };
            level + ((k * 37) % 11) as f64
        })
        .collect();
    check(&series, 4, Scale::new(1.0).unwrap(), routed());
}

/// The gate's choice on both sides of each of its thresholds keeps every bit, as does every forced
/// filter there, in every entry point at every thread count, tile shape and `k`: on a walk whose
/// flat run is longer than a window (constant windows) with gaps at its ends and inside, and on a
/// period of 7 ticks with a tick added every 11 (windows 77 apart repeat exactly, 7 apart nearly).
/// Each pair of window lengths straddles one of the gate's thresholds (256, 512 and 768).
#[test]
fn the_gate_keeps_every_bit_on_both_sides_of_its_thresholds() {
    let scale = Scale::new(MARKET_TICK).unwrap();
    let self_joins: Vec<_> = FAMILIES
        .into_iter()
        .filter(|(name, _, _)| !name.starts_with("ab-join"))
        .collect();
    for hi in [256, 512, 768] {
        let n = hi + hi / 4 + 60;
        let scaled = |at: usize| at * n / 1400;
        let mut dirty = walk(n);
        let level = dirty[scaled(150)];
        dirty[scaled(150)..scaled(1210)].fill(level);
        for at in [0, 1, scaled(90), scaled(1250), n - 1] {
            dirty[at] = f64::NAN;
        }
        let ties: Vec<f64> = (0..n)
            .map(|k| {
                (f64::from([3, 7, 1, 9, 4, 4, 2][k % 7]) + f64::from(u8::from(k % 11 == 0)))
                    * MARKET_TICK
            })
            .collect();
        for m in [hi - 1, hi] {
            check_families(&self_joins, &dirty, m, scale, routed());
            check_families(&self_joins, &ties, m, scale, routed());
        }
    }
}

/// A walk of `n` samples in the manner of [`long_walk`].
fn walk(n: usize) -> Vec<f64> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut price = 100.0;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            price += ((state >> 33) % 7) as f64 * MARKET_TICK - 3.0 * MARKET_TICK;
            price
        })
        .collect()
}
