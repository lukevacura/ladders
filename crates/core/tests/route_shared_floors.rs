#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! The survivor filter's floors shared between a self-join run's tiles and threads: one floor per
//! window as a row and as a column, raised by an atomic max as blocks run and tiles close, and read
//! before each gate (`FloorForce::Shared`, the library's route), against each tile's own partials
//! alone (`FloorForce::PerThread`). Every left, top-k and two-sided result stays equal on bits to
//! the plain pass at 1, 2 and 8 threads, on every tile shape, at k of 1, 4 and 16, under the PAA
//! filter, on ties, flats and gaps, and on the exact i64 and re-anchoring tile paths. An AB-join
//! neither reads nor raises the table.

#[path = "common/route.rs"]
mod route;

use ladde::batch::{
    ab_join_report_tiled, left_profile_report, left_profile_report_tiled, left_topk_profile_tiled,
    two_sided_profile_tiled, two_sided_topk_profile_tiled,
};
use ladde::entry::Entry;
use ladde::quantize::Scale;
use ladde::route::{FloorForce, RouteForce, SurvivorForce, forced_route};
use route::*;
use testkit::generators::MARKET_TICK;

/// Bands, chunks and tiles: one row and one lane, chunks narrower and wider than a part, a band
/// past one block of rows, and a tile of many chunks.
const SHAPES: [(usize, usize, usize); 4] = [(1, 1, 3), (4, 2, 16), (8, 64, 64), (64, 24, 120)];

const THREADS: [usize; 3] = [1, 2, 8];

const KS: [usize; 3] = [1, 4, 16];

const FLOORS: [FloorForce; 2] = [FloorForce::Shared, FloorForce::PerThread];

/// An entry point's bits.
type Ran = Vec<Bits>;

/// An entry point over every shape: series, m, scale, threads, k.
type Family = fn(&[f64], usize, Scale, usize, usize) -> Ran;

fn topk_rows(rows: &[(u64, Vec<Entry>)]) -> Vec<Bits> {
    rows.iter()
        .flat_map(|(row, entries)| topk_bits(*row, entries))
        .collect()
}

fn left_whole(series: &[f64], m: usize, scale: Scale, threads: usize, _k: usize) -> Ran {
    let (profile, _) = left_profile_report(series, nz(m), scale, nz(threads));
    profile_bits(&profile)
}

fn left_tiled(series: &[f64], m: usize, scale: Scale, threads: usize, _k: usize) -> Ran {
    let runs: Vec<_> = SHAPES
        .iter()
        .map(|&(b, c, t)| {
            left_profile_report_tiled(series, nz(m), scale, nz(threads), nz(b), nz(c), nz(t))
        })
        .collect();
    runs.iter().flat_map(|(p, _)| profile_bits(p)).collect()
}

fn left_topk(series: &[f64], m: usize, scale: Scale, threads: usize, k: usize) -> Ran {
    let runs: Vec<_> = SHAPES
        .iter()
        .map(|&(b, c, t)| {
            left_topk_profile_tiled(
                series,
                nz(m),
                scale,
                nz(threads),
                nz(b),
                nz(c),
                nz(t),
                nz(k),
            )
        })
        .collect();
    runs.iter().flat_map(|(p, _)| topk_rows(p)).collect()
}

fn two_sided(series: &[f64], m: usize, scale: Scale, threads: usize, _k: usize) -> Ran {
    let runs: Vec<_> = SHAPES
        .iter()
        .map(|&(b, c, t)| {
            two_sided_profile_tiled(series, nz(m), scale, nz(threads), nz(b), nz(c), nz(t))
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
    two_sided_topk_on(&SHAPES, series, m, scale, threads, k)
}

fn two_sided_topk_on(
    shapes: &[(usize, usize, usize)],
    series: &[f64],
    m: usize,
    scale: Scale,
    threads: usize,
    k: usize,
) -> Ran {
    let runs: Vec<_> = shapes
        .iter()
        .map(|&(b, c, t)| {
            two_sided_topk_profile_tiled(
                series,
                nz(m),
                scale,
                nz(threads),
                nz(b),
                nz(c),
                nz(t),
                nz(k),
            )
            .expect("small series fit")
        })
        .collect();
    runs.iter()
        .flat_map(|(p, _)| {
            p.iter().flat_map(|(row, t)| {
                [t.left, t.right]
                    .into_iter()
                    .chain(t.nearest.iter().copied())
                    .map(|e| bits(*row, e))
            })
        })
        .collect()
}

/// Every self-join entry point, and whether it keeps `k` neighbors.
const FAMILIES: [(&str, Family, bool); 5] = [
    ("left, whole series", left_whole, false),
    ("left tiled", left_tiled, false),
    ("left top-k", left_topk, true),
    ("two-sided", two_sided, false),
    ("two-sided top-k", two_sided_topk, true),
];

fn with(base: RouteForce, survivor: SurvivorForce, floors: FloorForce) -> RouteForce {
    RouteForce {
        survivor,
        floors,
        ..base
    }
}

/// One family under each floor source, its bits asserted equal to the plain pass's.
fn run_family(
    (name, family): (&str, Family),
    (series, m, scale): (&[f64], usize, Scale),
    (threads, k): (usize, usize),
    (base, producer): (RouteForce, SurvivorForce),
) {
    let plain = {
        let _g = forced_route(with(base, SurvivorForce::Off, FloorForce::Shared));
        family(series, m, scale, threads, k)
    };
    for floors in FLOORS {
        let g = forced_route(with(base, producer, floors));
        let got = family(series, m, scale, threads, k);
        if got != plain {
            let first = got.iter().zip(&plain).position(|(a, b)| a != b);
            panic!(
                "{name} {producer:?} {floors:?} {base:?}, {threads} threads, k {k}: first difference at {first:?} of {}: {:?} against {:?}",
                plain.len(),
                first.map(|i| got[i]),
                first.map(|i| plain[i])
            );
        }
        if floors == FloorForce::PerThread {
            assert_eq!(
                g.fired().floor_raises,
                0,
                "{name}: per-thread floors raise nothing"
            );
        }
    }
}

/// A sum of sines with a slow drift, in ticks: the kind of series the PAA bound prunes on.
fn smooth(n: usize) -> Vec<f64> {
    (0..n)
        .map(|p| {
            let t = p as f64;
            let wave = 40.0 * (t / 37.0).sin() + 25.0 * (t / 101.0).sin() + 10.0 * (t / 13.0).cos();
            (wave + t / 50.0).round() * MARKET_TICK
        })
        .collect()
}

#[test]
fn shared_floors_keep_every_bit_at_every_thread_count_shape_and_k() {
    let (series, m, scale) = (smooth(600), 16, Scale::new(MARKET_TICK).unwrap());
    for (name, family, keeps_k) in FAMILIES {
        let ks: &[usize] = if keeps_k { &KS } else { &[1] };
        for &threads in &THREADS {
            for &k in ks {
                let producer = SurvivorForce::Paa;
                run_family(
                    (name, family),
                    (&series, m, scale),
                    (threads, k),
                    (routed(), producer),
                );
            }
        }
    }
}

/// Exact repeats, flat runs (constant windows) and gaps: the table never takes a NaN or a missing
/// window's floor, and constant blocks stay visited.
#[test]
fn shared_floors_keep_every_bit_on_ties_flats_and_gaps() {
    let scale = Scale::new(1.0).unwrap();
    let mut series: Vec<f64> = (0..300)
        .map(|k| f64::from([3, 7, 1, 9, 4, 4, 2][k % 7]) + f64::from(u8::from(k % 53 == 0)))
        .collect();
    series[100..130].fill(5.0);
    for at in [0, 17, 64, 65, 299] {
        series[at] = f64::NAN;
    }
    for (name, family, keeps_k) in FAMILIES {
        let ks: &[usize] = if keeps_k { &KS } else { &[1] };
        for &threads in &THREADS {
            for &k in ks {
                let producer = SurvivorForce::Paa;
                run_family(
                    (name, family),
                    (&series, 6, scale),
                    (threads, k),
                    (routed(), producer),
                );
            }
        }
    }
}

/// The exact i64 tiles and the re-anchoring tiles raise and read the table too, and keep every
/// bit.
#[test]
fn shared_floors_keep_every_bit_on_the_wide_and_reanchoring_tiles() {
    let (series, m, scale) = (smooth(500), 16, Scale::new(MARKET_TICK).unwrap());
    let wide = RouteForce {
        wide: true,
        ..routed()
    };
    let cells = RouteForce {
        reanchor_cell: 7,
        ..routed()
    };
    for base in [wide, cells] {
        for (name, family, keeps_k) in FAMILIES {
            let k = if keeps_k { 4 } else { 1 };
            for &threads in &[1, 8] {
                run_family(
                    (name, family),
                    (&series, m, scale),
                    (threads, k),
                    (base, SurvivorForce::Paa),
                );
            }
        }
    }
}

/// The column pitfall at the engine: in two-sided top-k with bands of fewer rows than `k`, a
/// column partial holds room for only the band's rows, so its floor is above the run's `k`-th
/// best. Sharing it would skip pairs of the column's true top `k` (the study found this broke
/// top-k at k 4); a skip on such a floor shows in the bits.
#[test]
fn two_sided_topk_with_bands_shorter_than_k_keeps_every_bit() {
    let (series, m, scale) = (smooth(400), 16, Scale::new(MARKET_TICK).unwrap());
    for k in [4, 16] {
        for threads in [1, 2] {
            run_family(
                ("two-sided top-k, short bands", short_bands),
                (&series, m, scale),
                (threads, k),
                (routed(), SurvivorForce::Paa),
            );
        }
    }
}

/// Bands of one to three rows, each over tiles of several chunks.
const SHORT_BANDS: [(usize, usize, usize); 3] = [(1, 1, 3), (2, 8, 16), (3, 4, 40)];

fn short_bands(series: &[f64], m: usize, scale: Scale, threads: usize, k: usize) -> Ran {
    two_sided_topk_on(&SHORT_BANDS, series, m, scale, threads, k)
}

/// An AB-join's rows and columns are two series' windows, so it neither reads nor raises a table,
/// and its bits are unchanged.
#[test]
fn a_join_neither_reads_nor_raises_shared_floors() {
    let (series, m, scale) = (smooth(500), 16, Scale::new(MARKET_TICK).unwrap());
    let b: Vec<f64> = series.iter().rev().copied().collect();
    let join = || {
        ab_join_report_tiled(
            &series[..],
            &b[..],
            nz(m),
            scale,
            scale,
            nz(2),
            nz(8),
            nz(64),
            nz(64),
        )
        .0
    };
    let plain = {
        let _g = forced_route(RouteForce {
            survivor: SurvivorForce::Off,
            ..routed()
        });
        profile_bits(&join())
    };
    for floors in FLOORS {
        let g = forced_route(with(routed(), SurvivorForce::Paa, floors));
        assert_eq!(profile_bits(&join()), plain, "{floors:?}");
        assert_eq!(g.fired().floor_raises, 0, "{floors:?}");
    }
}
