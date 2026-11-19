//! Shared pieces of the route tests: the forced-route list, series strategies, bit helpers
//! and the fixed walk the counter tests run.
#![allow(dead_code)]

use core::num::NonZeroUsize;

use ladde::batch::left_profile_report_tiled;
use ladde::bounded::BoundedStream;
use ladde::entry::Entry;
use ladde::index::SampleIndex;
use ladde::quantize::{Scale, to_ticks};
use ladde::route::{
    ConversionForce, Device, Fired, RouteForce, SparseForce, SpillForce, forced_route,
};
use ladde::stream::GrowingStream;
use ladde::stream_api::Stream;
use proptest::prelude::*;
use testkit::generators::{MARKET_TICK, market_series_and_m};

/// The default force: the route production takes (the one-loop route where the PAA filter is on,
/// plain elsewhere).
pub fn routed() -> RouteForce {
    RouteForce::default()
}

#[path = "spill_ref.rs"]
pub mod spill_ref;

pub type Bits = (u64, u64, u64, u64);

/// Ticks with each gap holding the tick before it (`0` before any), and which were present.
fn ticks_and_presence(samples: &[Option<i64>]) -> (Vec<i64>, Vec<bool>) {
    let mut last = 0;
    let ticks = samples
        .iter()
        .map(|tick| {
            last = tick.unwrap_or(last);
            last
        })
        .collect();
    (ticks, samples.iter().map(Option::is_some).collect())
}

/// The exact left profile a stream reports over `samples`, as bits: every window with its
/// samples gets its exact nearest, whatever its variance numerator.
pub fn stream_profile_bits(samples: &[Option<i64>], m: usize) -> Vec<Bits> {
    let (ticks, had) = ticks_and_presence(samples);
    spill_ref::left_profile(&ticks, &had, m)
}

pub fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

pub fn bits(row: u64, e: Entry) -> Bits {
    (
        row,
        e.rho().to_bits(),
        e.squared_distance().to_bits(),
        e.index().get().unwrap_or(u64::MAX),
    )
}

pub fn profile_bits(profile: &[(u64, Entry)]) -> Vec<Bits> {
    profile.iter().map(|&(row, e)| bits(row, e)).collect()
}

pub fn stream_bits(rows: &[(SampleIndex, Entry)]) -> Vec<Bits> {
    rows.iter()
        .map(|&(row, e)| bits(row.get().expect("row is never NONE"), e))
        .collect()
}

/// The forces the route tests run, each on every device.
pub fn forces(seed: u64) -> Vec<RouteForce> {
    Device::ALL
        .into_iter()
        .flat_map(|device| {
            forces_on_cpu(seed)
                .into_iter()
                .map(move |force| RouteForce { device, ..force })
        })
        .collect()
}

fn forces_on_cpu(seed: u64) -> Vec<RouteForce> {
    let seed = seed | 1;
    let none = RouteForce::default();
    vec![
        none,
        RouteForce { wide: true, ..none },
        RouteForce {
            sparse: SparseForce::On,
            ..none
        },
        RouteForce {
            sparse: SparseForce::Off,
            wide: true,
            ..none
        },
        RouteForce {
            anchor_seed: seed,
            ..none
        },
        RouteForce {
            anchor_seed: seed,
            reanchor_every: 1,
            ..none
        },
        RouteForce {
            anchor_seed: seed ^ 0x55,
            reanchor_every: 3,
            ..none
        },
        RouteForce {
            per_tile: true,
            ..none
        },
        RouteForce {
            per_tile: true,
            anchor_seed: seed,
            ..none
        },
        RouteForce {
            wide: true,
            per_tile: true,
            ..none
        },
        RouteForce {
            reanchor_cell: 3,
            ..none
        },
        RouteForce {
            reanchor_cell: 1,
            anchor_seed: seed ^ 0xA5,
            ..none
        },
        RouteForce {
            reanchor_cell: 8,
            per_tile: true,
            ..none
        },
        RouteForce {
            reanchor_cell: 2,
            wide: true,
            ..none
        },
        RouteForce {
            spill: SpillForce::On,
            ..none
        },
        RouteForce {
            spill: SpillForce::Off,
            wide: true,
            ..none
        },
        RouteForce {
            constant_rows_by_lane: true,
            ..none
        },
        RouteForce {
            constant_rows_by_lane: true,
            wide: true,
            ..none
        },
        RouteForce {
            no_constant_class: true,
            ..none
        },
        RouteForce {
            no_constant_class: true,
            wide: true,
            per_tile: true,
            ..none
        },
        RouteForce {
            conversion: ConversionForce::Fast,
            per_tile: true,
            ..none
        },
        RouteForce {
            conversion: ConversionForce::FullRange,
            ..none
        },
        RouteForce {
            conversion: ConversionForce::FullRange,
            reanchor_cell: 3,
            ..none
        },
        RouteForce {
            resume_every: 1,
            ..none
        },
        RouteForce {
            resume_every: 3,
            per_tile: true,
            ..none
        },
        RouteForce {
            resume_every: 2,
            wide: true,
            ..none
        },
        RouteForce {
            resume_every: 4,
            reanchor_cell: 3,
            anchor_seed: seed,
            reanchor_every: 2,
            ..none
        },
    ]
}

/// A quantized series at scale 1 that jumps between levels around the one-anchor bound, with gaps
/// and flat runs: natural re-anchors and switches to i64 happen inside it.
pub fn jumpy(m: usize) -> impl Strategy<Value = (Vec<f64>, usize, Scale)> {
    let bound = (i32::MAX as usize / m) as f64;
    let levels = [0.0, bound / 2.0, bound, 2.0 * bound, 3.0 * bound, -bound];
    prop::collection::vec((0u8..14, -5i32..=5, 0u8..10), 8..60).prop_map(move |steps| {
        let mut level = 0.0;
        let mut last = 0.0;
        let series = steps
            .into_iter()
            .map(|(jump, noise, kind)| {
                if let Some(&next) = levels.get(jump as usize) {
                    level = next;
                }
                match kind {
                    0 => f64::NAN,
                    1 => last,
                    _ => {
                        last = level + f64::from(noise);
                        last
                    }
                }
            })
            .collect();
        (series, m, Scale::new(1.0).unwrap())
    })
}

pub fn cases() -> impl Strategy<Value = (Vec<f64>, usize, Scale)> {
    prop_oneof![
        market_series_and_m(0..48).prop_map(|(series, m)| (
            series,
            m,
            Scale::new(MARKET_TICK).unwrap()
        )),
        (2usize..=6).prop_flat_map(jumpy),
    ]
}

pub fn samples(series: &[f64], scale: Scale) -> Vec<Option<i64>> {
    series.iter().map(|&x| to_ticks(x, scale)).collect()
}

pub fn chunked(
    stream: &mut impl Stream,
    series: &[f64],
    chunk: usize,
) -> Vec<(SampleIndex, Entry)> {
    let mut out = Vec::new();
    for piece in series.chunks(chunk) {
        stream.update_many(piece, &mut out);
    }
    out
}

/// [`stream_profile_bits`]'s row `i` on only the `w` samples ending at that row's window, with
/// the neighbor index mapped back to the whole series.
pub fn stream_window_bits(samples: &[Option<i64>], m: usize, w: usize, i: usize) -> Bits {
    let end = i + m;
    let start = end.saturating_sub(w);
    let (_, rho, distance, index) = *stream_profile_bits(&samples[start..end], m)
        .last()
        .expect("the slice holds row i's own window");
    let index = if index == u64::MAX {
        index
    } else {
        index + start as u64
    };
    (i as u64, rho, distance, index)
}

pub fn thread_counts() -> Vec<usize> {
    vec![1, 2, 7, 16]
}

/// A long random walk that stays within one narrow anchor, so no route fires on its own.
pub fn long_walk() -> (Vec<f64>, usize, Scale) {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut price = 100.0;
    let series = (0..241)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            price += ((state >> 33) % 7) as f64 * MARKET_TICK - 3.0 * MARKET_TICK;
            price
        })
        .collect();
    (series, 8, Scale::new(MARKET_TICK).unwrap())
}

pub fn batch_run(force: RouteForce) -> Fired {
    let (series, m, scale) = long_walk();
    let guard = forced_route(force);
    for (band, chunk, tile) in [(1, 1, 7), (4, 2, 16), (64, 8, 64)] {
        let _ =
            left_profile_report_tiled(&series, nz(m), scale, nz(2), nz(band), nz(chunk), nz(tile));
    }
    guard.fired()
}

pub fn growing_run(force: RouteForce) -> Fired {
    let (series, m, scale) = long_walk();
    let guard = forced_route(force);
    let mut stream = GrowingStream::new(nz(m), scale);
    for &x in &series {
        let _ = stream.update(x);
    }
    guard.fired()
}

pub fn bounded_run(force: RouteForce) -> Fired {
    let (series, m, scale) = long_walk();
    let guard = forced_route(force);
    let mut stream = BoundedStream::new(nz(m), nz(40), scale).expect("w is at least m");
    for &x in &series {
        let _ = stream.update(x);
    }
    guard.fired()
}

/// A row's kept neighbors as bits, nearest first.
pub fn topk_bits(row: u64, entries: &[Entry]) -> Vec<Bits> {
    entries.iter().map(|&e| bits(row, e)).collect()
}

/// The exact `k` nearest left neighbors a stream reports per row over `samples`, as bits.
pub fn stream_topk_rows(samples: &[Option<i64>], m: usize, k: usize) -> Vec<(u64, Vec<Bits>)> {
    let (ticks, had) = ticks_and_presence(samples);
    spill_ref::left_topk(&ticks, &had, m, k)
        .into_iter()
        .enumerate()
        .map(|(row, entries)| (row as u64, entries))
        .collect()
}

/// [`stream_topk_rows`]'s row `i` on only the `w` samples ending at that row's window, with the
/// neighbor indices mapped back to the whole series.
pub fn stream_window_topk_bits(
    samples: &[Option<i64>],
    m: usize,
    w: usize,
    i: usize,
    k: usize,
) -> Vec<Bits> {
    let end = i + m;
    let start = end.saturating_sub(w);
    let (_, local) = stream_topk_rows(&samples[start..end], m, k)
        .pop()
        .expect("the slice holds row i's own window");
    local
        .into_iter()
        .map(|(_, rho, distance, index)| (i as u64, rho, distance, index + start as u64))
        .collect()
}

/// The growing stream's `k` neighbors per row over `series`, as bits.
pub fn growing_topk_rows(
    series: &[f64],
    m: usize,
    scale: Scale,
    k: usize,
) -> Vec<(u64, Vec<Bits>)> {
    let mut stream = GrowingStream::with_k(nz(m), scale, nz(k));
    series
        .iter()
        .filter_map(|&x| stream.update_topk(x).map(|(row, e)| (row, e.to_vec())))
        .map(|(row, e)| {
            let row = row.get().expect("row is never NONE");
            (row, topk_bits(row, &e))
        })
        .collect()
}

/// The bounded stream's `k` neighbors per row over `series`, as bits.
pub fn bounded_topk_rows(
    series: &[f64],
    m: usize,
    w: usize,
    scale: Scale,
    k: usize,
) -> Vec<(u64, Vec<Bits>)> {
    let mut stream = BoundedStream::with_k(nz(m), nz(w), scale, nz(k)).expect("w is at least m");
    series
        .iter()
        .filter_map(|&x| stream.update_topk(x).map(|(row, e)| (row, e.to_vec())))
        .map(|(row, e)| {
            let row = row.get().expect("row is never NONE");
            (row, topk_bits(row, &e))
        })
        .collect()
}
