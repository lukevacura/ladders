#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! The streams' cold route under every forced route: shocked series equal the exact reference
//! and the batch profile with its cold route over the same samples, whatever is forced, and a
//! forced spill route moves its counters.

#[path = "common/route.rs"]
mod route;
use route::spill_ref;
#[path = "common/spill_run.rs"]
mod spill_run;

use ladde::bounded::BoundedStream;
use ladde::quantize::Scale;
use ladde::route::{RouteForce, SpillForce, forced_route};
use ladde::stream::GrowingStream;
use route::{Bits, bits, forces, nz, samples, stream_profile_bits, stream_window_bits};

fn unit() -> Scale {
    Scale::new(1.0).unwrap()
}

fn walk(len: usize, seed: u64, level: f64) -> Vec<f64> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut value = level;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            value += (state % 9) as f64 - 4.0;
            value
        })
        .collect()
}

/// Walks at several levels with prints of ×1.5 to ×1000, adjacent, of both signs, beside gaps.
fn shocked(len: usize) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    for (seed, &level) in [1e6, 1e9, 1e12].iter().enumerate() {
        for &times in &[1.5, 1000.0] {
            let mut series = walk(len, seed as u64 + 1, level);
            for at in [3, len / 3, len / 3 + 1, len - 2] {
                series[at] *= times;
            }
            series[len / 2] *= -times;
            series[len / 3 + 2] = f64::NAN;
            out.push(series);
        }
    }
    out
}

fn growing_rows(series: &[f64], m: usize) -> Vec<Bits> {
    let mut stream = GrowingStream::new(nz(m), unit());
    series
        .iter()
        .filter_map(|&x| stream.update(x))
        .map(|(row, e)| bits(row.get().unwrap(), e))
        .collect()
}

fn bounded_rows(series: &[f64], m: usize, w: usize) -> Vec<Bits> {
    let mut stream = BoundedStream::new(nz(m), nz(w), unit()).unwrap();
    series
        .iter()
        .filter_map(|&x| stream.update(x))
        .map(|(row, e)| bits(row.get().unwrap(), e))
        .collect()
}

#[test]
fn shocked_streams_equal_the_exact_reference_on_every_route() {
    let (m, w) = (6, 30);
    for series in shocked(90) {
        let samples = samples(&series, unit());
        let expected = stream_profile_bits(&samples, m);
        let windowed: Vec<Bits> = (0..=series.len() - m)
            .map(|row| stream_window_bits(&samples, m, w, row))
            .collect();
        for (at, force) in forces(7).into_iter().enumerate() {
            for spill in [SpillForce::Auto, SpillForce::On, SpillForce::Off] {
                let _forced = forced_route(RouteForce { spill, ..force });
                assert_eq!(
                    growing_rows(&series, m),
                    expected,
                    "growing force {at} {spill:?}"
                );
                assert_eq!(
                    bounded_rows(&series, m, w),
                    windowed,
                    "bounded force {at} {spill:?}"
                );
            }
        }
    }
}

#[test]
fn shocked_streams_equal_the_batch_profile_with_its_cold_route() {
    let _forced = forced_route(RouteForce {
        spill: SpillForce::On,
        ..RouteForce::default()
    });
    for m in [4, 9] {
        for series in shocked(120) {
            let batch = spill_run::left_spill(&series, m, unit());
            assert_eq!(
                growing_rows(&series, m),
                spill_run::bits(&batch.entries),
                "m {m}"
            );
        }
    }
}

#[test]
fn a_forced_spill_route_moves_its_counters() {
    let series = &shocked(90)[3];
    let m = 6;
    let natural = {
        let _forced = forced_route(RouteForce::default());
        let mut stream = GrowingStream::new(nz(m), unit());
        for &x in series {
            let _ = stream.update(x);
        }
        stream.route_counts()
    };
    assert!(natural.withheld_ticks > 0);
    for spill in [SpillForce::On, SpillForce::Off] {
        let guard = forced_route(RouteForce {
            spill,
            ..RouteForce::default()
        });
        let mut stream = BoundedStream::new(nz(m), nz(30), unit()).unwrap();
        for &x in series {
            let _ = stream.update(x);
        }
        let (counts, fired) = (stream.route_counts(), guard.fired());
        match spill {
            SpillForce::On => {
                assert!(fired.spill_forced_on > 0);
                // Every present sample after the first is withheld.
                assert!(counts.withheld_ticks as usize >= series.len() - 4);
            }
            _ => {
                assert!(fired.spill_forced_off > 0);
                assert_eq!(counts.withheld_ticks, 0);
            }
        }
    }
}
