#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! A stream that looks right keeps, for each window it holds, its `k` nearest later neighbors and
//! its `k` nearest over both sides, equal on bits to the exact reference over the samples the
//! stream holds, on every route; the stream's own entries do not change.

#[path = "common/route.rs"]
mod route;

use ladde::bounded::BoundedStream;
use ladde::entry::Entry;
use ladde::route::forced_route;
use ladde::stream::GrowingStream;
use proptest::prelude::*;
use route::spill_ref::{self, Bits};
use route::{bits, cases, forces, nz, samples};

const KS: [usize; 4] = [1, 2, 3, 8];

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

fn row_bits(row: u64, entries: &[Entry]) -> Vec<Bits> {
    entries.iter().map(|&entry| bits(row, entry)).collect()
}

/// The reference's lists for the windows of `held` (a run of samples), indices mapped back by
/// `offset` windows.
fn reference(held: &[Option<i64>], m: usize, k: usize, offset: u64) -> Vec<(Vec<Bits>, Vec<Bits>)> {
    let (ticks, had) = ticks_and_presence(held);
    let right = spill_ref::right_topk(&ticks, &had, m, k);
    let both = spill_ref::two_sided_topk(&ticks, &had, m, k);
    let shift = |lists: &[Bits]| -> Vec<Bits> {
        lists
            .iter()
            .map(|&(row, rho, distance, index)| {
                let index = if index == u64::MAX {
                    index
                } else {
                    index + offset
                };
                (row + offset, rho, distance, index)
            })
            .collect()
    };
    right
        .iter()
        .zip(&both)
        .map(|(right, (_, _, both))| (shift(right), shift(both)))
        .collect()
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]

    #[test]
    fn the_growing_stream_looking_right_equals_the_reference_on_every_route(
        (series, m, scale) in cases(),
        seed in any::<u64>(),
    ) {
        let held = samples(&series, scale);
        for k in KS {
            let expected = reference(&held, m, k, 0);
            for force in forces(seed) {
                let _forced = forced_route(force);
                let mut stream = GrowingStream::with_right_side(nz(m), scale, nz(k));
                let mut plain = GrowingStream::with_k(nz(m), scale, nz(k));
                let mut out = Vec::new();
                for &x in &series {
                    let a = stream.update_topk(x).map(|(row, e)| (row, e.to_vec()));
                    let b = plain.update_topk(x).map(|(row, e)| (row, e.to_vec()));
                    prop_assert_eq!(
                        a.map(|(r, e)| row_bits(r.get().unwrap(), &e)),
                        b.map(|(r, e)| row_bits(r.get().unwrap(), &e)),
                        "{:?} k {}", force, k
                    );
                }
                let windows = stream.right_windows();
                prop_assert_eq!(windows.end as usize, expected.len());
                for window in windows {
                    stream.right_neighbors_into(window, &mut out);
                    prop_assert_eq!(
                        row_bits(window, &out),
                        expected[window as usize].0.clone(),
                        "right {:?} k {} window {}", force, k, window
                    );
                    stream.two_sided_neighbors_into(window, &mut out);
                    prop_assert_eq!(
                        row_bits(window, &out),
                        expected[window as usize].1.clone(),
                        "both {:?} k {} window {}", force, k, window
                    );
                }
            }
        }
    }
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]

    #[test]
    fn the_bounded_stream_looking_right_equals_the_reference_on_every_route(
        (series, m, scale) in cases(),
        seed in any::<u64>(),
        extra in 0usize..40,
    ) {
        let held = samples(&series, scale);
        let w = m + extra;
        let (ticks, had) = ticks_and_presence(&held);
        for k in KS {
            let right = spill_ref::right_topk(&ticks, &had, m, k);
            let both = spill_ref::two_sided_topk_where(&ticks, &had, m, k, &|i, j| i.abs_diff(j) <= extra);
            for force in forces(seed) {
                let _forced = forced_route(force);
                let mut stream = BoundedStream::with_right_side(nz(m), nz(w), scale, nz(k)).unwrap();
                let mut out = Vec::new();
                for &x in &series {
                    let _ = stream.update(x);
                }
                for window in stream.right_windows() {
                    stream.right_neighbors_into(window, &mut out);
                    prop_assert_eq!(
                        row_bits(window, &out),
                        right[window as usize].clone(),
                        "right {:?} k {} w {} window {}", force, k, w, window
                    );
                    stream.two_sided_neighbors_into(window, &mut out);
                    prop_assert_eq!(
                        row_bits(window, &out),
                        both[window as usize].2.clone(),
                        "both {:?} k {} w {} window {}", force, k, w, window
                    );
                }
            }
        }
    }
}
