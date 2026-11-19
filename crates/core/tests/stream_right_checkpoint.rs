#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! A stream that looks right checkpoints its right side: the restored stream goes on to the same
//! lists, and the same bytes, as one that never stopped, on every route.

#[path = "common/route.rs"]
mod route;

use ladde::bounded::BoundedStream;
use ladde::route::forced_route;
use ladde::stream::GrowingStream;
use proptest::prelude::*;
use route::spill_ref::Bits;
use route::{bits, cases, forces, nz};

fn lists_growing(stream: &GrowingStream) -> Vec<(Vec<Bits>, Vec<Bits>)> {
    let mut out = Vec::new();
    stream
        .right_windows()
        .map(|window| {
            stream.right_neighbors_into(window, &mut out);
            let right = out.iter().map(|&e| bits(window, e)).collect();
            stream.two_sided_neighbors_into(window, &mut out);
            (right, out.iter().map(|&e| bits(window, e)).collect())
        })
        .collect()
}

fn lists_bounded(stream: &BoundedStream) -> Vec<(Vec<Bits>, Vec<Bits>)> {
    let mut out = Vec::new();
    stream
        .right_windows()
        .map(|window| {
            stream.right_neighbors_into(window, &mut out);
            let right = out.iter().map(|&e| bits(window, e)).collect();
            stream.two_sided_neighbors_into(window, &mut out);
            (right, out.iter().map(|&e| bits(window, e)).collect())
        })
        .collect()
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]

    #[test]
    fn a_restored_stream_looking_right_goes_on_as_one_that_never_stopped(
        (series, m, scale) in cases(),
        seed in any::<u64>(),
        extra in 0usize..40,
        cut in 0usize..60,
        k in prop_oneof![Just(1usize), 2usize..6],
    ) {
        let cut = cut.min(series.len());
        let (head, tail) = series.split_at(cut);
        for force in forces(seed) {
            let _forced = forced_route(force);
            let mut growing = GrowingStream::with_right_side(nz(m), scale, nz(k));
            let mut whole = GrowingStream::with_right_side(nz(m), scale, nz(k));
            let mut bounded =
                BoundedStream::with_right_side(nz(m), nz(m + extra), scale, nz(k)).unwrap();
            let mut bounded_whole =
                BoundedStream::with_right_side(nz(m), nz(m + extra), scale, nz(k)).unwrap();
            for &x in head {
                let _ = (growing.update(x), whole.update(x));
                let _ = (bounded.update(x), bounded_whole.update(x));
            }
            let mut growing = GrowingStream::restore(&growing.checkpoint()).unwrap();
            let mut bounded = BoundedStream::restore(&bounded.checkpoint()).unwrap();
            prop_assert_eq!(lists_growing(&growing), lists_growing(&whole));
            prop_assert_eq!(lists_bounded(&bounded), lists_bounded(&bounded_whole));
            for &x in tail {
                let _ = (growing.update(x), whole.update(x));
                let _ = (bounded.update(x), bounded_whole.update(x));
            }
            prop_assert_eq!(lists_growing(&growing), lists_growing(&whole), "{:?}", force);
            prop_assert_eq!(lists_bounded(&bounded), lists_bounded(&bounded_whole), "{:?}", force);
            prop_assert_eq!(growing.checkpoint(), whole.checkpoint());
            prop_assert_eq!(bounded.checkpoint(), bounded_whole.checkpoint());
        }
    }
}
