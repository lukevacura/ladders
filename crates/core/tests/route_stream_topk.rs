#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! Streams that keep `k` neighbors per row equal the exact reference (the bounded one on only its
//! last `w` samples) on bits under every forced route, and keep their nearest neighbor equal to
//! the one-neighbor stream's.

#[path = "common/route.rs"]
mod route;

use ladde::bounded::BoundedStream;
use ladde::narrow::Path;
use ladde::route::{RouteForce, forced_route};
use ladde::stream::GrowingStream;
use proptest::prelude::*;
use route::*;

const KS: [usize; 4] = [2, 3, 8, 200];

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    #[test]
    fn growing_topk_equals_the_exact_reference_on_every_route(
        (series, m, scale) in cases(),
        seed in any::<u64>(),
    ) {
        let samples = samples(&series, scale);
        for k in KS {
            let expected = stream_topk_rows(&samples, m, k);
            for force in forces(seed) {
                let _forced = forced_route(force);
                prop_assert_eq!(
                    &growing_topk_rows(&series, m, scale, k),
                    &expected,
                    "{:?} k {}", force, k
                );
                let mut stream = GrowingStream::with_k(nz(m), scale, nz(k));
                for &x in &series {
                    let _ = stream.update(x);
                }
                if force.wide {
                    prop_assert_eq!(stream.path(), Path::Wide);
                }
            }
        }
    }

    #[test]
    fn bounded_topk_equals_the_windowed_exact_reference_on_every_route(
        (series, m, scale) in cases(),
        seed in any::<u64>(),
        extra in 0usize..40,
    ) {
        let samples = samples(&series, scale);
        let w = m + extra;
        for k in KS {
            for force in forces(seed) {
                let _forced = forced_route(force);
                for (row, got) in bounded_topk_rows(&series, m, w, scale, k) {
                    prop_assert_eq!(
                        got,
                        stream_window_topk_bits(&samples, m, w, row as usize, k),
                        "{:?} w {} k {}", force, w, k
                    );
                }
                let mut stream = BoundedStream::with_k(nz(m), nz(w), scale, nz(k))
                    .expect("w is at least m");
                for &x in &series {
                    let _ = stream.update(x);
                }
                if force.wide {
                    prop_assert_eq!(stream.path(), Path::Wide);
                }
            }
        }
    }

    #[test]
    fn the_nearest_of_k_is_the_one_neighbor_stream(
        (series, m, scale) in cases(),
        extra in 0usize..40,
        k in 2usize..9,
    ) {
        let _serial = forced_route(RouteForce::default());
        let mut one = GrowingStream::new(nz(m), scale);
        let mut many = GrowingStream::with_k(nz(m), scale, nz(k));
        let mut bounded_one = BoundedStream::new(nz(m), nz(m + extra), scale).unwrap();
        let mut bounded_many = BoundedStream::with_k(nz(m), nz(m + extra), scale, nz(k)).unwrap();
        for &x in &series {
            prop_assert_eq!(
                one.update(x).map(|(r, e)| bits(r.get().unwrap(), e)),
                many.update(x).map(|(r, e)| bits(r.get().unwrap(), e))
            );
            prop_assert_eq!(
                bounded_one.update(x).map(|(r, e)| bits(r.get().unwrap(), e)),
                bounded_many.update(x).map(|(r, e)| bits(r.get().unwrap(), e))
            );
        }
    }
}
