#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! A non-normalized stream that looks right checkpoints its right side: the restored stream goes
//! on to the same bytes as one that never stopped.

#[path = "common/nonnorm.rs"]
mod nonnorm;

use ladde::nonnorm_bounded::BoundedNonNormStream;
use ladde::nonnorm_stream::GrowingNonNormStream;
use nonnorm::*;
use proptest::prelude::*;

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]

    #[test]
    fn a_restored_stream_looking_right_goes_on_as_one_that_never_stopped(
        series in prop_oneof![series(), tie_series(), shocked()],
        m in 1usize..=6,
        k in prop_oneof![Just(1usize), 2usize..6],
        extra in 0usize..30,
        cut in 0usize..60,
    ) {
        let cut = cut.min(series.len());
        let (head, tail) = series.split_at(cut);
        let mut growing = GrowingNonNormStream::with_right_side(nz(m), unit(), nz(k));
        let mut whole = GrowingNonNormStream::with_right_side(nz(m), unit(), nz(k));
        let mut bounded =
            BoundedNonNormStream::with_right_side(nz(m), nz(m + extra), unit(), nz(k)).unwrap();
        let mut bounded_whole =
            BoundedNonNormStream::with_right_side(nz(m), nz(m + extra), unit(), nz(k)).unwrap();
        for &x in head {
            let _ = (growing.update(x), whole.update(x));
            let _ = (bounded.update(x), bounded_whole.update(x));
        }
        let mut growing = GrowingNonNormStream::restore(&growing.checkpoint()).unwrap();
        let mut bounded = BoundedNonNormStream::restore(&bounded.checkpoint()).unwrap();
        prop_assert_eq!(growing.checkpoint(), whole.checkpoint());
        prop_assert_eq!(bounded.checkpoint(), bounded_whole.checkpoint());
        for &x in tail {
            let _ = (growing.update(x), whole.update(x));
            let _ = (bounded.update(x), bounded_whole.update(x));
        }
        prop_assert_eq!(growing.checkpoint(), whole.checkpoint());
        prop_assert_eq!(bounded.checkpoint(), bounded_whole.checkpoint());
    }
}
