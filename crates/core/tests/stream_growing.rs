#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]

//! `GrowingStream`'s collected output always satisfies `testkit::validate::left_profile` and
//! `complete_rows`, checked on market-shaped series quantized to ticks. `testkit` is a
//! dev-dependency of `ladde`, and `ladde` is a dependency of `testkit`; that
//! cycle is fine here because this integration test links against the compiled library rather
//! than its source, but it is why this check can't live in `stream.rs`'s own unit tests (see
//! `naive_left_profile.rs`).

use core::num::NonZeroUsize;
use ladde::quantize::Scale;
use ladde::stream::GrowingStream;
use proptest::prelude::*;
use testkit::generators::{MARKET_TICK, market_series_and_m};
use testkit::validate::{complete_rows, left_profile};

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    #[test]
    fn stream_output_satisfies_left_profile_invariants((series, m) in market_series_and_m(0..48)) {
        let scale = Scale::new(MARKET_TICK).unwrap();
        let mut stream = GrowingStream::new(NonZeroUsize::new(m).unwrap(), scale);
        let profile: Vec<_> = series
            .iter()
            .filter_map(|&x| stream.update(x))
            .map(|(row, entry)| (row.get().expect("row is never NONE"), entry))
            .collect();

        prop_assert_eq!(left_profile(&profile, m), Ok(()));
        prop_assert_eq!(complete_rows(&profile, series.len(), m), Ok(()));
    }
}
