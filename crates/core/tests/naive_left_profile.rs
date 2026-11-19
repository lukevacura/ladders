#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]

//! `naive_left_profile`'s output always satisfies `testkit::validate::left_profile`, checked on
//! market-shaped series quantized to ticks. `testkit` is a dev-dependency of `ladde`,
//! and `ladde` is a dependency of `testkit`; that cycle is fine here because this
//! integration test links against the compiled library rather than its source, but it is why
//! this check can't live in `naive.rs`'s own unit tests.

use ladde::naive::naive_left_profile;
use ladde::quantize::{Scale, to_ticks};
use proptest::prelude::*;
use testkit::generators::market_series_and_m;
use testkit::validate::left_profile;

const TICK: f64 = 0.01;

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    #[test]
    fn naive_output_satisfies_left_profile_invariants((series, m) in market_series_and_m(0..48)) {
        let scale = Scale::new(TICK).unwrap();
        let ticks: Vec<Option<i64>> = series.iter().map(|&value| to_ticks(value, scale)).collect();
        let profile = naive_left_profile(&ticks, m);
        prop_assert_eq!(left_profile(&profile, m), Ok(()));
    }
}
