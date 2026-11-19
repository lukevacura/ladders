extern crate std;

use super::{ForwardFill, Sanitized};
use proptest::prelude::*;
use proptest::test_runner::TestCaseResult;
use std::vec::Vec;

fn ticks() -> impl Strategy<Value = Vec<Option<i64>>> {
    prop::collection::vec(prop::option::of(any::<i64>()), 0..64)
}

fn prices() -> impl Strategy<Value = Vec<Option<f64>>> {
    prop::collection::vec(
        prop::option::of(any::<u64>().prop_map(f64::from_bits)),
        0..64,
    )
}

fn same_bits(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}

fn push_all<T: Copy>(samples: &[Option<T>]) -> Vec<Sanitized<T>> {
    let mut fill = ForwardFill::default();
    samples.iter().map(|&sample| fill.push(sample)).collect()
}

fn last_present<T: Copy>(samples: &[Option<T>]) -> Option<T> {
    samples.iter().rev().find_map(|&sample| sample)
}

fn check_filled_value_is_last_present<T: Copy>(
    samples: &[Option<T>],
    same: fn(T, T) -> bool,
) -> TestCaseResult {
    for (k, out) in push_all(samples).into_iter().enumerate() {
        if samples[k].is_some() {
            continue;
        }
        if let Some(last) = last_present(&samples[..k]) {
            prop_assert!(matches!(out, Sanitized::Filled(v) if same(v, last)));
        }
    }
    Ok(())
}

fn check_present_passes_through<T: Copy>(
    samples: &[Option<T>],
    same: fn(T, T) -> bool,
) -> TestCaseResult {
    for (&sample, out) in samples.iter().zip(push_all(samples)) {
        if let Some(value) = sample {
            prop_assert!(matches!(out, Sanitized::Present(v) if same(v, value)));
        }
    }
    Ok(())
}

fn check_leading_missing_yields_nothing<T: Copy>(samples: &[Option<T>]) -> TestCaseResult {
    let leading = samples.iter().take_while(|sample| sample.is_none()).count();
    for (k, out) in push_all(samples).into_iter().enumerate() {
        prop_assert_eq!(matches!(out, Sanitized::Leading), k < leading);
    }
    Ok(())
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    #[test]
    fn filled_value_is_last_present_i64(samples in ticks()) {
        check_filled_value_is_last_present(&samples, |a, b| a == b)?;
    }

    #[test]
    fn filled_value_is_last_present_f64(samples in prices()) {
        check_filled_value_is_last_present(&samples, same_bits)?;
    }

    #[test]
    fn present_passes_through_i64(samples in ticks()) {
        check_present_passes_through(&samples, |a, b| a == b)?;
    }

    #[test]
    fn present_passes_through_f64(samples in prices()) {
        check_present_passes_through(&samples, same_bits)?;
    }

    #[test]
    fn leading_missing_yields_nothing_i64(samples in ticks()) {
        check_leading_missing_yields_nothing(&samples)?;
    }

    #[test]
    fn leading_missing_yields_nothing_f64(samples in prices()) {
        check_leading_missing_yields_nothing(&samples)?;
    }
}
