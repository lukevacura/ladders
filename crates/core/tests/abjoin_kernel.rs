#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! The AB-join row pass equals the direct exact AB-join reference on bits: random A and B of
//! different lengths, price levels and scales, gaps, flat runs, repeats across the two, B equal
//! to A, a window longer than either series, and a span too wide for an anchor on one side only.

use core::num::NonZeroUsize;

#[path = "common/spill_ref.rs"]
mod spill_ref;
#[path = "common/spill_run.rs"]
mod spill_run;

use ladde::batch::ab_join_single_threaded;
use ladde::entry::Entry;
use ladde::naive::naive_ab_join;
use ladde::quantize::{Scale, to_ticks};
use proptest::prelude::*;
use testkit::generators::{MARKET_TICK, market_series};

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

type Bits = (u64, u64, u64, u64);

fn bits(profile: &[(u64, Entry)]) -> Vec<Bits> {
    profile
        .iter()
        .map(|&(row, e)| {
            (
                row,
                e.rho().to_bits(),
                e.squared_distance().to_bits(),
                e.index().get().unwrap_or(u64::MAX),
            )
        })
        .collect()
}

fn samples(series: &[f64], scale: Scale) -> Vec<Option<i64>> {
    series.iter().map(|&x| to_ticks(x, scale)).collect()
}

fn scale(x: f64) -> Scale {
    Scale::new(x).unwrap()
}

fn assert_matches_naive(a: &[f64], b: &[f64], m: usize, scale_a: Scale, scale_b: Scale) {
    // The exact reference: a window with no distance in the hot passes' domain has one too.
    let expected = spill_run::exact_join(a, scale_a, b, scale_b, m);
    let actual = ab_join_single_threaded(a, b, nz(m), scale_a, scale_b);
    assert_eq!(bits(&actual), expected);
}

fn scales() -> impl Strategy<Value = Scale> {
    prop_oneof![Just(MARKET_TICK), Just(0.5), Just(1.0)].prop_map(scale)
}

/// A series at scale 1 jumping between levels around the one-anchor bound for `m`.
fn jumpy(m: usize) -> impl Strategy<Value = Vec<f64>> {
    let bound = (i32::MAX as usize / m) as f64;
    let levels = [0.0, bound, 3.0 * bound, -2.0 * bound];
    prop::collection::vec((0usize..8, -4i32..=4), 0..40).prop_map(move |steps| {
        let mut level = 0.0;
        steps
            .into_iter()
            .map(|(jump, noise)| {
                if let Some(&next) = levels.get(jump) {
                    level = next;
                }
                level + f64::from(noise)
            })
            .collect()
    })
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    #[test]
    fn random_pairs_equal_naive(
        m in 1usize..=6,
        a in (1usize..=6).prop_flat_map(|m| market_series(0..40, m)),
        b in (1usize..=6).prop_flat_map(|m| market_series(0..40, m)),
        scale_a in scales(),
        scale_b in scales(),
    ) {
        let expected = spill_run::exact_join(&a, scale_a, &b, scale_b, m);
        let actual = ab_join_single_threaded(&a, &b, nz(m), scale_a, scale_b);
        prop_assert_eq!(bits(&actual), expected);
    }

    #[test]
    fn a_series_joined_with_itself_equals_naive(
        m in 1usize..=6,
        a in (1usize..=6).prop_flat_map(|m| market_series(0..40, m)),
        scale_a in scales(),
    ) {
        assert_matches_naive(&a, &a, m, scale_a, scale_a);
    }

    #[test]
    fn a_series_joined_with_a_shifted_copy_equals_naive(
        m in 1usize..=6,
        a in (1usize..=6).prop_flat_map(|m| market_series(8..40, m)),
        cut in 0usize..8,
    ) {
        let b = a[cut..].to_vec();
        assert_matches_naive(&a, &b, m, scale(MARKET_TICK), scale(MARKET_TICK));
    }

    #[test]
    fn a_span_too_wide_for_one_side_equals_naive(
        m in 1usize..=6,
        wide in (1usize..=6).prop_flat_map(jumpy),
        flat in (1usize..=6).prop_flat_map(|m| market_series(0..40, m)),
        wide_first in any::<bool>(),
    ) {
        let one = scale(1.0);
        let flat_scale = scale(MARKET_TICK);
        if wide_first {
            assert_matches_naive(&wide, &flat, m, one, flat_scale);
        } else {
            assert_matches_naive(&flat, &wide, m, flat_scale, one);
        }
    }
}

#[test]
fn a_window_longer_than_either_series_gives_the_reference_shape() {
    let short: Vec<f64> = vec![1.0, 2.0];
    let long: Vec<f64> = vec![3.0, 1.0, 4.0, 1.0, 5.0];
    let one = scale(1.0);
    for (a, b) in [(&short, &long), (&long, &short), (&short, &short)] {
        assert_matches_naive(a, b, 3, one, one);
    }
}

#[test]
fn a_fixed_pair_has_pinned_entries() {
    let one = scale(1.0);
    let a = [1.0, 2.0, 4.0, 3.0, 5.0, 5.0, 5.0, f64::NAN];
    let b = [9.0, 1.0, 2.0, 4.0, 3.0, 8.0, 1.0, 2.0, 4.0, 3.0];
    let actual = ab_join_single_threaded(&a, &b, nz(4), one, one);
    let expected = naive_ab_join(&samples(&a, one), &samples(&b, one), 4);
    assert_eq!(bits(&actual), bits(&expected));
    let (_, first) = actual[0];
    assert_eq!(first.index().get(), Some(1));
    assert_eq!(first.rho().to_bits(), 1.0f64.to_bits());
    assert_eq!(first.squared_distance().to_bits(), 0.0f64.to_bits());
    let (_, flat) = actual[4];
    assert_eq!(flat.index().get(), None);
}
