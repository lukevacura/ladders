extern crate std;

use super::Stream;
use crate::bounded::BoundedStream;
use crate::entry::Entry;
use crate::index::SampleIndex;
use crate::quantize::Scale;
use crate::stream::GrowingStream;
use core::num::NonZeroUsize;
use proptest::prelude::*;
use std::vec::Vec;

const TICK: f64 = 0.01;

fn bits((row, entry): (SampleIndex, Entry)) -> (u64, u64, u64, u64) {
    (
        row.get().expect("row is never NONE"),
        entry.rho().to_bits(),
        entry.squared_distance().to_bits(),
        entry.index().get().unwrap_or(u64::MAX),
    )
}

/// A quantized walk that sometimes glitches a sample to NaN, so `update_many` sees both
/// a forward-filled gap and the plain present case.
fn series_with_gaps(
    len: impl Into<proptest::collection::SizeRange>,
) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec((-2i64..=2, any::<bool>()), len).prop_map(|steps| {
        steps
            .into_iter()
            .scan(0i64, |level, (step, is_gap)| {
                *level += step;
                Some(if is_gap {
                    f64::NAN
                } else {
                    *level as f64 * TICK
                })
            })
            .collect()
    })
}

/// `m`, a series at least `m` long, and a split point anywhere in it (including inside a
/// window, and at both ends) to feed `update_many` in two pieces.
fn series_m_and_split() -> impl Strategy<Value = (Vec<f64>, usize, usize)> {
    (2usize..=5).prop_flat_map(|m| {
        series_with_gaps(m..=40).prop_flat_map(move |series| {
            let len = series.len();
            (Just(series), Just(m), 0..=len)
        })
    })
}

/// `m`, `w`, a series, and a split point, for [`BoundedStream`].
fn series_m_w_and_split() -> impl Strategy<Value = (Vec<f64>, usize, usize, usize)> {
    (2usize..=5, 0usize..=10).prop_flat_map(|(m, extra)| {
        let w = m + extra;
        series_with_gaps(m..=40).prop_flat_map(move |series| {
            let len = series.len();
            (Just(series), Just(m), Just(w), 0..=len)
        })
    })
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    /// `GrowingStream::update_many`, split anywhere, returns exactly what the same series
    /// fed sample by sample to `update` returns, bit for bit.
    #[test]
    fn update_many_matches_update_for_growing_stream((series, m, split) in series_m_and_split()) {
        let scale = Scale::new(TICK).unwrap();

        let mut sequential = GrowingStream::new(NonZeroUsize::new(m).unwrap(), scale);
        let expected: Vec<_> = series.iter().filter_map(|&x| sequential.update(x)).map(bits).collect();

        let mut chunked = GrowingStream::new(NonZeroUsize::new(m).unwrap(), scale);
        let mut out = Vec::new();
        chunked.update_many(&series[..split], &mut out);
        chunked.update_many(&series[split..], &mut out);
        let actual: Vec<_> = out.into_iter().map(bits).collect();

        prop_assert_eq!(actual, expected);
    }
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    /// Same check for [`BoundedStream`].
    #[test]
    fn update_many_matches_update_for_bounded_stream((series, m, w, split) in series_m_w_and_split()) {
        let scale = Scale::new(TICK).unwrap();
        let new_stream = || BoundedStream::new(
            NonZeroUsize::new(m).unwrap(),
            NonZeroUsize::new(w).unwrap(),
            scale,
        )
        .expect("w is always m or more here");

        let mut sequential = new_stream();
        let expected: Vec<_> = series.iter().filter_map(|&x| sequential.update(x)).map(bits).collect();

        let mut chunked = new_stream();
        let mut out = Vec::new();
        chunked.update_many(&series[..split], &mut out);
        chunked.update_many(&series[split..], &mut out);
        let actual: Vec<_> = out.into_iter().map(bits).collect();

        prop_assert_eq!(actual, expected);
    }
}
