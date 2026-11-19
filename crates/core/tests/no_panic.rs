//! `GrowingStream`, `BoundedStream`, their non-normalized twins, `batch::left_profile` and
//! `Scale::new` never panic for any
//! input: an empty series, all-NaN, all-infinite, subnormal samples, values past `MAX_TICKS`,
//! any window length, any bound length (`BoundedStream` only), and any positive scale.

use core::num::NonZeroUsize;
use ladde::batch::left_profile;
use ladde::bounded::BoundedStream;
use ladde::index::SampleIndex;
use ladde::mdl::{maamp_mdl, mdl};
use ladde::motif::{
    Cutoffs, MotifParams, MultiMotifParams, mmotifs, mmotifs_nonnorm, motifs, motifs_nonnorm,
};
use ladde::nonnorm_bounded::BoundedNonNormStream;
use ladde::nonnorm_stream::GrowingNonNormStream;
use ladde::pattern::{
    MaxDistance, distance_profile, distance_profile_nonnorm, find_matches, find_matches_nonnorm,
};
use ladde::quantize::{MAX_TICKS, Scale};
use ladde::stream::GrowingStream;
use ladde::subspace::{Bits, Options, maamp_subspace, subspace};
use proptest::prelude::*;

/// A sample drawn to land on the boundaries every engine's arithmetic treats specially, not
/// just an ordinary finite value: NaN, both infinities, a genuine subnormal, both signed
/// zeros, the extremes of `f64`, and the tick count right at and right past
/// [`ladde::quantize::MAX_TICKS`].
fn adversarial_value() -> impl Strategy<Value = f64> {
    prop_oneof![
        3 => any::<f64>(),
        1 => Just(f64::NAN),
        1 => Just(f64::INFINITY),
        1 => Just(f64::NEG_INFINITY),
        1 => Just(0.0),
        1 => Just(-0.0),
        1 => Just(f64::MIN_POSITIVE / 2.0), // a genuine subnormal
        1 => Just(-f64::MIN_POSITIVE / 2.0),
        1 => Just(f64::MAX),
        1 => Just(f64::MIN),
        1 => Just(MAX_TICKS as f64),
        1 => Just(-(MAX_TICKS as f64)),
        1 => Just((MAX_TICKS + 1) as f64),
        1 => Just(-((MAX_TICKS + 1) as f64)),
    ]
}

fn adversarial_series() -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(adversarial_value(), 0..24)
}

/// Any positive scale a caller might pick, from the smallest subnormal through an ordinary
/// market tick to the largest finite value.
fn positive_scale() -> impl Strategy<Value = f64> {
    prop_oneof![
        Just(f64::MIN_POSITIVE),
        Just(1e-300),
        Just(0.01),
        Just(f64::MAX),
        1e-6..1e6,
    ]
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    /// Any `f64` bit pattern, including every non-finite and subnormal one: construction never
    /// panics, and it accepts exactly the positive finite values.
    #[test]
    fn scale_new_never_panics(bits in any::<u64>()) {
        let value = f64::from_bits(bits);
        let scale = Scale::new(value);
        prop_assert_eq!(scale.is_some(), value.is_finite() && value > 0.0);
    }
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    /// Any window length, any positive scale, any adversarial series: driving every sample
    /// through `update` never panics, however the window and the series length compare.
    #[test]
    fn growing_stream_never_panics(
        m in 1usize..32,
        scale_value in positive_scale(),
        series in adversarial_series(),
    ) {
        let m = NonZeroUsize::new(m).unwrap();
        let scale = Scale::new(scale_value).expect("positive_scale is always accepted");
        let mut stream = GrowingStream::new(m, scale);
        for &x in &series {
            let _ = stream.update(x);
        }
    }
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    /// Any window length, any positive scale, any adversarial series: never panics, whether the
    /// window is shorter or longer than the series itself.
    #[test]
    fn batch_left_profile_never_panics(
        m in 1usize..32,
        scale_value in positive_scale(),
        series in adversarial_series(),
    ) {
        let m = NonZeroUsize::new(m).unwrap();
        let scale = Scale::new(scale_value).expect("positive_scale is always accepted");
        let _ = left_profile(&series, m, scale);
    }
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    /// Same shape as [`growing_stream_never_panics`], with `w` drawn independently of `m` so
    /// `w < m`, `w == m` and `w` well past `m` are all just as reachable: construction never
    /// panics, `None` exactly when `w < m`, and driving every sample through whatever it
    /// returns never panics either.
    #[test]
    fn bounded_stream_never_panics(
        m in 1usize..32,
        w in 1usize..32,
        scale_value in positive_scale(),
        series in adversarial_series(),
    ) {
        let m = NonZeroUsize::new(m).unwrap();
        let w = NonZeroUsize::new(w).unwrap();
        let scale = Scale::new(scale_value).expect("positive_scale is always accepted");
        let stream = BoundedStream::new(m, w, scale);
        prop_assert_eq!(stream.is_some(), w.get() >= m.get());
        if let Some(mut stream) = stream {
            for &x in &series {
                let _ = stream.update(x);
            }
        }
    }
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    /// The non-normalized growing stream, driven like [`growing_stream_never_panics`].
    #[test]
    fn growing_nonnorm_stream_never_panics(
        m in 1usize..32,
        scale_value in positive_scale(),
        series in adversarial_series(),
    ) {
        let m = NonZeroUsize::new(m).unwrap();
        let scale = Scale::new(scale_value).expect("positive_scale is always accepted");
        let mut stream = GrowingNonNormStream::new(m, scale);
        for &x in &series {
            let _ = stream.update(x);
        }
        let _ = GrowingNonNormStream::restore(&stream.checkpoint()).expect("own checkpoint restores");
    }

    /// The non-normalized bounded stream, driven like [`bounded_stream_never_panics`].
    #[test]
    fn bounded_nonnorm_stream_never_panics(
        m in 1usize..32,
        w in 1usize..32,
        scale_value in positive_scale(),
        series in adversarial_series(),
    ) {
        let m = NonZeroUsize::new(m).unwrap();
        let w = NonZeroUsize::new(w).unwrap();
        let scale = Scale::new(scale_value).expect("positive_scale is always accepted");
        let stream = BoundedNonNormStream::new(m, w, scale);
        prop_assert_eq!(stream.is_some(), w.get() >= m.get());
        if let Some(mut stream) = stream {
            for &x in &series {
                let _ = stream.update(x);
            }
            let _ = BoundedNonNormStream::restore(&stream.checkpoint()).expect("own checkpoint restores");
        }
    }
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    /// Any window length, scales, bit count, pair of starts, `k` and `include` over adversarial
    /// dimensions (equally long or not): `subspace` and `maamp_subspace` never panic; so do `mdl` and `maamp_mdl` with starts of any count.
    #[test]
    fn subspace_never_panics(
        m in 1usize..12,
        scale_values in prop::collection::vec(positive_scale(), 3),
        rows in prop::collection::vec(adversarial_series(), 0..4),
        bits in 0u8..10,
        pair in (0usize..30, 0usize..30),
        k in 0usize..5,
        include in prop::collection::vec(0usize..6, 0..4),
        discords in any::<bool>(),
    ) {
        let m = NonZeroUsize::new(m).unwrap();
        let Ok(bits) = Bits::new(bits) else { return Ok(()); };
        let options = Options { include: &include, discords, bits, ..Options::default() };
        let set: Vec<(&[f64], Scale)> = rows
            .iter()
            .zip(&scale_values)
            .map(|(row, &s)| (&row[..], Scale::new(s).expect("positive_scale is always accepted")))
            .collect();
        let plain: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
        let _ = subspace(&set, m, pair, k, &options);
        let _ = maamp_subspace(&plain, m, pair, k, &options);
        let starts = [pair.0, pair.1, k, pair.0 + 1];
        let _ = mdl(&set, m, &starts[..rows.len().min(4)], &starts[1..=rows.len().min(3)], &options);
        let _ = maamp_mdl(&plain, m, &starts[..rows.len().min(4)], &starts[..rows.len().min(4)], &options);
    }
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    /// Multi-dimensional motifs under both metrics, for adversarial dimensions (equally long or
    /// not), profiles of any shape with non-finite distances and any neighbors, and any `k`,
    /// `include`, counts, thresholds and cutoffs.
    #[test]
    fn multi_dimensional_motifs_never_panic(
        m in 1usize..8,
        scale_values in prop::collection::vec(positive_scale(), 3),
        base in adversarial_series(),
        dimensions in 0usize..4,
        distances in adversarial_series(),
        neighbors in prop::collection::vec(0u64..40, 1..8),
        profile_rows in 0usize..8,
        k in prop_oneof![Just(None), (0usize..5).prop_map(Some)],
        include in prop::collection::vec(0usize..5, 0..3),
        threshold in prop_oneof![Just(None), Just(Some(f64::NAN)), Just(Some(-1.0)), Just(Some(2.5))],
        cutoff in prop_oneof![Just(None), Just(Some(f64::NAN)), Just(Some(f64::INFINITY)), Just(Some(1.0))],
        max_matches in prop_oneof![Just(None), Just(Some(0usize)), Just(Some(3usize))],
        atol in prop_oneof![Just(0.0), Just(f64::NAN), Just(1e-8)],
        min_neighbors in 0usize..3,
    ) {
        let m = NonZeroUsize::new(m).unwrap();
        // Every dimension a rotation of one series, so they are equally long.
        let rows: Vec<Vec<f64>> = (0..dimensions)
            .map(|r| {
                let mut row = base.clone();
                row.rotate_left((3 * r).min(base.len()));
                row
            })
            .collect();
        let set: Vec<(&[f64], Scale)> = rows
            .iter()
            .zip(&scale_values)
            .map(|(row, &s)| (&row[..], Scale::new(s).expect("positive_scale is always accepted")))
            .collect();
        let plain: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
        let windows = rows.first().map_or(0, |row| (row.len() + 1).saturating_sub(m.get()));
        // Rows of the right length, mostly as many as dimensions and sometimes not.
        let profile_rows = if profile_rows < 6 { dimensions } else { profile_rows - 5 };
        let distance: Vec<Vec<f64>> = (0..profile_rows)
            .map(|r| (0..windows).map(|i| distances.get((i + r) % distances.len().max(1)).copied().unwrap_or(0.0)).collect())
            .collect();
        let index: Vec<Vec<SampleIndex>> = (0..profile_rows)
            .map(|r| {
                (0..windows)
                    .map(|i| {
                        let at = neighbors[(i + r) % neighbors.len()];
                        SampleIndex::new(at).filter(|_| at < 38).unwrap_or(SampleIndex::NONE)
                    })
                    .collect()
            })
            .collect();
        let params = MultiMotifParams {
            min_neighbors,
            max_distance: threshold.map_or(MaxDistance::Default, MaxDistance::Value),
            cutoffs: cutoff.map_or(Cutoffs::Default, Cutoffs::All),
            max_matches,
            max_motifs: 3,
            atol,
            k,
            include: &include,
        };
        let _ = mmotifs(&set, m, &distance, &index, &params);
        let scale = Scale::new(scale_values[0]).expect("positive_scale is always accepted");
        let _ = mmotifs_nonnorm(&plain, m, scale, &distance, &index, &params);
    }
}

/// `BoundedStream::new` returns `None` when `w` is below `m`, the one shape no window that
/// long could ever stay inside, instead of panicking.
#[test]
fn bounded_stream_new_is_none_when_w_is_less_than_m() {
    let m = NonZeroUsize::new(2).unwrap();
    let w = NonZeroUsize::new(1).unwrap();
    let scale = Scale::new(1.0).unwrap();
    assert!(BoundedStream::new(m, w, scale).is_none());
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    /// Matching and motifs under both metrics, for any adversarial series, query, scale,
    /// threshold and counts (a scale so large every distance overflows included).
    #[test]
    fn matching_and_motifs_never_panic(
        m in 1usize..8,
        scale_value in positive_scale(),
        series in adversarial_series(),
        query in adversarial_series(),
        threshold in prop_oneof![
            Just(None), Just(Some(f64::NAN)), Just(Some(f64::INFINITY)), Just(Some(-1.0)),
            Just(Some(0.0)), Just(Some(2.5))
        ],
        cutoff in prop_oneof![Just(None), Just(Some(f64::NAN)), Just(Some(f64::INFINITY)), Just(Some(1.0))],
        max_matches in prop_oneof![Just(None), Just(Some(0usize)), Just(Some(3usize))],
        min_neighbors in 0usize..3,
    ) {
        let scale = Scale::new(scale_value).expect("positive_scale is always accepted");
        let max_distance = threshold.map_or(MaxDistance::Default, MaxDistance::Value);
        let _ = find_matches(&query, &series, scale, max_distance, max_matches);
        let _ = find_matches_nonnorm(&query, &series, scale, max_distance, max_matches);
        let _ = distance_profile(&query, &series, scale);
        let _ = distance_profile_nonnorm(&query, &series, scale);
        let params = MotifParams {
            min_neighbors,
            max_distance,
            cutoff,
            max_matches,
            max_motifs: 3, ..MotifParams::default() };
        let m = NonZeroUsize::new(m).unwrap();
        let one = NonZeroUsize::new(1).unwrap();
        let _ = motifs(&series, m, scale, one, &params);
        let _ = motifs_nonnorm(&series, m, scale, one, &params);
    }
}

/// A scale so large that every Euclidean distance overflows to infinity gives no motif, instead of
/// panicking: the only pair of windows is two ticks apart.
#[test]
fn a_scale_that_overflows_every_distance_gives_nothing() {
    let scale = Scale::new(1.7e308).unwrap();
    let series = [1.7e308, 0.0, -1.7e308];
    let m = NonZeroUsize::new(1).unwrap();
    let one = NonZeroUsize::new(1).unwrap();
    assert!(motifs_nonnorm(&series, m, scale, one, &MotifParams::default()).is_empty());
    let params = MotifParams {
        cutoff: Some(f64::INFINITY),
        max_distance: MaxDistance::Value(f64::INFINITY),
        ..MotifParams::default()
    };
    assert!(motifs_nonnorm(&series, m, scale, one, &params).is_empty());
}
