extern crate std;

use alloc::vec::Vec;
use core::num::NonZeroUsize;

use proptest::prelude::*;

use super::{
    Bits, Discretizer, EDGES, Options, Subspace, edge_below, edge_shift, linear_bin,
    maamp_subspace, subspace,
};
use crate::multidim::{MultiError, NonFinitePolicy};
use crate::naive::{naive_edge_below, naive_maamp_subspace, naive_subspace};
use crate::quantize::Scale;
use ethnum::I256;

fn m(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).unwrap()
}

fn bits(value: u8) -> Bits {
    Bits::new(value).unwrap()
}

fn scale(value: f64) -> Scale {
    Scale::new(value).unwrap()
}

fn options<'a>(include: &'a [usize], discords: bool, bits: Bits) -> Options<'a> {
    Options {
        include,
        discords,
        bits,
        non_finite: NonFinitePolicy::Missing,
    }
}

#[test]
fn bits_are_one_to_eight() {
    assert_eq!(Bits::new(0), Err(MultiError::BitsOutOfRange));
    assert_eq!(Bits::new(9), Err(MultiError::BitsOutOfRange));
    assert_eq!(Bits::new(255), Err(MultiError::BitsOutOfRange));
    for value in 1..=8 {
        let b = bits(value);
        assert_eq!(b.get(), value);
        assert_eq!(b.edge_count(), (1 << value) - 1);
        assert_eq!(b.middle_bin() as usize, (1 << (value - 1)) - 1);
    }
    assert_eq!(Bits::EIGHT, bits(8));
}

#[test]
fn the_edge_table_is_a_sorted_symmetric_normal_quantile_table() {
    assert_eq!(EDGES.len(), 255);
    assert_eq!(EDGES[127].to_bits(), 0.0f64.to_bits());
    for i in 0..127 {
        assert_eq!(EDGES[i], -EDGES[254 - i], "{i}");
    }
    for pair in EDGES.windows(2) {
        assert!(pair[0] < pair[1]);
    }
    // The quartile, and the lowest edge, correctly rounded (scipy gives -2.6600674686174592).
    assert_eq!(bits(2).edge(1), 0.0);
    assert_eq!(bits(2).edge(2), 0.6744897501960817);
    assert_eq!(bits(2).edge(0), -0.6744897501960817);
    assert_eq!(bits(1).edge(0), 0.0);
    assert_eq!(bits(8).edge(0), -2.6600674686174597);
}

#[test]
fn fewer_bits_use_every_stride_th_edge() {
    for value in 1..=8u8 {
        let b = bits(value);
        for i in 0..b.edge_count() {
            assert_eq!(b.edge(i), EDGES[((i + 1) << (8 - value)) - 1]);
        }
    }
}

#[test]
fn every_edge_square_fits_the_exact_comparison() {
    for &edge in EDGES.iter().filter(|&&edge| edge != 0.0) {
        assert!(edge_shift(edge) <= super::MAX_EDGE_SHIFT);
        assert!(edge_shift(edge) >= 1);
    }
}

fn isqrt(value: I256) -> I256 {
    let (mut low, mut high) = (I256::ZERO, I256::ONE << 100);
    while low < high {
        let middle = (low + high + 1) >> 1;
        if middle * middle <= value {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    low
}

/// The largest `a` with `a <= |edge| * sqrt(variance)`.
fn floor_offset(edge: f64, variance: u64) -> i128 {
    let mut value = edge.abs();
    let mut shift = 0;
    while value.fract() != 0.0 {
        value *= 2.0;
        shift += 1;
    }
    let mantissa = I256::from(value as u64);
    let squared = (mantissa * mantissa * I256::from(variance)) >> (2 * shift);
    i128::try_from(isqrt(squared)).unwrap()
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 1
    }
}

#[test]
fn a_sample_next_to_every_edge_is_decided_exactly() {
    let mut rng = Lcg(0x5eed);
    let mut variances: Vec<u64> = (1..40).collect();
    variances.extend((0..60).map(|_| rng.next() % (1 << 62) + 1));
    variances.extend([(1 << 63) - 1, (1 << 63) - 2, 1 << 62, 3 << 61]);
    let mut deciding = 0;
    for &edge in EDGES.iter() {
        for &variance in &variances {
            let floor = floor_offset(edge, variance);
            for a in (floor - 2..=floor + 2).flat_map(|a| [a, -a]) {
                let exact = naive_edge_below(edge, a, variance);
                assert_eq!(
                    edge_below(edge, a, variance),
                    exact,
                    "{edge} {a} {variance}"
                );
                deciding += usize::from(exact);
            }
            if edge > 0.0 {
                assert!(!edge_below(edge, floor, variance));
                assert!(edge_below(edge, floor + 1, variance));
            }
            if edge < 0.0 {
                assert!(!edge_below(edge, -floor - 1, variance));
            }
        }
    }
    assert!(deciding > 0);
}

#[test]
fn an_edge_equal_to_the_value_is_not_below_it() {
    // 0.5 · sqrt(4) = 1 exactly.
    assert!(!edge_below(0.5, 1, 4));
    assert!(edge_below(0.5, 2, 4));
    assert!(!edge_below(-0.5, -1, 4));
    assert!(edge_below(-0.5, 0, 4));
    assert!(!edge_below(-0.5, -2, 4));
    assert!(!edge_below(0.0, 0, 7));
    assert!(edge_below(0.0, 1, 7));
    assert!(!edge_below(0.0, -1, 7));
    assert!(edge_below(-0.25, 0, 7));
    assert!(!edge_below(0.25, 0, 7));
    for a in [-3, -2, -1, 0, 1, 2, 3] {
        for variance in [1, 4, 16, 64] {
            for edge in [0.5, -0.5, 0.25, -0.25, 1.0, -1.0, 2.0, -2.0] {
                assert_eq!(
                    edge_below(edge, a, variance),
                    naive_edge_below(edge, a, variance),
                    "{edge} {a} {variance}"
                );
            }
        }
    }
}

fn series_of(ticks: &[i64], tick: f64) -> Vec<f64> {
    ticks.iter().map(|&t| t as f64 * tick).collect()
}

#[test]
fn the_sample_at_the_mean_is_in_the_middle_bin_and_the_one_above_it_is_not() {
    // z = -sqrt(1.5), 0, +sqrt(1.5): edges strictly below 0 are 127 of them.
    let series = series_of(&[-1, 0, 1], 1.0);
    let only = [(&series[..], scale(1.0))];
    for value in 1..=8u8 {
        let b = bits(value);
        let found = Discretizer::z_normalized(&only, m(3), b).unwrap();
        let bins = found.window_bins(0, 0).unwrap();
        assert_eq!(bins[1], b.middle_bin(), "{value}");
        assert_eq!(bins[0] + bins[2], 2 * b.middle_bin() + 1, "{value}");
        assert!(bins[2] > b.middle_bin());
    }
}

#[test]
fn a_constant_window_is_all_middle_bins() {
    let series = alloc::vec![7.0; 6];
    let only = [(&series[..], scale(1.0))];
    for value in 1..=8u8 {
        let found = Discretizer::z_normalized(&only, m(4), bits(value)).unwrap();
        assert_eq!(
            found.window_bins(0, 1).unwrap(),
            alloc::vec![bits(value).middle_bin(); 4]
        );
    }
}

#[test]
fn a_window_with_a_gap_or_out_of_range_has_no_bins() {
    let gap = [1.0, 2.0, f64::NAN, 4.0, 5.0, 7.0];
    let found = Discretizer::z_normalized(&[(&gap[..], scale(1.0))], m(3), Bits::EIGHT).unwrap();
    for start in 0..=2 {
        assert!(found.window_bins(0, start).is_none(), "{start}");
    }
    assert!(found.window_bins(0, 3).is_some());
    // A variance numerator of 2^64 is beyond the arithmetic.
    let wild = [0.0, 1e30, 0.0, 1.0, 2.0, 4.0];
    let found = Discretizer::z_normalized(&[(&wild[..], scale(1.0))], m(3), Bits::EIGHT).unwrap();
    assert!(found.window_bins(0, 0).is_none());
    assert!(found.window_bins(0, 3).is_some());
}

#[test]
fn a_window_is_in_range_exactly_below_a_variance_numerator_of_two_to_the_sixty_three() {
    // With two ticks 0 and r the numerator is r squared.
    let inside = [0.0, 3037000499.0, 0.0, 1.0];
    let found = Discretizer::z_normalized(&[(&inside[..], scale(1.0))], m(2), Bits::EIGHT).unwrap();
    assert!(found.window_bins(0, 0).is_some());
    let outside = [0.0, 3037000500.0, 0.0, 1.0];
    let found =
        Discretizer::z_normalized(&[(&outside[..], scale(1.0))], m(2), Bits::EIGHT).unwrap();
    assert!(found.window_bins(0, 0).is_none());
    assert!(found.window_bins(0, 2).is_some());
}

#[test]
fn a_window_below_the_standard_deviation_threshold_is_not_scaled() {
    // Spread of 1e-9 per tick: standard deviation about 8e-10 in the samples' units.
    let tiny = [0.0, 1.0, 2.0, 3.0, 4.0].map(|t| t * 1e-9);
    let found = Discretizer::z_normalized(&[(&tiny[..], scale(1e-9))], m(5), Bits::EIGHT).unwrap();
    assert_eq!(
        found.window_bins(0, 0).unwrap(),
        alloc::vec![127, 127, 127, 128, 128],
        "not above the mean: 127; above it: 128"
    );
    // The same shape a thousand times larger is scaled by its own deviation.
    let wide = [0.0, 1.0, 2.0, 3.0, 4.0].map(|t| t * 1e-6);
    let found = Discretizer::z_normalized(&[(&wide[..], scale(1e-6))], m(5), Bits::EIGHT).unwrap();
    let bins = found.window_bins(0, 0).unwrap();
    assert_eq!(bins[2], 127);
    assert!(bins[0] < 30 && bins[4] > 220, "{bins:?}");
}

#[test]
fn a_deviation_of_exactly_the_threshold_is_scaled() {
    // Ticks 0 and 2 have variance numerator 4: a standard deviation of exactly one tick.
    let threshold = [0.0, 2e-7];
    let found =
        Discretizer::z_normalized(&[(&threshold[..], scale(1e-7))], m(2), Bits::EIGHT).unwrap();
    let bins = found.window_bins(0, 0).unwrap();
    assert!(bins[0] < 50 && bins[1] > 200, "{bins:?}");
    let below = [0.0, 2.0 * 9.9e-8];
    let found =
        Discretizer::z_normalized(&[(&below[..], scale(9.9e-8))], m(2), Bits::EIGHT).unwrap();
    assert_eq!(found.window_bins(0, 0).unwrap(), [127, 128]);
}

#[test]
fn the_deviation_rule_divides_the_offset_by_the_window_length() {
    // One tick of 60 among 5000 zeros at a tick of 1e-7: the deviation is 8.5e-8, below the
    // threshold, and the spike is 60 ticks above the mean, so one bin up.
    let mut series = alloc::vec![0.0; 5000];
    series[100] = 60.0 * 1e-7;
    let found =
        Discretizer::z_normalized(&[(&series[..], scale(1e-7))], m(5000), Bits::EIGHT).unwrap();
    let bins = found.window_bins(0, 0).unwrap();
    assert_eq!(bins[100], 128);
    assert!(
        bins.iter()
            .enumerate()
            .all(|(i, &bin)| i == 100 || bin == 127)
    );
}

#[test]
fn the_deviation_threshold_follows_the_scale_of_the_samples() {
    let ticks = [0.0, 3.0, 1.0, 4.0, 2.0];
    let unit: Vec<f64> = ticks.iter().map(|t| t * 1e-7).collect();
    let found = Discretizer::z_normalized(&[(&unit[..], scale(1e-7))], m(5), Bits::EIGHT).unwrap();
    // sigma = sqrt(2) * 1e-7 is not below the threshold.
    assert!(found.window_bins(0, 0).unwrap()[0] < 50);
    let small: Vec<f64> = ticks.iter().map(|t| t * 6e-8).collect();
    let found = Discretizer::z_normalized(&[(&small[..], scale(6e-8))], m(5), Bits::EIGHT).unwrap();
    // sigma = sqrt(2) * 6e-8 = 8.5e-8 is.
    assert_eq!(found.window_bins(0, 0).unwrap()[0], 127);
}

#[test]
fn linear_bins_round_ties_to_even() {
    // (x - 0) / 1 * 1 = 0.5: even is 0, so bin 1; 1.5 -> 2, 2.5 -> 2 with three steps.
    assert_eq!(linear_bin(0.5, 0.0, 1.0, bits(1)), 1);
    assert_eq!(linear_bin(0.0, 0.0, 1.0, bits(1)), 1);
    assert_eq!(linear_bin(1.0, 0.0, 1.0, bits(1)), 2);
    assert_eq!(linear_bin(0.5, 0.0, 1.0, bits(2)), 3);
    assert_eq!(linear_bin(1.5, 0.0, 3.0, bits(2)), 3);
    assert_eq!(linear_bin(7.0, 7.0, 7.0, Bits::EIGHT), 1);
    assert_eq!(linear_bin(255.0, 0.0, 255.0, Bits::EIGHT), 256);
    assert_eq!(linear_bin(1.0, -1e308, 1e308, Bits::EIGHT), 1);
}

fn dimension_set(rows: &[&[f64]]) -> Vec<Vec<f64>> {
    rows.iter().map(|row| row.to_vec()).collect()
}

fn borrowed(rows: &[Vec<f64>]) -> Vec<(&[f64], Scale)> {
    rows.iter().map(|row| (&row[..], scale(1.0))).collect()
}

fn plain(rows: &[Vec<f64>]) -> Vec<&[f64]> {
    rows.iter().map(|row| &row[..]).collect()
}

#[test]
fn dimensions_are_ordered_by_bin_distance_ties_to_the_smaller_dimension() {
    // Dimension 1 repeats exactly (distance 0) between the windows at 0 and 4; dimensions 0 and
    // 2 are two different shapes with equal distances.
    let rows = dimension_set(&[
        &[0.0, 1.0, 5.0, 2.0, 1.0, 0.0, 5.0, 2.0],
        &[0.0, 1.0, 3.0, 2.0, 0.0, 1.0, 3.0, 2.0],
        &[0.0, 1.0, 5.0, 2.0, 1.0, 0.0, 5.0, 2.0],
    ]);
    let set = borrowed(&rows);
    let found = |k| subspace(&set, m(4), (0, 4), k, &Options::default()).unwrap();
    assert_eq!(found(0).dimensions, [1]);
    assert_eq!(found(1).dimensions, [1, 0]);
    assert_eq!(found(2).dimensions, [1, 0, 2]);
    assert!(found(2).complete);
    let farthest = |k| subspace(&set, m(4), (0, 4), k, &options(&[], true, Bits::EIGHT)).unwrap();
    assert_eq!(farthest(0).dimensions, [0]);
    assert_eq!(farthest(1).dimensions, [0, 2]);
    assert_eq!(farthest(2).dimensions, [0, 2, 1]);
}

#[test]
fn include_dimensions_come_first_ordered_by_their_own_distance() {
    let rows = dimension_set(&[
        &[0.0, 1.0, 5.0, 2.0, 1.0, 0.0, 5.0, 2.0],
        &[0.0, 1.0, 3.0, 2.0, 0.0, 1.0, 3.0, 2.0],
        &[0.0, 4.0, 5.0, 1.0, 4.0, 0.0, 5.0, 2.0],
    ]);
    let set = borrowed(&rows);
    let found = |include: &[usize], discords| {
        subspace(
            &set,
            m(4),
            (0, 4),
            2,
            &options(include, discords, Bits::EIGHT),
        )
        .unwrap()
        .dimensions
    };
    let free = found(&[], false);
    assert_eq!(free.len(), 3);
    for include in [[2usize, 0], [0, 2], [2, 2], [0, 0]] {
        let got = found(&include, false);
        let front = &got[..include
            .iter()
            .collect::<alloc::collections::BTreeSet<_>>()
            .len()];
        let mut by_distance: Vec<usize> = free
            .iter()
            .copied()
            .filter(|dimension| include.contains(dimension))
            .collect();
        by_distance.dedup();
        assert_eq!(front, &by_distance[..], "{include:?}");
    }
    // Only the first k + 1 count: one included dimension at k = 0, even when it is the farthest.
    let first = subspace(&set, m(4), (0, 4), 0, &options(&[2], false, Bits::EIGHT)).unwrap();
    assert_eq!(first.dimensions, [2]);
    let farthest = subspace(&set, m(4), (0, 4), 0, &options(&[1], true, Bits::EIGHT)).unwrap();
    assert_eq!(farthest.dimensions, [1]);
}

#[test]
fn a_dimension_without_a_distance_sorts_last_in_both_directions() {
    let mut rows = dimension_set(&[
        &[0.0, 1.0, 5.0, 2.0, 1.0, 0.0, 5.0, 2.0],
        &[0.0, 1.0, 3.0, 2.0, 0.0, 1.0, 3.0, 2.0],
        &[0.0, 4.0, 5.0, 1.0, 4.0, 0.0, 5.0, 2.0],
    ]);
    rows[0][5] = f64::NAN;
    let set = borrowed(&rows);
    for discords in [false, true] {
        let found =
            |k| subspace(&set, m(4), (0, 4), k, &options(&[], discords, Bits::EIGHT)).unwrap();
        assert_eq!(found(2).dimensions.last(), Some(&0));
        assert!(found(0).complete);
        assert!(found(1).complete);
        assert!(!found(2).complete);
    }
    // Included, it still sorts last among the included, and the subspace is incomplete.
    let found = subspace(&set, m(4), (0, 4), 1, &options(&[0, 2], false, Bits::EIGHT)).unwrap();
    assert_eq!(found.dimensions, [2, 0]);
    assert!(!found.complete);
}

#[test]
fn gaps_at_the_ends_of_the_series_make_only_the_windows_they_touch_missing() {
    let mut leading = alloc::vec![f64::NAN, f64::NAN];
    leading.extend((0..8).map(|t| f64::from(t * t % 7)));
    let mut trailing: Vec<f64> = (0..8).map(|t| f64::from(t * t % 5)).collect();
    trailing.extend([f64::NAN, f64::NAN]);
    for series in [&leading, &trailing] {
        let plain = [&series[..]];
        let linear = Discretizer::linear(&plain, m(4), Bits::EIGHT).unwrap();
        let normalized =
            Discretizer::z_normalized(&[(&series[..], scale(1.0))], m(4), Bits::EIGHT).unwrap();
        for start in 0..=series.len() - 4 {
            let touched = series[start..start + 4].iter().any(|x| x.is_nan());
            assert_eq!(linear.window_bins(0, start).is_none(), touched, "{start}");
            assert_eq!(
                normalized.window_bins(0, start).is_none(),
                touched,
                "{start}"
            );
        }
    }
}

#[test]
fn infinities_outside_the_windows_leave_the_linear_range_to_the_finite_samples() {
    let finite: Vec<Vec<f64>> = (0..3)
        .map(|d| (0..30).map(|i| f64::from((i * (d + 3) + d) % 11)).collect())
        .collect();
    let mut wild = finite.clone();
    wild[1][27] = f64::INFINITY;
    wild[2][28] = f64::NEG_INFINITY;
    wild[0][29] = f64::NAN;
    let plain = |rows: &[Vec<f64>]| {
        let set: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
        (0..3)
            .map(|k| maamp_subspace(&set, m(5), (2, 14), k, &Options::default()).unwrap())
            .collect::<Vec<_>>()
    };
    assert_eq!(plain(&wild), plain(&finite));
    let set: Vec<&[f64]> = wild.iter().map(Vec::as_slice).collect();
    let found = Discretizer::linear(&set, m(5), Bits::EIGHT).unwrap();
    let reference: Vec<&[f64]> = finite.iter().map(Vec::as_slice).collect();
    let expected = Discretizer::linear(&reference, m(5), Bits::EIGHT).unwrap();
    for dimension in 0..3 {
        assert_eq!(
            found.window_bins(dimension, 2),
            expected.window_bins(dimension, 2)
        );
    }
}

#[test]
fn a_window_with_an_infinity_is_missing_when_not_normalized() {
    for bad in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        let mut series: Vec<f64> = (0..12).map(|i| f64::from(i % 5)).collect();
        series[4] = bad;
        let set = [&series[..]];
        let found = Discretizer::linear(&set, m(4), Bits::EIGHT).unwrap();
        for start in 0..=8 {
            assert_eq!(
                found.window_bins(0, start).is_none(),
                (1..=4).contains(&start),
                "{bad} {start}"
            );
        }
    }
}

#[test]
fn bad_arguments_are_errors() {
    let rows = dimension_set(&[
        &[0.0, 1.0, 5.0, 2.0, 1.0, 0.0],
        &[0.0, 1.0, 3.0, 2.0, 0.0, 1.0],
    ]);
    let set = borrowed(&rows);
    let ok = Options::default();
    assert_eq!(
        subspace::<&[f64]>(&[], m(3), (0, 1), 0, &ok),
        Err(MultiError::NoDimensions)
    );
    assert_eq!(
        maamp_subspace(&[], m(3), (0, 1), 0, &ok),
        Err(MultiError::NoDimensions)
    );
    let short = [1.0, 2.0];
    let uneven = [(&rows[0][..], scale(1.0)), (&short[..], scale(1.0))];
    assert_eq!(
        subspace(&uneven, m(3), (0, 1), 0, &ok),
        Err(MultiError::UnequalLengths)
    );
    assert_eq!(
        subspace(&set, m(7), (0, 0), 0, &ok),
        Err(MultiError::WindowTooLong)
    );
    assert_eq!(
        subspace(&set, m(3), (0, 1), 2, &ok),
        Err(MultiError::KOutOfRange)
    );
    assert_eq!(
        subspace(&set, m(3), (0, 1), usize::MAX, &ok),
        Err(MultiError::KOutOfRange)
    );
    assert_eq!(
        subspace(&set, m(3), (0, 1), 0, &options(&[2], false, Bits::EIGHT)),
        Err(MultiError::IncludeOutOfRange)
    );
    assert_eq!(
        subspace(&set, m(3), (0, 4), 0, &ok),
        Err(MultiError::WindowOutOfRange)
    );
    assert_eq!(
        subspace(&set, m(3), (4, 0), 0, &ok),
        Err(MultiError::WindowOutOfRange)
    );
    assert_eq!(
        subspace(&set, m(3), (usize::MAX, 0), 0, &ok),
        Err(MultiError::WindowOutOfRange)
    );
    assert!(subspace(&set, m(3), (3, 3), 1, &ok).is_ok());
    assert!(subspace(&set, m(6), (0, 0), 1, &ok).is_ok());
    let plain = plain(&rows);
    assert_eq!(
        maamp_subspace(&plain, m(3), (0, 4), 0, &ok),
        Err(MultiError::WindowOutOfRange)
    );
    assert_eq!(
        maamp_subspace(&plain, m(3), (0, 1), 2, &ok),
        Err(MultiError::KOutOfRange)
    );
}

#[test]
fn a_series_without_a_finite_sample_has_no_distances() {
    let nothing = [f64::NAN; 5];
    let rows = [&nothing[..]];
    let found = maamp_subspace(&rows, m(3), (0, 2), 0, &Options::default()).unwrap();
    assert_eq!(
        found,
        Subspace {
            dimensions: alloc::vec![0],
            complete: false
        }
    );
    let found = subspace(
        &[(&nothing[..], scale(1.0))],
        m(3),
        (0, 2),
        0,
        &Options::default(),
    );
    assert!(!found.unwrap().complete);
}

#[test]
fn non_normalized_dimensions_are_binned_between_the_extremes_of_every_dimension() {
    // Dimension 0 spans 0..8 inside windows; dimension 1 reaches 1000 elsewhere, which squeezes
    // dimension 0 into the lowest bins: the two windows then differ in no bin there.
    let rows = dimension_set(&[
        &[0.0, 1.0, 8.0, 0.0, 1.0, 8.0, 0.0, 1.0, 7.0],
        &[0.0, 500.0, 1000.0, 0.0, 600.0, 1000.0, 0.0, 400.0, 900.0],
    ]);
    let set = plain(&rows);
    let found = maamp_subspace(&set, m(3), (0, 3), 0, &Options::default()).unwrap();
    assert_eq!(found.dimensions, [0]);
    let found = maamp_subspace(&set, m(3), (0, 6), 0, &Options::default()).unwrap();
    assert_eq!(found.dimensions, [0]);
    let found = maamp_subspace(&set, m(3), (0, 3), 0, &options(&[], true, Bits::EIGHT)).unwrap();
    assert_eq!(found.dimensions, [1]);
}

fn grid_value() -> impl Strategy<Value = f64> {
    prop_oneof![
        8 => (-20i32..=20).prop_map(f64::from),
        1 => Just(f64::NAN),
        1 => Just(3.0),
    ]
}

fn dimension(len: usize) -> impl Strategy<Value = Vec<f64>> {
    prop_oneof![
        4 => proptest::collection::vec(grid_value(), len),
        1 => proptest::collection::vec(prop_oneof![Just(0.0), Just(1.0)], len),
        1 => Just(alloc::vec![2.0; len]),
        1 => (0.0f64..40.0, proptest::collection::vec(0u8..3, len))
            .prop_map(|(base, steps)| steps.iter().map(|&s| base + f64::from(s)).collect()),
        1 => proptest::collection::vec(any::<f64>(), len),
        1 => (0i64..(1 << 50), proptest::collection::vec(0u8..9, len))
            .prop_map(|(base, steps)| steps.iter().map(|&s| (base + i64::from(s)) as f64).collect()),
        1 => proptest::collection::vec(-(1i64 << 28)..(1 << 28), len)
            .prop_map(|ticks| ticks.iter().map(|&t| t as f64).collect()),
    ]
}

fn scale_choice() -> impl Strategy<Value = f64> {
    prop_oneof![
        3 => Just(1.0),
        2 => Just(0.25),
        1 => Just(2f64.powi(-30)),
        1 => Just(1e-9),
        1 => Just(2f64.powi(40)),
    ]
}

fn case() -> impl Strategy<Value = Case> {
    (1usize..=4, 3usize..=14).prop_flat_map(|(d, n)| {
        (
            proptest::collection::vec(dimension(n), d),
            proptest::collection::vec(scale_choice(), d),
            2usize..=n.min(6),
            any::<proptest::sample::Index>(),
            any::<proptest::sample::Index>(),
            any::<proptest::sample::Index>(),
            proptest::collection::vec(0usize..d + 1, 0..4),
            any::<bool>(),
            1u8..=8,
        )
            .prop_map(
                move |(rows, scales, window, a, b, k, include, discords, bits)| Case {
                    rows,
                    scales,
                    m: window,
                    first: a.index(n - window + 1),
                    second: b.index(n - window + 1),
                    k: k.index(d),
                    include,
                    discords,
                    bits,
                },
            )
    })
}

#[derive(Clone, Debug)]
struct Case {
    rows: Vec<Vec<f64>>,
    scales: Vec<f64>,
    m: usize,
    first: usize,
    second: usize,
    k: usize,
    include: Vec<usize>,
    discords: bool,
    bits: u8,
}

proptest! {
    #![proptest_config(testkit::generators::seeded(400))]

    #[test]
    fn z_normalized_matches_the_reference(case in case()) {
        let include: Vec<usize> = case.include.iter().copied().filter(|&i| i < case.rows.len()).collect();
        let set: Vec<(&[f64], Scale)> = case
            .rows
            .iter()
            .zip(&case.scales)
            .map(|(row, &s)| (&row[..], scale(s)))
            .collect();
        let b = bits(case.bits);
        let got = subspace(&set, m(case.m), (case.first, case.second), case.k, &options(&include, case.discords, b));
        let want = naive_subspace(&set, m(case.m), (case.first, case.second), case.k, &include, case.discords, b);
        prop_assert_eq!(got, want);
    }

    #[test]
    fn non_normalized_matches_the_reference(case in case()) {
        let include: Vec<usize> = case.include.iter().copied().filter(|&i| i < case.rows.len()).collect();
        let set = plain(&case.rows);
        let b = bits(case.bits);
        let got = maamp_subspace(&set, m(case.m), (case.first, case.second), case.k, &options(&include, case.discords, b));
        let want = naive_maamp_subspace(&set, m(case.m), (case.first, case.second), case.k, &include, case.discords, b);
        prop_assert_eq!(got, want);
    }

    #[test]
    fn an_out_of_range_include_is_an_error(case in case()) {
        let include = [case.rows.len() + case.k];
        let set = plain(&case.rows);
        let got = maamp_subspace(&set, m(case.m), (case.first, case.second), case.k, &options(&include, false, Bits::EIGHT));
        prop_assert_eq!(got, Err(MultiError::IncludeOutOfRange));
    }
}
