extern crate std;

use alloc::vec::Vec;
use core::num::NonZeroUsize;

use proptest::prelude::*;

use super::{LOG2, bit_size, describe, distinct_differences, maamp_mdl, mdl};
use crate::multidim::{MultiError, NonFinitePolicy};
use crate::naive::{naive_log2, naive_maamp_mdl, naive_mdl};
use crate::quantize::Scale;
use crate::subspace::{Bits, Discretizer, Options, Subspace};

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
fn the_logarithm_table_is_correctly_rounded_at_every_entry() {
    assert_eq!(LOG2.len(), 512);
    for (i, &logarithm) in LOG2.iter().enumerate() {
        let n = i as u64 + 1;
        assert_eq!(logarithm.to_bits(), naive_log2(n).to_bits(), "log2({n})");
    }
}

#[test]
fn the_logarithm_table_is_exact_at_powers_of_two_and_close_to_libm_elsewhere() {
    for (i, &logarithm) in LOG2.iter().enumerate() {
        let n = i + 1;
        if n.is_power_of_two() {
            assert_eq!(logarithm, f64::from(n.trailing_zeros()));
        } else {
            assert!((logarithm - libm::log2(n as f64)).abs() <= 2.0 * f64::EPSILON * logarithm);
        }
    }
    for pair in LOG2.windows(2) {
        assert!(pair[0] < pair[1]);
    }
}

#[test]
fn the_reference_logarithm_is_correct_beyond_the_table() {
    assert_eq!(naive_log2(1), 0.0);
    assert_eq!(naive_log2(1 << 40), 40.0);
    assert_eq!(naive_log2(3).to_bits(), 1.584962500721156f64.to_bits());
    assert_eq!(
        naive_log2(10).to_bits(),
        core::f64::consts::LOG2_10.to_bits()
    );
    for n in 513..2000 {
        let near = libm::log2(n as f64);
        assert!(
            (naive_log2(n) - near).abs() <= 2.0 * f64::EPSILON * near,
            "{n}"
        );
    }
}

#[test]
fn a_size_is_evaluated_in_stumpys_order() {
    // 8 * (2·2·3 − 1·3) + 3·log2(5) + 5·8, the sum formed left to right.
    let expected = (72.0 + 3.0 * LOG2[4]) + 40.0;
    assert_eq!(bit_size(8, 2, 1, 3, 5).to_bits(), expected.to_bits());
    assert_eq!(bit_size(8, 2, 1, 3, 1), 72.0 + 8.0);
    assert_eq!(bit_size(1, 3, 3, 4, 2), 1.0 * (24.0 - 12.0) + 12.0 + 2.0);
    // The sum is not reassociated: terms whose order matters.
    let large = bit_size(8, 1 << 30, 1, 1 << 20, 511);
    let whole = 8.0 * ((2u128 * (1 << 30) * (1 << 20) - (1 << 20)) as f64);
    assert_eq!(
        large.to_bits(),
        ((whole + (1 << 20) as f64 * LOG2[510]) + 511.0 * 8.0).to_bits()
    );
}

const DOCTEST: [[f64; 7]; 2] = [
    [584.0, -11.0, 23.0, 79.0, 1001.0, 0.0, -19.0],
    [1.0, 2.0, 4.0, 8.0, 16.0, 0.0, 32.0],
];

#[test]
fn stumpys_documented_example() {
    let rows: Vec<(&[f64], Scale)> = DOCTEST.iter().map(|row| (&row[..], scale(1.0))).collect();
    let found = mdl(&rows, m(3), &[0, 0], &[2, 4], &Options::default()).unwrap();
    assert_eq!(found.bit_sizes[0].to_bits(), 80.0f64.to_bits());
    assert_eq!(found.bit_sizes[1].to_bits(), 0x405b_e0a0_2756_f9c2);
    assert_eq!(found.subspaces[0].dimensions, [1]);
    assert_eq!(found.subspaces[1].dimensions, [0, 1]);
    assert!(found.subspaces.iter().all(|s| s.complete));
    let plain: Vec<&[f64]> = DOCTEST.iter().map(|row| &row[..]).collect();
    let found = maamp_mdl(&plain, m(3), &[0, 0], &[2, 4], &Options::default()).unwrap();
    assert_eq!(found.bit_sizes[0].to_bits(), 0x4059_3050_13ab_7ce1);
    assert_eq!(found.bit_sizes[1].to_bits(), 0x405b_e0a0_2756_f9c2);
    assert_eq!(found.subspaces[0].dimensions, [1]);
    assert_eq!(found.subspaces[1].dimensions, [1, 0]);
}

fn walks(d: usize, n: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % 17) as f64 - 8.0
    };
    (0..d)
        .map(|_| {
            let mut level = 0.0;
            (0..n)
                .map(|_| {
                    level += next();
                    level
                })
                .collect()
        })
        .collect()
}

fn borrowed(rows: &[Vec<f64>]) -> Vec<(&[f64], Scale)> {
    rows.iter().map(|row| (&row[..], scale(1.0))).collect()
}

#[test]
fn the_same_window_twice_has_one_distinct_difference() {
    let rows = walks(3, 30, 1);
    let set = borrowed(&rows);
    let found = mdl(&set, m(5), &[4, 4, 4], &[4, 4, 4], &Options::default()).unwrap();
    for (k, &size) in found.bit_sizes.iter().enumerate() {
        let subspace = k + 1;
        // 8 · (2·3·5 − s·5) + 0 + 1 · 8.
        assert_eq!(size, 8.0 * (30.0 - 5.0 * subspace as f64) + 8.0, "{k}");
    }
}

#[test]
fn the_number_of_distinct_differences_counts_every_dimension_of_the_subspace_together() {
    let pair = alloc::vec![
        Some((alloc::vec![3, 3, 3], alloc::vec![3, 4, 2])),
        Some((alloc::vec![5, 5, 5], alloc::vec![5, 5, 9])),
        None,
    ];
    let both = Subspace {
        dimensions: alloc::vec![0, 1],
        complete: true,
    };
    let only_first = Subspace {
        dimensions: alloc::vec![0],
        complete: true,
    };
    // Differences of dimension 0: {0, -1, 1}; of dimension 1: {0, -4}.
    assert_eq!(distinct_differences(&pair, &only_first, 4), 3);
    assert_eq!(distinct_differences(&pair, &both, 4), 4);
    let shifted = alloc::vec![Some((alloc::vec![0, 15], alloc::vec![15, 0]))];
    let wide = Subspace {
        dimensions: alloc::vec![0],
        complete: true,
    };
    assert_eq!(distinct_differences(&shifted, &wide, 4), 2);
}

#[test]
fn an_incomplete_subspace_has_an_infinite_size_and_the_others_a_finite_one() {
    let mut rows = walks(3, 30, 2);
    rows[1][7] = f64::NAN;
    let set = borrowed(&rows);
    let found = mdl(&set, m(5), &[3, 3, 3], &[20, 20, 20], &Options::default()).unwrap();
    assert!(found.bit_sizes[0].is_finite() && found.bit_sizes[1].is_finite());
    assert_eq!(found.bit_sizes[2], f64::INFINITY);
    assert_eq!(found.subspaces[2].dimensions.last(), Some(&1));
    assert_eq!(
        found
            .subspaces
            .iter()
            .map(|s| s.complete)
            .collect::<Vec<_>>(),
        [true, true, false]
    );
}

#[test]
fn bad_arguments_are_errors() {
    let rows = walks(2, 12, 3);
    let set = borrowed(&rows);
    let ok = Options::default();
    assert_eq!(
        mdl::<&[f64]>(&[], m(3), &[], &[], &ok).unwrap_err(),
        MultiError::NoDimensions
    );
    assert_eq!(
        mdl(&set, m(13), &[0, 0], &[0, 0], &ok).unwrap_err(),
        MultiError::WindowTooLong
    );
    assert_eq!(
        mdl(&set, m(3), &[0], &[0, 0], &ok).unwrap_err(),
        MultiError::WrongPairCount
    );
    assert_eq!(
        mdl(&set, m(3), &[0, 0], &[0], &ok).unwrap_err(),
        MultiError::WrongPairCount
    );
    assert_eq!(
        mdl(&set, m(3), &[0, 0], &[0, 0, 0], &ok).unwrap_err(),
        MultiError::WrongPairCount
    );
    assert_eq!(
        mdl(&set, m(3), &[0, 10], &[0, 0], &ok).unwrap_err(),
        MultiError::WindowOutOfRange
    );
    assert_eq!(
        mdl(&set, m(3), &[0, 0], &[0, usize::MAX], &ok).unwrap_err(),
        MultiError::WindowOutOfRange
    );
    assert!(mdl(&set, m(3), &[0, 9], &[9, 0], &ok).is_ok());
    assert_eq!(
        mdl(
            &set,
            m(3),
            &[0, 0],
            &[0, 0],
            &options(&[2], false, Bits::EIGHT)
        )
        .unwrap_err(),
        MultiError::IncludeOutOfRange
    );
    let plain: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    assert_eq!(
        maamp_mdl(&plain, m(3), &[0], &[0], &ok).unwrap_err(),
        MultiError::WrongPairCount
    );
    assert_eq!(
        maamp_mdl(&plain, m(3), &[10, 0], &[0, 0], &ok).unwrap_err(),
        MultiError::WindowOutOfRange
    );
    let uneven = [(&rows[0][..], scale(1.0)), (&rows[1][..11], scale(1.0))];
    assert_eq!(
        mdl(&uneven, m(3), &[0, 0], &[0, 0], &ok).unwrap_err(),
        MultiError::UnequalLengths
    );
}

#[test]
fn the_most_distinct_differences_possible_fit_the_table() {
    // Bins 0 and 255 against each other and every step between: 511 different differences.
    let first: Vec<u16> = (0..256).collect();
    let second: Vec<u16> = (0..256).rev().collect();
    let mut third: Vec<u16> = alloc::vec![0; 256];
    third[0] = 255;
    let pair = alloc::vec![Some((first, second)), Some((alloc::vec![255; 256], third))];
    let space = Subspace {
        dimensions: alloc::vec![0, 1],
        complete: true,
    };
    let distinct = distinct_differences(&pair, &space, 8);
    assert!(distinct <= 511);
    assert!(distinct > 255);
    let _ = bit_size(8, 2, 2, 256, distinct);
    let _ = bit_size(8, 2, 2, 256, 511);
    let _ = bit_size(8, 2, 2, 256, 512);
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
        1 => proptest::collection::vec(any::<f64>(), len),
        1 => (0.0f64..40.0, proptest::collection::vec(0u8..3, len))
            .prop_map(|(base, steps)| steps.iter().map(|&s| base + f64::from(s)).collect()),
    ]
}

#[derive(Clone, Debug)]
struct Case {
    rows: Vec<Vec<f64>>,
    scales: Vec<f64>,
    m: usize,
    first: Vec<usize>,
    second: Vec<usize>,
    include: Vec<usize>,
    discords: bool,
    bits: u8,
}

fn case() -> impl Strategy<Value = Case> {
    (1usize..=4, 3usize..=14).prop_flat_map(|(d, n)| {
        (
            proptest::collection::vec(dimension(n), d),
            proptest::collection::vec(
                prop_oneof![3 => Just(1.0), 1 => Just(0.25), 1 => Just(2f64.powi(-30))],
                d,
            ),
            2usize..=n.min(6),
            proptest::collection::vec(any::<proptest::sample::Index>(), d),
            proptest::collection::vec(any::<proptest::sample::Index>(), d),
            proptest::collection::vec(0usize..d, 0..3),
            any::<bool>(),
            1u8..=8,
        )
            .prop_map(
                move |(rows, scales, window, first, second, include, discords, bits)| Case {
                    rows,
                    scales,
                    m: window,
                    first: first.iter().map(|i| i.index(n - window + 1)).collect(),
                    second: second.iter().map(|i| i.index(n - window + 1)).collect(),
                    include,
                    discords,
                    bits,
                },
            )
    })
}

proptest! {
    #![proptest_config(testkit::generators::seeded(400))]

    #[test]
    fn z_normalized_matches_the_reference(case in case()) {
        let set: Vec<(&[f64], Scale)> = case.rows.iter().zip(&case.scales).map(|(row, &s)| (&row[..], scale(s))).collect();
        let b = bits(case.bits);
        let got = mdl(&set, m(case.m), &case.first, &case.second, &options(&case.include, case.discords, b)).unwrap();
        let (sizes, subspaces) = naive_mdl(&set, m(case.m), &case.first, &case.second, &case.include, case.discords, b).unwrap();
        prop_assert_eq!(got.subspaces, subspaces);
        prop_assert_eq!(
            got.bit_sizes.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            sizes.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn non_normalized_matches_the_reference(case in case()) {
        let set: Vec<&[f64]> = case.rows.iter().map(Vec::as_slice).collect();
        let b = bits(case.bits);
        let got = maamp_mdl(&set, m(case.m), &case.first, &case.second, &options(&case.include, case.discords, b)).unwrap();
        let (sizes, subspaces) = naive_maamp_mdl(&set, m(case.m), &case.first, &case.second, &case.include, case.discords, b).unwrap();
        prop_assert_eq!(got.subspaces, subspaces);
        prop_assert_eq!(
            got.bit_sizes.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            sizes.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_size_is_never_nan_and_its_subspace_has_k_plus_one_dimensions(case in case()) {
        let set: Vec<&[f64]> = case.rows.iter().map(Vec::as_slice).collect();
        let b = bits(case.bits);
        let got = maamp_mdl(&set, m(case.m), &case.first, &case.second, &options(&case.include, case.discords, b)).unwrap();
        prop_assert_eq!(got.bit_sizes.len(), case.rows.len());
        for (k, (size, subspace)) in got.bit_sizes.iter().zip(&got.subspaces).enumerate() {
            prop_assert!(!size.is_nan());
            prop_assert_eq!(subspace.dimensions.len(), k + 1);
            prop_assert_eq!(subspace.complete, size.is_finite());
        }
    }

    #[test]
    fn the_distinct_count_is_at_most_the_table(case in case()) {
        let set: Vec<&[f64]> = case.rows.iter().map(Vec::as_slice).collect();
        let b = bits(case.bits);
        let found = Discretizer::linear(&set, m(case.m), b).unwrap();
        let d = case.rows.len();
        let _ = describe(&found, &case.first, &case.second, &options(&[], false, b)).unwrap();
        let pair = found.pair(case.first[0], case.second[0]);
        if pair.iter().all(Option::is_some) {
            let space = Subspace { dimensions: (0..d).collect(), complete: true };
            let distinct = distinct_differences(&pair, &space, u32::from(b.get()));
            prop_assert!(distinct >= 1 && distinct < (1 << (u32::from(b.get()) + 1)));
        }
    }
}
