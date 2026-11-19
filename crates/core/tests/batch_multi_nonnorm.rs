#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! The non-normalized multi-dimensional batch profile equals the exact reference on bits at every
//! thread count and tile shape: on inputs of exact ties (constants, periods, scaled copies of a
//! dimension), gaps at either end, `include` and `discords`, and runs at levels around the `u64`
//! bound; with one dimension it is the non-normalized two-sided profile.

#[path = "common/multi.rs"]
mod multi;

use ladde::batch::multi::{maamp, maamp_report, maamp_report_tiled};
use ladde::batch::nonnorm::two_sided_profile;
use ladde::multidim::{MultiError, NonFinitePolicy};
use ladde::naive::naive_multi_profile_nonnorm;
use ladde::quantize::Scale;
use multi::*;
use proptest::prelude::*;

fn assert_equal(input: &Input, scale: Scale) {
    let samples: Vec<_> = input
        .dimensions
        .iter()
        .map(|series| ticks(series, scale))
        .collect();
    let refs: Vec<&[Option<i64>]> = samples.iter().map(Vec::as_slice).collect();
    let expected = naive_multi_profile_nonnorm(
        &refs,
        input.m,
        &input.include,
        input.discords,
        NonFinitePolicy::Missing,
    )
    .unwrap();
    let want = bits(&expected);
    let dimensions = slices(&input.dimensions);
    let n = input.dimensions[0].len();
    for threads in THREADS {
        for (band, chunk, tile) in shapes(n) {
            let (profile, _) = maamp_report_tiled(
                &dimensions,
                nz(input.m),
                scale,
                &input.include,
                input.discords,
                nz(threads),
                nz(band),
                nz(chunk),
                nz(tile),
            )
            .unwrap();
            assert_eq!(
                bits(&profile),
                want,
                "{input:?}, threads {threads}, shape {:?}",
                (band, chunk, tile)
            );
            assert_eq!(profile, expected, "the means, exactly");
        }
        let profile = maamp(
            &dimensions,
            nz(input.m),
            scale,
            &input.include,
            input.discords,
            nz(threads),
        )
        .unwrap();
        assert_eq!(bits(&profile), want, "default tiling, threads {threads}");
    }
}

proptest! {
    #![proptest_config(testkit::generators::seeded(40))]

    #[test]
    fn inputs_equal_the_reference(input in input(9)) {
        assert_equal(&input, unit());
    }
}

#[test]
fn tie_inputs_equal_the_reference() {
    for input in tie_inputs() {
        assert_equal(&input, unit());
    }
}

#[test]
fn more_than_twenty_tied_dimensions_equal_the_reference() {
    for count in [21, 24] {
        for (include, discords) in [(vec![], false), (vec![20, 3], true), (vec![0], false)] {
            assert_equal(&tied_copies(count, false, include, discords), unit());
        }
    }
}

#[test]
fn levels_around_the_u64_bound_equal_the_reference() {
    for seed in 0..6u64 {
        let dimensions: Vec<Vec<f64>> = (0..3u64)
            .map(|a| kind_series(8, 50, seed + a))
            .chain([kind_series(1, 50, 0)])
            .collect();
        for m in [3, 6] {
            assert_equal(
                &Input {
                    dimensions: dimensions.clone(),
                    m,
                    include: vec![],
                    discords: seed % 2 == 1,
                },
                unit(),
            );
        }
    }
}

#[test]
fn a_coarser_scale_is_read_back_exactly() {
    let mut input = tie_inputs().remove(5);
    for series in &mut input.dimensions {
        for x in series.iter_mut() {
            *x *= 0.5;
        }
    }
    assert_equal(&input, Scale::new(0.5).unwrap());
}

#[test]
fn one_dimension_is_the_two_sided_profile() {
    for kind in 0..9u8 {
        let series = kind_series(kind, 70, 5 + u64::from(kind));
        let profile = maamp(&[&series], nz(5), unit(), &[], false, nz(2)).unwrap();
        let sided = two_sided_profile(&series, nz(5), unit(), nz(2));
        assert_eq!(profile.distance[0].len(), sided.len());
        for (row, (_, sides)) in sided.iter().enumerate() {
            assert_eq!(
                profile.index[0][row],
                sides.nearest.index(),
                "kind {kind} row {row}"
            );
            let want = sides.nearest.index().get().map_or(f64::INFINITY, |_| {
                (sides.nearest.squared_ticks() as f64).sqrt()
            });
            assert_eq!(
                profile.distance[0][row].to_f64().to_bits(),
                want.to_bits(),
                "kind {kind} row {row}"
            );
        }
    }
}

#[test]
fn too_short_series_and_bad_arguments() {
    let series = vec![1.0, 2.0, 4.0];
    let empty = maamp(&[&series, &series], nz(5), unit(), &[], false, nz(1)).unwrap();
    assert_eq!(empty.distance, vec![vec![], vec![]]);
    let (profile, report) = maamp_report(&[&series], nz(3), unit(), &[], false, nz(1)).unwrap();
    assert!(profile.distance[0][0].is_missing() && profile.index[0][0].is_none());
    assert_eq!((report.tiles, report.wide_tiles), (1, 0));

    let one = [1.0, 2.0, 3.0, 5.0];
    let two = [1.0, 2.0, 3.0];
    let run = |dimensions: &[&[f64]], include: &[usize]| {
        maamp(dimensions, nz(3), unit(), include, false, nz(1)).map(|_| ())
    };
    assert_eq!(run(&[], &[]), Err(MultiError::NoDimensions));
    assert_eq!(run(&[&one, &two], &[]), Err(MultiError::UnequalLengths));
    assert_eq!(run(&[&one], &[1]), Err(MultiError::IncludeOutOfRange));
}
