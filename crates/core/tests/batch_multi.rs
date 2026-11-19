#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! The z-normalized multi-dimensional batch profile equals the exact reference on bits at every
//! thread count and tile shape: on inputs of exact ties (constants, periods, scaled copies of a
//! dimension), gaps at either end, `include` and `discords`; with one dimension it is the
//! two-sided profile; it equals STUMPY's `mstump` on random walks within a tolerance.

#[path = "common/multi.rs"]
mod multi;

use ladde::batch::multi::{mstump, mstump_report, mstump_report_tiled};
use ladde::batch::two_sided_profile;
use ladde::multidim::{MultiError, NonFinitePolicy};
use ladde::naive::naive_multi_profile;
use ladde::quantize::Scale;
use multi::*;
use proptest::prelude::*;

fn assert_equal(input: &Input, scales: &[Scale]) {
    let samples: Vec<_> = input
        .dimensions
        .iter()
        .zip(scales)
        .map(|(series, &scale)| ticks(series, scale))
        .collect();
    let refs: Vec<&[Option<i64>]> = samples.iter().map(Vec::as_slice).collect();
    let expected = naive_multi_profile(
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
            let (profile, _) = mstump_report_tiled(
                &dimensions,
                nz(input.m),
                scales,
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
        let profile = mstump(
            &dimensions,
            nz(input.m),
            scales,
            &input.include,
            input.discords,
            nz(threads),
        )
        .unwrap();
        assert_eq!(bits(&profile), want, "default tiling, threads {threads}");
    }
}

fn unit_scales(input: &Input) -> Vec<Scale> {
    vec![unit(); input.dimensions.len()]
}

proptest! {
    #![proptest_config(testkit::generators::seeded(40))]

    #[test]
    fn inputs_equal_the_reference(input in input(8)) {
        assert_equal(&input, &unit_scales(&input));
    }
}

#[test]
fn tie_inputs_equal_the_reference() {
    for input in tie_inputs() {
        assert_equal(&input, &unit_scales(&input));
    }
}

#[test]
fn more_than_twenty_tied_dimensions_equal_the_reference() {
    for count in [21, 24] {
        for (include, discords) in [(vec![], false), (vec![20, 3], true), (vec![0], false)] {
            let input = tied_copies(count, true, include, discords);
            assert_equal(&input, &unit_scales(&input));
        }
    }
}

#[test]
fn each_dimension_takes_its_own_scale() {
    let mut input = tie_inputs().remove(7);
    for series in &mut input.dimensions[1..] {
        for x in series.iter_mut() {
            *x *= 0.25;
        }
    }
    let quarter = Scale::new(0.25).unwrap();
    let scales = [unit(), quarter, quarter, quarter];
    assert_equal(&input, &scales[..input.dimensions.len()]);
}

#[test]
fn one_dimension_is_the_two_sided_profile() {
    for kind in 0..8u8 {
        let series = kind_series(kind, 70, 5 + u64::from(kind));
        let profile = mstump(&[&series], nz(5), &[unit()], &[], false, nz(2)).unwrap();
        let sided = two_sided_profile(&series, nz(5), unit(), nz(2));
        assert_eq!(profile.distance[0].len(), sided.len());
        for (row, (_, sides)) in sided.iter().enumerate() {
            assert_eq!(
                profile.index[0][row],
                sides.nearest.index(),
                "kind {kind} row {row}"
            );
            assert_eq!(
                profile.distance[0][row].to_f64().to_bits(),
                sides.nearest.squared_distance().sqrt().to_bits(),
                "kind {kind} row {row}"
            );
        }
    }
}

#[test]
fn too_short_series_and_bad_arguments() {
    let series = vec![1.0, 2.0, 4.0];
    let empty = mstump(&[&series, &series], nz(5), &[unit(); 2], &[], false, nz(1)).unwrap();
    assert_eq!(empty.distance, vec![vec![], vec![]]);
    let (profile, report) = mstump_report(&[&series], nz(3), &[unit()], &[], false, nz(1)).unwrap();
    assert_eq!(profile.distance[0].len(), 1);
    assert!(profile.distance[0][0].is_missing());
    assert!(profile.index[0][0].is_none());
    assert_eq!(report.tiles, 1);

    let one = [1.0, 2.0, 3.0, 5.0];
    let two = [1.0, 2.0, 3.0];
    let run = |dimensions: &[&[f64]], scales: &[Scale], include: &[usize]| {
        mstump(dimensions, nz(3), scales, include, false, nz(1)).map(|_| ())
    };
    assert_eq!(run(&[], &[], &[]), Err(MultiError::NoDimensions));
    assert_eq!(
        run(&[&one, &two], &[unit(); 2], &[]),
        Err(MultiError::UnequalLengths)
    );
    assert_eq!(run(&[&one], &[unit(); 2], &[]), Err(MultiError::ScaleCount));
    assert_eq!(
        run(&[&one], &[unit()], &[1]),
        Err(MultiError::IncludeOutOfRange)
    );
}

/// The golden random walks (`scripts/golden_mstump.py`): indices equal to the exact ones computed
/// independently of this library, distances within a tolerance of STUMPY's `mstump`.
#[test]
fn equals_the_exact_indices_and_stumpys_distances_on_random_walks() {
    const SCALE: f64 = 1.0 / (1u64 << 26) as f64;
    let names = testkit::golden::multi_names();
    assert!(names.len() >= 8);
    for name in names {
        let fixture = testkit::golden::load_multi(&name);
        let dimensions = slices(&fixture.series);
        let scales = vec![Scale::new(SCALE).unwrap(); dimensions.len()];
        let profile = mstump(
            &dimensions,
            nz(fixture.m),
            &scales,
            &fixture.include,
            fixture.discords,
            nz(3),
        )
        .unwrap();
        let mut finite = 0;
        for k in 0..dimensions.len() {
            for (row, (&expected, &exact)) in fixture.stumpy_p[k]
                .iter()
                .zip(&fixture.exact_i[k])
                .enumerate()
            {
                assert_eq!(
                    profile.index[k][row].get(),
                    (exact != u64::MAX).then_some(exact),
                    "{name} {k} {row}"
                );
                let value = profile.distance[k][row].to_f64();
                if expected.is_finite() {
                    finite += 1;
                    assert!(
                        (value - expected).abs() <= 1e-6 + 1e-6 * expected,
                        "{name} {k} {row}"
                    );
                } else {
                    assert!(value.is_infinite(), "{name} {k} {row}");
                }
            }
        }
        assert!(finite > 0, "{name}");
    }
}
