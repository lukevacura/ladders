//! The multi-dimensional reference against each fixture's exact indices, computed independently
//! of this library, and against STUMPY's own `mstump` beside them. An index must equal the exact
//! one. STUMPY's kernel is JIT-compiled with fastmath and sums in its own order, so its distances
//! agree within a tolerance, never bitwise; its indices equal the exact ones on these random walks,
//! where no two means nearly tie, and any fixture where they do not must be listed here.

use ladde::multidim::NonFinitePolicy;
use ladde::naive::naive_multi_profile;
use ladde::quantize::{Scale, to_ticks};
use testkit::golden::{MultiFixture, load_multi, multi_names};

/// The scale of `golden_stumpy`: quantizing moves a sample by at most 7.5e-9.
const SCALE: f64 = 1.0 / (1u64 << 26) as f64;
const ABSOLUTE_TOLERANCE: f64 = 1e-6;
const RELATIVE_TOLERANCE: f64 = 1e-6;

fn quantized(fixture: &MultiFixture) -> Vec<Vec<Option<i64>>> {
    let scale = Scale::new(SCALE).expect("SCALE is positive and finite");
    fixture
        .series
        .iter()
        .map(|series| series.iter().map(|&x| to_ticks(x, scale)).collect())
        .collect()
}

#[test]
fn reference_equals_the_exact_indices_and_stumpys_distances() {
    let names = multi_names();
    assert!(names.len() >= 8);
    for name in names {
        let fixture = load_multi(&name);
        let ticks = quantized(&fixture);
        let slices: Vec<&[Option<i64>]> = ticks.iter().map(Vec::as_slice).collect();
        let ours = naive_multi_profile(
            &slices,
            fixture.m,
            &fixture.include,
            fixture.discords,
            NonFinitePolicy::Missing,
        )
        .expect("fixture input is valid");
        let mut finite = 0;
        for k in 0..ticks.len() {
            for (row, &exact) in fixture.exact_i[k].iter().enumerate() {
                assert_eq!(
                    ours.index[k][row].get(),
                    (exact != u64::MAX).then_some(exact),
                    "{name} k {k} row {row}"
                );
                let value = ours.distance[k][row].to_f64();
                let expected = fixture.stumpy_p[k][row];
                if expected.is_finite() {
                    finite += 1;
                    assert!(
                        (value - expected).abs()
                            <= ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * expected,
                        "{name} k {k} row {row}: {value} vs {expected}"
                    );
                } else {
                    assert!(value.is_infinite(), "{name} k {k} row {row}");
                }
            }
        }
        assert!(finite > 0, "{name}");
        // STUMPY's indices beside the exact ones: no known difference in any fixture.
        assert_eq!(
            fixture.stumpy_i, fixture.exact_i,
            "{name}: STUMPY differs from the exact indices"
        );
    }
}
