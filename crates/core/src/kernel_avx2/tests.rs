extern crate std;

use super::{Kernel, detected, enabled, force_scalar, rank_converting, to_f64_four};
use crate::full_range::{
    self,
    tests::{any_magnitude, edges},
};
use crate::kernel_neon::tests::{RowInputs, floor_choices, neighbor_r_bias, scenario};
use crate::kernel_neon::{
    Conversion, GROUP_LANES, Lanes, RowScalars, SPAN_LANES, convert_body, scalar_groups,
    scalar_range,
};
use crate::route::{RouteForce, forced_route};
use proptest::prelude::*;
use std::{vec, vec::Vec};

/// Whether the AVX2 tests can run here; says so when they cannot.
fn avx2_here() -> bool {
    if detected() {
        return true;
    }
    std::eprintln!("skipped: this CPU has no AVX2, so the AVX2 kernel is not compared");
    false
}

/// [`rank_converting`]: the kernel with the conversion `row` holds, as the dispatcher picks it.
///
/// # Safety
/// As [`super::rank`]'s.
unsafe fn rank_as<const GROUPS: bool, const FLOORS: bool>(
    qt_m: &mut [i64],
    lanes: &Lanes,
    row: RowScalars,
    scratch: Option<&mut [f64]>,
    floors: &[f64],
    maxima: &mut [f64],
) -> (f64, u64) {
    // SAFETY: as the caller guarantees.
    unsafe { rank_converting::<GROUPS, FLOORS>(qt_m, lanes, row, scratch, floors, maxima) }
}

/// The conversions a row holding `conversion` may equally run with: its own and the full-range
/// one, which is exact on every term.
fn conversions(conversion: Conversion) -> Vec<Conversion> {
    let mut all = vec![conversion];
    if conversion != Conversion::FULL_RANGE {
        all.push(Conversion::FULL_RANGE);
    }
    all
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn lanes_of(inputs: &RowInputs) -> Lanes<'_> {
    Lanes {
        enter: &inputs.0,
        leave: &inputs.1,
        sum: &inputs.2,
        r: &inputs.3,
        bias: &inputs.4,
    }
}

/// A tick at an end of `i32`, near zero, or anywhere.
fn edge_tick() -> impl Strategy<Value = i32> {
    prop_oneof![
        Just(i32::MIN),
        Just(i32::MAX),
        Just(0),
        Just(1),
        Just(-1),
        any::<i32>()
    ]
}

/// A `qt` at an end of `i64`, near zero, or anywhere.
fn edge_qt() -> impl Strategy<Value = i64> {
    prop_oneof![
        Just(i64::MIN),
        Just(i64::MAX),
        Just(0),
        Just(-1),
        -1000i64..1000,
        any::<i64>()
    ]
}

fn edge_row_inputs(len: usize) -> impl Strategy<Value = RowInputs> {
    (
        prop::collection::vec(edge_tick(), len),
        prop::collection::vec(edge_tick(), len),
        prop::collection::vec(edge_tick(), len),
        prop::collection::vec(neighbor_r_bias(), len),
        (edge_tick(), edge_tick(), edge_tick(), 1u32..1_000_000),
    )
        .prop_map(|(enter, leave, sum, r_bias, (me, ml, si, ri))| {
            let (r, bias): (Vec<f64>, Vec<f64>) = r_bias.into_iter().unzip();
            let row = RowScalars {
                me,
                ml,
                si,
                ri: f64::from(ri) * 1e-6,
                conversion: Conversion::FULL_RANGE,
            };
            (enter, leave, sum, r, bias, row)
        })
}

/// Runs the AVX2 kernel and the scalar body on one row of every lane count `len`, comparing the
/// maximum, `qt`, `scratch`, the group maxima and the hit mask.
fn check_row(seed_qt: &[i64], inputs: &RowInputs, choices: &[u8]) -> Result<(), TestCaseError> {
    for conversion in conversions(inputs.5.conversion) {
        let mut inputs = inputs.clone();
        inputs.5.conversion = conversion;
        check_row_as(seed_qt, &inputs, choices)?;
    }
    Ok(())
}

/// [`check_row`] with the conversion `inputs` holds.
fn check_row_as(seed_qt: &[i64], inputs: &RowInputs, choices: &[u8]) -> Result<(), TestCaseError> {
    let len = seed_qt.len();
    let lanes = lanes_of(inputs);
    let row = inputs.5;

    let (mut qt_vec, mut qt_scalar) = (seed_qt.to_vec(), seed_qt.to_vec());
    let (mut out_vec, mut out_scalar) = (vec![0.0; len], vec![0.0; len]);
    // SAFETY: AVX2 detected by the caller; every slice holds `len` elements.
    let (max_vec, _) = unsafe {
        rank_as::<false, false>(&mut qt_vec, &lanes, row, Some(&mut out_vec), &[], &mut [])
    };
    let max_scalar = scalar_range(&mut qt_scalar, &lanes, row, Some(&mut out_scalar), 0, len);
    prop_assert_eq!(max_vec.to_bits(), max_scalar.to_bits());
    prop_assert_eq!(&qt_vec, &qt_scalar);
    prop_assert_eq!(bits(&out_vec), bits(&out_scalar));

    if len > SPAN_LANES {
        return Ok(());
    }
    assert_eq!(choices.len(), len, "one floor choice per lane");
    let floors: Vec<f64> = out_scalar
        .iter()
        .zip(choices)
        .map(|(&v, &c)| match c {
            0 => v,
            1 => v.next_up(),
            2 => f64::NEG_INFINITY,
            _ => f64::INFINITY,
        })
        .collect();
    let groups = len.div_ceil(GROUP_LANES);
    for with_floors in [false, true] {
        let floors = with_floors.then_some(floors.as_slice());
        let (mut qt_vec, mut qt_scalar) = (seed_qt.to_vec(), seed_qt.to_vec());
        let (mut out_vec, mut out_scalar) = (vec![0.0; len], vec![0.0; len]);
        let (mut max_vec, mut max_scalar) = (vec![0.0; groups], vec![0.0; groups]);
        // SAFETY: AVX2 detected by the caller; slices sized to `len` and `groups`, at most
        // `SPAN_GROUPS` of them.
        let hits_vec = unsafe {
            match floors {
                Some(f) => {
                    rank_as::<true, true>(
                        &mut qt_vec,
                        &lanes,
                        row,
                        Some(&mut out_vec),
                        f,
                        &mut max_vec,
                    )
                    .1
                }
                None => {
                    rank_as::<true, false>(
                        &mut qt_vec,
                        &lanes,
                        row,
                        Some(&mut out_vec),
                        &[],
                        &mut max_vec,
                    )
                    .1
                }
            }
        };
        let hits_scalar = scalar_groups(
            &mut qt_scalar,
            &lanes,
            row,
            &mut out_scalar,
            floors,
            &mut max_scalar,
        );
        prop_assert_eq!(hits_vec, hits_scalar);
        prop_assert_eq!(&qt_vec, &qt_scalar);
        prop_assert_eq!(bits(&out_vec), bits(&out_scalar));
        prop_assert_eq!(bits(&max_vec), bits(&max_scalar));
    }
    Ok(())
}

proptest! {
    #![proptest_config(testkit::generators::seeded(300))]

    /// Over several consecutive rows sharing one `qt`, the AVX2 kernel and the scalar body agree
    /// on the maximum, every `qt` and every ranking value, with and without `scratch`.
    #[test]
    fn avx2_rows_equal_the_scalar_body((use_scratch, seed_qt, rows) in scenario()) {
        if !avx2_here() {
            return Ok(());
        }
        let len = seed_qt.len();
        for conversion in conversions(Conversion::FULL_RANGE) {
        let (mut qt_vec, mut qt_scalar) = (seed_qt.clone(), seed_qt.clone());
        let mut out_vec = use_scratch.then(|| vec![0.0; len]);
        let mut out_scalar = use_scratch.then(|| vec![0.0; len]);
        for inputs in &rows {
            let lanes = lanes_of(inputs);
            let row = RowScalars { conversion, ..inputs.5 };
            // SAFETY: AVX2 detected above; every slice holds `len` elements.
            let (max_vec, _) = unsafe {
                rank_as::<false, false>(&mut qt_vec, &lanes, row, out_vec.as_deref_mut(), &[], &mut [])
            };
            let max_scalar = scalar_range(&mut qt_scalar, &lanes, inputs.5, out_scalar.as_deref_mut(), 0, len);
            prop_assert_eq!(max_vec.to_bits(), max_scalar.to_bits());
            prop_assert_eq!(&qt_vec, &qt_scalar);
            if let (Some(a), Some(b)) = (&out_vec, &out_scalar) {
                prop_assert_eq!(bits(a), bits(b));
            }
        }
        }
    }

    /// Group maxima, hit masks, scratch and `qt` at every lane count up to a full span, with
    /// ticks and `qt` at the ends of their ranges.
    #[test]
    fn avx2_groups_equal_the_scalar_body_at_the_edges(
        (seed_qt, inputs, choices) in (0usize..=SPAN_LANES).prop_flat_map(|len| {
            (prop::collection::vec(edge_qt(), len), edge_row_inputs(len), floor_choices(len))
        })
    ) {
        if !avx2_here() {
            return Ok(());
        }
        check_row(&seed_qt, &inputs, &choices)?;
    }

    /// Longer rows than a span, to the ungrouped kernel only.
    #[test]
    fn avx2_long_rows_equal_the_scalar_body(
        (seed_qt, inputs) in (SPAN_LANES + 1..SPAN_LANES + 300).prop_flat_map(|len| {
            (prop::collection::vec(edge_qt(), len), edge_row_inputs(len))
        })
    ) {
        if !avx2_here() {
            return Ok(());
        }
        check_row(&seed_qt, &inputs, &[])?;
    }

    /// The full-range conversion is `x as f64`, rounding included, on terms past `2⁵³`, near
    /// it and at both ends of `i64`.
    #[test]
    fn the_vector_conversion_rounds_like_as_f64(
        x in prop::array::uniform4(prop_oneof![
            any::<i64>(),
            any::<i64>().prop_map(|v| v >> 11),
            (-(1i64 << 54)..(1i64 << 54)),
            Just(i64::MIN),
            Just(i64::MAX),
        ])
    ) {
        if !avx2_here() {
            return Ok(());
        }
        prop_assert_eq!(
            to_f64_four::<{ convert_body::FULL_RANGE }>(x).map(f64::to_bits),
            x.map(|v| (v as f64).to_bits())
        );
    }
}

/// Every lane count through three groups, and a few at and past a full span, so each remainder width and each group boundary is
/// reached, on one fixed pattern with repeated values (ties) and one with constant lanes.
#[test]
fn every_lane_count_equals_the_scalar_body() {
    if !avx2_here() {
        return;
    }
    let past_a_span = [
        SPAN_LANES - 1,
        SPAN_LANES,
        SPAN_LANES + 1,
        SPAN_LANES + 3,
        2 * SPAN_LANES + 5,
    ];
    for len in (0..=3 * GROUP_LANES + 7).chain(past_a_span) {
        for pattern in 0..3i32 {
            let tick = |k: usize| match pattern {
                0 => 5,
                1 => (k % 4) as i32 - 2,
                _ => (k as i32).wrapping_mul(0x9E37_79B1_u32 as i32),
            };
            let col: Vec<i32> = (0..len).map(tick).collect();
            let inputs: RowInputs = (
                col.clone(),
                col.iter().map(|t| t.wrapping_add(3)).collect(),
                col,
                vec![1e-3; len],
                (0..len)
                    .map(|k| if k % 5 == 0 { 0.5 } else { 0.0 })
                    .collect(),
                RowScalars {
                    me: 7,
                    ml: -3,
                    si: 2,
                    ri: 0.25,
                    conversion: Conversion::FULL_RANGE,
                },
            );
            let seed: Vec<i64> = (0..len as i64).map(|k| k % 3 - 1).collect();
            let choices: Vec<u8> = (0..len).map(|k| (k % 4) as u8).collect();
            check_row(&seed, &inputs, &choices).unwrap();
        }
    }
}

/// A lane at the first and last position of a vector, of a group and of the tail, each the only
/// lane above its floor: the hit bit is its group's alone.
#[test]
fn a_hit_in_any_position_marks_its_group() {
    if !avx2_here() {
        return;
    }
    let len = 2 * GROUP_LANES + 5;
    let ticks = vec![1i32; len];
    let inputs: RowInputs = (
        ticks.clone(),
        ticks.clone(),
        ticks,
        vec![1.0; len],
        vec![0.0; len],
        RowScalars {
            me: 3,
            ml: 2,
            si: 1,
            ri: 1.0,
            conversion: Conversion::FULL_RANGE,
        },
    );
    for at in 0..len {
        let mut choices = vec![3u8; len];
        choices[at] = 0;
        check_row(&vec![0i64; len], &inputs, &choices).unwrap();
        let lanes = lanes_of(&inputs);
        let mut floors = vec![f64::INFINITY; len];
        floors[at] = f64::NEG_INFINITY;
        let mut maxima = vec![0.0; len.div_ceil(GROUP_LANES)];
        // SAFETY: AVX2 detected above; slices sized to `len` and its groups.
        let hits = unsafe {
            rank_as::<true, true>(
                &mut vec![0i64; len],
                &lanes,
                inputs.5,
                Some(&mut vec![0.0; len]),
                &floors,
                &mut maxima,
            )
            .1
        };
        assert_eq!(hits, 1 << (at / GROUP_LANES), "lane {at}");
    }
}

/// Ticks and `qt` small enough that every lane's cross term stays within `±2⁵¹` for a few rows.
fn in_range_row_inputs(len: usize) -> impl Strategy<Value = (Vec<i64>, RowInputs)> {
    let tick = || -(1i32 << 20)..1i32 << 20;
    (
        prop::collection::vec(-(1i64 << 49)..1i64 << 49, len),
        prop::collection::vec(tick(), len),
        prop::collection::vec(tick(), len),
        prop::collection::vec(tick(), len),
        prop::collection::vec(neighbor_r_bias(), len),
        (tick(), tick(), tick(), 1u32..1_000_000),
    )
        .prop_map(|(qt, enter, leave, sum, r_bias, (me, ml, si, ri))| {
            let (r, bias): (Vec<f64>, Vec<f64>) = r_bias.into_iter().unzip();
            let row = RowScalars {
                me,
                ml,
                si,
                ri: f64::from(ri) * 1e-6,
                conversion: Conversion::FAST,
            };
            (qt, (enter, leave, sum, r, bias, row))
        })
}

/// A row whose lanes' cross terms run through both ends of the fast conversion's range, zero
/// and its neighbors, each value at every position of a vector and in the scalar tail.
fn edge_terms_row(len: usize) -> (Vec<i64>, RowInputs) {
    const TERMS: [i64; 6] = [-(1 << 51), (1 << 51) - 1, 0, 1, -1, -(1 << 51) + 1];
    let (enter, leave, sum): (Vec<i32>, Vec<i32>, Vec<i32>) = (0..len)
        .map(|k| {
            let k = k as i32;
            (k % 13 - 6, k % 7 - 3, k % 5 - 2)
        })
        .collect();
    let row = RowScalars {
        me: 9,
        ml: -4,
        si: 3,
        ri: 1e-9,
        conversion: Conversion::FAST,
    };
    // `qt` chosen so the advanced lane's cross term is the term wanted there.
    let qt = (0..len)
        .map(|k| {
            let advance =
                i64::from(row.me) * i64::from(enter[k]) - i64::from(row.ml) * i64::from(leave[k]);
            TERMS[(k + k / 4) % TERMS.len()] + i64::from(row.si) * i64::from(sum[k]) - advance
        })
        .collect();
    (
        qt,
        (enter, leave, sum, vec![1e-9; len], vec![0.0; len], row),
    )
}

#[test]
fn the_fast_body_reaches_both_ends_of_its_range() {
    if !avx2_here() {
        return;
    }
    for len in (0..=2 * GROUP_LANES + 7).chain([SPAN_LANES, SPAN_LANES + 3]) {
        let (qt, inputs) = edge_terms_row(len);
        let choices: Vec<u8> = (0..len).map(|k| (k % 4) as u8).collect();
        check_row(&qt, &inputs, &choices).unwrap();
    }
}

proptest! {
    #![proptest_config(testkit::generators::seeded(200))]

    /// The fast conversion is `x as f64` on its whole range, ends included.
    #[test]
    fn the_fast_conversion_is_as_f64_inside_its_range(
        x in prop::array::uniform4(prop_oneof![
            -(1i64 << 51)..(1i64 << 51),
            -(1i64 << 20)..(1i64 << 20),
            Just(-(1i64 << 51)),
            Just((1i64 << 51) - 1),
            Just(0),
        ])
    ) {
        if !avx2_here() {
            return Ok(());
        }
        prop_assert_eq!(
            to_f64_four::<{ convert_body::FAST }>(x).map(f64::to_bits),
            x.map(|v| (v as f64).to_bits())
        );
    }

    /// With every cross term in range, the fast body equals the scalar body on every lane count
    /// up to past a span, over two consecutive rows.
    #[test]
    fn the_fast_body_equals_the_scalar_body(
        (seed_qt, first, second, choices) in (0usize..=SPAN_LANES + 9).prop_flat_map(|len| {
            (in_range_row_inputs(len), in_range_row_inputs(len), floor_choices(len))
                .prop_map(|((qt, first), (_, second), choices)| (qt, first, second, choices))
        })
    ) {
        if !avx2_here() {
            return Ok(());
        }
        let choices = if seed_qt.len() > SPAN_LANES { Vec::new() } else { choices };
        check_row(&seed_qt, &first, &choices)?;
        let mut advanced = seed_qt.clone();
        let len = advanced.len();
        scalar_range(&mut advanced, &lanes_of(&first), first.5, None, 0, len);
        check_row(&advanced, &second, &choices)?;
    }
}

proptest! {
    #![proptest_config(testkit::generators::seeded(5_000))]

    /// The full-range conversion is `x as f64` on every `i64`, and equals the scalar model it
    /// mirrors, in every lane.
    #[test]
    fn the_full_range_conversion_is_as_f64(x in prop::array::uniform4(any_magnitude())) {
        if !avx2_here() {
            return Ok(());
        }
        let got = to_f64_four::<{ convert_body::FULL_RANGE }>(x).map(f64::to_bits);
        prop_assert_eq!(got, x.map(|v| (v as f64).to_bits()));
        prop_assert_eq!(got, x.map(|v| full_range::model(v).to_bits()));
    }
}

/// The full-range conversion is `x as f64` on every edge value in every lane position.
#[test]
fn the_full_range_conversion_is_as_f64_on_every_edge() {
    if !avx2_here() {
        return;
    }
    let values = edges();
    for (k, &v) in values.iter().enumerate() {
        for lane in 0..4 {
            let mut x = [values[(k + 1) % values.len()]; 4];
            x[lane] = v;
            assert_eq!(
                to_f64_four::<{ convert_body::FULL_RANGE }>(x).map(f64::to_bits),
                x.map(|v| (v as f64).to_bits()),
                "{x:?}"
            );
        }
    }
}

/// A row whose lanes' cross terms are the full-range edge values, each at every position of a
/// vector and in the scalar tail, equals the scalar body under the full-range conversion.
#[test]
fn the_full_range_body_reaches_every_edge() {
    if !avx2_here() {
        return;
    }
    let terms = edges();
    for len in (0..=2 * GROUP_LANES + 7).chain([SPAN_LANES, SPAN_LANES + 3]) {
        for shift in 0..4 {
            let row = RowScalars {
                me: 0,
                ml: 0,
                si: 0,
                ri: 1e-12,
                conversion: Conversion::FULL_RANGE,
            };
            // Zero ticks: the cross term is `qt` itself.
            let qt: Vec<i64> = (0..len)
                .map(|k| terms[(k * 7 + shift) % terms.len()])
                .collect();
            let zeros = vec![0i32; len];
            let inputs: RowInputs = (
                zeros.clone(),
                zeros.clone(),
                zeros,
                vec![1e-9; len],
                vec![0.0; len],
                row,
            );
            let choices: Vec<u8> = (0..len).map(|k| (k % 4) as u8).collect();
            check_row(&qt, &inputs, &choices).unwrap();
        }
    }
}

/// Forcing the scalar body turns the AVX2 kernels off and the guard turns them back on.
#[test]
fn forcing_the_scalar_body_is_scoped() {
    let _serial = forced_route(RouteForce::default());
    let natural = enabled(Kernel::Narrow);
    assert_eq!(natural, detected());
    {
        let _scalar = force_scalar();
        assert!(!enabled(Kernel::Narrow));
        {
            let _nested = force_scalar();
            assert!(!enabled(Kernel::NonNormalized));
        }
        assert!(!enabled(Kernel::Narrow));
    }
    assert_eq!(enabled(Kernel::NonNormalized), natural);
}
