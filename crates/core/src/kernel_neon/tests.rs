extern crate std;

use super::{
    Conversion, GROUP_LANES, RowScalars, SPAN_LANES, rank_row, rank_row_groups,
    rank_row_groups_scalar, rank_row_scalar,
};
use proptest::prelude::*;
use std::{vec, vec::Vec};

/// A neighbor's own `(r, bias)` pair, matching one of the three kinds the contract gives a
/// fixed bias for: Valid (a positive `1/√V`, bias `0`), Constant (`r = 0`, bias `0.5`), or
/// non-finite (`r = 0`, bias `−∞`).
pub(crate) fn neighbor_r_bias() -> impl Strategy<Value = (f64, f64)> {
    prop_oneof![
        3 => (1u32..1_000_000).prop_map(|v| (f64::from(v) * 1e-6, 0.0_f64)),
        1 => Just((0.0, 0.5)),
        1 => Just((0.0, f64::NEG_INFINITY)),
    ]
}

pub(crate) fn row_scalars() -> impl Strategy<Value = RowScalars> {
    (any::<i32>(), any::<i32>(), any::<i32>(), 1u32..1_000_000).prop_map(|(me, ml, si, ri)| {
        RowScalars {
            me,
            ml,
            si,
            ri: f64::from(ri) * 1e-6,
            conversion: Conversion::FULL_RANGE,
        }
    })
}

/// One row's own `enter`, `leave`, `sum`, `r`, `bias` and [`RowScalars`], the same shape
/// [`rank_row`] and [`rank_row_scalar`] take.
pub(crate) type RowInputs = (Vec<i32>, Vec<i32>, Vec<i32>, Vec<f64>, Vec<f64>, RowScalars);

/// One row's full set of inputs at a given lag count: ticks and sums span the full `i32`
/// range, `r`/`bias` are drawn per lag from `neighbor_r_bias`.
pub(crate) fn row_inputs(len: usize) -> impl Strategy<Value = RowInputs> {
    (
        prop::collection::vec(any::<i32>(), len),
        prop::collection::vec(any::<i32>(), len),
        prop::collection::vec(any::<i32>(), len),
        prop::collection::vec(neighbor_r_bias(), len),
        row_scalars(),
    )
        .prop_map(|(enter, leave, sum, r_bias, row)| {
            let (r, bias): (Vec<f64>, Vec<f64>) = r_bias.into_iter().unzip();
            (enter, leave, sum, r, bias, row)
        })
}

/// `len` (0..2000) and `n_rows` (1..=5) chosen first, then everything their shape
/// depends on — the seed `qt_m`, `n_rows` rows of inputs, and whether to run in scratch
/// mode — drawn from strategies built with that `len`.
pub(crate) fn scenario() -> impl Strategy<Value = (bool, Vec<i64>, Vec<RowInputs>)> {
    (0usize..2000, 1usize..=5).prop_flat_map(|(len, n_rows)| {
        (
            any::<bool>(),
            prop::collection::vec(any::<i64>(), len),
            prop::collection::vec(row_inputs(len), n_rows),
        )
    })
}

proptest! {
    #![proptest_config(testkit::generators::seeded(300))]

    /// `rank_row` and `rank_row_scalar` agree bit for bit on the returned maximum, every
    /// lag's `qt_m`, and `scratch` (when present), over many consecutive rows sharing one
    /// `qt_m` buffer — so the comparison also covers whatever each row's own arithmetic
    /// carried into the next one, not just a single call from a fresh buffer.
    #[test]
    fn neon_matches_scalar_bit_for_bit((use_scratch, seed_qt, rows) in scenario()) {
        let len = seed_qt.len();
        let mut qt_neon = seed_qt.clone();
        let mut qt_scalar = seed_qt.clone();
        let mut scratch_neon: Option<Vec<f64>> = use_scratch.then(|| vec![0.0; len]);
        let mut scratch_scalar: Option<Vec<f64>> = use_scratch.then(|| vec![0.0; len]);

        for (enter, leave, sum, r, bias, row) in &rows {
            let max_neon = rank_row(
                &mut qt_neon, enter, leave, sum, r, bias, *row,
                scratch_neon.as_deref_mut(),
            );
            let max_scalar = rank_row_scalar(
                &mut qt_scalar, enter, leave, sum, r, bias, *row,
                scratch_scalar.as_deref_mut(),
            );

            prop_assert_eq!(max_neon.to_bits(), max_scalar.to_bits());
            prop_assert_eq!(&qt_neon, &qt_scalar);
            if let (Some(a), Some(b)) = (&scratch_neon, &scratch_scalar) {
                let a_bits: Vec<u64> = a.iter().map(|v| v.to_bits()).collect();
                let b_bits: Vec<u64> = b.iter().map(|v| v.to_bits()).collect();
                prop_assert_eq!(a_bits, b_bits);
            }
        }
    }
}

/// A floor per lane relative to the value the lane will take: the value itself (a hit), the
/// next value up (a miss), open, or closed.
pub(crate) fn floor_choices(len: usize) -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(0u8..4, len)
}

proptest! {
    #![proptest_config(testkit::generators::seeded(200))]

    /// `rank_row_groups` and its scalar oracle agree bit for bit on every group's maximum, the
    /// hit mask, `qt_m` and `scratch`, with and without floors; each maximum is its group's
    /// largest value and each hit bit says whether some lane of the group reaches its floor.
    #[test]
    fn groups_match_the_scalar_oracle_bit_for_bit(
        (seed_qt, row, choices) in (0usize..=SPAN_LANES).prop_flat_map(|len| {
            (prop::collection::vec(any::<i64>(), len), row_inputs(len), floor_choices(len))
        })
    ) {
        let (enter, leave, sum, r, bias, scalars) = &row;
        let len = seed_qt.len();
        let mut probe_qt = seed_qt.clone();
        let mut values = vec![0.0; len];
        rank_row_scalar(&mut probe_qt, enter, leave, sum, r, bias, *scalars, Some(&mut values));
        let floors: Vec<f64> = values
            .iter()
            .zip(&choices)
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
            let (mut qt_neon, mut qt_scalar) = (seed_qt.clone(), seed_qt.clone());
            let (mut s_neon, mut s_scalar) = (vec![0.0; len], vec![0.0; len]);
            let (mut max_neon, mut max_scalar) = (vec![0.0; groups], vec![0.0; groups]);
            let hits_neon = rank_row_groups(
                &mut qt_neon, enter, leave, sum, r, bias, *scalars, &mut s_neon, floors,
                &mut max_neon,
            );
            let hits_scalar = rank_row_groups_scalar(
                &mut qt_scalar, enter, leave, sum, r, bias, *scalars, &mut s_scalar, floors,
                &mut max_scalar,
            );
            prop_assert_eq!(hits_neon, hits_scalar);
            prop_assert_eq!(&qt_neon, &probe_qt);
            prop_assert_eq!(&qt_scalar, &probe_qt);
            let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            prop_assert_eq!(bits(&s_neon), bits(&values));
            prop_assert_eq!(bits(&s_scalar), bits(&values));
            prop_assert_eq!(bits(&max_neon), bits(&max_scalar));
            for (g, &max) in max_neon.iter().enumerate() {
                let lanes = g * GROUP_LANES..((g + 1) * GROUP_LANES).min(len);
                let largest = values[lanes.clone()].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                prop_assert_eq!(max, largest);
                let hit = floors.is_some_and(|f| lanes.clone().any(|j| values[j] >= f[j]));
                prop_assert_eq!(hits_neon >> g & 1 == 1, hit);
            }
        }
    }
}

/// One lane exactly on its floor, every other floor closed: the hit is that lane's group alone,
/// wherever the lane sits in a vector half or the scalar tail.
#[test]
fn a_single_lane_on_its_floor_marks_its_group() {
    let len = 2 * GROUP_LANES + 5;
    let ticks: Vec<i32> = (0..len as i32).map(|t| t % 17 - 8).collect();
    let r = vec![1e-3; len];
    let bias = vec![0.0; len];
    let row = RowScalars {
        me: 3,
        ml: 2,
        si: 1,
        ri: 1e-2,
        conversion: Conversion::FULL_RANGE,
    };
    let mut values = vec![0.0; len];
    rank_row_scalar(
        &mut vec![0i64; len],
        &ticks,
        &ticks,
        &ticks,
        &r,
        &bias,
        row,
        Some(&mut values),
    );
    for at in 0..len {
        let mut floors = vec![f64::INFINITY; len];
        floors[at] = values[at];
        let mut maxima = vec![0.0; len.div_ceil(GROUP_LANES)];
        let hits = rank_row_groups(
            &mut vec![0i64; len],
            &ticks,
            &ticks,
            &ticks,
            &r,
            &bias,
            row,
            &mut vec![0.0; len],
            Some(&floors),
            &mut maxima,
        );
        assert_eq!(hits, 1 << (at / GROUP_LANES), "lane {at}");
    }
}

/// A span at its limit reports all its groups, the last hit bit included.
#[test]
fn a_full_span_reports_every_group() {
    let len = SPAN_LANES;
    let mut qt = vec![0i64; len];
    let ticks = vec![0i32; len];
    let r = vec![1.0; len];
    let bias = vec![0.0; len];
    let mut floors = vec![f64::INFINITY; len];
    floors[len - 1] = 0.0;
    let mut scratch = vec![0.0; len];
    let mut maxima = vec![f64::NAN; len / GROUP_LANES];
    let row = RowScalars {
        me: 1,
        ml: 1,
        si: 1,
        ri: 1.0,
        conversion: Conversion::FULL_RANGE,
    };
    let hits = rank_row_groups(
        &mut qt,
        &ticks,
        &ticks,
        &ticks,
        &r,
        &bias,
        row,
        &mut scratch,
        Some(&floors),
        &mut maxima,
    );
    assert_eq!(hits, 1 << (SPAN_LANES / GROUP_LANES - 1));
    assert!(maxima.iter().all(|&m| m == 0.0));
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "no ranking lane may be NaN")]
fn rank_row_rejects_a_nan_scale_in_debug() {
    let mut qt_m = [0i64; 3];
    let ones = [0i32; 3];
    let r = [1.0, f64::NAN, 1.0];
    let bias = [0.0f64; 3];
    let row = RowScalars {
        me: 1,
        ml: 1,
        si: 1,
        ri: 1.0,
        conversion: Conversion::FULL_RANGE,
    };
    rank_row(&mut qt_m, &ones, &ones, &ones, &r, &bias, row, None);
}

#[test]
#[should_panic(expected = "must have one")]
fn rank_row_rejects_a_mismatched_slice_length() {
    let mut qt_m = [0i64; 3];
    let enter = [0i32; 3];
    let leave = [0i32; 3];
    let sum = [0i32; 2]; // one short
    let r = [0.0f64; 3];
    let bias = [0.0f64; 3];
    let row = RowScalars {
        me: 1,
        ml: 1,
        si: 1,
        ri: 1.0,
        conversion: Conversion::FULL_RANGE,
    };
    rank_row(&mut qt_m, &enter, &leave, &sum, &r, &bias, row, None);
}

/// Two Valid lanes straddling the 8-lane vector boundary (indices 7 and 8 of 16),
/// both with integer cross term 0, every other lane non-finite: the boundary lanes
/// round to `+0.0`, not `−0.0`, and NEON and the scalar body agree bit for bit, in
/// both scratch modes.
#[test]
fn neon_and_scalar_round_a_boundary_zero_cross_term_to_positive_zero() {
    let len = 16;
    let row = RowScalars {
        me: 0,
        ml: 0,
        si: 0,
        ri: 1.0,
        conversion: Conversion::FULL_RANGE,
    };
    let enter = vec![0i32; len];
    let leave = vec![0i32; len];
    let sum = vec![0i32; len];
    let mut r = vec![0.0f64; len];
    let mut bias = vec![f64::NEG_INFINITY; len];
    for j in [7, 8] {
        r[j] = 1.0;
        bias[j] = 0.0;
    }
    let positive_zero = 0.0_f64.to_bits();

    for use_scratch in [false, true] {
        let mut qt_neon = vec![0i64; len];
        let mut qt_scalar = qt_neon.clone();
        let mut scratch_neon: Option<Vec<f64>> = use_scratch.then(|| vec![1.0; len]);
        let mut scratch_scalar: Option<Vec<f64>> = use_scratch.then(|| vec![1.0; len]);

        let max_neon = rank_row(
            &mut qt_neon,
            &enter,
            &leave,
            &sum,
            &r,
            &bias,
            row,
            scratch_neon.as_deref_mut(),
        );
        let max_scalar = rank_row_scalar(
            &mut qt_scalar,
            &enter,
            &leave,
            &sum,
            &r,
            &bias,
            row,
            scratch_scalar.as_deref_mut(),
        );

        assert_eq!(max_neon.to_bits(), positive_zero);
        assert_eq!(max_scalar.to_bits(), positive_zero);

        if let (Some(a), Some(b)) = (&scratch_neon, &scratch_scalar) {
            assert_eq!(a[7].to_bits(), positive_zero);
            assert_eq!(b[7].to_bits(), positive_zero);
            assert_eq!(a[8].to_bits(), positive_zero);
            assert_eq!(b[8].to_bits(), positive_zero);
        }
    }
}

/// A later lane tying the running maximum numerically (`-0.0` against `+0.0`) must not
/// replace it: the first lane to reach the maximum keeps it, bit for bit.
#[test]
fn a_later_tying_lane_does_not_replace_the_maximum() {
    let row = RowScalars {
        me: 0,
        ml: 0,
        si: 0,
        ri: 1e-200,
        conversion: Conversion::FULL_RANGE,
    };
    let zero_i32 = [0i32; 2];
    let mut qt = [0i64, -1];
    // Lane 0 ranks `+0.0`; lane 1 underflows `(-1 * 1e-200) * 1e-200` to `-0.0`.
    let r = [1.0, 1e-200];
    let bias = [0.0, -0.0];
    let max = rank_row_scalar(
        &mut qt, &zero_i32, &zero_i32, &zero_i32, &r, &bias, row, None,
    );
    assert_eq!(max.to_bits(), 0.0_f64.to_bits());
}

// ---------- rank_row: pinned bits reject a reassociated product ----------

/// The same crafted triple [`crate::estimate`]'s own test uses: `(n · ri) · rj` and
/// `n · (ri · rj)` round to different bits, so pinning `rank_row`'s output against the
/// left-to-right value, lane by lane, catches a `lane_value` (scalar or NEON) that
/// reassociated the product — something a bit-for-bit NEON-against-scalar comparison alone
/// would miss if both copies reassociated the same way.
#[test]
fn rank_row_pins_the_left_to_right_product_on_a_few_lanes() {
    let n: i64 = -68_661_172_291_171;
    let ri = 1.0 / libm::sqrt(834_911_684_991_698_f64);
    let rj = 1.0 / libm::sqrt(275_720_901_775_960_f64);
    let left_to_right = (n as f64) * ri * rj;
    let reassociated = (n as f64) * (ri * rj);
    assert_ne!(
        left_to_right.to_bits(),
        reassociated.to_bits(),
        "these inputs must actually distinguish the two operation orders"
    );

    let row = RowScalars {
        me: 0,
        ml: 0,
        si: 0,
        ri,
        conversion: Conversion::FULL_RANGE,
    };
    // Spans both halves of the first vector group, the second group's own first half, and
    // the scalar tail past `vectored` (len 20, so 16..20): every lane shape the NEON loop
    // and its remainder can take.
    let pinned = [0usize, 7, 8, 19];
    let len = 20;
    let enter = vec![0i32; len];
    let leave = vec![0i32; len];
    let sum = vec![0i32; len];
    let mut r = vec![0.0f64; len];
    let mut bias = vec![f64::NEG_INFINITY; len];
    let mut qt_neon = vec![0i64; len];
    for &idx in &pinned {
        qt_neon[idx] = n;
        r[idx] = rj;
        bias[idx] = 0.0;
    }
    let mut qt_scalar = qt_neon.clone();
    let mut scratch_neon = vec![0.0f64; len];
    let mut scratch_scalar = vec![0.0f64; len];

    let max_neon = rank_row(
        &mut qt_neon,
        &enter,
        &leave,
        &sum,
        &r,
        &bias,
        row,
        Some(&mut scratch_neon),
    );
    let max_scalar = rank_row_scalar(
        &mut qt_scalar,
        &enter,
        &leave,
        &sum,
        &r,
        &bias,
        row,
        Some(&mut scratch_scalar),
    );

    assert_eq!(max_neon.to_bits(), left_to_right.to_bits());
    assert_eq!(max_scalar.to_bits(), left_to_right.to_bits());
    for &idx in &pinned {
        assert_eq!(scratch_neon[idx].to_bits(), left_to_right.to_bits());
        assert_eq!(scratch_scalar[idx].to_bits(), left_to_right.to_bits());
        assert_ne!(scratch_neon[idx].to_bits(), reassociated.to_bits());
        assert_ne!(scratch_scalar[idx].to_bits(), reassociated.to_bits());
    }
}
