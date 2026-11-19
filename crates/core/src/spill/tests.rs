extern crate std;

use super::*;
use crate::index::SampleIndex;
use num_bigint::BigInt;
use std::string::ToString;
use std::vec;

#[path = "../../tests/common/spill_ref.rs"]
pub(crate) mod spill_ref;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

fn bits_of(entries: &[(u64, Entry)]) -> Vec<spill_ref::Bits> {
    entries
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

/// Copies of one small pattern at very different scales and levels, with one-tick
/// perturbations, between short filler walks: many candidates within the estimate band.
fn copies(seed: u64, m: usize) -> Vec<i64> {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let base: Vec<i64> = (0..m).map(|_| (rng.next() % 61) as i64 - 30).collect();
    let mut ticks = Vec::new();
    for _ in 0..5 {
        for _ in 0..m / 2 + 2 {
            ticks.push((rng.next() % 9) as i64);
        }
        let scale: i64 = [1, 1 << 20, 1 << 34, 1 << 42, 1 << 46][(rng.next() % 5) as usize];
        let scale = if rng.next().is_multiple_of(6) {
            -scale
        } else {
            scale
        };
        let level = (rng.next() % (1 << 40)) as i64;
        let mut copy: Vec<i64> = base.iter().map(|&x| x * scale + level).collect();
        match rng.next() % 3 {
            0 => {}
            1 => copy[(rng.next() % m as u64) as usize] += 1,
            _ => copy[(rng.next() % m as u64) as usize] -= 1,
        }
        ticks.extend(copy);
    }
    ticks
}

/// The pairs a left pass offers over windows flagged `flagged`.
fn left_cover(flagged: &Flagged, m: usize) -> Cover<'_> {
    let marks = flagged.marks();
    let windows = marks.flags.len() as isize;
    Cover {
        rows: marks,
        cols: marks,
        offsets: -(windows - 1)..=-(crate::exclusion::zone(m) as isize + 1),
    }
}

fn cold_everywhere<A: Arith>(ticks: &[i64], had: &[bool], m: usize) -> Vec<spill_ref::Bits> {
    let truth = Truth::<A>::new(ticks, had, m).unwrap();
    let windows = truth.rows.windows.len();
    let flagged = Flagged::new(vec![false; ticks.len()], vec![true; windows]).unwrap();
    let hot: Vec<(u64, Entry)> = (0..windows as u64).map(|row| (row, Entry::NONE)).collect();
    let hot_runs = HotRuns::new(&hot, windows).unwrap();
    let mut entries = Vec::new();
    truth
        .merge_band(
            &left_cover(&flagged, m),
            0..windows,
            1,
            &hot_runs,
            &mut entries,
            &mut Counts::default(),
        )
        .unwrap();
    bits_of(&entries)
}

#[test]
fn both_tiers_rank_every_window_like_the_reference() {
    for seed in 0..60u64 {
        let m = 4 + (seed % 11) as usize;
        let ticks = copies(seed, m);
        let mut had = vec![true; ticks.len()];
        if seed % 4 == 0 {
            had[ticks.len() / 2] = false;
        }
        let expected = spill_ref::left_profile(&ticks, &had, m);
        assert_eq!(
            cold_everywhere::<i128>(&ticks, &had, m),
            expected,
            "i128, seed {seed}"
        );
        assert_eq!(
            cold_everywhere::<Wide<4>>(&ticks, &had, m),
            expected,
            "256-bit, seed {seed}"
        );
    }
}

fn reference_cmp(a: (&BigInt, &BigInt), b: (&BigInt, &BigInt)) -> Ordering {
    // (cross, neighbor variance) of two candidates of one row.
    let sign = |x: &BigInt| match x.sign() {
        num_bigint::Sign::Minus => -1i8,
        num_bigint::Sign::NoSign => 0,
        num_bigint::Sign::Plus => 1,
    };
    if sign(a.0) != sign(b.0) {
        return sign(a.0).cmp(&sign(b.0));
    }
    if sign(a.0) == 0 {
        return Ordering::Equal;
    }
    let magnitude = (a.0 * a.0 * b.1).cmp(&(b.0 * b.0 * a.1));
    if sign(a.0) > 0 {
        magnitude
    } else {
        magnitude.reverse()
    }
}

fn limb(rng: &mut Rng, bits: u32) -> u128 {
    let raw = (u128::from(rng.next()) << 64 | u128::from(rng.next())) >> (128 - bits);
    match rng.next() % 5 {
        0 => 1 << (bits - 1),
        1 => (1 << (bits - 1)) - 1,
        _ => raw,
    }
}

#[test]
fn candidates_of_one_row_compare_by_exact_rationals_on_both_tiers() {
    let mut rng = Rng(0xC0FF_EE00_1234_5678);
    for _ in 0..4000 {
        let row_variance = limb(&mut rng, 120).max(1);
        let make = |rng: &mut Rng| {
            let n = limb(rng, 120) as i128 * if rng.next().is_multiple_of(2) { 1 } else { -1 };
            let n = if rng.next().is_multiple_of(9) { 0 } else { n };
            (n, limb(rng, 120).max(1) as i128)
        };
        let ((n_a, v_a), (n_b, v_b)) = if rng.next().is_multiple_of(5) {
            let first = make(&mut rng);
            // The same ρ² with N doubled and Vj quadrupled: an exact tie.
            ((first.0, first.1), (first.0 * 2, first.1 * 4))
        } else {
            (make(&mut rng), make(&mut rng))
        };
        let expected = reference_cmp(
            (&BigInt::from(n_a), &BigInt::from(v_a)),
            (&BigInt::from(n_b), &BigInt::from(v_b)),
        );
        let narrow = |n: i128, v: i128| Spilled::<i128>::Valid {
            n,
            row_variance: row_variance as i128,
            neighbor_variance: v,
        };
        assert_eq!(narrow(n_a, v_a).cmp_in_row(narrow(n_b, v_b)), expected);
        let wide = |n: i128, v: i128| Spilled::<Wide<4>>::Valid {
            n: Wide::from_i128(n),
            row_variance: Wide::from_i128(row_variance as i128),
            neighbor_variance: Wide::from_i128(v),
        };
        assert_eq!(wide(n_a, v_a).cmp_in_row(wide(n_b, v_b)), expected);
    }
}

#[test]
fn constants_rank_against_valid_pairs_by_the_fixed_ratios() {
    // ρ² = 1 (both constant) and 1/4 (one constant) against a valid pair's N²/(Vi·Vj).
    let row = 40i128;
    let valid = |n: i128, v: i128| Spilled::<i128>::Valid {
        n,
        row_variance: row,
        neighbor_variance: v,
    };
    // With Vi = 40 and Vj = 10, N = 10 gives ρ² = 100/400 = 1/4 and N = 20 gives 1.
    assert_eq!(
        valid(10, 10).cmp_in_row(Spilled::OneConstant),
        Ordering::Equal
    );
    assert_eq!(
        valid(11, 10).cmp_in_row(Spilled::OneConstant),
        Ordering::Greater
    );
    assert_eq!(
        valid(9, 10).cmp_in_row(Spilled::OneConstant),
        Ordering::Less
    );
    assert_eq!(
        valid(-11, 10).cmp_in_row(Spilled::OneConstant),
        Ordering::Less
    );
    assert_eq!(
        Spilled::<i128>::BothConstant.cmp_in_row(Spilled::OneConstant),
        Ordering::Greater
    );
    assert_eq!(
        valid(20, 10).cmp_in_row(Spilled::BothConstant),
        Ordering::Equal
    );
    assert_eq!(valid(20, 10).cmp_in_row(valid(-20, 10)), Ordering::Greater);
    assert_eq!(valid(0, 10).cmp_in_row(valid(0, 99)), Ordering::Equal);
    assert_eq!(valid(0, 10).cmp_in_row(valid(-1, 99)), Ordering::Greater);
}

#[test]
fn the_reported_distance_numerator_is_exact() {
    let mut rng = Rng(99);
    for _ in 0..2000 {
        let (vi, vj) = (limb(&mut rng, 100).max(2), limb(&mut rng, 100).max(2));
        let bound = (vi as f64 * vj as f64).sqrt() as u128;
        let n = (u128::from(rng.next()) % bound.max(1)) as i128;
        let n = if rng.next().is_multiple_of(2) { n } else { -n };
        let gap = BigInt::from(vi) * BigInt::from(vj) - BigInt::from(n) * BigInt::from(n);
        if gap.sign() == num_bigint::Sign::Minus {
            continue;
        }
        let parts = Spilled::<Wide<4>>::Valid {
            n: Wide::from_i128(n),
            row_variance: Wide::from_i128(vi as i128),
            neighbor_variance: Wide::from_i128(vj as i128),
        }
        .distance_parts()
        .unwrap();
        let expected: f64 = gap.to_string().parse().unwrap();
        assert_eq!(parts.numerator.to_bits(), expected.to_bits());
        assert_eq!(parts.row_inv, 1.0 / libm::sqrt(vi as f64));
        let narrow = Spilled::<i128>::Valid {
            n,
            row_variance: vi as i128,
            neighbor_variance: vj as i128,
        }
        .distance_parts()
        .unwrap();
        assert_eq!(narrow.numerator.to_bits(), expected.to_bits());
    }
    assert!(Spilled::<i128>::BothConstant.distance_parts().is_none());
    assert!(Spilled::<i128>::OneConstant.distance_parts().is_none());
}

fn flags_from(mask: u64, len: usize) -> Vec<bool> {
    (0..len).map(|i| mask >> (i % 64) & 1 == 1).collect()
}

/// How many pairs `(i, i + offset)` with `offset` in `offsets` have a flagged window, one by one.
fn touching_pairs(rows: &[bool], cols: &[bool], offsets: core::ops::RangeInclusive<isize>) -> u64 {
    let mut touching = 0;
    for i in 0..rows.len() as isize {
        for offset in offsets.clone() {
            let j = i + offset;
            if (0..cols.len() as isize).contains(&j) {
                touching += u64::from(rows[i as usize] || cols[j as usize]);
            }
        }
    }
    touching
}

#[test]
fn the_counted_walk_is_what_the_pass_walks_and_the_pairs_that_touch_a_spilled_window() {
    let mut rng = Rng(7);
    let ticks: Vec<i64> = (0..160).map(|i| (i * i % 17) as i64).collect();
    let had = vec![true; ticks.len()];
    for m in [2usize, 5, 8, 16, 33] {
        let truth = Truth::<i128>::new(&ticks, &had, m).unwrap();
        let windows = truth.rows.windows.len();
        let mut masks: Vec<Vec<bool>> = vec![
            vec![true; windows],
            vec![false; windows],
            (0..windows).map(|i| i == 0).collect(),
            (0..windows).map(|i| i + 1 == windows).collect(),
            (0..windows).map(|i| i % 2 == 0).collect(),
            (0..windows).map(|i| (10..10 + m).contains(&i)).collect(),
            (0..windows).map(|i| i < windows / 2).collect(),
        ];
        for _ in 0..30 {
            masks.push(flags_from(rng.next() & rng.next(), windows));
        }
        for spilled in masks {
            let flagged = Flagged::new(vec![false; ticks.len()], spilled.clone()).unwrap();
            let cover = left_cover(&flagged, m);
            let mut walked = Counts::default();
            truth.cold_band(&cover, 0..windows, 1, &mut walked).unwrap();
            assert_eq!(cover.count(0..windows), walked, "m {m}, mask {spilled:?}");
            let touching = touching_pairs(&spilled, &spilled, cover.offsets.clone());
            assert_eq!(walked.cells, touching, "m {m}, mask {spilled:?}");
        }
    }
}

#[test]
fn runs_are_the_maximal_blocks_of_true() {
    assert!(runs(&[]).unwrap().is_empty());
    assert!(runs(&[false, false]).unwrap().is_empty());
    assert_eq!(runs(&[true]).unwrap(), vec![(0, 0)]);
    assert_eq!(
        runs(&[true, true, false, true, false, false, true, true]).unwrap(),
        vec![(0, 1), (3, 3), (6, 7)]
    );
}

#[test]
fn the_lone_deviation_limit_is_the_last_deviation_inside_the_exact_domain() {
    assert_eq!(lone_deviation_limit(0), u64::MAX);
    assert_eq!(lone_deviation_limit(1), u64::MAX);
    for m in (2usize..200).chain([1000, 4096, 1 << 20, 1 << 32]) {
        let limit = u128::from(lone_deviation_limit(m));
        let others = m as u128 - 1;
        assert!(others * limit * limit < 1 << 63, "m {m}");
        assert!(others * (limit + 1) * (limit + 1) >= 1 << 63, "m {m}");
    }
    assert_eq!(lone_deviation_limit(2), 3_037_000_499);
}

#[test]
fn the_tier_follows_the_range_times_the_window() {
    let spanning = |range: i64| [0, range];
    for (range, m, expected) in [
        (1i64 << 32, 1usize << 31, Some(Width::Narrow128)),
        (1 << 32, (1 << 32) - 1, Some(Width::Narrow128)),
        (1 << 32, 1 << 32, Some(Width::Wide256)),
        (1 << 53, 2047, Some(Width::Narrow128)),
        (1 << 53, 2048, Some(Width::Wide256)),
        (1 << 53, (1 << 33) - 1, Some(Width::Wide256)),
        (1 << 53, 1 << 33, None),
        (1, 1 << 63, Some(Width::Narrow128)),
    ] {
        assert_eq!(
            width_for(&spanning(range), m),
            expected,
            "range {range}, m {m}"
        );
    }
    assert_eq!(width_for(&[7, 7, 7], 1 << 40), Some(Width::Narrow128));
    assert_eq!(width_for(&[i64::MIN, i64::MAX], 1), Some(Width::Narrow128));
    assert_eq!(width_for(&[i64::MIN / 2, i64::MAX / 2], 1 << 30), None);
}

#[test]
fn an_outlier_is_measured_from_the_newest_present_tick_kept_before_it() {
    let m = 5;
    let limit = lone_deviation_limit(m) as i64;
    let mut ticks = vec![100i64; 3 * SCREEN_TICKS];
    // A slow climb is never an outlier; a far tick on it is, and so is one just past the limit
    // from the tick before it; a gap neither moves the level nor counts.
    for (p, tick) in ticks.iter_mut().enumerate().skip(SCREEN_TICKS) {
        *tick = 100 + (p - SCREEN_TICKS) as i64 * (limit / 4);
    }
    ticks[SCREEN_TICKS + 7] += 2 * limit;
    ticks[2 * SCREEN_TICKS + 3] = ticks[2 * SCREEN_TICKS + 2] - limit - 1;
    ticks[5] += limit;
    let mut had = vec![true; ticks.len()];
    had[SCREEN_TICKS + 9] = false;
    ticks[SCREEN_TICKS + 9] += 10 * limit;
    let flags = outlier_ticks(&ticks, &had, m).unwrap();
    let flagged: Vec<usize> = (0..ticks.len()).filter(|&p| flags[p]).collect();
    assert_eq!(flagged, vec![SCREEN_TICKS + 7, 2 * SCREEN_TICKS + 3]);
}

#[test]
fn a_level_held_past_m_outliers_becomes_the_new_level() {
    let m = 4;
    let far = lone_deviation_limit(m) as i64 + 1;
    let mut ticks = vec![0i64; 20];
    for tick in &mut ticks[5..15] {
        *tick = far;
    }
    let mut had = vec![true; 20];
    had[7] = false;
    let flags = outlier_ticks(&ticks, &had, m).unwrap();
    let flagged: Vec<usize> = (0..ticks.len()).filter(|&p| flags[p]).collect();
    // Four present ticks at the new level are outliers (the gap does not count); the fifth is
    // the new level, and the return to zero is an outlier for four ticks in turn.
    assert_eq!(flagged, vec![5, 6, 8, 9, 15, 16, 17, 18]);
}

#[test]
fn a_jump_across_a_stretchs_first_tick_is_an_outlier() {
    let m = 6;
    let far = lone_deviation_limit(m) as i64 + 1;
    let mut ticks = vec![0i64; 3 * SCREEN_TICKS];
    let mut had = vec![true; ticks.len()];
    ticks[SCREEN_TICKS] = far;
    // A gap before the third stretch: its first present tick is measured across it.
    had[2 * SCREEN_TICKS - 1] = false;
    ticks[2 * SCREEN_TICKS - 1] = far;
    ticks[2 * SCREEN_TICKS] = -far;
    let flags = outlier_ticks(&ticks, &had, m).unwrap();
    let flagged: Vec<usize> = (0..ticks.len()).filter(|&p| flags[p]).collect();
    assert_eq!(flagged, vec![SCREEN_TICKS, 2 * SCREEN_TICKS]);
}

#[test]
fn a_stretch_is_measured_from_the_newest_present_tick_before_it() {
    let m = 6;
    let limit = lone_deviation_limit(m) as i64;
    // A climb by the limit over the first stretch, a second stretch half the limit above its
    // top, and a third stretch past the limit above that.
    let mut ticks: Vec<i64> = (0..SCREEN_TICKS as i64)
        .map(|p| p * limit / (SCREEN_TICKS as i64 - 1))
        .collect();
    ticks.extend([limit + limit / 2; SCREEN_TICKS]);
    ticks.extend([2 * limit + limit / 2 + 1; SCREEN_TICKS]);
    let had = vec![true; ticks.len()];
    let flags = outlier_ticks(&ticks, &had, m).unwrap();
    let flagged: Vec<usize> = (0..ticks.len()).filter(|&p| flags[p]).collect();
    let third = 2 * SCREEN_TICKS;
    assert_eq!(flagged, (third..third + m).collect::<Vec<_>>());
}

#[test]
fn a_series_with_no_present_tick_has_no_outliers_and_a_short_last_stretch_works() {
    let m = 3;
    let limit = lone_deviation_limit(m) as i64;
    let mut ticks = vec![0i64; SCREEN_TICKS + 10];
    ticks[SCREEN_TICKS + 4] = 3 * limit;
    ticks[3] = 9 * limit;
    let mut had = vec![true; ticks.len()];
    for flag in &mut had[..SCREEN_TICKS] {
        *flag = false;
    }
    let flags = outlier_ticks(&ticks, &had, m).unwrap();
    assert_eq!(flags.iter().filter(|&&f| f).count(), 1);
    assert!(flags[SCREEN_TICKS + 4]);
    assert!(outlier_ticks(&[], &[], m).unwrap().is_empty());
    assert!(
        outlier_ticks(&ticks, &vec![false; ticks.len()], m)
            .unwrap()
            .iter()
            .all(|&f| !f)
    );
}

#[test]
fn spilled_windows_are_those_with_every_sample_that_hold_an_outlier_or_are_too_wide() {
    let m = 4;
    let limit = lone_deviation_limit(m) as i64;
    let mut ticks = vec![10i64; 60];
    ticks[20] += limit + 1;
    ticks[40] += limit / 2;
    let mut had = vec![true; 60];
    had[22] = false;
    let survey = Survey::new(&ticks, &had, nz(m)).unwrap().unwrap();
    // Windows 17..=20 hold the outlier; 19..=22 hold the gap.
    let expected: Vec<bool> = (0..=56).map(|i| i == 17 || i == 18).collect();
    assert_eq!(survey.spilled(), expected.as_slice());
    assert_eq!(survey.outliers().iter().filter(|&&f| f).count(), 1);
}

#[test]
fn windows_whose_variance_passes_the_exact_domain_are_spilled_without_an_outlier() {
    // Alternating between two levels 0.9 of the limit apart: no tick is further from the median
    // than the limit, yet every window's variance numerator is past 2^63.
    let m = 4;
    let limit = lone_deviation_limit(m) as i64;
    let gap = limit / 10 * 9;
    let ticks: Vec<i64> = (0..40).map(|i| if i % 2 == 0 { 0 } else { gap }).collect();
    let had = vec![true; 40];
    let survey = Survey::new(&ticks, &had, nz(m)).unwrap().unwrap();
    assert!(survey.outliers().iter().all(|&f| !f));
    let variance = (m as u128) * 2 * (gap as u128).pow(2) - (gap as u128 * 2).pow(2);
    assert!(variance >= 1 << 63, "the test's own premise");
    assert!(survey.spilled().iter().all(|&s| s));
}

#[test]
fn a_merge_keeps_the_hot_entry_when_the_cold_candidate_is_farther_or_equal_and_later() {
    // One spilled window whose only partners are worse than the hot neighbor's.
    let m = 4;
    let ticks: Vec<i64> = vec![0, 1, 0, 1, 0, 1, 0, 1, 5, 5, 5, 5, 0, 1, 0, 1];
    let had = vec![true; ticks.len()];
    let windows = ticks.len() - m + 1;
    let mut spilled = vec![false; windows];
    spilled[8] = true;
    let hot = spill_ref::left_profile_where(&ticks, &had, m, &|i, j| !spilled[i] && !spilled[j]);
    let mut entries: Vec<(u64, Entry)> = hot
        .iter()
        .map(|&b| {
            let index = SampleIndex::new(b.3).unwrap_or(SampleIndex::NONE);
            (
                b.0,
                Entry::new(f64::from_bits(b.1), f64::from_bits(b.2), index).unwrap(),
            )
        })
        .collect();
    let survey = Survey::with_spilled(&ticks, &had, nz(m), spilled)
        .unwrap()
        .unwrap();
    survey.merge_left(&mut entries).unwrap();
    assert_eq!(bits_of(&entries), spill_ref::left_profile(&ticks, &had, m));
}

#[test]
fn a_capacity_error_surfaces_instead_of_aborting() {
    assert!(reserved_vec::<ColdWindow<Wide<4>>>(u128::from(u64::MAX)).is_err());
}

#[test]
fn filled_ticks_do_not_move_the_level() {
    let m = 5;
    let limit = lone_deviation_limit(m) as i64;
    let mut ticks = vec![100i64; SCREEN_TICKS];
    let mut had = vec![true; SCREEN_TICKS];
    for position in 1..600 {
        ticks[position] = 50 * limit;
        had[position] = false;
    }
    let flags = outlier_ticks(&ticks, &had, m).unwrap();
    assert!(flags.iter().all(|&flag| !flag));
}

#[test]
fn the_first_present_tick_is_the_level() {
    let m = 5;
    let big = lone_deviation_limit(m) as i64 + 1;
    let ticks = [big, 0, 0, big];
    let flags = outlier_ticks(&ticks, &[false, true, true, true], m).unwrap();
    assert_eq!(flags, [false, false, false, true]);
}

#[test]
fn a_window_holding_an_outlier_only_at_its_last_tick_is_spilled_though_not_too_wide() {
    let m = 5;
    let limit = lone_deviation_limit(m) as i64;
    let mut ticks = vec![0i64; 60];
    for tick in &mut ticks[20..23] {
        *tick = (limit + 1) / 2;
    }
    ticks[24] = limit + 1;
    let had = vec![true; 60];
    let survey = Survey::new(&ticks, &had, nz(m)).unwrap().unwrap();
    assert!(survey.outliers()[24]);
    assert_eq!(survey.outliers().iter().filter(|&&f| f).count(), 1);
    assert!(survey.spilled()[20], "window 20 ends on the outlier");
    assert!(survey.spilled()[24], "window 24 starts on it");
}

#[test]
fn a_variance_of_exactly_two_to_the_63_is_too_wide() {
    // Two ticks of 2^30 among six: k(m-k)·D² = 8·2^60.
    let m = 6;
    let mut ticks = vec![0i64; 60];
    ticks[30] = 1 << 30;
    ticks[31] = 1 << 30;
    let had = vec![true; 60];
    let survey = Survey::new(&ticks, &had, nz(m)).unwrap().unwrap();
    assert!(survey.outliers().iter().all(|&flag| !flag));
    let spilled: Vec<usize> = (0..survey.spilled().len())
        .filter(|&i| survey.spilled()[i])
        .collect();
    assert_eq!(spilled, vec![26, 27, 28, 29, 30]);
}

#[test]
#[cfg(not(any(loom, shuttle)))]
fn the_callers_constant_flags_rank_cold_windows_as_the_hot_passes_rank_them() {
    // Plateaus make flat windows, so flagged-flat, unflagged-flat and flagged-valid windows all
    // occur; every window cold must equal the hot passes under the same flags.
    let _serial = crate::route::forced_route(crate::route::RouteForce::default());
    let scale = crate::quantize::Scale::new(1.0).unwrap();
    let mut rng = Rng(0x5EED);
    for m in [3usize, 4, 6, 9] {
        for round in 0..12u64 {
            let values: Vec<f64> = (0..90)
                .map(|i| {
                    let plateau = (i / 7) % 3 == 0;
                    if plateau {
                        40.0
                    } else {
                        (rng.next() % 7) as f64 + (i / 30) as f64 * 3.0
                    }
                })
                .collect();
            let windows = values.len() - m + 1;
            let flags: Vec<bool> = (0..windows)
                .map(|_| match round % 3 {
                    0 => rng.next().is_multiple_of(4),
                    1 => !rng.next().is_multiple_of(4),
                    _ => false,
                })
                .collect();
            let series = crate::stats::Series::with_constant_windows(&values, nz(m), &flags)
                .expect("one flag per window");
            let hot = crate::batch::left_profile(series, nz(m), scale);
            let ticks: Vec<i64> = values.iter().map(|&x| x as i64).collect();
            let had = vec![true; ticks.len()];
            let survey = Survey::with_spilled(&ticks, &had, nz(m), vec![true; windows])
                .unwrap()
                .unwrap()
                .flagged(Some(&flags), None);
            let mut cold: Vec<(u64, Entry)> =
                (0..windows as u64).map(|row| (row, Entry::NONE)).collect();
            survey.merge_left(&mut cold).unwrap();
            assert_eq!(bits_of(&cold), bits_of(&hot), "m {m}, round {round}");
        }
    }
}

/// The outlier rule written out directly: a present tick lying more than the limit from the
/// newest earlier present tick that is not an outlier, unless `m` present ticks since that one are.
fn outliers_by_the_rule(ticks: &[i64], had: &[bool], m: usize) -> Vec<bool> {
    let limit = lone_deviation_limit(m);
    let mut flags = vec![false; ticks.len()];
    for p in 0..ticks.len() {
        let Some(kept) = (0..p).rev().find(|&q| had[q] && !flags[q]) else {
            continue;
        };
        let run = (kept + 1..p).filter(|&q| had[q]).count();
        flags[p] = had[p] && ticks[p].abs_diff(ticks[kept]) > limit && run < m;
    }
    flags
}

/// Series of every shape the screen decides on: walks, jumps around the limit, plateaus, gaps.
fn screened_series(seed: u64) -> (Vec<i64>, Vec<bool>, usize) {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let m = 2 + (rng.next() % 12) as usize;
    let len = 40 + (rng.next() % 2600) as usize;
    let limit = lone_deviation_limit(m) as i64;
    let mut level = 1_000i64;
    let mut ticks = Vec::with_capacity(len);
    for i in 0..len {
        level += (rng.next() % 9) as i64 - 4;
        let jump = match rng.next() % 97 {
            _ if seed % 5 == 1 => 0,
            0 => limit + 1,
            1 => limit,
            2 => -limit - 3,
            3 => limit / 2,
            4 => limit / 5 * 3,
            5 => -(limit / 5 * 3),
            _ => 0,
        };
        let plateau = seed.is_multiple_of(3) && (i / 61) % 5 == 4;
        ticks.push(level + jump + if plateau { 3 * limit } else { 0 });
    }
    let had = (0..len)
        .map(|_| !seed.is_multiple_of(4) || !rng.next().is_multiple_of(17))
        .collect();
    (ticks, had, m)
}

#[test]
fn a_stretch_spanning_no_more_than_the_limit_is_skipped_without_changing_which_ticks_are_outliers()
{
    for seed in 0..120u64 {
        let (ticks, had, m) = screened_series(seed);
        assert_eq!(
            outlier_ticks(&ticks, &had, m).unwrap(),
            outliers_by_the_rule(&ticks, &had, m),
            "seed {seed}"
        );
    }
}

#[test]
fn nothing_to_rank_is_true_only_where_a_survey_finds_nothing() {
    let mut said_nothing = 0;
    for seed in 0..120u64 {
        let (ticks, had, m) = screened_series(seed);
        let mut stats = crate::stats::WindowStats::new(nz(m));
        let kinds: Vec<_> = ticks
            .iter()
            .zip(&had)
            .filter_map(|(&tick, &present)| stats.push(tick, present))
            .map(|window| window.kind)
            .collect();
        let unranked = |row: usize| kinds[row] == crate::stats::WindowKind::NonFinite;
        let any_unranked = (0..kinds.len()).any(unranked);
        let nothing = nothing_to_rank(&ticks, &had, m, any_unranked, unranked).unwrap();
        let Some(survey) = Survey::new(&ticks, &had, nz(m)).unwrap() else {
            continue;
        };
        let spilled = survey.spilled().iter().any(|&flag| flag);
        if nothing {
            said_nothing += 1;
            assert!(!spilled, "seed {seed}");
        }
        if spilled {
            assert!(!nothing, "seed {seed}");
        }
    }
    assert!(
        said_nothing > 10,
        "the screen decided {said_nothing} series"
    );
}

/// `survey`'s merged lists over `bands` of rows each (the last may be shorter), concatenated.
fn merged_in_bands(survey: &Survey, windows: usize, band: usize, k: usize) -> Vec<spill_ref::Bits> {
    let hot: Vec<(u64, Entry)> = (0..windows as u64).map(|row| (row, Entry::NONE)).collect();
    let runs = HotRuns::new(&hot, windows).unwrap();
    let mut out = Vec::new();
    let mut counts = Counts::default();
    for start in (0..windows).step_by(band) {
        survey
            .merge_rows(
                Reach::Left,
                start..(start + band).min(windows),
                nz(k),
                &runs,
                &mut out,
                &mut counts,
            )
            .unwrap();
    }
    bits_of(&out)
}

#[test]
fn rows_cut_into_bands_of_any_height_get_the_lists_of_the_whole_pass() {
    for seed in 0..24u64 {
        let m = 4 + (seed % 7) as usize;
        let ticks = copies(seed, m);
        let had = vec![true; ticks.len()];
        let windows = ticks.len() - m + 1;
        let mut rng = Rng(seed + 1);
        let spilled: Vec<bool> = match seed % 3 {
            0 => (0..windows).map(|_| rng.next().is_multiple_of(5)).collect(),
            1 => (0..windows).map(|i| (i / 9) % 3 == 1).collect(),
            _ => (0..windows).map(|i| i % (m + 3) == 0).collect(),
        };
        let survey = Survey::with_spilled(&ticks, &had, nz(m), spilled)
            .unwrap()
            .unwrap();
        for k in [1, 3] {
            let whole = merged_in_bands(&survey, windows, windows, k);
            for band in [1, 2, 3, 7, 16] {
                assert_eq!(
                    merged_in_bands(&survey, windows, band, k),
                    whole,
                    "seed {seed}, k {k}, band {band}"
                );
            }
        }
    }
}

/// The pairs of a pass of `reach` over `flagged`, and the neighbor flags of a join.
fn cover_over<'a>(
    rows: &'a Flagged,
    cols: Option<&'a Flagged>,
    reach: Reach,
    m: usize,
) -> Cover<'a> {
    let marks = rows.marks();
    let cols = cols.map_or(marks, Flagged::marks);
    let exclusion = crate::exclusion::zone(m) as isize;
    let (wr, wc) = (marks.flags.len() as isize, cols.flags.len() as isize);
    let offsets = match reach {
        Reach::Left => -(wr - 1)..=-(exclusion + 1),
        Reach::Right => exclusion + 1..=wr - 1,
        Reach::Join => -(wr - 1)..=wc - 1,
    };
    Cover {
        rows: marks,
        cols,
        offsets,
    }
}

#[test]
fn every_reach_walks_exactly_the_pairs_that_touch_a_spilled_window() {
    let mut rng = Rng(11);
    let ticks: Vec<i64> = (0..140).map(|i| (i * i % 19) as i64).collect();
    let other: Vec<i64> = (0..110).map(|i| (i * 7 % 13) as i64).collect();
    let (had, other_had) = (vec![true; ticks.len()], vec![true; other.len()]);
    for m in [2usize, 5, 9, 17] {
        let own = Truth::<i128>::new(&ticks, &had, m).unwrap();
        let join = Truth::<i128>::join((&ticks, &had), (&other, &other_had), m).unwrap();
        let (wr, wc) = (own.rows.windows.len(), join.cols().windows.len());
        for round in 0..24u64 {
            let rows = flags_from(rng.next() & rng.next() & rng.next(), wr);
            let cols = flags_from(rng.next() & rng.next(), wc);
            let (rows, cols) = match round {
                0 => (vec![true; wr], vec![true; wc]),
                1 => (vec![false; wr], vec![false; wc]),
                2 => (vec![false; wr], cols),
                3 => (rows, vec![false; wc]),
                _ => (rows, cols),
            };
            let row_flags = Flagged::new(vec![false; ticks.len()], rows.clone()).unwrap();
            let col_flags = Flagged::new(vec![false; other.len()], cols.clone()).unwrap();
            for reach in [Reach::Left, Reach::Right] {
                let cover = cover_over(&row_flags, None, reach, m);
                let mut walked = Counts::default();
                own.cold_band(&cover, 0..wr, 1, &mut walked).unwrap();
                assert_eq!(cover.count(0..wr), walked, "{reach:?}, m {m}");
                let touching = touching_pairs(&rows, &rows, cover.offsets.clone());
                assert_eq!(walked.cells, touching, "{reach:?}, m {m}, round {round}");
            }
            let cover = cover_over(&row_flags, Some(&col_flags), Reach::Join, m);
            let mut walked = Counts::default();
            join.cold_band(&cover, 0..wr, 1, &mut walked).unwrap();
            assert_eq!(cover.count(0..wr), walked, "join, m {m}");
            let touching = touching_pairs(&rows, &cols, cover.offsets.clone());
            assert_eq!(walked.cells, touching, "join, m {m}, round {round}");
        }
    }
}

/// `survey`'s merged lists over bands of `band` rows each, concatenated, with every hot row empty.
fn merged_over(
    survey: &Survey,
    reach: Reach,
    rows: usize,
    band: usize,
    k: usize,
) -> Vec<spill_ref::Bits> {
    let hot: Vec<(u64, Entry)> = (0..rows as u64).map(|row| (row, Entry::NONE)).collect();
    let runs = HotRuns::new(&hot, rows).unwrap();
    let mut out = Vec::new();
    let mut counts = Counts::default();
    for start in (0..rows).step_by(band) {
        survey
            .merge_rows(
                reach,
                start..(start + band).min(rows),
                nz(k),
                &runs,
                &mut out,
                &mut counts,
            )
            .unwrap();
    }
    bits_of(&out)
}

#[test]
fn right_and_join_rows_cut_into_bands_get_the_lists_of_the_whole_pass() {
    for seed in 0..16u64 {
        let m = 4 + (seed % 5) as usize;
        let ticks = copies(seed, m);
        let other = copies(seed + 50, m);
        let (had, other_had) = (vec![true; ticks.len()], vec![true; other.len()]);
        let windows = ticks.len() - m + 1;
        let mut rng = Rng(seed + 3);
        let spilled: Vec<bool> = (0..windows).map(|_| rng.next().is_multiple_of(4)).collect();
        let own = Survey::with_spilled(&ticks, &had, nz(m), spilled)
            .unwrap()
            .unwrap();
        let join = Survey::join((&ticks, &had), (&other, &other_had), nz(m))
            .unwrap()
            .unwrap();
        assert!(
            join.any_spilled(),
            "the copies at scale reach past the domain"
        );
        for k in [1, 3] {
            let whole = merged_over(&own, Reach::Right, windows, windows, k);
            let whole_join = merged_over(&join, Reach::Join, windows, windows, k);
            for band in [1, 2, 5, 16] {
                assert_eq!(merged_over(&own, Reach::Right, windows, band, k), whole);
                assert_eq!(
                    merged_over(&join, Reach::Join, windows, band, k),
                    whole_join,
                    "seed {seed}, k {k}, band {band}"
                );
            }
        }
    }
}

#[test]
fn a_join_takes_the_wider_tier_of_its_two_series() {
    let m = 3000;
    let narrow: Vec<i64> = (0..3060).map(|i| i64::from(i % 5)).collect();
    let wide: Vec<i64> = (0..3040)
        .map(|i| if i % 2 == 0 { 4_000_000_000_000_000 } else { -4_000_000_000_000_000 } + i64::from(i % 3))
        .collect();
    let (narrow_had, wide_had) = (vec![true; narrow.len()], vec![true; wide.len()]);
    assert_eq!(width_for(&narrow, m), Some(Width::Narrow128));
    assert_eq!(width_for(&wide, m), Some(Width::Wide256));
    for (rows, cols) in [
        ((&narrow, &narrow_had), (&wide, &wide_had)),
        ((&wide, &wide_had), (&narrow, &narrow_had)),
    ] {
        let join = Survey::join(
            (rows.0.as_slice(), rows.1.as_slice()),
            (cols.0.as_slice(), cols.1.as_slice()),
            nz(m),
        )
        .unwrap()
        .unwrap();
        assert_eq!(join.width(), Width::Wide256);
    }
}

#[test]
fn held_correlations_order_as_the_hot_comparison_does() {
    use crate::nearest::{ExactCorrelation, Held, Sharing, cmp_exact, variance_product};
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let draw = |rng: &mut Rng| -> (ExactCorrelation, Spilled<Wide<4>>) {
        match rng.next() % 8 {
            0 => (ExactCorrelation::BothConstant, Spilled::BothConstant),
            1 => (ExactCorrelation::OneConstant, Spilled::OneConstant),
            _ => {
                let (row, neighbor) = (1 + rng.next() % (1 << 50), 1 + rng.next() % (1 << 50));
                let bound = (u128::from(row) * u128::from(neighbor)).isqrt() as u64;
                let magnitude = rng.next() % (bound + 1);
                let n = if rng.next().is_multiple_of(2) {
                    i128::from(magnitude)
                } else {
                    -i128::from(magnitude)
                };
                (
                    ExactCorrelation::Valid {
                        sign: n.signum() as i8,
                        cross_squared: u128::from(magnitude) * u128::from(magnitude),
                        row_variance: row,
                        neighbor_variance: neighbor,
                        denominator: variance_product(row, neighbor),
                    },
                    Spilled::Valid {
                        n: Wide::from_i128(n),
                        row_variance: Wide::from_u64(row),
                        neighbor_variance: Wide::from_u64(neighbor),
                    },
                )
            }
        }
    };
    for _ in 0..2000 {
        let ((hot_a, wide_a), (hot_b, wide_b)) = (draw(&mut rng), draw(&mut rng));
        let expected = cmp_exact(hot_a, hot_b, Sharing::Nothing);
        assert_eq!(cmp_held(Held::Hot(hot_a), Held::Cold(wide_b)), expected);
        assert_eq!(cmp_held(Held::Cold(wide_a), Held::Hot(hot_b)), expected);
        assert_eq!(cmp_held(Held::Cold(wide_a), Held::Cold(wide_b)), expected);
    }
}
