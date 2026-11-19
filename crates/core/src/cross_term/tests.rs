extern crate std;

use super::{CatchUp, CrossTerms, Offsets, Side, narrow, wide};
use crate::batch::break_even;
use crate::kernel::Metric;
use crate::narrow::{self as narrowing, Anchor};
use crate::route::{CatchUpForce, RouteForce, forced_route};
use crate::znorm::{TickWindow, ZNorm};
use core::num::NonZeroUsize;
use proptest::prelude::*;
use std::vec;
use std::vec::Vec;

fn nz(m: usize) -> NonZeroUsize {
    NonZeroUsize::new(m).unwrap()
}

/// `Σ aₖ·bₖ` in i128, then reduced mod 2⁶⁴: the definition every form must equal.
fn exact_wrapped(a: &[i64], b: &[i64]) -> i64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| i128::from(x) * i128::from(y))
        .fold(0i128, |s, p| s.wrapping_add(p)) as i64
}

/// The largest offset the narrow arithmetic holds at `m`.
fn bound(m: usize) -> i64 {
    i64::from(i32::MAX) / m as i64
}

fn step_of(ticks: &[i64], at: usize, m: usize) -> TickWindow {
    let stats = crate::stats::Window {
        sum: 0,
        kind: crate::stats::WindowKind::NonFinite,
    };
    TickWindow {
        entering: ticks[at + m - 1],
        leaving: ticks[at - 1],
        stats,
    }
}

/// The wide recurrence from pair `(i, j)` through `run` steps.
fn advanced_wide(a: &[i64], b: &[i64], (i, j): (usize, usize), run: usize, m: usize) -> i64 {
    let mut qt = wide(&a[i..][..m], &b[j..][..m]);
    for k in 1..=run {
        qt = ZNorm::advance(qt, &step_of(a, i + k, m), &step_of(b, j + k, m));
    }
    qt
}

/// The narrow recurrence (`m·QT` on offsets) from pair `(i, j)` through `run` steps.
fn advanced_narrow(a: &[i32], b: &[i32], (i, j): (usize, usize), run: usize, m: usize) -> i64 {
    let mut qt = narrow(&a[i..][..m], &b[j..][..m], m);
    let scaled = |x: i32| (i64::from(x) * m as i64) as i32;
    for k in 1..=run {
        let (r, c) = (i + k, j + k);
        qt = narrowing::advance(
            qt,
            scaled(a[r + m - 1]),
            scaled(a[r - 1]),
            b[c + m - 1],
            b[c - 1],
        );
    }
    qt
}

fn wide_ticks(len: std::ops::Range<usize>) -> impl Strategy<Value = Vec<i64>> {
    prop::collection::vec(
        prop_oneof![any::<i64>(), -1000i64..1000, Just(i64::MAX), Just(i64::MIN),],
        len,
    )
}

fn narrow_offsets(m: usize, len: usize) -> impl Strategy<Value = Vec<i32>> {
    let b = bound(m) as i32;
    prop::collection::vec(prop_oneof![-b..=b, Just(b), Just(-b)], len)
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]

    #[test]
    fn wide_is_the_wrapped_exact_sum_at_any_tick(a in wide_ticks(0..40), b in wide_ticks(0..40)) {
        let len = a.len().min(b.len());
        prop_assert_eq!(wide(&a[..len], &b[..len]), exact_wrapped(&a[..len], &b[..len]));
    }

    #[test]
    fn narrow_is_m_times_the_exact_sum((m, a, b) in (1usize..300).prop_flat_map(|m| {
        (Just(m), narrow_offsets(m, m), narrow_offsets(m, m))
    })) {
        let wide_a: Vec<i64> = a.iter().map(|&x| i64::from(x)).collect();
        let wide_b: Vec<i64> = b.iter().map(|&x| i64::from(x)).collect();
        let exact = (m as i64).wrapping_mul(exact_wrapped(&wide_a, &wide_b));
        prop_assert_eq!(narrow(&a, &b, m), exact);
    }

    #[test]
    fn raw_ticks_read_as_offsets_equal_the_narrowed_copy(
        (m, a, b, at, bt) in (1usize..64).prop_flat_map(|m| {
            (Just(m), narrow_offsets(m, m), narrow_offsets(m, m), -(1i64 << 50)..(1i64 << 50), any::<i32>())
        })
    ) {
        let (anchor_a, anchor_b) = (Anchor { tick: at }, Anchor { tick: i64::from(bt) });
        let raw_a: Vec<i64> = a.iter().map(|&x| at + i64::from(x)).collect();
        let raw_b: Vec<i64> = b.iter().map(|&x| i64::from(bt) + i64::from(x)).collect();
        let side_a = Offsets { ticks: &raw_a, anchor: anchor_a, m: nz(m) };
        let side_b = Offsets { ticks: &raw_b, anchor: anchor_b, m: nz(m) };
        prop_assert_eq!(side_a.cross(0, side_b, 0, m, m), narrow(&a, &b, m));
    }

    #[test]
    fn a_resume_equals_the_wide_recurrence_at_any_tick(
        (m, a, b, i, j, run) in (1usize..12).prop_flat_map(|m| {
            (Just(m), wide_ticks(m + 1..60), wide_ticks(m + 1..60))
        }).prop_flat_map(|(m, a, b)| {
            let (ra, rb) = (a.len() - m, b.len() - m);
            (Just(m), Just(a), Just(b), 0..=ra, 0..=rb)
        }).prop_flat_map(|(m, a, b, i, j)| {
            let most = (a.len() - m - i).min(b.len() - m - j);
            (Just(m), Just(a), Just(b), Just(i), Just(j), 0..=most)
        })
    ) {
        let pairs = CrossTerms::new(&a[..], 0, &b[..], 0, m);
        let direct = pairs.resume((i, j), run);
        prop_assert_eq!(direct, Some(advanced_wide(&a, &b, (i, j), run, m)));
        prop_assert_eq!(direct, Some(pairs.at(i + run, j + run)));
        let most = (a.len() - m - i).min(b.len() - m - j);
        prop_assert_eq!(pairs.resume((i, j), most + 1), None);
    }

    #[test]
    fn a_resume_equals_the_narrow_recurrence_at_the_offset_bound(
        (m, a, b, i, j, run) in (1usize..40).prop_flat_map(|m| {
            (Just(m), 2 * m + 1..3 * m + 20).prop_flat_map(move |(m, len)| {
                (Just(m), narrow_offsets(m, len), narrow_offsets(m, len))
            })
        }).prop_flat_map(|(m, a, b)| {
            let last = a.len() - m;
            (Just(m), Just(a), Just(b), 0..=last, 0..=last)
        }).prop_flat_map(|(m, a, b, i, j)| {
            let most = (a.len() - m - i).min(b.len() - m - j);
            (Just(m), Just(a), Just(b), Just(i), Just(j), 0..=most)
        })
    ) {
        let pairs = CrossTerms::new(&a[..], 0, &b[..], 0, m);
        prop_assert_eq!(pairs.resume((i, j), run), Some(advanced_narrow(&a, &b, (i, j), run, m)));
    }
}

#[test]
fn a_resume_reaches_the_first_and_last_step_and_refuses_past_the_end() {
    let m = 4;
    let ticks: Vec<i64> = (0..20).map(|k| (k * 7919) % 101 - 50).collect();
    let pairs = CrossTerms::of(&ticks[..], m);
    let last = ticks.len() - m;
    // The diagonal of lag 3 runs from pair (3, 0) to (last, last − 3).
    let first = (3, 0);
    assert_eq!(pairs.resume(first, 0), Some(pairs.at(3, 0)));
    for run in [1, m, last - 3] {
        assert_eq!(
            pairs.resume(first, run),
            Some(advanced_wide(&ticks, &ticks, first, run, m)),
            "run {run}"
        );
    }
    assert_eq!(pairs.resume(first, last - 3 + 1), None);
    assert_eq!(pairs.resume(first, ticks.len()), None);
    assert_eq!(pairs.resume(first, usize::MAX), None);
    assert_eq!(pairs.resume((usize::MAX, 0), 1), None);
    assert_eq!(pairs.resume((last + 1, 0), 0), None);
}

#[test]
fn a_resume_from_before_the_sides_is_refused() {
    let ticks: Vec<i64> = (0..30).collect();
    let pairs = CrossTerms::new(&ticks[5..], 5, &ticks[2..], 2, 3);
    assert_eq!(pairs.resume((4, 9), 1), None);
    assert_eq!(pairs.resume((9, 1), 1), None);
    assert_eq!(
        pairs.resume((5, 2), 2),
        Some(wide(&ticks[7..10], &ticks[4..7]))
    );
}

#[test]
fn sides_from_any_first_window_name_the_same_pairs() {
    let m = 5;
    let a: Vec<i64> = (0..40).map(|k| (k * k) % 37 - 18).collect();
    let b: Vec<i64> = (0..33).map(|k| (k * 13) % 29 - 14).collect();
    let whole = CrossTerms::new(&a[..], 0, &b[..], 0, m);
    let part = CrossTerms::new(&a[6..30], 6, &b[3..], 3, m);
    for i in 6..=30 - m {
        for j in 3..=b.len() - m {
            assert_eq!(part.at(i, j), whole.at(i, j), "({i}, {j})");
            assert_eq!(part.at(i, j), wide(&a[i..i + m], &b[j..j + m]));
        }
    }
}

#[test]
fn the_state_before_row_zero_advances_into_its_first_pair() {
    for m in [1, 2, 7] {
        let a: Vec<i64> = (0..20).map(|k| (k * 31) % 17 - 8).collect();
        let b: Vec<i64> = (0..20).map(|k| (k * 11) % 23 - 11).collect();
        let pairs = CrossTerms::new(&a[..], 0, &b[..], 0, m);
        for j in 1..=b.len() - m {
            let before = pairs.before_first_row(j);
            // One step with a leaving row tick of zero: the column's leaving tick drops out.
            let reached = before.wrapping_add(a[m - 1].wrapping_mul(b[j + m - 1]));
            assert_eq!(reached, pairs.at(0, j), "m {m} j {j}");
        }
        let offsets: Vec<i32> = a.iter().map(|&x| x as i32).collect();
        let cols: Vec<i32> = b.iter().map(|&x| x as i32).collect();
        let narrow_pairs = CrossTerms::new(&offsets[..], 0, &cols[..], 0, m);
        for j in 1..=b.len() - m {
            let me = (i64::from(offsets[m - 1]) * m as i64) as i32;
            let reached =
                narrowing::advance(narrow_pairs.before_first_row(j), me, 0, cols[j + m - 1], 0);
            assert_eq!(reached, narrow_pairs.at(0, j), "narrow m {m} j {j}");
        }
    }
}

#[test]
fn every_side_agrees_on_the_same_windows() {
    let m = 6;
    let anchor = Anchor { tick: 1_000_000 };
    let raw: Vec<i64> = (0..30).map(|k| 1_000_000 + (k * 97) % 61 - 30).collect();
    let offsets: Vec<i32> = raw.iter().map(|&t| (t - anchor.tick) as i32).collect();
    let side = Offsets {
        ticks: &raw,
        anchor,
        m: nz(m),
    };
    let (by_offsets, by_raw) = (CrossTerms::of(&offsets[..], m), CrossTerms::of(side, m));
    let absolute = CrossTerms::of(&raw[..], m);
    for (i, j) in [(0, 0), (7, 2), (24, 24), (24, 0)] {
        assert_eq!(by_raw.at(i, j), by_offsets.at(i, j));
        let sum = |k: usize| raw[k..k + m].iter().sum::<i64>();
        let back = narrowing::absolute_qt(anchor, nz(m), by_raw.at(i, j), sum(i), sum(j));
        assert_eq!(back, absolute.at(i, j), "({i}, {j})");
    }
}

#[test]
fn a_forced_resume_replaces_each_carried_state_with_the_exact_one() {
    let m = 3;
    let ticks: Vec<i64> = (0..24).map(|k| (k * 7) % 11 - 5).collect();
    let pairs = CrossTerms::of(&ticks[..], m);
    // Row 9 carries pairs (8, j − 1) for neighbors j = 2..6.
    let truth: Vec<i64> = (2..6).map(|j| pairs.at(8, j - 1)).collect();
    {
        let guard = forced_route(RouteForce::default());
        let mut states = vec![0x5555_5555i64; 4];
        pairs.force_resume(9, (9, 2), &mut states);
        assert_eq!(states, vec![0x5555_5555i64; 4], "nothing is forced");
        assert_eq!(guard.fired().resumes, 0);
    }
    {
        let guard = forced_route(RouteForce {
            resume_every: 3,
            ..RouteForce::default()
        });
        let mut states = vec![0x5555_5555i64; 4];
        pairs.force_resume(8, (9, 2), &mut states);
        assert_eq!(
            states,
            vec![0x5555_5555i64; 4],
            "row 8 is not a multiple of 3"
        );
        pairs.force_resume(9, (9, 2), &mut states);
        assert_eq!(states, truth);
        // Other tests' engines may resume under this force too, so the count is a floor.
        assert!(guard.fired().resumes >= 4);
    }
}

#[test]
fn a_forced_resume_skips_what_the_sides_do_not_hold() {
    let m = 2;
    let ticks: Vec<i64> = (0..16).map(|k| k * 3 - 20).collect();
    let pairs = CrossTerms::new(&ticks[4..], 4, &ticks[4..], 4, m);
    let guard = forced_route(RouteForce {
        resume_every: 1,
        ..RouteForce::default()
    });
    // Row 0 carries the state before it, not a pair: nothing to resume.
    let mut states = vec![7i64; 3];
    pairs.force_resume(0, (0, 1), &mut states);
    assert_eq!(states, vec![7; 3]);
    // Neighbor 4's carried column window (3) is before the sides; 5 and 6 are held.
    let mut states = vec![7i64; 3];
    pairs.force_resume(9, (9, 4), &mut states);
    assert_eq!(states, vec![7, pairs.at(8, 4), pairs.at(8, 5)]);
    assert!(guard.fired().resumes >= 2);
}

/// Lane `l` of a part carrying pair `(row − 1 − run, first + l − 1 − run)` into row `row`, stale
/// by `run` steps: `pairs.at` of that pair, `garbage` where the sides do not hold it.
fn stale<S: Side>(
    pairs: &CrossTerms<S>,
    run: usize,
    (row, first): (usize, usize),
    lanes: usize,
) -> Vec<i64> {
    (0..lanes)
        .map(|l| pairs.at(row - 1 - run, first + l - 1 - run))
        .collect()
}

/// What every lane must carry after catching up: the pair `(row − 1, first + l − 1)`.
fn caught<S: Side>(pairs: &CrossTerms<S>, (row, first): (usize, usize), lanes: usize) -> Vec<i64> {
    (0..lanes)
        .map(|l| pairs.at(row - 1, first + l - 1))
        .collect()
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]

    /// Stepping a part's lanes through a run by the recurrence, resuming them by one dot product
    /// each, and resuming them together all reach the same exact states, on the wide, the narrow
    /// and the offsets arithmetic, at any tick and at the narrow offset bound.
    #[test]
    fn a_stepped_run_equals_its_resume_on_every_arithmetic(
        (m, a, b, run, row, first, lanes) in (1usize..24).prop_flat_map(|m| {
            (Just(m), 2 * m + 2..4 * m + 40)
        }).prop_flat_map(|(m, len)| {
            (Just(m), narrow_offsets(m, len), narrow_offsets(m, len), 0usize..len - 2 * m)
        }).prop_flat_map(|(m, a, b, run)| {
            let last = a.len() - m + 1;
            (Just(m), Just(a), Just(b), Just(run), run + 1..=last, run + 1..=last)
        }).prop_flat_map(|(m, a, b, run, row, first)| {
            let most = b.len() - m + 2 - first;
            (Just(m), Just(a), Just(b), Just(run), Just(row), Just(first), 1..=most)
        })
    ) {
        let at = (row, first);
        let narrow_pairs = CrossTerms::new(&a[..], 0, &b[..], 0, m);
        let wa: Vec<i64> = a.iter().map(|&x| i64::from(x) * 977 + 3).collect();
        let wb: Vec<i64> = b.iter().map(|&x| i64::from(x).wrapping_mul(i64::MAX / 7)).collect();
        let wide_pairs = CrossTerms::new(&wa[..], 0, &wb[..], 0, m);
        let anchor = Anchor { tick: 1 << 40 };
        let ra: Vec<i64> = a.iter().map(|&x| anchor.tick + i64::from(x)).collect();
        let rb: Vec<i64> = b.iter().map(|&x| anchor.tick + i64::from(x)).collect();
        let side = |ticks| Offsets { ticks, anchor, m: nz(m) };
        let raw_pairs = CrossTerms::new(side(&ra[..]), 0, side(&rb[..]), 0, m);
        fn all<S: Side>(pairs: &CrossTerms<S>, run: usize, at: (usize, usize), lanes: usize) -> Result<(), TestCaseError> {
            let truth = caught(pairs, at, lanes);
            let mut stepped = stale(pairs, run, at, lanes);
            prop_assert!(pairs.step_lanes(run, at, &mut stepped));
            prop_assert_eq!(&stepped, &truth);
            let mut resumed = vec![0x5a5a; lanes];
            prop_assert_eq!(pairs.resume_lanes(at, &mut resumed), lanes);
            prop_assert_eq!(&resumed, &truth);
            let one_by_one: Vec<i64> = (0..lanes)
                .map(|l| pairs.resume((at.0 - 1 - run, at.1 + l - 1 - run), run).unwrap())
                .collect();
            prop_assert_eq!(&one_by_one, &truth);
            Ok(())
        }
        all(&narrow_pairs, run, at, lanes)?;
        all(&wide_pairs, run, at, lanes)?;
        all(&raw_pairs, run, at, lanes)?;
    }
}

#[test]
fn a_step_needs_every_tick_of_its_run_in_the_sides() {
    let m = 3;
    let ticks: Vec<i64> = (0..40).map(|k| (k * 7) % 13 - 6).collect();
    // The sides start at row window 6 and column window 4, and end at window 37.
    let pairs = CrossTerms::new(&ticks[6..], 6, &ticks[4..], 4, m);
    let mut states = vec![1i64; 4];
    // Lane 0 carries pair (9 − run, 9 − run) into row 10: run 3 starts at row window 6, held.
    let at = (10, 10);
    let mut held = stale(&pairs, 3, at, 4);
    assert!(pairs.step_lanes(3, at, &mut held));
    assert_eq!(held, caught(&pairs, at, 4));
    // A run of 4 starts at row window 5, before the row side: refused, states untouched.
    assert!(!pairs.step_lanes(4, at, &mut states));
    assert_eq!(states, vec![1; 4]);
    // Column side: lane 0 at column window 6 − run, before window 4 from run 3.
    assert!(!pairs.step_lanes(3, (13, 7), &mut states));
    // The last lane's windows past the end of the column side.
    assert!(!pairs.step_lanes(0, (13, 36), &mut states));
    assert!(pairs.step_lanes(0, (13, 35), &mut states));
    // Overflowing runs and rows are refused, never a panic.
    assert!(!pairs.step_lanes(usize::MAX, at, &mut states));
    assert!(!pairs.step_lanes(1, (0, 0), &mut states));
    assert!(!pairs.step_lanes(1, (usize::MAX, usize::MAX), &mut states));
    assert_eq!(states, vec![1; 4]);
}

#[test]
fn a_lane_resume_replaces_only_the_lanes_the_sides_hold() {
    let m = 2;
    let ticks: Vec<i64> = (0..16).map(|k| k * 3 - 20).collect();
    let pairs = CrossTerms::new(&ticks[4..], 4, &ticks[4..], 4, m);
    let mut states = vec![7i64; 3];
    assert_eq!(pairs.resume_lanes((0, 1), &mut states), 0);
    // Lane 0's column window 3 is before the sides.
    assert_eq!(pairs.resume_lanes((9, 4), &mut states), 2);
    assert_eq!(states, vec![7, pairs.at(8, 4), pairs.at(8, 5)]);
    // Windows 14 and 15 end past the sides (window 14 is the last held).
    let mut states = vec![7i64; 3];
    assert_eq!(pairs.resume_lanes((9, 14), &mut states), 2);
    assert_eq!(states, vec![pairs.at(8, 13), pairs.at(8, 14), 7]);
    // A row window past the sides holds no lane.
    assert_eq!(pairs.resume_lanes((16, 6), &mut states), 0);
    assert_eq!(pairs.resume_lanes((usize::MAX, usize::MAX), &mut states), 0);
}

/// The rule steps a run shorter than the break-even and resumes any other, a forced branch is
/// taken whatever the run, a step the sides refuse resumes, and every branch reaches the same
/// states.
#[test]
fn the_rule_steps_short_runs_and_resumes_long_ones() {
    let m = 16;
    let even = break_even(m);
    assert!(even > 0);
    let ticks: Vec<i32> = (0..400).map(|k| (k * 37) % 51 - 25).collect();
    let pairs = CrossTerms::of(&ticks[..], m);
    let lanes = 9;
    let at = (300, 200);
    let run_with = |force: CatchUpForce, run: usize| {
        let _g = forced_route(RouteForce {
            catch_up: force,
            ..RouteForce::default()
        });
        let mut states = stale(&pairs, run, at, lanes);
        let how = pairs.advance_or_resume(run, at, &mut states);
        assert_eq!(states, caught(&pairs, at, lanes), "{force:?} run {run}");
        how
    };
    let stepped = |run: usize| CatchUp {
        resumed: 0,
        stepped: lanes,
        steps: lanes * run,
    };
    let resumed = CatchUp {
        resumed: lanes,
        stepped: 0,
        steps: 0,
    };
    for run in [1, even - 1] {
        assert_eq!(run_with(CatchUpForce::Rule, run), stepped(run), "run {run}");
    }
    for run in [even, even + 1, 3 * m] {
        assert_eq!(run_with(CatchUpForce::Rule, run), resumed, "run {run}");
    }
    for run in [1, even, 3 * m] {
        assert_eq!(run_with(CatchUpForce::Step, run), stepped(run), "run {run}");
        assert_eq!(run_with(CatchUpForce::Resume, run), resumed, "run {run}");
    }
    // A run longer than the sides hold before the part cannot step: it resumes.
    let _g = forced_route(RouteForce {
        catch_up: CatchUpForce::Step,
        ..RouteForce::default()
    });
    let mut states = vec![0; lanes];
    assert_eq!(pairs.advance_or_resume(250, at, &mut states), resumed);
    assert_eq!(states, caught(&pairs, at, lanes));
}

#[test]
fn the_break_even_grows_with_the_window() {
    let mut last = 0;
    for m in [1, 2, 4, 16, 64, 256, 1024, 4096, 1 << 20] {
        let even = break_even(m);
        assert!(even >= last, "m {m}");
        last = even;
    }
    assert!(break_even(usize::MAX) > 0);
}

#[cfg(target_arch = "x86_64")]
mod avx2_start_sum {
    use super::super::sum_of_products;
    use super::*;
    use crate::kernel_avx2::{detected, narrow_dot, narrow_lanes};
    use testkit::rng::splitmix64;

    /// The AVX2 bodies run here or the test fails, so an x86 machine without AVX2 cannot pass
    /// these tests without running them. A unit test detects the CPU with or without `std`.
    fn require_avx2() {
        assert!(
            detected(),
            "this CPU has no AVX2, so the AVX2 sums cannot run"
        );
    }

    /// Extremes, small values and anything, so the `i64` sums wrap.
    fn values(state: &mut u64, len: usize) -> Vec<i32> {
        (0..len)
            .map(|_| match splitmix64(state) % 6 {
                0 => i32::MIN,
                1 => i32::MAX,
                2 => (splitmix64(state) % 2001) as i32 - 1000,
                _ => splitmix64(state) as i32,
            })
            .collect()
    }

    fn scalar_dot(a: &[i32], b: &[i32]) -> i64 {
        sum_of_products(a, b, |x, y| i64::from(x).wrapping_mul(i64::from(y)))
    }

    fn avx_dot(a: &[i32], b: &[i32]) -> i64 {
        // SAFETY: every caller ran `require_avx2` first.
        unsafe { narrow_dot(a, b) }
    }

    #[test]
    fn dot_equals_the_scalar_sum() {
        require_avx2();
        let mut state = 0x5eed_57a2_7500_0001u64;
        let lens = (0..=70usize).chain([1024, 2048, 16384].into_iter().flat_map(|b| b..b + 17));
        for len in lens {
            let (a, b) = (values(&mut state, len), values(&mut state, len));
            assert_eq!(avx_dot(&a, &b), scalar_dot(&a, &b), "len {len}");
            let c = values(&mut state, len + 3);
            assert_eq!(avx_dot(&a, &c), scalar_dot(&a, &c), "unequal {len}");
            assert_eq!(avx_dot(&c, &a), scalar_dot(&c, &a), "unequal {len}");
        }
        for fill in [i32::MIN, i32::MAX, -1, 1] {
            for len in [15usize, 16, 17, 4099] {
                let a = vec![fill; len];
                assert_eq!(avx_dot(&a, &a), scalar_dot(&a, &a), "fill {fill} len {len}");
                let wide: Vec<i64> = a.iter().map(|&x| i64::from(x)).collect();
                assert_eq!(avx_dot(&a, &a), exact_wrapped(&wide, &wide));
            }
        }
    }

    #[test]
    fn lanes_equal_the_scalar_sums() {
        require_avx2();
        let mut state = 0x5eed_57a2_7500_0002u64;
        for m in [0usize, 1, 2, 3, 7, 8, 9, 15, 16, 17, 33, 64, 70] {
            for lanes in 0..=40usize {
                let a = values(&mut state, m);
                let b = values(&mut state, (lanes + m).saturating_sub(1));
                let mut got = vec![7i64; lanes];
                // SAFETY: `require_avx2` passed.
                unsafe { narrow_lanes(&a, &b, &mut got) };
                let want: Vec<i64> = (0..lanes).map(|l| scalar_dot(&a, &b[l..])).collect();
                assert_eq!(got, want, "m {m} lanes {lanes}");
            }
        }
        let a = vec![i32::MIN; 33];
        let b = vec![i32::MIN; 33 + 39];
        let mut got = vec![0i64; 40];
        // SAFETY: `require_avx2` passed.
        unsafe { narrow_lanes(&a, &b, &mut got) };
        assert!(got.iter().all(|&s| s == scalar_dot(&a, &b[..33])));
    }

    #[test]
    fn the_sides_take_the_route_and_the_force_restores_scalar() {
        require_avx2();
        let mut state = 0x5eed_57a2_7500_0003u64;
        let (m, lanes) = (37usize, 19usize);
        let a = values(&mut state, m + 4);
        let b = values(&mut state, m + lanes + 4);
        let (sa, sb) = (a.as_slice(), b.as_slice());
        let run = || {
            let start = narrow(&sa[1..][..m], &sb[2..][..m], m);
            let mut states = vec![0i64; lanes];
            sa.cross_lanes(1, sb, 2, m, &mut states);
            (start, states)
        };
        let guard = forced_route(RouteForce::default());
        let (natural, natural_lanes) = run();
        assert_eq!(guard.avx2_start_sums(), 2);
        drop(guard);
        let guard = forced_route(RouteForce {
            scalar_start_sum: true,
            ..RouteForce::default()
        });
        let (scalar, scalar_lanes) = run();
        assert_eq!(guard.avx2_start_sums(), 0);
        assert_eq!((natural, natural_lanes), (scalar, scalar_lanes));
    }
}
