extern crate std;

use super::{Anchor, Narrow, Path, advance, cross, reanchor, reanchor_pair};
use crate::kernel::Metric;
use crate::quantize::{Scale, to_ticks};
use crate::sanitize::{ForwardFill, Sanitized};
use crate::stats::{Window, WindowKind, WindowStats};
use crate::znorm::{TickWindow, ZNorm, centered_cross};
use core::num::NonZeroUsize;
use proptest::prelude::*;
use proptest::sample::Index;
use std::vec::Vec;
use testkit::generators::{MARKET_TICK, market_series_and_m};

// ---------- Path is a plain reporting enum ----------

#[test]
fn path_variants_are_distinct() {
    assert_ne!(Path::Narrow, Path::Wide);
}

// ---------- hand-worked case ----------

/// `m = 2`, window a = ticks `[101, 103]`, window b = ticks `[102, 104]`, anchor `100`:
/// offsets `[1, 3]` and `[2, 4]`, worked out by hand against the absolute formula
/// `N = m·QT − Sᵢ·Sⱼ = 2·21014 − 204·206 = 4`, then re-anchored to `105`.
#[test]
fn small_case_matches_hand_worked_values() {
    let m = NonZeroUsize::new(2).unwrap();
    let anchor = Anchor { tick: 100 };

    let oa = [1i32, 3];
    let ob = [2i32, 4];
    let qt: i64 = oa
        .iter()
        .zip(&ob)
        .map(|(&a, &b)| i64::from(a) * i64::from(b))
        .sum();
    let qt_m = 2 * qt;
    assert_eq!(qt_m, 28);
    let sa: i64 = oa.iter().map(|&o| i64::from(o)).sum();
    let sb: i64 = ob.iter().map(|&o| i64::from(o)).sum();
    assert_eq!((sa, sb), (4, 6));
    assert_eq!(cross(qt_m, sa as i32, sb as i32), 4);

    // same pair, directly from the absolute ticks.
    let w_a = TickWindow {
        entering: 0,
        leaving: 0,
        stats: window(204),
    };
    let w_b = TickWindow {
        entering: 0,
        leaving: 0,
        stats: window(206),
    };
    let qt_abs = ZNorm::start(&[101, 103], &w_a, &[102, 104], &w_b);
    assert_eq!(centered_cross(qt_abs, &w_a, &w_b, 2), 4);

    // re-anchor by +5: offsets become [-4, -2] and [-3, -1], same N by construction.
    let (qt_m2, sa2, sb2) = reanchor(5, m, qt_m, sa, sb);
    assert_eq!((qt_m2, sa2, sb2), (28, -6, -4));
    assert_eq!(cross(qt_m2, sa2 as i32, sb2 as i32), 4);
    anchor_offsets_match(&anchor, m, 1, 1); // sanity: offset(anchor.tick + 1) is 1
}

fn anchor_offsets_match(anchor: &Anchor, m: NonZeroUsize, delta: i64, expected: i32) {
    assert_eq!(anchor.offset(anchor.tick + delta, m), Some(expected));
}

fn window(sum: i64) -> Window {
    Window {
        sum,
        kind: WindowKind::Valid {
            variance_numerator: 1,
            inv_sqrt: 1.0,
        },
    }
}

// ---------- fits: exactly within the bound ----------

#[test]
fn fits_accepts_exactly_within_the_bound() {
    for &m in &[1usize, 2, 50, 256, 4096] {
        let m = NonZeroUsize::new(m).unwrap();
        let bound = i64::from(i32::MAX) / m.get() as i64;
        for offset in [bound, -bound] {
            assert!(Narrow::fits(offset, m), "m {}, offset {offset}", m.get());
        }
        for offset in [bound + 1, -(bound + 1)] {
            assert!(!Narrow::fits(offset, m), "m {}, offset {offset}", m.get());
        }
    }
}

#[test]
fn offset_is_none_outside_the_bound_and_some_within_it() {
    let anchor = Anchor {
        tick: 1_000_000_000,
    };
    let m = NonZeroUsize::new(4).unwrap();
    let bound = i64::from(i32::MAX) / 4;
    assert_eq!(anchor.offset(anchor.tick + bound, m), Some(bound as i32));
    assert_eq!(anchor.offset(anchor.tick - bound, m), Some(-(bound as i32)));
    assert_eq!(anchor.offset(anchor.tick + bound + 1, m), None);
    assert_eq!(anchor.offset(anchor.tick - bound - 1, m), None);
}

// ---------- property: narrow N and V match the i64 path's ----------

/// Non-finite or out-of-range samples are quantized then forward-filled, as in
/// `znorm`'s own tests.
fn to_tick_series(series: &[f64], scale: Scale) -> Vec<i64> {
    let mut fill = ForwardFill::default();
    series
        .iter()
        .map(|&x| match fill.push(to_ticks(x, scale)) {
            Sanitized::Present(t) | Sanitized::Filled(t) => t,
            Sanitized::Leading => 0,
        })
        .collect()
}

fn windows_of(ticks: &[i64], m: usize) -> Vec<Window> {
    let mut stats = WindowStats::new(NonZeroUsize::new(m).unwrap());
    ticks
        .iter()
        .filter_map(|&tick| stats.push(tick, true))
        .collect()
}

fn is_valid(window: Window) -> bool {
    matches!(window.kind, WindowKind::Valid { .. })
}

fn tick_window_at(ticks: &[i64], windows: &[Window], i: usize, m: usize) -> TickWindow {
    TickWindow {
        entering: ticks[i + m - 1],
        leaving: if i == 0 { 0 } else { ticks[i - 1] },
        stats: windows[i],
    }
}

/// Scales an in-bound offset by `m`, which [`Narrow::fits`]'s bound keeps within i32.
fn scale(offset: i32, m: usize) -> Option<i32> {
    i32::try_from(i64::from(offset) * m as i64).ok()
}

/// Walks diagonal `d` from its first cell with the narrow functions, checking `cross`
/// against `znorm`'s own `centered_cross` on the absolute ticks at every pair of valid
/// windows (`d = 0` pairs a window with itself, so this covers the variance numerator
/// too). Returns `None` to skip a diagonal where some tick falls outside the anchor's
/// bound, rather than fail the case.
fn check_diagonal_narrow(
    ticks: &[i64],
    windows: &[Window],
    m: NonZeroUsize,
    d: usize,
    anchor: Anchor,
) -> Option<()> {
    if d >= windows.len() {
        return Some(());
    }
    let count = windows.len() - d;
    let at = |i: usize| tick_window_at(ticks, windows, i, m.get());
    let offset_of = |tick: i64| anchor.offset(tick, m);

    let w0 = at(0);
    let wd = at(d);
    let mut qt_abs = ZNorm::start(&ticks[0..m.get()], &w0, &ticks[d..d + m.get()], &wd);

    let oa: Option<Vec<i32>> = ticks[0..m.get()].iter().map(|&t| offset_of(t)).collect();
    let ob: Option<Vec<i32>> = ticks[d..d + m.get()]
        .iter()
        .map(|&t| offset_of(t))
        .collect();
    let (oa, ob) = (oa?, ob?);
    let qt: i64 = oa
        .iter()
        .zip(&ob)
        .map(|(&a, &b)| i64::from(a) * i64::from(b))
        .sum();
    let mut qt_m = m.get() as i64 * qt;
    let mut sa: i64 = oa.iter().map(|&o| i64::from(o)).sum();
    let mut sb: i64 = ob.iter().map(|&o| i64::from(o)).sum();

    for i in 0..count {
        let j = i + d;
        if i > 0 {
            let wi = at(i);
            let wj = at(j);
            qt_abs = ZNorm::advance(qt_abs, &wi, &wj);

            let me_raw = offset_of(wi.entering)?;
            let ml_raw = offset_of(wi.leaving)?;
            let te = offset_of(wj.entering)?;
            let tl = offset_of(wj.leaving)?;
            let me = scale(me_raw, m.get())?;
            let ml = scale(ml_raw, m.get())?;
            qt_m = advance(qt_m, me, ml, te, tl);

            sa = sa - i64::from(ml_raw) + i64::from(me_raw);
            sb = sb - i64::from(tl) + i64::from(te);
        }

        if is_valid(windows[i]) && is_valid(windows[j]) {
            let expected = centered_cross(qt_abs, &at(i), &at(j), m.get());
            let si = i32::try_from(sa).ok()?;
            let sj = i32::try_from(sb).ok()?;
            assert_eq!(
                cross(qt_m, si, sj),
                expected,
                "i {i}, j {j}, d {d}, m {}",
                m.get()
            );
        }
    }
    Some(())
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    /// For a random market series near a 1e8 price level, every m it holds at, and a
    /// random anchor drawn from the series' own ticks, every diagonal's narrow `cross`
    /// matches `znorm`'s absolute `centered_cross`, at every pair of valid windows.
    #[test]
    fn narrow_matches_the_i64_path_for_n_and_v(
        (series, m) in market_series_and_m(2..40),
        anchor_at in any::<Index>(),
    ) {
        let scale_unit = Scale::new(MARKET_TICK).unwrap();
        let ticks = to_tick_series(&series, scale_unit);
        if ticks.is_empty() {
            return Ok(());
        }
        let windows = windows_of(&ticks, m);
        let m = NonZeroUsize::new(m).unwrap();
        let anchor = Anchor { tick: ticks[anchor_at.index(ticks.len())] };
        for d in 0..windows.len() {
            check_diagonal_narrow(&ticks, &windows, m, d, anchor);
        }
    }
}

// ---------- property: re-anchor matches building fresh at the new anchor ----------

/// `(m·QT, Sₐ, S_b)` for window pair `(i, j)` at `anchor`, straight from the offsets'
/// definitions: no recurrence, nothing shared with `advance`/`cross`/`reanchor`.
fn narrow_state_from_scratch(
    ticks: &[i64],
    i: usize,
    j: usize,
    m: NonZeroUsize,
    anchor: Anchor,
) -> Option<(i64, i64, i64)> {
    let offsets = |start: usize| -> Option<Vec<i32>> {
        ticks[start..start + m.get()]
            .iter()
            .map(|&t| anchor.offset(t, m))
            .collect()
    };
    let oa = offsets(i)?;
    let ob = offsets(j)?;
    let qt: i64 = oa
        .iter()
        .zip(&ob)
        .map(|(&a, &b)| i64::from(a) * i64::from(b))
        .sum();
    let qt_m = m.get() as i64 * qt;
    let sa: i64 = oa.iter().map(|&o| i64::from(o)).sum();
    let sb: i64 = ob.iter().map(|&o| i64::from(o)).sum();
    Some((qt_m, sa, sb))
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    /// Re-anchoring a diagonal's state by `delta`, then advancing it one more lag, gives
    /// exactly what building that state from scratch at the new anchor would: both right
    /// after the re-anchor, and after the following advance.
    #[test]
    fn reanchor_then_continue_matches_building_fresh_at_the_new_anchor(
        (series, m) in market_series_and_m(4..40),
        i_at in any::<Index>(),
        j_at in any::<Index>(),
        delta in -2_000_000i64..=2_000_000,
    ) {
        let scale_unit = Scale::new(MARKET_TICK).unwrap();
        let ticks = to_tick_series(&series, scale_unit);
        let m = NonZeroUsize::new(m).unwrap();
        if ticks.len() < m.get() + 1 {
            return Ok(());
        }
        let last_start = ticks.len() - m.get() - 1;
        let i = i_at.index(last_start + 1);
        let j = j_at.index(last_start + 1);

        let old_anchor = Anchor { tick: ticks[i] };
        let Some(new_tick) = old_anchor.tick.checked_add(delta) else { return Ok(()); };
        let new_anchor = Anchor { tick: new_tick };

        let Some((qt_m_old, sa_old, sb_old)) =
            narrow_state_from_scratch(&ticks, i, j, m, old_anchor) else { return Ok(()); };
        let Some((qt_m_fresh, sa_fresh, sb_fresh)) =
            narrow_state_from_scratch(&ticks, i, j, m, new_anchor) else { return Ok(()); };

        let (qt_m_re, sa_re, sb_re) = reanchor(delta, m, qt_m_old, sa_old, sb_old);
        prop_assert_eq!((qt_m_re, sa_re, sb_re), (qt_m_fresh, sa_fresh, sb_fresh));

        let (Some(me_raw), Some(te_raw), Some(ml_raw), Some(tl_raw)) = (
            new_anchor.offset(ticks[i + m.get()], m),
            new_anchor.offset(ticks[j + m.get()], m),
            new_anchor.offset(ticks[i], m),
            new_anchor.offset(ticks[j], m),
        ) else { return Ok(()); };
        let (Some(me), Some(ml)) = (scale(me_raw, m.get()), scale(ml_raw, m.get())) else {
            return Ok(());
        };

        let qt_m_next = advance(qt_m_re, me, ml, te_raw, tl_raw);
        let sa_next = sa_re - i64::from(ml_raw) + i64::from(me_raw);
        let sb_next = sb_re - i64::from(tl_raw) + i64::from(te_raw);

        let Some((qt_m_next_fresh, sa_next_fresh, sb_next_fresh)) =
            narrow_state_from_scratch(&ticks, i + 1, j + 1, m, new_anchor) else {
            return Ok(());
        };
        prop_assert_eq!(qt_m_next, qt_m_next_fresh);
        prop_assert_eq!(sa_next, sa_next_fresh);
        prop_assert_eq!(sb_next, sb_next_fresh);
    }
}

// ---------- extreme offsets: the bound itself, not the tens market series stay within ----------

const EXTREME_MS: [usize; 5] = [1, 2, 50, 256, 4096];

/// `⌊(2³¹−1)/m⌋`, independently of [`Narrow::fits`]'s own arithmetic.
fn bound(m: NonZeroUsize) -> i64 {
    i64::from(i32::MAX) / m.get() as i64
}

/// `N` for two windows of absolute ticks (or `V`, when `b` is `a`), straight from the
/// definitions in i128: the ground truth the narrow path is checked against here, with no
/// anchor and no wrapping anywhere.
fn exact_cross_i128(a: &[i64], b: &[i64]) -> i128 {
    let qt: i128 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| i128::from(x) * i128::from(y))
        .sum();
    let sa: i128 = a.iter().map(|&x| i128::from(x)).sum();
    let sb: i128 = b.iter().map(|&x| i128::from(x)).sum();
    i128::try_from(a.len()).unwrap() * qt - sa * sb
}

/// `(m·QT, Sₐ, S_b)` for two offset windows, straight from the definitions: like
/// [`narrow_state_from_scratch`], but without an anchor or a `fits` check, since every
/// offset here is already known to be in bound by construction.
fn narrow_from_offsets(oa: &[i32], ob: &[i32]) -> (i64, i64, i64) {
    let qt: i64 = oa
        .iter()
        .zip(ob)
        .map(|(&a, &b)| i64::from(a) * i64::from(b))
        .sum();
    let qt_m = oa.len() as i64 * qt;
    let sa: i64 = oa.iter().map(|&o| i64::from(o)).sum();
    let sb: i64 = ob.iter().map(|&o| i64::from(o)).sum();
    (qt_m, sa, sb)
}

/// Checks `cross` against [`exact_cross_i128`] for `(oa, ob)` at `anchor` (`N`, and at the
/// self-pair, `V`), then advances both windows with a newest tick exactly at the bound
/// (`lo` for `a`, `hi` for `b`) and checks `advance` the same way.
fn check_extreme_pair(anchor: i64, oa: &[i32], ob: &[i32], lo: i32, hi: i32) {
    let abs =
        |offsets: &[i32]| -> Vec<i64> { offsets.iter().map(|&o| anchor + i64::from(o)).collect() };
    let check = |oa: &[i32], ob: &[i32]| {
        let (qt_m, sa, sb) = narrow_from_offsets(oa, ob);
        let expected = i64::try_from(exact_cross_i128(&abs(oa), &abs(ob)))
            .expect("N fits i64 for offsets within the bound");
        let si = i32::try_from(sa).expect("row scalar fits i32 within the bound");
        let sj = i32::try_from(sb).expect("row scalar fits i32 within the bound");
        assert_eq!(cross(qt_m, si, sj), expected, "oa {oa:?}, ob {ob:?}");
    };
    check(oa, ob);
    check(oa, oa); // the self-pair: V by the same mechanism.

    let m = oa.len();
    let (qt_m0, sa0, sb0) = narrow_from_offsets(oa, ob);
    let mut window_a = oa.to_vec();
    let mut window_b = ob.to_vec();
    let leave_a = window_a.remove(0);
    let leave_b = window_b.remove(0);
    window_a.push(lo);
    window_b.push(hi);

    let me = scale(lo, m).expect("extreme entering offset scales within i32");
    let ml = scale(leave_a, m).expect("in-bound leaving offset scales within i32");
    let qt_m1 = advance(qt_m0, me, ml, hi, leave_b);
    let sa1 = sa0 - i64::from(leave_a) + i64::from(lo);
    let sb1 = sb0 - i64::from(leave_b) + i64::from(hi);
    assert_eq!((qt_m1, sa1, sb1), narrow_from_offsets(&window_a, &window_b));
    check(&window_a, &window_b);
}

#[test]
fn extreme_offsets_hit_the_exact_bound_at_every_m() {
    let anchor = 7_000_000_007i64;
    for &m_pick in &EXTREME_MS {
        let m = NonZeroUsize::new(m_pick).unwrap();
        let hi = i32::try_from(bound(m)).unwrap();
        let lo = -hi;
        let oa = std::vec![hi; m_pick];
        let ob = std::vec![lo; m_pick];
        check_extreme_pair(anchor, &oa, &ob, lo, hi);

        // `Anchor::offset` itself, not just the arithmetic downstream of it, draws the
        // line at this exact bound for every `m` here, not only the one `m` the older
        // `offset_is_none_outside_the_bound_and_some_within_it` unit test checks.
        let anchor_point = Anchor { tick: anchor };
        assert_eq!(anchor_point.offset(anchor + i64::from(hi), m), Some(hi));
        assert_eq!(anchor_point.offset(anchor - i64::from(hi), m), Some(lo));
        assert_eq!(anchor_point.offset(anchor + i64::from(hi) + 1, m), None);
        assert_eq!(anchor_point.offset(anchor - i64::from(hi) - 1, m), None);
    }
}

/// `m`, an anchor, and two offset windows of length `m` drawn uniformly from the full
/// `±⌊(2³¹−1)/m⌋` range `[`Narrow::fits`]'s bound allows, including its exact endpoints.
fn extreme_case() -> impl Strategy<Value = (i64, i32, i32, Vec<i32>, Vec<i32>)> {
    (
        prop::sample::select(&EXTREME_MS[..]),
        -1_000_000_000_000i64..=1_000_000_000_000i64,
    )
        .prop_flat_map(|(m_pick, anchor)| {
            let m = NonZeroUsize::new(m_pick).unwrap();
            let hi = i32::try_from(bound(m)).unwrap();
            let lo = -hi;
            let window = prop::collection::vec(lo..=hi, m_pick);
            (Just(anchor), Just(lo), Just(hi), window.clone(), window)
        })
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    /// Offsets drawn uniformly from the full `±⌊(2³¹−1)/m⌋` range, not the tens the market
    /// generators stay within, at every `m` the bound is keyed to: `cross` and `advance`
    /// still match the i128 ground truth from absolute ticks, for `N` and, at the
    /// self-pair, for `V`.
    #[test]
    fn extreme_offsets_match_the_i128_ground_truth((anchor, lo, hi, oa, ob) in extreme_case()) {
        check_extreme_pair(anchor, &oa, &ob, lo, hi);
    }
}

// ---------- re-anchoring near the bound, then rolling on past it ----------

/// Small enough that, after `delta` claims the rest of the bound, the old anchor's own
/// offsets (`delta` plus this) still fit `i32`, and small enough that ten further
/// `advance` steps at offsets this size trivially stay in bound too.
const JITTER: i64 = 500;

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    /// Re-anchoring by `delta` within a small margin of the bound — not the `≤ 2·10⁶` the
    /// other re-anchor property test stays under — from an old anchor whose own offsets
    /// are themselves that close to the bound, lands exactly where building fresh at the
    /// new anchor would; ten further `advance` steps at small, ordinary offsets keep
    /// agreeing with a fresh build at every one of them.
    #[test]
    fn reanchor_near_the_bound_then_ten_advances_matches_a_fresh_build(
        m_pick in prop::sample::select(&EXTREME_MS[..]),
        delta_sign in prop::bool::ANY,
        delta_margin in 0i64..=1_000,
        jitter_a in -JITTER..=JITTER,
        jitter_b in -JITTER..=JITTER,
        steps_a in prop::collection::vec(-JITTER..=JITTER, 10),
        steps_b in prop::collection::vec(-JITTER..=JITTER, 10),
    ) {
        let m = NonZeroUsize::new(m_pick).unwrap();
        let hi = bound(m);
        let delta_mag = hi - JITTER - delta_margin;
        let delta = if delta_sign { delta_mag } else { -delta_mag };

        let old_a = i32::try_from(delta + jitter_a).unwrap();
        let old_b = i32::try_from(delta + jitter_b).unwrap();
        let oa = std::vec![old_a; m_pick];
        let ob = std::vec![old_b; m_pick];
        let (qt_m_old, sa_old, sb_old) = narrow_from_offsets(&oa, &ob);

        let (mut qt_m, mut sa, mut sb) = reanchor(delta, m, qt_m_old, sa_old, sb_old);

        let mut window_a = std::vec![i32::try_from(jitter_a).unwrap(); m_pick];
        let mut window_b = std::vec![i32::try_from(jitter_b).unwrap(); m_pick];
        prop_assert_eq!((qt_m, sa, sb), narrow_from_offsets(&window_a, &window_b));

        for k in 0..10 {
            let enter_a = i32::try_from(steps_a[k]).unwrap();
            let enter_b = i32::try_from(steps_b[k]).unwrap();
            let leave_a = window_a.remove(0);
            let leave_b = window_b.remove(0);
            window_a.push(enter_a);
            window_b.push(enter_b);

            let me = scale(enter_a, m_pick).expect("small entering offset scales within i32");
            let ml = scale(leave_a, m_pick).expect("small leaving offset scales within i32");
            qt_m = advance(qt_m, me, ml, enter_b, leave_b);
            sa = sa - i64::from(leave_a) + i64::from(enter_a);
            sb = sb - i64::from(leave_b) + i64::from(enter_b);

            prop_assert_eq!((qt_m, sa, sb), narrow_from_offsets(&window_a, &window_b));
        }
    }
}

#[test]
fn reanchor_at_the_exact_bound_then_ten_advances_matches_a_fresh_build() {
    for &m_pick in &EXTREME_MS {
        let m = NonZeroUsize::new(m_pick).unwrap();
        let delta = bound(m); // the largest a re-anchor step can validly be.
        let old = i32::try_from(delta).unwrap();
        let oa = std::vec![old; m_pick];
        let ob = std::vec![old; m_pick];
        let (qt_m_old, sa_old, sb_old) = narrow_from_offsets(&oa, &ob);

        let (mut qt_m, mut sa, mut sb) = reanchor(delta, m, qt_m_old, sa_old, sb_old);
        let mut window_a = std::vec![0i32; m_pick];
        let mut window_b = std::vec![0i32; m_pick];
        assert_eq!(
            (qt_m, sa, sb),
            narrow_from_offsets(&window_a, &window_b),
            "m {m_pick}"
        );

        for k in 0..10i32 {
            let enter_a = (k * 37) % 97 - 48;
            let enter_b = (k * 53) % 97 - 48;
            let leave_a = window_a.remove(0);
            let leave_b = window_b.remove(0);
            window_a.push(enter_a);
            window_b.push(enter_b);

            let me = scale(enter_a, m_pick).unwrap();
            let ml = scale(leave_a, m_pick).unwrap();
            qt_m = advance(qt_m, me, ml, enter_b, leave_b);
            sa = sa - i64::from(leave_a) + i64::from(enter_a);
            sb = sb - i64::from(leave_b) + i64::from(enter_b);

            assert_eq!(
                (qt_m, sa, sb),
                narrow_from_offsets(&window_a, &window_b),
                "m {m_pick}, step {k}"
            );
        }
    }
}

/// `Anchor::can_cover` is the natural rule alone: it agrees with `covering` around the bound.
#[test]
#[cfg(not(any(loom, shuttle)))]
fn can_cover_agrees_with_covering_where_no_route_is_forced() {
    let _serial = crate::route::forced_route(crate::route::RouteForce::default());
    for m in [1usize, 2, 3, 7, 50, 1000] {
        let bound = i64::from(i32::MAX) / m as i64;
        let m = NonZeroUsize::new(m).unwrap();
        for span in [
            0,
            1,
            bound,
            2 * bound - 1,
            2 * bound,
            2 * bound + 1,
            3 * bound,
        ] {
            for low in [-5 * bound, -1, 0, 7] {
                let ticks = [low, low + span / 2, low + span];
                assert_eq!(
                    Anchor::can_cover(&ticks, m),
                    Anchor::covering(&ticks, m).is_some(),
                    "m {m}, span {span}"
                );
            }
        }
    }
    let m = NonZeroUsize::new(4).unwrap();
    assert!(Anchor::can_cover(&[], m) && Anchor::covering(&[], m).is_none());
}

// ---------- two independent anchors ----------

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    /// Moving the row and the column anchor apart gives the `m·QT` of a fresh dot product at the
    /// new anchors, whatever the old and new anchors are, as long as both windows fit both.
    #[test]
    fn reanchor_pair_equals_a_fresh_dot_product_at_the_new_anchors(
        m in 1usize..=40,
        row in prop::collection::vec(any::<i64>(), 40),
        col in prop::collection::vec(any::<i64>(), 40),
        anchors in any::<[i64; 4]>(),
        spread in any::<[u32; 2]>(),
    ) {
        let nm = NonZeroUsize::new(m).unwrap();
        let bound = i64::from(i32::MAX) / m as i64;
        // Windows near their anchors: every tick within the bound of both the old and new anchor.
        let near = |ticks: &[i64], old: i64, new: i64| -> Vec<i64> {
            let mid = old / 2 + new / 2;
            let reach = bound / 2;
            ticks[..m]
                .iter()
                .map(|&t| mid + t.rem_euclid(2 * reach + 1) - reach)
                .collect()
        };
        let (old_row, old_col) = (anchors[0] / 4, anchors[1] / 4);
        let (new_row, new_col) = (old_row + i64::from(spread[0] % (bound as u32).max(1)) - bound / 4, old_col + i64::from(spread[1] % (bound as u32).max(1)) - bound / 4);
        let rows = near(&row, old_row, new_row);
        let cols = near(&col, old_col, new_col);
        let dot = |a: i64, b: i64| -> i64 {
            rows.iter().zip(&cols).fold(0i64, |acc, (&x, &y)| {
                acc.wrapping_add((x - a).wrapping_mul(y - b))
            }).wrapping_mul(m as i64)
        };
        let sums = |a: i64, b: i64| -> (i64, i64) {
            (rows.iter().map(|&x| x - a).sum(), cols.iter().map(|&y| y - b).sum())
        };
        let (sx, sy) = sums(old_row, old_col);
        let moved = reanchor_pair(nm, new_row - old_row, new_col - old_col, dot(old_row, old_col), sx, sy);
        prop_assert_eq!(moved, dot(new_row, new_col));
    }
}

#[test]
fn a_span_of_twice_the_bound_fits_one_anchor_and_one_tick_more_does_not() {
    for m in [1usize, 2, 7, 64, 513] {
        let nm = NonZeroUsize::new(m).unwrap();
        let twice = 2 * (i64::from(i32::MAX) / m as i64);
        let lo = -123_456_789;
        let anchor = Anchor::fitting(lo, lo + twice, nm, 0).expect("the widest span fits");
        assert!(Narrow::fits(lo - anchor.tick, nm) && Narrow::fits(lo + twice - anchor.tick, nm));
        assert_eq!(Anchor::fitting(lo, lo + twice + 1, nm, 0), None);
    }
}
