#![cfg(not(any(loom, shuttle)))]

//! Non-normalized snippets (STUMPY's `aampdist_snippets`) equal a plain statement of the procedure
//! on exact integer ticks: every segment's profile built window by window from the squared tick
//! distances, STUMPY's value at the selected position, the pick, fractions and regimes shared with
//! the z-normalized reference; on random series, tie-heavy series, gaps and infinite samples, every
//! inner length, selection extreme and thread count. STUMPY drops the last segment when `m`
//! divides the length, the z-normalized snippets do not.

use core::num::NonZeroUsize;

#[path = "common/nonnorm_derived.rs"]
mod reference;
#[path = "common/snippets_pick.rs"]
mod snippets_pick;

use ladde::mpdist::Selection;
use ladde::quantize::Scale;
use ladde::snippets::{InnerLength, Snippets, SnippetsError, snippets, snippets_nonnorm};
use proptest::prelude::*;
use reference::{squared, ticks};
use snippets_pick::assemble;

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

fn scale(x: f64) -> Scale {
    Scale::new(x).unwrap()
}

#[derive(Clone, Copy, Debug)]
enum Pick {
    Percentage(f64),
    Rank(usize),
}

/// The selection `pick` names, or `None` for a percentage outside `[0, 1]`.
fn selection(pick: Pick) -> Option<Selection> {
    match pick {
        Pick::Percentage(p) => Selection::percentage(p),
        Pick::Rank(k) => Some(Selection::rank(k)),
    }
}

/// The nearest `s`-window distances, squared, of a segment inside a window and of the window
/// inside the segment, selected as STUMPY's `_select_P_ABBA_value` does and reported in `t`'s
/// units; a window with a gap has no distance and sorts as infinity.
fn mpdist(
    segment: &[Option<i64>],
    window: &[Option<i64>],
    s: usize,
    pick: Pick,
    scale: Scale,
) -> f64 {
    let m = segment.len();
    let j = m - s + 1;
    let sub = |w: &[Option<i64>], at: usize| -> Option<Vec<i64>> {
        w[at..at + s].iter().copied().collect()
    };
    let near = |from: &[Option<i64>], to: &[Option<i64>], a: usize| -> Option<u128> {
        let own = sub(from, a)?;
        (0..j)
            .filter_map(|b| sub(to, b).map(|other| squared(&own, &other)))
            .min()
    };
    let mut abba: Vec<u128> = (0..j).filter_map(|a| near(segment, window, a)).collect();
    abba.extend((0..j).filter_map(|b| near(window, segment, b)));
    abba.sort();
    let position = match pick {
        Pick::Percentage(share) => (share * (2 * m) as f64).ceil() as usize,
        Pick::Rank(k) => k,
    };
    let chosen = if position < abba.len() {
        Some(abba[position])
    } else {
        abba.last().copied()
    };
    chosen.map_or(f64::INFINITY, |sq| reference::distance(sq, scale))
}

/// STUMPY's `aampdist_snippets`, one window at a time.
fn expected(
    t: &[f64],
    m: usize,
    k: usize,
    scale: Scale,
    inner: InnerLength,
    pick: Pick,
) -> Snippets {
    let s = match inner {
        InnerLength::Share(share) => (share * m as f64).ceil() as usize,
        InnerLength::Length(s) => s,
    };
    let ticks = ticks(t, scale);
    let segments = if t.len().is_multiple_of(m) {
        t.len() / m - 1
    } else {
        t.len() / m
    };
    let len = t.len() - m + 1;
    let profiles: Vec<Vec<f64>> = (0..segments)
        .map(|i| {
            (0..len)
                .map(|p| mpdist(&ticks[i * m..(i + 1) * m], &ticks[p..p + m], s, pick, scale))
                .collect()
        })
        .collect();
    assemble(t, m, k, &profiles)
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|x| x.to_bits()).collect()
}

fn assert_same(got: &Snippets, want: &Snippets, what: &str) {
    assert_eq!(got.indices, want.indices, "{what}");
    assert_eq!(bits(&got.profiles), bits(&want.profiles), "{what}");
    assert_eq!(bits(&got.snippets), bits(&want.snippets), "{what}");
    assert_eq!(bits(&got.areas), bits(&want.areas), "{what}");
    assert_eq!(bits(&got.fractions), bits(&want.fractions), "{what}");
    assert_eq!(got.regimes, want.regimes, "{what}");
    assert_eq!(
        (got.m, got.profile_len),
        (want.m, want.profile_len),
        "{what}"
    );
}

fn run(
    t: &[f64],
    m: usize,
    k: usize,
    scale: Scale,
    threads: usize,
    inner: InnerLength,
    pick: Pick,
) -> Snippets {
    snippets_nonnorm(
        t,
        nz(m),
        nz(k),
        scale,
        nz(threads),
        inner,
        selection(pick).unwrap(),
    )
    .unwrap()
}

fn inners(m: usize) -> Vec<InnerLength> {
    vec![
        InnerLength::Share(1.0),
        InnerLength::Share(0.5),
        InnerLength::Length(m.div_ceil(2)),
        InnerLength::Length(m - 1),
        InnerLength::Length(m + 3),
    ]
}

fn picks() -> Vec<Pick> {
    vec![
        Pick::Percentage(0.05),
        Pick::Percentage(0.5),
        Pick::Percentage(1.0),
        Pick::Percentage(-1.0),
        Pick::Rank(0),
        Pick::Rank(3),
        Pick::Rank(usize::MAX),
    ]
}

fn assert_equals_reference(t: &[f64], m: usize, scale: Scale, ks: &[usize], threads: &[usize]) {
    for inner in inners(m) {
        let s = match inner {
            InnerLength::Share(share) => (share * m as f64).ceil() as usize,
            InnerLength::Length(s) => s,
        };
        if s > m {
            let got = snippets_nonnorm(t, nz(m), nz(1), scale, nz(1), inner, Selection::default());
            assert_eq!(
                got.map(|_| ()),
                Err(SnippetsError::InnerWindowAboveSegment),
                "{inner:?}"
            );
            continue;
        }
        if s < 3 {
            continue;
        }
        for pick in picks() {
            // A share outside [0, 1] is no selection at all.
            let Some(chosen) = selection(pick) else {
                continue;
            };
            if !chosen.fits(2 * (m - s + 1)) {
                let got = snippets_nonnorm(t, nz(m), nz(1), scale, nz(1), inner, chosen);
                assert_eq!(
                    got.map(|_| ()),
                    Err(SnippetsError::RankPastLast),
                    "{pick:?}"
                );
                continue;
            }
            for &k in ks {
                let want = expected(t, m, k, scale, inner, pick);
                for &threads in threads {
                    let got = run(t, m, k, scale, threads, inner, pick);
                    assert_same(
                        &got,
                        &want,
                        &format!("{inner:?} {pick:?} k {k} m {m} threads {threads}"),
                    );
                }
            }
        }
    }
}

proptest! {
    #![proptest_config(testkit::generators::seeded(30))]

    #[test]
    fn random_series_equal_the_reference(
        m in 3usize..=5,
        t in prop::collection::vec(
            prop_oneof![
                1 => Just(f64::NAN),
                20 => (0u8..9).prop_map(|x| f64::from(x) * 0.25),
            ],
            10..40,
        ),
    ) {
        prop_assume!(t.len() >= 2 * m);
        assert_equals_reference(&t, m, scale(0.25), &[1, 3], &[1, 3]);
    }

    #[test]
    fn wide_levels_equal_the_reference(
        m in 3usize..=4,
        t in prop::collection::vec(
            prop_oneof![Just(0.0), Just(1.0e9), Just(-3.0e9), Just(1.0e12), (0u8..4).prop_map(f64::from)],
            8..30,
        ),
    ) {
        prop_assume!(t.len() >= 2 * m);
        assert_equals_reference(&t, m, scale(1.0), &[2], &[2]);
    }
}

fn period(n: usize, shift: usize, cycle: &[f64]) -> Vec<f64> {
    (0..n).map(|i| cycle[(i + shift) % cycle.len()]).collect()
}

fn few_levels(n: usize, levels: u64, mut state: u64) -> Vec<f64> {
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) % levels) as f64
        })
        .collect()
}

#[test]
fn tie_heavy_series_equal_the_reference() {
    let one = scale(1.0);
    let series = vec![
        vec![3.0; 37],
        [vec![1.0; 12], vec![5.0; 12], vec![1.0; 14]].concat(),
        period(40, 0, &[3.0, 1.0]),
        period(44, 1, &[3.0, 1.0, 4.0, 1.0]),
        period(50, 0, &[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]),
        few_levels(45, 2, 7),
        few_levels(47, 3, 11),
        [
            period(20, 0, &[2.0, 7.0, 1.0]),
            period(20, 0, &[2.0, 7.0, 1.0]),
        ]
        .concat(),
    ];
    for t in &series {
        for m in [3, 4, 7] {
            assert_equals_reference(t, m, one, &[1, 2, 5], &[1, 2]);
        }
    }
}

#[test]
fn gaps_and_infinite_samples_at_the_start_the_end_and_between_equal_the_reference() {
    let one = scale(1.0);
    let mut state = 99u64;
    let walk: Vec<f64> = {
        let mut x = 0.0;
        (0..52)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                x += ((state >> 58) as i64 % 5 - 2) as f64;
                x
            })
            .collect()
    };
    for (lead, trail, every) in [(0, 0, 9), (5, 0, 0), (0, 6, 0), (4, 4, 11), (0, 0, 0)] {
        let mut t = walk.clone();
        for k in (0..t.len()).filter(|k| every != 0 && k % every == every - 1) {
            t[k] = f64::NAN;
        }
        t[..lead].fill(f64::NAN);
        let n = t.len();
        t[n - trail..].fill(f64::INFINITY);
        if every == 11 {
            t[20] = f64::NEG_INFINITY;
        }
        for m in [3, 5] {
            assert_equals_reference(&t, m, one, &[2], &[1, 2]);
        }
    }
}

#[test]
fn the_last_whole_segment_is_dropped_when_m_divides_the_length_as_stumpy_does() {
    let one = scale(1.0);
    let inner = InnerLength::Share(1.0);
    for (n, segments) in [(20, 1), (30, 2), (25, 2), (29, 2), (31, 3)] {
        let t = few_levels(n, 9, n as u64);
        let got = run(&t, 10, 8, one, 1, inner, Pick::Percentage(0.05));
        let picked: std::collections::BTreeSet<usize> = got.indices.iter().copied().collect();
        assert!(
            picked.iter().all(|&i| i / 10 < segments),
            "n {n}: {picked:?}"
        );
        assert_eq!(picked.len(), segments.min(8), "n {n}: {picked:?}");
        let z = snippets(&t, nz(10), nz(8), one, nz(1), inner, Selection::default()).unwrap();
        let z_segments: std::collections::BTreeSet<usize> = z.indices.iter().copied().collect();
        assert!(z_segments.len() <= n / 10);
    }
}

#[test]
fn the_two_metrics_give_different_profiles() {
    let t = few_levels(60, 7, 3);
    let one = scale(1.0);
    let inner = InnerLength::Share(1.0);
    let raw = run(&t, 10, 2, one, 1, inner, Pick::Percentage(0.05));
    let normalized = snippets(&t, nz(10), nz(2), one, nz(1), inner, Selection::default()).unwrap();
    assert_ne!(bits(&raw.profiles), bits(&normalized.profiles));
}

#[test]
fn distances_are_in_the_callers_units() {
    let t = few_levels(60, 7, 3);
    let inner = InnerLength::Share(1.0);
    let unit = run(&t, 10, 2, scale(1.0), 1, inner, Pick::Percentage(0.05));
    let scaled: Vec<f64> = t.iter().map(|x| x * 8.0).collect();
    let eight = run(&scaled, 10, 2, scale(8.0), 1, inner, Pick::Percentage(0.05));
    assert_eq!(unit.indices, eight.indices);
    for (a, b) in unit.profiles.iter().zip(&eight.profiles) {
        assert_eq!((a * 8.0).to_bits(), b.to_bits());
    }
}

#[test]
fn every_thread_count_gives_the_same_bits_on_a_longer_series() {
    let t = few_levels(900, 40, 5);
    let one = scale(1.0);
    let inner = InnerLength::Share(0.5);
    let first = run(&t, 30, 4, one, 1, inner, Pick::Percentage(0.05));
    assert!(first.profiles.iter().any(|x| x.is_finite()));
    for threads in [2, 3, 8, 64] {
        let got = run(&t, 30, 4, one, threads, inner, Pick::Percentage(0.05));
        assert_same(&got, &first, &format!("threads {threads}"));
    }
}

#[test]
fn a_tick_range_needing_more_than_128_bits_has_no_distance_and_never_panics() {
    let t: Vec<f64> = (0..40)
        .map(|i| if i % 2 == 0 { 0.0 } else { 9.0e15 })
        .collect();
    let got = run(
        &t,
        3,
        2,
        scale(1.0),
        1,
        InnerLength::Share(1.0),
        Pick::Rank(0),
    );
    assert!(got.profiles.iter().all(|x| x.is_finite()));
    let m: usize = 1 << 21;
    let huge: Vec<f64> = (0..2 * m + 3)
        .map(|i| if i % 2 == 0 { -9.0e15 } else { 9.0e15 })
        .collect();
    let got = run(
        &huge,
        m,
        1,
        scale(1.0),
        1,
        InnerLength::Length(m),
        Pick::Rank(0),
    );
    assert!(got.profiles.iter().all(|x| *x == f64::INFINITY));
}

#[test]
fn arguments_without_an_answer_are_errors_never_panics() {
    let t = few_levels(30, 5, 3);
    let one = scale(1.0);
    let call = |t: &[f64], m: usize, k: usize, inner: InnerLength| {
        snippets_nonnorm(t, nz(m), nz(k), one, nz(1), inner, Selection::default()).map(|_| ())
    };
    let whole = InnerLength::default();
    assert_eq!(call(&t, 16, 1, whole), Err(SnippetsError::SeriesTooShort));
    assert_eq!(call(&[], 1, 1, whole), Err(SnippetsError::SeriesTooShort));
    assert_eq!(
        call(&t, 5, 1, InnerLength::Length(0)),
        Err(SnippetsError::EmptyInnerWindow)
    );
    assert_eq!(
        call(&t, 5, 1, InnerLength::Share(f64::NAN)),
        Err(SnippetsError::NotANumber)
    );
    assert_eq!(
        call(&t, 5, 1, InnerLength::Length(2)),
        Err(SnippetsError::InnerWindowTooShort)
    );
    assert_eq!(call(&t, 15, 1, whole), Ok(()));
    assert_eq!(call(&t, 5, 1 << 12, whole), Ok(()));
    assert_eq!(call(&t, 5, usize::MAX, whole), Err(SnippetsError::Capacity));
    assert_eq!(
        call(&t, 5, 1, InnerLength::Length(usize::MAX)),
        Err(SnippetsError::InnerWindowAboveSegment)
    );
    assert_eq!(
        call(&t, 5, 1, InnerLength::Share(f64::INFINITY)),
        Err(SnippetsError::ShareOutOfRange)
    );
    let rank = |k: usize, p: f64| {
        let selection = if p.is_nan() {
            Selection::rank(k)
        } else {
            Selection::percentage(p).unwrap()
        };
        snippets_nonnorm(&t, nz(5), nz(2), one, nz(1), whole, selection).map(|_| ())
    };
    assert_eq!(rank(usize::MAX, f64::NAN), Err(SnippetsError::RankPastLast));
    assert_eq!(rank(1, f64::NAN), Ok(()));
    assert_eq!(rank(2, f64::NAN), Err(SnippetsError::RankPastLast));
    assert_eq!(rank(0, 1.0), Ok(()));
    assert!(Selection::percentage(f64::INFINITY).is_none());
}
