#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]

//! Snippets equal a plain statement of STUMPY's procedure: every segment's profile built window
//! by window from the brute-force MPdist (exact rational orders), the greedy pick over areas
//! summed pairwise, the fractions and the regimes; on random series with gaps, repeats and flat
//! runs, tie-heavy series, wide levels, every inner length and selection extreme, every thread
//! count.

#[path = "common/exact_pair.rs"]
mod exact_pair;
#[path = "common/mpdist_brute.rs"]
mod mpdist_brute;
#[path = "common/route.rs"]
mod route;
#[path = "common/snippets_pick.rs"]
mod snippets_pick;

use ladde::mpdist::Selection;
use ladde::naive::Ticks;
use ladde::quantize::{Scale, to_ticks};
use ladde::snippets::{InnerLength, Snippets, SnippetsError, snippets};
use ladde::stats::Series;
use mpdist_brute::{Pick, brute};
use proptest::prelude::*;
use route::{jumpy, nz};
use snippets_pick::assemble;
use testkit::generators::{MARKET_TICK, market_series};

fn scale(x: f64) -> Scale {
    Scale::new(x).unwrap()
}

/// The selection `pick` names, or `None` for a percentage outside `[0, 1]`.
fn selection(pick: Pick) -> Option<Selection> {
    match pick {
        Pick::Percentage(p) => Selection::percentage(p),
        Pick::Rank(k) => Some(Selection::rank(k)),
    }
}

/// STUMPY's `_get_all_profiles` and `snippets`, one window at a time.
fn reference(
    t: &[f64],
    m: usize,
    k: usize,
    scale: Scale,
    inner: InnerLength,
    pick: Pick,
) -> Snippets {
    reference_flagged(t, m, k, scale, inner, pick, None)
}

/// [`reference`] with the caller's flag for each window of the inner length being constant (one
/// per start in `t`), sliced to the windows of each segment and of each compared window as
/// STUMPY slices `T_subseq_isconstant`.
fn reference_flagged(
    t: &[f64],
    m: usize,
    k: usize,
    scale: Scale,
    inner: InnerLength,
    pick: Pick,
    flags: Option<&[bool]>,
) -> Snippets {
    let s = match inner {
        InnerLength::Share(share) => (share * m as f64).ceil() as usize,
        InnerLength::Length(s) => s,
    };
    let ticks: Vec<Option<i64>> = t.iter().map(|&x| to_ticks(x, scale)).collect();
    let segments = t.len() / m;
    let len = t.len() - m + 1;
    let profiles: Vec<Vec<f64>> = (0..segments)
        .map(|i| {
            (0..len)
                .map(|p| {
                    let window = |start: usize| {
                        let ticks = &ticks[start..start + m];
                        match flags {
                            Some(flags) => Ticks::with_constant_windows(
                                ticks,
                                &flags[start..start + m + 1 - s],
                            ),
                            None => Ticks::from(ticks),
                        }
                    };
                    brute(
                        (&t[i * m..(i + 1) * m], window(i * m)),
                        (&t[p..p + m], window(p)),
                        s,
                        pick,
                    )
                    .unwrap()
                })
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
    snippets(
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
        InnerLength::Share(0.0001),
        InnerLength::Length(1),
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

/// The library equal to the reference at every inner length, pick and `k` in `ks`.
fn assert_equals_reference(t: &[f64], m: usize, scale: Scale, ks: &[usize], threads: &[usize]) {
    for inner in inners(m) {
        let s = match inner {
            InnerLength::Share(share) => (share * m as f64).ceil() as usize,
            InnerLength::Length(s) => s,
        };
        if s > m {
            let got = snippets(t, nz(m), nz(1), scale, nz(1), inner, Selection::default());
            assert_eq!(
                got.map(|_| ()),
                Err(SnippetsError::InnerWindowAboveSegment),
                "{inner:?}"
            );
            continue;
        }
        if s < 3 {
            let got = snippets(t, nz(m), nz(1), scale, nz(1), inner, Selection::default());
            assert_eq!(
                got.map(|_| ()),
                Err(SnippetsError::InnerWindowTooShort),
                "{inner:?}"
            );
            continue;
        }
        for pick in picks() {
            // A share outside [0, 1] is no selection at all.
            let Some(chosen) = selection(pick) else {
                continue;
            };
            if !chosen.fits(2 * (m - s + 1)) {
                let got = snippets(t, nz(m), nz(1), scale, nz(1), inner, chosen);
                assert_eq!(
                    got.map(|_| ()),
                    Err(SnippetsError::RankPastLast),
                    "{pick:?}"
                );
                continue;
            }
            for &k in ks {
                let want = reference(t, m, k, scale, inner, pick);
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
    #![proptest_config(testkit::generators::seeded_default())]
    #[test]
    fn random_series_equal_the_reference(
        m in 3usize..=5,
        t in (3usize..=5).prop_flat_map(|m| market_series(2 * m..40, m)),
        tick in prop_oneof![Just(MARKET_TICK), Just(0.5), Just(1.0)],
    ) {
        prop_assume!(t.len() >= 2 * m);
        assert_equals_reference(&t, m, scale(tick), &[1, 3], &[1, 3]);
    }

    #[test]
    fn wide_levels_equal_the_reference(
        (t, m, one) in (3usize..=4).prop_flat_map(jumpy),
    ) {
        prop_assume!(t.len() >= 2 * m);
        assert_equals_reference(&t, m, one, &[2], &[2]);
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
fn gaps_at_the_start_the_end_and_between_equal_the_reference() {
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
        for m in [3, 5] {
            assert_equals_reference(&t, m, one, &[2], &[1, 2]);
        }
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
fn k_beyond_the_segment_count_picks_a_segment_again_as_stumpy_does() {
    let t = few_levels(30, 5, 3);
    let one = scale(1.0);
    let got = run(
        &t,
        10,
        7,
        one,
        1,
        InnerLength::Share(1.0),
        Pick::Percentage(0.05),
    );
    assert_eq!(got.indices.len(), 7);
    assert_same(
        &got,
        &reference(
            &t,
            10,
            7,
            one,
            InnerLength::Share(1.0),
            Pick::Percentage(0.05),
        ),
        "k 7 of 3 segments",
    );
}

#[test]
fn arguments_without_an_answer_are_errors_never_panics() {
    let t = few_levels(30, 5, 3);
    let one = scale(1.0);
    let call = |t: &[f64], m: usize, inner: InnerLength| {
        snippets(t, nz(m), nz(1), one, nz(1), inner, Selection::default()).map(|_| ())
    };
    assert_eq!(
        call(&t, 16, InnerLength::default()),
        Err(SnippetsError::SeriesTooShort)
    );
    assert_eq!(
        call(&t[..9], 5, InnerLength::default()),
        Err(SnippetsError::SeriesTooShort)
    );
    assert_eq!(
        call(&[], 1, InnerLength::default()),
        Err(SnippetsError::SeriesTooShort)
    );
    assert_eq!(
        call(&t, 5, InnerLength::Length(0)),
        Err(SnippetsError::EmptyInnerWindow)
    );
    assert_eq!(
        call(&t, 5, InnerLength::Share(0.0)),
        Err(SnippetsError::EmptyInnerWindow)
    );
    assert_eq!(
        call(&t, 5, InnerLength::Share(f64::NAN)),
        Err(SnippetsError::NotANumber)
    );
    assert_eq!(call(&t, 15, InnerLength::default()), Ok(()));
    assert_eq!(
        snippets(
            &t,
            nz(5),
            nz(usize::MAX),
            one,
            nz(1),
            InnerLength::default(),
            Selection::default()
        )
        .map(|_| ()),
        Err(SnippetsError::Capacity)
    );
    assert_eq!(
        call(&t, 5, InnerLength::Length(usize::MAX)),
        Err(SnippetsError::InnerWindowAboveSegment)
    );
    assert_eq!(
        call(&t, 5, InnerLength::Length(6)),
        Err(SnippetsError::InnerWindowAboveSegment)
    );
    assert_eq!(call(&t, 5, InnerLength::Length(5)), Ok(()));
    assert_eq!(
        call(&t, 5, InnerLength::Share(f64::INFINITY)),
        Err(SnippetsError::ShareOutOfRange)
    );
    assert_eq!(
        call(&t, 5, InnerLength::Share(1.0 + f64::EPSILON)),
        Err(SnippetsError::ShareOutOfRange)
    );
    assert_eq!(
        call(&t, 5, InnerLength::Share(-0.5)),
        Err(SnippetsError::ShareOutOfRange)
    );
    assert_eq!(call(&t, 5, InnerLength::Share(1.0)), Ok(()));
    assert_eq!(call(&t, 3, InnerLength::default()), Ok(()));
    assert_eq!(
        call(&t, 5, InnerLength::Length(2)),
        Err(SnippetsError::InnerWindowTooShort)
    );
    assert_eq!(call(&t, 5, InnerLength::Length(3)), Ok(()));
    assert_eq!(
        call(&t, 2, InnerLength::default()),
        Err(SnippetsError::InnerWindowTooShort)
    );
}

#[test]
fn flagged_windows_equal_the_reference() {
    let one = scale(1.0);
    let mut state = 0x3C6E_F372_FE94_F82Bu64;
    let mut next = |every: u64| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) % every
    };
    let mut series: Vec<Vec<f64>> = vec![
        (0..37).map(|_| next(3) as f64).collect(),
        (0..32).map(|_| next(2) as f64).collect(),
        (0..41).map(|_| next(6) as f64).collect(),
    ];
    for x in &mut series[0][8..17] {
        *x = 1.0;
    }
    series[2][20] = f64::NAN;
    for t in &series {
        for m in [8usize, 10] {
            for inner in [
                InnerLength::Share(1.0),
                InnerLength::Length(5),
                InnerLength::Share(0.6),
            ] {
                let s = match inner {
                    InnerLength::Share(share) => ((share * m as f64).ceil() as usize).min(m),
                    InnerLength::Length(s) => s.min(m),
                };
                for which in 0..4 {
                    let flags: Vec<bool> = (0..t.len() + 1 - s)
                        .map(|i| match which {
                            0 => t[i..i + s].iter().all(|x| *x == t[i]),
                            1 => false,
                            2 => true,
                            _ => next(3) == 0,
                        })
                        .collect();
                    for pick in [Pick::Percentage(0.05), Pick::Rank(1), Pick::Percentage(1.0)] {
                        let want = reference_flagged(t, m, 2, one, inner, pick, Some(&flags));
                        let flagged = Series::with_constant_windows(t, nz(s), &flags).unwrap();
                        for threads in [1, 3] {
                            let got = snippets(
                                flagged,
                                nz(m),
                                nz(2),
                                one,
                                nz(threads),
                                inner,
                                selection(pick).unwrap(),
                            )
                            .unwrap();
                            assert_same(
                                &got,
                                &want,
                                &format!("m {m} {inner:?} flags {which} {pick:?}"),
                            );
                        }
                    }
                }
            }
        }
    }
}
