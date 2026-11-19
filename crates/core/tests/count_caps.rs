//! A caller's count (k, n_regimes, max_matches, max_motifs, a width) never sizes an allocation
//! beyond what the input can produce: a huge count returns the result of a count equal to the
//! number of windows, and counts within range are unchanged.

use core::num::NonZeroUsize;

use ladde::batch::{ab_join_topk, discords, fluss, left_topk_profile, two_sided_topk_profile};
use ladde::discord::DiscordTracker;
use ladde::entry::Entry;
use ladde::floss::{Floss, regime_locations};
use ladde::index::SampleIndex;
use ladde::motif::{MotifParams, motifs};
use ladde::nearest::Nearest;
use ladde::pattern::{MaxDistance, find_matches};
use ladde::quantize::Scale;
use ladde::stats::WindowStats;
use proptest::prelude::*;

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

fn scale() -> Scale {
    Scale::new(1e-3).unwrap()
}

fn series(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) % 7) as f64
        })
        .collect()
}

fn shown<T: std::fmt::Debug + ?Sized>(value: &T) -> String {
    format!("{value:?}")
}

const HUGE: [usize; 3] = [4_278_255_360, 2_330_271_378, usize::MAX];

#[test]
fn left_topk_profile_with_a_huge_k_equals_k_at_the_window_count() {
    let s = series(40, 1);
    let windows = s.len() - 5 + 1;
    let at_windows = left_topk_profile(&s, nz(5), scale(), nz(1), nz(windows));
    for k in HUGE {
        let got = left_topk_profile(&s, nz(5), scale(), nz(2), nz(k));
        assert_eq!(shown(&got), shown(&at_windows));
    }
}

#[test]
fn two_sided_topk_profile_with_a_huge_k_equals_k_at_the_window_count() {
    let s = series(40, 1);
    let windows = s.len() - 5 + 1;
    let at_windows = two_sided_topk_profile(&s, nz(5), scale(), nz(1), nz(windows)).unwrap();
    for k in HUGE {
        let got = two_sided_topk_profile(&s, nz(5), scale(), nz(2), nz(k)).unwrap();
        assert_eq!(shown(&got), shown(&at_windows));
    }
    for k in HUGE {
        assert!(
            two_sided_topk_profile(&[1.0, 2.0], nz(5), scale(), nz(1), nz(k))
                .unwrap()
                .is_empty()
        );
    }
}

#[test]
fn ab_join_topk_with_a_huge_k_equals_k_at_the_window_count() {
    let (a, b) = (series(40, 1), series(30, 2));
    let at_windows = ab_join_topk(&a, &b, nz(5), scale(), scale(), nz(1), nz(b.len() - 4)).unwrap();
    for k in HUGE {
        let got = ab_join_topk(&a, &b, nz(5), scale(), scale(), nz(2), nz(k)).unwrap();
        assert_eq!(shown(&got), shown(&at_windows));
        assert!(
            ab_join_topk(&[1.0, 2.0], &b, nz(5), scale(), scale(), nz(1), nz(k))
                .unwrap()
                .is_empty()
        );
        let short = ab_join_topk(&a, &[1.0, 2.0], nz(5), scale(), scale(), nz(1), nz(k)).unwrap();
        assert!(short.iter().all(|(_, entries)| entries.is_empty()));
    }
}

/// Gaps leave few or no finite windows: a huge `k` still equals `k` at the window count, and
/// neither side nor the join has anything to keep for a window with no finite neighbor.
#[test]
fn huge_k_on_gap_series_equals_k_at_the_window_count() {
    let all_gaps = vec![f64::NAN; 60];
    let mut mostly_gaps = series(60, 3);
    for (i, x) in mostly_gaps.iter_mut().enumerate() {
        if i % 30 >= 8 {
            *x = f64::NAN;
        }
    }
    let windows = 60 - 5 + 1;
    for s in [&all_gaps, &mostly_gaps] {
        let at_windows = two_sided_topk_profile(s, nz(5), scale(), nz(1), nz(windows)).unwrap();
        let join_at = ab_join_topk(s, s, nz(5), scale(), scale(), nz(1), nz(windows)).unwrap();
        for k in HUGE {
            let got = two_sided_topk_profile(s, nz(5), scale(), nz(2), nz(k)).unwrap();
            assert_eq!(shown(&got), shown(&at_windows));
            let got = ab_join_topk(s, s, nz(5), scale(), scale(), nz(2), nz(k)).unwrap();
            assert_eq!(shown(&got), shown(&join_at));
        }
    }
    let rows = two_sided_topk_profile(&all_gaps, nz(5), scale(), nz(1), nz(usize::MAX)).unwrap();
    assert_eq!(rows.len(), windows);
    assert!(rows.iter().all(|(_, sides)| sides.nearest.is_empty()));
}

#[test]
fn left_topk_profile_with_a_huge_k_on_a_series_without_windows_is_empty() {
    for k in HUGE {
        assert!(left_topk_profile(&[1.0, 2.0], nz(5), scale(), nz(1), nz(k)).is_empty());
        assert!(left_topk_profile(&[], nz(5), scale(), nz(1), nz(k)).is_empty());
    }
}

#[test]
fn discords_with_a_huge_k_on_an_empty_series_is_empty() {
    for k in HUGE {
        assert!(discords(&[], nz(5), scale(), nz(1), k).is_empty());
        assert!(discords(&[1.0, 2.0], nz(5), scale(), nz(1), k).is_empty());
    }
}

#[test]
fn discords_with_a_huge_k_equals_k_at_the_window_count() {
    let s = series(60, 2);
    let windows = s.len() - 6 + 1;
    let at_windows = discords(&s, nz(6), scale(), nz(1), windows);
    for k in HUGE {
        assert_eq!(discords(&s, nz(6), scale(), nz(2), k), at_windows);
    }
}

#[test]
fn fluss_with_a_huge_n_regimes_equals_the_window_count() {
    let s = series(80, 3);
    let windows = s.len() - 6 + 1;
    let at_windows = fluss(&s, nz(6), scale(), nz(1), 3, windows + 1, 2, None).unwrap();
    for n_regimes in HUGE {
        let got = fluss(&s, nz(6), scale(), nz(1), 3, n_regimes, 2, None).unwrap();
        assert_eq!(got.1, at_windows.1);
        assert_eq!(shown(&got.0), shown(&at_windows.0));
    }
}

#[test]
fn regime_locations_with_a_huge_count_is_bounded_by_the_curve() {
    let curve = [0.5, 0.2, 0.9, 0.1, 0.7];
    for n_regimes in HUGE {
        assert_eq!(
            regime_locations(&curve, n_regimes, 1, 1),
            regime_locations(&curve, curve.len() + 1, 1, 1)
        );
        assert!(regime_locations(&[], n_regimes, 1, 1).is_empty());
    }
}

#[test]
fn a_discord_tracker_with_a_huge_k_lists_what_the_scores_allow() {
    let s = series(60, 4);
    let m = nz(6);
    let windows = s.len() - 6 + 1;
    let mut capped = DiscordTracker::new(m, windows);
    let mut huge = DiscordTracker::new(m, usize::MAX);
    let mut windowed = DiscordTracker::windowed(m, windows, nz(40)).unwrap();
    let mut windowed_huge = DiscordTracker::windowed(m, usize::MAX, nz(40)).unwrap();
    let mut at_horizon = DiscordTracker::windowed(m, 40 - 6 + 1, nz(40)).unwrap();
    let mut stream = ladde::stream::GrowingStream::new(m, scale());
    for &x in &s {
        if let Some((row, score)) = stream.update_exact(x) {
            capped.update(row, score);
            huge.update(row, score);
            windowed.update(row, score);
            windowed_huge.update(row, score);
            at_horizon.update(row, score);
        }
    }
    assert_eq!(huge.top(), capped.top());
    assert_eq!(windowed_huge.top(), windowed.top());
    assert_eq!(windowed_huge.top(), at_horizon.top());
}

#[test]
fn nearest_with_a_huge_k_keeps_every_offered_candidate() {
    let mut stats = WindowStats::new(nz(3));
    let row = [1, 2, 9]
        .into_iter()
        .filter_map(|tick| stats.push(tick, true))
        .last()
        .unwrap();
    for k in HUGE {
        assert!(
            Nearest::with_k(&row, 3, nz(k), usize::MAX)
                .finish_k()
                .is_empty()
        );
    }
}

#[test]
fn floss_with_a_huge_width_runs() {
    let mut floss = Floss::new(nz(4), nz(usize::MAX), usize::MAX, 3, 2);
    for row in 0..20u64 {
        floss.update(SampleIndex::new(row).unwrap(), Entry::NONE);
    }
    assert_eq!(floss.curve().len(), 20);
}

#[test]
fn motifs_and_matches_with_huge_counts_equal_unbounded_ones() {
    let s = series(80, 5);
    let params = |max_matches, max_motifs, min_neighbors| MotifParams {
        min_neighbors,
        max_distance: MaxDistance::Default,
        cutoff: None,
        max_matches,
        max_motifs,
        ..MotifParams::default()
    };
    let windows = s.len() - 6 + 1;
    let tight = motifs(
        &s,
        nz(6),
        scale(),
        nz(1),
        &params(Some(windows), windows, 1),
    );
    let huge = motifs(
        &s,
        nz(6),
        scale(),
        nz(2),
        &params(Some(usize::MAX), usize::MAX, 1),
    );
    assert_eq!(shown(&huge), shown(&tight));
    let none = motifs(&s, nz(6), scale(), nz(1), &params(None, 1, usize::MAX));
    assert!(none.is_empty());
    let query = &s[10..16];
    let all = find_matches(query, &s, scale(), MaxDistance::Default, None).unwrap();
    let huge = find_matches(query, &s, scale(), MaxDistance::Default, Some(usize::MAX)).unwrap();
    assert_eq!(shown(&huge), shown(&all));
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    #[test]
    fn counts_up_to_the_window_count_are_unchanged_and_beyond_it_equal_it(
        n in 8usize..60, seed in 0u64..1000, extra in 0usize..50,
    ) {
        let m = 5;
        let s = series(n, seed);
        let windows = n - m + 1;
        let exact = |k: usize| left_topk_profile(&s, nz(m), scale(), nz(1), nz(k));
        let beyond = exact(windows + extra + 1);
        prop_assert_eq!(shown(&beyond), shown(&exact(windows)));
        let within = 1 + extra % windows;
        let reference = exact(within);
        for (row, kept) in &reference {
            prop_assert!(kept.len() <= within, "row {row}");
            prop_assert_eq!(
                shown(kept),
                shown(&beyond.iter().find(|(r, _)| r == row).unwrap().1[..kept.len()])
            );
        }
        let picks = discords(&s, nz(m), scale(), nz(1), windows);
        prop_assert_eq!(discords(&s, nz(m), scale(), nz(1), windows + extra + 1), picks.clone());
        prop_assert_eq!(
            &discords(&s, nz(m), scale(), nz(1), within)[..],
            &picks[..picks.len().min(within)]
        );
    }
}

/// STUMPY's regime extraction written out: `n_regimes - 1` times the first minimum, then the
/// cuts on either side of it set to 1.
fn reference_regimes(curve: &[f64], n_regimes: usize, l: usize, excl_factor: usize) -> Vec<usize> {
    let reach = l * excl_factor;
    let mut left = curve.to_vec();
    let mut found = Vec::new();
    for _ in 0..n_regimes - 1 {
        let mut at = 0;
        for (i, &v) in left.iter().enumerate() {
            if v < left[at] {
                at = i;
            }
        }
        found.push(at);
        for v in &mut left[at.saturating_sub(reach)..(at + reach).min(curve.len())] {
            *v = 1.0;
        }
    }
    found
}

#[test]
fn regime_locations_return_one_per_window_at_most_and_follow_the_reference_below_it() {
    for seed in 0..40u64 {
        let curve: Vec<f64> = series(30, seed).iter().map(|v| v / 7.0).collect();
        let windows = curve.len();
        for (l, excl_factor) in [(1, 1), (2, 1), (1, 3), (3, 0)] {
            for n_regimes in 1..=windows {
                assert_eq!(
                    regime_locations(&curve, n_regimes, l, excl_factor),
                    reference_regimes(&curve, n_regimes, l, excl_factor),
                    "seed {seed} l {l} excl {excl_factor} n {n_regimes}"
                );
            }
            for n_regimes in [windows + 1, windows + 2, 1000, usize::MAX] {
                let got = regime_locations(&curve, n_regimes, l, excl_factor);
                assert_eq!(got.len(), windows, "n {n_regimes}");
                assert_eq!(got, reference_regimes(&curve, windows + 1, l, excl_factor));
            }
        }
    }
}
