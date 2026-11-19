#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]

//! The non-normalized derived procedures (matching, motifs and snippets) give the same bits on every forced
//! route, and the wide route fires in the pair arithmetic exactly when it is forced.

use core::num::NonZeroUsize;

#[path = "common/nonnorm_derived.rs"]
mod reference;

use ladde::motif::{MotifParams, motifs_nonnorm};
use ladde::mpdist::{Selection, aampdist};
use ladde::ostinato::aamp_ostinato;
use ladde::pattern::{MaxDistance, distance_profile_nonnorm, find_matches_nonnorm};
use ladde::quantize::Scale;
use ladde::route::{RouteForce, forced_route};
use ladde::snippets::{InnerLength, snippets_nonnorm};
use reference::{Group, Pick, brute_aamp_ostinato, brute_aampdist, brute_motifs, profile, ticks};

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

fn unit() -> Scale {
    Scale::new(1.0).unwrap()
}

fn wide() -> RouteForce {
    RouteForce {
        wide: true,
        ..RouteForce::default()
    }
}

/// A walk of a few ticks around zero with exact repeats: no pair needs more than `u64`.
fn calm_walk() -> Vec<f64> {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut price = 0i64;
    let mut series: Vec<f64> = (0..90)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            price += ((state >> 33) % 7) as i64 - 3;
            price as f64
        })
        .collect();
    let (head, tail) = series.split_at_mut(60);
    tail[..12].copy_from_slice(&head[10..22]);
    series
}

fn as_group(matches: &[(u64, ladde::nonnorm::NonNormNeighbor)]) -> Group {
    matches
        .iter()
        .map(|&(s, h)| (s, h.squared_ticks()))
        .collect()
}

#[test]
fn matches_and_motifs_equal_the_reference_on_every_route() {
    let series = calm_walk();
    let m = 12;
    let query = series[10..10 + m].to_vec();
    let exact = profile(
        &ticks(&query, unit())
            .into_iter()
            .map(Option::unwrap)
            .collect::<Vec<_>>(),
        &ticks(&series, unit()),
    );
    let params = MotifParams {
        max_motifs: 3,
        max_matches: None,
        cutoff: Some(f64::INFINITY),
        max_distance: MaxDistance::Value(60.0),
        ..MotifParams::default()
    };
    let want_motifs = brute_motifs(&series, m, unit(), &params);
    assert!(!want_motifs.is_empty());
    for force in [RouteForce::default(), wide()] {
        let _forced = forced_route(force);
        let hits = distance_profile_nonnorm(&query, &series, unit()).unwrap();
        for (hit, want) in hits.iter().zip(&exact) {
            assert_eq!(hit.index().get().is_some(), want.is_some(), "{force:?}");
            if let Some(sq) = want {
                assert_eq!(hit.squared_ticks(), *sq, "{force:?}");
            }
        }
        let matches =
            find_matches_nonnorm(&query, &series, unit(), MaxDistance::Value(60.0), None).unwrap();
        assert_eq!(
            as_group(&matches),
            reference::brute_matches(&exact, m, unit(), MaxDistance::Value(60.0), None, None),
            "{force:?}"
        );
        let found: Vec<Group> = motifs_nonnorm(&series, nz(m), unit(), nz(2), &params)
            .iter()
            .map(|g| as_group(g))
            .collect();
        assert_eq!(found, want_motifs, "{force:?}");
    }
}

#[test]
fn the_wide_route_fires_in_matching_only_when_forced() {
    let series = calm_walk();
    let query = series[10..22].to_vec();
    let run = |force: RouteForce| {
        let guard = forced_route(force);
        let _ = find_matches_nonnorm(&query, &series, unit(), MaxDistance::Default, None);
        guard.fired().wide_switches
    };
    assert_eq!(run(RouteForce::default()), 0);
    // One decision per window the query meets.
    assert_eq!(run(wide()), series.len() - query.len() + 1);
}

#[test]
fn the_wide_route_fires_in_motifs_only_when_forced() {
    let series = calm_walk();
    let run = |force: RouteForce| {
        let guard = forced_route(force);
        let _ = motifs_nonnorm(&series, nz(12), unit(), nz(1), &MotifParams::default());
        guard.fired().wide_switches
    };
    assert_eq!(run(RouteForce::default()), 0);
    assert!(run(wide()) > 0);
}

/// A pair spanning more than `u64` runs wide on its own, and the answer is the same as forced.
#[test]
fn a_pair_past_the_u64_bound_runs_wide_without_being_forced() {
    let mut series = calm_walk();
    for x in &mut series[30..42] {
        *x += (1u64 << 40) as f64;
    }
    let query = series[30..42].to_vec();
    let natural = {
        let guard = forced_route(RouteForce::default());
        let hits = distance_profile_nonnorm(&query, &series, unit()).unwrap();
        assert_eq!(guard.fired().wide_switches, 0);
        hits
    };
    let _forced = forced_route(wide());
    let again = distance_profile_nonnorm(&query, &series, unit()).unwrap();
    assert_eq!(natural, again);
    let far = natural[0].squared_ticks();
    assert!(far > u128::from(u64::MAX), "{far}");
}

/// Three series of one walk: two views and a shifted copy of it, so every pair has near windows.
fn calm_set() -> Vec<Vec<f64>> {
    let walk = calm_walk();
    vec![
        walk[..70].to_vec(),
        walk[20..].to_vec(),
        walk[5..80].to_vec(),
    ]
}

#[test]
fn mpdist_and_consensus_equal_the_reference_on_every_route() {
    let set = calm_set();
    let refs: Vec<&[f64]> = set.iter().map(Vec::as_slice).collect();
    let want_consensus = brute_aamp_ostinato(&set, 12, unit()).unwrap();
    assert!(want_consensus.0.is_finite());
    for force in [RouteForce::default(), wide()] {
        let _forced = forced_route(force);
        for (pick, selection) in [
            (Pick::Percentage(0.05), Selection::default()),
            (Pick::Rank(3), Selection::rank(3)),
            (Pick::Percentage(1.0), Selection::percentage(1.0).unwrap()),
        ] {
            let want = brute_aampdist(&set[0], &set[1], 12, unit(), pick).unwrap();
            let got = aampdist(&set[0], &set[1], nz(12), unit(), nz(2), selection).unwrap();
            assert_eq!(got.to_bits(), want.to_bits(), "{force:?} {pick:?}");
        }
        let got = aamp_ostinato(&refs, unit(), nz(12), nz(2)).unwrap();
        assert_eq!(
            (got.radius.to_bits(), got.series, got.window),
            (
                want_consensus.0.to_bits(),
                want_consensus.1,
                want_consensus.2
            ),
            "{force:?}"
        );
    }
}

#[test]
fn the_wide_route_fires_in_mpdist_and_consensus_only_when_forced() {
    let set = calm_set();
    let refs: Vec<&[f64]> = set.iter().map(Vec::as_slice).collect();
    let run = |force: RouteForce| {
        let guard = forced_route(force);
        let _ = aampdist(
            &set[0],
            &set[1],
            nz(12),
            unit(),
            nz(1),
            Selection::default(),
        );
        let after_dist = guard.fired().wide_switches;
        let _ = aamp_ostinato(&refs, unit(), nz(12), nz(1));
        (after_dist, guard.fired().wide_switches)
    };
    assert_eq!(run(RouteForce::default()), (0, 0));
    let (dist, both) = run(wide());
    assert!(dist > 0 && both > dist, "{dist} {both}");
}

#[test]
fn snippets_equal_themselves_on_every_route_and_the_wide_route_fires_only_when_forced() {
    let series = calm_walk();
    let run = |force: RouteForce, threads: usize| {
        let guard = forced_route(force);
        let found = snippets_nonnorm(
            &series,
            nz(12),
            nz(3),
            unit(),
            nz(threads),
            InnerLength::Share(0.5),
            Selection::default(),
        )
        .unwrap();
        (found, guard.fired().wide_switches)
    };
    let (natural, natural_switches) = run(RouteForce::default(), 1);
    assert!(natural.profiles.iter().any(|x| x.is_finite()));
    assert_eq!(natural_switches, 0);
    for threads in [1, 3] {
        let (forced, switches) = run(wide(), threads);
        assert_eq!(forced, natural);
        assert_eq!(switches, 1);
    }
}
