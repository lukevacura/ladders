#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]

//! The diagonal-major walk over a subset of the diagonals equals the exact reference over exactly
//! those pairs, and the tiled pass, on bits, at every thread count; the anytime profiles choose
//! the walk or the tiled pass by run length, equal either way under every forced choice, and the
//! forced choice fires.

#[path = "common/route.rs"]
mod route;

use ladde::batch::{
    ab_join_topk_over_tiled, ab_join_topk_over_walked, two_sided_topk_profile_over_tiled,
    two_sided_topk_profile_over_walked,
};
use ladde::entry::{Entry, TwoSidedTopk};
use ladde::naive::{naive_ab_join_topk_on, naive_two_sided_topk_on};
use ladde::quantize::Scale;
use ladde::route::{RouteForce, SparseForce, forced_route};
use ladde::scrump::{Explicit, Scrump, ScrumpJoin};
use proptest::prelude::*;
use route::{cases, nz, samples};

type EntryBits = (u64, u64, u64);

fn one(e: Entry) -> EntryBits {
    (
        e.rho().to_bits(),
        e.squared_distance().to_bits(),
        e.index().get().unwrap_or(u64::MAX),
    )
}

fn bits(profile: &[(u64, TwoSidedTopk)]) -> Vec<(u64, [EntryBits; 2], Vec<EntryBits>)> {
    profile
        .iter()
        .map(|(row, s)| {
            (
                *row,
                [s.left, s.right].map(one),
                s.nearest.iter().copied().map(one).collect(),
            )
        })
        .collect()
}

fn join_bits(rows: &[(u64, Vec<Entry>)]) -> Vec<(u64, Vec<EntryBits>)> {
    rows.iter()
        .map(|(row, es)| (*row, es.iter().copied().map(one).collect()))
        .collect()
}

/// Diagonal sets of a self-join of `windows` windows at `m`: none, one, scattered, runs.
fn subsets(windows: usize, m: usize, salt: u64) -> Vec<Vec<i64>> {
    let first = ladde::exclusion::zone(m) as i64 + 1;
    let all: Vec<i64> = (first..windows as i64).collect();
    let mut state = salt | 1;
    let mut random = |per_mille: u64| -> Vec<i64> {
        all.iter()
            .copied()
            .filter(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (state >> 33) % 1000 < per_mille
            })
            .collect()
    };
    vec![
        vec![],
        all.iter().copied().take(1).collect(),
        all.iter().copied().step_by(2).collect(),
        all.iter().copied().step_by(7).collect(),
        random(100),
        random(500),
        all.clone(),
        vec![-1, 0, first - 1, windows as i64, i64::MIN, i64::MAX],
    ]
}

const THREADS: [usize; 3] = [1, 2, 7];

fn assert_self(series: &[f64], m: usize, scale: Scale, k: usize, salt: u64) {
    let windows = (series.len() + 1).saturating_sub(m);
    let samples = samples(series, scale);
    for diagonals in subsets(windows, m, salt) {
        let expected = bits(&naive_two_sided_topk_on(&samples, m, nz(k), &diagonals));
        let tiled = two_sided_topk_profile_over_tiled(
            series,
            nz(m),
            scale,
            nz(1),
            nz(5),
            nz(3),
            nz(6),
            nz(k),
            &diagonals,
        )
        .unwrap()
        .0;
        assert_eq!(bits(&tiled), expected, "tiled");
        for threads in THREADS {
            let walked = two_sided_topk_profile_over_walked(
                series,
                nz(m),
                scale,
                nz(threads),
                nz(k),
                &diagonals,
            )
            .unwrap();
            assert_eq!(
                bits(&walked),
                expected,
                "k {k}, threads {threads}, {} diagonals",
                diagonals.len()
            );
        }
    }
}

proptest! {
    #![proptest_config(testkit::generators::seeded(10))]

    #[test]
    fn the_walk_equals_the_reference((series, m, scale) in cases(), salt in any::<u64>()) {
        let _serial = forced_route(RouteForce::default());
        for k in [1, 3] {
            assert_self(&series, m, scale, k, salt);
        }
    }

    #[test]
    fn the_join_walk_equals_the_reference(
        (a, m, sa) in cases(),
        (b, _, sb) in cases(),
        salt in any::<u64>(),
    ) {
        let _serial = forced_route(RouteForce::default());
        let wins = |s: &[f64]| (s.len() + 1).saturating_sub(m);
        if wins(&a) == 0 || wins(&b) == 0 {
            return Ok(());
        }
        let all: Vec<i64> = (1 - wins(&a) as i64..wins(&b) as i64).collect();
        let (xa, xb) = (samples(&a, sa), samples(&b, sb));
        let sets = [
            vec![],
            all.iter().copied().step_by(3).collect(),
            all.iter().copied().filter(|d| (*d as u64 ^ salt) % 5 < 2).collect(),
            all.clone(),
            vec![i64::MIN, i64::MAX, -(wins(&a) as i64)],
        ];
        for diagonals in sets {
            for k in [1, 2] {
                let expected = join_bits(&naive_ab_join_topk_on(&xa, &xb, m, nz(k), &diagonals));
                let tiled = ab_join_topk_over_tiled(
                    &a, &b, nz(m), sa, sb, nz(1), nz(5), nz(3), nz(6), nz(k), &diagonals,
                ).unwrap().0;
                prop_assert_eq!(join_bits(&tiled), expected.clone());
                for threads in THREADS {
                    let walked = ab_join_topk_over_walked(
                        &a, &b, nz(m), sa, sb, nz(threads), nz(k), &diagonals,
                    ).unwrap();
                    prop_assert_eq!(join_bits(&walked), expected.clone(), "threads {}", threads);
                }
            }
        }
    }
}

#[test]
fn the_walk_equals_the_reference_on_ties_and_gaps() {
    let _serial = forced_route(RouteForce::default());
    let scale = Scale::new(1.0).unwrap();
    let flat = vec![7.0; 70];
    let runs: Vec<f64> = (0..90).map(|k| f64::from(((k / 11) % 3) as u8)).collect();
    let period: Vec<f64> = (0..80)
        .map(|k| f64::from([0, 3, 1, 4, 1, 5][k % 6]))
        .collect();
    let mut gapped: Vec<f64> = (0..80)
        .map(|k| f64::from(((k * k / 5 + k) % 11) as u8))
        .collect();
    for k in [0, 1, 2, 39, 40, 41, 77, 78, 79] {
        gapped[k] = f64::NAN;
    }
    for (series, m) in [(&flat, 4), (&runs, 5), (&period, 6), (&gapped, 6)] {
        for k in [1, 2, 200] {
            assert_self(series, m, scale, k, 9);
        }
    }
}

/// A series long enough that one band of the walk is not the whole series.
#[test]
fn the_walk_over_several_bands_equals_the_tiled_pass() {
    let _serial = forced_route(RouteForce::default());
    let series: Vec<f64> = (0..5000)
        .map(|k| f64::from(((k * k + 3 * k) % 41) as u8) + f64::from((k / 700) as u8))
        .collect();
    let (m, scale) = (6, Scale::new(1.0).unwrap());
    let diagonals: Vec<i64> = (3..4990).filter(|d| d % 97 < 3 || d % 389 == 0).collect();
    let walked =
        two_sided_topk_profile_over_walked(&series, nz(m), scale, nz(3), nz(2), &diagonals)
            .unwrap();
    let tiled = two_sided_topk_profile_over_tiled(
        &series,
        nz(m),
        scale,
        nz(3),
        nz(64),
        nz(32),
        nz(64),
        nz(2),
        &diagonals,
    )
    .unwrap()
    .0;
    assert_eq!(bits(&walked), bits(&tiled));
    assert!(walked.iter().any(|(_, s)| s.left.index().get().is_some()));
}

/// The anytime profiles under every forced choice of walk or tiles equal the reference, and the
/// forced choice is counted.
#[test]
fn a_forced_choice_of_walk_or_tiles_fires_and_changes_no_bit() {
    let series: Vec<f64> = (0..200)
        .map(|k| f64::from(((k * 7 + k / 3) % 13) as u8))
        .collect();
    let (m, scale) = (6, Scale::new(1.0).unwrap());
    let samples = samples(&series, scale);
    let (first, count) = Scrump::diagonals(series.len(), nz(m)).unwrap();
    let descending: Vec<i64> = (first..first + count as i64).rev().collect();
    let run = |force: SparseForce| {
        let guard = forced_route(RouteForce {
            sparse: force,
            ..RouteForce::default()
        });
        let mut scrump = Scrump::new(
            &series,
            nz(m),
            scale,
            nz(2),
            nz(2),
            1.0,
            Explicit(descending.clone()),
        )
        .unwrap();
        scrump.update().unwrap();
        let profile = bits(&scrump.profile().unwrap());
        (profile, guard.fired())
    };
    let expected = bits(&naive_two_sided_topk_on(&samples, m, nz(2), &descending));
    // One run of every diagonal: the run length tiles it.
    let (auto, fired) = run(SparseForce::Auto);
    assert_eq!(auto, expected);
    assert_eq!((fired.sparse_forced_on, fired.sparse_forced_off), (0, 0));
    let (on, fired) = run(SparseForce::On);
    assert_eq!(on, expected);
    assert_eq!((fired.sparse_forced_on, fired.sparse_forced_off), (1, 0));
    let (off, fired) = run(SparseForce::Off);
    assert_eq!(off, expected);
    assert_eq!((fired.sparse_forced_on, fired.sparse_forced_off), (0, 0));

    // Scattered diagonals: the run length walks them.
    let scattered: Vec<i64> = (first..first + count as i64)
        .filter(|d| d % 3 == 0)
        .chain((first..first + count as i64).filter(|d| d % 3 != 0))
        .collect();
    let run_scattered = |force: SparseForce| {
        let guard = forced_route(RouteForce {
            sparse: force,
            ..RouteForce::default()
        });
        let mut scrump = Scrump::new(
            &series,
            nz(m),
            scale,
            nz(1),
            nz(2),
            0.5,
            Explicit(scattered.clone()),
        )
        .unwrap();
        scrump.update().unwrap();
        let visited = scrump.visited().to_vec();
        (bits(&scrump.profile().unwrap()), visited, guard.fired())
    };
    let (auto, visited, fired) = run_scattered(SparseForce::Auto);
    assert_eq!(
        auto,
        bits(&naive_two_sided_topk_on(&samples, m, nz(2), &visited))
    );
    assert_eq!((fired.sparse_forced_on, fired.sparse_forced_off), (0, 0));
    let (off, _, fired) = run_scattered(SparseForce::Off);
    assert_eq!(off, auto);
    assert!(fired.sparse_forced_off > 0);
}

#[test]
fn the_join_chooses_by_run_length_and_a_forced_choice_changes_no_bit() {
    let a: Vec<f64> = (0..90)
        .map(|k| f64::from(((k * 7 + k / 3) % 13) as u8))
        .collect();
    let b: Vec<f64> = (0..70)
        .map(|k| f64::from(((k * 5 + k / 2) % 11) as u8))
        .collect();
    let scale = Scale::new(1.0).unwrap();
    let (first, count) = ScrumpJoin::diagonals(a.len(), b.len(), nz(5)).unwrap();
    let order: Vec<i64> = (first..first + count as i64).rev().collect();
    let mut results = Vec::new();
    for force in [SparseForce::Auto, SparseForce::On, SparseForce::Off] {
        let guard = forced_route(RouteForce {
            sparse: force,
            ..RouteForce::default()
        });
        let mut join = ScrumpJoin::new(
            &a,
            &b,
            nz(5),
            (scale, scale),
            nz(2),
            nz(2),
            1.0,
            Explicit(order.clone()),
        )
        .unwrap();
        join.update().unwrap();
        results.push((join_bits(&join.profile().unwrap()), guard.fired()));
    }
    assert_eq!(results[0].0, results[1].0);
    assert_eq!(results[0].0, results[2].0);
    assert_eq!(results[1].1.sparse_forced_on, 1);
    assert_eq!(results[2].1.sparse_forced_off, 0);
}

/// One chunk holding a long run and scattered singles: the tiled pass and the walk each take their
/// part, and the merged profile is the reference's.
#[test]
fn a_chunk_of_a_long_run_and_scattered_diagonals_is_split_and_merged() {
    let _serial = forced_route(RouteForce::default());
    let series: Vec<f64> = (0..240)
        .map(|k| f64::from(((k * k / 3 + 7 * k) % 29) as u8))
        .collect();
    let (m, scale) = (6, Scale::new(1.0).unwrap());
    let (first, count) = Scrump::diagonals(series.len(), nz(m)).unwrap();
    let all: Vec<i64> = (first..first + count as i64).collect();
    let chosen: Vec<i64> = all
        .iter()
        .copied()
        .filter(|d| (40..70).contains(d) || d % 5 == 0)
        .collect();
    let mut rest: Vec<i64> = all
        .iter()
        .copied()
        .filter(|d| !chosen.contains(d))
        .collect();
    rest.sort_by_key(|d| (d % 7, *d));
    let order: Vec<i64> = chosen.iter().chain(&rest).copied().collect();
    let series_ticks = samples(&series, scale);
    for k in [1, 3] {
        let mut scrump = Scrump::new(
            &series,
            nz(m),
            scale,
            nz(2),
            nz(k),
            0.5,
            Explicit(order.clone()),
        )
        .unwrap();
        scrump.update().unwrap();
        let visited = scrump.visited().to_vec();
        assert!(visited.len() > chosen.len());
        assert_eq!(
            bits(&scrump.profile().unwrap()),
            bits(&naive_two_sided_topk_on(&series_ticks, m, nz(k), &visited))
        );
    }
    let b: Vec<f64> = (0..200)
        .map(|k| f64::from(((k * 5 + k / 2) % 19) as u8))
        .collect();
    let (jfirst, jcount) = ScrumpJoin::diagonals(series.len(), b.len(), nz(m)).unwrap();
    let jall: Vec<i64> = (jfirst..jfirst + jcount as i64).collect();
    let jchosen: Vec<i64> = jall
        .iter()
        .copied()
        .filter(|d| (-20..10).contains(d) || d % 4 == 0)
        .collect();
    let mut jrest: Vec<i64> = jall
        .iter()
        .copied()
        .filter(|d| !jchosen.contains(d))
        .collect();
    jrest.sort_by_key(|d| (d.rem_euclid(7), *d));
    let jorder: Vec<i64> = jchosen.iter().chain(&jrest).copied().collect();
    let mut join = ScrumpJoin::new(
        &series,
        &b,
        nz(m),
        (scale, scale),
        nz(2),
        nz(2),
        0.5,
        Explicit(jorder),
    )
    .unwrap();
    join.update().unwrap();
    let b_ticks = samples(&b, scale);
    let expected = naive_ab_join_topk_on(&series_ticks, &b_ticks, m, nz(2), join.visited());
    assert_eq!(join_bits(&join.profile().unwrap()), join_bits(&expected));
}

/// The run length that tiles: a series with exactly `run` diagonals is one run of that length.
/// Every run shorter than the target's threshold is walked and every other run is tiled, so
/// forcing the other way is counted for exactly one, and every route gives the same bits.
#[test]
fn short_runs_walk_and_longer_runs_tile() {
    let (m, scale) = (8, Scale::new(1.0).unwrap());
    let mut walked = Vec::new();
    for run in 1..=40usize {
        // Windows `run + 3` with an exclusion zone of 2 leave diagonals 3..run + 3.
        let series: Vec<f64> = (0..run + 3 + m - 1)
            .map(|k| f64::from(((k * 7 + k / 3) % 13) as u8))
            .collect();
        let (first, count) = Scrump::diagonals(series.len(), nz(m)).unwrap();
        assert_eq!((first, count), (3, run));
        let ticks = samples(&series, scale);
        let expected = bits(&naive_two_sided_topk_on(
            &ticks,
            m,
            nz(2),
            &(first..first + count as i64).collect::<Vec<_>>(),
        ));
        let mut counted = Vec::new();
        for force in [SparseForce::Auto, SparseForce::On, SparseForce::Off] {
            let guard = forced_route(RouteForce {
                sparse: force,
                ..RouteForce::default()
            });
            let mut scrump = Scrump::new(
                &series,
                nz(m),
                scale,
                nz(1),
                nz(2),
                1.0,
                Explicit((first..first + count as i64).collect()),
            )
            .unwrap();
            scrump.update().unwrap();
            assert_eq!(bits(&scrump.profile().unwrap()), expected, "run {run}");
            let fired = guard.fired();
            counted.push((fired.sparse_forced_on, fired.sparse_forced_off));
        }
        // Forcing on is counted for a run that tiles, forcing off for one that walks.
        let walks = match counted[..] {
            [(0, 0), (0, 0), (0, 1)] => true,
            [(0, 0), (1, 0), (0, 0)] => false,
            _ => panic!("run {run}: {counted:?}"),
        };
        walked.push(walks);
    }
    let threshold = walked.iter().position(|&w| !w).expect("a run of 40 tiles");
    assert!(threshold > 1, "a run of one walks");
    assert!(walked[threshold..].iter().all(|&w| !w), "{walked:?}");
}
