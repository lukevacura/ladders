#![cfg(all(feature = "oracle", not(any(loom, shuttle))))]
//! The best-pair search (`motif_pairs`): a two-sided run that keeps one global floor, the highest
//! floor any row has reached or the floor of a pair the constant class holds. Its pair is the one
//! the full two-sided profile gives under the pick rule, on bits, at 1, 2 and 8 threads, with the
//! filter forced off, left to the gate and forced to PAA, and under the forced
//! routes of the other families, on near-ties, exact repeats (three copies tie their pairs),
//! constant runs and gaps. The floor is installed only by the best-pair run: a normal two-sided,
//! left or top-k run fires none of its counters, and the force that turns it off leaves the
//! pair unchanged.

#[path = "common/route.rs"]
mod route;

use ladde::batch::{
    left_profile_report, two_sided_profile, two_sided_profile_tiled, two_sided_topk_profile_tiled,
};
use ladde::entry::TwoSided;
use ladde::motif::{MotifPair, motif_pairs, pick_pair};
use ladde::quantize::Scale;
use ladde::route::{Fired, FloorForce, RouteForce, SurvivorForce, forced_route};
use proptest::prelude::*;
use route::*;
use testkit::generators::{
    MARKET_TICK, market_series, seeded_default, with_flat_runs, with_gaps, with_repeats,
};

const THREADS: [usize; 3] = [1, 2, 8];

/// The filters a run is forced to: the production gate's choice (PAA at every window length for
/// this run), none, and PAA forced.
const FILTERS: [SurvivorForce; 3] = [SurvivorForce::Gate, SurvivorForce::Off, SurvivorForce::Paa];

type Pair = Option<(usize, usize, u64)>;

fn bits_of(pair: Option<MotifPair>) -> Pair {
    pair.map(|p| (p.a, p.b, p.distance.to_bits()))
}

/// The pair of the full plain profile: the filter forced off, so every entry is exact.
fn full_pair(series: &[f64], m: usize, scale: Scale, base: RouteForce) -> Pair {
    let _g = forced_route(RouteForce {
        survivor: SurvivorForce::Off,
        ..base
    });
    let profile = two_sided_profile(series, nz(m), scale, nz(2));
    let picked = bits_of(pick_pair(series, nz(m), scale, &profile));
    assert!(picked.is_none_or(|(a, b, _)| a < b), "a < b");
    picked
}

fn search(series: &[f64], m: usize, scale: Scale, threads: usize) -> Pair {
    bits_of(
        motif_pairs(series, nz(m), scale, nz(threads))
            .expect("small series fit")
            .first()
            .copied(),
    )
}

/// `series` searched at 1, 2 and 8 threads under every filter route and `base`, each held to the
/// full profile's pair.
fn check(what: &str, (series, m, scale): (&[f64], usize, Scale), base: RouteForce) -> Pair {
    let expected = full_pair(series, m, scale, base);
    for threads in THREADS {
        for survivor in FILTERS {
            let _g = forced_route(RouteForce { survivor, ..base });
            assert_eq!(
                search(series, m, scale, threads),
                expected,
                "{what}, m {m}, {threads} threads, {survivor:?}"
            );
        }
    }
    expected
}

/// A sum of sines with a slow drift, in ticks: the kind of series the PAA bound prunes on.
fn smooth(n: usize) -> Vec<f64> {
    (0..n)
        .map(|p| {
            let t = p as f64;
            let wave = 40.0 * (t / 37.0).sin() + 25.0 * (t / 101.0).sin() + 10.0 * (t / 13.0).cos();
            (wave + t / 50.0).round() * MARKET_TICK
        })
        .collect()
}

/// `n` samples of `levels` integer levels from a fixed linear congruential sequence.
fn few_levels(n: usize, levels: u64, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) % levels) as f64
        })
        .collect()
}

/// Integer levels of a slow wave with noise, so that no window is constant.
fn wavy(n: usize) -> Vec<f64> {
    let noise = few_levels(n, 4, 21);
    smooth(n)
        .into_iter()
        .zip(noise)
        .map(|(x, noise)| (x / MARKET_TICK / 4.0).round() + noise)
        .collect()
}

fn ticks() -> Scale {
    Scale::new(MARKET_TICK).unwrap()
}

fn unit() -> Scale {
    Scale::new(1.0).unwrap()
}

#[test]
fn the_pair_is_the_full_profiles_on_a_smooth_series() {
    let series = smooth(600);
    for m in [16, 64] {
        assert!(check("smooth", (&series, m, ticks()), routed()).is_some());
    }
}

/// A few levels tie many pairs at each correlation, the nearest among them.
#[test]
fn the_pair_is_the_full_profiles_on_few_levels() {
    for (levels, seed) in [(2, 1), (3, 2), (5, 3)] {
        let series = few_levels(420, levels, seed);
        for m in [8, 16] {
            check("few levels", (&series, m, unit()), routed());
        }
    }
}

/// A stretch repeated exactly: its windows pair at distance zero with each other only, and with
/// three copies every pair of them ties, so the pair is the earliest window's.
#[test]
fn the_pair_is_the_full_profiles_on_exact_repeats() {
    let mut series = few_levels(480, 9, 4);
    let stretch = series[40..90].to_vec();
    series[200..250].copy_from_slice(&stretch);
    series[390..440].copy_from_slice(&stretch);
    for m in [8, 16, 32] {
        let pair = check("exact repeats", (&series, m, unit()), routed());
        let (a, b, distance) = pair.expect("a pair");
        assert_eq!((a, b, distance), (40, 200, 0f64.to_bits()), "m {m}");
    }
    let period: Vec<f64> = (0..400)
        .map(|k| f64::from([3, 7, 1, 9, 4, 4, 2][k % 7]))
        .collect();
    check("one period", (&period, 12, unit()), routed());
}

/// Two copies of a stretch one tick apart in one sample are a near-tie of the nearest pair, and
/// the copy of that near-pair is nearer still.
#[test]
fn the_pair_is_the_full_profiles_on_near_ties() {
    let mut series = few_levels(500, 40, 11);
    let stretch = series[60..110].to_vec();
    series[260..310].copy_from_slice(&stretch);
    series[420..470].copy_from_slice(&stretch);
    series[300] += 1.0;
    series[450] += 1.0;
    for m in [10, 24] {
        check("near-ties", (&series, m, unit()), routed());
    }
    let mut smooth = smooth(500);
    let stretch = smooth[100..140].to_vec();
    smooth[330..370].copy_from_slice(&stretch);
    smooth[335] += MARKET_TICK;
    check("smooth near-tie", (&smooth, 16, ticks()), routed());
}

/// Flat runs: constant windows pair at distance zero when two lie outside each other's zone, and
/// one flat run alone leaves the pair to the rest.
#[test]
fn the_pair_is_the_full_profiles_on_constant_runs() {
    let mut two = few_levels(500, 7, 5);
    two[60..100].fill(5.0);
    two[300..330].fill(2.0);
    let mut one: Vec<f64> = smooth(500)
        .into_iter()
        .map(|x| (x / MARKET_TICK).round())
        .collect();
    one[220..245].fill(3.0);
    let mut short = few_levels(300, 4, 6);
    short[100..112].fill(1.0);
    for m in [8, 16] {
        check("two flat runs", (&two, m, unit()), routed());
        check("one flat run", (&one, m, unit()), routed());
        check("a flat run of one zone", (&short, m, unit()), routed());
    }
}

/// Nearly every window has a nearly perfect neighbor at a short lag, so the rows' floors are close
/// to one early on; the best pair, an exact repeat of a noisy stretch, lies far behind them. A
/// global floor above the rows' would skip the block that holds it.
#[test]
fn the_pair_is_the_full_profiles_when_it_lies_far_behind_near_perfect_ones() {
    let mut series = wavy(900);
    let stretch = series[100..180].to_vec();
    series[700..780].copy_from_slice(&stretch);
    for m in [16, 32] {
        let pair = check("a far repeat", (&series, m, unit()), routed());
        let (a, b, distance) = pair.expect("a pair");
        assert!(
            (100..180).contains(&a) && (700..780).contains(&b),
            "{a} {b}"
        );
        assert_eq!(distance, 0f64.to_bits());
    }
}

/// Gaps: windows over a missing sample have no distance and set no floor.
#[test]
fn the_pair_is_the_full_profiles_across_gaps() {
    let mut series: Vec<f64> = (0..300)
        .map(|k| f64::from([3, 7, 1, 9, 4, 4, 2][k % 7]) + f64::from(u8::from(k % 53 == 0)))
        .collect();
    series[100..130].fill(5.0);
    for at in [0, 17, 64, 65, 299] {
        series[at] = f64::NAN;
    }
    let mut walk = smooth(600);
    for at in [0, 5, 150, 151, 152, 400, 598, 599] {
        walk[at] = f64::NAN;
    }
    check("ties, a flat run and gaps", (&series, 6, unit()), routed());
    check("smooth with gaps", (&walk, 16, ticks()), routed());
}

/// A series with no pair has no answer, and nothing is padded.
#[test]
fn a_series_with_no_pair_has_an_empty_answer() {
    let some = |series: &[f64], m: usize| {
        motif_pairs(series, nz(m), unit(), nz(2)).expect("small series fit")
    };
    assert!(
        some(&[1.0, 2.0, 3.0], 8).is_empty(),
        "shorter than a window"
    );
    assert!(
        some(&few_levels(9, 5, 1), 8).is_empty(),
        "two windows in one zone"
    );
    assert!(some(&[f64::NAN; 40], 8).is_empty(), "no finite window");
    assert!(some(&[], 4).is_empty(), "no sample");
    let pairs = some(&few_levels(60, 5, 2), 8);
    assert_eq!(pairs.len(), 1);
    assert!(pairs[0].a < pairs[0].b || pairs[0].b < pairs[0].a);
}

/// The forced routes of the other families (the exact i64 tiles, re-anchoring, spill, tile
/// shapes, the constant class off, resumes) leave the pair unchanged.
#[test]
fn the_pair_is_the_full_profiles_under_the_other_families_forces() {
    let smooth = smooth(300);
    let mut flat = few_levels(300, 6, 8);
    flat[40..70].fill(3.0);
    flat[200..230].fill(3.0);
    for force in forces(0x5eed) {
        for (series, m, scale) in [(&smooth, 16, ticks()), (&flat, 12, unit())] {
            let expected = full_pair(series, m, scale, force);
            for threads in [1, 8] {
                let _g = forced_route(force);
                assert_eq!(
                    search(series, m, scale, threads),
                    expected,
                    "{force:?}, m {m}, {threads} threads"
                );
            }
        }
    }
}

/// The pair of a one-thread search, what fired, and the area the one-loop route skipped.
fn run_fired(series: &[f64], m: usize, scale: Scale, force: RouteForce) -> (Pair, Fired, usize) {
    let g = forced_route(force);
    let pair = search(series, m, scale, 1);
    (pair, g.fired(), g.work().skipped_area)
}

/// On one thread, where the work is fixed: the floor rises from the rows' pairs, the filter is
/// asked at a window length the production gate leaves off, and blocks are skipped under the
/// floor. With the force that turns the floor off, none of that happens and the pair is the same.
#[test]
fn the_global_floor_rises_and_skips_and_its_force_turns_it_off() {
    let (series, m) = (wavy(900), 16);
    let (on, fired, skipped) = run_fired(&series, m, unit(), routed());
    assert!(fired.gate_on >= 1, "asked at m {m}: {fired:?}");
    assert!(fired.global_rises > 0, "{fired:?}");
    assert!(skipped > 0, "{fired:?}");
    assert_eq!(fired.global_seeds, 0, "{fired:?}");
    let (off, fired, _) = run_fired(
        &series,
        m,
        unit(),
        RouteForce {
            no_global_floor: true,
            ..routed()
        },
    );
    assert_eq!(
        (fired.global_rises, fired.global_seeds),
        (0, 0),
        "{fired:?}"
    );
    assert!(fired.paa_tables > 0, "the filter is still asked: {fired:?}");
    assert_eq!(on, off);
    assert_eq!(on, full_pair(&series, m, unit(), routed()));
}

/// The floor lives in the run's shared table and in a filter that reads blocks: without either
/// (no filter, a table per tile, a PAA table that cannot be built) the run is the plain pass and
/// the pair is still the full profile's.
#[test]
fn the_global_floor_needs_a_filter_and_the_shared_table() {
    let (series, m) = (smooth(600), 16);
    let expected = full_pair(&series, m, ticks(), routed());
    for force in [
        RouteForce {
            survivor: SurvivorForce::Off,
            ..routed()
        },
        RouteForce {
            floors: FloorForce::PerThread,
            ..routed()
        },
        RouteForce {
            paa_build_fails: true,
            ..routed()
        },
    ] {
        let (pair, fired, _) = run_fired(&series, m, ticks(), force);
        assert_eq!(fired.global_rises, 0, "{force:?}: {fired:?}");
        assert_eq!(pair, expected, "{force:?}");
    }
}

/// Two constant windows outside each other's zone are a pair of distance zero the class knows
/// before the pass: the floor starts from it, and blocks are skipped under it from the first
/// row. One flat run no longer than its zone gives none, and neither does a run without the
/// class; the pair is the full profile's in each.
#[test]
fn the_constant_class_seeds_the_global_floor() {
    let mut two = wavy(700);
    two[100..140].fill(5.0);
    two[420..460].fill(2.0);
    let m = 16;
    let expected = full_pair(&two, m, unit(), routed());
    let (pair, fired, skipped) = run_fired(&two, m, unit(), routed());
    assert_eq!(fired.global_seeds, 1, "{fired:?}");
    assert!(skipped > 0, "{fired:?}");
    assert_eq!(pair, expected);
    assert_eq!(pair.map(|(_, _, distance)| distance), Some(0f64.to_bits()));
    let (pair, fired, _) = run_fired(
        &two,
        m,
        unit(),
        RouteForce {
            no_constant_class: true,
            ..routed()
        },
    );
    assert_eq!(fired.global_seeds, 0, "{fired:?}");
    assert_eq!(pair, expected);
    // A flat run of one zone and a window: its windows are all within the zone of each other.
    let mut within = wavy(500);
    within[100..100 + m + 4].fill(7.0);
    let expected = full_pair(&within, m, unit(), routed());
    let (pair, fired, _) = run_fired(&within, m, unit(), routed());
    assert_eq!(fired.global_seeds, 0, "{fired:?}");
    assert_eq!(pair, expected);
    within[300..300 + m].fill(7.0);
    let expected = full_pair(&within, m, unit(), routed());
    let (pair, fired, _) = run_fired(&within, m, unit(), routed());
    assert_eq!(fired.global_seeds, 1, "{fired:?}");
    assert_eq!(pair, expected);
}

/// A normal run keeps no global floor: two-sided, left and top-k runs fire none of its counters,
/// under the forced filters too, and give the plain pass's bits.
#[test]
fn other_runs_keep_no_global_floor() {
    let (series, m) = (smooth(600), 16);
    let rows = |profile: &[(u64, TwoSided)]| -> Vec<Bits> {
        profile
            .iter()
            .flat_map(|&(row, t)| [t.left, t.right, t.nearest].map(|e| bits(row, e)))
            .collect()
    };
    let topk = |threads: usize| -> Vec<Bits> {
        let (profile, _) = two_sided_topk_profile_tiled(
            &series,
            nz(m),
            ticks(),
            nz(threads),
            nz(8),
            nz(64),
            nz(64),
            nz(4),
        )
        .expect("small series fit");
        profile
            .iter()
            .flat_map(|(row, t)| {
                [t.left, t.right]
                    .into_iter()
                    .chain(t.nearest.iter().copied())
                    .map(|e| bits(*row, e))
            })
            .collect()
    };
    for threads in THREADS {
        for survivor in FILTERS {
            let g = forced_route(RouteForce {
                survivor,
                ..routed()
            });
            let two = rows(&two_sided_profile(&series, nz(m), ticks(), nz(threads)));
            let (shaped, _) = two_sided_profile_tiled(
                &series,
                nz(m),
                ticks(),
                nz(threads),
                nz(8),
                nz(64),
                nz(64),
            );
            let left = profile_bits(&left_profile_report(&series, nz(m), ticks(), nz(threads)).0);
            let k_best = topk(threads);
            let fired = g.fired();
            drop(g);
            assert_eq!(
                (fired.global_rises, fired.global_seeds),
                (0, 0),
                "{threads} threads, {survivor:?}: {fired:?}"
            );
            let plain = {
                let _g = forced_route(RouteForce {
                    survivor: SurvivorForce::Off,
                    ..routed()
                });
                (
                    rows(&two_sided_profile(&series, nz(m), ticks(), nz(threads))),
                    profile_bits(&left_profile_report(&series, nz(m), ticks(), nz(threads)).0),
                    topk(threads),
                )
            };
            assert!(
                (two.clone(), left, k_best) == plain,
                "{threads} threads, {survivor:?}: the plain profiles"
            );
            assert!(rows(&shaped) == plain.0, "{threads} threads, {survivor:?}");
        }
    }
}

proptest! {
    #![proptest_config(seeded_default())]

    /// Market walks with flat runs, gaps and exact repeats.
    #[test]
    fn the_pair_is_the_full_profiles_on_market_series(
        (m, series) in prop_oneof![Just(8usize), Just(16), Just(32)]
            .prop_flat_map(|m| (Just(m), market_series(120..260, m)))
    ) {
        check("market", (&series, m, ticks()), routed());
    }

    /// Few integer levels (ties everywhere), with flat runs, gaps and repeats.
    #[test]
    fn the_pair_is_the_full_profiles_on_tie_heavy_series(
        (m, series) in prop_oneof![Just(6usize), Just(12)].prop_flat_map(|m| {
            let base = prop::collection::vec(0u8..4, 90..200)
                .prop_map(|v| v.into_iter().map(f64::from).collect::<Vec<_>>());
            let shaped = prop_oneof![
                base.clone(),
                with_flat_runs(base.clone(), m),
                with_gaps(base.clone()),
                with_repeats(base.clone(), m),
                with_gaps(with_flat_runs(with_repeats(base, m), m)),
            ];
            (Just(m), shaped)
        })
    ) {
        check("tie heavy", (&series, m, unit()), routed());
    }
}
