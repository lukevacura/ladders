extern crate std;

use super::{
    BLOCK_ROWS, FRESH_GROUPS, PAR_RUNS, PARTIALS_BYTES, PEAK_LIVE_BANDS, RowAcc,
    SINGLE_THREADED_BELOW_ROWS, TILE_RENDEZVOUS, TILES_PER_THREAD, VISITS, cut_tiles,
    default_chunk_lanes, eligible_count, left_profile, left_profile_threads,
    left_profile_threads_tiled, single_threaded_below_rows, threaded_workers, tile_plan,
};
use crate::entry::Entry;
use crate::quantize::Scale;
use core::num::NonZeroUsize;
use std::vec::Vec;

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

/// A market-offset random walk: stays on the narrow path, ties are rare.
fn walk(n: usize) -> Vec<f64> {
    let mut state = 0x2545_F491_4F6C_DD1D_u64;
    let mut level = 1e10;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            level += ((state >> 56) as i64 % 5 - 2) as f64;
            level
        })
        .collect()
}

fn bits(profile: &[(u64, Entry)]) -> Vec<(u64, u64, u64, u64)> {
    profile
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

// ---------- every eligible pair evaluated exactly once, on the threaded engine ----------

/// Every eligible `(row, neighbor)` pair is offered to its row exactly once across all tiles
/// and sub-chunks: none twice (a tile's first sub-chunk run again, or a partial merged
/// twice, changes no bit but shows here), none missing, none inside the exclusion zone. A pair
/// the constant class settles is accounted for without being offered: the tiles leave some of
/// them out, a block at a time.
#[test]
fn every_eligible_pair_is_offered_exactly_once_across_tiles() {
    let m = 3usize;
    let exclusion = crate::exclusion::zone(m);
    let series = walk(150);
    let scale = Scale::new(1.0).unwrap();
    let samples: Vec<Option<i64>> = series
        .iter()
        .map(|&x| crate::quantize::to_ticks(x, scale))
        .collect();
    let settled = super::class_settles(&samples, m, exclusion);
    let mut left_out = 0;
    let n_windows = series.len() - m + 1;
    let n = series.len();
    // (band_rows, chunk_lanes, tile_lanes): tiles of one to many sub-chunks, ragged edges.
    let plans = [
        (1, 1, 1),
        (2, 1, 3),
        (3, 2, 4),
        (5, 3, 3),
        (8, 7, 14),
        (13, 5, 25),
        (100, 64, 128),
        (n + 5, n + 11, n + 11),
    ];
    for threads in [2, 3, 4] {
        for (band, chunk, tile) in plans {
            VISITS.with(|v| v.borrow_mut().clear());
            left_profile_threads_tiled(
                &series,
                nz(m),
                scale,
                nz(threads),
                nz(band),
                nz(chunk),
                nz(tile),
            );
            let mut seen = VISITS.with(|v| v.borrow().clone());
            seen.sort();
            let rows: Vec<usize> = seen.iter().map(|&(row, _)| row).collect();
            let expected_rows: Vec<usize> = (exclusion + 1..n_windows).collect();
            let plan = std::format!("{threads} threads, {band}/{chunk}/{tile}");
            assert_eq!(rows, expected_rows, "{plan}");
            for (row, mut visits) in seen {
                visits.sort();
                let expected: Vec<usize> = (0..eligible_count(row, exclusion))
                    .filter(|&j| visits.binary_search(&j).is_ok() || !settled(row, j))
                    .collect();
                assert_eq!(visits, expected, "{plan}, row {row}");
                left_out += eligible_count(row, exclusion) - visits.len();
            }
        }
    }
    assert!(left_out > 0, "the series has blocks the class settles");
}

#[test]
fn tile_width_and_band_height_never_change_a_bit() {
    let m = 5usize;
    let series = walk(400);
    let scale = Scale::new(1.0).unwrap();
    let expected = bits(&left_profile(&series, nz(m), scale));
    assert!(expected.iter().any(|&(_, _, _, i)| i != u64::MAX));
    for threads in [2, 5] {
        for (band, chunk, tile) in [(7, 3, 3), (7, 3, 9), (64, 16, 50), (1000, 64, 1000)] {
            let got = left_profile_threads_tiled(
                &series,
                nz(m),
                scale,
                nz(threads),
                nz(band),
                nz(chunk),
                nz(tile),
            );
            assert_eq!(bits(&got), expected, "{threads} {band}/{chunk}/{tile}");
        }
    }
}

// ---------- the single-threaded threshold ----------

/// The cutoff grows with the threads asked, within its bounds.
#[test]
fn the_cutoff_grows_with_the_threads_asked_within_bounds() {
    let cutoffs: Vec<usize> = [1, 2, 4, 8, 10, 16, 64, usize::MAX]
        .map(single_threaded_below_rows)
        .into();
    assert!(cutoffs.windows(2).all(|w| w[0] <= w[1]), "{cutoffs:?}");
    assert!(cutoffs[0] < cutoffs[4] && cutoffs[4] < cutoffs[7]);
    assert_eq!(cutoffs[7], SINGLE_THREADED_BELOW_ROWS);
}

/// Below the cutoff for the threads asked `left_profile_threads` is `left_profile`'s own engine;
/// at it, the threaded one runs. Both give the same bits, at every thread count.
#[test]
fn small_series_take_the_single_threaded_path() {
    let m = 3usize;
    let scale = Scale::new(1.0).unwrap();
    let runs = || PAR_RUNS.with(|r| r.get());
    for threads in [2, 4, 8, 10] {
        let cutoff = single_threaded_below_rows(threads);
        for (windows, threaded) in [(cutoff - 1, false), (cutoff, true)] {
            let series = walk(windows + m - 1);
            let before = runs();
            let got = left_profile_threads(&series, nz(m), scale, nz(threads));
            assert_eq!(
                runs() - before,
                usize::from(threaded),
                "{windows} windows, {threads} threads"
            );
            assert_eq!(
                bits(&got),
                bits(&left_profile(&series, nz(m), scale)),
                "{windows} windows, {threads} threads"
            );
        }
    }
    // One thread never runs the threaded engine, at any size.
    let series = walk(SINGLE_THREADED_BELOW_ROWS + m - 1);
    let before = runs();
    left_profile_threads(&series, nz(m), scale, nz(1));
    assert_eq!(runs(), before);
}

/// `need` tiles are inside at once only if `need` threads run tiles: the gate holds every tile
/// until that many have arrived, and gives up (a failure) only after a long hang guard.
#[test]
fn the_threaded_engine_runs_tiles_on_as_many_threads_as_asked() {
    let m = 3usize;
    let scale = Scale::new(1.0).unwrap();
    let series = walk(SINGLE_THREADED_BELOW_ROWS * 4 + m - 1);
    let expected = bits(&left_profile(&series, nz(m), scale));
    for threads in [2, 4] {
        assert_eq!(threaded_workers(series.len() - m + 1, m, threads), threads);
        TILE_RENDEZVOUS.with(|gate| gate.set(threads));
        let got = left_profile_threads(&series, nz(m), scale, nz(threads));
        TILE_RENDEZVOUS.with(|gate| gate.set(0));
        assert_eq!(bits(&got), expected, "{threads} threads");
    }
}

/// Only a window of bands' accumulators is alive at once, however many bands there are: a
/// band's rows are finished and its accumulators dropped when its last tile merges.
#[test]
fn live_accumulators_stay_within_a_window_of_bands() {
    let m = 3usize;
    let series = walk(3_000);
    let scale = Scale::new(1.0).unwrap();
    let expected = bits(&left_profile(&series, nz(m), scale));
    for threads in [2usize, 4] {
        let got =
            left_profile_threads_tiled(&series, nz(m), scale, nz(threads), nz(20), nz(64), nz(256));
        let bands = 3_000usize.div_ceil(20);
        let peak = PEAK_LIVE_BANDS.with(|p| p.get());
        assert_eq!(bits(&got), expected);
        assert!(bands > 100);
        assert!(
            peak <= 2 * threads + 2,
            "{threads} threads: {peak} of {bands}"
        );
        // A band opens with the accumulators a finished band left, so the groups made new are
        // those of the bands alive at once, not one per band.
        let fresh = FRESH_GROUPS.with(|p| p.get());
        assert!(
            fresh <= peak,
            "{threads} threads: {fresh} made new, {peak} alive at most"
        );
    }
}

#[test]
fn workers_are_threads_capped_by_tiles_and_one_when_single_threaded() {
    assert_eq!(threaded_workers(100_000, 50, 1), 1);
    assert_eq!(
        threaded_workers(single_threaded_below_rows(8) - 1, 50, 8),
        1
    );
    assert!(threaded_workers(single_threaded_below_rows(8), 50, 8) > 1);
    for t in [2, 3, 8, 16] {
        assert_eq!(threaded_workers(100_000, 50, t), t);
    }
    // Tiles cap it: a series just past the threshold with a huge band can't feed everyone.
    let (band, tile) = tile_plan(SINGLE_THREADED_BELOW_ROWS, 50, 4096);
    let tiles = cut_tiles(SINGLE_THREADED_BELOW_ROWS, 13, band, tile, None).len();
    assert_eq!(
        threaded_workers(SINGLE_THREADED_BELOW_ROWS, 50, 4096),
        tiles.min(4096)
    );
}

// ---------- tile_plan ----------

const SIZES: [usize; 9] = [
    60,
    600,
    2_500,
    10_000,
    50_000,
    100_000,
    1_000_000,
    10_000_000,
    1_000_000_000,
];
const WINDOWS: [usize; 6] = [3, 8, 50, 256, 1000, 5000];
const THREADS: [usize; 6] = [1, 2, 3, 4, 8, 16];

fn count(n_windows: usize, m: usize, plan: (usize, usize)) -> usize {
    // Exact only where `cut_tiles` is affordable.
    cut_tiles(n_windows, crate::exclusion::zone(m), plan.0, plan.1, None).len()
}

#[test]
fn a_plan_keeps_a_thread_scratch_within_budget_at_any_size() {
    let chunk = default_chunk_lanes();
    for n in SIZES {
        for m in WINDOWS {
            for t in THREADS {
                let n_windows = n.saturating_sub(m - 1);
                let (band, tile) = tile_plan(n_windows, m, t);
                let ctx = std::format!("n {n}, m {m}, T {t}: {band} x {tile}");
                assert!(band >= 1, "{ctx}");
                assert!(band * size_of::<RowAcc>() <= PARTIALS_BYTES, "{ctx}");
                assert!(tile * size_of::<i64>() <= PARTIALS_BYTES, "{ctx}");
                assert!(tile >= chunk && tile % chunk == 0, "{ctx}");
            }
        }
    }
}

#[test]
fn a_plan_cuts_at_least_the_asked_tiles_when_the_series_allows() {
    let chunk = default_chunk_lanes();
    for n in [600_usize, 2_500, 10_000, 50_000, 100_000] {
        for m in WINDOWS {
            for t in THREADS {
                let n_windows = n.saturating_sub(m - 1);
                let plan = tile_plan(n_windows, m, t);
                let target = TILES_PER_THREAD * t;
                let finest = count(n_windows, m, (BLOCK_ROWS, chunk));
                let got = count(n_windows, m, plan);
                let ctx = std::format!("n {n}, m {m}, T {t}: {plan:?} cuts {got}, finest {finest}");
                // Tiles are never fewer than threads while the series can be cut that fine.
                assert!(got >= t.min(finest), "{ctx}");
                if finest >= 2 * target {
                    assert!(got >= target, "{ctx}");
                }
            }
        }
    }
}

#[test]
fn a_plan_asks_for_more_tiles_with_more_threads() {
    // 10⁵ windows at m = 50: the target decides the cut, and 16 threads need more tiles.
    let counts: Vec<usize> = [2, 4, 8, 16]
        .map(|t| count(100_000, 50, tile_plan(100_000, 50, t)))
        .to_vec();
    for (pair, t) in counts.windows(2).zip([2, 4, 8]) {
        assert!(pair[1] > pair[0], "{t} -> {}: {counts:?}", t * 2);
    }
    for (got, t) in counts.iter().zip([2, 4, 8, 16]) {
        assert!(*got >= TILES_PER_THREAD * t, "T {t}: {counts:?}");
    }
}

/// The start cost is weighed, not floored: a short series with a large `m` takes bands far
/// shorter than the old `16 m`, and cuts enough tiles for eight threads; a long one takes
/// tall bands and wide tiles.
#[test]
fn a_plan_weighs_the_start_cost_against_the_tail() {
    let (band, _) = tile_plan(10_000 - 999, 1000, 8);
    assert!((256..=2048).contains(&band), "{band}");
    assert!(count(10_000 - 999, 1000, tile_plan(10_000 - 999, 1000, 8)) >= 16);
    // Dearer starts (larger m) lengthen the band.
    let (small, _) = tile_plan(100_000, 50, 8);
    let (large, _) = tile_plan(100_000, 1000, 8);
    assert!(large > small, "{small} {large}");
    let chunk = default_chunk_lanes();
    let (band, tile) = tile_plan(1_000_000, 50, 8);
    assert!(band >= 1024 && tile >= 4 * chunk, "{band} {tile}");
}
