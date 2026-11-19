extern crate std;

use super::{
    BLOCK_ROWS, Quantized, RIGHT_GROUP, RightBook, Tile, TilePlan, default_chunk_lanes,
    eligible_count, is_finite, join_list_len, left_profile, left_profile_with_blocks,
    left_profile_with_chunk, profile_on, side_counts, tile_columns,
};
use crate::entry::Entry;
use crate::naive::naive_left_profile;
use crate::quantize::{Scale, to_ticks};
use core::num::NonZeroUsize;
use proptest::collection::SizeRange;
use proptest::prelude::*;
use std::vec::Vec;
use testkit::generators::{MARKET_TICK, market_series, market_series_and_m};

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

fn bits(entry: Entry) -> (u64, u64, u64) {
    (
        entry.rho().to_bits(),
        entry.squared_distance().to_bits(),
        entry.index().get().unwrap_or(u64::MAX),
    )
}

/// Windows finite at `positions` of `len`, the rest gaps.
fn finite_at(len: usize, positions: &[usize]) -> Vec<crate::stats::Window> {
    use crate::stats::{Window, WindowKind};
    let gap = Window {
        sum: 0,
        kind: WindowKind::NonFinite,
    };
    let mut windows = std::vec![gap; len];
    for &p in positions {
        windows[p].kind = WindowKind::Constant;
    }
    windows
}

/// A `k` past what a row can reach counts every reachable window: both sides equal a direct count
/// over every pair outside the exclusion zone, for several exclusion widths.
#[test]
fn a_k_past_each_rows_reach_counts_the_windows_it_reaches() {
    let windows = finite_at(38, &[1, 14, 22, 28, 31, 32, 33, 34, 37]);
    let finite = |i: usize| is_finite(&windows[i]);
    for exclusion in [0, 1, 3, 7] {
        for k in [1, 2, 5, 100, usize::MAX] {
            let (mut left, mut right) = ((0u128, 0u128), (0u128, 0u128));
            for i in (0..windows.len()).filter(|&i| finite(i)) {
                let reach =
                    |others: &mut dyn Iterator<Item = usize>| others.filter(|&j| finite(j)).count();
                let before = reach(&mut (0..i.saturating_sub(exclusion)));
                let after = reach(&mut ((i + exclusion + 1).min(windows.len())..windows.len()));
                for (side, reach) in [(&mut left, before), (&mut right, after)] {
                    let kept = reach.min(k);
                    side.0 += kept as u128;
                    side.1 += u128::from(kept == 0);
                }
            }
            let gaps = windows.iter().filter(|w| !is_finite(w)).count() as u128;
            let (got_left, got_right) = side_counts(
                windows.len(),
                finite,
                NonZeroUsize::new(k).unwrap(),
                exclusion,
            );
            assert_eq!(got_left.kept, left.0, "left, exclusion {exclusion}, k {k}");
            assert_eq!(
                got_right.kept, right.0,
                "right, exclusion {exclusion}, k {k}"
            );
            assert_eq!(got_left.list_len(), left.0 + left.1 + gaps);
            assert_eq!(got_right.list_len(), right.0 + right.1 + gaps);
        }
    }
}

/// Each side keeps its own count: on gappy windows the left side's differs from the right's.
#[test]
fn each_side_counts_the_neighbors_it_keeps() {
    let windows = finite_at(38, &[1, 14, 22, 28, 31, 32, 33, 34, 37]);
    let (left, right) = side_counts(
        windows.len(),
        |i| is_finite(&windows[i]),
        NonZeroUsize::new(3).unwrap(),
        3,
    );
    assert_eq!((left.kept, right.kept), (21, 15));
    // One list entry per kept neighbor, and one empty entry for each row that keeps none: the
    // 29 gaps and the finite rows with no neighbor on that side.
    assert_eq!(left.list_len(), 21 + 29 + 1);
    assert_eq!(right.list_len(), 15 + 29 + 2);
    let k_one = side_counts(
        windows.len(),
        |i| is_finite(&windows[i]),
        NonZeroUsize::MIN,
        3,
    );
    assert_eq!((k_one.0.list_len(), k_one.1.list_len()), (38, 38));
}

/// A join keeps `k` entries for each complete window of `a` and one empty entry for every other.
#[test]
fn a_join_lists_k_entries_per_finite_row_and_one_for_each_other() {
    let k = |k: usize| NonZeroUsize::new(k).unwrap();
    assert_eq!(join_list_len(10, 7, k(3)), 3 + 7 * 3);
    assert_eq!(join_list_len(10, 10, k(1)), 10);
    assert_eq!(join_list_len(5, 0, k(usize::MAX)), 5);
    assert_eq!(
        join_list_len(usize::MAX, usize::MAX, k(usize::MAX)),
        usize::MAX as u128 * usize::MAX as u128
    );
}

/// A top-`k` count past what the reservation limit once refused (`k` of 4 at more than 2^24 / 4
/// windows) is counted and reserved: only a size that cannot be allocated is a capacity error.
/// The counting is checked directly, since running 4.5 million windows is not a cheap test.
#[test]
fn k_of_four_past_the_reservation_limit_over_four_is_accepted() {
    use crate::capacity::reserved_vec;
    let windows = finite_at(4_500_000, &(0..4_500_000).collect::<Vec<_>>());
    assert!(4 * windows.len() > 1 << 24);
    let (left, right) = side_counts(
        windows.len(),
        |i| is_finite(&windows[i]),
        NonZeroUsize::new(4).unwrap(),
        2,
    );
    for side in [left, right] {
        assert!(side.list_len() > 1 << 24);
        assert!(reserved_vec::<(u64, Entry)>(side.list_len()).is_ok());
    }
    let gaps = finite_at(4_500_000, &[]);
    let (left, right) = side_counts(
        gaps.len(),
        |i| is_finite(&gaps[i]),
        NonZeroUsize::new(usize::MAX).unwrap(),
        2,
    );
    assert_eq!((left.kept, right.kept), (0, 0));
}

/// Through tiles that split every column's best pairs across partials and a merge into the
/// book, each column ends holding its `k` largest correlations to the windows on its right,
/// in order: the merge keeps every partial's top values, not only their best.
#[test]
fn tile_merges_leave_each_column_its_k_best_right_correlations() {
    let (m, e, k) = (8usize, 2usize, 3usize);
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut level = 0.0;
    let series: Vec<f64> = (0..260)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            level += ((state >> 58) % 5) as f64 - 2.0;
            level
        })
        .collect();
    let windows = series.len() - m + 1;
    let z: Vec<Vec<f64>> = (0..windows)
        .map(|i| {
            let w = &series[i..i + m];
            let mean = w.iter().sum::<f64>() / m as f64;
            let sd = (w.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / m as f64).sqrt();
            w.iter().map(|x| (x - mean) / sd).collect()
        })
        .collect();
    let rho =
        |i: usize, j: usize| z[i].iter().zip(&z[j]).map(|(a, b)| a * b).sum::<f64>() / m as f64;

    let scale = Scale::new(1.0).unwrap();
    let mu = NonZeroUsize::new(m).unwrap();
    let k_nz = NonZeroUsize::new(k).unwrap();
    let quantized = Quantized::build(&series, mu, scale);
    for (threads, band_rows, chunk_lanes, tile_lanes) in [(1, 4, 2, 6), (3, 3, 2, 4), (2, 40, 8, 8)]
    {
        let plan = TilePlan {
            band_rows,
            chunk_lanes,
            tile_lanes,
            k: k_nz,
            survivor: Default::default(),
        };
        let class = quantized.constant_class(None);
        let book = RightBook::new(&quantized.windows, k_nz, e, Vec::new(), class).unwrap();
        profile_on(
            &quantized,
            quantized.narrow_anchor(mu),
            mu,
            k_nz,
            NonZeroUsize::new(threads).unwrap(),
            chunk_lanes,
            BLOCK_ROWS,
            Some(plan),
            Vec::new(),
            Some(&book),
            None,
        );
        for column in 0..windows {
            let mut best: Vec<f64> = (column + e + 1..windows)
                .map(|row| rho(row, column))
                .collect();
            best.sort_by(|a, b| b.total_cmp(a));
            best.truncate(k);
            let held = book
                .groups
                .with_column(column, |group| group.top_of(column).to_vec());
            assert_eq!(held.len(), k.min(windows.saturating_sub(column + e + 1)));
            for (got, want) in held.iter().zip(&best) {
                assert!(
                    (got - want).abs() < 1e-9,
                    "column {column}: {held:?} vs {best:?}"
                );
            }
        }
    }
}

#[test]
fn a_tile_s_columns_run_from_its_furthest_start_to_its_last_neighbor() {
    let tile = |rows, qa, qb| Tile {
        band: 0,
        rows,
        column: 0,
        qa,
        qb,
    };
    assert_eq!(tile_columns(&tile(10..14, 2, 5), 3), 1..8);
    assert_eq!(tile_columns(&tile(4..6, 0, 9), 3), 0..2);
}

/// The most right-side candidates any accumulator held at once on `series`.
fn peak_right_candidates(
    series: &[f64],
    m: usize,
    threads: usize,
    plan: TilePlan,
    k: NonZeroUsize,
) -> usize {
    let scale = Scale::new(1.0).unwrap();
    let m = NonZeroUsize::new(m).unwrap();
    let quantized = Quantized::build(series, m, scale);
    let book = RightBook::new(
        &quantized.windows,
        k,
        crate::exclusion::zone(m.get()),
        Vec::new(),
        quantized.constant_class(None),
    )
    .unwrap();
    profile_on(
        &quantized,
        quantized.narrow_anchor(m),
        m,
        k,
        NonZeroUsize::new(threads).unwrap(),
        plan.chunk_lanes,
        BLOCK_ROWS,
        Some(plan),
        Vec::new(),
        Some(&book),
        None,
    );
    book.peak_cands.load(crate::sync::Ordering::SeqCst)
}

/// With every pair an exact tie the right side keeps a bounded number of candidates per kept
/// neighbor, the same bound at any series length.
#[test]
fn heavy_ties_keep_the_right_side_candidates_bounded() {
    let plan = TilePlan {
        band_rows: 128,
        chunk_lanes: 64,
        tile_lanes: 1024,
        k: NonZeroUsize::MIN,
        survivor: Default::default(),
    };
    let m = 8;
    let columns = RIGHT_GROUP.max(plan.band_rows + plan.tile_lanes + m);
    let bound = |k: usize| 16 * columns * k + 2 * 64 + BLOCK_ROWS * plan.chunk_lanes;
    let constant = |n: usize| std::vec![1.0; n];
    let repeat = |n: usize| -> Vec<f64> { (0..n).map(|k| [0.0, 3.0, 1.0, 4.0][k % 4]).collect() };
    let binary = |n: usize| -> Vec<f64> {
        (0..n)
            .map(|k| f64::from((k * k / 3 + k / 5) as u8 & 1))
            .collect()
    };
    for threads in [1, 2, 7] {
        for n in [1500, 4000] {
            for series in [constant(n), repeat(n), binary(n)] {
                for k in [1, 4] {
                    let peak = peak_right_candidates(&series, m, threads, plan, nz(k));
                    let bound = bound(k);
                    assert!(
                        peak <= bound,
                        "n {n}, k {k}, threads {threads}: {peak} > {bound}"
                    );
                }
            }
        }
    }
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    /// Every row `left_profile` returns, in order, equals `naive_left_profile`'s row on the
    /// same quantized samples, bit for bit: the one invariant this node exists to close.
    #[test]
    fn batch_eq_naive((series, m) in market_series_and_m(0..48)) {
        let scale = Scale::new(MARKET_TICK).unwrap();
        let samples: Vec<Option<i64>> = series.iter().map(|&x| to_ticks(x, scale)).collect();
        let expected = naive_left_profile(&samples, m);
        let actual = left_profile(&series, NonZeroUsize::new(m).unwrap(), scale);

        prop_assert_eq!(actual.len(), expected.len());
        for (&(row, entry), &(expected_row, expected_entry)) in actual.iter().zip(&expected) {
            prop_assert_eq!(row, expected_row);
            prop_assert_eq!(bits(entry), bits(expected_entry));
        }
    }
}

// ---------- the tiled narrow path: matches naive at every chunk length ----------

/// A market series and `m` in `{3, 4, 50}` — the window lengths the spec's own close-it-with
/// section names — so the tiled path is checked at both a tiny window and one close to the
/// cost the bench measures.
fn tiled_series_and_m(len: impl Into<SizeRange>) -> impl Strategy<Value = (Vec<f64>, usize)> {
    let len = len.into();
    prop_oneof![Just(3usize), Just(4usize), Just(50usize)]
        .prop_flat_map(move |m| (market_series(len.clone(), m), Just(m)))
}

/// `series` at `m`, checked against `naive` bit for bit at every length in `chunk_lengths`,
/// the chunk length changes only locality, never a bit.
fn assert_tiled_matches_naive(series: &[f64], m: usize, scale: Scale, chunk_lengths: &[usize]) {
    let samples: Vec<Option<i64>> = series.iter().map(|&x| to_ticks(x, scale)).collect();
    let expected = naive_left_profile(&samples, m);
    for &chunk_lanes in chunk_lengths {
        let actual =
            left_profile_with_chunk(series, NonZeroUsize::new(m).unwrap(), scale, chunk_lanes);
        assert_eq!(actual.len(), expected.len(), "chunk_lanes {chunk_lanes}");
        for (&(row, entry), &(expected_row, expected_entry)) in actual.iter().zip(&expected) {
            assert_eq!(row, expected_row);
            assert_eq!(
                bits(entry),
                bits(expected_entry),
                "chunk_lanes {chunk_lanes}, row {row}"
            );
        }
    }
}

proptest! {
    #![proptest_config(testkit::generators::seeded(24))]
    /// `left_profile_with_chunk` matches `naive` bit for bit at several chunk lengths — `1`,
    /// a small odd value, the default, and one larger than the series: the tiling is purely
    /// a loop-order change, so no chunk length changes a bit — on series long enough to
    /// span several blocks and chunks either side of [`super::BLOCK_ROWS`].
    #[test]
    fn tiled_matches_naive_across_chunk_lengths((series, m) in tiled_series_and_m(0..300)) {
        let scale = Scale::new(MARKET_TICK).unwrap();
        let chunk_lengths = [1usize, 37, default_chunk_lanes(), series.len() + 11];
        assert_tiled_matches_naive(&series, m, scale, &chunk_lengths);
    }
}

/// Every value identical: every window is Constant-kind, so every row (not just every
/// neighbor) takes the fallback path's own Constant-row handling — the tiled engine never
/// even starts a block. Checked anyway, at every chunk length, so a future change can't
/// silently assume a block is always non-empty.
#[test]
fn all_constant_series_matches_naive_at_every_chunk_length() {
    let series = std::vec![100.0_f64; 64];
    let scale = Scale::new(MARKET_TICK).unwrap();
    let chunk_lengths = [1usize, 5, default_chunk_lanes()];
    assert_tiled_matches_naive(&series, 4, scale, &chunk_lengths);
}

/// A row whose best Valid neighbor is only middlingly correlated (ρ ≈ 0.29, well inside
/// `(0, 0.5)`), plus one flat run (a Constant neighbor, ρ̂ fixed at `0.5`): the Constant
/// neighbor must still win. A Constant neighbor given `r = 0` with no real `bias` added
/// (mutant (a)), or a chunk containing one never routed through its own real `bias` at all
/// (mutant (e), [`chunk_has_non_valid`]), both rank it `0.0` in the hot loop instead of
/// `0.5` — still non-negative, so it is *not* caught by an all-negatively-correlated row
/// (every real candidate there is already below `0.0`), but here it is `0.0 < 0.29`, low
/// enough to be dropped from the candidate list before `Nearest` ever sees it, which is
/// exactly the failure mode the real `bias` dispatch exists to rule out. Every block is isolated by a NaN
/// gap at least `m` long on each side, so no window can ever straddle two blocks into some
/// other, unplanned-for correlation.
#[test]
fn a_constant_neighbor_beats_a_middling_positive_correlation() {
    let m = 4usize;
    let base = [1.0_f64, 2.0, 3.0, 5.0];
    let medium = [1.0_f64, 6.0, 9.0, 4.0]; // ρ ≈ 0.29 against `base`
    let mut series: Vec<f64> = Vec::new();
    for _ in 0..20 {
        series.extend_from_slice(&medium); // Valid, ρ ≈ 0.29 against the row
        series.extend(std::iter::repeat_n(f64::NAN, m));
    }
    series.extend(std::iter::repeat_n(7.0, m)); // Constant, ρ̂ = 0.5 against the row
    series.extend(std::iter::repeat_n(f64::NAN, m));
    for _ in 0..3 {
        series.extend_from_slice(&medium);
        series.extend(std::iter::repeat_n(f64::NAN, m));
    }
    series.extend_from_slice(&base); // the row

    let scale = Scale::new(MARKET_TICK).unwrap();
    let chunk_lengths = [1usize, 3, 37, default_chunk_lanes()];
    assert_tiled_matches_naive(&series, m, scale, &chunk_lengths);

    // Pinned directly too: the Constant neighbor (not the middling-correlation one) wins.
    let samples: Vec<Option<i64>> = series.iter().map(|&x| to_ticks(x, scale)).collect();
    let expected = naive_left_profile(&samples, m);
    let row = expected.len() - 1;
    assert_eq!(expected[row].1.rho(), 0.5);
}

/// An exact repeat beats a one-tick near miss even though both clamp to the same `ρ̂`
/// within `2ε`: only the exact fallback, fed every candidate the near-tie band admits, can
/// tell them apart. Mutant (b) (the candidate filter at `M_running` instead of
/// `M_running − 2ε`) drops the exact match from the candidate list here, since it is seen
/// one eligible neighbor before the row's own running maximum catches up to the near miss
/// two blocks later — `row`'s own fixed-window version of this case, driven here through
/// [`left_profile_with_chunk`] so the tiled candidate list (not `advance_and_rank_narrow`'s)
/// is what gets exercised.
#[test]
fn exact_repeat_beats_a_near_miss_through_the_tiled_path() {
    const A: [i64; 4] = [235_732_534, -190_400_685, 248_938_659, -501_918_519];
    const B: [i64; 4] = [235_732_534, -190_400_685, 248_938_660, -501_918_519];
    const FILLER: [i64; 4] = [0, 0, 0, 0];
    let m = 4usize;
    let mut ticks: Vec<i64> = Vec::new();
    ticks.extend_from_slice(&A); // window 0: the near miss
    ticks.extend_from_slice(&FILLER);
    ticks.extend_from_slice(&B); // window 8: the exact match
    ticks.extend_from_slice(&FILLER);
    ticks.extend_from_slice(&B); // window 16 (the last row): an exact repeat of window 8

    let series: Vec<f64> = ticks.iter().map(|&t| t as f64).collect();
    let scale = Scale::new(1.0).unwrap();
    let chunk_lengths = [1usize, 3, default_chunk_lanes()];
    assert_tiled_matches_naive(&series, m, scale, &chunk_lengths);

    // Pinned directly too, matching `row`'s own version of this case: the winner is the
    // exact match (window 8), not the near miss (window 0).
    for &chunk_lanes in &chunk_lengths {
        let actual =
            left_profile_with_chunk(&series, NonZeroUsize::new(m).unwrap(), scale, chunk_lanes);
        let (row, entry) = actual[16];
        assert_eq!(row, 16);
        assert_eq!(entry.index().get(), Some(8));
        assert_eq!(entry.rho(), 1.0);
        assert_eq!(entry.squared_distance(), 0.0);
    }
}

/// A walk with a flat stretch and a gap (so Constant and non-finite rows and neighbors take
/// the same pass), long enough that every chunk and block size below splits it many ways.
fn lagged_series() -> Vec<f64> {
    let mut state = 0x5EED_u64;
    let mut level = 1_000.0;
    let mut series = Vec::new();
    for k in 0..260 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        level += ((state >> 33) % 7) as f64 * MARKET_TICK - 3.0 * MARKET_TICK;
        series.push(match k {
            60..=79 => 1_000.0,
            120 => f64::NAN,
            _ => level,
        });
    }
    series
}

/// Every chunk width and every block height — 1, small, odd, the defaults and larger than
/// the series — gives the same bits as `naive`, and the output is not trivial: most rows
/// have a neighbor and they do not all pick the same one.
#[test]
fn chunk_width_and_block_height_never_change_a_bit() {
    let series = lagged_series();
    let m = 8usize;
    let scale = Scale::new(MARKET_TICK).unwrap();
    let samples: Vec<Option<i64>> = series.iter().map(|&x| to_ticks(x, scale)).collect();
    let expected = naive_left_profile(&samples, m);
    let n = series.len();

    let with_neighbor = expected.iter().filter(|(_, e)| e.index().get().is_some());
    assert!(with_neighbor.clone().count() > n / 2);
    let distinct: std::collections::BTreeSet<_> =
        with_neighbor.map(|(_, e)| e.index().get()).collect();
    assert!(
        distinct.len() > 10,
        "only {} distinct neighbors",
        distinct.len()
    );

    let chunks = [1, 2, 3, 7, 64, default_chunk_lanes(), n + 11];
    let blocks = [1, 2, 3, BLOCK_ROWS, 13, 100, n + 5];
    for chunk_lanes in chunks {
        for block_rows in blocks {
            let actual = left_profile_with_blocks(
                &series,
                NonZeroUsize::new(m).unwrap(),
                scale,
                chunk_lanes,
                block_rows,
            );
            assert_eq!(actual.len(), expected.len());
            for (&(row, entry), &(expected_row, expected_entry)) in actual.iter().zip(&expected) {
                assert_eq!(row, expected_row);
                assert_eq!(
                    bits(entry),
                    bits(expected_entry),
                    "chunk {chunk_lanes}, block {block_rows}, row {row}"
                );
            }
        }
    }
}

const GENERIC_SERIES: [f64; 16] = [
    1.0, 8.0, 2.0, 50.0, 60.0, 10.0, 3.0, 7.0, 42.0, 5.0, 9.0, 13.0, 61.0, 2.0, 48.0, 11.0,
];

/// A few generic rows of a plain series, pinned directly against `naive`'s own bits for
/// `(ρ, d², index)`, at a small, non-default chunk length — not a differential check, a
/// literal one, so a future change to any of the three has to look here and decide.
/// A few generic rows of a plain series, pinned directly against `naive`'s own bits for
/// `(ρ, d², index)`, at a small, non-default chunk length — not a differential check, a
/// literal one, so a future change to any of the three has to look here and decide.
#[test]
fn generic_rows_match_pinned_bits() {
    let m = 4usize;
    let scale = Scale::new(1.0).unwrap();
    let samples: Vec<Option<i64>> = GENERIC_SERIES.iter().map(|&x| to_ticks(x, scale)).collect();
    let expected = naive_left_profile(&samples, m);
    let actual = left_profile_with_chunk(&GENERIC_SERIES, NonZeroUsize::new(m).unwrap(), scale, 3);
    assert_eq!(actual.len(), expected.len());

    let pinned: [(u64, u64, u64, u64); 3] = [
        (6, 0x3fe7b5013de7d708, 0x400095fd843051f2, 2),
        (9, 0x3fef8a8d4a9c2c9e, 0x3fbd5cad58f4d889, 0),
        (12, 0x3fe73be1d5c1d16b, 0x4001883c547c5d2a, 8),
    ];
    for (row, rho_bits, dist_bits, index) in pinned {
        let (actual_row, entry) = actual[row as usize];
        assert_eq!(actual_row, row);
        assert_eq!(entry.rho().to_bits(), rho_bits, "row {row}");
        assert_eq!(entry.squared_distance().to_bits(), dist_bits, "row {row}");
        assert_eq!(entry.index().get(), Some(index), "row {row}");
        assert_eq!(bits(entry), bits(expected[row as usize].1), "row {row}");
    }
}

// ---------- AB-join ----------

/// Every `(a window, b window)` pair is fed to its row's pass exactly once, at every chunk
/// width, whichever side is longer.
#[test]
fn every_ab_pair_is_evaluated_exactly_once_at_every_chunk_width() {
    let scale = Scale::new(MARKET_TICK).unwrap();
    let m = NonZeroUsize::new(3).unwrap();
    let long = lagged_series();
    let short = &long[5..21];
    for (a, b) in [(&long[..], short), (short, &long[..])] {
        let n_b = b.len() - m.get() + 1;
        for chunk_lanes in [1, 2, 5, 64, default_chunk_lanes()] {
            super::VISITS.with(|v| v.borrow_mut().clear());
            let joined = super::ab_join_chunked(a, b, m, scale, scale, chunk_lanes);
            let samples = |s: &[f64]| s.iter().map(|&x| to_ticks(x, scale)).collect::<Vec<_>>();
            let expected = crate::naive::naive_ab_join(&samples(a), &samples(b), m.get());
            for (&(_, got), &(_, want)) in joined.iter().zip(&expected) {
                assert_eq!(bits(got), bits(want), "chunk {chunk_lanes}");
            }
            let mut seen = super::VISITS.with(|v| v.borrow().clone());
            seen.sort();
            assert_eq!(seen.len(), joined.len());
            for (row, ((seen_row, mut visits), (joined_row, _))) in
                seen.into_iter().zip(joined).enumerate()
            {
                assert_eq!((row, row as u64), (seen_row, joined_row));
                visits.sort();
                assert_eq!(visits, (0..n_b).collect::<Vec<_>>(), "chunk {chunk_lanes}");
            }
        }
    }
}

// ---------- every eligible pair evaluated exactly once ----------

/// On the engine itself: every eligible `(row, neighbor)` pair is fed to its row's pass
/// exactly once — none twice, none missing, none inside the exclusion zone — at every chunk
/// width and block height. A pair the constant class settles is accounted for without being fed:
/// the pass leaves some of them out, a block at a time.
#[test]
fn every_eligible_pair_is_evaluated_exactly_once_at_every_tiling() {
    let m: usize = 3;
    let exclusion = crate::exclusion::zone(m);
    let series = lagged_series();
    let scale = Scale::new(MARKET_TICK).unwrap();
    let num_windows = series.len() - m + 1;
    let n = series.len();
    let samples: Vec<Option<i64>> = series.iter().map(|&x| to_ticks(x, scale)).collect();
    let settled = super::class_settles(&samples, m, exclusion);
    let mut left_out = 0;

    for chunk_lanes in [1, 2, 3, 7, 64, default_chunk_lanes(), n + 11] {
        for block_rows in [1, 2, 3, BLOCK_ROWS, 13, 100, n + 5] {
            super::VISITS.with(|v| v.borrow_mut().clear());
            left_profile_with_blocks(
                &series,
                NonZeroUsize::new(m).unwrap(),
                scale,
                chunk_lanes,
                block_rows,
            );
            let mut seen = super::VISITS.with(|v| v.borrow().clone());
            seen.sort();
            let rows: Vec<usize> = seen.iter().map(|&(row, _)| row).collect();
            let expected_rows: Vec<usize> = (exclusion + 1..num_windows).collect();
            assert_eq!(
                rows, expected_rows,
                "chunk {chunk_lanes}, block {block_rows}"
            );
            for (row, mut visits) in seen {
                visits.sort();
                let expected: Vec<usize> = (0..eligible_count(row, exclusion))
                    .filter(|&j| visits.binary_search(&j).is_ok() || !settled(row, j))
                    .collect();
                assert_eq!(
                    visits, expected,
                    "chunk {chunk_lanes}, block {block_rows}, row {row}"
                );
                left_out += eligible_count(row, exclusion) - visits.len();
            }
        }
    }
    assert!(left_out > 0, "the series has blocks the class settles");
}

#[test]
fn eligible_count_excludes_the_zone_and_includes_everything_outside_it() {
    // m = 7: exclusion = ceil(7/4) = 2.
    let exclusion = crate::exclusion::zone(7usize);
    assert_eq!(exclusion, 2);
    assert_eq!(eligible_count(0, exclusion), 0);
    assert_eq!(eligible_count(2, exclusion), 0); // row == exclusion: still none eligible
    assert_eq!(eligible_count(3, exclusion), 1); // j = 0 just outside the zone
    assert_eq!(eligible_count(10, exclusion), 8);
}

// ---------- the narrow path: the whole-series anchor decision ----------

fn bound(m: usize) -> i64 {
    i64::from(i32::MAX) / m as i64
}

#[test]
fn narrow_anchor_covers_a_market_level_series() {
    let m = NonZeroUsize::new(50).unwrap();
    let scale = Scale::new(1.0).unwrap();
    let level = 10_000_000_000i64;
    let series: Vec<f64> = (0..5_000)
        .map(|k| (level + (k % 97) as i64 - 48) as f64)
        .collect();
    let quantized = Quantized::build(&series, m, scale);
    assert!(
        quantized.narrow_anchor(m).is_some(),
        "a market-level series' own tiny span fits comfortably within one anchor"
    );
}

#[test]
fn narrow_anchor_is_none_past_twice_the_bound() {
    let m = NonZeroUsize::new(4).unwrap();
    let scale = Scale::new(1.0).unwrap();
    let b = bound(m.get());
    // Span exactly `2 * b`: still one anchor's worth.
    let at_bound: Vec<f64> = std::vec![0.0, (2 * b) as f64, 0.0, (2 * b) as f64];
    let quantized = Quantized::build(&at_bound, m, scale);
    assert!(quantized.narrow_anchor(m).is_some());

    // One past it: no anchor could cover both ends any longer.
    let past_bound: Vec<f64> = std::vec![0.0, (2 * b + 1) as f64, 0.0, (2 * b + 1) as f64];
    let quantized = Quantized::build(&past_bound, m, scale);
    assert!(quantized.narrow_anchor(m).is_none());
}

/// Runs `left_profile` on `series` and checks every row against `naive_left_profile` bit
/// for bit.
fn assert_left_profile_matches_naive(series: &[f64], m: usize, scale: Scale) {
    let samples: Vec<Option<i64>> = series.iter().map(|&x| to_ticks(x, scale)).collect();
    let expected = naive_left_profile(&samples, m);
    let actual = left_profile(series, NonZeroUsize::new(m).unwrap(), scale);
    assert_eq!(actual.len(), expected.len());
    for (&(row, entry), &(expected_row, expected_entry)) in actual.iter().zip(&expected) {
        assert_eq!(row, expected_row);
        assert_eq!(bits(entry), bits(expected_entry), "row {row}");
    }
}

/// A series whose tick span is exactly `2 * bound(m)`: the tightest span a single anchor
/// can still cover, so `narrow_anchor` takes the narrow path (mirrors `stream.rs`'s own
/// `a_span_of_exactly_twice_the_bound_stays_narrow`, through `left_profile` instead of a
/// stream's incremental re-anchoring). Bit-exact against `naive`.
#[test]
fn batch_eq_naive_at_exactly_twice_the_bound() {
    let m = 4;
    let scale = Scale::new(1.0).unwrap();
    let b = bound(m);
    let series: Vec<f64> = std::vec![
        0.0,
        1.0,
        2.0,
        (2 * b) as f64,
        (2 * b - 1) as f64,
        (2 * b - 2) as f64,
        1.0,
        2.0,
    ];
    let m_nz = NonZeroUsize::new(m).unwrap();
    assert!(
        Quantized::build(&series, m_nz, scale)
            .narrow_anchor(m_nz)
            .is_some(),
        "a span of exactly 2 * bound still fits one anchor"
    );
    assert_left_profile_matches_naive(&series, m, scale);
}

/// One tick past the boundary above: no anchor can cover both ends any longer, so
/// `narrow_anchor` must return `None` and `left_profile` must run the wide fallback
/// (mirrors `stream.rs`'s own `a_span_one_past_twice_the_bound_switches_to_wide`). Still
/// bit-exact against `naive`. Catches the anchor bound off by one (`> 2 * bound` relaxed
/// to `> 2 * bound + 1`): under that mutant this span wrongly takes the narrow path, whose
/// offsets no longer fit `Narrow::fits`, so this is where the mismatch against `naive`
/// (or the narrowing's own `debug_assert`) surfaces.
#[test]
fn batch_eq_naive_one_past_twice_the_bound() {
    let m = 4;
    let scale = Scale::new(1.0).unwrap();
    let b = bound(m);
    let series: Vec<f64> = std::vec![
        0.0,
        1.0,
        2.0,
        (2 * b + 1) as f64,
        (2 * b) as f64,
        (2 * b - 1) as f64,
        1.0,
        2.0,
    ];
    let m_nz = NonZeroUsize::new(m).unwrap();
    assert!(
        Quantized::build(&series, m_nz, scale)
            .narrow_anchor(m_nz)
            .is_none(),
        "a span one past 2 * bound no longer fits any one anchor"
    );
    assert_left_profile_matches_naive(&series, m, scale);
}

/// A quantized series at scale 1 stepping between levels around the one-anchor bound of `m`, so
/// that some tiles of any cut need the exact pass and others do not.
#[cfg(not(any(loom, shuttle)))]
fn stepped(len: usize, m: usize, seed: u64) -> Quantized {
    let bound = bound(m) as f64;
    let levels = [0.0, bound, 3.0 * bound, -2.0 * bound];
    let mut state = seed | 1;
    let mut level = 0.0;
    let series: Vec<f64> = (0..len)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            if i % 23 == 0 {
                level = levels[(state % 4) as usize];
            }
            level + (state >> 8) as f64 % 5.0
        })
        .collect();
    Quantized::build(&series, nz(m), Scale::new(1.0).unwrap())
}

/// The tile predicates the cold route prices the exact pass with are the ones the engines run by:
/// the tiles they call wide are exactly the tiles the engines report as wide.
#[test]
#[cfg(not(any(loom, shuttle)))]
fn the_tiles_the_cold_route_calls_wide_are_the_tiles_the_engines_run_wide() {
    use super::ab_join::ab_join_on;
    use super::cold::{JoinCut, wide_cells_join, wide_cells_triangle};
    use super::plan::TileCut;
    use super::tile::{join_tile_is_wide, triangle_tile_is_wide};
    let _serial = crate::route::forced_route(crate::route::RouteForce::default());
    for (len, m, seed) in [
        (300usize, 6usize, 1u64),
        (260, 4, 2),
        (410, 9, 3),
        (150, 3, 4),
    ] {
        let (single, threads) = (NonZeroUsize::MIN, nz(2));
        let plan = TilePlan::fixed(nz(7), nz(3), nz(9));
        let q = stepped(len, m, seed);
        let windows = q.windows.len();

        let (_, report) = profile_on(
            &q,
            None,
            nz(m),
            single,
            threads,
            3,
            BLOCK_ROWS,
            Some(plan),
            Vec::new(),
            None,
            None,
        );
        let cut = TileCut::triangle(windows, m, plan, None);
        let wide = cut
            .tiles
            .iter()
            .filter(|tile| triangle_tile_is_wide(&q.ticks, nz(m), tile, 3))
            .count();
        assert_eq!(wide, report.wide_tiles, "triangle, len {len}, m {m}");
        assert!(wide > 0 && wide < report.tiles, "the series mixes both");
        let key = |tile: &Tile| (tile.band, tile.rows.clone(), tile.qa, tile.qb);
        let keys = |tiles: &[Tile]| tiles.iter().map(key).collect::<Vec<_>>();
        let mut visited = Vec::new();
        TileCut::visit_triangle(windows, m, plan, |tile| visited.push(key(tile)));
        assert_eq!(visited, keys(&cut.tiles));
        assert!(wide_cells_triangle(&q.ticks, nz(m), plan) > 0);

        let other = stepped(len / 2 + 7, m, seed + 10);
        let (_, report) = ab_join_on(
            &q,
            &other,
            nz(m),
            threads,
            3,
            Some(plan),
            single,
            Vec::new(),
            None,
        );
        let (n_a, n_b) = (windows, other.windows.len());
        let join = TileCut::join(n_a, n_b, plan, None);
        let wide = join
            .tiles
            .iter()
            .filter(|tile| join_tile_is_wide(&q.ticks, &other.ticks, nz(m), tile, 3))
            .count();
        assert_eq!(wide, report.wide_tiles, "join, len {len}, m {m}");
        let mut visited = Vec::new();
        TileCut::visit_join(n_a, n_b, plan, |tile| visited.push(key(tile)));
        assert_eq!(visited, keys(&join.tiles));
        // A tile wider than the matrix is cut down to it, in the visit as in the cut.
        let wide = TilePlan::fixed(nz(7), nz(3), nz(100_000));
        let mut visited = Vec::new();
        TileCut::visit_join(n_a, n_b, wide, |tile| visited.push(key(tile)));
        assert_eq!(visited, keys(&TileCut::join(n_a, n_b, wide, None).tiles));
        let mut visited = Vec::new();
        TileCut::visit_triangle(windows, m, wide, |tile| visited.push(key(tile)));
        assert_eq!(
            visited,
            keys(&TileCut::triangle(windows, m, wide, None).tiles)
        );
        let pairs = (n_a * n_b) as u64;
        let cells = wide_cells_join(&q.ticks, &other.ticks, nz(m), JoinCut::Tiles(plan));
        assert!(cells > 0 && cells < pairs, "join, len {len}, m {m}");
        let one_pass = wide_cells_join(&q.ticks, &other.ticks, nz(m), JoinCut::OnePass);
        assert_eq!(one_pass, pairs, "neither fits one anchor");
    }
}

/// The cross-term bound counts every window the hot pass reads, whatever its kind.
mod cross_range {
    use super::{Quantized, nz};
    use crate::columns::CrossRange;
    use crate::quantize::Scale;
    use crate::stats::{Series, WindowKind};
    use std::{vec, vec::Vec};

    fn unit() -> Scale {
        Scale::new(1.0).unwrap()
    }

    #[test]
    fn windows_with_gaps_carry_the_widest_window() {
        // Leading gaps read as zero ticks, far from every sample.
        let mut series = vec![f64::NAN; 3];
        series.extend((0..40).map(|k| 1.0e8 + f64::from(k % 3)));
        let q = Quantized::build(&series, nz(8), unit());
        assert_eq!(q.range, CrossRange::over(&q.ticks, nz(8)));
        assert!(!q.range.fits_fast_conversion());
        let valid: Vec<u64> = q
            .windows
            .iter()
            .filter_map(|w| match w.kind {
                WindowKind::Valid {
                    variance_numerator, ..
                } => Some(variance_numerator),
                _ => None,
            })
            .collect();
        assert!(!valid.is_empty() && valid.iter().all(|&v| v < 1 << 51));
    }

    #[test]
    fn flagged_constant_windows_still_count() {
        let m = nz(4);
        let series: Vec<f64> = (0..30).map(|k| f64::from(k % 7) * 1.0e8).collect();
        let flags = vec![true; 27];
        let flagged = Series::with_constant_windows(&series, m, &flags).unwrap();
        let q = Quantized::build(flagged, m, unit());
        assert!(q.windows.iter().all(|w| w.kind == WindowKind::Constant));
        assert_eq!(q.range, Quantized::build(&series, m, unit()).range);
        assert!(!q.range.fits_fast_conversion());
    }

    #[test]
    fn stood_in_windows_are_measured_after_the_stand_in() {
        let m = nz(4);
        let mut series = vec![0.0; 30];
        series[11] = 1000.0;
        series[12] = 1.0e9;
        let mut q = Quantized::build(&series, m, unit());
        assert!(!q.range.fits_fast_conversion());
        let had = vec![true; 30];
        let mut outliers = vec![false; 30];
        outliers[12] = true;
        let spilled: Vec<bool> = (0..27).map(|row| (9..=12).contains(&row)).collect();
        q.standing_in(&had, &outliers, &spilled, m).unwrap();
        assert_eq!(q.ticks[12], 1000);
        // The widest windows now hold both thousands; each of them is spilled, so non-finite.
        assert_eq!(q.range, CrossRange::over(&[0, 0, 1000, 1000], m));
        assert_eq!(q.range, CrossRange::over(&q.ticks, m));
        assert!((9..=11).all(|row| q.windows[row].kind == WindowKind::NonFinite));
        assert!(q.range.fits_fast_conversion());
    }

    #[test]
    fn a_threaded_build_takes_the_same_bound() {
        let series: Vec<f64> = (0..5000)
            .map(|k| {
                if k % 997 == 0 {
                    f64::NAN
                } else {
                    f64::from((k * 37) % 101) * 1.0e6
                }
            })
            .collect();
        for m in [1, 3, 64] {
            let one = Quantized::build(&series, nz(m), unit());
            let many = Quantized::build_on(
                &series,
                nz(m),
                unit(),
                nz(8),
                crate::batch::plan::Engine::Ungated,
            );
            assert_eq!(one.range, many.range, "m {m}");
            assert_eq!(one.range, CrossRange::over(&one.ticks, nz(m)), "m {m}");
        }
    }
}
