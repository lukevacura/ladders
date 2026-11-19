extern crate std;

use super::{Columns, RowInputs, advance_and_rank, resize_and_shift};
use crate::entry::Entry;
use crate::estimate::{F64, Producer, band_floor};
use crate::kernel::Metric;
use crate::kernel_neon::Conversion;
use crate::nearest::Nearest;
use crate::quantize::{Scale, to_ticks};
use crate::stats::{Window, WindowKind, WindowStats};
use crate::znorm::{TickWindow, ZNorm};
use core::num::NonZeroUsize;
use proptest::prelude::*;
use std::vec::Vec;
use testkit::generators::{MARKET_TICK, market_series_and_m};

fn bits(entry: Entry) -> (u64, u64, u64) {
    (
        entry.rho().to_bits(),
        entry.squared_distance().to_bits(),
        entry.index().get().unwrap_or(u64::MAX),
    )
}

// ---------- the fast pass's fixed ratios and its routing through `rank_estimate` are
// ---------- pinned directly in `columns.rs`, the one place that now encodes them.

/// Ticks built straight from clean `i64` values, with no missing sample: used only to
/// hand-build a series whose exact tick content is chosen directly, not quantized from
/// `f64`.
fn windows_of(ticks: &[i64], m: usize) -> Vec<Window> {
    let mut stats = WindowStats::new(NonZeroUsize::new(m).unwrap());
    ticks
        .iter()
        .filter_map(|&tick| stats.push(tick, true))
        .collect()
}

/// Quantized, with a missing tick (filled with the last valid one, or `0` before any) kept
/// distinct from a present one — the same distinction `GrowingStream` and `stats::classify`
/// need to tell a gap from a merely flat run.
fn sanitize_to_ticks(series: &[f64], scale: Scale) -> Vec<(i64, bool)> {
    let mut last = 0i64;
    series
        .iter()
        .map(|&x| match to_ticks(x, scale) {
            Some(tick) => {
                last = tick;
                (tick, true)
            }
            None => (last, false),
        })
        .collect()
}

/// The sanitized ticks and every completed window's statistics, for a quantized series
/// that may contain gaps.
fn ticks_and_windows_with_gaps(series: &[f64], m: usize, scale: Scale) -> (Vec<i64>, Vec<Window>) {
    let sanitized = sanitize_to_ticks(series, scale);
    let ticks: Vec<i64> = sanitized.iter().map(|&(tick, _)| tick).collect();
    let mut stats = WindowStats::new(NonZeroUsize::new(m).unwrap());
    let windows: Vec<Window> = sanitized
        .iter()
        .filter_map(|&(tick, had_tick)| stats.push(tick, had_tick))
        .collect();
    (ticks, windows)
}

struct Setup {
    ticks: Vec<i64>,
    sum: Vec<i64>,
    r: Vec<f64>,
    bias: Vec<f64>,
    variance: Vec<u64>,
    windows: Vec<Window>,
}

fn setup(ticks: Vec<i64>, windows: Vec<Window>) -> Setup {
    let mut sum = Vec::with_capacity(windows.len());
    let mut r = Vec::with_capacity(windows.len());
    let mut bias = Vec::with_capacity(windows.len());
    let mut variance = Vec::with_capacity(windows.len());
    for &window in &windows {
        super::push_window(window, &mut sum, &mut r, &mut bias, &mut variance);
    }
    Setup {
        ticks,
        sum,
        r,
        bias,
        variance,
        windows,
    }
}

fn columns(setup: &Setup) -> Columns<'_> {
    Columns {
        ticks: &setup.ticks,
        sum: &setup.sum,
        r: &setup.r,
        bias: &setup.bias,
        variance: &setup.variance,
    }
}

/// The row's exact entry, independent of `advance_and_rank`: a fresh `ZNorm::start` for
/// every eligible pair (bit-identical to any incremental path, since wrapping arithmetic
/// is exact ring arithmetic and so is path-independent) offered to `Nearest` in index
/// order.
fn reference_entry(setup: &Setup, m: usize, exclusion: usize, row: usize) -> (Entry, Vec<i64>) {
    let window = setup.windows[row];
    let mut nearest = Nearest::new(&window, m);
    let row_tw = TickWindow {
        entering: setup.ticks[row + m - 1],
        leaving: if row == 0 { 0 } else { setup.ticks[row - 1] },
        stats: window,
    };
    let count = row.saturating_sub(exclusion);
    let mut qts = Vec::with_capacity(count);
    for j in 0..count {
        let neighbor = setup.windows[j];
        let neighbor_tw = TickWindow {
            entering: setup.ticks[j + m - 1],
            leaving: if j == 0 { 0 } else { setup.ticks[j - 1] },
            stats: neighbor,
        };
        let qt = ZNorm::start(
            &setup.ticks[row..row + m],
            &row_tw,
            &setup.ticks[j..j + m],
            &neighbor_tw,
        );
        let n = crate::znorm::centered_cross(qt, &row_tw, &neighbor_tw, m);
        let index = crate::index::SampleIndex::new(j as u64).unwrap();
        nearest.offer(n, &neighbor, index);
        qts.push(qt);
    }
    (nearest.finish(), qts)
}

/// Drives `advance_and_rank` row by row over the whole series, checking every row's
/// `Entry` and every diagonal's final `QT` against `reference_entry`, bit for bit.
fn check_series(series: &[f64], m: usize) {
    let scale = Scale::new(MARKET_TICK).unwrap();
    if series.len() < m {
        return;
    }
    let (ticks, windows) = ticks_and_windows_with_gaps(series, m, scale);
    let setup = setup(ticks, windows);
    let cols = columns(&setup);
    let exclusion = crate::exclusion::zone(m);

    let mut qt: Vec<i64> = Vec::new();
    let mut scratch: Vec<f64> = Vec::new();
    for row in 0..setup.windows.len() {
        let count = row.saturating_sub(exclusion);
        resize_and_shift(&mut qt, count);
        scratch.resize(count, 0.0);
        let row_inputs = RowInputs {
            row,
            window: setup.windows[row],
        };
        let entry = advance_and_rank(&mut qt, &cols, &row_inputs, m, exclusion, &mut scratch);
        let (expected_entry, expected_qt) = reference_entry(&setup, m, exclusion, row);
        assert_eq!(
            bits(entry),
            bits(expected_entry),
            "row {row}, m {m}, series {series:?}"
        );
        // `advance_and_rank`'s `qt[j]` and `expected_qt[j]` are both indexed by neighbor
        // directly.
        for (j, (&value, &expected)) in qt.iter().zip(&expected_qt).enumerate() {
            assert_eq!(
                value, expected,
                "row {row}, neighbor {j}, m {m}, series {series:?}"
            );
        }
    }
}

/// Calls `advance_and_rank` for every row `0..=through_row` in order, maintaining `qt`
/// and `scratch` the way a real caller must (each diagonal's state only exists because
/// every earlier row built it up): returns `through_row`'s `Entry` and its final `qt`.
fn drive(
    cols: &Columns,
    windows: &[Window],
    m: usize,
    exclusion: usize,
    through_row: usize,
) -> (Entry, Vec<i64>) {
    let mut qt: Vec<i64> = Vec::new();
    let mut scratch: Vec<f64> = Vec::new();
    let mut entry = Entry::NONE;
    for (row, &window) in windows.iter().enumerate().take(through_row + 1) {
        let count = row.saturating_sub(exclusion);
        resize_and_shift(&mut qt, count);
        scratch.resize(count, 0.0);
        let row_inputs = RowInputs { row, window };
        entry = advance_and_rank(&mut qt, cols, &row_inputs, m, exclusion, &mut scratch);
    }
    (entry, qt)
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    /// Every row's `Entry` and every diagonal's `QT`, driving `advance_and_rank` over
    /// market-shaped series (gaps, flat runs, exact repeats included), match a reference
    /// built independently from `ZNorm::start` and `Nearest`, bit for bit.
    #[test]
    fn matches_the_reference_bit_for_bit((series, m) in market_series_and_m(0..40)) {
        check_series(&series, m);
    }
}

// ---------- targeted: an exact repeat beats a near miss even when ρ̂ ties ----------

#[test]
fn exact_repeat_beats_a_near_miss_even_when_rho_hat_ties() {
    // `A` is a one-tick near miss of `B`; the row repeats `B` exactly. Both pairs' ρ̂
    // clamp to exactly 1.0 (within 2ε of each other), so only the exact fallback inside
    // `Nearest` can tell the true exact match (window 8, ρ² = 1/1 exactly) from the near
    // miss (window 0). Windows overlap for `m > 1`, so `A`, a constant filler (long
    // enough to isolate every straddling window far from either block) and `B` each sit
    // in their own block, well apart.
    const A: [i64; 4] = [235_732_534, -190_400_685, 248_938_659, -501_918_519];
    const B: [i64; 4] = [235_732_534, -190_400_685, 248_938_660, -501_918_519];
    const FILLER: [i64; 4] = [0, 0, 0, 0];
    let m: usize = 4;
    let exclusion = crate::exclusion::zone(m); // 1
    let mut ticks = Vec::new();
    ticks.extend_from_slice(&A); // window 0: the near miss
    ticks.extend_from_slice(&FILLER);
    ticks.extend_from_slice(&B); // window 8: the exact match
    ticks.extend_from_slice(&FILLER);
    ticks.extend_from_slice(&B); // window 16: the row, an exact repeat of window 8

    let windows = windows_of(&ticks, m);
    let setup = setup(ticks, windows);
    let cols = columns(&setup);

    let row = 16;
    let (entry, _) = drive(&cols, &setup.windows, m, exclusion, row);
    assert_eq!(entry.index().get(), Some(8));
    assert_eq!(entry.rho(), 1.0);
    assert_eq!(entry.squared_distance(), 0.0);
}

// ---------- targeted: the chunk-skip rescan's own blind spot ----------

/// The row's unique exact match sits at window 63 — `RESCAN_CHUNK − 1`, the last lane of
/// `rescan_near_tie`'s own first chunk — with 76 further windows after it (`count = 139`,
/// past two full chunks) so the scenario cannot be satisfied by scanning only one chunk by
/// accident. Every other tick comes from a generic walk, so nothing else ties or nears the
/// planted match; `reference_entry` (the brute-force oracle, independent of any chunking)
/// confirms window 63 really is this row's true best before checking the real path against
/// it. A chunk-bounds off-by-one that drops a chunk's own last qualifying lane would skip
/// window 63 here and return the wrong neighbor (or none).
#[test]
fn a_winner_at_a_rescan_chunk_s_last_lane_is_not_dropped() {
    let m: usize = 4;
    let exclusion = crate::exclusion::zone(m); // 1
    let row = 140; // count = 139: more than two 64-lane chunks
    let winner = 63; // RESCAN_CHUNK - 1

    let mut state = 0xC0FFEE_u64;
    let mut ticks: Vec<i64> = (0..row + m)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) % 2001) as i64 - 1000
        })
        .collect();
    // Far outside the generic walk's own `[-1000, 1000)` range, so the planted exact
    // match can't coincide with anything the walk happens to produce elsewhere.
    const PATTERN: [i64; 4] = [500_000, -300_000, 700_000, -900_000];
    ticks[winner..winner + m].copy_from_slice(&PATTERN);
    ticks[row..row + m].copy_from_slice(&PATTERN);

    let windows = windows_of(&ticks, m);
    let setup = setup(ticks, windows);
    let cols = columns(&setup);

    let (expected_entry, _) = reference_entry(&setup, m, exclusion, row);
    assert_eq!(
        expected_entry.index().get(),
        Some(winner as u64),
        "construction check: the planted exact match must be the row's own true best"
    );

    let (entry, _) = drive(&cols, &setup.windows, m, exclusion, row);
    assert_eq!(bits(entry), bits(expected_entry));
}

// ---------- targeted: a non-finite neighbor never wins, and never shifts M ----------

#[test]
fn non_finite_neighbor_is_excluded_from_the_maximum() {
    let m: usize = 3;
    let exclusion = crate::exclusion::zone(m); // 1
    // window 0: a gap (non-finite); window 1: valid; window 2: valid; window 3: the row.
    let scale = Scale::new(MARKET_TICK).unwrap();
    let series = [f64::NAN, 1.0, 2.0, 9.0, 3.0, 5.0, 1.0, 3.0, 5.0, 4.0];
    let (ticks, windows) = ticks_and_windows_with_gaps(&series, m, scale);
    let setup = setup(ticks, windows);
    assert_eq!(setup.windows[0].kind, WindowKind::NonFinite);
    let cols = columns(&setup);

    let row = 4;
    let (entry, _) = drive(&cols, &setup.windows, m, exclusion, row);
    let (expected_entry, _) = reference_entry(&setup, m, exclusion, row);
    assert_eq!(bits(entry), bits(expected_entry));
    assert_ne!(entry.index().get(), Some(0));
}

// ---------- targeted: a non-finite row still advances every diagonal ----------

#[test]
fn non_finite_row_is_none_but_still_advances_every_diagonal() {
    let m: usize = 3;
    let exclusion = crate::exclusion::zone(m);
    let scale = Scale::new(MARKET_TICK).unwrap();
    // The row itself (the last window) contains a gap.
    let series = [1.0, 2.0, 9.0, 3.0, 5.0, 1.0, f64::NAN, 5.0];
    let (ticks, windows) = ticks_and_windows_with_gaps(&series, m, scale);
    let row = windows.len() - 1;
    let setup = setup(ticks, windows);
    assert_eq!(setup.windows[row].kind, WindowKind::NonFinite);
    let cols = columns(&setup);

    let (entry, qt) = drive(&cols, &setup.windows, m, exclusion, row);
    assert_eq!(bits(entry), bits(Entry::NONE));
    let (_, expected_qt) = reference_entry(&setup, m, exclusion, row);
    for (j, (&value, &expected)) in qt.iter().zip(&expected_qt).enumerate() {
        assert_eq!(value, expected, "neighbor {j}");
    }
}

// ---------- RowAcc / pass_row_narrow: tile-size independence and top-k ----------

/// Every row of `series` through `pass_row_narrow` with `tile_len`-lane chunks and `k`
/// nearest, driven the way `GrowingStream` drives it (`resize_and_shift` keeps each
/// diagonal in place). `None` if the series' ticks do not fit one anchor.
fn narrow_rows(
    series: &[f64],
    m: usize,
    tile_len: usize,
    k: usize,
    split: bool,
) -> Option<Vec<Vec<Entry>>> {
    use super::{ChunkCols, NarrowColumns, RowAcc, pass_chunk};
    use crate::columns;
    use crate::kernel_neon::RowScalars;
    use crate::narrow::{self, Anchor};
    use crate::stats::WindowKind;
    let scale = Scale::new(MARKET_TICK).unwrap();
    if series.len() < m {
        return None;
    }
    let m_nz = NonZeroUsize::new(m).unwrap();
    let (ticks, windows) = ticks_and_windows_with_gaps(series, m, scale);
    let setup = setup(ticks, windows);
    let anchor = Anchor {
        tick: *setup.ticks.iter().min().unwrap(),
    };
    if setup
        .ticks
        .iter()
        .any(|&t| anchor.offset(t, m_nz).is_none())
    {
        return None;
    }
    let ticks_off: Vec<i32> = setup
        .ticks
        .iter()
        .map(|&t| narrow::tick_offset(anchor, m_nz, t))
        .collect();
    let sum_off: Vec<i32> = setup
        .windows
        .iter()
        .map(|w| narrow::sum_offset(anchor, m_nz, w.sum))
        .collect();
    let cols = NarrowColumns {
        ticks: &ticks_off,
        sum: &sum_off,
        r: &setup.r,
        bias: &setup.bias,
        variance: &setup.variance,
        conversion: Conversion::FULL_RANGE,
    };
    let exclusion = crate::exclusion::zone(m);
    let k = NonZeroUsize::new(k).unwrap();
    let mut acc = RowAcc::new();
    let mut other = RowAcc::new();
    let mut qt: Vec<i64> = Vec::new();
    let mut tile = std::vec![0.0f64; tile_len];
    let mut out = Vec::new();
    for (row, &window) in setup.windows.iter().enumerate() {
        let count = row.saturating_sub(exclusion);
        resize_and_shift(&mut qt, count);
        let mut nearest = Nearest::with_k(&window, m, k, usize::MAX);
        acc.begin(&window, k, usize::MAX);
        let inputs = RowInputs { row, window };
        if split && count > 0 {
            // Chunks alternate between two accumulators, merged at the end: the tiling
            // and threading shape `merge` exists for.
            other.begin(&window, k, usize::MAX);
            let scalars = RowScalars {
                me: (i64::from(ticks_off[row + m - 1]) * m as i64) as i32,
                ml: (i64::from(ticks_off[row - 1]) * m as i64) as i32,
                si: sum_off[row],
                ri: match window.kind {
                    WindowKind::Valid { inv_sqrt, .. } => inv_sqrt,
                    _ => 0.0,
                },
                conversion: Conversion::FULL_RANGE,
            };
            let mut start = 0i64;
            for t in 0..m {
                start = start.wrapping_add(
                    i64::from(ticks_off[row + t]).wrapping_mul(i64::from(ticks_off[t])),
                );
            }
            qt[0] = start.wrapping_mul(m as i64);
            let nb0 =
                |_: usize| columns::window_of(0, setup.r[0], setup.bias[0], setup.variance[0]);
            acc.start_lane(
                qt[0],
                &ChunkCols {
                    enter: &[],
                    leave: &[],
                    sum: &sum_off[..1],
                    r: &setup.r[..1],
                    bias: &setup.bias[..1],
                    first: 0,
                    neighbor: &nb0,
                },
                scalars,
            );
            let (mut lo, mut turn) = (1usize, false);
            while lo < count {
                let hi = (lo + tile_len).min(count);
                let nb = |l: usize| {
                    let j = lo + l;
                    columns::window_of(0, setup.r[j], setup.bias[j], setup.variance[j])
                };
                pass_chunk(
                    &mut qt[lo..hi],
                    ChunkCols {
                        enter: &ticks_off[lo + m - 1..hi + m - 1],
                        leave: &ticks_off[lo - 1..hi - 1],
                        sum: &sum_off[lo..hi],
                        r: &setup.r[lo..hi],
                        bias: &setup.bias[lo..hi],
                        first: lo,
                        neighbor: &nb,
                    },
                    scalars,
                    if turn { &mut other } else { &mut acc },
                    &mut tile,
                );
                turn = !turn;
                lo = hi;
            }
            acc.merge(&mut other);
            acc.finish(&mut nearest);
        } else {
            super::pass_row_narrow(
                &mut qt,
                &cols,
                &inputs,
                sum_off[row],
                m,
                exclusion,
                &mut acc,
                &mut tile,
                &mut nearest,
            );
        }
        out.push(nearest.finish_k());
    }
    Some(out)
}

fn all_bits(rows: &[Vec<Entry>]) -> Vec<Vec<(u64, u64, u64)>> {
    rows.iter()
        .map(|row| row.iter().map(|&e| bits(e)).collect())
        .collect()
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    /// Cutting a row into chunks of any length changes no bit, and k = 1 equals the wide
    /// `advance_and_rank` reference entry.
    #[test]
    fn chunk_size_never_changes_a_bit((series, m) in market_series_and_m(0..60)) {
        let Some(base) = narrow_rows(&series, m, 1_000, 1, false) else { return Ok(()); };
        for tile_len in [1usize, 7, 64, super::DEFAULT_TILE_LANES] {
            let rows = narrow_rows(&series, m, tile_len, 1, false).unwrap();
            prop_assert_eq!(all_bits(&rows), all_bits(&base));
        }
        for tile_len in [1usize, 7] {
            let rows = narrow_rows(&series, m, tile_len, 1, true).unwrap();
            prop_assert_eq!(all_bits(&rows), all_bits(&base));
        }
        let scale = Scale::new(MARKET_TICK).unwrap();
        let (ticks, windows) = ticks_and_windows_with_gaps(&series, m, scale);
        let setup = setup(ticks, windows);
        for (row, entries) in base.iter().enumerate() {
            let (expected, _) = reference_entry(&setup, m, crate::exclusion::zone(m), row);
            let got = entries.first().copied().unwrap_or(Entry::NONE);
            prop_assert_eq!(bits(got), bits(expected), "row {}", row);
        }
    }

    /// k = 2 and 3 equal the naive exact k nearest, at every chunk length.
    #[test]
    fn top_k_equals_naive((series, m) in market_series_and_m(0..60)) {
        let scale = Scale::new(MARKET_TICK).unwrap();
        let samples: Vec<Option<i64>> = sanitize_to_ticks(&series, scale)
            .into_iter()
            .map(|(tick, had)| had.then_some(tick))
            .collect();
        for k in [2usize, 3] {
            let expected = crate::naive::naive_left_topk(&samples, m, NonZeroUsize::new(k).unwrap());
            for tile_len in [1usize, 7, 64] {
                let Some(rows) = narrow_rows(&series, m, tile_len, k, tile_len % 2 == 1) else { return Ok(()); };
                prop_assert_eq!(rows.len(), expected.len());
                for (got, (_, want)) in rows.iter().zip(&expected) {
                    prop_assert_eq!(
                        got.iter().map(|&e| bits(e)).collect::<Vec<_>>(),
                        want.iter().map(|&e| bits(e)).collect::<Vec<_>>()
                    );
                }
            }
        }
    }
}

#[test]
fn a_market_series_fits_one_anchor_and_finds_neighbors() {
    let series: Vec<f64> = (0..40)
        .map(|i| 100.0 + f64::from((i * 7) % 11) * MARKET_TICK)
        .collect();
    let rows = narrow_rows(&series, 4, 7, 2, true).expect("fits one anchor");
    assert!(rows.iter().skip(8).all(|r| r.len() == 2));
}

/// The candidate band is inclusive: a lane sitting at exactly `band_floor` of the
/// running maximum is kept, one a single ulp below is not.
#[test]
fn a_lane_at_the_floor_exactly_is_a_candidate() {
    use super::{ChunkCols, RowAcc};
    let window = windows_of(&[1, 2, 9, 3, 4], 4)[0];
    let top = 0.9;
    let floor = band_floor(top, F64::DELTA);
    let below = f64::from_bits(floor.to_bits() - 1);
    let neighbor = |_: usize| window;
    let kept = |tile: [f64; 3]| {
        let mut acc = RowAcc::new();
        acc.begin(&window, NonZeroUsize::MIN, 1);
        let cols = ChunkCols {
            enter: &[0; 3],
            leave: &[0; 3],
            sum: &[0; 3],
            r: &[0.0; 3],
            bias: &[0.0; 3],
            first: 0,
            neighbor: &neighbor,
        };
        absorb(&mut acc, &tile, &cols);
        acc.cands.len()
    };
    assert_eq!(kept([floor, top, 0.0]), 2);
    assert_eq!(kept([below, top, 0.0]), 1);
}

/// The exact one-nearest pass offers a lane at exactly `band_floor` of the row's maximum to the
/// exact order, and not one a single ulp below. Both neighbors are constant, so they tie
/// exactly and the smaller index wins when it is offered; the maximum is zero, whose floor
/// `-2δ` is exact.
#[test]
fn the_exact_row_offers_a_lane_at_the_floor_exactly() {
    let ticks = [5, 5, 5, 5, 5, 6, 9];
    let m = 4;
    let floor = band_floor(0.0, F64::DELTA);
    let below = f64::from_bits(floor.to_bits() + 1);
    assert!(below < floor);
    let winner = |first_lane: f64| {
        let mut setup = setup(ticks.to_vec(), windows_of(&ticks, m));
        setup.bias[0] = first_lane;
        setup.bias[1] = 0.0;
        let row = RowInputs {
            row: 3,
            window: setup.windows[3],
        };
        let mut qt = [0i64; 2];
        let mut scratch = [0.0; 2];
        let exact =
            super::advance_and_rank_exact(&mut qt, &columns(&setup), &row, m, 1, &mut scratch)
                .expect("a valid row has a nearest neighbor");
        assert_eq!(scratch[1], 0.0);
        exact.entry().index().get()
    };
    assert_eq!(winner(floor), Some(0));
    assert_eq!(winner(below), Some(1));
}

/// A block whose maximum sits exactly on the floor is rescanned, and its lane at the floor
/// kept, wherever in the block's eight-lane parts it lies.
#[test]
fn a_block_whose_maximum_is_the_floor_is_read() {
    use super::{ChunkCols, RowAcc};
    use crate::kernel_neon::GROUP_LANES;
    let window = windows_of(&[1, 2, 9, 3, 4], 4)[0];
    let top = 0.9;
    let floor = band_floor(top, F64::DELTA);
    let neighbor = |_: usize| window;
    let len = 2 * GROUP_LANES;
    let (ints, zeros) = (std::vec![0i32; len], std::vec![0.0; len]);
    for at in [GROUP_LANES, GROUP_LANES + 9, len - 1] {
        let mut tile = std::vec![0.0; len];
        tile[0] = top;
        tile[at] = floor;
        let mut acc = RowAcc::new();
        acc.begin(&window, NonZeroUsize::MIN, 1);
        let cols = ChunkCols {
            enter: &ints,
            leave: &ints,
            sum: &ints,
            r: &zeros,
            bias: &zeros,
            first: 0,
            neighbor: &neighbor,
        };
        absorb(&mut acc, &tile, &cols);
        let kept: Vec<usize> = acc.cands.iter().map(|c| c.index).collect();
        assert_eq!(kept, [0, at]);
    }
}

/// The prune that runs once candidates pile up keeps a lane exactly at the running floor.
#[test]
fn the_prune_keeps_lanes_exactly_at_the_floor() {
    use super::acc::PRUNE_AT;
    use super::{ChunkCols, RowAcc};
    let window = windows_of(&[1, 2, 9, 3, 4], 4)[0];
    let top = 0.9;
    let floor = band_floor(top, F64::DELTA);
    let len = PRUNE_AT + 6;
    let mut tile = std::vec![floor; len];
    tile[0] = top;
    let neighbor = |lane: usize| scaled(window, lane + 1);
    let zeros = std::vec![0.0; len];
    let ints = std::vec![0i32; len];
    let mut acc = RowAcc::new();
    acc.begin(&window, NonZeroUsize::MIN, 1);
    let cols = ChunkCols {
        enter: &ints,
        leave: &ints,
        sum: &ints,
        r: &zeros,
        bias: &zeros,
        first: 0,
        neighbor: &neighbor,
    };
    absorb_with_n::<super::RankRow>(&mut acc, &tile, &cols, n_of_scaled);
    assert_eq!(acc.cands.len(), len);
}

/// Lanes at the floor that tie the lane before them exactly, from larger indices, are not
/// admitted: the smaller index wins the tie.
#[test]
fn a_lane_tying_the_one_before_it_is_not_admitted() {
    use super::acc::PRUNE_AT;
    use super::{ChunkCols, RowAcc};
    let window = windows_of(&[1, 2, 9, 3, 4], 4)[0];
    let top = 0.9;
    let floor = band_floor(top, F64::DELTA);
    let len = PRUNE_AT + 6;
    let mut tile = std::vec![floor; len];
    tile[0] = top;
    let neighbor = |_: usize| window;
    let zeros = std::vec![0.0; len];
    let ints = std::vec![0i32; len];
    let mut acc = RowAcc::new();
    acc.begin(&window, NonZeroUsize::MIN, 1);
    let cols = ChunkCols {
        enter: &ints,
        leave: &ints,
        sum: &ints,
        r: &zeros,
        bias: &zeros,
        first: 0,
        neighbor: &neighbor,
    };
    absorb(&mut acc, &tile, &cols);
    let kept: Vec<usize> = acc.cands.iter().map(|c| c.index).collect();
    assert_eq!(kept, [0, 1]);
}

/// A lane whose correlation is exactly one outranks every later lane, whatever its cross
/// term and variance, once `k` such lanes of one cross term and variance are held; the scan
/// stops there.
#[test]
fn a_run_of_correlation_one_ends_the_scan_of_larger_indices() {
    use super::{ChunkCols, RowAcc};
    let window = windows_of(&[1, 2, 9, 3, 4], 4)[0];
    let WindowKind::Valid {
        variance_numerator, ..
    } = window.kind
    else {
        panic!("a valid window");
    };
    let len = 40;
    let (zeros, ints) = (std::vec![0.0; len], std::vec![0i32; len]);
    let tile = std::vec![1.0; len];
    let held_after = |k: usize, scale: &dyn Fn(usize) -> usize| {
        let neighbor = |lane: usize| scaled(window, scale(lane));
        let cols = ChunkCols {
            enter: &ints,
            leave: &ints,
            sum: &ints,
            r: &zeros,
            bias: &zeros,
            first: 0,
            neighbor: &neighbor,
        };
        let mut acc = RowAcc::new();
        acc.begin(&window, NonZeroUsize::new(k).unwrap(), usize::MAX);
        // A neighbor `s` times larger has a cross term of `s` times the row's variance.
        absorb_with_n::<super::RankRow>(&mut acc, &tile, &cols, |lane| {
            (variance_numerator * scale(lane) as u64) as i64
        });
        acc.cands.len()
    };
    for k in [1, 2, 3] {
        assert_eq!(held_after(k, &|_| 1), k, "one key, k {k}");
    }
    assert_eq!(
        held_after(2, &|lane| lane + 1),
        len,
        "every lane its own key"
    );
}

/// A row whose lanes all tie exactly keeps a bounded candidate list, and what stays is the
/// smallest index.
#[test]
fn a_row_s_exact_ties_collapse_to_the_smallest_index() {
    use super::acc::PRUNE_AT;
    use super::{ChunkCols, RowAcc};
    let window = windows_of(&[1, 2, 9, 3, 4], 4)[0];
    let len = 100;
    let (zeros, ints) = (std::vec![0.0; len], std::vec![0i32; len]);
    let tile = std::vec![0.5; len];
    let mut acc = RowAcc::new();
    acc.begin(&window, NonZeroUsize::MIN, 1);
    for chunk in 0..20 {
        let first = chunk * len;
        let neighbor = |lane: usize| scaled(window, first + lane + 1);
        let cols = ChunkCols {
            enter: &ints,
            leave: &ints,
            sum: &ints,
            r: &zeros,
            bias: &zeros,
            first,
            neighbor: &neighbor,
        };
        absorb_with_n::<super::RankRow>(&mut acc, &tile, &cols, |lane| n_of_scaled(first + lane));
        assert!(acc.cands.len() <= 4 * PRUNE_AT + len);
    }
    acc.resolve_ties();
    let kept: std::vec::Vec<usize> = acc.cands.iter().map(|c| c.index).collect();
    assert!(kept.contains(&0));
}

/// With `k` past the one-by-one limit, exact ties still collapse to the `k` smallest indices,
/// on either side of the limit, and the list stays within twice `k` plus one chunk.
#[test]
fn a_large_k_keeps_the_smallest_indices_of_exact_ties() {
    use super::{ChunkCols, RowAcc};
    let window = windows_of(&[1, 2, 9, 3, 4], 4)[0];
    let len = 100;
    let (zeros, ints) = (std::vec![0.0; len], std::vec![0i32; len]);
    let tile = std::vec![0.5; len];
    for k in [crate::nearest::SELECT_ALL_ABOVE, 150, 700] {
        let mut acc = RowAcc::new();
        acc.begin(&window, NonZeroUsize::new(k).unwrap(), usize::MAX);
        for chunk in 0..20 {
            let first = chunk * len;
            let neighbor = |lane: usize| scaled(window, first + lane + 1);
            let cols = ChunkCols {
                enter: &ints,
                leave: &ints,
                sum: &ints,
                r: &zeros,
                bias: &zeros,
                first,
                neighbor: &neighbor,
            };
            absorb_with_n::<super::RankRow>(&mut acc, &tile, &cols, |lane| {
                n_of_scaled(first + lane)
            });
            assert!(
                acc.cands.len() <= 2 * k.max(super::acc::RESOLVE_AT) + len,
                "k {k}"
            );
        }
        let mut nearest = Nearest::with_k(&window, 4, NonZeroUsize::new(k).unwrap(), 2000);
        acc.finish(&mut nearest);
        let kept: Vec<u64> = nearest
            .finish_k()
            .iter()
            .map(|e| e.index().get().unwrap())
            .collect();
        assert_eq!(kept, (0..k as u64).collect::<Vec<_>>(), "k {k}");
    }
}

/// Folds a ranked `tile` into `acc` as the chunk `cols` describes, every exact `n` zero.
fn absorb(acc: &mut super::RowAcc, tile: &[f64], cols: &super::ChunkCols) {
    absorb_as::<super::RankRow>(acc, tile, cols);
}

/// [`absorb`] as producer `P` ranked the tile.
fn absorb_as<P: Producer>(acc: &mut super::RowAcc, tile: &[f64], cols: &super::ChunkCols) {
    absorb_with_n::<P>(acc, tile, cols, |_| 0);
}

/// [`absorb_as`] with lane `lane`'s exact `n` as `n_of` gives it.
fn absorb_with_n<P: Producer>(
    acc: &mut super::RowAcc,
    tile: &[f64],
    cols: &super::ChunkCols,
    n_of: impl Fn(usize) -> i64,
) {
    let mut maxima = std::vec![0.0; tile.len().div_ceil(crate::kernel_neon::GROUP_LANES)];
    super::chunk::group_maxima(tile, &mut maxima);
    acc.absorb_with(
        crate::kbest::Blocks::<P>::new(tile, &maxima),
        cols.first,
        cols.neighbor,
        n_of,
        |lane| crate::columns::is_constant(cols.r[lane], cols.bias[lane]),
    );
}

/// The neighbors of ties that share no key: neighbor `s` has `s²` times the variance of
/// `base` and, with `n_of_scaled`, `s` times its cross term, so every pair's correlation is
/// the same exactly while no two pairs agree on a cross term and a variance.
fn scaled(base: Window, s: usize) -> Window {
    let WindowKind::Valid {
        variance_numerator, ..
    } = base.kind
    else {
        panic!("a valid base window");
    };
    let variance = variance_numerator * (s * s) as u64;
    Window {
        sum: 0,
        kind: WindowKind::Valid {
            variance_numerator: variance,
            inv_sqrt: 1.0 / libm::sqrt(variance as f64),
        },
    }
}

/// The cross term of the neighbor [`scaled`] makes of absolute index `index`.
fn n_of_scaled(index: usize) -> i64 {
    (index + 1) as i64
}

/// An exact producer: its estimates are the correlations themselves.
struct Exact;

impl Producer for Exact {
    const DELTA: f64 = 0.0;
}

/// The band follows the producer that ranked the lanes: a lane just under the best is a
/// candidate under the kernel's bound and not under an exact producer's, and a row fed by
/// both keeps the wider band.
#[test]
fn the_band_is_the_producer_s() {
    use super::{ChunkCols, RowAcc};
    let window = windows_of(&[1, 2, 9, 3, 4], 4)[0];
    let neighbor = |_: usize| window;
    let cols = ChunkCols {
        enter: &[0; 2],
        leave: &[0; 2],
        sum: &[0; 2],
        r: &[0.0; 2],
        bias: &[0.0; 2],
        first: 0,
        neighbor: &neighbor,
    };
    let tile = [0.9, (0.9f64).next_down()];
    let fresh = || {
        let mut acc = RowAcc::new();
        acc.begin(&window, NonZeroUsize::MIN, 1);
        acc
    };
    let mut acc = fresh();
    absorb_as::<Exact>(&mut acc, &tile, &cols);
    assert_eq!(acc.cands.len(), 1);
    let mut acc = fresh();
    absorb_as::<super::RankRow>(&mut acc, &tile, &cols);
    assert_eq!(acc.cands.len(), 2);
    let mut acc = fresh();
    absorb_as::<Exact>(&mut acc, &[0.1], &cols);
    absorb_as::<super::WideRank>(&mut acc, &[0.1], &cols);
    absorb_as::<Exact>(&mut acc, &tile, &cols);
    assert_eq!(acc.floor(), band_floor(0.9, F64::DELTA));
}

proptest! {
    #![proptest_config(testkit::generators::seeded(200))]

    /// Skipping blocks that cannot enter the `k` largest or reach the floor leaves the same
    /// `k` largest and the same candidates, in the same order, as reading every lane.
    #[test]
    fn skipping_blocks_changes_nothing(
        tiles in prop::collection::vec(
            prop::collection::vec(prop_oneof![
                4 => (0u32..64).prop_map(|v| f64::from(v) / 64.0),
                1 => Just(f64::NEG_INFINITY),
            ], 0..300),
            1..6,
        ),
        k in 1usize..20,
    ) {
        use super::{ChunkCols, RowAcc};
        let window = windows_of(&[1, 2, 9, 3, 4], 4)[0];
        let neighbor = |_: usize| window;
        let k = NonZeroUsize::new(k).unwrap();
        let mut acc = RowAcc::new();
        acc.begin(&window, k, usize::MAX);
        let mut all: std::vec::Vec<(f64, usize)> = std::vec::Vec::new();
        let mut first = 0;
        for tile in &tiles {
            let ints = std::vec![0i32; tile.len()];
            let zeros = std::vec![0.0; tile.len()];
            let cols = ChunkCols {
                enter: &ints,
                leave: &ints,
                sum: &ints,
                r: &zeros,
                bias: &zeros,
                first,
                neighbor: &neighbor,
            };
            absorb(&mut acc, tile, &cols);
            all.extend(tile.iter().copied().zip(first..));
            first += tile.len();
            // Reading every lane: the k largest finite values, then every lane at or above
            // the floor of the k-th that the row has not yet dropped.
            let mut finite: std::vec::Vec<f64> =
                all.iter().map(|a| a.0).filter(|v| v.is_finite()).collect();
            finite.sort_by(|a, b| b.total_cmp(a));
            finite.truncate(k.get());
            let mut top = acc.top.clone();
            top.sort_by(|a, b| b.total_cmp(a));
            prop_assert_eq!(&top, &finite);
        }
        let floor = acc.floor();
        let kept: std::vec::Vec<usize> =
            acc.cands.iter().filter(|c| c.rho_hat >= floor).map(|c| c.index).collect();
        let expected: std::vec::Vec<usize> = all
            .iter()
            .filter(|&&(v, _)| v.is_finite() && v >= floor)
            .map(|&(_, index)| index)
            .collect();
        prop_assert!(kept.iter().all(|i| expected.contains(i)));
        if all.len() <= super::acc::PRUNE_AT {
            prop_assert_eq!(kept, expected);
        }
    }

    /// The wide pass reports each block's maximum as the largest of its lanes.
    #[test]
    fn wide_block_maxima_are_their_lanes_largest(
        values in prop::collection::vec(-1000i64..1000, 1..400),
        seed in any::<i64>(),
    ) {
        use super::chunk_wide::rank_wide_groups;
        use super::{WideChunkCols, WideRow};
        use crate::kernel_neon::GROUP_LANES;
        let len = values.len();
        let neighbor = |_: usize| windows_of(&[1, 2, 9, 3, 4], 4)[0];
        let zeros = std::vec![0i64; len];
        let r = std::vec![1e-3; len];
        let bias = std::vec![0.0; len];
        let cols = WideChunkCols {
            enter: &zeros,
            leave: &zeros,
            sum: &zeros,
            r: &r,
            bias: &bias,
            first: 0,
            neighbor: &neighbor,
        };
        let row = WideRow { enter: 0, leave: 0, sum: 0 };
        let mut qt = values.clone();
        let mut tile = std::vec![0.0; len];
        let groups = len.div_ceil(GROUP_LANES);
        let mut maxima = std::vec![f64::NAN; groups];
        rank_wide_groups((&mut qt, &mut tile, &mut maxima, 0..len), &cols, row, |q, j| {
            (q.wrapping_add(seed) % 997) as f64 * r[j]
        });
        for (g, &max) in maxima.iter().enumerate() {
            let lanes = &tile[g * GROUP_LANES..((g + 1) * GROUP_LANES).min(len)];
            prop_assert_eq!(max, lanes.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
        }
    }
}

/// A part started from the shared floors admits a pair at them and none below, and its own
/// top only ever raises them.
#[test]
fn a_seeded_column_admits_only_what_the_shared_floor_allows() {
    let mut shared = column_acc(1);
    for row in 4..8 {
        shared.offer_lane::<super::RankRow>(0, id(row), ranked(0.9), || 0);
    }
    let floor = band_floor(0.9, F64::DELTA);
    let mut part = column_acc(1);
    part.raise_floors(&shared, 0..1);
    part.offer_lane::<super::RankRow>(0, id(5), ranked(floor.next_down()), || 0);
    part.offer_lane::<super::RankRow>(0, id(6), ranked(floor), || 0);
    part.offer_lane::<super::RankRow>(0, id(7), ranked(0.1), || 0);
    assert_eq!(part.cands.len(), 1);
    assert_eq!(part.floors(0, 1), &[floor]);
}

fn constant_window() -> Window {
    Window {
        sum: 0,
        kind: WindowKind::Constant,
    }
}

fn column_acc(columns: usize) -> super::ColumnAcc {
    let mut acc = super::ColumnAcc::new();
    acc.begin(
        0..columns,
        &std::vec![constant_window(); columns],
        NonZeroUsize::MIN,
        |_| usize::MAX,
    );
    acc
}

/// Pairs all tied exactly on `ρ̂` pile up without bound by the floor alone; `compact` keeps
/// each column's exact nearest, the smallest row.
#[test]
fn exact_ties_collapse_to_the_smallest_row() {
    let windows = std::vec![constant_window(); 5000];
    let mut acc = super::ColumnAcc::new();
    acc.begin(0..2, &windows, NonZeroUsize::MIN, |_| usize::MAX);
    for row in (2..5000).rev() {
        for column in 0..2 {
            acc.offer_lane::<super::RankRow>(
                column,
                id(row),
                super::RankedLane {
                    rho_hat: 1.0,
                    n: row as i64,
                },
                || 0,
            );
        }
        acc.compact(&windows);
        assert!(acc.cands.len() <= 8 * super::column::COLUMN_PRUNE_AT + 2);
    }
    acc.compact(&windows);
    acc.keep_exact_nearest(&windows);
    let rows: std::vec::Vec<usize> = acc.cands.iter().map(|c| c.row).collect();
    assert_eq!(rows, std::vec![2, 2]);
}

/// With `k` of two the floor is the second best's: a pair between the two floors stays, and
/// exact ties collapse to the two smallest rows.
#[test]
fn a_column_keeps_the_pairs_near_its_k_th_best() {
    let two = NonZeroUsize::new(2).unwrap();
    let windows = std::vec![constant_window(); 5000];
    let mut acc = super::ColumnAcc::new();
    acc.begin(0..1, &windows, two, |_| usize::MAX);
    acc.offer_lane::<super::RankRow>(0, id(5), ranked(0.9), || 0);
    acc.offer_lane::<super::RankRow>(0, id(6), ranked(0.5), || 0);
    acc.offer_lane::<super::RankRow>(0, id(7), ranked(0.1), || 0);
    let rows: std::vec::Vec<usize> = acc.cands.iter().map(|c| c.row).collect();
    assert_eq!(rows, std::vec![5, 6]);

    acc.begin(0..2, &windows, two, |_| usize::MAX);
    for row in (2..5000).rev() {
        for column in 0..2 {
            acc.offer_lane::<super::RankRow>(
                column,
                id(row),
                super::RankedLane {
                    rho_hat: 1.0,
                    n: row as i64,
                },
                || 0,
            );
        }
        acc.compact(&windows);
        assert!(acc.cands.len() <= 8 * 2 * super::column::COLUMN_PRUNE_AT + 4);
    }
    acc.compact(&windows);
    acc.keep_exact_nearest(&windows);
    acc.sort();
    let rows: std::vec::Vec<usize> = acc.cands.iter().map(|c| c.row).collect();
    assert_eq!(rows.len(), 4);
    assert_eq!(rows.iter().filter(|&&r| r == 2).count(), 2);
    assert_eq!(rows.iter().filter(|&&r| r == 3).count(), 2);
}

/// A column's list holds its `k` largest, descending, however they arrive and merge, and its
/// floor follows the `k`-th.
#[test]
fn a_column_keeps_its_k_largest_in_order_through_inserts_and_merges() {
    let three = NonZeroUsize::new(3).unwrap();
    let windows = std::vec![constant_window(); 4];
    let floor = |best| band_floor(best, F64::DELTA);
    let mut mine = super::ColumnAcc::new();
    mine.begin(0..1, &windows, three, |_| usize::MAX);
    for (row, rho) in [(1, 0.2), (2, 0.9), (3, 0.5), (4, 0.7), (5, 0.1)] {
        mine.offer_lane::<super::RankRow>(0, id(row), ranked(rho), || 0);
    }
    assert_eq!(mine.top_of(0), std::vec![0.9, 0.7, 0.5]);
    assert_eq!(mine.floor, std::vec![floor(0.5)]);

    let mut theirs = super::ColumnAcc::new();
    theirs.begin(0..1, &windows, three, |_| usize::MAX);
    for (row, rho) in [(6, 0.85), (7, 0.8), (8, 0.3)] {
        theirs.offer_lane::<super::RankRow>(0, id(row), ranked(rho), || 0);
    }
    theirs.sort();
    mine.merge(&theirs, 0..1, &windows);
    assert_eq!(mine.top_of(0), std::vec![0.9, 0.85, 0.8]);
    assert_eq!(mine.floor, std::vec![floor(0.8)]);
}

/// `finish` keeps a pair at the final floor: with every window constant the pairs tie
/// exactly, so the pair at the floor, from the smaller row, wins if it was kept.
#[test]
fn finish_keeps_a_pair_exactly_at_the_final_floor() {
    let windows = std::vec![constant_window(); 8];
    let mut acc = super::ColumnAcc::new();
    acc.begin(0..1, &windows, NonZeroUsize::MIN, |_| usize::MAX);
    acc.offer_lane::<super::RankRow>(0, id(5), ranked(0.9), || 0);
    acc.offer_lane::<super::RankRow>(0, id(2), ranked(band_floor(0.9, F64::DELTA)), || 0);
    let mut out = std::vec::Vec::new();
    acc.finish(&windows, 3, &mut out);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].1.index().get(), Some(2));
}

/// A column owns `min(k, rows that can reach it)` slots: none when nothing can.
#[test]
fn a_column_owns_only_the_slots_its_rows_can_fill() {
    let three = NonZeroUsize::new(3).unwrap();
    let windows = std::vec![constant_window(); 3];
    let mut acc = super::ColumnAcc::new();
    acc.begin(0..3, &windows, three, |column| column);
    assert_eq!(acc.offset, std::vec![0, 0, 1, 3]);
    acc.begin(0..3, &windows, three, |_| 0);
    assert!(acc.top.is_empty());
}

fn ranked(rho_hat: f64) -> super::RankedLane {
    super::RankedLane { rho_hat, n: 0 }
}

/// A row as the columns know it: its index, and a constant window's variance.
fn id(row: usize) -> super::RowId {
    super::RowId {
        index: row,
        variance: 0,
        offers_from: 0,
        finite_before: 0,
    }
}

#[test]
fn a_column_keeps_a_pair_at_its_floor_and_drops_one_below() {
    let floor = band_floor(0.9, F64::DELTA);
    let mut acc = column_acc(1);
    acc.offer_lane::<super::RankRow>(0, id(5), ranked(0.9), || 0);
    acc.offer_lane::<super::RankRow>(0, id(6), ranked(floor), || 0);
    assert_eq!(acc.cands.len(), 2);
    acc.offer_lane::<super::RankRow>(0, id(7), ranked(floor.next_down()), || 0);
    assert_eq!(acc.cands.len(), 2);
}

#[test]
fn pairs_far_below_a_column_s_best_are_pruned_as_they_pile_up() {
    let windows = std::vec![constant_window(); 200];
    let mut acc = super::ColumnAcc::new();
    acc.begin(0..1, &windows, NonZeroUsize::MIN, |_| usize::MAX);
    for row in 0..200 {
        acc.offer_lane::<super::RankRow>(0, id(row), ranked(row as f64 * 1e-3), || 0);
        acc.compact(&windows);
    }
    assert!(acc.cands.len() <= super::column::COLUMN_PRUNE_AT);
    assert_eq!(acc.cands.last().map(|c| c.row), Some(199));
}

#[test]
fn merging_a_better_partial_raises_the_floor_for_later_ones() {
    let mut mine = column_acc(2);
    mine.offer_lane::<super::RankRow>(0, id(5), ranked(0.1), || 0);
    let mut better = column_acc(2);
    better.offer_lane::<super::RankRow>(0, id(9), ranked(0.9), || 0);
    better.sort();
    mine.merge(&better, 0..2, &[]);
    let mut worse = column_acc(2);
    worse.offer_lane::<super::RankRow>(0, id(7), ranked(0.5), || 0);
    worse.sort();
    mine.merge(&worse, 0..2, &[]);
    let rows: std::vec::Vec<usize> = mine.cands.iter().map(|c| c.row).collect();
    assert_eq!(rows, std::vec![5, 9]);
}

proptest! {
    #![proptest_config(testkit::generators::seeded(100))]

    /// Grouping the candidates by column keeps every candidate and puts each column's in one
    /// run, whether few candidates lie over many columns or many over few.
    #[test]
    fn sorting_the_candidates_groups_them_by_column(
        columns in 1usize..40,
        picks in prop::collection::vec((0usize..1000, any::<i64>()), 0..120),
    ) {
        let windows = std::vec![constant_window(); columns];
        let mut acc = super::ColumnAcc::new();
        acc.begin(0..columns, &windows, NonZeroUsize::MIN, |_| usize::MAX);
        for (row, &(pick, n)) in picks.iter().enumerate() {
            acc.cands.push(super::column::ColumnCand {
                column: pick % columns,
                row,
                n,
                rho_hat: 0.5,
            });
        }
        let mut expected: Vec<(usize, usize, i64)> =
            acc.cands.iter().map(|c| (c.column, c.row, c.n)).collect();
        acc.sort();
        let mut got: Vec<(usize, usize, i64)> =
            acc.cands.iter().map(|c| (c.column, c.row, c.n)).collect();
        prop_assert!(got.windows(2).all(|w| w[0].0 <= w[1].0));
        got.sort_unstable();
        expected.sort_unstable();
        prop_assert_eq!(got, expected);
    }
}

/// A valid window of this variance numerator.
fn valid_window(variance: u64) -> Window {
    Window {
        sum: 0,
        kind: WindowKind::Valid {
            variance_numerator: variance,
            inv_sqrt: 1.0 / libm::sqrt(variance as f64),
        },
    }
}

/// Column 0 of variance 4 with rows 1.. of the variances given, and an accumulator over it.
fn column_of_variance_four(
    row_variances: &[u64],
    k: usize,
) -> (std::vec::Vec<Window>, super::ColumnAcc) {
    let mut windows = std::vec![valid_window(4)];
    windows.extend(row_variances.iter().map(|&v| valid_window(v)));
    let mut acc = super::ColumnAcc::new();
    acc.begin(0..1, &windows, NonZeroUsize::new(k).unwrap(), |_| {
        usize::MAX
    });
    (windows, acc)
}

/// Offers row `row`, of variance `variance`, to column 0 (of variance 4) at cross term `n`.
fn offer_valid(acc: &mut super::ColumnAcc, row: usize, variance: u64, n: i64, rho_hat: f64) {
    let id = super::RowId {
        index: row,
        variance,
        offers_from: 0,
        finite_before: 0,
    };
    acc.offer_lane::<super::RankRow>(0, id, super::RankedLane { rho_hat, n }, || 4);
}

fn admitted_rows(acc: &super::ColumnAcc) -> std::vec::Vec<usize> {
    acc.cands.iter().map(|c| c.row).collect()
}

/// A row that bettered the column's best starts no run; later rows that tie the last row
/// admitted without bettering it exactly (the same cross term and variance, the same
/// estimate) lose the tie to it and are not admitted; a different cross term or variance is
/// another pair, even at the same correlation.
#[test]
fn a_column_does_not_admit_a_row_that_ties_the_run_before_it() {
    let (_, mut acc) = column_of_variance_four(&[9, 9, 9, 36, 36], 1);
    offer_valid(&mut acc, 1, 9, 3, 0.5);
    offer_valid(&mut acc, 2, 9, 3, 0.5);
    offer_valid(&mut acc, 3, 9, 3, 0.5);
    assert_eq!(admitted_rows(&acc), [1, 2]);
    // Twice the cross term at four times the variance: the same correlation, another key.
    offer_valid(&mut acc, 4, 36, 6, 0.5);
    offer_valid(&mut acc, 5, 36, 6, 0.5);
    assert_eq!(admitted_rows(&acc), [1, 2, 4]);
}

/// With `k` of two a pair is skipped only after two before it in a run tie it.
#[test]
fn a_column_admits_k_rows_of_a_tie_before_it_skips() {
    let (_, mut acc) = column_of_variance_four(&[9; 6], 2);
    for row in 1..=6 {
        offer_valid(&mut acc, row, 9, 3, 0.5);
    }
    // Two fill the column's best, two more make the run.
    assert_eq!(admitted_rows(&acc), [1, 2, 3, 4]);
}

/// Two rows of correlation exactly one (its cross term squared is the product of the
/// variances) in a run beat every later row, whatever its cross term and variance: none is
/// admitted after them.
#[test]
fn a_column_holding_a_correlation_of_one_admits_no_later_row() {
    let (windows, mut acc) = column_of_variance_four(&[9, 16, 25, 36], 1);
    offer_valid(&mut acc, 1, 9, 6, 1.0);
    offer_valid(&mut acc, 2, 16, 8, 1.0);
    offer_valid(&mut acc, 3, 25, 10, 1.0);
    assert_eq!(admitted_rows(&acc), [1, 2]);
    // The kernel reads nothing more of the column, except in a checked build, which reads on
    // to assert the order of the rows.
    let floor = band_floor(1.0, F64::DELTA);
    let read = acc.floors(0, 1)[0];
    if cfg!(debug_assertions) {
        assert_eq!(read, floor);
    } else {
        assert_eq!(read, f64::INFINITY);
    }
    // The column still reports it: the settled column keeps the floor its candidates meet.
    acc.compact(&windows);
    acc.sort();
    let mut out = std::vec::Vec::new();
    acc.finish(&windows, 4, &mut out);
    assert_eq!(out[0].1.index().get(), Some(1));
}

/// A cross term one short of perfect is not a correlation of one: later rows still count.
#[test]
fn a_nearly_perfect_row_settles_nothing() {
    let (_, mut acc) = column_of_variance_four(&[9, 16, 25], 1);
    offer_valid(&mut acc, 1, 9, 5, 1.0);
    offer_valid(&mut acc, 2, 16, 7, 1.0);
    offer_valid(&mut acc, 3, 25, 9, 1.0);
    assert_eq!(admitted_rows(&acc), [1, 2, 3]);
}

/// With `k` of two a correlation of one settles the column once two rows of one key are held
/// in a run.
#[test]
fn a_column_settles_on_k_rows_of_a_correlation_of_one() {
    let (_, mut acc) = column_of_variance_four(&[9, 9, 9, 9, 16], 2);
    for row in 1..=4 {
        offer_valid(&mut acc, row, 9, 6, 1.0);
    }
    offer_valid(&mut acc, 5, 16, 8, 1.0);
    assert_eq!(admitted_rows(&acc), [1, 2, 3, 4]);
}

#[test]
fn a_non_finite_column_takes_no_pair() {
    let finite = Window {
        sum: 0,
        kind: WindowKind::Constant,
    };
    let gap = Window {
        sum: 0,
        kind: WindowKind::NonFinite,
    };
    let mut acc = super::ColumnAcc::new();
    acc.begin(0..2, &[finite, gap], NonZeroUsize::MIN, |_| usize::MAX);
    acc.offer_chunk::<super::RankRow>(0, id(4), &[0.5, 0.5], |_| 0, |_| 0);
    let columns: std::vec::Vec<usize> = acc.cands.iter().map(|c| c.column).collect();
    assert_eq!(columns, std::vec![0]);
}

/// A constant row's neighbors as `kind(index)` lays them out (`2` constant, `1` valid, `0`
/// non-finite), fed in chunks of `len`: the indices it holds as candidates, and its `k` nearest.
#[cfg(not(any(loom, shuttle)))]
fn constant_row_fed(
    kind: &dyn Fn(usize) -> u8,
    k: usize,
    chunks: usize,
    len: usize,
) -> (Vec<usize>, Vec<(u64, u64, u64)>) {
    use super::{ChunkCols, RowAcc};
    let row = constant_window();
    let (zeros, ints) = (std::vec![0.0; len], std::vec![0i32; len]);
    let k = NonZeroUsize::new(k).unwrap();
    let mut acc = RowAcc::new();
    acc.begin(&row, k, usize::MAX);
    for chunk in 0..chunks {
        let first = chunk * len;
        let tile: Vec<f64> = (first..first + len)
            .map(|index| {
                let (r, bias) = crate::columns::r_bias(&neighbor_of_kind(kind(index)));
                crate::columns::constant_row_rank(r, bias)
            })
            .collect();
        let neighbor = |lane: usize| neighbor_of_kind(kind(first + lane));
        let cols = ChunkCols {
            enter: &ints,
            leave: &ints,
            sum: &ints,
            r: &zeros,
            bias: &zeros,
            first,
            neighbor: &neighbor,
        };
        absorb(&mut acc, &tile, &cols);
    }
    let held = acc.cands.iter().map(|c| c.index).collect();
    let mut nearest = Nearest::with_k(&row, 4, k, usize::MAX);
    acc.finish(&mut nearest);
    let kept = nearest.finish_k().into_iter().map(bits).collect();
    (held, kept)
}

/// A neighbor of kind `kind`, as [`constant_row_fed`] reads it.
#[cfg(not(any(loom, shuttle)))]
fn neighbor_of_kind(kind: u8) -> Window {
    match kind {
        2 => constant_window(),
        1 => windows_of(&[1, 2, 9, 3, 4], 4)[0],
        _ => Window {
            sum: 0,
            kind: WindowKind::NonFinite,
        },
    }
}

/// The `k` nearest of a constant row, every neighbor offered to the exact reduction.
#[cfg(not(any(loom, shuttle)))]
fn constant_row_reference(kind: &dyn Fn(usize) -> u8, k: usize, n: usize) -> Vec<(u64, u64, u64)> {
    let row = constant_window();
    let mut nearest = Nearest::with_k(&row, 4, NonZeroUsize::new(k).unwrap(), usize::MAX);
    nearest.select((0..n).map(|index| {
        let index_of = crate::index::SampleIndex::new(index as u64).unwrap();
        (0, neighbor_of_kind(kind(index)), index_of)
    }));
    nearest.finish_k().into_iter().map(bits).collect()
}

/// A constant row holds only the first `k` of its constant neighbors and the first `k` of its
/// valid ones, and none of the valid ones once `k` constant ones came first; its `k` nearest
/// are the exact reduction's, whichever way its lanes are admitted.
#[test]
#[cfg(not(any(loom, shuttle)))]
fn a_constant_row_admits_only_the_first_k_lanes_of_each_kind() {
    let (chunks, len) = (4, 150);
    let n = chunks * len;
    let late_constants = |index: usize| match index {
        i if i % 11 == 0 => 0,
        i if i >= 230 && i % 13 == 0 => 2,
        _ => 1,
    };
    let early_constants = |index: usize| match index {
        i if i % 11 == 0 => 0,
        i if i % 2 == 0 => 2,
        _ => 1,
    };
    let first_of = |kind: &dyn Fn(usize) -> u8, of: u8, k: usize| -> Vec<usize> {
        (0..n).filter(|&i| kind(i) == of).take(k).collect()
    };
    for k in [1, 2, 3, 8, 40] {
        let (held, kept) = constant_row_fed(&late_constants, k, chunks, len);
        let mut expected = first_of(&late_constants, 1, k);
        expected.extend(first_of(&late_constants, 2, k));
        expected.sort_unstable();
        let mut held = held;
        held.sort_unstable();
        assert_eq!(held, expected, "late constants, k {k}");
        assert_eq!(kept, constant_row_reference(&late_constants, k, n), "k {k}");

        let (held, kept) = constant_row_fed(&early_constants, k, chunks, len);
        assert_eq!(
            held,
            first_of(&early_constants, 2, k),
            "early constants, k {k}"
        );
        assert_eq!(
            kept,
            constant_row_reference(&early_constants, k, n),
            "k {k}"
        );

        let by_lane = crate::route::forced_route(crate::route::RouteForce {
            constant_rows_by_lane: true,
            ..Default::default()
        });
        for kind in [&late_constants as &dyn Fn(usize) -> u8, &early_constants] {
            let (_, kept) = constant_row_fed(kind, k, chunks, len);
            assert_eq!(kept, constant_row_reference(kind, k, n), "by lane, k {k}");
        }
        assert!(by_lane.fired().constant_row_spans_by_lane > 0);
    }
}
