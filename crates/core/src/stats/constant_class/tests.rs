extern crate std;

use super::{ConstantClass, JoinClass, NO_GAP, RowClass, RowsClass};
use crate::stats::{Window, WindowKind};
use core::num::NonZeroUsize;
use core::ops::Range;
use proptest::prelude::*;
use std::vec::Vec;

fn window(kind: u8) -> Window {
    let kind = match kind {
        0 => WindowKind::NonFinite,
        1 => WindowKind::Constant,
        _ => WindowKind::Valid {
            variance_numerator: u64::from(kind),
            inv_sqrt: 1.0,
        },
    };
    Window { sum: 0, kind }
}

fn constant(w: &Window) -> bool {
    w.kind == WindowKind::Constant
}

fn finite(w: &Window) -> bool {
    w.kind != WindowKind::NonFinite
}

/// [`ConstantClass::row`] by its definitions, counted window by window.
fn row_by_definition(windows: &[Window], e: usize, row: usize, k: usize) -> RowClass {
    let constants: Vec<usize> = (0..windows.len())
        .filter(|&j| constant(&windows[j]))
        .collect();
    let constants_through = constants.get(k - 1).map_or(u32::MAX, |&c| c as u32);
    if constant(&windows[row]) {
        // The first column with fewer than `k` constant windows between its zone and the row.
        let offers_from = (0..=row)
            .find(|&j| (j + e + 1..row).filter(|&c| constant(&windows[c])).count() < k)
            .expect("the row's own column has none between");
        return RowClass {
            constants_through,
            settled: offers_from as u32,
        };
    }
    let left_constants = (0..row.saturating_sub(e))
        .filter(|&j| constant(&windows[j]))
        .count();
    let finite_before = (0..row).filter(|&j| finite(&windows[j])).count() as u32;
    let seeded = finite(&windows[row]) && left_constants >= k;
    RowClass {
        constants_through,
        settled: finite_before | if seeded { super::SEEDED } else { 0 },
    }
}

fn kinds() -> impl Strategy<Value = Vec<u8>> {
    kinds_up_to(60)
}

fn kinds_up_to(windows: usize) -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(
        prop_oneof![1 => Just(0u8), 3 => Just(1u8), 2 => Just(2u8)],
        1..windows,
    )
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    #[test]
    fn rows_match_their_definitions(kinds in kinds(), e in 0usize..6, k in 1usize..5) {
        let windows: Vec<Window> = kinds.iter().map(|&k| window(k)).collect();
        let Some(class) = ConstantClass::of(&windows, e) else {
            prop_assert!(!windows.iter().any(constant));
            return Ok(());
        };
        let k_nz = NonZeroUsize::new(k).unwrap();
        for row in 0..windows.len() {
            prop_assert_eq!(class.row(row, k_nz), row_by_definition(&windows, e, row, k), "row {}", row);
        }
    }

    #[test]
    fn column_cuts_match_their_definition(kinds in kinds(), e in 0usize..6, k in 1usize..5) {
        let windows: Vec<Window> = kinds.iter().map(|&k| window(k)).collect();
        let Some(class) = ConstantClass::of(&windows, e) else { return Ok(()) };
        let n = windows.len();
        let finite_before = |i: usize| (0..i).filter(|&j| finite(&windows[j])).count();
        for start in 0..n {
            for end in start..=n {
                let mut got = Vec::new();
                class.column_cuts(start..end, NonZeroUsize::new(k).unwrap(), |c, cut| got.push((c, cut)));
                let want: Vec<(usize, usize)> = (start..end)
                    .filter(|&c| constant(&windows[c]))
                    .map(|c| (c, finite_before((c + e + 1).min(n)) + k))
                    .collect();
                prop_assert_eq!(&got, &want);
                // A row at or past the cut has `k` finite windows between the column's zone and
                // itself; one before it has fewer.
                for &(c, cut) in &got {
                    for row in (c + e + 1).min(n)..n {
                        let between = (c + e + 1..row).filter(|&j| finite(&windows[j])).count();
                        prop_assert_eq!(finite_before(row) >= cut, between >= k);
                    }
                }
            }
        }
    }

    #[test]
    fn column_seeds_match_their_definition(
        kinds in kinds(),
        e in 0usize..6,
        k in 1usize..5,
        start in 0usize..60,
    ) {
        let windows: Vec<Window> = kinds.iter().map(|&k| window(k)).collect();
        let Some(class) = ConstantClass::of(&windows, e) else { return Ok(()) };
        let n = windows.len();
        let start = start.min(n);
        let reach = |c: usize| (c + e + 1..n).filter(|&j| finite(&windows[j])).count();
        let slots = |c: usize| if finite(&windows[c]) { k.min(reach(c)) } else { 0 };
        let mut got = Vec::new();
        class.column_seeds(start..n, slots, |c, rho| got.push((c, rho)));
        let want: Vec<(usize, f64)> = (start..n)
            .filter_map(|c| {
                let s = slots(c);
                if s == 0 {
                    return None;
                }
                let past = (c + e + 1..n).filter(|&j| constant(&windows[j])).count();
                match (constant(&windows[c]), past >= s) {
                    (true, true) => Some((c, 1.0)),
                    (true, false) | (false, true) => Some((c, 0.5)),
                    (false, false) => None,
                }
            })
            .collect();
        prop_assert_eq!(got, want);
    }
}

/// The exact correlation of windows `a` and `b` where their kinds fix it: none with a non-finite
/// window, one between constant windows, one half between a constant and an ordinary one; between
/// two ordinary windows, whatever `ordinary` gives the pair.
fn correlation(windows: &[Window], ordinary: &[f64], a: usize, b: usize) -> Option<f64> {
    if !finite(&windows[a]) || !finite(&windows[b]) {
        return None;
    }
    Some(match (constant(&windows[a]), constant(&windows[b])) {
        (true, true) => 1.0,
        (true, false) | (false, true) => 0.5,
        (false, false) => ordinary[(a * 31 + b * 17) % ordinary.len()],
    })
}

/// `nearest[row][column]`: whether the pair is among the row's `k` nearest left neighbors or the
/// column's `k` nearest right ones, the larger correlation first and ties to the smaller index.
fn nearest(windows: &[Window], ordinary: &[f64], e: usize, k: usize) -> Vec<Vec<bool>> {
    let n = windows.len();
    let mut nearest = std::vec![std::vec![false; n]; n];
    let ranked = |own: usize, others: Range<usize>| {
        let mut ranked: Vec<(f64, usize)> = others
            .filter_map(|other| Some((correlation(windows, ordinary, own, other)?, other)))
            .collect();
        ranked.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        ranked.truncate(k);
        ranked
    };
    let pairs = (0..n).flat_map(|own| {
        let left = ranked(own, 0..own.saturating_sub(e)).into_iter();
        let right = ranked(own, (own + e + 1).min(n)..n).into_iter();
        left.map(move |(_, column)| (own, column))
            .chain(right.map(move |(_, row)| (row, own)))
    });
    for (row, column) in pairs {
        nearest[row][column] = true;
    }
    nearest
}

/// How far before `row` its `k`-th earlier window of the kind that settles its offers lies:
/// constant for a constant row, finite for an ordinary one; `None` with fewer than `k`, and `0`
/// for a non-finite row.
fn gap_by_definition(windows: &[Window], row: usize, k: usize) -> Option<usize> {
    if !finite(&windows[row]) {
        return Some(0);
    }
    let settles: fn(&Window) -> bool = if constant(&windows[row]) {
        constant
    } else {
        finite
    };
    let kth = (0..row)
        .rev()
        .filter(|&j| settles(&windows[j]))
        .nth(k - 1)?;
    Some(row - kth)
}

/// The block rule, counted window by window: the three conditions of [`super::ClassBlocks`].
fn settles_by_definition(
    windows: &[Window],
    (e, k): (usize, usize),
    rows: Range<usize>,
    columns: Range<usize>,
    lag: usize,
) -> bool {
    let ordinary = |w: &Window| finite(w) && !constant(w);
    let no_ordinary_pair = !windows[rows.clone()].iter().any(ordinary)
        || !windows[columns.clone()].iter().any(ordinary);
    let gaps: Option<Vec<usize>> = rows.map(|row| gap_by_definition(windows, row, k)).collect();
    let past_gap = gaps.is_some_and(|gaps| gaps.iter().all(|&gap| lag > gap + e));
    let first = |kind: fn(&Window) -> bool| {
        (0..windows.len())
            .filter(|&j| kind(&windows[j]))
            .take(k)
            .any(|j| columns.contains(&j))
    };
    no_ordinary_pair && past_gap && !first(constant) && !first(finite)
}

/// Every block of up to `rows` rows and `columns` columns of `n` windows, the columns wholly
/// before the rows.
fn blocks(
    n: usize,
    rows: usize,
    columns: usize,
) -> impl Iterator<Item = (Range<usize>, Range<usize>)> {
    (0..n).flat_map(move |r0| {
        (r0 + 1..=(r0 + rows).min(n)).flat_map(move |r1| {
            (0..r0).flat_map(move |c0| {
                (c0 + 1..=(c0 + columns).min(r0)).map(move |c1| (r0..r1, c0..c1))
            })
        })
    })
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    /// The block query is the rule: its three conditions, counted window by window, at every
    /// block, lag and `k`.
    #[test]
    fn blocks_match_the_rule(kinds in kinds_up_to(26), e in 0usize..4, k in 1usize..4) {
        let windows: Vec<Window> = kinds.iter().map(|&k| window(k)).collect();
        let Some(class) = ConstantClass::of(&windows, e) else { return Ok(()) };
        let asked = class.blocks(NonZeroUsize::new(k).unwrap()).unwrap();
        for (rows, columns) in blocks(windows.len(), 3, 4) {
            let read = asked.rows(rows.clone(), &columns);
            for lag in 1..=rows.start - columns.start {
                let got = read.is_some_and(|read| asked.settles(&read, &columns, lag));
                let want = settles_by_definition(&windows, (e, k), rows.clone(), columns.clone(), lag);
                prop_assert_eq!(got, want, "rows {:?}, columns {:?}, lag {}", &rows, &columns, lag);
            }
        }
    }

    /// No pair of a block the class settles is among its row's nearest left neighbors or its
    /// column's nearest right ones, whatever the correlations of the ordinary pairs are.
    #[test]
    fn a_settled_block_holds_no_nearest_pair(
        kinds in kinds_up_to(26),
        e in 0usize..4,
        k in 1usize..4,
        ordinary in prop::collection::vec(
            prop_oneof![Just(-1.0), Just(0.25), Just(0.5), Just(0.75), Just(1.0)],
            7,
        ),
    ) {
        let windows: Vec<Window> = kinds.iter().map(|&k| window(k)).collect();
        let Some(class) = ConstantClass::of(&windows, e) else { return Ok(()) };
        let asked = class.blocks(NonZeroUsize::new(k).unwrap()).unwrap();
        let nearest = nearest(&windows, &ordinary, e, k);
        for (rows, columns) in blocks(windows.len(), 3, 4) {
            let Some(read) = asked.rows(rows.clone(), &columns) else { continue };
            for lag in e + 1..=rows.start - columns.start {
                if !asked.settles(&read, &columns, lag) {
                    continue;
                }
                for row in rows.clone() {
                    for column in columns.clone() {
                        prop_assert!(
                            row - column < lag || !nearest[row][column],
                            "pair ({}, {}) of rows {:?}, columns {:?}, lag {}",
                            row, column, &rows, &columns, lag
                        );
                    }
                }
            }
        }
    }
}

/// The blocks the class settles for `rows` against `columns` at `lag`, at exclusion zone `e`
/// keeping `k` neighbors, over windows of `kinds` (`N` non-finite, `C` constant, `V` ordinary).
fn settles(
    kinds: &str,
    (e, k): (usize, usize),
    rows: Range<usize>,
    columns: Range<usize>,
    lag: usize,
) -> bool {
    let windows: Vec<Window> = kinds
        .bytes()
        .map(|kind| {
            window(match kind {
                b'N' => 0,
                b'C' => 1,
                _ => 2,
            })
        })
        .collect();
    let class = ConstantClass::of(&windows, e).unwrap();
    let asked = class.blocks(NonZeroUsize::new(k).unwrap()).unwrap();
    let got = asked
        .rows(rows.clone(), &columns)
        .is_some_and(|read| asked.settles(&read, &columns, lag));
    assert_eq!(
        got,
        settles_by_definition(&windows, (e, k), rows, columns, lag)
    );
    got
}

/// The first condition at its boundary: one ordinary column against rows that hold one ordinary
/// window is one pair the kinds do not fix.
#[test]
fn a_block_pairing_two_ordinary_windows_is_not_settled() {
    // Rows 8..10 are a constant and an ordinary window; both have a settling window one back.
    assert!(settles("CCCCCVCCVC", (0, 1), 8..10, 2..5, 4));
    assert!(!settles("CCCCVVCCVC", (0, 1), 8..10, 2..5, 4));
    assert!(settles("CCCCNVCCVC", (0, 1), 8..10, 2..5, 4));
    // With no ordinary row the columns' kinds do not matter.
    assert!(settles("CCVVVVCCCC", (0, 1), 8..10, 2..5, 4));
}

/// The second condition at its boundary, for a constant row: the lag is one more than the
/// distance back to the constant window before it plus the zone. One less in the lag, one more in
/// the distance, or one more in the zone, and the row's offer may be a column's nearest.
#[test]
fn a_constant_row_is_settled_past_the_constant_window_before_it() {
    // Row 8's earlier constant window is 5: three back.
    assert!(settles("CCVVVCVVC", (1, 1), 8..9, 1..2, 5));
    assert!(!settles("CCVVVCVVC", (1, 1), 8..9, 1..2, 4));
    assert!(settles("CCVVVVCVC", (1, 1), 8..9, 1..2, 4));
    assert!(!settles("CCVVCVVVC", (1, 1), 8..9, 1..2, 5));
    assert!(settles("CCVVVCVVC", (0, 1), 8..9, 1..2, 4));
    assert!(!settles("CCVVVCVVC", (2, 1), 8..9, 1..2, 5));
    // Keeping two, the window that settles the offer is the second constant one back.
    assert!(!settles("CCVCVCVVC", (1, 2), 8..9, 2..3, 6));
    assert!(settles("CCVCVCVVC", (1, 2), 8..9, 2..3, 7));
    // A row with no constant window before it is settled at no lag.
    assert!(!settles("VVVVVVVVC", (1, 1), 8..9, 1..2, 7));
}

/// The second condition for an ordinary row: the window that settles its offers is the finite
/// one before it, constant or not.
#[test]
fn an_ordinary_row_is_settled_past_the_finite_window_before_it() {
    // Row 8's earlier finite window is 5.
    assert!(settles("CCNNNVNNV", (1, 1), 8..9, 1..2, 5));
    assert!(!settles("CCNNNVNNV", (1, 1), 8..9, 1..2, 4));
    assert!(settles("CCNNNNCNV", (1, 1), 8..9, 1..2, 4));
    assert!(!settles("CCNNVNNNV", (1, 1), 8..9, 1..2, 5));
    assert!(!settles("CCNNNVNNV", (2, 1), 8..9, 1..2, 5));
    // Keeping two, it is the second finite window back, 3.
    assert!(!settles("CCNCNVNNV", (1, 2), 8..9, 2..3, 6));
    assert!(settles("CCNCNVNNV", (1, 2), 8..9, 2..3, 7));
}

/// The rows' gap is the largest of theirs, and a non-finite row adds nothing to it.
#[test]
fn a_block_of_rows_is_settled_past_its_largest_gap() {
    let windows: Vec<Window> = [1, 1, 2, 2, 1, 0, 2, 1].map(window).into();
    let class = ConstantClass::of(&windows, 1).unwrap();
    let asked = class.blocks(NonZeroUsize::MIN).unwrap();
    let gap = |rows: Range<usize>| asked.rows(rows, &(0..1)).map(|read: RowsClass| read.gap);
    assert_eq!(gap(4..5), Some(3));
    assert_eq!(gap(5..6), Some(0));
    assert_eq!(gap(6..7), Some(2));
    assert_eq!(gap(7..8), Some(3));
    assert_eq!(gap(4..8), Some(3));
    assert_eq!(gap(3..5), Some(3));
    assert_eq!(gap(0..2), None);
    assert_eq!(asked.gap(0), NO_GAP);
    let two = class.blocks(NonZeroUsize::new(2).unwrap()).unwrap();
    assert_eq!(two.gap(4), 4);
    assert_eq!(two.gap(6), 3);
    assert_eq!(two.gap(1), NO_GAP);
}

/// The third condition at its boundary: the columns end just before, or start just past, the
/// first constant window and the first finite one; with `k` kept, the first `k` of each.
#[test]
fn a_block_holding_a_first_window_is_not_settled() {
    // The first finite window is 1 and the first constant one is 3; the rows are constant.
    let kinds = "NVVCVCCCCCCC";
    assert!(!settles(kinds, (0, 1), 10..12, 1..3, 7));
    assert!(settles(kinds, (0, 1), 10..12, 2..3, 7));
    assert!(!settles(kinds, (0, 1), 10..12, 2..4, 7));
    assert!(!settles(kinds, (0, 1), 10..12, 3..5, 7));
    assert!(settles(kinds, (0, 1), 10..12, 4..6, 7));
    assert!(settles(kinds, (0, 1), 10..12, 0..1, 7));
    // Keeping two: the finite windows 1 and 2, the constant windows 3 and 5.
    assert!(!settles(kinds, (0, 2), 10..12, 2..3, 7));
    assert!(!settles(kinds, (0, 2), 10..12, 5..6, 7));
    assert!(settles(kinds, (0, 2), 10..12, 4..5, 7));
    assert!(settles(kinds, (0, 2), 10..12, 6..8, 5));
}

/// The route harness forces the pass to visit every block: the class answers nothing.
#[test]
#[cfg(not(any(loom, shuttle)))]
fn the_block_query_is_none_when_forced_off() {
    let windows = [window(1), window(2), window(1)];
    let class = ConstantClass::of(&windows, 0).unwrap();
    assert!(class.blocks(NonZeroUsize::MIN).is_some());
    let _route = crate::route::forced_route(crate::route::RouteForce {
        no_constant_blocks: true,
        ..crate::route::RouteForce::default()
    });
    assert!(class.blocks(NonZeroUsize::MIN).is_none());
}

#[test]
fn no_constant_window_builds_no_class() {
    let windows = [window(2), window(0), window(3)];
    assert!(ConstantClass::of(&windows, 1).is_none());
    assert!(ConstantClass::of(&[], 1).is_none());
}

#[test]
fn a_row_the_class_settles_nothing_for_admits_every_constant_lane() {
    assert_eq!(RowClass::NONE.constants_through(), usize::MAX);
    let windows = [window(1), window(2), window(1), window(1)];
    let class = ConstantClass::of(&windows, 0).unwrap();
    let k = NonZeroUsize::new(2).unwrap();
    assert_eq!(class.row(3, k).constants_through(), 2);
    let k = NonZeroUsize::new(4).unwrap();
    assert_eq!(class.row(3, k).constants_through(), usize::MAX);
}

#[test]
fn a_seed_merges_in_and_leaves_what_is_settled() {
    let windows = [window(1), window(2), window(2), window(2)];
    // Row one's zone covers the constant window; row three's does not.
    let class = ConstantClass::of(&windows, 1).unwrap();
    let k = NonZeroUsize::new(1).unwrap();
    let seeded = class.row(3, k);
    assert_eq!((seeded.settled(), seeded.seed()), (3, 0.5));
    let mut row = class.row(1, k);
    assert_eq!((row.settled(), row.seed()), (1, f64::NEG_INFINITY));
    row.merge_seed(&seeded);
    assert_eq!((row.settled(), row.seed()), (1, 0.5));
    assert_eq!(RowClass::NONE.seed(), f64::NEG_INFINITY);
}

/// [`ConstantClass::top_pair`] by its definition: the constant pair outside each other's zone
/// that is first by the earlier window and then the later.
fn top_pair_by_definition(windows: &[Window], e: usize) -> Option<(usize, usize)> {
    let constants: Vec<usize> = (0..windows.len())
        .filter(|&j| constant(&windows[j]))
        .collect();
    constants
        .iter()
        .flat_map(|&a| constants.iter().map(move |&b| (a, b)))
        .find(|&(a, b)| b > a + e)
}

#[test]
fn the_top_pair_is_the_first_constant_pair_outside_each_others_zone() {
    let class = |kinds: &[u8], e: usize| {
        let windows: Vec<Window> = kinds.iter().map(|&k| window(k)).collect();
        ConstantClass::of(&windows, e)
    };
    assert!(class(&[2, 2, 2], 1).is_none());
    assert_eq!(class(&[1, 2, 2, 2], 1).unwrap().top_pair(), None);
    assert_eq!(class(&[1, 1, 1, 2], 2).unwrap().top_pair(), None);
    assert_eq!(class(&[1, 1, 1, 1], 1).unwrap().top_pair(), Some((0, 2)));
    assert_eq!(class(&[2, 1, 0, 1, 1], 1).unwrap().top_pair(), Some((1, 3)));
    assert_eq!(class(&[2, 1, 2, 2, 1], 2).unwrap().top_pair(), Some((1, 4)));
    assert_eq!(class(&[1, 1, 1, 1], 0).unwrap().top_pair(), Some((0, 1)));
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    #[test]
    fn the_top_pair_matches_its_definition(kinds in kinds(), e in 0usize..6) {
        let windows: Vec<Window> = kinds.iter().map(|&k| window(k)).collect();
        if let Some(class) = ConstantClass::of(&windows, e) {
            prop_assert_eq!(class.top_pair(), top_pair_by_definition(&windows, e));
        }
    }
}

/// The exact correlation of window `i` of `a` and window `j` of `b` where their kinds fix it:
/// none with a non-finite window, one between constant windows, one half between a constant and
/// an ordinary one; between two ordinary windows, whatever `ordinary` gives the pair.
fn join_correlation(
    a: &[Window],
    b: &[Window],
    ordinary: &[f64],
    i: usize,
    j: usize,
) -> Option<f64> {
    if !finite(&a[i]) || !finite(&b[j]) {
        return None;
    }
    Some(match (constant(&a[i]), constant(&b[j])) {
        (true, true) => 1.0,
        (true, false) | (false, true) => 0.5,
        (false, false) => ordinary[(i * 31 + j * 17) % ordinary.len()],
    })
}

/// The `k` nearest windows of `b` to window `row` of `a`: the larger correlation first, ties to
/// the smaller index.
fn join_nearest(a: &[Window], b: &[Window], ordinary: &[f64], row: usize, k: usize) -> Vec<usize> {
    let mut ranked: Vec<(f64, usize)> = (0..b.len())
        .filter_map(|j| Some((join_correlation(a, b, ordinary, row, j)?, j)))
        .collect();
    ranked.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)));
    ranked.truncate(k);
    ranked.into_iter().map(|(_, j)| j).collect()
}

fn windows_of(kinds: &[u8]) -> Vec<Window> {
    kinds.iter().map(|&kind| window(kind)).collect()
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    /// No pair of a block the join's class settles is among its row's `k` nearest windows of `b`,
    /// whatever the correlations of the ordinary pairs are, for every block of rows and columns.
    #[test]
    fn a_settled_join_block_holds_no_nearest_pair(
        a_kinds in kinds_up_to(20),
        b_kinds in kinds_up_to(20),
        k in 1usize..5,
        ordinary in prop::collection::vec(
            prop_oneof![Just(-1.0), Just(0.25), Just(0.5), Just(0.75), Just(1.0)],
            7,
        ),
    ) {
        let (a, b) = (windows_of(&a_kinds), windows_of(&b_kinds));
        let Some(class) = JoinClass::of(&a, &b, NonZeroUsize::new(k).unwrap()) else {
            prop_assert!(!a.iter().chain(&b).any(constant));
            return Ok(());
        };
        let nearest: Vec<Vec<usize>> =
            (0..a.len()).map(|row| join_nearest(&a, &b, &ordinary, row, k)).collect();
        for r0 in 0..a.len() {
            for r1 in r0 + 1..=(r0 + 3).min(a.len()) {
                let rows = (r0..r1)
                    .map(|row| class.row_class(row))
                    .reduce(|all, row| all.join(row))
                    .unwrap();
                for c0 in 0..b.len() {
                    for c1 in c0 + 1..=(c0 + 4).min(b.len()) {
                        if !class.settles(rows, &(c0..c1)) {
                            continue;
                        }
                        for (row, nearest) in (r0..r1).zip(&nearest[r0..r1]) {
                            for column in c0..c1 {
                                prop_assert!(
                                    !nearest.contains(&column),
                                    "pair ({}, {}) in rows {:?}, columns {:?}",
                                    row, column, r0..r1, c0..c1
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// The runs of slots the join's class leaves to compute are ascending and disjoint inside the
    /// range, and every pair of a slot it leaves out is none of its row's `k` nearest windows.
    #[test]
    fn the_join_runs_leave_out_no_nearest_pair(
        a_kinds in kinds_up_to(24),
        b_kinds in kinds_up_to(24),
        k in 1usize..4,
        part in 1usize..6,
        at in (0usize..30, 1usize..4, 0usize..30, 1usize..30),
        ordinary in prop::collection::vec(
            prop_oneof![Just(-1.0), Just(0.25), Just(0.5), Just(0.75), Just(1.0)],
            7,
        ),
    ) {
        let (a, b) = (windows_of(&a_kinds), windows_of(&b_kinds));
        let Some(class) = JoinClass::of(&a, &b, NonZeroUsize::new(k).unwrap()) else {
            return Ok(());
        };
        let (n_a, n_b) = (a.len(), b.len());
        let r0 = at.0 % n_a;
        let rows = r0..(r0 + at.1).min(n_a);
        // Slots of the band: `r0` through the last row plus `n_b - 1`.
        let s0 = at.2 % (rows.end + n_b - 1);
        let slots = s0..(s0 + at.3).min(rows.end + n_b - 1);
        let block = rows
            .clone()
            .map(|row| class.row_class(row))
            .reduce(|all, row| all.join(row))
            .unwrap();
        let mut runs = Vec::new();
        class.unsettled_in(block, &rows, &slots, part, |run| runs.push(run));
        prop_assert!(runs.iter().all(|run| slots.start <= run.start && run.start < run.end && run.end <= slots.end));
        prop_assert!(runs.windows(2).all(|pair| pair[0].end < pair[1].start));
        for run in &runs {
            // A run is cut at parts' ends, so a part is left out whole or computed whole.
            prop_assert!(run.start == slots.start || run.start.is_multiple_of(part));
            prop_assert!(run.end == slots.end || run.end.is_multiple_of(part));
        }
        let nearest: Vec<Vec<usize>> =
            rows.clone().map(|row| join_nearest(&a, &b, &ordinary, row, k)).collect();
        for slot in slots.clone().filter(|slot| !runs.iter().any(|run| run.contains(slot))) {
            for row in rows.clone() {
                let Some(column) = (row + n_b - 1).checked_sub(slot).filter(|&c| c < n_b) else {
                    continue;
                };
                prop_assert!(
                    !nearest[row - rows.start].contains(&column),
                    "slot {} pairs row {} with its nearest column {}", slot, row, column
                );
            }
        }
    }
}

/// The join's rule at its boundaries: a constant row needs the first `k` finite and constant
/// columns, an ordinary row every ordinary column and the first `k` constant ones.
#[test]
fn the_join_class_settles_what_the_kinds_fix() {
    // b: ordinary, constant, ordinary, constant, constant, ordinary.
    let b = windows_of(&[2, 1, 2, 1, 1, 2]);
    let class = |a: &[u8], k| JoinClass::of(&windows_of(a), &b, NonZeroUsize::new(k).unwrap());
    // A constant row: only the first finite window (0) and the first constant (1) at k = 1.
    let constant_row = class(&[1], 1).unwrap();
    let row = constant_row.row_class(0);
    assert!(!constant_row.settles(row, &(0..1)));
    assert!(!constant_row.settles(row, &(1..2)));
    assert!(constant_row.settles(row, &(2..6)));
    // At k = 2 the second finite window and the second constant one join them.
    let constant_row = class(&[1], 2).unwrap();
    let row = constant_row.row_class(0);
    assert!(!constant_row.settles(row, &(1..2)));
    assert!(constant_row.settles(row, &(2..3)));
    assert!(!constant_row.settles(row, &(3..4)));
    assert!(constant_row.settles(row, &(4..6)));
    // An ordinary row: every ordinary column and the first constant one.
    let ordinary_row = class(&[2], 1).unwrap();
    let row = ordinary_row.row_class(0);
    assert!(!ordinary_row.settles(row, &(0..1)));
    assert!(!ordinary_row.settles(row, &(1..2)));
    assert!(!ordinary_row.settles(row, &(2..3)));
    assert!(ordinary_row.settles(row, &(3..5)));
    assert!(!ordinary_row.settles(row, &(4..6)));
    // A non-finite row needs nothing.
    let none = class(&[0, 1], 1).unwrap();
    assert!(none.settles(none.row_class(0), &(0..6)));
    // A block with a constant and an ordinary row needs both rows' columns.
    let both = class(&[1, 2], 1).unwrap();
    let rows = both.row_class(0).join(both.row_class(1));
    assert!(!both.settles(rows, &(2..3)));
    assert!(both.settles(rows, &(3..5)));
}

#[test]
fn a_join_without_constant_windows_builds_no_class() {
    let ordinary = windows_of(&[2, 0, 3]);
    let k = NonZeroUsize::MIN;
    assert!(JoinClass::of(&ordinary, &ordinary, k).is_none());
    assert!(JoinClass::of(&[], &[], k).is_none());
    // One constant window on either side is a class.
    let constants = windows_of(&[2, 1]);
    assert!(JoinClass::of(&constants, &ordinary, k).is_some());
    assert!(JoinClass::of(&ordinary, &constants, k).is_some());
}

/// The route harness forces the pass to visit every block: the join builds no class.
#[test]
#[cfg(not(any(loom, shuttle)))]
fn the_join_class_is_none_when_forced_off() {
    let windows = windows_of(&[1, 2, 1]);
    assert!(JoinClass::of(&windows, &windows, NonZeroUsize::MIN).is_some());
    let _route = crate::route::forced_route(crate::route::RouteForce {
        no_constant_blocks: true,
        ..crate::route::RouteForce::default()
    });
    assert!(JoinClass::of(&windows, &windows, NonZeroUsize::MIN).is_none());
}

/// A part inside a long constant stretch of `b` is left out for constant rows, and the parts
/// holding the first finite and the first constant window are not.
#[test]
fn the_join_runs_leave_out_the_parts_inside_a_constant_stretch() {
    let mut kinds = std::vec![1u8; 40];
    kinds[0] = 2;
    kinds[30] = 2;
    let b = windows_of(&kinds);
    let a = windows_of(&[1; 4]);
    let class = JoinClass::of(&a, &b, NonZeroUsize::MIN).unwrap();
    let rows = 0..4;
    let block = (0..4)
        .map(|row| class.row_class(row))
        .reduce(|all, row| all.join(row))
        .unwrap();
    let mut runs = Vec::new();
    // Slots 0..43 of the band (rows 0..4 plus 39): slot `s` meets column `r + 39 - s`.
    class.unsettled_in(block, &rows, &(0..43), 4, |run| runs.push(run));
    let left_out: usize = 43 - runs.iter().map(|run| run.len()).sum::<usize>();
    assert!(left_out > 0, "{runs:?}");
    // The first finite window (0) and the first constant one (1) are never left out; the
    // ordinary window 30 is, for constant rows.
    for row in 0..4 {
        for column in [0, 1] {
            let slot = row + 39 - column;
            assert!(
                runs.iter().any(|run| run.contains(&slot)),
                "{slot} in {runs:?}"
            );
        }
    }
}
