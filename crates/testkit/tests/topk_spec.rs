//! `Nearest::with_k`/`finish_k` against an oracle independent of `nearest.rs`: windows are
//! built only through the public `stats::WindowStats` API, each pair's exact centered cross
//! term `n = m·Σab − Sa·Sb` is recomputed in i128 straight from ticks, and candidates are
//! ranked by their exact correlation (sign, then `ρ²` as a `BigRational`), ties going to the
//! smaller index. Each oracle entry's own bits come from `Nearest::new` + a single
//! `offer` + `finish`, which the spec requires `with_k`'s matching entry to equal bit for bit.

use core::cmp::Ordering;
use core::num::NonZeroUsize;

use ladde::entry::Entry;
use ladde::estimate::rank_estimate;
use ladde::index::SampleIndex;
use ladde::nearest::Nearest;
use ladde::stats::{Window, WindowKind, WindowStats};
use num_bigint::BigInt;
use num_rational::BigRational;
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::TestRunner;

/// `(tick, had_tick)` pairs for one window, in push order.
type Ticks = Vec<(i64, bool)>;

// ---------- building windows only through the public `WindowStats` API ----------

/// `ticks.len()` must equal `m`; pushes every tick and returns the window completed by the
/// last one.
fn window_from_ticks(ticks: &[(i64, bool)]) -> Window {
    let m = NonZeroUsize::new(ticks.len()).expect("m > 0");
    let mut stats = WindowStats::new(m);
    let mut window = None;
    for &(tick, had_tick) in ticks {
        window = stats.push(tick, had_tick);
    }
    window.expect("pushing exactly m ticks completes the window on the last push")
}

/// `n = m·Σ(a·b) − Sa·Sb`, exact in i128, straight from the ticks (the gap placeholder's
/// numeric value is irrelevant: a window with any gap is `NonFinite` and never ranked).
fn exact_cross_term(a: &[(i64, bool)], b: &[(i64, bool)]) -> i64 {
    let m = a.len() as i128;
    let dot: i128 = a
        .iter()
        .zip(b)
        .map(|(&(x, _), &(y, _))| i128::from(x) * i128::from(y))
        .sum();
    let sum_a: i128 = a.iter().map(|&(x, _)| i128::from(x)).sum();
    let sum_b: i128 = b.iter().map(|&(x, _)| i128::from(x)).sum();
    i64::try_from(m * dot - sum_a * sum_b).expect("tick magnitudes in this file keep n within i64")
}

fn index(i: u64) -> SampleIndex {
    SampleIndex::new(i).expect("test indices are below NONE's sentinel")
}

fn bits(entry: Entry) -> (u64, u64, u64) {
    (
        entry.rho().to_bits(),
        entry.squared_distance().to_bits(),
        entry.index().get().unwrap_or(u64::MAX),
    )
}

// ---------- the independent oracle: exact correlation as a signed BigRational ----------

#[derive(Clone)]
enum OracleScore {
    BothConstant,
    OneConstant,
    Valid { sign: i8, rho_squared: BigRational },
}

/// `None` when the pair is ineligible (either window non-finite), matching `offer`'s own
/// filter.
fn oracle_score(row: &Window, neighbor: &Window, n: i64) -> Option<OracleScore> {
    match (row.kind, neighbor.kind) {
        (WindowKind::NonFinite, _) | (_, WindowKind::NonFinite) => None,
        (WindowKind::Constant, WindowKind::Constant) => Some(OracleScore::BothConstant),
        (WindowKind::Constant, WindowKind::Valid { .. })
        | (WindowKind::Valid { .. }, WindowKind::Constant) => Some(OracleScore::OneConstant),
        (
            WindowKind::Valid {
                variance_numerator: vi,
                ..
            },
            WindowKind::Valid {
                variance_numerator: vj,
                ..
            },
        ) => {
            let n_big = BigInt::from(n);
            let denominator = BigInt::from(vi) * BigInt::from(vj);
            Some(OracleScore::Valid {
                sign: n.signum() as i8,
                rho_squared: BigRational::new(&n_big * &n_big, denominator),
            })
        }
    }
}

/// Nearest-first order between two scores of matching sign semantics: `Greater` means `a` is
/// nearer. Signs differ first; equal signs (both zero is the only way two `Valid` scores
/// share sign `0`) compare `ρ²` — reversed for the negative side, since a larger magnitude
/// there is a more negative, hence farther, `ρ`.
fn oracle_rank(a: &OracleScore, b: &OracleScore) -> Ordering {
    fn sign(s: &OracleScore) -> i8 {
        match s {
            OracleScore::BothConstant | OracleScore::OneConstant => 1,
            OracleScore::Valid { sign, .. } => *sign,
        }
    }
    fn magnitude(s: &OracleScore) -> BigRational {
        match s {
            OracleScore::BothConstant => BigRational::from_integer(BigInt::from(1)),
            OracleScore::OneConstant => BigRational::new(BigInt::from(1), BigInt::from(4)),
            OracleScore::Valid { rho_squared, .. } => rho_squared.clone(),
        }
    }
    let (sign_a, sign_b) = (sign(a), sign(b));
    if sign_a != sign_b {
        return sign_a.cmp(&sign_b);
    }
    if sign_a == 0 {
        return Ordering::Equal;
    }
    let magnitude_order = magnitude(a).cmp(&magnitude(b));
    if sign_a > 0 {
        magnitude_order
    } else {
        magnitude_order.reverse()
    }
}

/// The oracle's own top-`k`: ranks every eligible `(index, window, n)` candidate, ties to the
/// smaller index, takes `k`, and reads off each winner's bits from a fresh `Nearest::new` +
/// single `offer` + `finish` (per the spec, this must equal `with_k`'s own entry).
fn oracle_topk(row: &Window, m: usize, candidates: &[(u64, Window, i64)], k: usize) -> Vec<Entry> {
    let mut eligible: Vec<(u64, Window, i64, OracleScore)> = candidates
        .iter()
        .filter_map(|&(candidate_index, window, n)| {
            oracle_score(row, &window, n).map(|score| (candidate_index, window, n, score))
        })
        .collect();
    eligible.sort_by(|(index_a, _, _, score_a), (index_b, _, _, score_b)| {
        match oracle_rank(score_a, score_b) {
            Ordering::Greater => Ordering::Less,
            Ordering::Less => Ordering::Greater,
            Ordering::Equal => index_a.cmp(index_b),
        }
    });
    eligible
        .into_iter()
        .take(k)
        .map(|(candidate_index, window, n, _)| {
            let mut nearest = Nearest::new(row, m);
            nearest.offer(n, &window, index(candidate_index));
            nearest.finish()
        })
        .collect()
}

fn entries_match(actual: &[Entry], expected: &[Entry]) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(&a, &e)| bits(a) == bits(e))
}

// ---------- market-shaped tick generation, biased toward constant/non-finite/tied cases ----------

fn tick_range() -> core::ops::RangeInclusive<i64> {
    -500..=500
}

/// One window's `m` ticks: usually plain random values, sometimes all equal (`Constant`),
/// sometimes carrying a gap (`NonFinite`).
fn ticks_strategy(m: usize) -> impl Strategy<Value = Ticks> {
    prop_oneof![
        6 => prop::collection::vec(tick_range().prop_map(|t| (t, true)), m),
        1 => tick_range().prop_map(move |t| alloc_vec_of((t, true), m)),
        1 => prop::collection::vec(
            prop_oneof![9 => tick_range().prop_map(|t| (t, true)), 1 => Just((0i64, false))],
            m,
        ),
    ]
}

fn alloc_vec_of<T: Clone>(value: T, len: usize) -> Vec<T> {
    core::iter::repeat_n(value, len).collect()
}

/// Overwrites some candidates with an exact copy of the row, an exact copy of another
/// candidate, or a one-tick near miss of the row, so exact repeats and `ρ̂`-ties (whose exact
/// correlation differs) show up regularly.
fn inject_ties(row: Ticks, candidates: Vec<Ticks>) -> impl Strategy<Value = (Ticks, Vec<Ticks>)> {
    let len = candidates.len();
    let original = candidates.clone();
    (
        prop::collection::vec(0usize..4, len),
        prop::collection::vec(0usize..len.max(1), len),
    )
        .prop_map(move |(modes, sources)| {
            let mut candidates = candidates.clone();
            for (i, (&mode, &source)) in modes.iter().zip(&sources).enumerate() {
                match mode {
                    1 => candidates[i] = row.clone(),
                    2 if len > 1 => {
                        let source = if source == i {
                            (source + 1) % len
                        } else {
                            source
                        };
                        candidates[i] = original[source].clone();
                    }
                    3 => {
                        let mut near = row.clone();
                        if let Some((value, had_tick)) = near.first_mut()
                            && *had_tick
                        {
                            *value = value.wrapping_add(1);
                        }
                        candidates[i] = near;
                    }
                    _ => {}
                }
            }
            (row.clone(), candidates)
        })
}

fn case_strategy() -> impl Strategy<Value = (usize, Ticks, Vec<Ticks>)> {
    (1usize..=5, 0usize..=10).prop_flat_map(|(m, num_candidates)| {
        (
            ticks_strategy(m),
            prop::collection::vec(ticks_strategy(m), num_candidates),
        )
            .prop_flat_map(move |(row, candidates)| {
                inject_ties(row, candidates).prop_map(move |(row, candidates)| (m, row, candidates))
            })
    })
}

fn shuffled(len: usize, runner: &mut TestRunner) -> Vec<usize> {
    Just((0..len).collect::<Vec<usize>>())
        .prop_shuffle()
        .new_tree(runner)
        .expect("shuffle strategy produces a value")
        .current()
}

const CASES: u32 = 600;
/// `k` values the spec names explicitly; `usize::MAX` is a stand-in for "k = the number of
/// candidates" (exercised per case below, since it depends on `candidates.len()`).
const FIXED_K: [usize; 4] = [1, 2, 3, 7];

// ---------- property: with_k matches the oracle for every k, offer order, tie, gap ----------

#[test]
fn with_k_matches_the_independent_oracle_across_k_offer_order_and_degenerate_cases() {
    let mut runner = TestRunner::deterministic();
    let strategy = case_strategy();
    let mut nontrivial = 0usize;
    let mut total = 0usize;

    for _ in 0..CASES {
        let (m, row_ticks, candidate_ticks) = strategy
            .new_tree(&mut runner)
            .expect("strategy produces a value")
            .current();
        let row = window_from_ticks(&row_ticks);
        let candidates: Vec<(u64, Window, i64)> = candidate_ticks
            .iter()
            .enumerate()
            .map(|(i, ticks)| {
                (
                    i as u64,
                    window_from_ticks(ticks),
                    exact_cross_term(&row_ticks, ticks),
                )
            })
            .collect();
        let order = shuffled(candidates.len(), &mut runner);
        let all_k = candidates.len().max(1);

        for &k in FIXED_K.iter().chain([all_k].iter()) {
            total += 1;
            let expected = oracle_topk(&row, m, &candidates, k);
            let mut nearest = Nearest::with_k(
                &row,
                m,
                NonZeroUsize::new(k).expect("k values here are all nonzero"),
                usize::MAX,
            );
            for &i in &order {
                let (candidate_index, window, n) = &candidates[i];
                nearest.offer(*n, window, index(*candidate_index));
            }
            let actual = nearest.finish_k();

            assert!(
                entries_match(&actual, &expected),
                "m {m}, k {k}, row {row_ticks:?}, candidates {candidate_ticks:?}, order {order:?}:\n\
                 actual {actual:?}\nexpected {expected:?}"
            );
            if !expected.is_empty() {
                nontrivial += 1;
                assert!(
                    actual[0].squared_distance().is_finite(),
                    "a real winner's distance is never non-finite"
                );
            }
        }
    }

    assert!(
        nontrivial * 4 > total,
        "only {nontrivial} of {total} (case, k) combinations had any eligible candidate; \
         the generator is too degenerate to exercise with_k at all"
    );
}

// ---------- requirement 3: with_k(.., 1) equals new, bit for bit, entry by entry ----------

#[test]
fn with_k_one_equals_new_on_bits() {
    let mut runner = TestRunner::deterministic();
    let strategy = case_strategy();

    for _ in 0..CASES {
        let (m, row_ticks, candidate_ticks) = strategy
            .new_tree(&mut runner)
            .expect("strategy produces a value")
            .current();
        let row = window_from_ticks(&row_ticks);
        let candidates: Vec<(u64, Window, i64)> = candidate_ticks
            .iter()
            .enumerate()
            .map(|(i, ticks)| {
                (
                    i as u64,
                    window_from_ticks(ticks),
                    exact_cross_term(&row_ticks, ticks),
                )
            })
            .collect();
        let order = shuffled(candidates.len(), &mut runner);

        let mut via_new = Nearest::new(&row, m);
        let mut via_with_k = Nearest::with_k(&row, m, NonZeroUsize::new(1).unwrap(), usize::MAX);
        for &i in &order {
            let (candidate_index, window, n) = &candidates[i];
            via_new.offer(*n, window, index(*candidate_index));
            via_with_k.offer(*n, window, index(*candidate_index));
        }
        let from_new = via_new.finish();
        let from_with_k = via_with_k.finish_k();

        assert!(from_with_k.len() <= 1);
        let from_with_k_entry = from_with_k.first().copied().unwrap_or(Entry::NONE);
        assert_eq!(
            bits(from_new),
            bits(from_with_k_entry),
            "row {row_ticks:?}, candidates {candidate_ticks:?}, order {order:?}"
        );
    }
}

// ---------- property: near-tie pairs at adversarial magnitude (targets the 2*epsilon band) ----------

/// `2⁻⁴⁹`, the fast-reject margin; hardcoded here (not reachable from `estimate`'s private
/// `EPSILON`) since it is a fixed, documented constant, not an implementation detail.
const RHO_HAT_EPSILON: f64 = 1.0 / (1u64 << 49) as f64;

/// Large-magnitude base ticks with a single tick nudged by `±1`, the construction that makes
/// `ρ̂` barely move while the exact correlation still changes outright: searches many bases
/// and nudge positions for a pair whose own `ρ̂` (via the public `rank_estimate`) differs from
/// a third, clearly-distinct candidate's by something inside the `(epsilon, 2*epsilon)` band
/// the fast-reject margin must cover.
#[test]
fn near_tie_pairs_at_adversarial_magnitude_match_the_oracle() {
    let mut runner = TestRunner::deterministic();
    let base_strategy = prop::collection::vec(-(1i64 << 30)..=(1i64 << 30), 4);
    let mut narrowest_band_seen = f64::INFINITY;

    for _ in 0..4000 {
        let base = base_strategy
            .new_tree(&mut runner)
            .expect("strategy produces a value")
            .current();
        let mut near_miss = base.clone();
        near_miss[0] = near_miss[0].wrapping_add(1);
        let row_ticks: Vec<(i64, bool)> = base.iter().map(|&t| (t, true)).collect();
        let exact_repeat_ticks: Vec<(i64, bool)> = base.iter().map(|&t| (t, true)).collect();
        let near_miss_ticks: Vec<(i64, bool)> = near_miss.iter().map(|&t| (t, true)).collect();

        let row = window_from_ticks(&row_ticks);
        let candidates = [
            (
                0u64,
                window_from_ticks(&exact_repeat_ticks),
                &exact_repeat_ticks,
            ),
            (1u64, window_from_ticks(&near_miss_ticks), &near_miss_ticks),
        ];
        let (rho_hat_a, rho_hat_b) = match (candidates[0].1.kind, candidates[1].1.kind) {
            (
                WindowKind::Valid {
                    inv_sqrt: inv_a, ..
                },
                WindowKind::Valid {
                    inv_sqrt: inv_b, ..
                },
            ) => {
                let row_inv = match row.kind {
                    WindowKind::Valid { inv_sqrt, .. } => inv_sqrt,
                    _ => continue,
                };
                let n_a = exact_cross_term(&row_ticks, candidates[0].2);
                let n_b = exact_cross_term(&row_ticks, candidates[1].2);
                (
                    rank_estimate(n_a, row_inv, inv_a).clamp(-1.0, 1.0),
                    rank_estimate(n_b, row_inv, inv_b).clamp(-1.0, 1.0),
                )
            }
            _ => continue,
        };
        let band = (rho_hat_a - rho_hat_b).abs();
        if band > 0.0 {
            narrowest_band_seen = narrowest_band_seen.min(band);
        }

        let built_candidates: Vec<(u64, Window, i64)> = candidates
            .iter()
            .map(|&(idx, window, ticks)| (idx, window, exact_cross_term(&row_ticks, ticks)))
            .collect();

        for &k in &[1usize, 2] {
            let expected = oracle_topk(&row, 4, &built_candidates, k);
            let mut nearest = Nearest::with_k(&row, 4, NonZeroUsize::new(k).unwrap(), usize::MAX);
            for &(idx, window, n) in &built_candidates {
                nearest.offer(n, &window, index(idx));
            }
            let actual = nearest.finish_k();
            assert!(
                entries_match(&actual, &expected),
                "base {base:?}, k {k}: actual {actual:?}, expected {expected:?}, \
                 rho_hat_a {rho_hat_a}, rho_hat_b {rho_hat_b}"
            );
        }
    }

    eprintln!(
        "near_tie_pairs_at_adversarial_magnitude: narrowest nonzero |rho_hat_a - rho_hat_b| \
         seen = {narrowest_band_seen:e} (epsilon = {RHO_HAT_EPSILON:e}, 2*epsilon = {:e})",
        2.0 * RHO_HAT_EPSILON
    );
}

// ---------- deterministic cases (literal expected indices) ----------

#[test]
fn deterministic_top_three_of_five_literal_indices() {
    // Five independent windows, all valid, none constant, chosen so their correlations with
    // the row are strictly ordered: index 2 (best) > 0 > 4 > 1 > 3 (worst).
    let row_ticks = [(10i64, true), (20, true), (15, true), (40, true)];
    let candidates_ticks: [[(i64, bool); 4]; 5] = [
        [(10, true), (22, true), (14, true), (39, true)], // 0: close
        [(40, true), (15, true), (20, true), (10, true)], // 1: reversed, poor match
        [(11, true), (21, true), (16, true), (41, true)], // 2: closest
        [(3, true), (90, true), (-5, true), (2, true)],   // 3: worst
        [(12, true), (18, true), (17, true), (38, true)], // 4: middling
    ];
    let row = window_from_ticks(&row_ticks);
    let candidates: Vec<(u64, Window, i64)> = candidates_ticks
        .iter()
        .enumerate()
        .map(|(i, ticks)| {
            (
                i as u64,
                window_from_ticks(ticks),
                exact_cross_term(&row_ticks, ticks),
            )
        })
        .collect();

    let expected = oracle_topk(&row, 4, &candidates, 3);
    let expected_indices: Vec<u64> = expected.iter().map(|e| e.index().get().unwrap()).collect();

    let mut nearest = Nearest::with_k(&row, 4, NonZeroUsize::new(3).unwrap(), usize::MAX);
    for &(idx, window, n) in &candidates {
        nearest.offer(n, &window, index(idx));
    }
    let actual = nearest.finish_k();
    let actual_indices: Vec<u64> = actual.iter().map(|e| e.index().get().unwrap()).collect();

    assert_eq!(actual_indices, expected_indices);
    assert_eq!(actual_indices.len(), 3);
    assert!(entries_match(&actual, &expected));
}

#[test]
fn tie_at_the_cutoff_keeps_the_smaller_index_regardless_of_offer_order() {
    // A: unique best. B and C: bit-identical windows (so their correlation with the row is
    // an exact tie); B has the smaller index. With k = 2, B must be kept and C dropped,
    // whichever order B and C are offered in.
    let row_ticks = [(100i64, true), (50, true), (80, true)];
    let a_ticks = [(101i64, true), (49, true), (81, true)];
    let tied_ticks = [(5i64, true), (200, true), (-30, true)];

    let row = window_from_ticks(&row_ticks);
    let a = window_from_ticks(&a_ticks);
    let tied = window_from_ticks(&tied_ticks);
    let n_a = exact_cross_term(&row_ticks, &a_ticks);
    let n_tied = exact_cross_term(&row_ticks, &tied_ticks);

    for (first, second) in [(1u64, 7u64), (7u64, 1u64)] {
        let mut nearest = Nearest::with_k(&row, 3, NonZeroUsize::new(2).unwrap(), usize::MAX);
        nearest.offer(n_a, &a, index(0));
        for i in [first, second] {
            nearest.offer(n_tied, &tied, index(i));
        }
        let actual = nearest.finish_k();
        let actual_indices: Vec<u64> = actual.iter().map(|e| e.index().get().unwrap()).collect();
        assert_eq!(
            actual_indices,
            vec![0u64, 1u64],
            "offer order {first}, {second}"
        );
    }
}

#[test]
fn fewer_candidates_than_k_returns_all_of_them() {
    let row_ticks = [(1i64, true), (2, true), (9, true)];
    let candidates_ticks = [
        [(3i64, true), (5, true), (1, true)],
        [(0, true), (0, true), (0, true)],
    ];
    let row = window_from_ticks(&row_ticks);

    let mut nearest = Nearest::with_k(&row, 3, NonZeroUsize::new(5).unwrap(), usize::MAX);
    for (i, ticks) in candidates_ticks.iter().enumerate() {
        let window = window_from_ticks(ticks);
        let n = exact_cross_term(&row_ticks, ticks);
        nearest.offer(n, &window, index(i as u64));
    }
    let actual = nearest.finish_k();
    assert_eq!(actual.len(), 2);
    for entry in &actual {
        assert!(entry.squared_distance().is_finite());
    }
}

#[test]
fn zero_candidates_returns_an_empty_vec_and_finish_matches() {
    let row_ticks = [(1i64, true), (2, true), (9, true)];
    let row = window_from_ticks(&row_ticks);
    let nearest = Nearest::with_k(&row, 3, NonZeroUsize::new(4).unwrap(), usize::MAX);
    assert!(nearest.finish_k().is_empty());
}

#[test]
fn non_finite_row_returns_no_entries_for_any_k() {
    let row_ticks = [(1i64, true), (2, false), (9, true)]; // the middle tick is a gap
    let row = window_from_ticks(&row_ticks);
    assert_eq!(row.kind, WindowKind::NonFinite);

    let neighbor_ticks = [(3i64, true), (5, true), (1, true)];
    let neighbor = window_from_ticks(&neighbor_ticks);
    let n = exact_cross_term(&row_ticks, &neighbor_ticks);

    for k in [1usize, 3, 10] {
        let mut nearest = Nearest::with_k(&row, 3, NonZeroUsize::new(k).unwrap(), usize::MAX);
        nearest.offer(n, &neighbor, index(0));
        assert!(nearest.finish_k().is_empty(), "k {k}");
    }
}

#[test]
fn constant_and_non_finite_neighbors_are_ranked_or_excluded_correctly() {
    // Row is valid. Offer: a constant neighbor (rho = 1/2 fixed), a non-finite neighbor
    // (must be excluded entirely), and a valid neighbor that beats the constant one.
    let row_ticks = [(1i64, true), (5, true), (2, true)];
    let row = window_from_ticks(&row_ticks);

    let constant_ticks = [(9i64, true), (9, true), (9, true)];
    let constant = window_from_ticks(&constant_ticks);

    let non_finite_ticks = [(1i64, true), (2, false), (3, true)];
    let non_finite = window_from_ticks(&non_finite_ticks);
    assert_eq!(non_finite.kind, WindowKind::NonFinite);

    let valid_ticks = [(1i64, true), (6, true), (3, true)];
    let valid = window_from_ticks(&valid_ticks);

    let candidates: Vec<(u64, Window, i64)> = vec![
        (0, constant, exact_cross_term(&row_ticks, &constant_ticks)),
        (
            1,
            non_finite,
            exact_cross_term(&row_ticks, &non_finite_ticks),
        ),
        (2, valid, exact_cross_term(&row_ticks, &valid_ticks)),
    ];

    let expected = oracle_topk(&row, 3, &candidates, 3);
    assert!(
        expected.iter().all(|e| e.index().get() != Some(1)),
        "the non-finite neighbor must never appear: {expected:?}"
    );

    let mut nearest = Nearest::with_k(&row, 3, NonZeroUsize::new(3).unwrap(), usize::MAX);
    for &(idx, window, n) in &candidates {
        nearest.offer(n, &window, index(idx));
    }
    let actual = nearest.finish_k();
    assert!(entries_match(&actual, &expected));
    assert_eq!(
        actual.len(),
        2,
        "only the constant and valid neighbors are eligible"
    );
}

#[test]
fn finish_equals_finish_k_first_entry_or_none() {
    let row_ticks = [(1i64, true), (2, true), (9, true)];
    let row = window_from_ticks(&row_ticks);

    // With no offers at all: finish is NONE, finish_k is empty.
    let empty = Nearest::with_k(&row, 3, NonZeroUsize::new(3).unwrap(), usize::MAX);
    assert!(empty.finish_k().is_empty());
    let empty_finish = Nearest::with_k(&row, 3, NonZeroUsize::new(3).unwrap(), usize::MAX);
    assert_eq!(bits(empty_finish.finish()), bits(Entry::NONE));

    // With offers: finish's bits equal finish_k's first entry.
    let neighbor_ticks = [(3i64, true), (5, true), (1, true)];
    let neighbor = window_from_ticks(&neighbor_ticks);
    let n = exact_cross_term(&row_ticks, &neighbor_ticks);

    let mut for_finish_k = Nearest::with_k(&row, 3, NonZeroUsize::new(3).unwrap(), usize::MAX);
    for_finish_k.offer(n, &neighbor, index(0));
    let first_of_finish_k = for_finish_k.finish_k().into_iter().next().unwrap();

    let mut for_finish = Nearest::with_k(&row, 3, NonZeroUsize::new(3).unwrap(), usize::MAX);
    for_finish.offer(n, &neighbor, index(0));
    let from_finish = for_finish.finish();

    assert_eq!(bits(from_finish), bits(first_of_finish_k));
}

#[test]
fn exact_repeat_outranks_a_tied_near_miss_at_k_two() {
    // Same construction as nearest.rs's own tie-breaking pin: a one-tick near miss (index 0) and an
    // exact repeat (index 1) whose rho_hat both clamp to the same value, so only the exact
    // fallback can order them. At k = 2 both appear, but the exact repeat must come first.
    const A: [i64; 4] = [235_732_534, -190_400_685, 248_938_659, -501_918_519];
    const B: [i64; 4] = [235_732_534, -190_400_685, 248_938_660, -501_918_519];
    let row_ticks: Vec<(i64, bool)> = B.iter().map(|&t| (t, true)).collect();
    let a_ticks: Vec<(i64, bool)> = A.iter().map(|&t| (t, true)).collect();
    let b_ticks: Vec<(i64, bool)> = B.iter().map(|&t| (t, true)).collect();

    let row = window_from_ticks(&row_ticks);
    let a = window_from_ticks(&a_ticks);
    let b = window_from_ticks(&b_ticks);
    let n_a = exact_cross_term(&row_ticks, &a_ticks);
    let n_b = exact_cross_term(&row_ticks, &b_ticks);

    for (first, second) in [(0u64, 1u64), (1u64, 0u64)] {
        let mut nearest = Nearest::with_k(&row, 4, NonZeroUsize::new(2).unwrap(), usize::MAX);
        for i in [first, second] {
            if i == 0 {
                nearest.offer(n_a, &a, index(0));
            } else {
                nearest.offer(n_b, &b, index(1));
            }
        }
        let actual = nearest.finish_k();
        let actual_indices: Vec<u64> = actual.iter().map(|e| e.index().get().unwrap()).collect();
        assert_eq!(
            actual_indices,
            vec![1u64, 0u64],
            "offer order {first}, {second}"
        );
        assert_eq!(actual[0].rho(), 1.0);
        assert_eq!(actual[0].squared_distance(), 0.0);
    }
}
