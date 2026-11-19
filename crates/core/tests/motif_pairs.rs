//! The pair of `motif_pairs`: the pair of highest exact correlation, ties to the smaller `(a, b)`,
//! `a < b`, against a brute force over every pair of windows in exact integer arithmetic, at 1, 2
//! and 8 threads, on ties, exact repeats and flat runs; and the series on which a pair's two
//! profile entries differ by a rounding.

use core::cmp::Ordering;
use core::num::NonZeroUsize;
use ladde::batch::two_sided_profile;
use ladde::entry::Entry;
use ladde::index::SampleIndex;
use ladde::motif::{motif_pairs, pick_pair};
use ladde::quantize::Scale;
use proptest::prelude::*;
use testkit::generators::{seeded_default, with_flat_runs, with_repeats};

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

/// A correlation as `(sign, numerator, denominator)` of its square: its value is
/// `sign * numerator / denominator` under a root.
type Rho = (i8, i128, i128);

fn cmp_rho(a: Rho, b: Rho) -> Ordering {
    match a.0.cmp(&b.0) {
        Ordering::Equal if a.0 == 0 => Ordering::Equal,
        Ordering::Equal => {
            let by_square = (a.1 * b.2).cmp(&(b.1 * a.2));
            if a.0 > 0 {
                by_square
            } else {
                by_square.reverse()
            }
        }
        unequal => unequal,
    }
}

/// The exact correlation of the windows of `x` at `i` and `j`, with the constant windows as the
/// library has them: one between two, one half between a constant and any other.
fn rho(x: &[i128], m: usize, i: usize, j: usize) -> Rho {
    let window = |s: usize| {
        let w = &x[s..s + m];
        let (sum, squares) = (
            w.iter().sum::<i128>(),
            w.iter().map(|v| v * v).sum::<i128>(),
        );
        (sum, m as i128 * squares - sum * sum)
    };
    let ((si, vi), (sj, vj)) = (window(i), window(j));
    match (vi == 0, vj == 0) {
        (true, true) => (1, 1, 1),
        (true, false) | (false, true) => (1, 1, 4),
        (false, false) => {
            let cross: i128 = (0..m).map(|k| x[i + k] * x[j + k]).sum();
            let n = m as i128 * cross - si * sj;
            (n.signum() as i8, n * n, vi * vj)
        }
    }
}

/// The best pair by brute force: highest exact correlation, ties to the smaller `(a, b)`.
fn brute_force(series: &[f64], m: usize) -> Option<(usize, usize)> {
    let x: Vec<i128> = series.iter().map(|&v| v as i128).collect();
    let windows = (x.len() + 1).checked_sub(m)?;
    let zone = m.div_ceil(4);
    let mut best: Option<((usize, usize), Rho)> = None;
    for i in 0..windows {
        for j in i + zone + 1..windows {
            let r = rho(&x, m, i, j);
            if best.is_none_or(|(_, held)| cmp_rho(r, held) == Ordering::Greater) {
                best = Some(((i, j), r));
            }
        }
    }
    best.map(|(pair, _)| pair)
}

fn searched(series: &[f64], m: usize, threads: usize) -> Option<(usize, usize)> {
    let scale = Scale::new(1.0).unwrap();
    let pairs = motif_pairs(series, nz(m), scale, nz(threads)).expect("small series fit");
    assert!(pairs.len() <= 1);
    pairs.first().map(|p| {
        assert!(p.a < p.b, "{p:?}");
        assert!(p.distance.is_finite() && p.distance >= 0.0, "{p:?}");
        (p.a, p.b)
    })
}

/// One pair with one exact correlation whose two profile entries differ by one place: window 0's
/// distance to window 6 is 2.345009105857163 and window 6's to window 0 is
/// 2.3450091058571627, so ranking on the entries would give the later window first.
#[test]
fn a_pair_whose_entries_differ_by_a_rounding_is_reported_earlier_window_first() {
    let series: Vec<f64> = [
        674, 986, 474, 901, 1797, 459, 649, 667, 123, 575, 1431, 1085, 27, 235, 950,
    ]
    .map(f64::from)
    .to_vec();
    assert_eq!(brute_force(&series, 8), Some((0, 6)));
    for threads in [1, 2, 8] {
        assert_eq!(searched(&series, 8, threads), Some((0, 6)), "{threads}");
    }
    let scale = Scale::new(1.0).unwrap();
    let pair = motif_pairs(&series[..], nz(8), scale, nz(1)).unwrap()[0];
    assert_eq!(pair.distance.to_bits(), 2.345009105857163f64.to_bits());
}

/// Three copies of a stretch tie every pair of them; the smallest `(a, b)` is the first two.
#[test]
fn equal_correlations_go_to_the_smaller_pair() {
    let mut state = 12345u64;
    let mut series: Vec<f64> = (0..300)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) % 1000) as f64
        })
        .collect();
    let stretch = series[20..70].to_vec();
    series[120..170].copy_from_slice(&stretch);
    series[230..280].copy_from_slice(&stretch);
    for m in [8, 16] {
        assert_eq!(brute_force(&series, m), Some((20, 120)), "m {m}");
        for threads in [1, 2, 8] {
            assert_eq!(searched(&series, m, threads), Some((20, 120)), "m {m}");
        }
    }
}

#[test]
fn no_pair_gives_an_empty_answer() {
    assert_eq!(searched(&[1.0, 2.0, 3.0], 8, 1), None);
    assert_eq!(
        searched(&[1.0, 5.0, 2.0, 7.0, 3.0, 9.0, 4.0, 8.0, 1.0], 8, 2),
        None
    );
    assert_eq!(searched(&[f64::NAN; 40], 8, 1), None);
    assert_eq!(searched(&[], 4, 1), None);
}

/// Two different pairs whose exact correlations are equal (the first six windows' shape recurs
/// shifted) but whose f64 entries are a rounding apart the other way round: the least entry
/// (window 7 to window 2, 0.30199641080498973) is of the pair (2, 7), while the tie rule's pair
/// is (1, 21), whose entries are 0.30199641080498985. Found by a random search; the full
/// profile's least entry is the wrong pair here.
#[test]
fn equal_pairs_with_entries_ordered_against_the_tie_rule_go_to_the_smaller_pair_a() {
    let series: Vec<f64> = [
        3, 1, 1, 0, 4, 1, 4, 1, 1, 4, 1, 3, 2, 2, 4, 1, 5, 4, 0, 1, 2, 3, 3, 3, 5,
    ]
    .map(f64::from)
    .to_vec();
    pinned(&series, 4, (1, 21), 0x3fe195d91b3b158d);
}

/// As above, on a series of large values with a few levels (m 4, windows 1 and 10 against 2 and
/// 11).
#[test]
fn equal_pairs_with_entries_ordered_against_the_tie_rule_go_to_the_smaller_pair_b() {
    let series: Vec<f64> = [
        100003, 100001, 100003, 100003, 100001, 100000, 100002, 100002, 100006, 100004, 100001,
        100003, 100003, 100000, 100000, 100002, 100000, 100004, 100006, 100004,
    ]
    .map(f64::from)
    .to_vec();
    pinned(&series, 4, (1, 10), 0x3fe195d91b3b158d);
}

/// As above at m 15 on 99 samples: the least entry is of (1, 39), the tie rule's pair is (0, 38).
#[test]
fn equal_pairs_with_entries_ordered_against_the_tie_rule_go_to_the_smaller_pair_c() {
    let series: Vec<f64> = [
        100002, 100002, 100001, 100002, 100000, 100004, 100006, 100006, 100006, 100004, 100006,
        100004, 100002, 100006, 100005, 100003, 100004, 100001, 100002, 100004, 100000, 100006,
        100005, 100003, 100003, 100000, 100000, 100002, 100001, 100006, 100005, 100000, 100002,
        100002, 100000, 100000, 100000, 100001, 100003, 100002, 100001, 100002, 100000, 100004,
        100006, 100006, 100006, 100004, 100006, 100004, 100002, 100006, 100005, 100002, 100006,
        100002, 100003, 100001, 100004, 100003, 100000, 100006, 100000, 100002, 100005, 100005,
        100002, 100001, 100002, 100006, 100004, 100004, 100006, 100003, 100004, 100000, 100000,
        100003, 100000, 100002, 100002, 100004, 100004, 100006, 100006, 100004, 100005, 100002,
        100004, 100002, 100006, 100004, 100006, 100002, 100006, 100001, 100000, 100000, 100005,
    ]
    .map(f64::from)
    .to_vec();
    pinned(&series, 15, (0, 38), 0x3fde8e01c4e0233c);
}

fn pinned(series: &[f64], m: usize, pair: (usize, usize), distance_bits: u64) {
    assert_eq!(brute_force(series, m), Some(pair));
    let scale = Scale::new(1.0).unwrap();
    for threads in [1, 2, 8] {
        let found = motif_pairs(series, nz(m), scale, nz(threads)).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!((found[0].a, found[0].b), pair, "{threads} threads");
        assert_eq!(
            found[0].distance.to_bits(),
            distance_bits,
            "{threads} threads"
        );
    }
}

/// `pick_pair` on a profile whose f64 entries order two different pairs against their exact
/// order: the better pair's entries are one `2^-45` relative step above the worse pair's, so the
/// least entry is the worse pair's, and only the band and the exact ranking find the better one.
#[test]
fn the_band_and_the_exact_ranking_pick_the_better_pair_over_the_least_entry() {
    let mut state = 99u64;
    let series: Vec<f64> = (0..120)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) % 1000) as f64
        })
        .collect();
    let (m, scale) = (8, Scale::new(1.0).unwrap());
    let mut profile = two_sided_profile(&series[..], nz(m), scale, nz(1));
    let (a, b) = brute_force(&series, m).expect("a pair");
    let a_entry = profile[a].1.nearest;
    assert_eq!(a_entry.index().get(), Some(b as u64));
    let better = a_entry.squared_distance();
    // The nearest pair outside the better pair's windows: the worse pair.
    let (c, d) = profile
        .iter()
        .filter(|&&(w, sides)| {
            let nb = sides.nearest.index().get().unwrap_or(w);
            ![a as u64, b as u64].contains(&w) && ![a as u64, b as u64].contains(&nb)
        })
        .min_by(|x, y| {
            x.1.nearest
                .squared_distance()
                .total_cmp(&y.1.nearest.squared_distance())
        })
        .map(|&(w, sides)| (w as usize, sides.nearest.index().get().unwrap() as usize))
        .expect("a second pair");
    let (c, d) = (c.min(d), c.max(d));
    assert!(
        profile[c].1.nearest.squared_distance() > better,
        "the pairs differ"
    );
    let lowered = better * (1.0 - 1.0 / (1u64 << 45) as f64);
    for (from, to) in [(a, b), (b, a), (c, d), (d, c)] {
        let held = profile[from].1.nearest;
        let squared = if [a, b].contains(&from) {
            better
        } else {
            lowered
        };
        let entry = Entry::new(held.rho(), squared, SampleIndex::new(to as u64).unwrap()).unwrap();
        profile[from].1.nearest = entry;
    }
    let picked = pick_pair(&series[..], nz(m), scale, &profile).expect("a pair");
    assert_eq!(
        (picked.a, picked.b),
        (a, b),
        "the worse pair has the least entry"
    );
    assert_eq!(picked.distance.to_bits(), better.sqrt().to_bits());
}

proptest! {
    #![proptest_config(seeded_default())]

    /// Few small integer levels (ties everywhere), plain or with flat runs and exact repeats.
    #[test]
    fn the_pair_is_the_brute_force_pair(
        (m, series) in prop_oneof![Just(4usize), Just(8), Just(12), Just(16)].prop_flat_map(|m| {
            let base = prop::collection::vec(0u8..6, 40..200)
                .prop_map(|v| v.into_iter().map(f64::from).collect::<Vec<_>>());
            let shaped = prop_oneof![
                base.clone(),
                with_flat_runs(base.clone(), m),
                with_repeats(base.clone(), m),
                with_flat_runs(with_repeats(base, m), m),
            ];
            (Just(m), shaped)
        })
    ) {
        let expected = brute_force(&series, m);
        for threads in [1, 2, 8] {
            prop_assert_eq!(searched(&series, m, threads), expected, "{} threads", threads);
        }
    }

    /// Wider values, so that exact correlations rarely tie and near-ties are rounding apart.
    #[test]
    fn the_pair_is_the_brute_force_pair_on_wide_values(
        m in 4usize..=10,
        series in prop::collection::vec(0u16..2000, 30..80)
            .prop_map(|v| v.into_iter().map(f64::from).collect::<Vec<_>>()),
    ) {
        let expected = brute_force(&series, m);
        for threads in [1, 8] {
            prop_assert_eq!(searched(&series, m, threads), expected, "{} threads", threads);
        }
    }

    /// Large values with a few levels and near repeats: the series on which two different pairs
    /// tie exactly while their f64 entries differ by a rounding.
    #[test]
    fn the_pair_is_the_brute_force_pair_on_offset_values_with_few_levels(
        m in 4usize..=16,
        base in prop::collection::vec(0u8..7, 15..200),
        repeat in (0usize..200, 0usize..200, 0usize..=16),
    ) {
        let mut series: Vec<f64> = base.iter().map(|&v| 100_000.0 + f64::from(v)).collect();
        let (from, to, extra) = repeat;
        let len = m + extra % (m + 1);
        if series.len() > 2 * len + 2 {
            let a = from % (series.len() - 2 * len);
            let b = a + len + to % (series.len() - a - 2 * len + 1);
            for k in 0..len {
                series[b + k] = series[a + k];
            }
        }
        let expected = brute_force(&series, m);
        for threads in [1, 8] {
            prop_assert_eq!(searched(&series, m, threads), expected, "{} threads", threads);
        }
    }
}
