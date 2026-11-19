extern crate std;

use super::naive_multi_profile;
use crate::index::SampleIndex;
use crate::multidim::{MeanDistance, MultiError, MultiProfile, NonFinitePolicy};
use crate::naive::naive_two_sided;
use alloc::vec::Vec;
use core::cmp::Ordering;

type Ticks = Vec<Option<i64>>;

fn profile(dimensions: &[Ticks], m: usize, include: &[usize], discords: bool) -> MultiProfile {
    let slices: Vec<&[Option<i64>]> = dimensions.iter().map(Vec::as_slice).collect();
    naive_multi_profile(&slices, m, include, discords, NonFinitePolicy::Missing).unwrap()
}

fn ticks(values: &[i64]) -> Ticks {
    values.iter().copied().map(Some).collect()
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self, below: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) % below
    }
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    Walk,
    Constant,
    ConstantRuns,
    Period(usize),
    FewLevels,
    OffsetRepeats,
}

const KINDS: [Kind; 9] = [
    Kind::Walk,
    Kind::Constant,
    Kind::ConstantRuns,
    Kind::Period(2),
    Kind::Period(4),
    Kind::Period(7),
    Kind::FewLevels,
    Kind::OffsetRepeats,
    Kind::Walk,
];

fn series(kind: Kind, n: usize, seed: u64) -> Ticks {
    let mut rng = Lcg(seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xD1B54A32D192ED03);
    let mut level = 0i64;
    let motif: Vec<i64> = (0..8).map(|_| rng.next(50) as i64).collect();
    let pattern: Vec<i64> = (0..7).map(|_| rng.next(40) as i64).collect();
    (0..n)
        .map(|t| {
            Some(match kind {
                Kind::Walk => {
                    level += rng.next(21) as i64 - 10;
                    level
                }
                Kind::Constant => 5,
                Kind::ConstantRuns => (t / 7 % 3) as i64 * 4,
                Kind::Period(p) => pattern[t % p],
                Kind::FewLevels => rng.next(3) as i64,
                Kind::OffsetRepeats => motif[t % 8] + 100 * (t / 8 % 3) as i64,
            })
        })
        .collect()
}

fn with_gaps(mut samples: Ticks, at: &[usize]) -> Ticks {
    for &position in at {
        samples[position] = None;
    }
    samples
}

fn eligible(i: usize, j: usize, m: usize) -> bool {
    i.abs_diff(j) > crate::exclusion::zone(m)
}

/// The distance as `f64` arithmetic on the ticks: nothing shared with the reference.
fn float_distance(samples: &[Option<i64>], i: usize, j: usize, m: usize) -> f64 {
    let window = |start: usize| -> Option<Vec<f64>> {
        samples[start..start + m]
            .iter()
            .map(|tick| tick.map(|t| t as f64))
            .collect()
    };
    let (Some(a), Some(b)) = (window(i), window(j)) else {
        return f64::INFINITY;
    };
    let flat = |w: &[f64]| w.iter().all(|&x| x == w[0]);
    match (flat(&a), flat(&b)) {
        (true, true) => return 0.0,
        (true, false) | (false, true) => return (m as f64).sqrt(),
        (false, false) => {}
    }
    let normalize = |w: &[f64]| -> Vec<f64> {
        let mean = w.iter().sum::<f64>() / m as f64;
        let std = (w.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / m as f64).sqrt();
        w.iter().map(|x| (x - mean) / std).collect()
    };
    let (za, zb) = (normalize(&a), normalize(&b));
    za.iter()
        .zip(&zb)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// STUMPY's procedure on `f64`: `means[k][j]` for row `i`.
fn float_means(
    dimensions: &[Ticks],
    m: usize,
    include: &[usize],
    discords: bool,
    i: usize,
) -> Vec<Vec<f64>> {
    let windows = dimensions[0].len() - m + 1;
    let d = dimensions.len();
    let mut means = alloc::vec![alloc::vec![f64::INFINITY; windows]; d];
    for j in (0..windows).filter(|&j| eligible(i, j, m)) {
        let values: Vec<f64> = dimensions
            .iter()
            .map(|samples| float_distance(samples, i, j, m))
            .collect();
        let mut included: Vec<usize> = Vec::new();
        for &dimension in include {
            if !included.contains(&dimension) {
                included.push(dimension);
            }
        }
        let mut rest: Vec<usize> = (0..d).filter(|a| !included.contains(a)).collect();
        rest.sort_by(|&a, &b| {
            let order = values[a].partial_cmp(&values[b]).unwrap();
            if discords && values[a].is_finite() && values[b].is_finite() {
                order.reverse()
            } else {
                order
            }
        });
        let mut sum = 0.0;
        for (k, dimension) in included.into_iter().chain(rest).enumerate() {
            sum += values[dimension];
            means[k][j] = sum / (k + 1) as f64;
        }
    }
    means
}

const TOLERANCE: f64 = 1e-6;

fn assert_matches_float_procedure(
    dimensions: &[Ticks],
    m: usize,
    include: &[usize],
    discords: bool,
    context: &str,
) {
    let result = profile(dimensions, m, include, discords);
    let windows = dimensions[0].len() - m + 1;
    for i in 0..windows {
        let means = float_means(dimensions, m, include, discords, i);
        for (k, row) in means.iter().enumerate() {
            let best = row.iter().copied().fold(f64::INFINITY, f64::min);
            let ours = result.distance[k][i];
            let chosen = result.index[k][i].get();
            if best.is_infinite() {
                assert!(ours.is_missing(), "{context}: k {k} row {i}");
                assert_eq!(chosen, None, "{context}: k {k} row {i}");
            } else {
                assert!(
                    (ours.to_f64() - best).abs() <= TOLERANCE,
                    "{context}: k {k} row {i}: {} vs {best}",
                    ours.to_f64()
                );
                let j = chosen.expect("a finite value has a neighbor") as usize;
                assert!(
                    (row[j] - best).abs() <= TOLERANCE,
                    "{context}: k {k} row {i}: index {j}"
                );
            }
        }
    }
}

#[test]
fn hand_computed_two_dimensions() {
    let sqrt8 = libm::sqrt(8.0);
    let near = |value: MeanDistance, expected: f64| {
        assert!(
            (value.to_f64() - expected).abs() < 1e-12,
            "{} vs {expected}",
            value.to_f64()
        );
    };
    let at = |index: SampleIndex, expected: u64| assert_eq!(index.get(), Some(expected));
    let dimensions = [
        ticks(&[0, 1, 0, 1, 0, 1, 0, 1]),
        ticks(&[0, 0, 1, 1, 0, 0, 1, 1]),
    ];

    // Row 0, columns 2, 3, 4: dimension 0 gives 0, 4, 0 and dimension 1 gives 4, sqrt 8, 0.
    let plain = profile(&dimensions, 4, &[], false);
    near(plain.distance[0][0], 0.0);
    at(plain.index[0][0], 2);
    near(plain.distance[1][0], 0.0);
    at(plain.index[1][0], 4);

    // Row 1, columns 3 and 4: (0, 4) and (4, sqrt 8).
    near(plain.distance[0][1], 0.0);
    at(plain.index[0][1], 3);
    near(plain.distance[1][1], 2.0);
    at(plain.index[1][1], 3);

    // Dimension 1 first: row 1 gives (4, 0) and (sqrt 8, 4).
    let included = profile(&dimensions, 4, &[1], false);
    near(included.distance[0][1], sqrt8);
    at(included.index[0][1], 4);
    near(included.distance[1][1], 2.0);
    at(included.index[1][1], 3);

    // Farthest first: row 1 gives (4, 0) and (4, sqrt 8), so k = 0 ties at 4 and k = 1 is 2.
    let farthest = profile(&dimensions, 4, &[], true);
    near(farthest.distance[0][1], 4.0);
    at(farthest.index[0][1], 3);
    near(farthest.distance[1][1], 2.0);
    at(farthest.index[1][1], 3);
}

#[test]
fn one_dimension_equals_the_two_sided_profile() {
    for (case, kind) in KINDS.into_iter().enumerate() {
        for m in [3, 4, 8] {
            for gaps in [&[][..], &[9, 10, 25][..]] {
                let samples = with_gaps(series(kind, 40, case as u64), gaps);
                let multi = profile(core::slice::from_ref(&samples), m, &[], false);
                let reference = naive_two_sided(&samples, m);
                assert_eq!(multi.distance[0].len(), reference.len());
                for (i, (_, two_sided)) in reference.iter().enumerate() {
                    let nearest = two_sided.nearest;
                    let expected = libm::sqrt(nearest.squared_distance());
                    assert_eq!(
                        multi.distance[0][i].to_f64().to_bits(),
                        expected.to_bits(),
                        "{kind:?} m {m} row {i}"
                    );
                    assert_eq!(multi.index[0][i], nearest.index(), "{kind:?} m {m} row {i}");
                }
            }
        }
    }
}

#[test]
fn first_row_is_the_minimum_over_dimensions() {
    for (case, kind) in KINDS.into_iter().enumerate() {
        for m in [3, 5] {
            let dimensions: Vec<Ticks> = (0..3)
                .map(|a| series(kind, 36, (case * 3 + a) as u64 + 100))
                .collect();
            let joined = profile(&dimensions, m, &[], false);
            let singles: Vec<_> = dimensions
                .iter()
                .map(|samples| naive_two_sided(samples, m))
                .collect();
            for i in 0..joined.distance[0].len() {
                let roots: Vec<f64> = singles
                    .iter()
                    .map(|single| libm::sqrt(single[i].1.nearest.squared_distance()))
                    .collect();
                let best = roots.iter().copied().fold(f64::INFINITY, f64::min);
                assert_eq!(
                    joined.distance[0][i].to_f64().to_bits(),
                    best.to_bits(),
                    "{kind:?} m {m} row {i}"
                );
                let expected = (0..3)
                    .filter(|&a| roots[a] == best)
                    .filter_map(|a| singles[a][i].1.nearest.index().get())
                    .min();
                assert_eq!(joined.index[0][i].get(), expected, "{kind:?} m {m} row {i}");
            }
        }
    }
}

#[test]
fn permuting_the_dimensions_changes_nothing() {
    for (case, kind) in KINDS.into_iter().enumerate() {
        for m in [3, 6] {
            let dimensions: Vec<Ticks> = (0..4)
                .map(|a| series(kind, 36, (case * 4 + a) as u64 + 200))
                .collect();
            let reference = profile(&dimensions, m, &[], false);
            for order in [[3, 2, 1, 0], [1, 3, 0, 2], [2, 0, 3, 1]] {
                let permuted: Vec<Ticks> = order.iter().map(|&a| dimensions[a].clone()).collect();
                let got = profile(&permuted, m, &[], false);
                if matches!(kind, Kind::Walk) {
                    assert_eq!(got, reference, "{kind:?} m {m} order {order:?}");
                    continue;
                }
                // Exactly tied distances of different dimensions can differ in the last bit,
                // so their summation order moves the mean by an ulp, and with it the winner
                // among columns whose means are equal up to that ulp.
                for k in 0..4 {
                    for (a, b) in got.distance[k].iter().zip(&reference.distance[k]) {
                        let (a, b) = (a.to_f64(), b.to_f64());
                        assert!(
                            a == b || (a - b).abs() <= 4.0 * f64::EPSILON * b,
                            "{kind:?} m {m} order {order:?} k {k}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn scaling_one_dimension_by_a_power_of_two_changes_nothing() {
    for (case, kind) in KINDS.into_iter().enumerate() {
        let dimensions: Vec<Ticks> = (0..3)
            .map(|a| series(kind, 36, (case * 3 + a) as u64 + 300))
            .collect();
        let mut scaled = dimensions.clone();
        scaled[1] = scaled[1].iter().map(|t| t.map(|t| t << 3)).collect();
        scaled[2] = scaled[2].iter().map(|t| t.map(|t| t * 4 + 17)).collect();
        for discords in [false, true] {
            assert_eq!(
                profile(&scaled, 5, &[], discords),
                profile(&dimensions, 5, &[], discords),
                "{kind:?} discords {discords}"
            );
        }
    }
}

#[test]
fn identical_dimensions_repeat_the_one_dimensional_profile() {
    for (case, kind) in KINDS.into_iter().enumerate() {
        let samples = series(kind, 36, case as u64 + 400);
        let single = profile(core::slice::from_ref(&samples), 4, &[], false);
        let copies = alloc::vec![samples; 4];
        let joined = profile(&copies, 4, &[], false);
        for k in 0..4 {
            assert_eq!(joined.index[k], single.index[0], "{kind:?} k {k}");
            for (a, b) in joined.distance[k].iter().zip(&single.distance[0]) {
                // Equal values summed k + 1 times and divided back may be an ulp off.
                assert!(
                    (a.to_f64() - b.to_f64()).abs() <= 4.0 * f64::EPSILON * b.to_f64().max(1.0),
                    "{kind:?} k {k}"
                );
            }
        }
    }
}

#[test]
fn procedure_matches_float_arithmetic_on_walks() {
    for seed in 0..3 {
        let dimensions: Vec<Ticks> = (0..4)
            .map(|a| series(Kind::Walk, 40, seed * 4 + a + 500))
            .collect();
        for (include, discords) in [
            (&[][..], false),
            (&[2][..], false),
            (&[3, 1][..], false),
            (&[1, 1, 0][..], false),
            (&[][..], true),
            (&[2][..], true),
        ] {
            assert_matches_float_procedure(
                &dimensions,
                6,
                include,
                discords,
                &std::format!("seed {seed} include {include:?} discords {discords}"),
            );
        }
    }
}

#[test]
fn procedure_matches_float_arithmetic_on_ties_and_gaps() {
    for (case, kind) in KINDS.into_iter().enumerate() {
        let dimensions: Vec<Ticks> = (0..3)
            .map(|a| {
                let samples = series(kind, 30, (case * 3 + a) as u64 + 600);
                if a == 1 {
                    with_gaps(samples, &[0, 11, 12, 29])
                } else {
                    samples
                }
            })
            .collect();
        for m in [3, 4, 7] {
            for (include, discords) in [(&[][..], false), (&[1][..], false), (&[][..], true)] {
                assert_matches_float_procedure(
                    &dimensions,
                    m,
                    include,
                    discords,
                    &std::format!("{kind:?} m {m} include {include:?} discords {discords}"),
                );
            }
        }
    }
}

#[test]
fn constant_dimensions_are_all_zero_and_pick_the_first_eligible_window() {
    let dimensions = alloc::vec![ticks(&[7; 20]); 3];
    let result = profile(&dimensions, 4, &[], false);
    for k in 0..3 {
        for i in 0..17 {
            assert_eq!(result.distance[k][i].to_f64(), 0.0);
            let first = if i > 1 { 0 } else { i + 2 };
            assert_eq!(
                result.index[k][i].get(),
                Some(first as u64),
                "k {k} row {i}"
            );
        }
    }
}

#[test]
fn a_gap_removes_only_the_dimension_it_touches() {
    for gap in [0usize, 14, 29] {
        let m = 5;
        let clean = series(Kind::Walk, 30, 700);
        let gapped = with_gaps(series(Kind::Walk, 30, 701), &[gap]);
        let dimensions = [clean.clone(), gapped];
        let covered = |window: usize| window <= gap && gap < window + m;
        let windows = 30 - m + 1;

        let plain = profile(&dimensions, m, &[], false);
        let first_alone = profile(core::slice::from_ref(&clean), m, &[], false);
        let discord = profile(&dimensions, m, &[], true);
        let gap_first = profile(&dimensions, m, &[1], false);
        for i in 0..windows {
            let reachable = (0..windows).any(|j| eligible(i, j, m) && !covered(j));
            if covered(i) {
                assert!(plain.distance[1][i].is_missing(), "gap {gap} row {i}");
                assert_eq!(plain.index[1][i], SampleIndex::NONE);
                assert_eq!(plain.distance[0][i], first_alone.distance[0][i]);
                assert_eq!(plain.index[0][i], first_alone.index[0][i]);
                assert!(gap_first.distance[0][i].is_missing());
                // The gapped dimension sorts last in discords mode too: only k = 1 is missing.
                assert_eq!(discord.distance[0][i], first_alone.distance[0][i]);
                assert_eq!(discord.index[0][i], first_alone.index[0][i]);
                assert!(discord.distance[1][i].is_missing());
            } else if reachable {
                for (result, k) in [(&plain, 1), (&discord, 1)] {
                    let j = result.index[k][i].get().expect("a clean neighbor") as usize;
                    assert!(!covered(j), "gap {gap} row {i} k {k}");
                    assert!(!result.distance[k][i].is_missing());
                }
            }
        }
    }
}

#[test]
fn input_errors_are_values() {
    let one = ticks(&[1, 2, 3, 4, 5, 6]);
    let short = ticks(&[1, 2, 3]);
    let run = |dimensions: &[&[Option<i64>]], include: &[usize]| {
        naive_multi_profile(dimensions, 3, include, false, NonFinitePolicy::Missing)
    };
    assert_eq!(run(&[], &[]).unwrap_err(), MultiError::NoDimensions);
    assert_eq!(
        run(&[&one, &short], &[]).unwrap_err(),
        MultiError::UnequalLengths
    );
    assert_eq!(
        run(&[&one, &one], &[2]).unwrap_err(),
        MultiError::IncludeOutOfRange
    );
    assert_eq!(
        run(&[&one, &one], &[0, 5]).unwrap_err(),
        MultiError::IncludeOutOfRange
    );
}

#[test]
fn repeated_includes_count_once() {
    let dimensions: Vec<Ticks> = (0..3).map(|a| series(Kind::Walk, 30, a + 800)).collect();
    assert_eq!(
        profile(&dimensions, 4, &[2, 0, 2, 0], false),
        profile(&dimensions, 4, &[2, 0], false)
    );
}

#[test]
fn degenerate_windows_have_no_neighbors() {
    let dimensions = [series(Kind::Walk, 10, 900), series(Kind::Walk, 10, 901)];
    for m in [0, 11] {
        let result = profile(&dimensions, m, &[], false);
        assert_eq!(result.distance.len(), 2);
        assert!(result.distance.iter().all(Vec::is_empty));
        assert!(result.index.iter().all(Vec::is_empty));
    }
    // m = 8 leaves 3 windows, all inside each other's exclusion zone of 2.
    let result = profile(&dimensions, 8, &[], false);
    for k in 0..2 {
        assert!(result.distance[k].iter().all(|value| value.is_missing()));
        assert!(result.index[k].iter().all(|index| index.is_none()));
    }
}

/// Dimensions that are exact scalings of one another have exactly tied distances whose rounded
/// values can differ in the last bit, so the order they are summed in shows in the result.
#[test]
fn exactly_tied_dimensions_are_summed_in_dimension_order() {
    let m = 8;
    let factors = [1i64, 3, 5, 7, 9, 11];
    let mut found = false;
    for seed in 0..400 {
        let base = series(Kind::Walk, 11, seed + 1000);
        let dimensions: Vec<Ticks> = factors
            .iter()
            .map(|&f| base.iter().map(|t| t.map(|t| t * f)).collect())
            .collect();
        // Windows 0 and 3 are the only eligible pair, so each dimension alone gives its
        // distance for that pair.
        let roots: Vec<f64> = dimensions
            .iter()
            .map(|samples| {
                profile(core::slice::from_ref(samples), m, &[], false).distance[0][0].to_f64()
            })
            .collect();
        let mean = |order: &[usize]| {
            let mut sum = 0.0;
            for &a in order {
                sum += roots[a];
            }
            sum / order.len() as f64
        };
        let forward: Vec<usize> = (0..6).collect();
        let backward: Vec<usize> = (0..6).rev().collect();
        if (2..=6)
            .all(|count| mean(&forward[..count]).to_bits() == mean(&backward[..count]).to_bits())
        {
            continue;
        }
        found = true;
        let joined = profile(&dimensions, m, &[], false);
        for k in 0..6 {
            assert_eq!(
                joined.distance[k][0].to_f64().to_bits(),
                mean(&forward[..=k]).to_bits(),
                "seed {seed} k {k}"
            );
        }
        let discord = profile(&dimensions, m, &[], true);
        for k in 0..6 {
            assert_eq!(
                discord.distance[k][0].to_f64().to_bits(),
                mean(&forward[..=k]).to_bits(),
                "seed {seed} discords k {k}"
            );
        }
        break;
    }
    assert!(found, "no seed made the summation order visible");
}

#[test]
fn means_order_by_value() {
    let dimensions = [series(Kind::Walk, 30, 950), series(Kind::Walk, 30, 951)];
    let result = profile(&dimensions, 4, &[], false);
    for i in 0..result.distance[0].len() {
        assert_ne!(
            result.distance[0][i]
                .to_f64()
                .total_cmp(&result.distance[1][i].to_f64()),
            Ordering::Greater,
            "the best single dimension is no farther than the best pair, row {i}"
        );
    }
}

#[test]
fn a_missing_dimension_sorts_last_in_discords_mode() {
    let clean = [series(Kind::Walk, 30, 960), series(Kind::Walk, 30, 961)];
    let gapped = with_gaps(series(Kind::Walk, 30, 962), &[14]);
    let dimensions = [clean[0].clone(), gapped, clean[1].clone()];
    let present = [clean[0].clone(), clean[1].clone()];
    let m = 5;
    let with_gap = profile(&dimensions, m, &[], true);
    let without = profile(&present, m, &[], true);
    for i in (0..=30 - m).filter(|&i| i <= 14 && 14 < i + m) {
        // The two present dimensions, largest first; the gapped one only enters at k = 2.
        for k in 0..2 {
            assert!(!with_gap.distance[k][i].is_missing(), "k {k} row {i}");
            assert_eq!(
                with_gap.distance[k][i], without.distance[k][i],
                "k {k} row {i}"
            );
            assert_eq!(with_gap.index[k][i], without.index[k][i], "k {k} row {i}");
        }
        assert!(with_gap.distance[2][i].is_missing());
    }
}

#[test]
fn a_z_normalized_root_is_within_its_stated_error() {
    use super::dimension_distance;
    use crate::multidim::{DimensionDistance, Terms};
    use crate::naive::window_stats;
    use crate::nearest::ExactDistance;

    assert_eq!(
        ExactDistance::ROOT_RELATIVE_ERROR,
        1.0 / (1u64 << 47) as f64
    );
    let m = 5;
    let mut checked = 0;
    for kind in KINDS {
        for seed in 0..3 {
            let samples = series(kind, 36, seed);
            let windows: Vec<_> = (0..=36 - m).map(|s| window_stats(&samples, s, m)).collect();
            for i in 0..windows.len() {
                for j in 0..windows.len() {
                    if i.abs_diff(j) <= 1 {
                        continue;
                    }
                    let Some(distance) = dimension_distance(&samples, &windows, m, i, j) else {
                        continue;
                    };
                    let root = distance.root();
                    let mut terms = Terms::new();
                    distance.push_exact(false, &mut terms);
                    if root == 0.0 {
                        assert_eq!(terms.sign(), Ordering::Equal, "{kind:?} {seed} {i} {j}");
                        continue;
                    }
                    // root / (1 + τ) ≤ d ≤ root / (1 − τ), with τ = 2^-47.
                    terms.push_scaled(true, root, 1 << 47, (1 << 47) + 1);
                    assert_ne!(terms.sign(), Ordering::Less, "{kind:?} {seed} {i} {j}");
                    terms.clear();
                    distance.push_exact(false, &mut terms);
                    terms.push_scaled(true, root, 1 << 47, (1 << 47) - 1);
                    assert_ne!(terms.sign(), Ordering::Greater, "{kind:?} {seed} {i} {j}");
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 2000, "{checked} pairs");
}
