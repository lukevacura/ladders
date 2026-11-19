use super::roots::vectors;
use super::{
    DimensionDistance, IncludeOrder, MeanDistance, MeanOrder, MeanWork, Terms, error_ratio,
    order_dimensions, order_dimensions_into,
};
use crate::entry::Entry;
use crate::index::SampleIndex;
use crate::nearest::{ExactCorrelation, ExactDistance};
use crate::nonnorm::SquaredTicks;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Ordering;

fn distance(squared: f64) -> Option<ExactDistance> {
    let entry = Entry::new(0.5, squared, SampleIndex::new(0).unwrap()).unwrap();
    Some(ExactDistance::from_parts(
        entry,
        ExactCorrelation::OneConstant,
        1,
    ))
}

/// The order of the mean of `a` against the mean of `b`, which have the same length.
fn order_of<D: DimensionDistance>(a: &[Option<D>], b: &[Option<D>]) -> Ordering {
    let ranking = MeanOrder::new(a.len().max(1), D::ROOT_RELATIVE_ERROR);
    ranking.compare(
        &mut MeanWork::new(),
        &MeanDistance::of(a),
        a,
        &MeanDistance::of(b),
        b,
    )
}

fn ticks(values: &[u128]) -> Vec<Option<SquaredTicks>> {
    values.iter().map(|&v| Some(SquaredTicks(v))).collect()
}

#[test]
fn mean_is_summed_left_to_right() {
    // The big root is 1e16, where adding 1.0 is lost; the order of the terms changes the bits.
    let root = 1e16f64;
    let big = distance(root * root);
    let one = distance(1.0);
    let forward = MeanDistance::of(&[one, one, big]).to_f64();
    let backward = MeanDistance::of(&[big, one, one]).to_f64();
    assert_eq!(forward.to_bits(), ((1.0 + 1.0 + root) / 3.0f64).to_bits());
    assert_eq!(backward.to_bits(), ((root + 1.0 + 1.0) / 3.0f64).to_bits());
    assert_ne!(forward.to_bits(), backward.to_bits());
}

#[test]
fn mean_averages_roots_not_squares() {
    let value = MeanDistance::of(&[distance(1.0), distance(9.0)]).to_f64();
    assert_eq!(value, 2.0);
}

#[test]
fn a_dimension_without_a_distance_is_missing() {
    assert!(MeanDistance::of(&[distance(1.0), None]).is_missing());
    assert!(MeanDistance::<ExactDistance>::of(&[None]).is_missing());
    assert!(MeanDistance::<ExactDistance>::of(&[]).is_missing());
    assert_eq!(
        MeanDistance::<ExactDistance>::of(&[None]),
        MeanDistance::MISSING
    );
    assert!(!MeanDistance::of(&[distance(1.0)]).is_missing());
    assert_eq!(order_of(&[distance(1e300)], &[None]), Ordering::Less);
    assert_eq!(order_of(&[None], &[distance(1e300)]), Ordering::Greater);
    assert_eq!(order_of::<ExactDistance>(&[None], &[None]), Ordering::Equal);
}

#[test]
fn equal_inputs_compare_equal_and_order_follows_the_value() {
    let a = [distance(1.0), distance(4.0)];
    let c = [distance(4.0), distance(4.0)];
    assert_eq!(order_of(&a, &a), Ordering::Equal);
    assert_eq!(MeanDistance::of(&a), MeanDistance::of(&a));
    assert_eq!(order_of(&a, &c), Ordering::Less);
    assert_eq!(order_of(&c, &a), Ordering::Greater);
}

#[test]
fn one_dimension_is_ordered_by_its_exact_distance() {
    // Same rounded distance, different exact correlation: the exact one decides.
    let entry = Entry::new(0.5, 2.0, SampleIndex::new(0).unwrap()).unwrap();
    let with = |cross_squared: u128| {
        Some(ExactDistance::from_parts(
            entry,
            ExactCorrelation::Valid {
                sign: 1,
                cross_squared,
                row_variance: 4,
                neighbor_variance: 4,
                denominator: 16,
            },
            4,
        ))
    };
    let (nearer, farther) = ([with(9)], [with(8)]);
    assert_eq!(
        MeanDistance::of(&nearer).to_f64().to_bits(),
        MeanDistance::of(&farther).to_f64().to_bits()
    );
    assert_eq!(order_of(&nearer, &farther), Ordering::Less);
    assert_eq!(order_of(&farther, &nearer), Ordering::Greater);
    assert_eq!(order_of(&nearer, &nearer), Ordering::Equal);
}

#[test]
fn means_of_roots_that_agree_in_f64_are_ordered_exactly() {
    // √(x² + f²) − x ≈ f²/2x: these two sums of two roots differ by about 1e-13, far under
    // the spacing of f64 at their size, so the values are equal bits and the order is exact.
    let near = vectors::PLAIN
        .iter()
        .filter(|(plus, minus, sign)| plus.len() == 2 && minus.len() == 2 && *sign != 0)
        .filter(|(plus, minus, _)| plus.iter().chain(minus.iter()).all(|&r| r > 1u128 << 40))
        .collect::<Vec<_>>();
    assert!(near.len() >= 20);
    let mut tied_in_f64 = 0;
    for &&(plus, minus, sign) in &near {
        let (a, b) = (ticks(plus), ticks(minus));
        if MeanDistance::of(&a).to_f64() == MeanDistance::of(&b).to_f64() {
            tied_in_f64 += 1;
        }
        let want = i32::from(sign).cmp(&0);
        assert_eq!(order_of(&a, &b), want, "{plus:?} against {minus:?}");
        assert_eq!(order_of(&b, &a), want.reverse());
    }
    assert!(
        tied_in_f64 >= 10,
        "{tied_in_f64} of {} tie in f64",
        near.len()
    );
}

#[test]
fn means_of_equal_length_follow_the_exact_sign_of_every_vector() {
    for &(plus, minus, sign) in vectors::PLAIN {
        if plus.len() != minus.len() {
            continue;
        }
        let want = i32::from(sign).cmp(&0);
        assert_eq!(
            order_of(&ticks(plus), &ticks(minus)),
            want,
            "{plus:?} {minus:?}"
        );
    }
}

#[test]
fn sums_that_agree_as_numbers_but_not_term_by_term_are_equal() {
    // √2 + √8 = √18 + √0 and √50 + √0 = √2 + √32 exactly; f64 sums differ in the last bit.
    assert_eq!(order_of(&ticks(&[2, 8]), &ticks(&[18, 0])), Ordering::Equal);
    assert_eq!(
        order_of(&ticks(&[50, 0]), &ticks(&[2, 32])),
        Ordering::Equal
    );
    assert_eq!(
        order_of(&ticks(&[2, 8, 18]), &ticks(&[72, 0, 0])),
        Ordering::Equal
    );
    assert_eq!(order_of(&ticks(&[2, 8]), &ticks(&[19, 0])), Ordering::Less);
}

#[test]
fn each_prefix_is_of_on_every_prefix() {
    let roots = [3.0, 0.1, 7.0, 1e16, 1.0, 2.5];
    let sets: [Vec<Option<ExactDistance>>; 4] = [
        roots.iter().map(|r| distance(r * r)).collect(),
        vec![distance(1.0), None, distance(4.0)],
        vec![None, distance(1.0)],
        vec![],
    ];
    for dimensions in sets {
        let mut seen = 0;
        MeanDistance::each_prefix(&dimensions, |k, mean| {
            assert_eq!(k, seen);
            seen += 1;
            let want = MeanDistance::of(&dimensions[..=k]);
            assert_eq!(mean.to_f64().to_bits(), want.to_f64().to_bits());
            assert_eq!(mean, want);
            assert_eq!(mean.is_missing(), want.is_missing());
        });
        assert_eq!(seen, dimensions.len());
    }
}

/// The order the contract states, by a stable sort of the dimensions: nearest first (farthest
/// first for discords), a missing key last, ties to the smaller dimension, `include` in front.
fn stable_order(
    keys: &[Option<u32>],
    include: &[usize],
    farthest: bool,
    order: IncludeOrder,
) -> Vec<usize> {
    let mut sorted: Vec<usize> = (0..keys.len()).collect();
    sorted.sort_by(|&a, &b| match (keys[a], keys[b]) {
        (Some(x), Some(y)) if farthest => y.cmp(&x),
        (Some(x), Some(y)) => x.cmp(&y),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    });
    let (front, back): (Vec<usize>, Vec<usize>) =
        sorted.into_iter().partition(|d| include.contains(d));
    match order {
        IncludeOrder::AsGiven => include.iter().copied().chain(back).collect(),
        IncludeOrder::ByDistance => front.into_iter().chain(back).collect(),
    }
}

#[test]
fn many_tied_dimensions_keep_ties_to_the_smaller() {
    // Past twenty elements an in-place sort is no longer an insertion sort, so only the
    // tie-break keeps equal keys in dimension order.
    for count in [21usize, 22, 33, 64] {
        let keys: Vec<Option<u32>> = (0..count)
            .map(|d| {
                if d % 7 == 3 {
                    None
                } else {
                    Some((d % 2) as u32)
                }
            })
            .collect();
        let mut out = Vec::new();
        for include in [vec![], vec![5], vec![count - 1, 3, 8]] {
            for farthest in [false, true] {
                for order in [IncludeOrder::AsGiven, IncludeOrder::ByDistance] {
                    order_dimensions_into(&mut out, &keys, &include, farthest, order);
                    assert_eq!(
                        out,
                        stable_order(&keys, &include, farthest, order),
                        "{count} {include:?} {farthest} {order:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn dimensions_are_ordered_in_place_with_ties_to_the_smaller() {
    let keys = [Some(4), Some(1), None, Some(4), Some(0), Some(4), None];
    let mut out = Vec::with_capacity(keys.len());
    for (include, farthest, order, want) in [
        (&[][..], false, IncludeOrder::AsGiven, [4, 1, 0, 3, 5, 2, 6]),
        (&[][..], true, IncludeOrder::AsGiven, [0, 3, 5, 1, 4, 2, 6]),
        (
            &[6, 3][..],
            false,
            IncludeOrder::AsGiven,
            [6, 3, 4, 1, 0, 5, 2],
        ),
        (
            &[6, 3][..],
            false,
            IncludeOrder::ByDistance,
            [3, 6, 4, 1, 0, 5, 2],
        ),
        (
            &[2][..],
            true,
            IncludeOrder::ByDistance,
            [2, 0, 3, 5, 1, 4, 6],
        ),
    ] {
        order_dimensions_into(&mut out, &keys, include, farthest, order);
        assert_eq!(out, want, "{include:?} {farthest} {order:?}");
        assert_eq!(out.capacity(), keys.len(), "no growth");
        assert_eq!(order_dimensions(&keys, include, farthest, order), want);
    }
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }

    /// A radicand of any size up to 127 bits, small and power-of-two ones included.
    fn radicand(&mut self) -> u128 {
        let bits = 1 + self.next() % 127;
        let raw = (u128::from(self.next()) << 64 | u128::from(self.next())) >> (128 - bits);
        match self.next() % 8 {
            0 => 1u128 << (bits - 1),
            1 => (1u128 << bits) - 1,
            _ => raw.max(1),
        }
    }
}

#[test]
fn the_root_of_a_squared_distance_is_within_its_stated_error() {
    assert_eq!(SquaredTicks::ROOT_RELATIVE_ERROR, 1.0 / (1u64 << 51) as f64);
    let mut rng = Lcg(3);
    for _ in 0..3000 {
        let radicand = rng.radicand();
        let root = SquaredTicks(radicand).root();
        // root / (1 + τ) ≤ √R ≤ root / (1 − τ), with τ = 2^-51.
        let mut terms = Terms::new();
        terms.push_root(false, radicand);
        terms.push_scaled(true, root, 1 << 51, (1 << 51) + 1);
        assert_ne!(terms.sign(), Ordering::Less, "{radicand}");
        terms.clear();
        terms.push_root(false, radicand);
        terms.push_scaled(true, root, 1 << 51, (1 << 51) - 1);
        assert_ne!(terms.sign(), Ordering::Greater, "{radicand}");
    }
}

#[test]
fn a_mean_is_within_the_error_the_order_allows_for() {
    let ranking = MeanOrder::new(8, SquaredTicks::ROOT_RELATIVE_ERROR);
    let mut rng = Lcg(9);
    for round in 0..2000 {
        let count = 1 + round % 8;
        let radicands: Vec<u128> = (0..count).map(|_| rng.radicand()).collect();
        let value = MeanDistance::of(&ticks(&radicands)).to_f64();
        // The order treats two values as different only past a ratio 1 + 3.75·(error bound), so
        // the mean's own error must stay well inside a third of the excess.
        let allowed = (ranking.ratio(count - 1) - 1.0) / 3.0;
        let scale = 1u64 << 60;
        let excess = (allowed * scale as f64).ceil() as u64;
        let k = count as u64;
        // sum ≥ value·k / (1 + allowed) and sum ≤ value·k / (1 − allowed).
        let mut terms = Terms::new();
        for &radicand in &radicands {
            terms.push_root(false, radicand);
        }
        terms.push_scaled(true, value, k * scale, scale + excess);
        assert_ne!(terms.sign(), Ordering::Less, "{radicands:?}");
        terms.clear();
        for &radicand in &radicands {
            terms.push_root(false, radicand);
        }
        terms.push_scaled(true, value, k * scale, scale - excess);
        assert_ne!(terms.sign(), Ordering::Greater, "{radicands:?}");
    }
}

#[test]
fn the_ratio_between_decided_values_grows_with_the_count_and_the_root_error() {
    let narrow = MeanOrder::new(40, SquaredTicks::ROOT_RELATIVE_ERROR);
    let wide = MeanOrder::new(40, ExactDistance::ROOT_RELATIVE_ERROR);
    for k in 1..40 {
        assert!(narrow.ratio(k) > narrow.ratio(k - 1));
        assert!(wide.ratio(k) > narrow.ratio(k));
    }
    assert!(narrow.ratio(0) > 1.0);
    // Past a count where the rounding bound itself is no longer small nothing is decided by value.
    assert!(error_ratio(1 << 50, 0.0).is_infinite());
    assert!(error_ratio(1 << 40, 0.0).is_finite());
}
/// A dimension's distance whose exact value (`exact`, a whole number) and f64 root disagree by
/// an ulp; two are equal when their exact values are.
#[derive(Clone, Copy, Debug)]
struct Noisy {
    exact: u8,
    ulps: i8,
}

impl PartialEq for Noisy {
    fn eq(&self, other: &Self) -> bool {
        self.exact == other.exact
    }
}

impl Eq for Noisy {}

impl PartialOrd for Noisy {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Noisy {
    fn cmp(&self, other: &Self) -> Ordering {
        self.exact.cmp(&other.exact)
    }
}

impl DimensionDistance for Noisy {
    fn root(self) -> f64 {
        let base = f64::from(self.exact);
        f64::from_bits((base.to_bits() as i64 + i64::from(self.ulps)) as u64)
    }

    const ROOT_RELATIVE_ERROR: f64 = 1.0 / (1u64 << 51) as f64;

    fn push_exact(self, negative: bool, terms: &mut Terms) {
        terms.push_root(negative, u128::from(self.exact) * u128::from(self.exact));
    }

    fn fixed(self) -> Option<bool> {
        None
    }
}

/// The order of means of the same number of dimensions is a total order, exact whatever their
/// f64 values round to: antisymmetric, transitive, a sort of any mix is consistent, and means
/// whose exact distances are equal compare equal even when their f64 values differ.
#[test]
fn mean_order_is_total_over_means_of_one_count() {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next = |bound: u64| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) % bound
    };
    let ranking = MeanOrder::new(3, Noisy::ROOT_RELATIVE_ERROR);
    let mut equal_with_different_values = 0;
    for k in 1..=3usize {
        for _ in 0..60 {
            let mut sets: Vec<Vec<Option<Noisy>>> = vec![vec![None; k]];
            for _ in 0..16 {
                sets.push(
                    (0..k)
                        .map(|_| {
                            Some(Noisy {
                                exact: 1 + next(4) as u8,
                                ulps: next(3) as i8 - 1,
                            })
                        })
                        .collect(),
                );
            }
            let mut work = MeanWork::new();
            let mut cmp = |a: &Vec<Option<Noisy>>, b: &Vec<Option<Noisy>>| {
                let (x, y) = (MeanDistance::of(a), MeanDistance::of(b));
                let order = ranking.compare(&mut work, &x, a, &y, b);
                if order == Ordering::Equal && x.to_f64() != y.to_f64() {
                    equal_with_different_values += 1;
                }
                order
            };
            for a in &sets {
                for b in &sets {
                    assert_eq!(cmp(a, b), cmp(b, a).reverse(), "{a:?} {b:?}");
                    for c in &sets {
                        if cmp(a, b) != Ordering::Greater && cmp(b, c) != Ordering::Greater {
                            assert_ne!(cmp(a, c), Ordering::Greater, "{a:?} {b:?} {c:?}");
                        }
                    }
                }
            }
            let mut sorted = sets.clone();
            sorted.sort_by(&mut cmp);
            assert!(
                sorted
                    .windows(2)
                    .all(|w| cmp(&w[0], &w[1]) != Ordering::Greater)
            );
            assert!(sorted.last().unwrap().iter().all(Option::is_none));
        }
    }
    assert!(equal_with_different_values > 0);
}

/// A bound looser than the arithmetic needs is still correct but sends near-ties that the values
/// could decide to the exact search, so the excess over 1 is pinned to `3.75·(τ + γ)`: at least
/// the `2.1·e` the proof needs, and not a million times more.
#[test]
fn the_ratio_between_decided_values_is_tight() {
    let unit = 1.0 / (1u64 << 53) as f64;
    for root_error in [
        SquaredTicks::ROOT_RELATIVE_ERROR,
        ExactDistance::ROOT_RELATIVE_ERROR,
    ] {
        for count in [1usize, 2, 3, 8, 64, 1000] {
            let operations = (count + 1) as f64 * unit;
            let error = root_error + operations / (1.0 - operations);
            let excess = error_ratio(count, root_error) - 1.0;
            assert!(
                excess >= 2.1 * error,
                "{count}: {excess:e} against {error:e}"
            );
            assert!(
                excess <= 3.8 * error,
                "{count}: {excess:e} against {error:e}"
            );
        }
    }
    // And a pair of ordinary means is decided by value: well apart, and a few ulps apart.
    let ranking = MeanOrder::new(3, ExactDistance::ROOT_RELATIVE_ERROR);
    let (a, b) = (
        MeanDistance::<ExactDistance>::new(1.0, 2),
        MeanDistance::new(1.0 + 1e-9, 2),
    );
    assert_eq!(ranking.by_value(1, &a, &b), Some(Ordering::Less));
}
