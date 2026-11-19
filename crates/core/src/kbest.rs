//! The one k-best keeper every engine ranks through: a short list, best first, under a strict
//! order the caller gives, that takes a new item only when it beats the list's last once the list
//! is full. Insertion shifts at most the items the new one beats, one slot each, from the back:
//! no search over the whole list, no memmove call, and no allocation once the list's capacity
//! holds its `k`. Ties are whatever the order says: an item that does not beat a kept one stays
//! behind it, so an index in the order sends ties to the smaller index.
//!
//! Two other forms serve lists whose shape a large `k` would make quadratic: a bound heap with the
//! worst at the root, for a list read only through its `k`-th best, and a selection that ranks
//! many offers at once. Both keep the same items the list would.

use alloc::vec::Vec;
use core::cmp::Ordering;
use core::marker::PhantomData;

use crate::estimate::{Producer, band_floor};

/// Lanes per block a [`Producer`] reports one maximum for.
pub(crate) const BLOCK_LANES: usize = 64;

/// Whether a block whose producer value is `value` (its largest estimate, or an upper bound of
/// its exact correlations) is below `floor`, so a consumer need not read it: strict, so a block
/// that reaches a floor exactly, and so may hold a tie, is always read. The one rule every block
/// gate applies, after the kernel to a lane block's maximum and before it to a block of pairs.
#[inline(always)]
pub(crate) fn below_floor(value: f64, floor: f64) -> bool {
    value < floor
}

/// The lower of two floors as [`below_floor`] reads them: `below_floor(v, lower_floor(a, b))` is
/// `below_floor(v, a) && below_floor(v, b)` for every `v`, so a block's floors folded through it
/// are cleared exactly when `below_floor` clears every one of them. A NaN floor, which nothing is below,
/// is kept.
#[inline(always)]
pub(crate) fn lower_floor(a: f64, b: f64) -> f64 {
    if a.is_nan() || a <= b { a } else { b }
}

/// One span of a producer's estimates, with the maximum of each [`BLOCK_LANES`]-lane block (the
/// last may be shorter): what the keeper reads, a block at a time, only where a block's maximum
/// clears its floor.
pub(crate) struct Blocks<'a, P> {
    estimates: &'a [f64],
    maxima: &'a [f64],
    producer: PhantomData<P>,
}

impl<'a, P: Producer> Blocks<'a, P> {
    /// `maxima[b]` is the largest of `estimates`' block `b`.
    #[inline(always)]
    pub(crate) fn new(estimates: &'a [f64], maxima: &'a [f64]) -> Self {
        debug_assert_eq!(maxima.len(), estimates.len().div_ceil(BLOCK_LANES));
        Self {
            estimates,
            maxima,
            producer: PhantomData,
        }
    }

    /// Block `b`'s estimates.
    #[inline(always)]
    pub(crate) fn block(&self, b: usize) -> &'a [f64] {
        let lo = b * BLOCK_LANES;
        &self.estimates[lo..(lo + BLOCK_LANES).min(self.estimates.len())]
    }

    /// Every block's maximum, in order.
    #[inline(always)]
    pub(crate) fn maxima(&self) -> &'a [f64] {
        self.maxima
    }

    /// The largest estimate of the span, `−∞` when it is empty.
    #[inline(always)]
    pub(crate) fn maximum(&self) -> f64 {
        self.maxima
            .iter()
            .fold(f64::NEG_INFINITY, |max, &m| max.max(m))
    }

    /// Calls `hit(lane, estimate)` for every lane at or above `floor`, ascending, reading only
    /// the blocks whose maximum clears it, eight lanes at a time, until `hit` returns `false`:
    /// the lanes after that one are not read.
    #[inline(always)]
    pub(crate) fn at_or_above(&self, floor: f64, mut hit: impl FnMut(usize, f64) -> bool) {
        for (b, &max) in self.maxima.iter().enumerate() {
            if below_floor(max, floor) {
                continue;
            }
            for (part, lanes) in self.block(b).chunks(8).enumerate() {
                if !lanes.iter().fold(false, |any, &v| any | (v >= floor)) {
                    continue;
                }
                for (offset, &v) in lanes.iter().enumerate() {
                    if v >= floor && !hit(b * BLOCK_LANES + part * 8 + offset, v) {
                        return;
                    }
                }
            }
        }
    }
}

/// The run of pairs a row or a column last admitted that tie exactly, by what fixes the exact
/// correlation against the others of the same row or column: the estimate, the cross term and the
/// other window's variance. The run's `count` pairs arrived in ascending index order, `index`
/// being the last. A later pair from a larger index that ties the run, once `count` reaches the
/// `k` the row or column keeps, loses the tie to `k` pairs and need not be admitted; and when the
/// correlation is exactly one, no pair from a larger index can beat a run of `k`, whatever it is.
#[derive(Clone, Copy)]
pub(crate) struct Champion {
    index: u64,
    n: i64,
    variance: u64,
    rho_hat: f64,
    /// Saturates: a run is never taken for longer than it is.
    count: u32,
    perfect: bool,
}

impl Champion {
    /// Before any pair: ties and outranks nothing.
    pub(crate) const NONE: Self = Self {
        index: u64::MAX,
        n: 0,
        variance: 0,
        rho_hat: f64::NEG_INFINITY,
        count: 0,
        perfect: false,
    };

    /// The pair at `index` with cross term `n` against a window of this variance. A zero cross
    /// term fixes the correlation by the estimate alone, so the variance is dropped.
    #[inline(always)]
    pub(crate) fn of(index: u64, n: i64, variance: u64, rho_hat: f64) -> Self {
        Self {
            index,
            n,
            variance: if n == 0 { 0 } else { variance },
            rho_hat,
            count: 1,
            perfect: false,
        }
    }

    /// The run once `pair` is admitted: longer by it when it ties the run from a larger index,
    /// else just `pair`, with whether its correlation is exactly one: a pair of constants (`1.0`),
    /// or a cross term whose square is the product of the two windows' variances, `own` (read
    /// only when needed) and `other`. Only an estimate within the near-tie band of one, under the
    /// bound of the producer `P` that made it, can be.
    #[inline(always)]
    pub(crate) fn then<P: Producer>(
        self,
        pair: Self,
        own: impl FnOnce() -> u64,
        other: u64,
    ) -> Self {
        if self.ties(&pair) && self.index < pair.index {
            return Self {
                index: pair.index,
                count: self.count.saturating_add(1),
                ..self
            };
        }
        let perfect = pair.rho_hat >= band_floor(1.0, P::DELTA)
            && if pair.n == 0 {
                pair.rho_hat == 1.0
            } else {
                let cross = u128::from(pair.n.unsigned_abs());
                pair.n > 0 && cross * cross == crate::nearest::variance_product(own(), other)
            };
        Self { perfect, ..pair }
    }

    fn ties(&self, pair: &Self) -> bool {
        self.n == pair.n && self.variance == pair.variance && self.rho_hat == pair.rho_hat
    }

    /// Whether `k` pairs of the run beat any pair at `index`, whatever it is: they are exactly
    /// perfect, and from smaller indices.
    #[inline(always)]
    pub(crate) fn outranks_index(&self, index: u64, k: usize) -> bool {
        self.perfect && self.holds(k) && self.index < index
    }

    /// Whether `k` pairs of the run beat `pair`, which comes later from a larger index: it ties
    /// them, or they are perfect.
    #[inline(always)]
    pub(crate) fn beats(&self, pair: &Self, k: usize) -> bool {
        self.holds(k) && self.index < pair.index && (self.perfect || self.ties(pair))
    }

    /// Whether the run holds `k` perfect pairs.
    pub(crate) fn is_settled(&self, k: usize) -> bool {
        self.perfect && self.holds(k)
    }

    /// Whether the run has `k` pairs.
    #[inline(always)]
    fn holds(&self, k: usize) -> bool {
        self.count as usize >= k
    }
}

/// Puts `item` into `list` (best first under `better`, at most `k` long): appended and moved
/// forward while the list has fewer than `k`; once it has `k`, kept only when it beats the last,
/// which falls off. Returns whether `item` was kept. `better(a, b)` is `a` strictly before `b`.
#[inline(always)]
pub(crate) fn keep<T: Copy>(
    list: &mut Vec<T>,
    k: usize,
    item: T,
    better: impl Fn(&T, &T) -> bool,
) -> bool {
    if list.len() < k {
        list.push(item);
        settle(list, item, better);
        return true;
    }
    keep_in(list, item, better)
}

/// [`keep`] on fixed `slots` whose first `*len` hold the list: a short list grows into the next
/// slot, a full one (`*len` equal to the slots) keeps `item` only when it beats the last.
/// Returns whether `item` was kept.
#[inline(always)]
pub(crate) fn keep_within<T: Copy>(
    slots: &mut [T],
    len: &mut usize,
    item: T,
    better: impl Fn(&T, &T) -> bool,
) -> bool {
    if *len < slots.len() {
        *len += 1;
        settle(&mut slots[..*len], item, better);
        return true;
    }
    keep_in(slots, item, better)
}

/// Puts `item` into the bound `heap` of at most `k` items, kept with the worst at the root (no
/// parent is better than its children): pushed while there are fewer than `k`, else it replaces
/// the root when it beats it. Returns whether it was kept. The heap's order is not the best-first
/// order: it serves a list whose only read is its worst, its `k`-th best, in a logarithmic number
/// of moves whatever `k` is.
#[inline(always)]
pub(crate) fn heap_keep<T: Copy>(
    heap: &mut Vec<T>,
    k: usize,
    item: T,
    better: impl Fn(&T, &T) -> bool,
) -> bool {
    if heap.len() < k {
        heap.push(item);
        let mut at = heap.len() - 1;
        while at > 0 {
            let parent = (at - 1) / 2;
            if !better(&heap[parent], &heap[at]) {
                break;
            }
            heap.swap(parent, at);
            at = parent;
        }
        return true;
    }
    heap_keep_in(heap, item, better)
}

/// [`heap_keep`] on full fixed slots (a heap with the worst at the root): `item` replaces the root
/// when it beats it. Returns whether it was kept.
#[inline(always)]
pub(crate) fn heap_keep_in<T: Copy>(
    slots: &mut [T],
    item: T,
    better: impl Fn(&T, &T) -> bool,
) -> bool {
    match slots.first() {
        Some(root) if better(&item, root) => {
            slots[0] = item;
            sift_down(slots, &better);
            true
        }
        _ => false,
    }
}

/// Moves the root of `slots` down past every child worse than it.
#[inline]
fn sift_down<T: Copy>(slots: &mut [T], better: &impl Fn(&T, &T) -> bool) {
    let mut at = 0;
    loop {
        let mut worst = at;
        for child in [2 * at + 1, 2 * at + 2] {
            if child < slots.len() && better(&slots[worst], &slots[child]) {
                worst = child;
            }
        }
        if worst == at {
            return;
        }
        slots.swap(at, worst);
        at = worst;
    }
}

/// Leaves `items` as the `k` best of them, best first, as [`keep`] would have kept them offered
/// one by one: a selection and a sort, so the work grows with the items and not with `k` times
/// them. `better` must be a strict total order on the items.
pub(crate) fn select<T: Copy>(items: &mut Vec<T>, k: usize, better: impl Fn(&T, &T) -> bool) {
    let order = |a: &T, b: &T| {
        if better(a, b) {
            Ordering::Less
        } else if better(b, a) {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    };
    if items.len() > k {
        match k.checked_sub(1) {
            Some(last) => {
                items.select_nth_unstable_by(last, order);
            }
            None => items.clear(),
        }
        items.truncate(k);
    }
    items.sort_unstable_by(order);
}

/// [`keep`] with only the rejection inlined: for a caller whose loop is tight enough that an
/// inlined insertion costs every pair more than a call costs the rare insert.
#[inline(always)]
pub(crate) fn keep_rarely<T: Copy, F: Fn(&T, &T) -> bool>(
    list: &mut Vec<T>,
    k: usize,
    item: T,
    better: F,
) -> bool {
    #[inline(never)]
    fn insert<T: Copy, F: Fn(&T, &T) -> bool>(list: &mut Vec<T>, k: usize, item: T, better: F) {
        keep(list, k, item, better);
    }
    let enters = list.len() < k || list.last().is_some_and(|last| better(&item, last));
    if enters {
        insert(list, k, item, better);
    }
    enters
}

/// [`keep`] on a full list of fixed length (`slots`, best first): `item` replaces the last when it
/// beats it, then moves forward past every item it beats. Returns whether it was kept.
#[inline(always)]
pub(crate) fn keep_in<T: Copy>(slots: &mut [T], item: T, better: impl Fn(&T, &T) -> bool) -> bool {
    match slots.last() {
        Some(last) if better(&item, last) => {
            settle(slots, item, better);
            true
        }
        _ => false,
    }
}

/// Moves `item`, which takes the last of `slots` (non-empty), forward past every slot it beats.
#[inline(always)]
fn settle<T: Copy>(slots: &mut [T], item: T, better: impl Fn(&T, &T) -> bool) {
    let mut at = slots.len() - 1;
    while at > 0 && better(&item, &slots[at - 1]) {
        slots[at] = slots[at - 1];
        at -= 1;
    }
    slots[at] = item;
}

#[cfg(test)]
mod tests {
    use super::{Champion, keep, keep_in};
    use crate::estimate::F64;
    use crate::nearest::variance_product;
    use alloc::vec::Vec;
    use proptest::prelude::*;

    /// The `k` best of `items` by sorting: ascending value, ties to the earlier arrival.
    fn sorted_best(items: &[(u8, usize)], k: usize) -> Vec<(u8, usize)> {
        let mut all = items.to_vec();
        all.sort_by_key(|&(value, arrival)| (value, arrival));
        all.truncate(k);
        all
    }

    proptest! {
        #![proptest_config(testkit::generators::seeded(300))]

        /// Whatever the arrival order, the list is the `k` smallest values, ascending, equal
        /// values in arrival order: the order compares values only, so an equal later item
        /// never passes an earlier one, and never enters a full list whose last it only ties.
        #[test]
        fn keeps_the_k_best_with_ties_in_arrival_order(
            values in prop::collection::vec(0u8..8, 0..60),
            k in 0usize..12,
        ) {
            let items: Vec<(u8, usize)> = values.iter().copied().zip(0..).collect();
            let mut list = Vec::with_capacity(k);
            for &item in &items {
                let before = list.clone();
                let kept = keep(&mut list, k, item, |a: &(u8, usize), b| a.0 < b.0);
                prop_assert_eq!(kept, list != before);
            }
            prop_assert_eq!(list.capacity(), k);
            prop_assert_eq!(list, sorted_best(&items, k));
        }

        /// The fixed-length form is the full-list case of [`keep`]: slots start at the worst
        /// value and fill with the best.
        #[test]
        fn fixed_slots_keep_the_k_best(values in prop::collection::vec(0u8..8, 0..60), k in 0usize..12) {
            let items: Vec<(u8, usize)> = values.iter().copied().zip(0..).collect();
            let mut slots = alloc::vec![(u8::MAX, usize::MAX); k];
            for &item in &items {
                keep_in(&mut slots, item, |a: &(u8, usize), b| a.0 < b.0);
            }
            let mut expected = sorted_best(&items, k);
            expected.resize(k, (u8::MAX, usize::MAX));
            prop_assert_eq!(slots, expected);
        }

        /// The heap holds the same items as the list, and its root is the list's last.
        #[test]
        fn the_heap_holds_the_list_s_items_with_its_last_at_the_root(
            values in prop::collection::vec(0u8..8, 0..80),
            k in 1usize..20,
        ) {
            let items: Vec<(u8, usize)> = values.iter().copied().zip(0..).collect();
            let better = |a: &(u8, usize), b: &(u8, usize)| a.0 < b.0;
            let (mut list, mut heap) = (Vec::new(), Vec::new());
            let mut slots = alloc::vec![(u8::MAX, usize::MAX); k];
            for &item in &items {
                prop_assert_eq!(super::heap_keep(&mut heap, k, item, better), keep(&mut list, k, item, better));
                super::heap_keep_in(&mut slots, item, better);
                prop_assert_eq!(heap.first().map(|h| h.0), list.last().map(|l| l.0));
            }
            let values = |v: &[(u8, usize)]| {
                let mut v: Vec<u8> = v.iter().map(|x| x.0).collect();
                v.sort_unstable();
                v
            };
            prop_assert_eq!(values(&heap), values(&list));
            let mut padded = list.clone();
            padded.resize(k, (u8::MAX, usize::MAX));
            prop_assert_eq!(values(&slots), values(&padded));
        }

        /// Ranking everything at once keeps exactly what keeping one by one does, in order.
        #[test]
        fn selecting_at_once_equals_keeping_one_by_one(
            values in prop::collection::vec(0u8..8, 0..80),
            k in 0usize..20,
        ) {
            let items: Vec<(u8, usize)> = values.iter().copied().zip(0..).collect();
            let better = |a: &(u8, usize), b: &(u8, usize)| (a.0, a.1) < (b.0, b.1);
            let mut list = Vec::new();
            for &item in &items {
                keep(&mut list, k, item, better);
            }
            let mut all = items.clone();
            super::select(&mut all, k, better);
            prop_assert_eq!(all, list);
        }

        /// The form over fixed slots and a length keeps what [`keep`] keeps.
        #[test]
        fn slots_with_a_length_keep_the_k_best(
            values in prop::collection::vec(0u8..8, 0..60),
            k in 0usize..12,
        ) {
            let items: Vec<(u8, usize)> = values.iter().copied().zip(0..).collect();
            let mut slots = alloc::vec![(0, 0); k];
            let mut len = 0;
            for &item in &items {
                super::keep_within(&mut slots, &mut len, item, |a: &(u8, usize), b| a.0 < b.0);
            }
            let expected = sorted_best(&items, k);
            prop_assert_eq!(&slots[..len], expected.as_slice());
        }
    }

    #[test]
    fn an_item_tying_a_full_list_s_last_is_dropped() {
        let mut list = alloc::vec![1, 2];
        assert!(!keep(&mut list, 2, 2, |a, b| a < b));
        assert!(keep(&mut list, 2, 0, |a, b| a < b));
        assert_eq!(list, [0, 1]);
    }

    /// The row's variance, and for each kind of pair its cross term, the other window's variance
    /// and estimate: the correlations squared are 1, 1/2, 1/4 and 9/64 exactly, and the first is
    /// perfect.
    const OWN: u64 = 4;
    const KINDS: [(i64, u64, f64); 4] = [
        (6, 9, 1.0),
        (6, 18, core::f64::consts::FRAC_1_SQRT_2),
        (3, 9, 0.5),
        (3, 16, 0.375),
    ];

    /// Whether kind `a` is nearer than kind `b`, by the squared correlations cross-multiplied.
    fn nearer_kind(a: usize, b: usize) -> bool {
        let (na, va, _) = KINDS[a];
        let (nb, vb, _) = KINDS[b];
        let left = u128::from(na.unsigned_abs()).pow(2) * variance_product(OWN, vb);
        let right = u128::from(nb.unsigned_abs()).pow(2) * variance_product(OWN, va);
        left > right
    }

    /// `(index, kind)` pairs, the order the engines keep: nearer kind first, ties to the smaller
    /// index.
    fn before(a: &(usize, usize), b: &(usize, usize)) -> bool {
        nearer_kind(a.1, b.1) || (a.1 == b.1 && a.0 < b.0)
    }

    /// The pairs offered to a keeper, once through a champion and once straight.
    fn kept(chunks: &[Vec<usize>], k: usize, skipping: bool) -> Vec<(usize, usize)> {
        let mut list = Vec::new();
        let mut champion = Champion::NONE;
        let mut base = chunks.iter().map(Vec::len).sum::<usize>();
        for chunk in chunks {
            base -= chunk.len();
            for (offset, &kind) in chunk.iter().enumerate() {
                let (n, other, rho_hat) = KINDS[kind];
                let index = base + offset;
                let pair = Champion::of(index as u64, n, other, rho_hat);
                if skipping && champion.beats(&pair, k) {
                    continue;
                }
                champion = champion.then::<F64>(pair, || OWN, other);
                keep(&mut list, k, (index, kind), before);
            }
        }
        list
    }

    proptest! {
        #![proptest_config(testkit::generators::seeded(400))]

        /// Skipping a pair that `k` pairs before it tie, or that follow `k` perfect ones, keeps
        /// what offering everything keeps: in one ascending run, as a column sees its rows, and
        /// in runs that ascend but follow each other downwards, as a row sees its neighbors.
        #[test]
        fn skipping_pairs_that_lose_a_tie_keeps_the_k_best(
            chunks in prop::collection::vec(prop::collection::vec(0usize..4, 0..14), 1..5),
            k in 1usize..5,
        ) {
            prop_assert_eq!(kept(&chunks, k, true), kept(&chunks, k, false));
        }
    }

    #[test]
    fn a_run_of_k_perfect_pairs_outranks_every_later_index() {
        let (n, other, rho_hat) = KINDS[0];
        let mut champion = Champion::NONE;
        for index in 0..3 {
            assert!(!champion.outranks_index(index + 1, 3));
            champion = champion.then::<F64>(Champion::of(index, n, other, rho_hat), || OWN, other);
        }
        assert!(champion.is_settled(3) && !champion.is_settled(4));
        assert!(champion.outranks_index(3, 3));
        assert!(
            !champion.outranks_index(2, 3),
            "a smaller index is not outranked"
        );
        assert!(!champion.outranks_index(3, 4));
    }

    #[test]
    fn only_a_correlation_of_exactly_one_is_perfect() {
        for (kind, &(n, other, rho_hat)) in KINDS.iter().enumerate() {
            let champion =
                Champion::NONE.then::<F64>(Champion::of(0, n, other, rho_hat), || OWN, other);
            assert_eq!(champion.is_settled(1), kind == 0);
        }
        // A cross term one short of the variances' root has a squared correlation just under one.
        let champion = Champion::NONE.then::<F64>(Champion::of(0, 5, 9, 1.0), || OWN, 9);
        assert!(!champion.is_settled(1));
        // A pair of constants (a zero cross term at an estimate of one) is perfect; a single
        // constant (an estimate of one half) is not.
        assert!(
            Champion::NONE
                .then::<F64>(Champion::of(0, 0, 0, 1.0), || 0, 0)
                .is_settled(1)
        );
        assert!(
            !Champion::NONE
                .then::<F64>(Champion::of(0, 0, 9, 0.5), || 0, 9)
                .is_settled(1)
        );
    }

    /// A producer whose band reaches from one down to zero.
    struct Half;

    impl crate::estimate::Producer for Half {
        const DELTA: f64 = 0.5;
    }

    /// The perfect check's band is the producer's: an exactly perfect pair whose estimate is at
    /// the floor `1 − 2·δ` is perfect, one just below it is not, and the f64 producer's narrower
    /// band leaves both out.
    #[test]
    fn the_perfect_check_reads_the_producers_band() {
        let (n, other, _) = KINDS[0];
        let settled = |rho_hat: f64, half: bool| {
            let pair = Champion::of(0, n, other, rho_hat);
            let champion = if half {
                Champion::NONE.then::<Half>(pair, || OWN, other)
            } else {
                Champion::NONE.then::<F64>(pair, || OWN, other)
            };
            champion.is_settled(1)
        };
        assert!(settled(0.0, true));
        assert!(!settled((-0.0f64).next_down(), true));
        assert!(!settled(0.0, false));
        assert!(settled(1.0, false));
    }

    #[test]
    fn a_zero_cross_term_ties_whatever_the_variances() {
        let first = Champion::of(1, 0, 9, 0.0);
        let later = Champion::of(2, 0, 16, 0.0);
        assert!(first.beats(&later, 1));
        assert!(!later.beats(&first, 1), "the smaller index is not beaten");
        assert!(
            !first.beats(&Champion::of(2, 0, 16, 0.5), 1),
            "another estimate is another tie"
        );
        assert!(
            !first.beats(&Champion::of(2, 3, 9, 0.0), 1),
            "another cross term too"
        );
    }
}
