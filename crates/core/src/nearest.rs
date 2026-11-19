//! The per-row reduction from each neighbor's exact centered cross term to the row's exact
//! nearest-neighbor [`Entry`]: `ρ̂` ranks every candidate in O(1), falling back to an exact
//! comparison only on a near-tie, so the winner matches `naive`'s independently computed one
//! bit for bit.

use core::cmp::Ordering;
use core::num::NonZeroUsize;

use alloc::vec::Vec;

use crate::entry::Entry;
use crate::estimate::{F64, Producer, band_floor, rank_estimate};
use crate::index::SampleIndex;
use crate::kbest;
use crate::kernel::Reducer;
use crate::stats::{Window, WindowKind};
use crate::wide::Wide;

/// A candidate's exact correlation, independent of floating-point rounding: `sign(N)` and
/// `N²` for a valid pair (against the row's own `Vi` and this neighbor's `Vj`), or the fixed
/// ratio for a constant pair (always positive): both constant is `1/1`, one constant `1/4`.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ExactCorrelation {
    BothConstant,
    OneConstant,
    Valid {
        sign: i8,
        cross_squared: u128,
        row_variance: u64,
        neighbor_variance: u64,
        /// `Vi · Vj`, precomputed once (checked, never saturating: a silently capped value
        /// here would be a wrong comparison, not merely a slow one).
        denominator: u128,
    },
}

/// `Vi · Vj`, exact in u128 (both factors are below 2⁶³, so the product never overflows).
pub(crate) fn variance_product(row_variance: u64, neighbor_variance: u64) -> u128 {
    u128::from(row_variance) * u128::from(neighbor_variance)
}

/// `a · b` against `c · d`, exact, where every factor is below `2^126` (so the products
/// themselves can exceed u128).
pub(crate) fn cmp_wide_products(a: u128, b: u128, c: u128, d: u128) -> Ordering {
    Wide::product(a, b).cmp(&Wide::product(c, d))
}

/// `a · b` against `c · d`, exact: a factor the two products have in common cancels, so two
/// correlations with equal cross terms or equal variances are ordered by one u128 comparison and
/// only an unrelated pair needs the 256-bit products. Zero factors make both sides zero.
#[inline(always)]
fn cmp_cross_products(a: u128, b: u128, c: u128, d: u128) -> Ordering {
    if a == 0 || b == 0 || c == 0 || d == 0 {
        return cmp_wide_products(a, b, c, d);
    }
    if a == c {
        return b.cmp(&d);
    }
    if b == d {
        return a.cmp(&c);
    }
    if (a | b | c | d) >> 64 == 0 {
        return (a * b).cmp(&(c * d));
    }
    cmp_wide_products(a, b, c, d)
}

/// `a/b` against `c/d` as plain u128 products. The caller never reaches this with two
/// valid-valid ratios (that pair cancels `Vi` and goes through the wide product instead), so
/// one side here is always a constant ratio with numerator and denominator at most `4`; the
/// other, at most `2^126`, times that small constant stays below `2^128` and never needs
/// widening.
fn cmp_ratio(a: u128, b: u128, c: u128, d: u128) -> Ordering {
    let overflow = "a constant ratio's factor of at most 4 keeps this product within u128";
    let left = a.checked_mul(d).expect(overflow);
    let right = c.checked_mul(b).expect(overflow);
    left.cmp(&right)
}

/// What two compared correlations have in common, so that variance cancels from `ρ²`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Sharing {
    /// The same row: candidates of one row, sharing `Vi`.
    Row,
    /// The same neighbor: two rows against one query, sharing `Vj`.
    Neighbor,
    /// Different rows and neighbors: the full cross product with both variance products.
    Nothing,
}

/// Which of two same-sign, nonzero correlations has the larger `ρ²` (so is nearer). Two
/// valid pairs cancel whatever variance they share (or cross-multiply the whole denominators
/// when they share none); anything touching a constant pair's fixed ratio falls back to a
/// plain cross product instead.
fn cmp_magnitude(a: ExactCorrelation, b: ExactCorrelation, sharing: Sharing) -> Ordering {
    use ExactCorrelation::{BothConstant, OneConstant, Valid};
    if let (
        Valid {
            cross_squared: na,
            row_variance: via,
            neighbor_variance: vja,
            denominator: den_a,
            ..
        },
        Valid {
            cross_squared: nb,
            row_variance: vib,
            neighbor_variance: vjb,
            denominator: den_b,
            ..
        },
    ) = (a, b)
    {
        return match sharing {
            Sharing::Row => cmp_cross_products(na, vjb.into(), nb, vja.into()),
            Sharing::Neighbor => cmp_cross_products(na, vib.into(), nb, via.into()),
            Sharing::Nothing => cmp_cross_products(na, den_b, nb, den_a),
        };
    }
    let ratio = |correlation: ExactCorrelation| -> (u128, u128) {
        match correlation {
            BothConstant => (1, 1),
            OneConstant => (1, 4),
            Valid {
                cross_squared,
                denominator,
                ..
            } => (cross_squared, denominator),
        }
    };
    let (num_a, den_a) = ratio(a);
    let (num_b, den_b) = ratio(b);
    cmp_ratio(num_a, den_a, num_b, den_b)
}

fn sign(correlation: ExactCorrelation) -> i8 {
    match correlation {
        ExactCorrelation::BothConstant | ExactCorrelation::OneConstant => 1,
        ExactCorrelation::Valid { sign, .. } => sign,
    }
}

/// The exact nearest-first order of two correlations: by sign, then by `ρ²` — reversed for a
/// negative sign, since a larger magnitude there is a more negative, so farther, `ρ`.
/// `Greater` means `a` is nearer. `sharing` says which variance the two have in common; the
/// answer is the same for any true choice, only the cost differs.
pub(crate) fn cmp_exact(a: ExactCorrelation, b: ExactCorrelation, sharing: Sharing) -> Ordering {
    let sign_a = sign(a);
    let sign_b = sign(b);
    if sign_a != sign_b {
        return sign_a.cmp(&sign_b);
    }
    if sign_a == 0 {
        return Ordering::Equal;
    }
    let magnitude_order = cmp_magnitude(a, b, sharing);
    if sign_a > 0 {
        magnitude_order
    } else {
        magnitude_order.reverse()
    }
}

/// Lanes per chunk-skip group in [`rescan_near_tie`]: wide enough to amortize the branchless
/// reduction's own cost over many lanes, narrow enough that one hit still only rescans 64 of
/// them, not the whole row.
const RESCAN_CHUNK: usize = 64;

/// A near-tie band's own scan, chunk-skipped: `scratch[j] ≥ threshold` is rare (the near-tie
/// band is usually one lane wide, `row::advance_and_rank`'s own threshold comment proves why),
/// so a plain element-by-element scan spends almost every lane on a branch that never taken.
/// Each `RESCAN_CHUNK`-wide group is first reduced with one branchless fold — `scratch` is
/// never NaN, so `|`-ing a plain `≥` across the chunk can't silently drop a lane the way a
/// NaN-aware comparison could — and only a chunk that fold flags is scanned lane by lane.
/// `on_hit(j)` runs for every `j` whose `scratch[j]` clears `threshold`, ascending, the same set
/// in the same order a plain scan over the whole slice would produce: `chunks()` sizes the last
/// group to whatever is left, so the tail is scanned by the exact same code as every full
/// chunk, not a separate (and separately bug-prone) special case.
#[inline]
pub(crate) fn rescan_near_tie(scratch: &[f64], threshold: f64, mut on_hit: impl FnMut(usize)) {
    for (chunk_index, chunk) in scratch.chunks(RESCAN_CHUNK).enumerate() {
        let any_hit = chunk
            .iter()
            .fold(false, |hit, &rho_hat| hit | (rho_hat >= threshold));
        if any_hit {
            let base = chunk_index * RESCAN_CHUNK;
            for (offset, &rho_hat) in chunk.iter().enumerate() {
                if rho_hat >= threshold {
                    on_hit(base + offset);
                }
            }
        }
    }
}

/// What a correlation must offer the shared ranking: the exact order of two candidates of one
/// row, and the pieces of the winner's reported distance. The hot passes' [`ExactCorrelation`]
/// and the cold route's own ([`crate::spill`]) both implement it, so one [`nearer`] and one
/// [`winner_entry`] serve both.
pub(crate) trait Ranked: Copy {
    /// `Greater` when `self` is the exact nearer of two candidates sharing a row.
    fn cmp_in_row(self, other: Self) -> Ordering;

    /// `Some` for a valid-valid pair: the reported distance's numerator and both reciprocal roots.
    fn distance_parts(self) -> Option<DistanceParts>;
}

/// `Vi·Vj − N²` as the nearest `f64`, with `1/√Vi` and `1/√Vj`.
pub(crate) struct DistanceParts {
    pub(crate) numerator: f64,
    pub(crate) row_inv: f64,
    pub(crate) neighbor_inv: f64,
}

impl Ranked for ExactCorrelation {
    fn cmp_in_row(self, other: Self) -> Ordering {
        cmp_exact(self, other, Sharing::Row)
    }

    fn distance_parts(self) -> Option<DistanceParts> {
        match self {
            Self::Valid {
                cross_squared,
                row_variance,
                neighbor_variance,
                denominator,
                ..
            } => {
                let distance_numerator = denominator
                    .checked_sub(cross_squared)
                    .expect("cross_squared is at most denominator (Cauchy-Schwarz)");
                Some(DistanceParts {
                    numerator: distance_numerator as f64,
                    row_inv: 1.0 / libm::sqrt(row_variance as f64),
                    neighbor_inv: 1.0 / libm::sqrt(neighbor_variance as f64),
                })
            }
            Self::BothConstant | Self::OneConstant => None,
        }
    }
}

/// One row's candidate neighbor: the ranking `ρ̂` (fast, approximate) and the exact
/// correlation it approximates (used only to settle a near-tie or a true tie).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Candidate<C = ExactCorrelation> {
    pub(crate) index: u64,
    pub(crate) rho_hat: f64,
    pub(crate) correlation: C,
}

/// Whether estimate `a` already beats estimate `b`: `b` falls below `a`'s band floor, the band
/// within which only the exact correlations decide. Bound to [`F64`]: it only compares estimates
/// that [`rank_estimate`] made from the exact cross term (as [`Nearest`] recomputes them), never
/// a filter's own.
#[inline(always)]
pub(crate) fn clearly_nearer(a: f64, b: f64) -> bool {
    b < band_floor(a, F64::DELTA)
}

/// `true` if `a` is the exact nearest of the two: `ρ̂` alone decides outside `2ε` of each
/// other; within that band, or exactly on it, the exact correlation decides, ties going to
/// the smaller index.
// Always inlined into the offer loops, so the generic form costs no more than the plain one did.
#[inline(always)]
pub(crate) fn nearer<C: Ranked>(a: Candidate<C>, b: Candidate<C>) -> bool {
    if clearly_nearer(a.rho_hat, b.rho_hat) {
        return true;
    }
    if clearly_nearer(b.rho_hat, a.rho_hat) {
        return false;
    }
    match a.correlation.cmp_in_row(b.correlation) {
        Ordering::Greater => true,
        Ordering::Less => false,
        Ordering::Equal => a.index < b.index,
    }
}

/// The reported distance: for a valid-valid winner with positive ρ, `Vi·Vj − N²`
/// recomputed exactly and divided by `1 + ρ̂`; otherwise the ranking formula, which already
/// gives `0` for a double constant match and `m` for a single one.
pub(crate) fn winner_entry<C: Ranked>(m: usize, winner: Candidate<C>) -> Entry {
    let rho = winner.rho_hat;
    let parts = if rho > 0.0 {
        winner.correlation.distance_parts()
    } else {
        None
    };
    let squared_distance = match parts {
        Some(parts) => {
            let inverse_product = parts.row_inv * parts.neighbor_inv;
            (2.0 * m as f64) * parts.numerator * inverse_product * inverse_product / (1.0 + rho)
        }
        None => (2.0 * m as f64) * (1.0 - rho),
    };
    let index = SampleIndex::new(winner.index).expect("neighbor index is below NONE's sentinel");
    Entry::new(rho, squared_distance, index).expect("rho and squared_distance are never NaN")
}

/// One window's distance to its nearest neighbor, carrying the exact correlation it was
/// ranked by.
///
/// The order is the exact order of distances across windows: the nearer window sorts first,
/// and exactly equal distances (such as a window and a scaled copy of it) compare `Equal`, for
/// the caller to break. Two entries far apart in `ρ̂` are ordered by it alone; within the band
/// where rounding in either estimate could reverse them, the exact correlations decide, so the
/// result never depends on the rounding of each window's own reciprocal root. Any two entries
/// compare, whether or not they share a row or a neighbor.
#[derive(Clone, Copy, Debug)]
pub struct ExactDistance {
    entry: Entry,
    correlation: Held,
    window: usize,
}

/// The exact correlation an [`ExactDistance`] ranks by: as the hot passes hold it, or as the
/// cold route holds a pair with a variance numerator past `2⁶³`.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Held {
    Hot(ExactCorrelation),
    Cold(crate::spill::WideCorrelation),
}

impl ExactDistance {
    /// The reported entry.
    pub fn entry(&self) -> Entry {
        self.entry
    }

    /// The window length the distance is of.
    pub(crate) fn window(&self) -> usize {
        self.window
    }

    /// The exact correlation the distance ranks by.
    pub(crate) fn correlation(&self) -> Held {
        self.correlation
    }

    /// An entry with whatever correlation a test or the reference wants it ranked by, for
    /// windows of length `window`.
    #[cfg(any(test, feature = "oracle"))]
    pub(crate) fn from_parts(entry: Entry, correlation: ExactCorrelation, window: usize) -> Self {
        Self::held(entry, Held::Hot(correlation), window)
    }

    /// The reported distance of `winner`, for windows of length `m`.
    pub(crate) fn of_winner(m: usize, winner: Candidate) -> Self {
        Self::held(winner_entry(m, winner), Held::Hot(winner.correlation), m)
    }

    /// `entry` ranked by `correlation`, for windows of length `window`.
    pub(crate) fn held(entry: Entry, correlation: Held, window: usize) -> Self {
        Self {
            entry,
            correlation,
            window,
        }
    }

    /// The exact order given what the two entries share; [`Ord::cmp`] is
    /// [`Sharing::Nothing`], which is right for any pair. The estimates decide outside the band
    /// of [`F64`], which made every entry's.
    pub(crate) fn cmp_sharing(&self, other: &Self, sharing: Sharing) -> Ordering {
        let (a, b) = (self.entry.rho(), other.entry.rho());
        if b < band_floor(a, F64::DELTA) {
            return Ordering::Less;
        }
        if a < band_floor(b, F64::DELTA) {
            return Ordering::Greater;
        }
        match (other.correlation, self.correlation) {
            (Held::Hot(a), Held::Hot(b)) => cmp_exact(a, b, sharing),
            (a, b) => crate::spill::cmp_held(a, b),
        }
    }
}

impl Ord for ExactDistance {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cmp_sharing(other, Sharing::Nothing)
    }
}

impl PartialOrd for ExactDistance {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for ExactDistance {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for ExactDistance {}

/// The row's own validity, as `offer` needs it: a non-finite row ignores every candidate;
/// otherwise a candidate's `ρ̂` depends on whether this row is constant, and if not, on its
/// variance and precomputed reciprocal square root, the `r_i` the ranking multiplies by.
#[derive(Clone, Copy)]
enum RowKind {
    NonFinite,
    Constant,
    Valid { inv_sqrt: f64, variance: u64 },
}

fn row_kind(window: &Window) -> RowKind {
    match window.kind {
        WindowKind::NonFinite => RowKind::NonFinite,
        WindowKind::Constant => RowKind::Constant,
        WindowKind::Valid {
            variance_numerator,
            inv_sqrt,
        } => RowKind::Valid {
            inv_sqrt,
            variance: variance_numerator,
        },
    }
}

/// What `Nearest` keeps: either the single best (exactly [`Self::new`]'s own representation,
/// so `k = 1` costs what it always has), or, for `k > 1`, the `k` nearest so far, nearest
/// first, in a `Vec` reserved to `min(k, bound)` (up to a fixed limit) once in [`Nearest::with_k`].
enum Kept {
    One(Option<Candidate>),
    Many(Vec<Candidate>, NonZeroUsize),
}

/// Inserts `candidate` into `heap` (nearest first under [`nearer`], length at most `k`) through
/// the one k-best keeper: after every kept candidate it does not beat, the farthest falling off a
/// full list, and nothing kept when it beats none of a full list. Never reallocates once `heap`
/// holds room for `k`.
pub(crate) fn insert_topk<C: Ranked>(
    heap: &mut Vec<Candidate<C>>,
    k: usize,
    candidate: Candidate<C>,
) {
    kbest::keep(heap, k, candidate, |a, b| nearer(*a, *b));
}

/// The candidate a row of kind `row` ranks neighbor `index` by, from the pair's exact centered
/// cross term `n`; `None` when either window is non-finite.
fn ranked(row: RowKind, n: i64, neighbor: &Window, index: SampleIndex) -> Option<Candidate> {
    let (rho_hat, correlation) = match (row, neighbor.kind) {
        (RowKind::NonFinite, _) | (_, WindowKind::NonFinite) => return None,
        (RowKind::Constant, WindowKind::Constant) => (1.0, ExactCorrelation::BothConstant),
        (RowKind::Constant, WindowKind::Valid { .. })
        | (RowKind::Valid { .. }, WindowKind::Constant) => (0.5, ExactCorrelation::OneConstant),
        (
            RowKind::Valid {
                inv_sqrt: row_inv,
                variance: row_variance,
            },
            WindowKind::Valid {
                variance_numerator: neighbor_variance,
                inv_sqrt: neighbor_inv,
            },
        ) => {
            let rho_hat = rank_estimate(n, row_inv, neighbor_inv).clamp(-1.0, 1.0);
            let magnitude = u128::from(n.unsigned_abs());
            let correlation = ExactCorrelation::Valid {
                sign: n.signum() as i8,
                cross_squared: magnitude * magnitude,
                row_variance,
                neighbor_variance,
                denominator: variance_product(row_variance, neighbor_variance),
            };
            (rho_hat, correlation)
        }
    };
    Some(Candidate {
        index: index.get().expect("a candidate's index is never NONE"),
        rho_hat,
        correlation,
    })
}

/// The candidate window `row` ranks `neighbor` (window `index`) by, from the pair's exact centered
/// cross term `n`; `None` when either window is non-finite. What a column's side of the pass
/// ranks a later window by, with the column as the row.
pub(crate) fn candidate_of(
    row: &Window,
    n: i64,
    neighbor: &Window,
    index: SampleIndex,
) -> Option<Candidate> {
    ranked(row_kind(row), n, neighbor, index)
}

/// The `k` above which [`Nearest::select`] ranks all its offers at once instead of keeping `k`
/// as they come.
pub(crate) const SELECT_ALL_ABOVE: usize = 64;

/// One row's reduction over its eligible neighbors, offered in any order: ranks each by `ρ̂`
/// in O(1), falls back to the exact comparison only on a near-tie, and computes the reported
/// distance once, in [`Self::finish`] or [`Self::finish_k`], for the kept winners alone.
#[doc(hidden)]
pub struct Nearest {
    row: RowKind,
    m: usize,
    kept: Kept,
}

impl Nearest {
    /// A non-finite row never produces or receives a candidate: every `offer` after this is a
    /// no-op, and `finish` is always [`Entry::NONE`]. Equivalent to `with_k(row, m, 1)`.
    pub fn new(row: &Window, m: usize) -> Self {
        Self::with_k(row, m, NonZeroUsize::MIN, 1)
    }

    /// Keeps the `k` exact nearest candidates ever offered, nearest first under the exact
    /// order (ties to the smaller index). `k = 1` allocates nothing beyond what `new` always
    /// has; `k > 1` reserves `min(k, bound)` candidates here, once, `bound` being the most
    /// candidates the row's input can offer, so `offer` never allocates (past a fixed limit the
    /// reservation stops there and `offer` grows it).
    pub fn with_k(row: &Window, m: usize, k: NonZeroUsize, bound: usize) -> Self {
        let kept = if k.get() == 1 {
            Kept::One(None)
        } else {
            Kept::Many(
                Vec::with_capacity(crate::capacity::reserve_for(k.get(), bound)),
                k,
            )
        };
        Self {
            row: row_kind(row),
            m,
            kept,
        }
    }

    /// Starts a new row on this reducer: forgets every kept candidate, keeping their storage, and
    /// keeps the `k` nearest of at most `bound` neighbors from here on. Equivalent to
    /// [`Self::with_k`] on a fresh one, except that a repeat row reserves nothing it already holds,
    /// so a stream that keeps one reducer allocates only while its counts are still growing.
    pub fn restart(&mut self, row: &Window, m: usize, k: NonZeroUsize, bound: usize) {
        self.row = row_kind(row);
        self.m = m;
        if k.get() == 1 {
            self.kept = Kept::One(None);
            return;
        }
        let wanted = crate::capacity::reserve_for(k.get(), bound);
        if let Kept::Many(heap, kept_k) = &mut self.kept {
            heap.clear();
            *kept_k = k;
            heap.reserve(wanted);
        } else {
            self.kept = Kept::Many(Vec::with_capacity(wanted), k);
        }
    }

    /// `n` is the pair's exact centered cross term. Ignored if this row is non-finite or
    /// `neighbor` is.
    pub fn offer(&mut self, n: i64, neighbor: &Window, index: SampleIndex) {
        let Some(candidate) = ranked(self.row, n, neighbor, index) else {
            return;
        };
        match &mut self.kept {
            Kept::One(best) => {
                if best.is_none_or(|current| nearer(candidate, current)) {
                    *best = Some(candidate);
                }
            }
            Kept::Many(heap, k) => insert_topk(heap, k.get(), candidate),
        }
    }

    /// Offers every one of `offers`, as [`Self::offer`] takes each, with the same result. Up to
    /// [`SELECT_ALL_ABOVE`] they are offered one by one; past it they join the kept ones and all are
    /// ranked at once, so the work grows with the offers and not with `k` times them. That needs
    /// room for the kept ones and the offers together; past what `k` reserves, it grows.
    pub(crate) fn select(&mut self, offers: impl Iterator<Item = (i64, Window, SampleIndex)>) {
        let row = self.row;
        match &mut self.kept {
            Kept::Many(heap, k) if k.get() > SELECT_ALL_ABOVE => {
                heap.extend(
                    offers.filter_map(|(n, neighbor, index)| ranked(row, n, &neighbor, index)),
                );
                kbest::select(heap, k.get(), |a, b| nearer(*a, *b));
            }
            _ => {
                for (n, neighbor, index) in offers {
                    self.offer(n, &neighbor, index);
                }
            }
        }
    }

    /// Reserves room for [`Self::select`] to rank `offers` at once without allocating.
    pub(crate) fn reserve_selection(&mut self, offers: usize) {
        if let Kept::Many(heap, k) = &mut self.kept
            && k.get() > SELECT_ALL_ABOVE
        {
            heap.reserve(offers.saturating_sub(heap.len()));
        }
    }

    /// The estimate below which a further candidate cannot enter the kept set: the band floor of
    /// the worst kept estimate once `k` are kept, `-inf` before, under [`F64`], which made every
    /// kept estimate. A candidate whose estimate is under it is clearly farther than every kept
    /// one, so offering it changes nothing.
    pub(crate) fn admission_floor(&self) -> f64 {
        match &self.kept {
            Kept::One(best) => {
                best.map_or(f64::NEG_INFINITY, |c| band_floor(c.rho_hat, F64::DELTA))
            }
            Kept::Many(heap, k) if heap.len() == k.get() => heap
                .last()
                .map_or(f64::NEG_INFINITY, |c| band_floor(c.rho_hat, F64::DELTA)),
            Kept::Many(..) => f64::NEG_INFINITY,
        }
    }

    /// A reduction that only selects: [`Self::kept_indices`] names the `k` exact nearest, and
    /// the distances are never reported, so no window length is needed.
    pub(crate) fn selecting(row: &Window, k: NonZeroUsize, bound: usize) -> Self {
        Self::with_k(row, 0, k, bound)
    }

    /// The exact nearest candidates offered so far, nearest first.
    pub(crate) fn kept(&self) -> impl Iterator<Item = Candidate> + '_ {
        let (one, many): (_, &[Candidate]) = match &self.kept {
            Kept::One(best) => (*best, &[]),
            Kept::Many(heap, _) => (None, heap),
        };
        one.into_iter().chain(many.iter().copied())
    }

    /// The indices of the exact nearest candidates offered so far, nearest first.
    pub(crate) fn kept_indices(&self) -> impl Iterator<Item = u64> + '_ {
        let (one, many): (_, &[Candidate]) = match &self.kept {
            Kept::One(best) => (*best, &[]),
            Kept::Many(heap, _) => (None, heap),
        };
        one.into_iter().chain(many.iter().copied()).map(|c| c.index)
    }

    /// The reported distance for the winner, bit for bit; [`Entry::NONE`] if the row had no
    /// eligible neighbor. Equals [`Self::finish_k`]'s first entry, or `Entry::NONE` if empty.
    pub fn finish(self) -> Entry {
        let m = self.m;
        match self.kept {
            Kept::One(best) => best.map_or(Entry::NONE, |winner| winner_entry(m, winner)),
            Kept::Many(heap, _) => heap
                .into_iter()
                .next()
                .map_or(Entry::NONE, |winner| winner_entry(m, winner)),
        }
    }

    /// [`Self::finish`] as an [`ExactDistance`], `None` if there was no winner.
    pub(crate) fn finish_exact(self) -> Option<ExactDistance> {
        let m = self.m;
        self.finish_winner()
            .map(|winner| ExactDistance::of_winner(m, winner))
    }

    /// The winner itself, `None` if there was none: [`Self::finish_exact`] before it is reported.
    pub(crate) fn finish_winner(self) -> Option<Candidate> {
        match self.kept {
            Kept::One(best) => best,
            Kept::Many(heap, _) => heap.into_iter().next(),
        }
    }

    /// Appends `row`'s kept neighbors, nearest first: one `(row, entry)` per neighbor, or one
    /// [`Entry::NONE`] when it has none. With `k` of one it is [`Self::finish`] and allocates
    /// nothing.
    pub(crate) fn push_kept(self, row: u64, k: NonZeroUsize, results: &mut Vec<(u64, Entry)>) {
        if k == NonZeroUsize::MIN {
            results.push((row, self.finish()));
            return;
        }
        let before = results.len();
        results.extend(self.finish_k().into_iter().map(|e| (row, e)));
        if results.len() == before {
            results.push((row, Entry::NONE));
        }
    }

    /// [`Self::finish_k`] into `out` (cleared first), leaving this reducer empty for the next
    /// [`Self::restart`]; allocates nothing when `out` has room. Returns the nearest neighbor with
    /// the exact correlation it was ranked by, `None` if there is none.
    pub fn drain_k(&mut self, out: &mut Vec<Entry>) -> Option<ExactDistance> {
        out.clear();
        let m = self.m;
        let mut nearest = None;
        let mut push = |winner: Candidate| {
            let entry = winner_entry(m, winner);
            nearest.get_or_insert(ExactDistance {
                entry,
                correlation: Held::Hot(winner.correlation),
                window: m,
            });
            out.push(entry);
        };
        match &mut self.kept {
            Kept::One(best) => best.take().into_iter().for_each(&mut push),
            Kept::Many(heap, _) => heap.drain(..).for_each(&mut push),
        }
        nearest
    }

    /// At most `k` entries (fewer if fewer candidates were eligible), nearest first under
    /// the exact order, ties to the smaller index; each entry's bits equal what `finish`
    /// reports when that neighbor is the only one offered.
    pub fn finish_k(self) -> Vec<Entry> {
        let m = self.m;
        match self.kept {
            Kept::One(best) => best
                .into_iter()
                .map(|winner| winner_entry(m, winner))
                .collect(),
            Kept::Many(heap, _) => heap
                .into_iter()
                .map(|winner| winner_entry(m, winner))
                .collect(),
        }
    }
}

/// `Nearest` as the kernel's [`Reducer`]: its candidate is the exact pair `offer` already
/// takes (never a ranked [`Entry`]), so this delegates straight to the inherent
/// methods above.
impl Reducer for Nearest {
    type Candidate = (i64, Window, SampleIndex);
    type Output = Entry;

    fn offer(&mut self, (n, neighbor, index): Self::Candidate) {
        self.offer(n, &neighbor, index);
    }

    fn finish(self) -> Entry {
        self.finish()
    }
}

/// Test-only exact comparison by plain signed rationals, shared by the tests that check an
/// order against the production comparator.
#[cfg(test)]
pub(crate) mod exact_oracle {
    /// A window's exact correlation as a signed rational `±num/den` of its square, from direct
    /// i128 sums over ticks (no shared code with the production comparison).
    #[derive(Clone, Copy)]
    pub(crate) struct Exact {
        pub(crate) sign: i8,
        pub(crate) num: u128,
        pub(crate) den: u128,
    }

    /// 256-bit product of two u128, as (high, low).
    fn wide(a: u128, b: u128) -> (u128, u128) {
        let (a0, a1) = (a & u128::from(u64::MAX), a >> 64);
        let (b0, b1) = (b & u128::from(u64::MAX), b >> 64);
        let (p00, p01, p10, p11) = (a0 * b0, a0 * b1, a1 * b0, a1 * b1);
        let mid = (p00 >> 64) + (p01 & u128::from(u64::MAX)) + (p10 & u128::from(u64::MAX));
        let low = (p00 & u128::from(u64::MAX)) | (mid << 64);
        let high = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
        (high, low)
    }

    /// `true` when `a` is strictly nearer than `b`: larger correlation, by cross-multiplying
    /// the squared ratios exactly.
    pub(crate) fn exactly_nearer(a: Exact, b: Exact) -> bool {
        if a.sign != b.sign {
            return a.sign > b.sign;
        }
        let (left, right) = (wide(a.num, b.den), wide(b.num, a.den));
        match a.sign {
            0 => false,
            1 => left > right,
            _ => left < right,
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    #[test]
    fn an_estimate_on_the_floor_is_not_clearly_beaten() {
        use super::clearly_nearer;
        let floor = 1.0 - 2.0 * EPSILON;
        assert!(!clearly_nearer(1.0, floor));
        assert!(clearly_nearer(1.0, floor - EPSILON / 4.0));
        assert!(!clearly_nearer(floor, 1.0));
    }

    use super::{
        Candidate, ExactCorrelation, ExactDistance, Nearest, RESCAN_CHUNK, RowKind, Sharing,
        cmp_cross_products, cmp_exact, cmp_wide_products, nearer, rescan_near_tie, row_kind,
        variance_product,
    };
    use crate::entry::Entry;
    use crate::estimate::EPSILON;
    use crate::index::SampleIndex;
    use crate::stats::{Window, WindowKind};
    use core::cmp::Ordering;
    use core::num::NonZeroUsize;
    use proptest::prelude::*;
    use proptest::strategy::ValueTree;
    use proptest::test_runner::TestRunner;
    use std::vec::Vec;

    /// An `ExactCorrelation::Valid` with the given sign and `ρ² = cross_squared / (row_variance
    /// · neighbor_variance)`, for pinning the comparator's behavior directly without a real
    /// window or offer.
    fn valid_correlation(
        sign: i8,
        cross_squared: u128,
        row_variance: u64,
        neighbor_variance: u64,
    ) -> ExactCorrelation {
        ExactCorrelation::Valid {
            sign,
            cross_squared,
            row_variance,
            neighbor_variance,
            denominator: variance_product(row_variance, neighbor_variance),
        }
    }

    /// A window built straight from raw ticks, independent of `stats::WindowStats`: the same
    /// classification `stats::classify` uses (missing beats flat beats out-of-range), but
    /// computed here from a plain slice rather than a rolling ring.
    fn window_of(ticks: &[i64]) -> Window {
        let m = ticks.len() as i128;
        let sum: i128 = ticks.iter().map(|&t| i128::from(t)).sum();
        let sum_sq: i128 = ticks.iter().map(|&t| i128::from(t) * i128::from(t)).sum();
        let min = *ticks.iter().min().expect("m > 0");
        let max = *ticks.iter().max().expect("m > 0");
        let kind = if max == min {
            WindowKind::Constant
        } else {
            let variance_numerator = m * sum_sq - sum * sum;
            if (1..1i128 << 63).contains(&variance_numerator) {
                let variance_numerator = variance_numerator as u64;
                WindowKind::Valid {
                    variance_numerator,
                    inv_sqrt: 1.0 / libm::sqrt(variance_numerator as f64),
                }
            } else {
                WindowKind::NonFinite
            }
        };
        Window {
            sum: sum as i64,
            kind,
        }
    }

    fn non_finite_window() -> Window {
        Window {
            sum: 0,
            kind: WindowKind::NonFinite,
        }
    }

    /// `N = m·Σxᵢxⱼ − Sᵢ·Sⱼ`, straight from the definitions in i128: no wrapping, no
    /// recurrence, nothing shared with `znorm`'s `centered_cross` or the kernel it feeds.
    fn exact_n(row: &[i64], neighbor: &[i64]) -> i64 {
        let m = row.len() as i128;
        let qt: i128 = row
            .iter()
            .zip(neighbor)
            .map(|(&a, &b)| i128::from(a) * i128::from(b))
            .sum();
        let sum_row: i128 = row.iter().map(|&t| i128::from(t)).sum();
        let sum_neighbor: i128 = neighbor.iter().map(|&t| i128::from(t)).sum();
        (m * qt - sum_row * sum_neighbor) as i64
    }

    fn index(i: u64) -> SampleIndex {
        SampleIndex::new(i).expect("test indices are below NONE's sentinel")
    }

    fn bits(entry: crate::entry::Entry) -> (u64, u64, u64) {
        (
            entry.rho().to_bits(),
            entry.squared_distance().to_bits(),
            entry.index().get().unwrap_or(u64::MAX),
        )
    }

    // ---------- literal bits, cross-checked against naive's own pinned values ----------

    /// Same series, window and rows as `naive`'s `generic_valid_pair_pins_rho_and_distance_bits`:
    /// feeding `Nearest` the same candidates must land on the exact same bits.
    const GENERIC_SERIES: [i64; 8] = [1, 8, 2, 50, 60, 10, 3, 7];

    #[test]
    fn generic_valid_pair_matches_naives_pinned_bits() {
        let m = 3;
        let windows: Vec<Window> = (0..=GENERIC_SERIES.len() - m)
            .map(|start| window_of(&GENERIC_SERIES[start..start + m]))
            .collect();

        let mut row2 = Nearest::new(&windows[2], m);
        let n = exact_n(&GENERIC_SERIES[2..5], &GENERIC_SERIES[0..3]);
        row2.offer(n, &windows[0], index(0));
        let entry2 = row2.finish();
        assert_eq!(entry2.index(), index(0));
        assert_eq!(entry2.rho().to_bits(), 0x3fde59b497f4fdc1);
        assert_eq!(entry2.squared_distance().to_bits(), 0x40093cb88e0841af);

        // Offered in reverse order (j = 1 before j = 0): order never changes the winner.
        let mut row3 = Nearest::new(&windows[3], m);
        let n1 = exact_n(&GENERIC_SERIES[3..6], &GENERIC_SERIES[1..4]);
        let n0 = exact_n(&GENERIC_SERIES[3..6], &GENERIC_SERIES[0..3]);
        row3.offer(n1, &windows[1], index(1));
        row3.offer(n0, &windows[0], index(0));
        let entry3 = row3.finish();
        assert_eq!(entry3.index(), index(0));
        assert_eq!(entry3.rho().to_bits(), 0x3fe1921c00da2997);
        assert_eq!(entry3.squared_distance().to_bits(), 0x4005a4d5feb8c19d);
    }

    // ---------- exact repeat beats a near miss, even when the ranking ties ----------

    #[test]
    fn exact_repeat_beats_a_near_miss_even_when_ranking_ties() {
        // B repeats itself (index 1) and A is a one-tick near miss of both (index 0); both
        // pairs' ranking ρ̂ clamp to exactly 1 (within 2ε of each other), so only the exact
        // fallback can tell the true exact match (B, ρ² = 1/1 exactly) from the near miss.
        const A: [i64; 4] = [235_732_534, -190_400_685, 248_938_659, -501_918_519];
        const B: [i64; 4] = [235_732_534, -190_400_685, 248_938_660, -501_918_519];
        let row = window_of(&B);
        let a = window_of(&A);
        let b = window_of(&B);

        for (first, second) in [(0u64, 1u64), (1u64, 0u64)] {
            let mut nearest = Nearest::new(&row, 4);
            for i in [first, second] {
                if i == 0 {
                    nearest.offer(exact_n(&B, &A), &a, index(0));
                } else {
                    nearest.offer(exact_n(&B, &B), &b, index(1));
                }
            }
            let entry = nearest.finish();
            assert_eq!(entry.index(), index(1));
            assert_eq!(entry.rho(), 1.0);
            assert_eq!(entry.squared_distance(), 0.0);
        }
    }

    // ---------- a negative near miss beats the farther exact anti-correlation ----------

    #[test]
    fn negative_near_miss_is_nearer_than_exact_anti_correlation() {
        // ρ = −1 (exact anti-correlation) is the single farthest point (d² = 4m
        // exactly); a near miss just short of −1 is slightly nearer. Negating every tick of a
        // window flips the sign of its cross term with anything while leaving every variance
        // unchanged, so the same two sequences that give a near-exact *positive* match
        // elsewhere give a near-exact *negative* one here.
        const A: [i64; 4] = [235_732_534, -190_400_685, 248_938_659, -501_918_519];
        const B: [i64; 4] = [235_732_534, -190_400_685, 248_938_660, -501_918_519];
        let negated_b: [i64; 4] = B.map(|t| -t);
        let row = window_of(&negated_b);
        let a = window_of(&A);
        let b = window_of(&B);

        let mut nearest = Nearest::new(&row, 4);
        nearest.offer(exact_n(&negated_b, &B), &b, index(0));
        nearest.offer(exact_n(&negated_b, &A), &a, index(1));
        let entry = nearest.finish();
        assert_eq!(entry.rho(), -1.0);
        // The near miss (index 1), not the farther exact anti-correlation (index 0, smaller).
        // The exact distance is only recomputed for ρ > 0, so both report the same ranking
        // distance here (4m); it is the *choice* of winner, not its reported distance, that
        // the exact comparison protects for a negative ρ.
        assert_eq!(entry.index(), index(1));
        assert_eq!(entry.squared_distance(), 16.0);
    }

    // ---------- constant and one-constant pairs ----------

    #[test]
    fn both_constant_pair_has_distance_zero() {
        let row = window_of(&[5, 5, 5]);
        let neighbor = window_of(&[9, 9, 9]);
        let mut nearest = Nearest::new(&row, 3);
        nearest.offer(0, &neighbor, index(0));
        let entry = nearest.finish();
        assert_eq!(entry.index(), index(0));
        assert_eq!(entry.rho(), 1.0);
        assert_eq!(entry.squared_distance(), 0.0);
    }

    #[test]
    fn one_constant_pair_has_distance_m_either_side() {
        let constant = window_of(&[7, 7, 7]);
        let valid = window_of(&[1, 2, 9]);

        let mut constant_row = Nearest::new(&constant, 3);
        constant_row.offer(0, &valid, index(0));
        let entry = constant_row.finish();
        assert_eq!(entry.index(), index(0));
        assert_eq!(entry.rho(), 0.5);
        assert_eq!(entry.squared_distance(), 3.0);

        let mut valid_row = Nearest::new(&valid, 3);
        valid_row.offer(0, &constant, index(0));
        let entry = valid_row.finish();
        assert_eq!(entry.index(), index(0));
        assert_eq!(entry.rho(), 0.5);
        assert_eq!(entry.squared_distance(), 3.0);
    }

    // ---------- non-finite row and neighbor ----------

    #[test]
    fn non_finite_row_is_none_regardless_of_offers() {
        let row = non_finite_window();
        let mut nearest = Nearest::new(&row, 3);
        let neighbor = window_of(&[1, 2, 9]);
        nearest.offer(5, &neighbor, index(0));
        assert_eq!(bits(nearest.finish()), bits(crate::entry::Entry::NONE));
    }

    #[test]
    fn non_finite_neighbor_is_ignored_not_chosen() {
        let row = window_of(&[1, 2, 9]);
        let real_neighbor = window_of(&[3, 5, 1]);
        let n = exact_n(&[1, 2, 9], &[3, 5, 1]);

        let mut nearest = Nearest::new(&row, 3);
        nearest.offer(0, &non_finite_window(), index(0));
        nearest.offer(n, &real_neighbor, index(1));
        let entry = nearest.finish();
        assert_eq!(entry.index(), index(1));
        assert!(entry.squared_distance().is_finite());
    }

    #[test]
    fn non_finite_row_ignores_a_non_finite_neighbor_too() {
        let mut nearest = Nearest::new(&non_finite_window(), 3);
        nearest.offer(0, &non_finite_window(), index(0));
        assert_eq!(bits(nearest.finish()), bits(crate::entry::Entry::NONE));
    }

    // ---------- property: matches naive's independently computed entry, in any order ----------

    fn tick_strategy() -> impl Strategy<Value = i64> {
        -500i64..=500
    }

    fn series_and_m() -> impl Strategy<Value = (Vec<i64>, usize)> {
        (2usize..=5).prop_flat_map(|m| (prop::collection::vec(tick_strategy(), m..=10), Just(m)))
    }

    fn shuffled(len: usize, runner: &mut TestRunner) -> Vec<usize> {
        let order: Vec<usize> = (0..len).collect();
        Just(order)
            .prop_shuffle()
            .new_tree(runner)
            .expect("shuffle strategy produces a value")
            .current()
    }

    #[test]
    fn matches_naive_bit_for_bit_in_any_order() {
        let mut runner = TestRunner::deterministic();
        let strategy = series_and_m();
        let cases = 512;

        for _ in 0..cases {
            let (series, m) = strategy
                .new_tree(&mut runner)
                .expect("strategy produces a value")
                .current();
            let samples: Vec<Option<i64>> = series.iter().copied().map(Some).collect();
            let naive_profile = crate::naive::naive_left_profile(&samples, m);
            let e = crate::exclusion::zone(m);
            let num_windows = series.len() - m + 1;
            let windows: Vec<Window> = (0..num_windows)
                .map(|start| window_of(&series[start..start + m]))
                .collect();

            for i in 0..num_windows {
                let mut nearest = Nearest::new(&windows[i], m);
                if i > e {
                    let candidates = i - e;
                    let order = shuffled(candidates, &mut runner);
                    for j in order {
                        let n = exact_n(&series[i..i + m], &series[j..j + m]);
                        nearest.offer(n, &windows[j], index(j as u64));
                    }
                }
                let entry = nearest.finish();
                let (row, expected) = naive_profile[i];
                assert_eq!(row, i as u64);
                assert_eq!(
                    bits(entry),
                    bits(expected),
                    "series {series:?}, m {m}, row {i}"
                );
            }
        }
    }

    // `row_kind` is exercised only through `Nearest::new`/`offer` above; this pins its mapping
    // directly so a change to `WindowKind`'s variants is caught here, not only end to end.
    #[test]
    fn row_kind_maps_every_window_kind() {
        assert!(matches!(row_kind(&non_finite_window()), RowKind::NonFinite));
        assert!(matches!(
            row_kind(&window_of(&[4, 4, 4])),
            RowKind::Constant
        ));
        assert!(matches!(
            row_kind(&window_of(&[1, 2, 9])),
            RowKind::Valid { .. }
        ));
    }

    // ---------- the comparator itself, pinned directly (no window needed) ----------

    #[test]
    fn sign_decides_before_magnitude() {
        // Same |ρ|² scale, opposite sign: positive must win regardless of magnitude.
        let positive_small = valid_correlation(1, 1, 10, 10);
        let negative_large = valid_correlation(-1, 81, 10, 10);
        assert_eq!(
            cmp_exact(positive_small, negative_large, Sharing::Row),
            Ordering::Greater
        );
        assert_eq!(
            cmp_exact(negative_large, positive_small, Sharing::Row),
            Ordering::Less
        );
    }

    #[test]
    fn exact_tie_keeps_the_smaller_index() {
        let tied = valid_correlation(1, 4, 2, 8); // ρ² = 4/16 = 0.25, same on both sides
        let smaller = Candidate {
            index: 2,
            rho_hat: 0.5,
            correlation: tied,
        };
        let larger = Candidate {
            index: 9,
            rho_hat: 0.5,
            correlation: tied,
        };
        assert!(nearer(smaller, larger));
        assert!(!nearer(larger, smaller));
    }

    #[test]
    fn one_constant_ratio_is_one_quarter_not_one_half() {
        // ρ² = 0.3 sits strictly between 1/4 and 1/2: it must beat a one-constant pair
        // (ρ² = 1/4), which it would not if the fixed ratio were mistakenly 1/2.
        let one_constant = ExactCorrelation::OneConstant;
        let valid = valid_correlation(1, 3, 1, 10); // ρ² = 3/10
        assert_eq!(
            cmp_exact(valid, one_constant, Sharing::Row),
            Ordering::Greater
        );
        assert_eq!(cmp_exact(one_constant, valid, Sharing::Row), Ordering::Less);
    }

    #[test]
    fn ranking_tolerance_is_symmetric_and_falls_back_within_it() {
        let tiny = f64::EPSILON;
        let band = 2.0 * EPSILON;
        let base = Candidate {
            index: 5,
            rho_hat: 0.5,
            correlation: ExactCorrelation::BothConstant,
        };
        let candidate_at = |rho_hat: f64| Candidate {
            index: 7,
            rho_hat,
            correlation: ExactCorrelation::BothConstant,
        };

        // Clearly outside the band: ρ̂ alone decides, despite the larger index.
        assert!(nearer(candidate_at(0.5 + band + tiny), base));
        assert!(!nearer(base, candidate_at(0.5 + band + tiny)));
        assert!(!nearer(candidate_at(0.5 - band - tiny), base));
        assert!(nearer(base, candidate_at(0.5 - band - tiny)));

        // Inside the band on either side: falls back to the exact tie (equal here), where
        // the smaller index (base, 5) wins over the larger (7).
        assert!(!nearer(candidate_at(0.5 + band - tiny), base));
        assert!(nearer(base, candidate_at(0.5 + band - tiny)));
        assert!(!nearer(candidate_at(0.5 - band + tiny), base));
        assert!(nearer(base, candidate_at(0.5 - band + tiny)));
    }

    /// Every band here cuts at the f64 producer's floor: `clearly_nearer`, both sides of the
    /// exact order of two entries, and a row's admission floor for one kept and for `k` kept.
    /// Pinned at an estimate of zero, whose floor `−2δ` is exact, so one ULP of `δ` moves each.
    #[test]
    fn every_band_here_cuts_at_the_f64_producers_floor() {
        use super::clearly_nearer;
        use crate::estimate::{F64, Producer, band_floor};
        let floor = band_floor(0.0, F64::DELTA);
        assert_eq!(floor.to_bits(), (-2.0 * EPSILON).to_bits());
        let below = floor.next_down();
        assert!(!clearly_nearer(0.0, floor));
        assert!(clearly_nearer(0.0, below));

        let entry = |rho_hat: f64, index: u64| {
            let index = SampleIndex::new(index).expect("a valid index");
            Entry::new(rho_hat, 1.0, index).expect("finite")
        };
        let farther = ExactDistance::from_parts(entry(0.0, 0), valid_correlation(1, 1, 1, 5), 1);
        let nearer_at = |rho_hat| {
            ExactDistance::from_parts(entry(rho_hat, 1), valid_correlation(1, 1, 1, 4), 1)
        };
        assert_eq!(farther.cmp(&nearer_at(floor)), Ordering::Greater);
        assert_eq!(farther.cmp(&nearer_at(below)), Ordering::Less);
        assert_eq!(nearer_at(floor).cmp(&farther), Ordering::Less);
        assert_eq!(nearer_at(below).cmp(&farther), Ordering::Greater);

        let row = window_of(&[1, 2, 9]);
        let neighbor = window_of(&[9, 2, 1]);
        for k in [1, 2] {
            let mut nearest = Nearest::with_k(&row, 3, NonZeroUsize::new(k).unwrap(), k);
            for i in 0..k as u64 {
                nearest.offer(0, &neighbor, index(i));
            }
            assert_eq!(nearest.admission_floor().to_bits(), floor.to_bits());
        }
    }

    /// Two entries whose `ρ̂` differ by 1.5·`EPSILON`, inside the band, where the lower one is
    /// exactly nearer: the exact order wins, in both argument orders.
    #[test]
    fn a_rho_hat_gap_inside_the_band_defers_to_the_exact_order() {
        let entry = |rho_hat: f64| {
            let index = SampleIndex::new(0).expect("zero is a valid index");
            Entry::new(rho_hat, 1.0, index).expect("finite")
        };
        let higher_rho_hat = ExactDistance::from_parts(
            entry(0.5 + 1.5 * EPSILON),
            valid_correlation(1, 1, 1, 5), // ρ² = 1/5
            1,
        );
        let nearer = ExactDistance::from_parts(
            entry(0.5),
            valid_correlation(1, 1, 1, 4), // ρ² = 1/4
            1,
        );
        assert_eq!(higher_rho_hat.cmp(&nearer), Ordering::Greater);
        assert_eq!(nearer.cmp(&higher_rho_hat), Ordering::Less);
    }

    // ---------- gaps the reducer-exact verifier found: a literal pin, an order property,
    // ---------- and a named "something beats nothing" test ----------

    /// A row's offered candidates all share one `row_variance` (as every real `Nearest` row
    /// does): `cmp_magnitude`'s wide-product shortcut cancels that shared `Vi`, so comparing
    /// candidates built from independently random `row_variance` would exercise a case
    /// `nearer` never actually faces.
    fn correlation_strategy(row_variance: u64) -> impl Strategy<Value = ExactCorrelation> {
        prop_oneof![
            1 => Just(ExactCorrelation::BothConstant),
            1 => Just(ExactCorrelation::OneConstant),
            3 => (
                prop_oneof![Just(-1i8), Just(0i8), Just(1i8)],
                0u128..(1u128 << 40),
                1u64..(1u64 << 40),
            )
                .prop_map(move |(sign, cross_squared, neighbor_variance)| {
                    valid_correlation(sign, cross_squared, row_variance, neighbor_variance)
                }),
        ]
    }

    fn candidate_strategy(row_variance: u64) -> impl Strategy<Value = Candidate> {
        (
            0u64..1000,
            -1.0f64..=1.0,
            correlation_strategy(row_variance),
        )
            .prop_map(|(index, rho_hat, correlation)| Candidate {
                index,
                rho_hat,
                correlation,
            })
    }

    /// Three candidates from one shared row, for the order property below.
    fn candidate_triple() -> impl Strategy<Value = (Candidate, Candidate, Candidate)> {
        (1u64..(1u64 << 62)).prop_flat_map(|row_variance| {
            (
                candidate_strategy(row_variance),
                candidate_strategy(row_variance),
                candidate_strategy(row_variance),
            )
        })
    }

    proptest! {
        #![proptest_config(testkit::generators::seeded_default())]
        /// `nearer` (equivalently `cmp_exact`, which it wraps with the ρ̂ band) is a strict
        /// total order over candidates from one row: irreflexive, asymmetric, and transitive.
        /// `Nearest`'s top-k insertion relies on exactly this to keep its kept candidates
        /// sorted by a single pairwise comparison.
        #[test]
        fn nearer_is_a_strict_total_order((a, b, c) in candidate_triple()) {
            prop_assert!(!nearer(a, a), "irreflexive");
            prop_assert!(!(nearer(a, b) && nearer(b, a)), "asymmetric");
            if nearer(a, b) && nearer(b, c) {
                prop_assert!(nearer(a, c), "transitive");
            }
        }
    }

    /// `a · b` as eight little-endian 32-bit limbs, by a different decomposition than
    /// `Wide::product`'s.
    fn reference_product(a: u128, b: u128) -> [u32; 8] {
        let limbs = |x: u128| {
            [
                x as u32,
                (x >> 32) as u32,
                (x >> 64) as u32,
                (x >> 96) as u32,
            ]
        };
        let mut out = [0u64; 9];
        for (i, &x) in limbs(a).iter().enumerate() {
            let mut carry = 0u64;
            for (j, &y) in limbs(b).iter().enumerate() {
                let total = out[i + j] + u64::from(x) * u64::from(y) + carry;
                out[i + j] = total & 0xffff_ffff;
                carry = total >> 32;
            }
            out[i + 4] += carry;
        }
        core::array::from_fn(|i| out[i] as u32)
    }

    fn reference_cmp_products(a: u128, b: u128, c: u128, d: u128) -> Ordering {
        let left = reference_product(a, b);
        let right = reference_product(c, d);
        left.iter().rev().cmp(right.iter().rev())
    }

    fn below_two_to_the_126() -> impl Strategy<Value = u128> {
        prop_oneof![
            any::<u128>().prop_map(|x| x >> 2),
            Just((1u128 << 126) - 1),
            Just(1u128 << 125),
            0u128..4,
        ]
    }

    /// Products within one unit of each other whose limbs all carry: a dropped carry in the
    /// middle or the top limb flips these. Each row is `a, b, c, d` and the order of `a·b`
    /// against `c·d`.
    #[test]
    fn near_equal_wide_products_keep_every_carry() {
        let witnesses: [(u128, u128, u128, u128, Ordering); 15] = [
            (
                32447098705601090803096464345569421272,
                57604828351084230019395926409867080227,
                57940533149498494384878894067521424763,
                32259101700085344457784066857943520551,
                Ordering::Less,
            ),
            (
                32288140498170673667276966218621481022,
                18791017934734309463137595869596579375,
                37550030841094870080887234933917274090,
                16157830329032443084679927974854162426,
                Ordering::Less,
            ),
            (
                77769540214770078501165620125533067346,
                14476965719644821423938153629633360239,
                70388992263038078386143078669334619928,
                15994929484350198879155400064350641213,
                Ordering::Less,
            ),
            (
                41511932887746660878656019804353589819,
                49026036790424829656985656701849882344,
                38661157476362079382029351859988667857,
                52641092037677355921794655488764391641,
                Ordering::Greater,
            ),
            (
                54043372346880837895403303269519642964,
                1853759158323542772109917946202012334,
                73353657568672045637158466335033979509,
                1365758705909520657576352439049693239,
                Ordering::Less,
            ),
            (
                75196754733771706515845085046227981895,
                58478711075473328695981106958930331168,
                75738403539882863360247927520955603711,
                58060496239187932104671653966391132829,
                Ordering::Less,
            ),
            (
                13356554606064292829138957268166416095,
                40290480384251338321858378858075079494,
                9469792555646304040152328225669567776,
                56827221736335869365506596809174344582,
                Ordering::Less,
            ),
            (
                10467436767346343765513228283061834550,
                23301941754473205162514179907492572924,
                17883488833568616370044985743531320535,
                13638927176977675775573782700850505571,
                Ordering::Less,
            ),
            (
                39584296198669177838262224702455480906,
                639536651579648371016628536996848471,
                19857194345266952053336147380606117792,
                1274883440523307055782960568073593914,
                Ordering::Less,
            ),
            (
                13930583612872808128332466911802322669,
                59057764136210470269220043785075910404,
                40907751446017314606860765423572065739,
                20111325902976221720562992357298459764,
                Ordering::Less,
            ),
            (
                13356554606064292829138957268166416095,
                40290480384251338321858378858075079494,
                9469792555646304040152328225669567776,
                56827221736335869365506596809174344582,
                Ordering::Less,
            ),
            (
                60728198001463042195443858915300935499,
                24814664252030976228923812385514155667,
                43426612343792460092462083825545999064,
                34701068370408409123199410040067214782,
                Ordering::Less,
            ),
            (
                66842585825861579043580682572965377709,
                24050357965599260094188445424374030856,
                68690461913597763296404367231512719245,
                23403367391536331693922752048656937627,
                Ordering::Less,
            ),
            (
                69238663658415209048086832422886756651,
                8542400020373164050092449066930190992,
                43510221070530343520908641849991847330,
                13593687811594642647912205398852600075,
                Ordering::Less,
            ),
            (
                39584296198669177838262224702455480906,
                639536651579648371016628536996848471,
                19857194345266952053336147380606117792,
                1274883440523307055782960568073593914,
                Ordering::Less,
            ),
        ];
        for (a, b, c, d, expected) in witnesses {
            assert_eq!(reference_cmp_products(a, b, c, d), expected);
            assert_eq!(cmp_wide_products(a, b, c, d), expected);
            assert_eq!(cmp_wide_products(c, d, a, b), expected.reverse());
        }
    }

    proptest! {
        #![proptest_config(testkit::generators::seeded_default())]
        /// `a·b = c·d` exactly by construction (`a = xy`, `b = zw`, `c = xz`, `d = yw`), then `d`
        /// nudged by one: products up to `2^252` that differ only in their lowest bit, which
        /// only a correct carry through every limb gets right.
        #[test]
        fn wide_products_one_apart_compare_exactly(
            factors in prop::array::uniform4(prop_oneof![
                1u128..(1u128 << 63),
                Just((1u128 << 63) - 2),
                (1u128 << 62)..(1u128 << 63),
            ]),
            nudge in 0u128..2,
        ) {
            let [x, y, z, w] = factors;
            let (a, b, c, d) = (x * y, z * w, x * z, y * w + nudge);
            prop_assert_eq!(cmp_wide_products(a, b, c, d), reference_cmp_products(a, b, c, d));
            let expected = if nudge == 0 { Ordering::Equal } else { Ordering::Less };
            prop_assert_eq!(cmp_wide_products(a, b, c, d), expected);
        }
    }

    proptest! {
        #![proptest_config(testkit::generators::seeded_default())]
        #[test]
        fn wide_products_compare_like_the_reference(
            a in below_two_to_the_126(),
            b in below_two_to_the_126(),
            c in below_two_to_the_126(),
            d in below_two_to_the_126(),
        ) {
            prop_assert_eq!(cmp_wide_products(a, b, c, d), reference_cmp_products(a, b, c, d));
            prop_assert_eq!(cmp_wide_products(a, b, a, b), Ordering::Equal);
        }

        /// Cancelling a shared factor, and comparing small products in 128 bits, order the
        /// products as the 256-bit reference does: whichever factors are shared, zero or small.
        #[test]
        fn cross_products_compare_like_the_reference(
            factors in prop::array::uniform4(prop_oneof![
                Just(0u128),
                Just(1u128),
                0u128..(1u128 << 32),
                0u128..(1u128 << 64),
                below_two_to_the_126(),
            ]),
            shared in 0u8..4,
        ) {
            let [a, b, mut c, mut d] = factors;
            match shared {
                0 => c = a,
                1 => d = b,
                _ => {}
            }
            prop_assert_eq!(cmp_cross_products(a, b, c, d), reference_cmp_products(a, b, c, d));
        }

        /// With nothing shared, the order is that of `N²/(Vi·Vj)` as an exact rational; with
        /// a variance shared, the cheaper arms agree with it.
        #[test]
        fn the_nothing_arm_is_the_exact_rational_order(
            na in 0u128..(1u128 << 100),
            nb in 0u128..(1u128 << 100),
            variances in prop::array::uniform4(1u64..(1u64 << 62)),
            sign in prop_oneof![Just(-1i8), Just(1i8)],
        ) {
            let [via, vja, vib, vjb] = variances;
            let a = valid_correlation(sign, na, via, vja);
            let b = valid_correlation(sign, nb, vib, vjb);
            let by_cross_product = reference_cmp_products(
                na,
                variance_product(vib, vjb),
                nb,
                variance_product(via, vja),
            );
            let expected = if sign > 0 { by_cross_product } else { by_cross_product.reverse() };
            prop_assert_eq!(cmp_exact(a, b, Sharing::Nothing), expected);
            prop_assert_eq!(cmp_exact(b, a, Sharing::Nothing), expected.reverse());

            let same_row = valid_correlation(sign, nb, via, vjb);
            prop_assert_eq!(
                cmp_exact(a, same_row, Sharing::Row),
                cmp_exact(a, same_row, Sharing::Nothing)
            );
            let same_neighbor = valid_correlation(sign, nb, vib, vja);
            prop_assert_eq!(
                cmp_exact(a, same_neighbor, Sharing::Neighbor),
                cmp_exact(a, same_neighbor, Sharing::Nothing)
            );
        }
    }

    /// Past what any single u128 product holds: `ρ²` of one tick apart at the largest
    /// variances still orders correctly.
    #[test]
    fn the_nothing_arm_separates_neighbors_at_the_largest_variances() {
        let largest = (1u64 << 63) - 1;
        let denominator = variance_product(largest, largest);
        let nearer = valid_correlation(1, denominator - 1, largest, largest);
        let farther = valid_correlation(1, denominator - 2, largest, largest);
        assert_eq!(
            cmp_exact(nearer, farther, Sharing::Nothing),
            Ordering::Greater
        );
        let smaller_denominator = variance_product(largest - 1, largest);
        let other_variances = valid_correlation(1, smaller_denominator - 1, largest - 1, largest);
        assert_eq!(
            cmp_exact(nearer, other_variances, Sharing::Nothing),
            Ordering::Greater
        );
    }

    /// A row with one real, eligible neighbor never reports [`Entry::NONE`]: offering
    /// anything real always beats offering nothing.
    #[test]
    fn a_real_neighbor_is_chosen_over_no_neighbor() {
        let row = window_of(&[1, 2, 9]);
        let neighbor = window_of(&[3, 5, 1]);
        let n = exact_n(&[1, 2, 9], &[3, 5, 1]);

        let mut nearest = Nearest::new(&row, 3);
        nearest.offer(n, &neighbor, index(0));
        let entry = nearest.finish();

        assert_ne!(bits(entry), bits(crate::entry::Entry::NONE));
        assert!(entry.squared_distance().is_finite());
    }

    // ---------- with_k/finish_k: nearest-first order and tie-at-the-cutoff eviction ----------

    #[test]
    fn with_k_keeps_the_nearest_two_nearest_first() {
        // An exact repeat of the row (rho = 1, the best any pair can do), a shifted copy
        // (same shape as the row, so still a strong positive match but not exact), and a
        // reversal (negative match, the worst of the three): a strict, hand-verified order.
        const ROW: [i64; 3] = [1, 2, 9];
        const EXACT: [i64; 3] = [1, 2, 9];
        const SHIFTED: [i64; 3] = [1, 3, 8];
        const REVERSED: [i64; 3] = [9, 2, 1];
        let row = window_of(&ROW);
        let exact = window_of(&EXACT);
        let shifted = window_of(&SHIFTED);
        let reversed = window_of(&REVERSED);

        let mut nearest = Nearest::with_k(&row, 3, NonZeroUsize::new(2).unwrap(), 2);
        nearest.offer(exact_n(&ROW, &REVERSED), &reversed, index(2));
        nearest.offer(exact_n(&ROW, &EXACT), &exact, index(0));
        nearest.offer(exact_n(&ROW, &SHIFTED), &shifted, index(1));

        let kept = nearest.finish_k();
        let kept_indices: Vec<u64> = kept.iter().map(|e| e.index().get().unwrap()).collect();
        assert_eq!(kept_indices, [0u64, 1u64]);
        assert_eq!(kept[0].rho(), 1.0);
        assert!(kept[0].squared_distance() < kept[1].squared_distance());
    }

    #[test]
    fn with_k_tie_at_the_cutoff_keeps_the_smaller_index() {
        // A unique best (X) and two bit-identical candidates (Y, offered at two different
        // indices): their exact correlation ties, so only the smaller index may survive
        // k = 2's cutoff, whichever order the two Ys are offered in.
        const ROW: [i64; 3] = [1, 2, 9];
        const X: [i64; 3] = [1, 3, 8];
        const Y: [i64; 3] = [9, 2, 1];
        let row = window_of(&ROW);
        let x = window_of(&X);
        let y = window_of(&Y);

        for (first, second) in [(5u64, 9u64), (9u64, 5u64)] {
            let mut nearest = Nearest::with_k(&row, 3, NonZeroUsize::new(2).unwrap(), 2);
            nearest.offer(exact_n(&ROW, &X), &x, index(0));
            for i in [first, second] {
                nearest.offer(exact_n(&ROW, &Y), &y, index(i));
            }
            let kept = nearest.finish_k();
            let kept_indices: Vec<u64> = kept.iter().map(|e| e.index().get().unwrap()).collect();
            assert_eq!(kept_indices, [0u64, 5u64], "offer order {first}, {second}");
        }
    }

    // ---------- rescan_near_tie: the chunk-skip fold accepts a lane at the threshold exactly

    /// `rescan_near_tie`'s own chunk-skip fold must accept a lane sitting at exactly
    /// `threshold` (`M − 2ε`), not only one strictly above it: two chunks, each with exactly
    /// one qualifying lane — chunk 0's own lane `0` holds the row's maximum `M`; chunk 1's own
    /// lane `0` (absolute index `RESCAN_CHUNK`) holds exactly `M − 2·EPSILON`, the near-tie
    /// band's own lower edge, landing in a chunk other than the maximum's. Folding with `>`
    /// instead of `>=` reports chunk 1 as having no hit at all, silently skipping it.
    #[test]
    fn rescan_near_tie_accepts_a_lane_at_the_threshold_exactly() {
        const M_RHO: f64 = 0.75; // a representative mid-range maximum; nothing special about it
        let threshold = M_RHO - 2.0 * EPSILON;
        let len = 2 * RESCAN_CHUNK;
        let mut scratch = std::vec![f64::NEG_INFINITY; len];
        scratch[0] = M_RHO;
        scratch[RESCAN_CHUNK] = threshold;

        let mut hits: Vec<usize> = Vec::new();
        rescan_near_tie(&scratch, threshold, |j| hits.push(j));

        assert_eq!(hits, std::vec![0, RESCAN_CHUNK]);
    }
}
