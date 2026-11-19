//! The cold exact route for windows the hot passes cannot rank: a window holding a tick far from
//! its neighbors' level, or whose variance numerator passes `2⁶³` (it has no distance today). The
//! hot passes see such a window as non-finite; this module ranks every pair that touches one from
//! the exact ticks and folds the results into the hot entries, so each entry is the exact
//! nearest with the hot passes' own `ρ̂` and distance.
//!
//! Two exact tiers, chosen by the series' range `R` and window length `m`: `i128` while
//! `m·R < 2⁶⁴`, and a 256-bit tier while `m·R < 2⁸⁶`. Beyond that the route is not available
//! ([`Survey::new`] returns `None`). Ranking is the shared `nearer` and `winner_entry`
//! over a wide correlation (`Spilled`); near-ties are settled on 512-bit products.
//!
//! A pass covers the pairs of one series' windows with its own earlier or later windows (left or
//! right of a profile), or of one series' windows with another's (a join), whichever touch a
//! spilled window, row by row, so a band of rows is independent of the others.
//!
//! The outliers are defined from the data and the exact domain alone, by one causal rule the
//! streams apply sample by sample and batch over the whole series (`is_outlier`): a tick is an
//! outlier when it lies further from the newest present tick before it that is not one than
//! [`lone_deviation_limit`], the deviation at which a window holding it with every other tick at
//! that level would leave the exact domain, unless the `m` before it were outliers. The narrow
//! anchor's bound, a tile plan and the cost model never enter the rule.

use alloc::borrow::Cow;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::fmt::Debug;
use core::num::NonZeroUsize;
use core::ops::Range;

use crate::capacity::{CapacityError, reserved_vec};
use crate::entry::Entry;
use crate::nearest::{
    Candidate, DistanceParts, ExactCorrelation, Held, Ranked, clearly_nearer, nearer,
    variance_product,
};
use crate::stats::{Window, WindowKind};
use crate::wide::{Wide, i128_to_f64};

mod pass;
pub(crate) mod stream;

pub use self::pass::Counts;
use self::pass::{Cover, Marks};
pub(crate) use self::pass::{HotRuns, Reach};

/// The bytes of cold candidates one band of rows may hold at once.
const BAND_BYTES: usize = 8 << 20;

/// Ticks the outlier scan screens at once: a stretch whose present ticks, with the level it
/// starts from, span no more than the limit holds no outlier.
const SCREEN_TICKS: usize = 1024;

/// The variance numerator at and past which a window leaves the exact domain of the hot passes.
const EXACT_VARIANCE_LIMIT: u64 = 1 << 63;

/// The largest deviation from the level that a tick may have: `D` with `(m − 1)·D² < 2⁶³`, the
/// variance numerator of a window of `m` whose other ticks all sit at the level. A window of one
/// tick has no variance, so nothing is an outlier there.
pub fn lone_deviation_limit(m: usize) -> u64 {
    match (m as u64).checked_sub(1) {
        None | Some(0) => u64::MAX,
        Some(others) => ((EXACT_VARIANCE_LIMIT - 1) / others).isqrt(),
    }
}

/// The outlier rule, for a present `tick` read after earlier samples: it is an outlier when it
/// lies further than [`lone_deviation_limit`] from `level`, the newest earlier present tick that
/// is not one, and fewer than `m` present ticks since that one were outliers (`run`), so a level
/// that persists becomes the new level. Gaps are skipped; the first present tick is never one.
pub(crate) fn is_outlier(m: usize, tick: i64, level: i64, run: impl FnOnce() -> usize) -> bool {
    tick.abs_diff(level) > lone_deviation_limit(m) && run() < m
}

/// `is_outlier` applied to a series in order: the level and the run so far.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct OutlierScan {
    level: Option<i64>,
    run: usize,
}

impl OutlierScan {
    /// Whether the present `tick` is withheld, `engaged` deciding from what the rule says.
    pub(crate) fn step(&mut self, m: usize, tick: i64, engaged: impl FnOnce(bool) -> bool) -> bool {
        let withheld = self
            .level
            .is_some_and(|level| engaged(is_outlier(m, tick, level, || self.run)));
        if withheld {
            self.withhold();
        } else {
            self.keep(tick);
        }
        withheld
    }

    /// The newest present tick kept, the level the next one is measured from.
    pub(crate) fn level(&self) -> Option<i64> {
        self.level
    }

    /// The present ticks withheld since [`Self::level`].
    pub(crate) fn run(&self) -> usize {
        self.run
    }

    /// Records a present tick kept: the new level.
    pub(crate) fn keep(&mut self, tick: i64) {
        (self.level, self.run) = (Some(tick), 0);
    }

    /// Records a present tick withheld.
    pub(crate) fn withhold(&mut self) {
        self.run += 1;
    }
}

/// `len` flags, all false, or a capacity error.
fn cleared(len: usize) -> Result<Vec<bool>, CapacityError> {
    let mut flags = reserved_vec::<bool>(len as u128)?;
    flags.resize(len, false);
    Ok(flags)
}

/// Calls `found(p)` for each outlier `ticks[p]` (present where `had[p]`), in order, until it
/// returns false.
///
/// A stretch of [`SCREEN_TICKS`] whose present ticks and the level it starts from span no more
/// than the limit holds none: each tick's level is one of them. It leaves the newest present tick
/// as the level and no run, so only the other stretches are read tick by tick.
pub(crate) fn each_outlier(
    ticks: &[i64],
    had: &[bool],
    m: usize,
    mut found: impl FnMut(usize) -> bool,
) {
    let limit = lone_deviation_limit(m);
    let mut scan = OutlierScan::default();
    for (index, stretch) in ticks.chunks(SCREEN_TICKS).enumerate() {
        let base = index * SCREEN_TICKS;
        let had = &had[base..base + stretch.len()];
        let (mut low, mut high) = scan
            .level
            .map_or((i64::MAX, i64::MIN), |level| (level, level));
        for (&tick, &measured) in stretch.iter().zip(had) {
            low = low.min(if measured { tick } else { i64::MAX });
            high = high.max(if measured { tick } else { i64::MIN });
        }
        if low > high || high.abs_diff(low) <= limit {
            if let Some(newest) = had.iter().rposition(|&measured| measured) {
                scan = OutlierScan {
                    level: Some(stretch[newest]),
                    run: 0,
                };
            }
            continue;
        }
        for (offset, (&tick, &measured)) in stretch.iter().zip(had).enumerate() {
            if measured && scan.step(m, tick, |outlier| outlier) && !found(base + offset) {
                return;
            }
        }
    }
}

/// Which present ticks are outliers by `is_outlier`: `flags[p]` for `ticks[p]`, present when
/// `had[p]`.
pub fn outlier_ticks(ticks: &[i64], had: &[bool], m: usize) -> Result<Vec<bool>, CapacityError> {
    let mut flags = cleared(ticks.len())?;
    each_outlier(ticks, had, m, |p| {
        flags[p] = true;
        true
    });
    Ok(flags)
}

/// Whether no window can be ranked cold: no tick is an outlier, and every window the hot passes
/// cannot rank (`unranked(i)`, only if `any_unranked`) has a missing sample. A `true` is exact; a
/// `false` only says a survey must look.
pub(crate) fn nothing_to_rank(
    ticks: &[i64],
    had: &[bool],
    m: usize,
    any_unranked: bool,
    unranked: impl Fn(usize) -> bool,
) -> Result<bool, CapacityError> {
    if any_unranked {
        let mut missing = had[..m.min(had.len())].iter().filter(|&&h| !h).count();
        for row in 0..(ticks.len() + 1).saturating_sub(m) {
            if row > 0 {
                missing += usize::from(!had[row + m - 1]);
                missing -= usize::from(!had[row - 1]);
            }
            if missing == 0 && unranked(row) {
                return Ok(false);
            }
        }
    }
    let mut found = false;
    each_outlier(ticks, had, m, |_| {
        found = true;
        false
    });
    Ok(!found)
}

/// Exact signed arithmetic of one tier. Addition, subtraction and multiplication wrap at the
/// tier's width; the tier is chosen so that every cross term and variance fits it.
pub(crate) trait Arith: Copy + Debug {
    const ZERO: Self;
    fn from_i64(value: i64) -> Self;
    fn from_u64(value: u64) -> Self;
    /// `a · b`, exactly.
    fn product(a: i64, b: i64) -> Self;
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn mul(self, other: Self) -> Self;
    fn is_negative(self) -> bool;
    fn is_zero(self) -> bool;
    fn magnitude(self) -> Wide<8>;
    /// The value rounded to the nearest `f64`, ties to even.
    fn to_f64(self) -> f64;

    /// The value on 256 bits.
    fn to_wide4(self) -> Wide<4>;

    /// The value when it is nonnegative and below `2⁶³`.
    fn below_2_63(self) -> Option<u64> {
        let magnitude = self.magnitude();
        (!self.is_negative() && magnitude.bit_length() < 64).then(|| magnitude.low_u64())
    }
}

impl Arith for i128 {
    const ZERO: Self = 0;

    fn from_i64(value: i64) -> Self {
        i128::from(value)
    }

    fn from_u64(value: u64) -> Self {
        i128::from(value)
    }

    fn product(a: i64, b: i64) -> Self {
        i128::from(a) * i128::from(b)
    }

    fn add(self, other: Self) -> Self {
        self.wrapping_add(other)
    }

    fn sub(self, other: Self) -> Self {
        self.wrapping_sub(other)
    }

    fn mul(self, other: Self) -> Self {
        self.wrapping_mul(other)
    }

    fn is_negative(self) -> bool {
        self < 0
    }

    fn is_zero(self) -> bool {
        self == 0
    }

    fn magnitude(self) -> Wide<8> {
        Wide::<2>::from_u128(self.unsigned_abs()).widen()
    }

    fn to_f64(self) -> f64 {
        i128_to_f64(self)
    }

    fn to_wide4(self) -> Wide<4> {
        Wide::from_i128(self)
    }
}

impl Arith for Wide<4> {
    const ZERO: Self = Wide::ZERO;

    fn from_i64(value: i64) -> Self {
        Wide::from_i128(i128::from(value))
    }

    fn from_u64(value: u64) -> Self {
        Wide::from_u64(value)
    }

    fn product(a: i64, b: i64) -> Self {
        Wide::from_i128(i128::from(a) * i128::from(b))
    }

    fn add(self, other: Self) -> Self {
        self.wrapping_add(&other)
    }

    fn sub(self, other: Self) -> Self {
        self.wrapping_sub(&other)
    }

    fn mul(self, other: Self) -> Self {
        self.wrapping_mul(&other)
    }

    fn is_negative(self) -> bool {
        Wide::is_negative(&self)
    }

    fn is_zero(self) -> bool {
        Wide::is_zero(&self)
    }

    fn magnitude(self) -> Wide<8> {
        self.unsigned_abs().widen()
    }

    fn to_f64(self) -> f64 {
        self.to_f64_signed()
    }

    fn to_wide4(self) -> Wide<4> {
        self
    }
}

/// The exact arithmetic a series needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Width {
    /// `m·R < 2⁶⁴`: every cross term and variance fits `i128`.
    Narrow128,
    /// `m·R < 2⁸⁶`: they fit a 256-bit integer, and every product a comparison forms fits 512 bits.
    Wide256,
}

/// The tier for `ticks` at window length `m`, `None` when `m·R ≥ 2⁸⁶` (`R` the range of `ticks`).
pub fn width_for(ticks: &[i64], m: usize) -> Option<Width> {
    let (Some(&low), Some(&high)) = (ticks.iter().min(), ticks.iter().max()) else {
        return Some(Width::Narrow128);
    };
    let range = (i128::from(high) - i128::from(low)) as u128;
    let reach = range.checked_mul(m as u128)?;
    if reach < 1 << 64 {
        Some(Width::Narrow128)
    } else if reach < 1 << 86 {
        Some(Width::Wide256)
    } else {
        None
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Missing,
    Constant,
    Valid,
}

/// One window's exact statistics.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ColdWindow<A> {
    kind: Kind,
    sum: A,
    variance: A,
    inv_sqrt: f64,
}

impl<A: Arith> ColdWindow<A> {
    /// A window that is no candidate.
    pub(crate) const MISSING: Self = Self {
        kind: Kind::Missing,
        sum: A::ZERO,
        variance: A::ZERO,
        inv_sqrt: 0.0,
    };

    /// The window of exact `sum` and variance numerator `variance`, under the caller's flag for
    /// it as the hot passes apply one: flagged constant is constant, and equal ticks not flagged
    /// constant are valid with a zero variance, so every correlation with them is zero.
    pub(crate) fn classify(missing: bool, sum: A, variance: A, flag: Option<bool>) -> Self {
        let kind = match (missing, flag) {
            (true, _) => Kind::Missing,
            (false, Some(true)) => Kind::Constant,
            (false, None) if variance.is_zero() => Kind::Constant,
            _ => Kind::Valid,
        };
        let inv_sqrt = match kind {
            Kind::Valid if variance.is_zero() => 1.0,
            Kind::Valid => 1.0 / libm::sqrt(variance.to_f64()),
            _ => 0.0,
        };
        Self {
            kind,
            sum,
            variance,
            inv_sqrt,
        }
    }

    /// The exact window of `ticks`, every one present.
    pub(crate) fn of_ticks(ticks: &[i64], flag: Option<bool>) -> Self {
        let (sum, sum_sq) = ticks
            .iter()
            .fold((A::ZERO, A::ZERO), |(sum, sum_sq), &tick| {
                (
                    sum.add(A::from_i64(tick)),
                    sum_sq.add(A::product(tick, tick)),
                )
            });
        let variance = A::from_u64(ticks.len() as u64)
            .mul(sum_sq)
            .sub(sum.mul(sum));
        Self::classify(false, sum, variance, flag)
    }

    /// A window as the hot passes hold it (the caller's flag already applied), with its exact
    /// `sum`; a non-finite one is missing.
    pub(crate) fn of_hot(window: &Window, sum: A) -> Self {
        let (kind, variance, inv_sqrt) = match window.kind {
            WindowKind::NonFinite => (Kind::Missing, A::ZERO, 0.0),
            WindowKind::Constant => (Kind::Constant, A::ZERO, 0.0),
            WindowKind::Valid {
                variance_numerator,
                inv_sqrt,
            } => (Kind::Valid, A::from_u64(variance_numerator), inv_sqrt),
        };
        Self {
            kind,
            sum,
            variance,
            inv_sqrt,
        }
    }

    pub(crate) fn is_constant(&self) -> bool {
        self.kind == Kind::Constant
    }

    /// Whether the window is present and its variance numerator at least `2⁶³`, past what the
    /// hot passes hold, whatever its flag makes it.
    pub(crate) fn too_wide(&self) -> bool {
        self.kind != Kind::Missing && self.variance.below_2_63().is_none()
    }
}

/// `Σ a[k]·b[k]`, exactly.
pub(crate) fn dot<A: Arith>(a: &[i64], b: &[i64]) -> A {
    a.iter()
        .zip(b)
        .fold(A::ZERO, |total, (&x, &y)| total.add(A::product(x, y)))
}

/// The candidate of `row` against `neighbor` (window `index`), whose dot product is `qt`: `None`
/// for a pair with a missing window, or one whose estimate `skip` says is clearly out of the
/// running.
pub(crate) fn cold_candidate<A: Arith>(
    m: usize,
    (row, neighbor): (&ColdWindow<A>, &ColdWindow<A>),
    qt: A,
    index: u64,
    skip: impl FnOnce(f64) -> bool,
) -> Option<Candidate<Spilled<A>>> {
    let (rho_hat, correlation) = match (row.kind, neighbor.kind) {
        (Kind::Missing, _) | (_, Kind::Missing) => return None,
        (Kind::Constant, Kind::Constant) => (1.0, Spilled::BothConstant),
        (Kind::Constant, _) | (_, Kind::Constant) => (0.5, Spilled::OneConstant),
        (Kind::Valid, Kind::Valid) => {
            let n = A::from_u64(m as u64).mul(qt).sub(row.sum.mul(neighbor.sum));
            let rho_hat = ((n.to_f64() * row.inv_sqrt) * neighbor.inv_sqrt).clamp(-1.0, 1.0);
            if skip(rho_hat) {
                return None;
            }
            let correlation = Spilled::Valid {
                n,
                row_variance: row.variance,
                neighbor_variance: neighbor.variance,
            };
            (rho_hat, correlation)
        }
    };
    Some(Candidate {
        index,
        rho_hat,
        correlation,
    })
}

/// Whether `cold` is exactly nearer than `hot`, the hot passes' entry for the same row; `exact`
/// builds the hot neighbor's exact candidate and runs only to settle a near-tie.
pub(crate) fn cold_nearer<A: Arith>(
    cold: Candidate<Spilled<A>>,
    hot: Entry,
    exact: impl FnOnce() -> Option<Candidate<Spilled<A>>>,
) -> bool {
    if clearly_nearer(cold.rho_hat, hot.rho()) {
        return true;
    }
    if clearly_nearer(hot.rho(), cold.rho_hat) {
        return false;
    }
    exact().is_some_and(|exact| nearer(cold, exact))
}

/// A cold pair's exact correlation, held at the widest tier.
pub(crate) type WideCorrelation = Spilled<Wide<4>>;

/// The exact correlation `held` as `(sign, ρ² numerator, ρ² denominator)`, exact on 1024 bits.
fn fraction(held: Held) -> (i8, Wide<16>, Wide<16>) {
    match held {
        Held::Hot(ExactCorrelation::BothConstant) => (1, Wide::from_u64(1), Wide::from_u64(1)),
        Held::Hot(ExactCorrelation::OneConstant) => (1, Wide::from_u64(1), Wide::from_u64(4)),
        Held::Hot(ExactCorrelation::Valid {
            sign,
            cross_squared,
            denominator,
            ..
        }) => (
            sign,
            Wide::from_u128(cross_squared),
            Wide::from_u128(denominator),
        ),
        Held::Cold(correlation) => {
            let (numerator, denominator) = correlation.ratio();
            (correlation.sign(), numerator.widen(), denominator.widen())
        }
    }
}

/// The exact order of two held correlations of any rows: `Greater` when `a` is nearer, as
/// `cmp_exact` orders the hot ones.
pub(crate) fn cmp_held(a: Held, b: Held) -> Ordering {
    let ((sign_a, num_a, den_a), (sign_b, num_b, den_b)) = (fraction(a), fraction(b));
    if sign_a != sign_b {
        return sign_a.cmp(&sign_b);
    }
    if sign_a == 0 {
        return Ordering::Equal;
    }
    let order = num_a.wrapping_mul(&den_b).cmp(&num_b.wrapping_mul(&den_a));
    if sign_a > 0 { order } else { order.reverse() }
}

/// Which list a merge takes its next neighbor from.
#[derive(Clone, Copy)]
pub(crate) enum Pick {
    Hot(usize),
    Cold(usize),
}

/// Merges `hot` hot entries and the cold candidates `cold` of one row, both nearest first and
/// sharing no neighbor, calling `pick` for at most `k` of them, nearest first; `cold_wins` says
/// whether a cold candidate is nearer than a hot entry (by its position).
pub(crate) fn merge_nearest<A: Arith>(
    hot: usize,
    cold: &[Candidate<Spilled<A>>],
    k: usize,
    mut cold_wins: impl FnMut(Candidate<Spilled<A>>, usize) -> bool,
    mut pick: impl FnMut(Pick),
) {
    let (mut at_hot, mut at_cold) = (0, 0);
    for _ in 0..k {
        let take_cold = match (at_hot < hot, cold.get(at_cold)) {
            (false, None) => break,
            (true, None) => false,
            (false, Some(_)) => true,
            (true, Some(&candidate)) => cold_wins(candidate, at_hot),
        };
        if take_cold {
            pick(Pick::Cold(at_cold));
            at_cold += 1;
        } else {
            pick(Pick::Hot(at_hot));
            at_hot += 1;
        }
    }
}

/// A candidate's exact correlation on a tier: `sign(N)` and `|N|` with both variances for a valid
/// pair, the fixed ratios for a constant one (both constant `1/1`, one constant `1/4`).
#[derive(Clone, Copy, Debug)]
pub(crate) enum Spilled<A> {
    BothConstant,
    OneConstant,
    Valid {
        n: A,
        row_variance: A,
        neighbor_variance: A,
    },
}

impl<A: Arith> Spilled<A> {
    /// The correlation an [`ExactDistance`](crate::nearest::ExactDistance) holds: as the hot
    /// passes hold it when both variances are below `2⁶³`, else on 256 bits.
    pub(crate) fn held(&self) -> Held {
        self.hot_form().map_or_else(
            || {
                Held::Cold(match *self {
                    Self::BothConstant => Spilled::BothConstant,
                    Self::OneConstant => Spilled::OneConstant,
                    Self::Valid {
                        n,
                        row_variance,
                        neighbor_variance,
                    } => Spilled::Valid {
                        n: n.to_wide4(),
                        row_variance: row_variance.to_wide4(),
                        neighbor_variance: neighbor_variance.to_wide4(),
                    },
                })
            },
            Held::Hot,
        )
    }

    /// The same correlation as the hot passes hold it, when both variances are below `2⁶³`.
    pub(crate) fn hot_form(&self) -> Option<ExactCorrelation> {
        Some(match *self {
            Self::BothConstant => ExactCorrelation::BothConstant,
            Self::OneConstant => ExactCorrelation::OneConstant,
            Self::Valid {
                n,
                row_variance,
                neighbor_variance,
            } => {
                let (row_variance, neighbor_variance) =
                    (row_variance.below_2_63()?, neighbor_variance.below_2_63()?);
                let magnitude = u128::from(n.magnitude().low_u64());
                ExactCorrelation::Valid {
                    sign: self.sign(),
                    cross_squared: magnitude * magnitude,
                    row_variance,
                    neighbor_variance,
                    denominator: variance_product(row_variance, neighbor_variance),
                }
            }
        })
    }

    fn sign(&self) -> i8 {
        match self {
            Self::Valid { n, .. } if n.is_zero() => 0,
            Self::Valid { n, .. } if n.is_negative() => -1,
            _ => 1,
        }
    }

    /// `(numerator, denominator)` of `ρ²`, exact on 512 bits.
    fn ratio(&self) -> (Wide<8>, Wide<8>) {
        match self {
            Self::BothConstant => (Wide::from_u64(1), Wide::from_u64(1)),
            Self::OneConstant => (Wide::from_u64(1), Wide::from_u64(4)),
            Self::Valid {
                n,
                row_variance,
                neighbor_variance,
            } => {
                let magnitude = n.magnitude();
                (
                    magnitude.wrapping_mul(&magnitude),
                    row_variance
                        .magnitude()
                        .wrapping_mul(&neighbor_variance.magnitude()),
                )
            }
        }
    }

    fn cmp_magnitude(&self, other: &Self) -> Ordering {
        if let (
            Self::Valid {
                n: n_a,
                neighbor_variance: v_a,
                ..
            },
            Self::Valid {
                n: n_b,
                neighbor_variance: v_b,
                ..
            },
        ) = (self, other)
        {
            // One row: its variance cancels from both ratios.
            let (a, b) = (n_a.magnitude(), n_b.magnitude());
            let left = a.wrapping_mul(&a).wrapping_mul(&v_b.magnitude());
            let right = b.wrapping_mul(&b).wrapping_mul(&v_a.magnitude());
            return left.cmp(&right);
        }
        let ((num_a, den_a), (num_b, den_b)) = (self.ratio(), other.ratio());
        num_a.wrapping_mul(&den_b).cmp(&num_b.wrapping_mul(&den_a))
    }
}

impl<A: Arith> Ranked for Spilled<A> {
    fn cmp_in_row(self, other: Self) -> Ordering {
        let (sign_a, sign_b) = (self.sign(), other.sign());
        if sign_a != sign_b {
            return sign_a.cmp(&sign_b);
        }
        if sign_a == 0 {
            return Ordering::Equal;
        }
        let order = self.cmp_magnitude(&other);
        if sign_a > 0 { order } else { order.reverse() }
    }

    fn distance_parts(self) -> Option<DistanceParts> {
        let Self::Valid {
            n,
            row_variance,
            neighbor_variance,
        } = self
        else {
            return None;
        };
        let magnitude = n.magnitude();
        let product = row_variance
            .magnitude()
            .wrapping_mul(&neighbor_variance.magnitude());
        Some(DistanceParts {
            numerator: product
                .wrapping_sub(&magnitude.wrapping_mul(&magnitude))
                .to_f64(),
            row_inv: 1.0 / libm::sqrt(row_variance.to_f64()),
            neighbor_inv: 1.0 / libm::sqrt(neighbor_variance.to_f64()),
        })
    }
}

/// One series' exact ticks and the exact statistics of its windows.
struct Side<'a, A> {
    ticks: Cow<'a, [i64]>,
    windows: Vec<ColdWindow<A>>,
}

impl<'a, A: Arith> Side<'a, A> {
    /// `had[p]` is false for a tick that was filled, not measured.
    fn new(ticks: Cow<'a, [i64]>, had: &[bool], m: usize) -> Result<Self, CapacityError> {
        let count = (ticks.len() + 1).saturating_sub(m);
        let mut windows = reserved_vec::<ColdWindow<A>>(count as u128)?;
        let (mut sum, mut sum_sq, mut missing) = (A::ZERO, A::ZERO, 0usize);
        let scale = A::from_u64(m as u64);
        for (position, &tick) in ticks.iter().enumerate() {
            sum = sum.add(A::from_i64(tick));
            sum_sq = sum_sq.add(A::product(tick, tick));
            missing += usize::from(!had[position]);
            if position >= m {
                let departing = ticks[position - m];
                sum = sum.sub(A::from_i64(departing));
                sum_sq = sum_sq.sub(A::product(departing, departing));
                missing -= usize::from(!had[position - m]);
            }
            if position + 1 < m {
                continue;
            }
            let variance = scale.mul(sum_sq).sub(sum.mul(sum));
            windows.push(ColdWindow::classify(missing > 0, sum, variance, None));
        }
        Ok(Self { ticks, windows })
    }

    /// Takes each present window's constancy from the caller's flag for it: a flagged window is
    /// constant, and an unflagged window whose ticks are all equal correlates with nothing (zero,
    /// as the hot passes rank it).
    fn flag(&mut self, flags: &[bool]) {
        for (window, &flagged) in self.windows.iter_mut().zip(flags) {
            window.kind = match (window.kind, flagged) {
                (Kind::Missing, _) => Kind::Missing,
                (_, true) => Kind::Constant,
                (Kind::Constant, false) => {
                    window.inv_sqrt = 1.0;
                    Kind::Valid
                }
                (kind, false) => kind,
            };
        }
    }
}

/// The exact ticks of the series a pass ranks, and of the series its neighbors come from.
struct Truth<'a, A> {
    m: usize,
    rows: Side<'a, A>,
    /// The neighbors' series, when it is not the rows' own (a join).
    cols: Option<Side<'a, A>>,
}

impl<'a, A: Arith> Truth<'a, A> {
    fn new(
        ticks: impl Into<Cow<'a, [i64]>>,
        had: &[bool],
        m: usize,
    ) -> Result<Self, CapacityError> {
        Ok(Self {
            m,
            rows: Side::new(ticks.into(), had, m)?,
            cols: None,
        })
    }

    fn join(
        (ticks, had): (impl Into<Cow<'a, [i64]>>, &[bool]),
        (col_ticks, col_had): (impl Into<Cow<'a, [i64]>>, &[bool]),
        m: usize,
    ) -> Result<Self, CapacityError> {
        Ok(Self {
            m,
            rows: Side::new(ticks.into(), had, m)?,
            cols: Some(Side::new(col_ticks.into(), col_had, m)?),
        })
    }

    /// The series the neighbors come from.
    fn cols(&self) -> &Side<'a, A> {
        self.cols.as_ref().unwrap_or(&self.rows)
    }

    /// `Σ rows[i+k]·cols[j+k]`, exactly.
    fn dot(&self, i: usize, j: usize) -> A {
        let m = self.m;
        dot(&self.rows.ticks[i..i + m], &self.cols().ticks[j..j + m])
    }

    /// Row `i`'s candidate for neighbor `j` with dot product `qt`, from the windows of the rows'
    /// series and of the neighbors' series, unless a window is missing or `skip` says its
    /// estimate is clearly out of the running.
    fn candidate(
        &self,
        (rows, cols): (&[ColdWindow<A>], &[ColdWindow<A>]),
        (i, j): (usize, usize),
        qt: A,
        skip: impl FnOnce(f64) -> bool,
    ) -> Option<Candidate<Spilled<A>>> {
        cold_candidate(self.m, (&rows[i], &cols[j]), qt, j as u64, skip)
    }

    /// The exact candidate of one pair, computed from scratch (`m` multiplies).
    fn pair(&self, i: usize, j: usize) -> Option<Candidate<Spilled<A>>> {
        let windows = (&self.rows.windows[..], &self.cols().windows[..]);
        self.candidate(windows, (i, j), self.dot(i, j), |_| false)
    }
}

/// Maximal runs `(first, last)` of consecutive `true` flags.
fn runs(flags: &[bool]) -> Result<Vec<(usize, usize)>, CapacityError> {
    let mut out = Vec::new();
    let mut start = None;
    for (position, &flag) in flags.iter().enumerate() {
        match (flag, start) {
            (true, None) => start = Some(position),
            (false, Some(first)) => {
                out.try_reserve(1).map_err(|_| CapacityError)?;
                out.push((first, position - 1));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(first) = start {
        out.try_reserve(1).map_err(|_| CapacityError)?;
        out.push((first, flags.len() - 1));
    }
    Ok(out)
}

/// One series' outliers and spilled windows, with the runs and counts a pass reads.
struct Flagged {
    outliers: Vec<bool>,
    spilled: Vec<bool>,
    runs: Vec<(usize, usize)>,
    /// `before[i]`: the spilled windows among the first `i`.
    before: Vec<usize>,
}

impl Flagged {
    fn new(outliers: Vec<bool>, spilled: Vec<bool>) -> Result<Self, CapacityError> {
        let mut flagged = Self {
            outliers,
            spilled,
            runs: Vec::new(),
            before: Vec::new(),
        };
        flagged.index()?;
        Ok(flagged)
    }

    /// Recomputes the runs and counts from the flags.
    fn index(&mut self) -> Result<(), CapacityError> {
        let mut before = reserved_vec::<usize>(self.spilled.len() as u128 + 1)?;
        let mut count = 0;
        before.push(0);
        for &flag in &self.spilled {
            count += usize::from(flag);
            before.push(count);
        }
        self.runs = runs(&self.spilled)?;
        self.before = before;
        Ok(())
    }

    fn marks(&self) -> Marks<'_> {
        Marks {
            flags: &self.spilled,
            runs: &self.runs,
            before: &self.before,
        }
    }

    /// The same flags, without the outliers (a pass reads only the spilled windows).
    fn restricted(&self) -> Self {
        Self {
            outliers: Vec::new(),
            spilled: self.spilled.clone(),
            runs: self.runs.clone(),
            before: self.before.clone(),
        }
    }

    /// Keeps as spilled only the windows `keep` accepts.
    fn keep(&mut self, keep: impl Fn(usize) -> bool) -> Result<(), CapacityError> {
        for (window, flag) in self.spilled.iter_mut().enumerate() {
            *flag = *flag && keep(window);
        }
        self.index()
    }
}

enum Tier<'a> {
    Narrow(Truth<'a, i128>),
    Wide(Truth<'a, Wide<4>>),
}

/// A series' outliers and the windows the cold route ranks, with the exact statistics to rank
/// them from; or, for a join, those of both series.
pub struct Survey<'a> {
    tier: Tier<'a>,
    rows: Flagged,
    /// The neighbors' series of a join.
    cols: Option<Flagged>,
    m: usize,
}

/// A series' ticks, borrowed or owned, and which of them were measured.
type Measured<'a, 'b> = (Cow<'a, [i64]>, &'b [bool]);

/// The exact statistics at one tier, and each series' spilled windows.
type Flagging<'a, A> = (Truth<'a, A>, Flagged, Option<Flagged>);

impl<'a> Survey<'a> {
    /// Surveys `ticks` (`had[p]` false for a filled tick) at window length `m`. `Ok(None)` when
    /// the series is outside the cold route's exact domain ([`width_for`]).
    pub fn new(
        ticks: &'a [i64],
        had: &[bool],
        m: NonZeroUsize,
    ) -> Result<Option<Self>, CapacityError> {
        assert_eq!(ticks.len(), had.len(), "one measured flag per tick");
        Self::build((Cow::Borrowed(ticks), had), None, m.get(), None)
    }

    /// [`Self::new`] on ticks it keeps, so the survey outlives the caller's borrow.
    pub(crate) fn owned(
        ticks: Vec<i64>,
        had: &[bool],
        m: NonZeroUsize,
    ) -> Result<Option<Survey<'static>>, CapacityError> {
        assert_eq!(ticks.len(), had.len(), "one measured flag per tick");
        Survey::build((Cow::Owned(ticks), had), None, m.get(), None)
    }

    /// [`Self::new`] with the cold route's windows chosen by the caller instead of found from the
    /// data, for tests and measurement: no tick is an outlier, `spilled[i]` says whether window
    /// `i` is ranked cold.
    pub fn with_spilled(
        ticks: &'a [i64],
        had: &[bool],
        m: NonZeroUsize,
        spilled: Vec<bool>,
    ) -> Result<Option<Self>, CapacityError> {
        assert_eq!(ticks.len(), had.len(), "one measured flag per tick");
        assert_eq!(
            spilled.len(),
            (ticks.len() + 1).saturating_sub(m.get()),
            "one flag per window"
        );
        Self::build((Cow::Borrowed(ticks), had), None, m.get(), Some(spilled))
    }

    /// [`Self::join_owned`] on borrowed ticks.
    #[cfg(test)]
    pub(crate) fn join(
        rows: (&'a [i64], &[bool]),
        cols: (&'a [i64], &[bool]),
        m: NonZeroUsize,
    ) -> Result<Option<Self>, CapacityError> {
        let borrowed = |(ticks, had)| (Cow::Borrowed(ticks), had);
        Self::join_of(borrowed(rows), borrowed(cols), m)
    }

    /// Surveys a join of the windows of `rows` against those of `cols`, the exact domain being
    /// that of both, keeping their ticks. `Ok(None)` when either is outside it.
    pub(crate) fn join_owned(
        (rows, row_had): (Vec<i64>, &[bool]),
        (cols, col_had): (Vec<i64>, &[bool]),
        m: NonZeroUsize,
    ) -> Result<Option<Survey<'static>>, CapacityError> {
        Survey::join_of((Cow::Owned(rows), row_had), (Cow::Owned(cols), col_had), m)
    }

    fn join_of(
        rows: Measured<'a, '_>,
        cols: Measured<'a, '_>,
        m: NonZeroUsize,
    ) -> Result<Option<Self>, CapacityError> {
        assert_eq!(rows.0.len(), rows.1.len(), "one measured flag per tick");
        assert_eq!(cols.0.len(), cols.1.len(), "one measured flag per tick");
        Self::build(rows, Some(cols), m.get(), None)
    }

    fn build(
        rows: Measured<'a, '_>,
        cols: Option<Measured<'a, '_>>,
        m: usize,
        chosen: Option<Vec<bool>>,
    ) -> Result<Option<Self>, CapacityError> {
        let width = match (
            width_for(&rows.0, m),
            cols.as_ref().map(|cols| width_for(&cols.0, m)),
        ) {
            (None, _) | (_, Some(None)) => return Ok(None),
            (Some(width), None) => width,
            (Some(width), Some(Some(other))) => width.max(other),
        };
        Ok(Some(match width {
            Width::Narrow128 => {
                let (truth, rows, cols) = Self::flag_windows::<i128>(rows, cols, m, chosen)?;
                Self {
                    tier: Tier::Narrow(truth),
                    rows,
                    cols,
                    m,
                }
            }
            Width::Wide256 => {
                let (truth, rows, cols) = Self::flag_windows::<Wide<4>>(rows, cols, m, chosen)?;
                Self {
                    tier: Tier::Wide(truth),
                    rows,
                    cols,
                    m,
                }
            }
        }))
    }

    /// The exact statistics at one tier, and which windows of each series are spilled.
    fn flag_windows<A: Arith>(
        rows: Measured<'a, '_>,
        cols: Option<Measured<'a, '_>>,
        m: usize,
        chosen: Option<Vec<bool>>,
    ) -> Result<Flagging<'a, A>, CapacityError> {
        let (row_had, col_had) = (rows.1, cols.as_ref().map(|cols| cols.1));
        let truth = match cols {
            None => Truth::new(rows.0, rows.1, m)?,
            Some(cols) => Truth::join(rows, cols, m)?,
        };
        let found = |side: &Side<A>, had: &[bool]| {
            let outliers = outlier_ticks(&side.ticks, had, m)?;
            let spilled = spilled_windows(side, m, &outliers)?;
            Flagged::new(outliers, spilled)
        };
        let row_flags = match chosen {
            Some(spilled) => Flagged::new(cleared(truth.rows.ticks.len())?, spilled)?,
            None => found(&truth.rows, row_had)?,
        };
        let col_flags = match (&truth.cols, col_had) {
            (Some(side), Some(had)) => Some(found(side, had)?),
            _ => None,
        };
        Ok((truth, row_flags, col_flags))
    }

    /// Takes each present window's constancy from the caller's flag for it (`None`: the default
    /// rule, a window is constant when its ticks are equal), as the hot passes do; `cols` for a
    /// join's neighbors.
    pub(crate) fn flagged(mut self, rows: Option<&[bool]>, cols: Option<&[bool]>) -> Self {
        match &mut self.tier {
            Tier::Narrow(truth) => flag_truth(truth, rows, cols),
            Tier::Wide(truth) => flag_truth(truth, rows, cols),
        }
        self
    }

    /// Keeps as spilled only the rows' windows `rows` accepts and the neighbors' windows `cols`
    /// accepts (a self-join has one series, judged by `rows`).
    pub(crate) fn keep_spilled_where(
        &mut self,
        rows: impl Fn(usize) -> bool,
        cols: impl Fn(usize) -> bool,
    ) -> Result<(), CapacityError> {
        self.rows.keep(rows)?;
        if let Some(flagged) = &mut self.cols {
            flagged.keep(cols)?;
        }
        Ok(())
    }

    /// Whether this is a join of two series.
    pub(crate) fn is_join(&self) -> bool {
        self.cols.is_some()
    }

    /// Whether any window is ranked by the cold route.
    pub(crate) fn any_spilled(&self) -> bool {
        !self.rows.runs.is_empty() || self.cols.as_ref().is_some_and(|cols| !cols.runs.is_empty())
    }

    /// The rows whose lists a pass of `reach` can change: a spilled row, or a row with a spilled
    /// neighbor to offer. Any other row keeps what the hot passes found.
    pub(crate) fn affected_rows(&self, reach: Reach) -> Range<usize> {
        let cover = self.cover(reach);
        let rows = cover.rows.flags.len() as isize;
        let (mut first, mut end) = (isize::MAX, 0);
        if let (Some(&(start, _)), Some(&(_, last))) =
            (cover.rows.runs.first(), cover.rows.runs.last())
        {
            (first, end) = (start as isize, last as isize + 1);
        }
        if let (Some(&(start, _)), Some(&(_, last))) =
            (cover.cols.runs.first(), cover.cols.runs.last())
        {
            let reached = (start as isize - *cover.offsets.end()).max(0)
                ..(last as isize - *cover.offsets.start() + 1).min(rows);
            if reached.start < reached.end {
                (first, end) = (first.min(reached.start), end.max(reached.end));
            }
        }
        if first < end {
            first as usize..end as usize
        } else {
            0..0
        }
    }

    /// The most rows of one band at `k` neighbors per row, so that a band's candidates stay
    /// within [`BAND_BYTES`].
    pub(crate) fn band_limit(&self, k: usize) -> usize {
        let candidate = match &self.tier {
            Tier::Narrow(_) => core::mem::size_of::<Candidate<Spilled<i128>>>(),
            Tier::Wide(_) => core::mem::size_of::<Candidate<Spilled<Wide<4>>>>(),
        };
        let per_row = candidate
            .saturating_mul(k)
            .saturating_add(2 * core::mem::size_of::<usize>());
        (BAND_BYTES / per_row).max(1)
    }

    pub fn width(&self) -> Width {
        match self.tier {
            Tier::Narrow(_) => Width::Narrow128,
            Tier::Wide(_) => Width::Wide256,
        }
    }

    /// `outliers()[p]`: whether tick `p` is an outlier.
    pub fn outliers(&self) -> &[bool] {
        &self.rows.outliers
    }

    /// `spilled()[i]`: whether window `i` is ranked by the cold route: every sample present, and
    /// it holds an outlier or its variance numerator is at least `2⁶³`.
    pub fn spilled(&self) -> &[bool] {
        &self.rows.spilled
    }

    /// A join's neighbors' outliers (the same as [`Self::outliers`] for one series).
    pub(crate) fn col_outliers(&self) -> &[bool] {
        &self.cols.as_ref().unwrap_or(&self.rows).outliers
    }

    /// A join's neighbors' spilled windows (the same as [`Self::spilled`] for one series).
    pub(crate) fn col_spilled(&self) -> &[bool] {
        &self.cols.as_ref().unwrap_or(&self.rows).spilled
    }

    /// The pairs `reach` offers.
    fn cover(&self, reach: Reach) -> Cover<'_> {
        self.cover_of(&self.rows, self.cols.as_ref(), reach)
    }

    /// The pairs `reach` offers over `rows`' and `cols`' spilled windows (`cols` for a join).
    fn cover_of<'b>(
        &self,
        rows: &'b Flagged,
        cols: Option<&'b Flagged>,
        reach: Reach,
    ) -> Cover<'b> {
        let rows = rows.marks();
        let cols = cols.map_or(rows, Flagged::marks);
        let exclusion = crate::exclusion::zone(self.m) as isize;
        let (rows_n, cols_n) = (rows.flags.len() as isize, cols.flags.len() as isize);
        // A reach the survey does not offer has no offsets.
        let (lowest, highest) = match (reach, self.is_join()) {
            (Reach::Left, false) => (-(rows_n - 1), -(exclusion + 1)),
            (Reach::Right, false) => (exclusion + 1, rows_n - 1),
            (Reach::Join, true) => (-(rows_n - 1), cols_n - 1),
            _ => (1, 0),
        };
        Cover {
            rows,
            cols,
            offsets: lowest..=highest,
        }
    }

    /// The pairs and chains the passes of `reaches` walk together.
    pub(crate) fn walk_over(&self, reaches: &[Reach]) -> Counts {
        let mut counts = Counts::default();
        for &reach in reaches {
            let cover = self.cover(reach);
            counts += cover.count(0..cover.rows.flags.len());
        }
        counts
    }

    /// The pairs and chains the passes of `reaches` walk if only the windows `rows` accepts (and,
    /// for a join, the neighbors' windows `cols` accepts) of those spilled are.
    pub(crate) fn walk_where(
        &self,
        reaches: &[Reach],
        rows: impl Fn(usize) -> bool,
        cols: impl Fn(usize) -> bool,
    ) -> Result<Counts, CapacityError> {
        let mut kept_rows = self.rows.restricted();
        kept_rows.keep(rows)?;
        let kept_cols = match &self.cols {
            Some(flagged) => {
                let mut restricted = flagged.restricted();
                restricted.keep(cols)?;
                Some(restricted)
            }
            None => None,
        };
        let mut counts = Counts::default();
        for &reach in reaches {
            let cover = self.cover_of(&kept_rows, kept_cols.as_ref(), reach);
            counts += cover.count(0..cover.rows.flags.len());
        }
        Ok(counts)
    }

    /// The pairs and chains a cold pass walks: a self-join's left profile, or a join.
    pub fn walk(&self) -> Result<Counts, CapacityError> {
        Ok(self.walk_over(&[self.default_reach()]))
    }

    fn default_reach(&self) -> Reach {
        if self.cols.is_some() {
            Reach::Join
        } else {
            Reach::Left
        }
    }

    /// Appends the final lists of rows `band` to `out`: for each row, the `k` nearest, nearest
    /// first, of its entries in `hot` (what the hot passes produced for `reach` with every spilled
    /// window non-finite) and of its pairs with a spilled window; one empty entry for a row with
    /// none. A left or right pass offers the neighbors outside the exclusion zone (`m/4` rounded
    /// up), a join every window of the other series. Adds what was walked to `counts`.
    pub(crate) fn merge_rows(
        &self,
        reach: Reach,
        band: Range<usize>,
        k: NonZeroUsize,
        hot: &HotRuns,
        out: &mut Vec<(u64, Entry)>,
        counts: &mut Counts,
    ) -> Result<(), CapacityError> {
        let cover = self.cover(reach);
        match &self.tier {
            Tier::Narrow(truth) => truth.merge_band(&cover, band, k.get(), hot, out, counts),
            Tier::Wide(truth) => truth.merge_band(&cover, band, k.get(), hot, out, counts),
        }
    }

    /// Whether any of row `row`'s window and the neighbors `x` and `y` is spilled.
    pub(crate) fn touches(&self, row: usize, x: usize, y: usize) -> bool {
        let neighbors = &self.cols.as_ref().unwrap_or(&self.rows).spilled;
        self.rows.spilled[row] || neighbors[x] || neighbors[y]
    }

    /// Whether neighbor `x` is exactly nearer to row `row` than neighbor `y` is, ties to the
    /// smaller index.
    pub(crate) fn nearer_neighbor(&self, row: usize, x: usize, y: usize) -> bool {
        match &self.tier {
            Tier::Narrow(truth) => truth.nearer_neighbor(row, x, y),
            Tier::Wide(truth) => truth.nearer_neighbor(row, x, y),
        }
    }

    /// Folds the cold route into the left profile `results` (one `(row, entry)` per window, in
    /// row order) that the hot passes produced with every spilled window non-finite: each entry
    /// becomes the exact nearest over the left neighbors outside the exclusion zone (`m/4`
    /// rounded up). Returns what was walked.
    pub fn merge_left(&self, results: &mut [(u64, Entry)]) -> Result<Counts, CapacityError> {
        assert_eq!(
            results.len(),
            self.rows.spilled.len(),
            "one result per window"
        );
        let mut counts = Counts::default();
        if self.any_spilled() {
            let mut merged = reserved_vec(results.len() as u128)?;
            let hot = HotRuns::new(results, results.len())?;
            let band = 0..results.len();
            let k = NonZeroUsize::MIN;
            self.merge_rows(Reach::Left, band, k, &hot, &mut merged, &mut counts)?;
            results.copy_from_slice(&merged);
        }
        Ok(counts)
    }
}

/// Takes the windows' constancy from the caller's flags.
fn flag_truth<A: Arith>(truth: &mut Truth<A>, rows: Option<&[bool]>, cols: Option<&[bool]>) {
    if let Some(flags) = rows {
        truth.rows.flag(flags);
    }
    if let (Some(side), Some(flags)) = (&mut truth.cols, cols) {
        side.flag(flags);
    }
}

/// Windows with every sample present that hold an outlier, or whose variance numerator is at
/// least `2⁶³`.
fn spilled_windows<A: Arith>(
    side: &Side<A>,
    m: usize,
    outliers: &[bool],
) -> Result<Vec<bool>, CapacityError> {
    let windows = side.windows.len();
    let mut spilled = cleared(windows)?;
    let mut next_outlier = usize::MAX;
    let limit = Wide::<8>::from_u64(EXACT_VARIANCE_LIMIT);
    for position in (0..side.ticks.len()).rev() {
        if outliers[position] {
            next_outlier = position;
        }
        if position < windows {
            let window = &side.windows[position];
            let holds_outlier = next_outlier != usize::MAX && next_outlier - position < m;
            let too_wide = window.kind == Kind::Valid && window.variance.magnitude() >= limit;
            spilled[position] = window.kind != Kind::Missing && (holds_outlier || too_wide);
        }
    }
    Ok(spilled)
}

/// What a run needs of the cold route, and what today's routes would spend instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub width: Width,
    pub m: usize,
    /// What the cold route walks when it takes every spilled window.
    pub walk: Counts,
    /// What it walks whatever the model says: the pairs of the spilled windows that have no
    /// distance without it (a part of `walk`). Taking the rest adds `walk` less `required`.
    pub required: Counts,
    /// Pairs today's routes run on the exact i64 pass (their tiles that do not fit one anchor);
    /// zero when everything fits.
    pub wide_cells_today: u64,
}

/// The cost model's verdict.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    /// Whether the cold route runs, after any forced route.
    pub engage: bool,
    /// What the model says on its own.
    pub natural: bool,
    /// What taking every spilled window adds to the cold walk the data requires anyway.
    pub cold_cost: u64,
    pub saved_cost: u64,
}

impl Survey<'_> {
    /// The cost model's input for a left profile, or for a join.
    pub fn shape(&self, wide_cells_today: u64) -> Result<Shape, CapacityError> {
        let required = Counts::default();
        self.shape_over(&[self.default_reach()], wide_cells_today, required)
    }

    /// The cost model's input for the passes of `reaches` together, when `required` is what the
    /// cold route walks whatever the model says.
    pub(crate) fn shape_over(
        &self,
        reaches: &[Reach],
        wide_cells_today: u64,
        required: Counts,
    ) -> Result<Shape, CapacityError> {
        Ok(Shape {
            width: self.width(),
            m: self.m,
            walk: self.walk_over(reaches),
            required,
            wide_cells_today,
        })
    }
}

#[cfg(test)]
pub(crate) mod tests;
