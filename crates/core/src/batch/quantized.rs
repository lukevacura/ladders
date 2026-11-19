use super::plan::{Choice, Engine, RunShape, SlotRuns, survivor_filter};
use super::survivor::Filter;
use crate::bound_paa::{Coefficients, Segments};
use crate::capacity::{CapacityError, clamp_count};
use crate::columns::CrossRange;
use crate::estimate::{F64, Producer, band_floor};
use crate::narrow::{self, Anchor};
use crate::quantize::{Scale, to_ticks};
use crate::row::{Columns, push_window};
use crate::sanitize::{ForwardFill, Sanitized};
use crate::stats::{ConstantClass, Series, Window, WindowKind, WindowStats};
use alloc::boxed::Box;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

/// The windows of one block of a threaded build: a block starts from the last tick that quantized
/// before it and reads `m - 1` ticks past it, so it is at least twice `m`.
#[cfg(all(feature = "parallel", not(loom)))]
const QUANTIZE_BLOCK: usize = 1024;

/// `series` quantized and forward-filled into sanitized ticks, and every completed window's
/// statistics in [`Columns`]'s own struct-of-arrays layout, oldest first, with the bound on every
/// cross term between those windows ([`CrossRange`]), the survivor filter the run installs and,
/// when that filter reads them, every window's PAA coefficients ([`Coefficients`]), both kept
/// current with the ticks.
pub(crate) struct Quantized {
    pub(super) ticks: Vec<i64>,
    pub(super) sum: Vec<i64>,
    pub(super) r: Vec<f64>,
    pub(super) bias: Vec<f64>,
    pub(super) variance: Vec<u64>,
    pub(super) windows: Vec<Window>,
    pub(super) range: CrossRange,
    /// The survivor filter the run installs, as installed from [`Self::choice`] with the PAA
    /// coefficients; every frontier of the run reads it from here.
    pub(super) filter: Filter,
    /// The run's survivor filter as decided once from its shape before the build
    /// ([`survivor_filter`]), kept so the coefficients taken again over new ticks install the
    /// same choice.
    choice: Choice,
    /// Whether the run reports only its best pair ([`Engine::BestPair`]).
    best_pair: bool,
    /// `None` unless the run's filter reads PAA coefficients and their proof covers the series.
    pub(super) paa: Option<Box<Coefficients>>,
    /// The constant windows of a self-join's windows as one class, kept current with them;
    /// `None` for any other run, or when the class settles nothing.
    class: Option<Box<ConstantClass>>,
}

impl Quantized {
    /// The series of a run the gate leaves off ([`Engine::Ungated`]: a pass over chosen
    /// diagonals, a walk), on one thread.
    pub(crate) fn build<'a>(series: impl Into<Series<'a>>, m: NonZeroUsize, scale: Scale) -> Self {
        Self::build_on(series, m, scale, NonZeroUsize::MIN, Engine::Ungated)
    }

    /// [`Self::build`] for a run of `engine` on up to `threads` threads, built on them when the
    /// series has windows enough: blocks of windows are built from their own start and joined in
    /// order, so the result is the same for any `threads`.
    pub(crate) fn build_on<'a>(
        series: impl Into<Series<'a>>,
        m: NonZeroUsize,
        scale: Scale,
        threads: NonZeroUsize,
        engine: Engine,
    ) -> Self {
        let choice = survivor_filter(RunShape {
            m: m.get(),
            engine,
            threads: threads.get(),
        });
        let mut built = Self::build_windows_on(series.into(), m, scale, threads);
        built.range = CrossRange::over(&built.ticks, m);
        built.choice = choice;
        built.best_pair = engine == Engine::BestPair;
        built.install_filter(m);
        if !matches!(engine, Engine::Ungated | Engine::Join | Engine::JoinTopK) {
            built.class =
                ConstantClass::of(&built.windows, crate::exclusion::zone(m.get())).map(Box::new);
        }
        built
    }

    /// Where the global floor of a run that offers every pair to a row keeping one neighbor
    /// (`offers_all`) starts, `None` when it keeps none: a best-pair run keeps one, open unless the
    /// constant class holds two windows outside each other's zone, whose pair is a correlation of
    /// one, and unless the route harness forces it off.
    pub(super) fn global_floor(&self, offers_all: bool) -> Option<f64> {
        if !(self.best_pair && offers_all) || crate::route::no_global_floor() {
            return None;
        }
        let known = self.class.as_deref().and_then(ConstantClass::top_pair);
        if known.is_none() {
            return Some(f64::NEG_INFINITY);
        }
        crate::route::count_global_seed();
        Some(band_floor(1.0, F64::DELTA))
    }

    /// The constant class of a self-join that offers every pair; `None` with `runs`, whose pairs
    /// need not include the class's earliest members.
    pub(super) fn constant_class(&self, runs: Option<&SlotRuns>) -> Option<&ConstantClass> {
        self.class.as_deref().filter(|_| runs.is_none())
    }

    /// Installs the run's filter over the current ticks and windows, with the PAA coefficients
    /// when it reads them ([`Choice::install`]).
    fn install_filter(&mut self, m: NonZeroUsize) {
        let (filter, paa) = self.choice.install(|| self.coefficients(m));
        self.filter = filter;
        self.paa = paa;
    }

    /// The PAA coefficients of the current ticks and windows, `None` where they cannot be built.
    fn coefficients(&self, m: NonZeroUsize) -> Option<Box<Coefficients>> {
        if crate::route::paa_build_fails() {
            return None;
        }
        let built = Coefficients::over(
            &self.ticks,
            (&self.sum, &self.r, &self.bias),
            m.get(),
            Segments::for_window(m.get()),
        )?;
        crate::route::count_paa_table();
        Some(Box::new(built))
    }

    /// [`Self::build_on`] without the cross-term bound.
    fn build_windows_on(
        series: Series<'_>,
        m: NonZeroUsize,
        scale: Scale,
        threads: NonZeroUsize,
    ) -> Self {
        let windows = (series.values().len() + 1).saturating_sub(m.get());
        #[cfg(all(feature = "parallel", not(loom)))]
        {
            let block = QUANTIZE_BLOCK.max(m.get().saturating_mul(2));
            let blocks = windows.div_ceil(block);
            if threads.get() > 1 && blocks > 1 {
                let parts = crate::par::for_each_chunk(threads, blocks, |b| {
                    let span = b * block..((b + 1) * block).min(windows);
                    Self::build_span(series, m, scale, span, windows)
                });
                return Self::join(parts);
            }
        }
        let _ = threads;
        Self::build_span(series, m, scale, 0..windows, windows)
    }

    /// The ticks and windows `span` of the `windows` windows own, started from the last tick
    /// that quantized before the span. The last span owns the ticks to the end of the series.
    fn build_span(
        series: Series<'_>,
        m: NonZeroUsize,
        scale: Scale,
        span: core::ops::Range<usize>,
        windows: usize,
    ) -> Self {
        let values = series.values();
        let mut fill = ForwardFill::<i64>::default();
        if let Some(tick) = values[..span.start]
            .iter()
            .rev()
            .find_map(|&x| to_ticks(x, scale))
        {
            let _ = fill.push(Some(tick));
        }
        let mut stats = WindowStats::new(m);
        let owned_end = if span.end == windows {
            values.len()
        } else {
            span.end
        };
        let read_end = if span.end == windows {
            values.len()
        } else {
            span.end + m.get() - 1
        };
        let mut ticks = Vec::with_capacity(owned_end - span.start);
        let mut sum = Vec::with_capacity(span.len());
        let mut r = Vec::with_capacity(span.len());
        let mut bias = Vec::with_capacity(span.len());
        let mut variance = Vec::with_capacity(span.len());
        let mut kept = Vec::with_capacity(span.len());

        for (p, &x) in values[span.start..read_end].iter().enumerate() {
            let (tick, had_tick) = match fill.push(to_ticks(x, scale)) {
                Sanitized::Present(t) => (t, true),
                Sanitized::Filled(t) => (t, false),
                Sanitized::Leading => (0, false),
            };
            if span.start + p < owned_end {
                ticks.push(tick);
            }
            if let Some(window) = stats.push(tick, had_tick) {
                let window = series.window(span.start + kept.len(), window, m);
                push_window(window, &mut sum, &mut r, &mut bias, &mut variance);
                kept.push(window);
            }
        }

        Self {
            ticks,
            sum,
            r,
            bias,
            variance,
            windows: kept,
            range: CrossRange::UNKNOWN,
            filter: Filter::Off,
            choice: Choice::default(),
            best_pair: false,
            paa: None,
            class: None,
        }
    }

    /// The spans' parts in series order as one.
    #[cfg(all(feature = "parallel", not(loom)))]
    fn join(parts: Vec<Self>) -> Self {
        fn joined<T: Copy>(parts: &[Quantized], of: impl Fn(&Quantized) -> &Vec<T>) -> Vec<T> {
            let mut all = Vec::with_capacity(parts.iter().map(|p| of(p).len()).sum());
            for part in parts {
                all.extend_from_slice(of(part));
            }
            all
        }
        Self {
            ticks: joined(&parts, |p| &p.ticks),
            sum: joined(&parts, |p| &p.sum),
            r: joined(&parts, |p| &p.r),
            bias: joined(&parts, |p| &p.bias),
            variance: joined(&parts, |p| &p.variance),
            windows: joined(&parts, |p| &p.windows),
            range: CrossRange::UNKNOWN,
            filter: Filter::Off,
            choice: Choice::default(),
            best_pair: false,
            paa: None,
            class: None,
        }
    }

    /// The ticks as the hot passes see them once the cold route takes over `outliers`: each outlier
    /// tick, and each filled tick that repeats one, becomes the last tick that is neither (the
    /// first such tick for a leading run), so the series can fit one anchor; and every window in
    /// `spilled` is non-finite. `had[p]` says whether tick `p` was measured. The cross-term bound
    /// is taken again over the new ticks, and the run's filter installed again over them with its
    /// PAA coefficients ([`Choice::install`]).
    pub(super) fn standing_in(
        &mut self,
        had: &[bool],
        outliers: &[bool],
        spilled: &[bool],
        m: NonZeroUsize,
    ) -> Result<(), CapacityError> {
        let first_clean = (0..self.ticks.len())
            .find(|&p| had[p] && !outliers[p])
            .map(|p| self.ticks[p]);
        let mut replaced = Vec::new();
        replaced
            .try_reserve_exact(self.ticks.len())
            .map_err(|_| CapacityError)?;
        let (mut last, mut seen) = (None, false);
        for (p, &tick) in self.ticks.iter().enumerate() {
            if had[p] && !outliers[p] {
                last = Some(tick);
            }
            let stand_in = last.or(first_clean).unwrap_or(0);
            replaced.push(if outliers[p] || (!had[p] && seen) {
                stand_in
            } else {
                tick
            });
            seen |= had[p];
        }
        let old = core::mem::replace(&mut self.ticks, replaced);
        let mu = m.get();
        let mut shift = 0i64;
        for p in 0..self.ticks.len() {
            shift = shift.wrapping_add(self.ticks[p].wrapping_sub(old[p]));
            if p >= mu {
                shift = shift.wrapping_sub(self.ticks[p - mu].wrapping_sub(old[p - mu]));
            }
            if p + 1 >= mu {
                let row = p + 1 - mu;
                self.sum[row] = self.sum[row].wrapping_add(shift);
                self.windows[row].sum = self.sum[row];
            }
        }
        for (row, _) in spilled.iter().enumerate().filter(|&(_, &spilled)| spilled) {
            self.windows[row].kind = WindowKind::NonFinite;
            let (r, bias, variance) = crate::columns::encode(&self.windows[row]);
            self.r[row] = r;
            self.bias[row] = bias;
            self.variance[row] = variance;
        }
        self.range = CrossRange::over(&self.ticks, m);
        self.install_filter(m);
        if self.class.is_some() {
            self.class =
                ConstantClass::of(&self.windows, crate::exclusion::zone(m.get())).map(Box::new);
        }
        Ok(())
    }

    /// The ticks, window sums and kinds `(r, bias)`, for tests outside the batch module.
    #[cfg(test)]
    pub(crate) fn parts(&self) -> (&[i64], &[i64], &[f64], &[f64]) {
        (&self.ticks, &self.sum, &self.r, &self.bias)
    }

    /// The windows that can be a neighbor: every one that is finite.
    pub(super) fn finite_windows(&self) -> usize {
        self.windows
            .iter()
            .filter(|w| !matches!(w.kind, WindowKind::NonFinite))
            .count()
    }

    /// A caller's `k` limited to what a row can keep: the finite windows (at least one, so the
    /// count stays a count). Reservations sized by it never exceed what the input can fill.
    pub(crate) fn clamp_k(&self, k: NonZeroUsize) -> NonZeroUsize {
        NonZeroUsize::new(clamp_count(k.get(), self.finite_windows())).unwrap_or(NonZeroUsize::MIN)
    }

    pub(super) fn columns(&self) -> Columns<'_> {
        Columns {
            ticks: &self.ticks,
            sum: &self.sum,
            r: &self.r,
            bias: &self.bias,
            variance: &self.variance,
        }
    }

    /// A single anchor covering every tick in the whole series, or `None` when the series' tick
    /// range is too wide for any one anchor (`None` as well when there are no ticks).
    pub(super) fn narrow_anchor(&self, m: NonZeroUsize) -> Option<Anchor> {
        let whole = Anchor::covering(&self.ticks, m)?;
        crate::route::keep_whole_series_anchor().then_some(whole)
    }

    /// Every tick and window sum, narrowed to `anchor` once for the whole series.
    pub(super) fn narrow_columns(&self, anchor: Anchor, m: NonZeroUsize) -> (Vec<i32>, Vec<i32>) {
        let (mut ticks, mut sums) = (Vec::new(), Vec::new());
        narrow::narrow_ticks_into(anchor, m, &self.ticks, &mut ticks);
        narrow::narrow_sums_into(anchor, m, &self.sum, &mut sums);
        (ticks, sums)
    }
}
