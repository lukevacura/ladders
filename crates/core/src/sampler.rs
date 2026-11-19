//! Market events become kernel samples here; ordering is enforced at this boundary, not
//! in the kernel.

use core::num::NonZeroU64;

/// One market event in, zero or more samples out, emitted via callback (no allocation).
pub trait Sampler {
    /// Feeds one event; `emit` is called once per sample this event produces, in order.
    fn on_event(
        &mut self,
        ts_ns: u64,
        price: f64,
        size: f64,
        emit: impl FnMut(f64),
    ) -> Result<(), SamplerError>;

    /// Caller-supplied heartbeat: closes any bars that end at or before `ts_ns`.
    fn advance_to(&mut self, ts_ns: u64, emit: impl FnMut(f64)) -> Result<(), SamplerError>;
}

/// A timestamp strictly earlier than one already given to [`Sampler::on_event`] or
/// [`Sampler::advance_to`]; equal timestamps are in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SamplerError {
    OutOfOrder { last_ns: u64, got_ns: u64 },
}

/// Marks a sampler whose samples land on fixed boundaries, independent of exactly when
/// events arrive within them. [`Ticks`] does not implement this: each event is its own
/// sample.
///
/// ```compile_fail
/// use ladde::sampler::{AlignedSampler, Ticks};
///
/// fn needs_aligned<S: AlignedSampler>(_sampler: S) {}
/// needs_aligned(Ticks::default());
/// ```
pub trait AlignedSampler: Sampler {}

/// Emits every event's price unchanged; rejects an event older than the last one seen.
#[derive(Clone, Copy, Debug, Default)]
pub struct Ticks {
    last_ns: Option<u64>,
}

impl Ticks {
    fn admit(&mut self, ts_ns: u64) -> Result<(), SamplerError> {
        if let Some(last_ns) = self.last_ns
            && ts_ns < last_ns
        {
            return Err(SamplerError::OutOfOrder {
                last_ns,
                got_ns: ts_ns,
            });
        }
        self.last_ns = Some(ts_ns);
        Ok(())
    }
}

impl Sampler for Ticks {
    fn on_event(
        &mut self,
        ts_ns: u64,
        price: f64,
        _size: f64,
        mut emit: impl FnMut(f64),
    ) -> Result<(), SamplerError> {
        self.admit(ts_ns)?;
        emit(price);
        Ok(())
    }

    fn advance_to(&mut self, ts_ns: u64, _emit: impl FnMut(f64)) -> Result<(), SamplerError> {
        self.admit(ts_ns)
    }
}

/// How a bar with no events in it is reported.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmptyBar {
    /// Repeats the last emitted value (`0.0` for [`BarField::LogReturn`]).
    ForwardFill,
    /// `NaN`.
    Gap,
}

/// The value a closed bar emits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BarField {
    /// The bar's last price.
    Close,
    /// `Σ(price·size) / Σsize`, or the bar's close when `Σsize == 0`.
    Vwap,
    /// `ln(close / previous close)`, relative to the last non-empty bar; nothing for the
    /// first one. A non-positive previous close yields NaN or ±inf, as `ln` does.
    LogReturn,
}

/// A bar's running totals while it is still open; `index` is `ts_ns / interval_ns`.
#[derive(Clone, Copy, Debug)]
struct OpenBar {
    index: u64,
    has_data: bool,
    close: f64,
    sum_price_size: f64,
    sum_size: f64,
}

impl OpenBar {
    fn empty(index: u64) -> Self {
        Self {
            index,
            has_data: false,
            close: 0.0,
            sum_price_size: 0.0,
            sum_size: 0.0,
        }
    }

    fn new(index: u64, price: f64, size: f64) -> Self {
        let mut bar = Self::empty(index);
        bar.push(price, size);
        bar
    }

    fn push(&mut self, price: f64, size: f64) {
        self.has_data = true;
        self.close = price;
        self.sum_price_size += price * size;
        self.sum_size += size;
    }
}

/// Bars of `interval_ns`, aligned to multiples of it from timestamp `0`. Nothing is
/// emitted before the first event.
#[derive(Clone, Copy, Debug)]
pub struct TimeBars {
    interval_ns: NonZeroU64,
    empty: EmptyBar,
    field: BarField,
    last_ns: Option<u64>,
    open: Option<OpenBar>,
    /// The last non-empty bar's close; untouched by empty bars, so a `LogReturn` after a
    /// `Gap` is still relative to the last real close.
    last_close: Option<f64>,
    /// The last non-empty bar's emitted value; `ForwardFill`'s fill for `Close`/`Vwap`.
    last_value: Option<f64>,
}

impl TimeBars {
    pub fn new(interval_ns: NonZeroU64, empty: EmptyBar, field: BarField) -> Self {
        Self {
            interval_ns,
            empty,
            field,
            last_ns: None,
            open: None,
            last_close: None,
            last_value: None,
        }
    }

    fn admit(&mut self, ts_ns: u64) -> Result<(), SamplerError> {
        if let Some(last_ns) = self.last_ns
            && ts_ns < last_ns
        {
            return Err(SamplerError::OutOfOrder {
                last_ns,
                got_ns: ts_ns,
            });
        }
        self.last_ns = Some(ts_ns);
        Ok(())
    }

    fn bar_index(&self, ts_ns: u64) -> u64 {
        ts_ns / self.interval_ns.get()
    }

    /// Closes `bar`, then fills every bar strictly between it and `target` under the
    /// empty-bar policy. `target` becomes the new open bar, left to the caller.
    fn roll(&mut self, bar: OpenBar, target: u64, emit: &mut impl FnMut(f64)) {
        self.close(bar, emit);
        for _ in (bar.index + 1)..target {
            self.fill_empty(emit);
        }
    }

    fn close(&mut self, bar: OpenBar, emit: &mut impl FnMut(f64)) {
        if !bar.has_data {
            self.fill_empty(emit);
            return;
        }
        match self.field {
            BarField::Close => {
                self.last_value = Some(bar.close);
                emit(bar.close);
            }
            BarField::Vwap => {
                let vwap = if bar.sum_size == 0.0 {
                    bar.close
                } else {
                    bar.sum_price_size / bar.sum_size
                };
                self.last_value = Some(vwap);
                emit(vwap);
            }
            BarField::LogReturn => {
                if let Some(previous_close) = self.last_close {
                    emit(libm::log(bar.close / previous_close));
                }
            }
        }
        self.last_close = Some(bar.close);
    }

    fn fill_empty(&mut self, emit: &mut impl FnMut(f64)) {
        match self.empty {
            EmptyBar::Gap => emit(f64::NAN),
            EmptyBar::ForwardFill => match self.field {
                BarField::LogReturn => emit(0.0),
                BarField::Close | BarField::Vwap => {
                    if let Some(value) = self.last_value {
                        emit(value);
                    }
                }
            },
        }
    }
}

impl Sampler for TimeBars {
    fn on_event(
        &mut self,
        ts_ns: u64,
        price: f64,
        size: f64,
        mut emit: impl FnMut(f64),
    ) -> Result<(), SamplerError> {
        self.admit(ts_ns)?;
        let index = self.bar_index(ts_ns);
        self.open = Some(match self.open.take() {
            Some(mut bar) if bar.index == index => {
                bar.push(price, size);
                bar
            }
            Some(bar) => {
                self.roll(bar, index, &mut emit);
                OpenBar::new(index, price, size)
            }
            None => OpenBar::new(index, price, size),
        });
        Ok(())
    }

    fn advance_to(&mut self, ts_ns: u64, mut emit: impl FnMut(f64)) -> Result<(), SamplerError> {
        self.admit(ts_ns)?;
        let Some(bar) = self.open.take() else {
            return Ok(());
        };
        let index = self.bar_index(ts_ns);
        if bar.index >= index {
            self.open = Some(bar);
            return Ok(());
        }
        self.roll(bar, index, &mut emit);
        self.open = Some(OpenBar::empty(index));
        Ok(())
    }
}

impl AlignedSampler for TimeBars {}

#[cfg(test)]
mod tests;

const _: fn() = || {
    fn needs_aligned<S: AlignedSampler>() {}
    needs_aligned::<TimeBars>();
};

#[cfg(test)]
mod timebars_tests;
