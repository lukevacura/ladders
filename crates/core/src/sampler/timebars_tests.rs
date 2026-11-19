extern crate std;

use super::{BarField, EmptyBar, Sampler, SamplerError, TimeBars};
use core::num::NonZeroU64;
use proptest::prelude::*;
use std::vec::Vec;

fn any_f64() -> impl Strategy<Value = f64> {
    any::<u64>().prop_map(f64::from_bits)
}

fn finite_price() -> impl Strategy<Value = f64> {
    (1u32..=100_000).prop_map(|t| t as f64 * 0.01)
}

fn nonzero_size() -> impl Strategy<Value = f64> {
    (1u32..=100_000).prop_map(|t| t as f64 * 0.01)
}

fn interval() -> impl Strategy<Value = NonZeroU64> {
    (1u64..=50).prop_map(|n| NonZeroU64::new(n).unwrap())
}

fn empty_bar() -> impl Strategy<Value = EmptyBar> {
    prop_oneof![Just(EmptyBar::ForwardFill), Just(EmptyBar::Gap)]
}

fn bar_field() -> impl Strategy<Value = BarField> {
    prop_oneof![
        Just(BarField::Close),
        Just(BarField::Vwap),
        Just(BarField::LogReturn),
    ]
}

/// Element `k` is `start` plus the sum of `steps[..k]`, so a zero step repeats the
/// previous timestamp and the sequence never decreases.
fn scan_steps_from(start: u64, steps: Vec<u64>) -> Vec<u64> {
    steps
        .into_iter()
        .scan(start, |ts, step| {
            let value = *ts;
            *ts += step;
            Some(value)
        })
        .collect()
}

fn non_decreasing_timestamps(len: usize) -> impl Strategy<Value = Vec<u64>> {
    (0u64..=500, prop::collection::vec(0u64..=200, len))
        .prop_map(|(start, steps)| scan_steps_from(start, steps))
}

fn events() -> impl Strategy<Value = Vec<(u64, f64, f64)>> {
    (1usize..12).prop_flat_map(|len| {
        (
            non_decreasing_timestamps(len),
            prop::collection::vec((any_f64(), any_f64()), len),
        )
            .prop_map(|(timestamps, rest)| {
                timestamps
                    .into_iter()
                    .zip(rest)
                    .map(|(ts, (price, size))| (ts, price, size))
                    .collect()
            })
    })
}

/// Offsets sorted within `[0, interval)`, so timestamps built from them stay inside
/// one bar.
fn interval_and_offsets() -> impl Strategy<Value = (NonZeroU64, Vec<u64>)> {
    interval().prop_flat_map(|interval| {
        let span = interval.get();
        prop::collection::vec(0u64..span, 1..12).prop_map(move |mut offsets| {
            offsets.sort_unstable();
            (interval, offsets)
        })
    })
}

fn run(sampler: &mut TimeBars, events: &[(u64, f64, f64)]) -> Vec<u64> {
    let mut out = Vec::new();
    for &(ts, price, size) in events {
        sampler
            .on_event(ts, price, size, |x| out.push(x.to_bits()))
            .unwrap();
    }
    out
}

fn run_with_heartbeats(
    interval: NonZeroU64,
    empty: EmptyBar,
    field: BarField,
    events: &[(u64, f64, f64)],
    heartbeat_before: &[bool],
    final_ts: u64,
) -> Vec<u64> {
    let mut sampler = TimeBars::new(interval, empty, field);
    let mut out = Vec::new();
    for (i, &(ts, price, size)) in events.iter().enumerate() {
        if heartbeat_before.get(i).copied().unwrap_or(false) {
            sampler.advance_to(ts, |x| out.push(x.to_bits())).unwrap();
        }
        sampler
            .on_event(ts, price, size, |x| out.push(x.to_bits()))
            .unwrap();
    }
    sampler
        .advance_to(final_ts, |x| out.push(x.to_bits()))
        .unwrap();
    out
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    /// However many events land inside one bar, none of them close it.
    #[test]
    fn bars_close_only_when_ended(
        (interval, offsets) in interval_and_offsets(),
        empty in empty_bar(),
        field in bar_field(),
        bar in 0u64..100,
        values in prop::collection::vec((any_f64(), any_f64()), 1..12),
    ) {
        let start = bar * interval.get();
        let n = offsets.len().min(values.len());
        let events: Vec<(u64, f64, f64)> = offsets[..n]
            .iter()
            .map(|&o| start + o)
            .zip(values[..n].iter().copied())
            .map(|(ts, (price, size))| (ts, price, size))
            .collect();

        let mut sampler = TimeBars::new(interval, empty, field);
        let emitted = run(&mut sampler, &events);
        prop_assert!(emitted.is_empty());
    }

    /// Wherever heartbeats land among the same events, the samples that come out are
    /// the same once a shared final heartbeat passes the last bar's end.
    #[test]
    fn heartbeat_interleaving_is_irrelevant(
        interval in interval(),
        empty in empty_bar(),
        field in bar_field(),
        events in events(),
        heartbeats_a in prop::collection::vec(any::<bool>(), 0..20),
        heartbeats_b in prop::collection::vec(any::<bool>(), 0..20),
    ) {
        let last_ns = events.last().unwrap().0;
        let final_ts = last_ns + interval.get();
        let a = run_with_heartbeats(interval, empty, field, &events, &heartbeats_a, final_ts);
        let b = run_with_heartbeats(interval, empty, field, &events, &heartbeats_b, final_ts);
        prop_assert_eq!(a, b);
    }

    /// Samples already emitted from a prefix are untouched by whatever is appended
    /// after it, down to the bit.
    #[test]
    fn sampler_causality(
        interval in interval(),
        empty in empty_bar(),
        field in bar_field(),
        prefix in events(),
        suffix_steps in prop::collection::vec(0u64..=200, 0..12),
        suffix_values in prop::collection::vec((any_f64(), any_f64()), 0..12),
    ) {
        let suffix_len = suffix_steps.len().min(suffix_values.len());
        let start = prefix.last().map_or(0, |&(ts, _, _)| ts);
        let suffix_timestamps = scan_steps_from(start, suffix_steps[..suffix_len].to_vec());
        let suffix: Vec<(u64, f64, f64)> = suffix_timestamps
            .into_iter()
            .zip(suffix_values[..suffix_len].iter().copied())
            .map(|(ts, (price, size))| (ts, price, size))
            .collect();

        let prefix_emitted = run(&mut TimeBars::new(interval, empty, field), &prefix);

        let mut extended = TimeBars::new(interval, empty, field);
        let mut combined_emitted = run(&mut extended, &prefix);
        combined_emitted.extend(run(&mut extended, &suffix));

        prop_assert_eq!(&combined_emitted[..prefix_emitted.len()], &prefix_emitted[..]);
    }

    /// `ForwardFill` repeats the last real value exactly `gap` times; `Gap` emits
    /// exactly `gap` `NaN`s.
    #[test]
    fn empty_bar_policy(
        interval in interval(),
        empty in empty_bar(),
        field in bar_field(),
        bar0 in 0u64..50,
        gap in 0u64..20,
        price0 in finite_price(),
        size0 in nonzero_size(),
        price1 in finite_price(),
        size1 in nonzero_size(),
    ) {
        let span = interval.get();
        let ts0 = bar0 * span;
        let bar1 = bar0 + gap + 1;
        let ts1 = bar1 * span;
        let final_ts = ts1 + span;

        let mut sampler = TimeBars::new(interval, empty, field);
        let mut emitted = Vec::new();
        sampler.on_event(ts0, price0, size0, |x| emitted.push(x)).unwrap();
        sampler.on_event(ts1, price1, size1, |x| emitted.push(x)).unwrap();
        sampler.advance_to(final_ts, |x| emitted.push(x)).unwrap();

        let vwap0 = price0 * size0 / size0;
        let vwap1 = price1 * size1 / size1;
        let (first, last, fill): (Option<f64>, f64, f64) = match (field, empty) {
            (BarField::Close, EmptyBar::ForwardFill) => (Some(price0), price1, price0),
            (BarField::Close, EmptyBar::Gap) => (Some(price0), price1, f64::NAN),
            (BarField::Vwap, EmptyBar::ForwardFill) => (Some(vwap0), vwap1, vwap0),
            (BarField::Vwap, EmptyBar::Gap) => (Some(vwap0), vwap1, f64::NAN),
            (BarField::LogReturn, EmptyBar::ForwardFill) => {
                (None, libm::log(price1 / price0), 0.0)
            }
            (BarField::LogReturn, EmptyBar::Gap) => {
                (None, libm::log(price1 / price0), f64::NAN)
            }
        };

        let mut expected: Vec<f64> = first.into_iter().collect();
        expected.extend(core::iter::repeat_n(fill, gap as usize));
        expected.push(last);

        prop_assert_eq!(emitted.len(), expected.len());
        for (a, b) in emitted.iter().zip(expected.iter()) {
            prop_assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    /// A strictly earlier timestamp is rejected and leaves every bit of state as it
    /// was, for both `on_event` and `advance_to`.
    #[test]
    fn rejects_out_of_order(
        interval in interval(),
        empty in empty_bar(),
        field in bar_field(),
        events in events(),
        bad_offset in 1u64..=1_000_000,
        bad_price in any_f64(),
        bad_size in any_f64(),
        next_price in any_f64(),
        next_size in any_f64(),
    ) {
        let last_ns = events.last().unwrap().0;
        prop_assume!(last_ns > 0);
        let bad_ts = last_ns.saturating_sub(bad_offset);

        let mut sampler = TimeBars::new(interval, empty, field);
        run(&mut sampler, &events);

        let mut emitted = false;
        let rejected = sampler.on_event(bad_ts, bad_price, bad_size, |_| emitted = true);
        prop_assert_eq!(rejected, Err(SamplerError::OutOfOrder { last_ns, got_ns: bad_ts }));
        prop_assert!(!emitted);

        let rejected_again = sampler.on_event(bad_ts, bad_price, bad_size, |_| emitted = true);
        prop_assert_eq!(rejected_again, Err(SamplerError::OutOfOrder { last_ns, got_ns: bad_ts }));
        prop_assert!(!emitted);

        let mut heartbeat_emitted = false;
        let heartbeat = sampler.advance_to(bad_ts, |_| heartbeat_emitted = true);
        prop_assert_eq!(heartbeat, Err(SamplerError::OutOfOrder { last_ns, got_ns: bad_ts }));
        prop_assert!(!heartbeat_emitted);

        let mut clean = TimeBars::new(interval, empty, field);
        run(&mut clean, &events);
        let mut clean_next = Vec::new();
        clean
            .on_event(last_ns, next_price, next_size, |x| clean_next.push(x.to_bits()))
            .unwrap();

        let mut next_emitted = Vec::new();
        sampler
            .on_event(last_ns, next_price, next_size, |x| next_emitted.push(x.to_bits()))
            .unwrap();

        prop_assert_eq!(next_emitted, clean_next);
    }
}

#[test]
fn bar_closes_only_once_an_event_reaches_its_end() {
    let interval = NonZeroU64::new(10).unwrap();
    let mut sampler = TimeBars::new(interval, EmptyBar::ForwardFill, BarField::Close);
    let mut emitted = Vec::new();
    sampler.on_event(9, 1.0, 1.0, |x| emitted.push(x)).unwrap();
    assert!(emitted.is_empty());
    sampler.on_event(10, 2.0, 1.0, |x| emitted.push(x)).unwrap();
    assert_eq!(emitted, [1.0]);
}

#[test]
fn advance_to_before_the_open_bars_end_emits_nothing() {
    let interval = NonZeroU64::new(10).unwrap();
    let mut sampler = TimeBars::new(interval, EmptyBar::ForwardFill, BarField::Close);
    let mut emitted = Vec::new();
    sampler.on_event(0, 1.0, 1.0, |x| emitted.push(x)).unwrap();
    sampler.advance_to(9, |x| emitted.push(x)).unwrap();
    assert!(emitted.is_empty());
    sampler.advance_to(10, |x| emitted.push(x)).unwrap();
    assert_eq!(emitted, [1.0]);
}

#[test]
fn close_emits_last_price_in_bar() {
    let interval = NonZeroU64::new(10).unwrap();
    let mut sampler = TimeBars::new(interval, EmptyBar::ForwardFill, BarField::Close);
    let mut emitted = Vec::new();
    sampler.on_event(0, 1.0, 1.0, |x| emitted.push(x)).unwrap();
    sampler.on_event(5, 2.0, 1.0, |x| emitted.push(x)).unwrap();
    sampler.on_event(10, 3.0, 1.0, |x| emitted.push(x)).unwrap();
    sampler.advance_to(20, |x| emitted.push(x)).unwrap();
    assert_eq!(emitted, [2.0, 3.0]);
}

#[test]
fn vwap_is_size_weighted_average() {
    let interval = NonZeroU64::new(10).unwrap();
    let mut sampler = TimeBars::new(interval, EmptyBar::ForwardFill, BarField::Vwap);
    let mut emitted = Vec::new();
    sampler.on_event(0, 10.0, 1.0, |x| emitted.push(x)).unwrap();
    sampler.on_event(1, 20.0, 3.0, |x| emitted.push(x)).unwrap();
    sampler.advance_to(10, |x| emitted.push(x)).unwrap();
    assert_eq!(emitted, [17.5]);
}

#[test]
fn vwap_falls_back_to_close_when_size_is_zero() {
    let interval = NonZeroU64::new(10).unwrap();
    let mut sampler = TimeBars::new(interval, EmptyBar::ForwardFill, BarField::Vwap);
    let mut emitted = Vec::new();
    sampler.on_event(0, 10.0, 0.0, |x| emitted.push(x)).unwrap();
    sampler.on_event(1, 20.0, 0.0, |x| emitted.push(x)).unwrap();
    sampler.advance_to(10, |x| emitted.push(x)).unwrap();
    assert_eq!(emitted, [20.0]);
}

#[test]
fn log_return_skips_the_first_bar_then_is_relative_to_the_previous_close() {
    let interval = NonZeroU64::new(10).unwrap();
    let mut sampler = TimeBars::new(interval, EmptyBar::ForwardFill, BarField::LogReturn);
    let mut emitted = Vec::new();
    sampler
        .on_event(0, 100.0, 1.0, |x| emitted.push(x))
        .unwrap();
    sampler
        .on_event(10, 110.0, 1.0, |x| emitted.push(x))
        .unwrap();
    sampler.advance_to(20, |x| emitted.push(x)).unwrap();
    assert_eq!(emitted, [libm::log(110.0 / 100.0)]);
}

#[test]
fn log_return_after_a_gap_is_relative_to_the_last_real_close() {
    let interval = NonZeroU64::new(10).unwrap();
    let mut sampler = TimeBars::new(interval, EmptyBar::Gap, BarField::LogReturn);
    let mut emitted = Vec::new();
    sampler
        .on_event(0, 100.0, 1.0, |x| emitted.push(x))
        .unwrap();
    sampler
        .on_event(20, 150.0, 1.0, |x| emitted.push(x))
        .unwrap();
    sampler.advance_to(30, |x| emitted.push(x)).unwrap();
    assert_eq!(emitted.len(), 2);
    assert!(emitted[0].is_nan());
    assert_eq!(emitted[1], libm::log(150.0 / 100.0));
}

#[test]
fn empty_bars_closed_one_heartbeat_at_a_time_keep_the_last_real_close() {
    let interval = NonZeroU64::new(10).unwrap();
    let mut sampler = TimeBars::new(interval, EmptyBar::Gap, BarField::LogReturn);
    let mut emitted = Vec::new();
    sampler
        .on_event(0, 100.0, 1.0, |x| emitted.push(x))
        .unwrap();
    sampler.advance_to(10, |x| emitted.push(x)).unwrap();
    sampler.advance_to(20, |x| emitted.push(x)).unwrap();
    sampler
        .on_event(25, 150.0, 1.0, |x| emitted.push(x))
        .unwrap();
    sampler.advance_to(30, |x| emitted.push(x)).unwrap();
    assert_eq!(emitted.len(), 2);
    assert!(emitted[0].is_nan());
    assert_eq!(emitted[1], libm::log(150.0 / 100.0));
}
