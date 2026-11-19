extern crate std;

use super::{Sampler, SamplerError, Ticks};
use proptest::prelude::*;
use std::vec::Vec;

fn any_f64() -> impl Strategy<Value = f64> {
    any::<u64>().prop_map(f64::from_bits)
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
    (0u64..=1000, prop::collection::vec(0u64..=1000, len))
        .prop_map(|(start, steps)| scan_steps_from(start, steps))
}

fn events() -> impl Strategy<Value = Vec<(u64, f64, f64)>> {
    (1usize..20).prop_flat_map(|len| {
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

fn run(sampler: &mut Ticks, events: &[(u64, f64, f64)]) -> Vec<u64> {
    let mut out = Vec::new();
    for &(ts, price, size) in events {
        sampler
            .on_event(ts, price, size, |x| out.push(x.to_bits()))
            .unwrap();
    }
    out
}

/// Calls `advance_to(ts, ..)` right before `on_event` wherever `heartbeat_before[i]`
/// is `true`; the heartbeat's timestamp always matches the event's, so it is always
/// in order and never changes which event comes next.
fn run_with_heartbeats(events: &[(u64, f64, f64)], heartbeat_before: &[bool]) -> Vec<u64> {
    let mut sampler = Ticks::default();
    let mut out = Vec::new();
    for (i, &(ts, price, size)) in events.iter().enumerate() {
        if heartbeat_before.get(i).copied().unwrap_or(false) {
            sampler.advance_to(ts, |x| out.push(x.to_bits())).unwrap();
        }
        sampler
            .on_event(ts, price, size, |x| out.push(x.to_bits()))
            .unwrap();
    }
    out
}

proptest! {
    #![proptest_config(testkit::generators::seeded_default())]
    /// Whether, or where, the caller sends heartbeats between events never changes
    /// which samples come out, and each one is the event's price, not its size.
    #[test]
    fn sampler_determinism(
        events in events(),
        heartbeats_a in prop::collection::vec(any::<bool>(), 0..20),
        heartbeats_b in prop::collection::vec(any::<bool>(), 0..20),
    ) {
        let expected: Vec<u64> = events.iter().map(|&(_, price, _)| price.to_bits()).collect();
        prop_assert_eq!(run_with_heartbeats(&events, &heartbeats_a), expected.clone());
        prop_assert_eq!(run_with_heartbeats(&events, &heartbeats_b), expected);
    }

    /// Samples already emitted from a prefix are untouched by whatever is appended
    /// after it, down to the bit.
    #[test]
    fn sampler_causality(
        prefix in events(),
        suffix_steps in prop::collection::vec(0u64..=1000, 0..20),
        suffix_values in prop::collection::vec((any_f64(), any_f64()), 0..20),
    ) {
        let suffix_len = suffix_steps.len().min(suffix_values.len());
        let start = prefix.last().map_or(0, |&(ts, _, _)| ts);
        let suffix_timestamps = scan_steps_from(start, suffix_steps[..suffix_len].to_vec());
        let suffix: Vec<(u64, f64, f64)> = suffix_timestamps
            .into_iter()
            .zip(suffix_values[..suffix_len].iter().copied())
            .map(|(ts, (price, size))| (ts, price, size))
            .collect();

        let prefix_emitted = run(&mut Ticks::default(), &prefix);

        let mut extended = Ticks::default();
        let mut combined_emitted = run(&mut extended, &prefix);
        combined_emitted.extend(run(&mut extended, &suffix));

        prop_assert_eq!(&combined_emitted[..prefix_emitted.len()], &prefix_emitted[..]);
    }

    /// An earlier timestamp is rejected, emits nothing, leaves state exactly as it
    /// was, and the event that was always next proceeds as if it had never arrived.
    #[test]
    fn sampler_rejects_out_of_order(
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

        let mut sampler = Ticks::default();
        run(&mut sampler, &events);

        let mut emitted = false;
        let rejected = sampler.on_event(bad_ts, bad_price, bad_size, |_| emitted = true);
        prop_assert_eq!(rejected, Err(SamplerError::OutOfOrder { last_ns, got_ns: bad_ts }));
        prop_assert!(!emitted);

        // The same bad call again must fail identically: state did not move to `bad_ts`.
        let rejected_again = sampler.on_event(bad_ts, bad_price, bad_size, |_| emitted = true);
        prop_assert_eq!(rejected_again, Err(SamplerError::OutOfOrder { last_ns, got_ns: bad_ts }));
        prop_assert!(!emitted);

        // The event that was always next, at the original last timestamp, still works.
        let mut next_emitted = None;
        sampler.on_event(last_ns, next_price, next_size, |x| next_emitted = Some(x.to_bits())).unwrap();
        prop_assert_eq!(next_emitted, Some(next_price.to_bits()));

        // `advance_to` enforces the same order.
        let mut heartbeat_emitted = false;
        let heartbeat = sampler.advance_to(bad_ts, |_| heartbeat_emitted = true);
        prop_assert_eq!(heartbeat, Err(SamplerError::OutOfOrder { last_ns, got_ns: bad_ts }));
        prop_assert!(!heartbeat_emitted);
    }
}

#[test]
fn equal_timestamp_is_accepted() {
    let mut sampler = Ticks::default();
    let mut emitted = Vec::new();
    sampler
        .on_event(10, 1.0, 100.0, |x| emitted.push(x))
        .unwrap();
    sampler
        .on_event(10, 2.0, 200.0, |x| emitted.push(x))
        .unwrap();
    assert_eq!(emitted, [1.0, 2.0]);
}

#[test]
fn out_of_order_is_rejected_without_changing_state() {
    let mut sampler = Ticks::default();
    sampler.on_event(10, 1.0, 100.0, |_| {}).unwrap();

    let err = sampler
        .on_event(5, 9.9, 9.9, |_| panic!("must not emit"))
        .unwrap_err();
    assert_eq!(
        err,
        SamplerError::OutOfOrder {
            last_ns: 10,
            got_ns: 5
        }
    );

    let err_again = sampler
        .on_event(5, 9.9, 9.9, |_| panic!("must not emit"))
        .unwrap_err();
    assert_eq!(
        err_again,
        SamplerError::OutOfOrder {
            last_ns: 10,
            got_ns: 5
        }
    );

    let mut emitted = None;
    sampler
        .on_event(10, 3.0, 300.0, |x| emitted = Some(x))
        .unwrap();
    assert_eq!(emitted, Some(3.0));
}

#[test]
fn advance_to_rejects_earlier_timestamp() {
    let mut sampler = Ticks::default();
    sampler.on_event(10, 1.0, 100.0, |_| {}).unwrap();
    assert_eq!(
        sampler.advance_to(5, |_| panic!("must not emit")),
        Err(SamplerError::OutOfOrder {
            last_ns: 10,
            got_ns: 5
        })
    );
}

#[test]
fn on_event_emits_price_not_size() {
    let mut sampler = Ticks::default();
    let mut emitted = None;
    sampler
        .on_event(1, 42.0, 7.0, |x| emitted = Some(x))
        .unwrap();
    assert_eq!(emitted, Some(42.0));
}
