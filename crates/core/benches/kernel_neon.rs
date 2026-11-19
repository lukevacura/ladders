//! `kernel_neon` bench: ns per lag for [`rank_row`], at `n ∈ {2 000, 19 000, 100 000}`, in
//! both max mode (`scratch: None`) and scratch mode (`scratch: Some(..)`).
//!
//! Hand-timed with `std::time::Instant` rather than the `criterion` crate, for the same reason
//! `row`'s bench is (see its own module doc): `criterion` pulls in `serde_json`, whose blanket
//! `PartialEq<Value>` impls make this crate's own unannotated empty-slice comparisons
//! ambiguous at the type-checker.
//!
//! Every lane is given a Valid neighbor (`r > 0`, `bias = 0.0`): the kernel is branch-free, so
//! a lane's cost does not depend on which of the three biases it carries, and this keeps the
//! setup simple. The timed region is one `rank_row` call over the whole row, repeated after a
//! warm-up; `qt_m` is reset from a snapshot before every call (timed or not) so repeated calls
//! never see state a real stream wouldn't have handed them twice.

use ladde::kernel_neon::{Conversion, RowScalars, rank_row};
use std::hint::black_box;
use std::time::Instant;

const WARMUP_ITERS: u32 = 200;
const TIMED_ITERS: u32 = 2_000;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

struct Setup {
    enter: Vec<i32>,
    leave: Vec<i32>,
    sum: Vec<i32>,
    r: Vec<f64>,
    bias: Vec<f64>,
    qt_snapshot: Vec<i64>,
}

fn build(n: usize, seed: u64) -> Setup {
    let mut state = seed;
    let mut next_i32 = |lo: i32, hi: i32| -> i32 {
        let span = (hi - lo) as u64 + 1;
        lo + (splitmix64(&mut state) % span) as i32
    };
    let enter: Vec<i32> = (0..n).map(|_| next_i32(-1_000_000, 1_000_000)).collect();
    let leave: Vec<i32> = (0..n).map(|_| next_i32(-1_000_000, 1_000_000)).collect();
    let sum: Vec<i32> = (0..n).map(|_| next_i32(-20_000_000, 20_000_000)).collect();
    let r: Vec<f64> = (0..n)
        .map(|_| 1e-6 * ((splitmix64(&mut state) % 1000) as f64 + 1.0))
        .collect();
    let bias = vec![0.0f64; n];
    let qt_snapshot: Vec<i64> = (0..n)
        .map(|_| (splitmix64(&mut state) as i64).wrapping_sub(i64::MAX / 2))
        .collect();
    Setup {
        enter,
        leave,
        sum,
        r,
        bias,
        qt_snapshot,
    }
}

/// Mean nanoseconds per lag of `TIMED_ITERS` whole-row calls at `n`, after `WARMUP_ITERS`
/// unmeasured calls, with and without a scratch column.
fn bench_one(n: usize) -> (f64, f64) {
    let setup = build(n, 0xC0FFEE ^ n as u64);
    let mut qt = setup.qt_snapshot.clone();
    let mut scratch = vec![0.0f64; n];

    let mut run = |use_scratch: bool| -> f64 {
        let mut total = std::time::Duration::ZERO;
        for i in 0..(WARMUP_ITERS + TIMED_ITERS) {
            qt.copy_from_slice(&setup.qt_snapshot);
            let row = RowScalars {
                me: black_box(50i32.wrapping_mul(i as i32)),
                ml: black_box(150),
                si: black_box(7),
                ri: black_box(1e-6),
                conversion: Conversion::FULL_RANGE,
            };
            let scratch_arg = use_scratch.then_some(scratch.as_mut_slice());
            let start = Instant::now();
            black_box(rank_row(
                black_box(&mut qt),
                black_box(&setup.enter),
                black_box(&setup.leave),
                black_box(&setup.sum),
                black_box(&setup.r),
                black_box(&setup.bias),
                row,
                scratch_arg,
            ));
            if i >= WARMUP_ITERS {
                total += start.elapsed();
            }
        }
        total.as_secs_f64() * 1e9 / (f64::from(TIMED_ITERS) * n as f64)
    };

    let max_mode = run(false);
    let scratch_mode = run(true);
    (max_mode, scratch_mode)
}

fn main() {
    println!("kernel_neon: ns per lag");
    for &n in &[2_000usize, 19_000, 100_000] {
        let (max_mode, scratch_mode) = bench_one(n);
        println!(
            "  n = {n:>7}  max mode {max_mode:.3} ns/lag  scratch mode {scratch_mode:.3} ns/lag"
        );
    }
}
