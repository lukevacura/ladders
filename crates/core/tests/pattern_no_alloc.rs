//! `PatternStream::push` allocates nothing after construction (its own test binary: the counting
//! allocator only counts once it is the global allocator, like `nearest_no_alloc.rs`).

use core::num::NonZeroUsize;
use ladde::pattern::{PatternLibrary, PatternStream};
use ladde::quantize::Scale;
use testkit::alloc_count::{CountingAllocator, allocations_in};

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn push_never_allocates_after_construction() {
    let m = 16;
    let mut lib = PatternLibrary::new(NonZeroUsize::new(m).unwrap(), Scale::new(0.01).unwrap());
    for k in 0..5 {
        let q: Vec<f64> = (0..m)
            .map(|i| 1e8 + ((i * (k + 3)) % 7) as f64 * 0.01)
            .collect();
        lib.register(&q).unwrap();
    }
    let mut stream = PatternStream::new(lib);
    let mut x = 1e8;
    let mut state = 12345u64;
    let mut finite = 0usize;
    let (_, count) = allocations_in(|| {
        for i in 0..20_000u32 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let step = ((state >> 33) % 21) as f64 * 0.01 - 0.1;
            x += step;
            // gaps, flat runs, and a rising/falling range to churn the monotonic deques
            let s = if i % 97 == 5 {
                f64::NAN
            } else if (i / 50) % 4 == 3 {
                1e8
            } else {
                x
            };
            finite += stream
                .push(s)
                .iter()
                .filter(|e| e.squared_distance().is_finite())
                .count();
        }
    });
    assert!(finite > 20_000, "{finite}");
    assert_eq!(count, 0);
}
