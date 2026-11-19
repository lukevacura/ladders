//! A global allocator that counts heap operations per thread, so a test can prove a code
//! path allocates nothing.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

const LOG_SLOTS: usize = 256;

thread_local! {
    static COUNT: Cell<usize> = const { Cell::new(0) };
    static LOG_MIN: Cell<usize> = const { Cell::new(usize::MAX) };
    static LOG_LEN: Cell<usize> = const { Cell::new(0) };
    static LOG: [Cell<(usize, bool)>; LOG_SLOTS] =
        const { [const { Cell::new((0, false)) }; LOG_SLOTS] };
}

fn note(bytes: usize, realloc: bool) {
    COUNT.with(|count| count.set(count.get() + 1));
    if bytes >= LOG_MIN.with(Cell::get) {
        let len = LOG_LEN.with(Cell::get);
        if len < LOG_SLOTS {
            LOG.with(|log| log[len].set((bytes, realloc)));
            LOG_LEN.with(|l| l.set(len + 1));
        }
    }
}

/// Counts `alloc`, `alloc_zeroed` and `realloc` calls made by the installing thread, then
/// delegates to [`System`]. Install with `#[global_allocator]` in a test binary.
pub struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note(layout.size(), false);
        // SAFETY: `layout` is exactly what the caller passed; forwarded unchanged to `System`.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note(layout.size(), false);
        // SAFETY: `layout` is exactly what the caller passed; forwarded unchanged to `System`.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` and `layout` are exactly what the caller passed; forwarded unchanged to
        // `System`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note(new_size, true);
        // SAFETY: `ptr`, `layout` and `new_size` are exactly what the caller passed; forwarded
        // unchanged to `System`.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

/// Runs `f`, returning its result paired with the heap allocations counted on this thread
/// while it ran. Calls nest: an inner `allocations_in` reports only what happened inside its
/// own closure.
pub fn allocations_in<R>(f: impl FnOnce() -> R) -> (R, usize) {
    let before = COUNT.with(Cell::get);
    let result = f();
    let after = COUNT.with(Cell::get);
    (result, after - before)
}

/// One heap request of at least the logged size: its byte count, and whether it resized an
/// existing block (`realloc`) rather than made a new one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LargeRequest {
    pub bytes: usize,
    pub realloc: bool,
}

/// Runs `f`, returning its result with every heap request of at least `min_bytes` the installing
/// thread made while it ran, in order (the first 256 only). Does not nest.
pub fn large_requests_in<R>(min_bytes: usize, f: impl FnOnce() -> R) -> (R, Vec<LargeRequest>) {
    LOG_LEN.with(|len| len.set(0));
    LOG_MIN.with(|min| min.set(min_bytes));
    let result = f();
    LOG_MIN.with(|min| min.set(usize::MAX));
    let len = LOG_LEN.with(Cell::get);
    let requests = LOG.with(|log| {
        log[..len]
            .iter()
            .map(|slot| {
                let (bytes, realloc) = slot.get();
                LargeRequest { bytes, realloc }
            })
            .collect()
    });
    (result, requests)
}
