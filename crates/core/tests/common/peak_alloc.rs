//! A process-wide peak-heap tracker for the memory tests: each test binary that includes this
//! module gets it as its global allocator (it counts allocations on every thread), and
//! `peak_of` reports the most heap a closure held at once beyond what was held before it.
#![allow(dead_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static LARGE_MIN: AtomicUsize = AtomicUsize::new(usize::MAX);
static LARGE: AtomicUsize = AtomicUsize::new(0);

struct Peak;

fn note_size(size: usize) {
    if size >= LARGE_MIN.load(Ordering::Relaxed) && !on_main_thread() {
        LARGE.fetch_add(1, Ordering::Relaxed);
    }
}

/// Whether the calling thread is the process's first thread. libtest runs each test on a thread
/// of its own and keeps its bookkeeping (the running test's handle, its output) on the first one,
/// right as the test starts: under load that lands inside a count, so counts leave it out.
fn on_main_thread() -> bool {
    #[cfg(target_os = "macos")]
    {
        unsafe extern "C" {
            fn pthread_main_np() -> i32;
        }
        // SAFETY: takes nothing and only reads the calling thread's state; it never allocates.
        unsafe { pthread_main_np() == 1 }
    }
    #[cfg(target_os = "linux")]
    {
        unsafe extern "C" {
            fn gettid() -> i32;
            fn getpid() -> i32;
        }
        // SAFETY: both take nothing and only read the calling thread's and process's ids.
        unsafe { gettid() == getpid() }
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        false
    }
}

fn grow(by: usize) {
    let now = CURRENT.fetch_add(by, Ordering::SeqCst) + by;
    PEAK.fetch_max(now, Ordering::SeqCst);
}

// SAFETY: every call is forwarded unchanged to `System`; the counters are only bookkeeping.
unsafe impl GlobalAlloc for Peak {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note_size(layout.size());
        grow(layout.size());
        // SAFETY: `layout` is the caller's, forwarded unchanged.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note_size(layout.size());
        grow(layout.size());
        // SAFETY: `layout` is the caller's, forwarded unchanged.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        CURRENT.fetch_sub(layout.size(), Ordering::SeqCst);
        // SAFETY: `ptr` and `layout` are the caller's, forwarded unchanged.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note_size(new_size);
        if new_size >= layout.size() {
            grow(new_size - layout.size());
        } else {
            CURRENT.fetch_sub(layout.size() - new_size, Ordering::SeqCst);
        }
        // SAFETY: `ptr`, `layout` and `new_size` are the caller's, forwarded unchanged.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Peak = Peak;

pub fn peak_of(f: impl FnOnce()) -> usize {
    let base = CURRENT.load(Ordering::SeqCst);
    PEAK.store(base, Ordering::SeqCst);
    f();
    PEAK.load(Ordering::SeqCst) - base
}

/// How many allocations (or reallocations) of at least `min_bytes` the closure makes, on any
/// thread but the process's first, which never runs a test. Unlike a peak, it does not depend on
/// how the threads interleave.
pub fn large_allocations_of(min_bytes: usize, f: impl FnOnce()) -> usize {
    assert!(
        !on_main_thread(),
        "allocation counts leave out the process's first thread, so they cannot be taken on it"
    );
    LARGE.store(0, Ordering::SeqCst);
    LARGE_MIN.store(min_bytes, Ordering::SeqCst);
    f();
    LARGE_MIN.store(usize::MAX, Ordering::SeqCst);
    LARGE.load(Ordering::SeqCst)
}

/// Runs `run` at each thread count and asserts every run made the same number of allocations of
/// at least `min_bytes`: everything of that size belongs to the caller or to the plan, so a worker
/// that copied a column of its own would add one per worker. Returns that number.
pub fn assert_equal_large_allocations(
    min_bytes: usize,
    threads: &[usize],
    label: &str,
    run: impl Fn(usize),
) -> usize {
    let counts: Vec<usize> = threads
        .iter()
        .map(|&t| large_allocations_of(min_bytes, || run(t)))
        .collect();
    assert!(
        counts.iter().all(|&c| c == counts[0]),
        "{label}: allocations of {min_bytes} bytes or more at {threads:?} threads: {counts:?}"
    );
    counts[0]
}

/// How many times a worker's largest scratch buffer fits in half a column: twice, so a buffer of
/// as many elements that grew by doubling stays below half a column too.
const HEADROOM: usize = 2;

/// The largest buffer the tile plan sizes in a worker's scratch, in bytes: the tile's lag state
/// (an `i64` per lane of the tile), the rows its survivor gate left lanes out at (a `usize` per
/// lane of the tile, allocated when the gate first leaves a block out), one chunk's rho (an `f64`
/// per lane of the chunk), and the band's accumulators and scalars (one of each per band row).
fn largest_scratch((band_rows, chunk_lanes, tile_lanes): (usize, usize, usize)) -> usize {
    use ladde::kernel_neon::RowScalars;
    use ladde::row::RowAcc;

    [
        tile_lanes * size_of::<i64>(),
        tile_lanes * size_of::<usize>(),
        chunk_lanes * size_of::<f64>(),
        band_rows * size_of::<RowAcc>(),
        band_rows * size_of::<RowScalars>(),
    ]
    .into_iter()
    .max()
    .unwrap()
}

/// A threaded run for the counts above, sized by the model of a worker's buffers instead of by
/// fixed lengths: the series is the shortest whose half column is `HEADROOM` times the buffers of
/// a band of `band_rows` and a chunk of `chunk_lanes`, and the tiles are the widest, in whole
/// chunks, whose lag state leaves the same headroom. So the sizes of the library's types move the
/// run, not its verdict. The accumulators of the model are the z-normalized engine's; the
/// non-normalized and multi-dimensional engines keep a partial of their own, private to the
/// library, per band row in their place.
pub struct CountedRun {
    pub samples: usize,
    pub band_rows: usize,
    pub chunk_lanes: usize,
    pub tile_lanes: usize,
}

impl CountedRun {
    pub fn new(band_rows: usize, chunk_lanes: usize) -> Self {
        let half_column = HEADROOM * largest_scratch((band_rows, chunk_lanes, 0));
        let samples = half_column.div_ceil(HALF_COLUMN_PER_SAMPLE);
        let lanes = samples * HALF_COLUMN_PER_SAMPLE / (HEADROOM * size_of::<i64>());
        Self {
            samples,
            band_rows,
            chunk_lanes,
            tile_lanes: lanes / chunk_lanes * chunk_lanes,
        }
    }

    /// Half a column of the run's series, the size the counts start at.
    pub fn half_column(&self) -> usize {
        self.samples * HALF_COLUMN_PER_SAMPLE
    }

    pub fn shape(&self) -> (usize, usize, usize) {
        (self.band_rows, self.chunk_lanes, self.tile_lanes)
    }
}

/// Half of a column's eight bytes per sample.
const HALF_COLUMN_PER_SAMPLE: usize = 4;

/// Asserts what makes the counts above schedule-independent: half a column is `HEADROOM` times a
/// worker's largest scratch buffer under the tile plan ([`largest_scratch`]), and is no less than
/// the list of one small record per tile a worker keeps, were it to claim every tile and the list
/// to have just doubled. `tiles` is the number of tiles the run cut, the most any one worker can
/// claim.
pub fn assert_workers_hold_less_than_half_a_column(
    half_column: usize,
    shape: (usize, usize, usize),
    tiles: usize,
) {
    let scratch_largest = largest_scratch(shape);
    assert!(
        HEADROOM * scratch_largest <= half_column,
        "a worker's largest buffer ({scratch_largest} bytes) must fit {HEADROOM} times in half a \
         column ({half_column} bytes)"
    );
    let claimed = tiles * size_of::<(usize, bool)>();
    assert!(
        2 * claimed <= half_column,
        "a worker claiming all {tiles} tiles keeps {claimed} bytes of records, which doubled \
         pass half a column ({half_column} bytes)"
    );
}
