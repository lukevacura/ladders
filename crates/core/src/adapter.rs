//! One writer appends; any number of readers observe the prefix it has published.
//!
//! The core stream is single-threaded and holds no shared state (see `sync.rs` on
//! why that's enough reason to keep atomics out of it). This module is for callers
//! who want to read a stream's output from another thread anyway.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::mem::MaybeUninit;

use crate::capacity::CapacityError;
use crate::sync::{self, AtomicUsize, Ordering};

/// The publisher has written every slot; no further push succeeds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Full;

struct Shared<T> {
    slots: Box<[sync::UnsafeCell<MaybeUninit<T>>]>,
    published: AtomicUsize,
}

// SAFETY: once a slot is published it is never written again, only read, and
// any number of readers may hold `&T` to it at once (multiple `Reader`s, each
// on their own thread). Sharing those concurrent `&T` references across
// threads is exactly what `T: Sync` means, so bounding on it here is what
// makes `&Shared<T>` shareable: the same requirement `RwLock<T>` has, not
// `Mutex<T>`'s weaker one.
unsafe impl<T: Sync> Sync for Shared<T> {}

/// Appends entries up to a fixed, once-allocated capacity. `push` is the only
/// way to reach a slot, so at most one writer may exist per [`published`] pair.
pub struct Publisher<T: Copy> {
    shared: sync::Arc<Shared<T>>,
    written: usize,
}

/// Observes the prefix a [`Publisher`] has published so far. `Clone`, so any
/// number of readers may observe the same channel.
pub struct Reader<T: Copy> {
    shared: sync::Arc<Shared<T>>,
}

/// A single-writer, multi-reader channel over a fixed capacity, allocated once.
/// `Err(`[`CapacityError`]`)` when that many slots cannot be allocated; nothing is allocated then.
///
/// Building the channel only needs `T: Copy`, but sending a [`Reader`] to
/// another thread additionally needs `T: `[`Sync`]: cloned readers on
/// different threads then hold concurrent `&T` into the same slots, which is
/// exactly what `Sync` promises is sound.
///
/// A `Send`-but-not-`Sync` `Copy` type builds a channel fine, but a `Reader`
/// over it can't cross threads:
///
/// ```compile_fail
/// use core::cell::Cell;
/// use core::ptr::NonNull;
///
/// #[derive(Clone, Copy)]
/// struct Evil(NonNull<Cell<u64>>);
/// unsafe impl Send for Evil {} // still not Sync
///
/// let (_publisher, reader) = ladde::adapter::published::<Evil>(1).unwrap();
/// std::thread::spawn(move || drop(reader)); // `Reader<Evil>` is not `Send`
/// ```
pub fn published<T: Copy>(capacity: usize) -> Result<(Publisher<T>, Reader<T>), CapacityError> {
    let mut slots = Vec::new();
    slots
        .try_reserve_exact(capacity)
        .map_err(|_| CapacityError)?;
    // SAFETY: the reservation holds `capacity` slots, and a slot is an `UnsafeCell` over
    // `MaybeUninit`, valid uninitialized; setting the length instead of pushing keeps a
    // zero-sized slot with a huge capacity from looping.
    #[cfg(not(loom))]
    unsafe {
        slots.set_len(capacity);
    }
    // loom's cell registers itself on construction, so each slot is built.
    #[cfg(loom)]
    slots.extend((0..capacity).map(|_| sync::UnsafeCell::new(MaybeUninit::uninit())));
    let slots = slots.into_boxed_slice();
    let shared = sync::Arc::new(Shared {
        slots,
        published: AtomicUsize::new(0),
    });
    let reader = Reader {
        shared: sync::Arc::clone(&shared),
    };
    let publisher = Publisher { shared, written: 0 };
    Ok((publisher, reader))
}

impl<T: Copy> Publisher<T> {
    /// Writes `value` to the next slot and publishes it. `Err(`[`Full`]`)` once
    /// every slot has been written.
    pub fn push(&mut self, value: T) -> Result<(), Full> {
        let slot = self.shared.slots.get(self.written).ok_or(Full)?;
        // SAFETY: slot `written` sits at or past `published`, so no reader has
        // read it yet, and the single writer never writes a slot twice.
        slot.with_mut(|ptr| unsafe { *ptr = MaybeUninit::new(value) });
        self.written += 1;
        // Release: pairs with a reader's acquire-load in `snapshot`, so
        // observing this store also makes the write above, and every write
        // before it, visible.
        self.shared.published.store(self.written, Ordering::Release);
        Ok(())
    }
}

impl<T: Copy> Clone for Reader<T> {
    fn clone(&self) -> Self {
        Self {
            shared: sync::Arc::clone(&self.shared),
        }
    }
}

impl<T: Copy> Reader<T> {
    /// The number of entries published as of this call: the one acquire-load
    /// both `snapshot` bodies below build on, so a weaker ordering here is
    /// wrong for either of them identically.
    fn published_len(&self) -> usize {
        // Acquire: pairs with the writer's release-store in `push`, so every
        // slot below the returned length was fully written before this
        // thread observed that length, and no one writes a published slot
        // again.
        self.shared.published.load(Ordering::Acquire)
    }

    /// The published prefix as of this call.
    #[cfg(not(loom))]
    pub fn snapshot(&self) -> &[T] {
        let len = self.published_len();
        // SAFETY: `published_len`'s acquire-load means every slot below
        // `len` is fully written and never written again. `sync::UnsafeCell<U>`
        // here is `#[repr(transparent)]` over `U`, and `MaybeUninit<T>` is
        // documented to share `T`'s size and alignment, so the initialized
        // prefix may be read back as `&[T]`.
        unsafe { core::slice::from_raw_parts(self.shared.slots.as_ptr().cast::<T>(), len) }
    }

    /// The published prefix as of this call.
    ///
    /// loom's cell carries extra bookkeeping per slot, so a slot is not
    /// layout-compatible with `T`; this copies the prefix out instead of the
    /// reinterpret cast the non-loom `snapshot` uses.
    #[cfg(loom)]
    pub fn snapshot(&self) -> Vec<T> {
        (0..self.published_len())
            .map(|i| {
                // SAFETY: `i` is below `published_len`'s acquire-load, so the
                // writer finished this slot before this thread observed that
                // length.
                self.shared.slots[i].with(|ptr| unsafe { (*ptr).assume_init() })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests;
