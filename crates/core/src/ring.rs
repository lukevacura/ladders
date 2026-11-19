//! The last `capacity` records, readable as one contiguous slice.

use alloc::boxed::Box;
use alloc::vec;
use core::num::NonZeroUsize;

/// Writes each record at both `k` and `k + capacity`, so any run of the latest
/// `capacity` records is contiguous in the doubled storage.
pub struct MirroredRing<T: Copy> {
    storage: Box<[T]>,
    capacity: usize,
    next: usize,
    len: usize,
}

impl<T: Copy> MirroredRing<T> {
    /// The only allocation the ring makes.
    pub fn new(capacity: NonZeroUsize, fill: T) -> Self {
        let capacity = capacity.get();
        Self {
            storage: vec![fill; 2 * capacity].into_boxed_slice(),
            capacity,
            next: 0,
            len: 0,
        }
    }

    pub fn push(&mut self, value: T) {
        self.storage[self.next] = value;
        self.storage[self.next + self.capacity] = value;
        self.next += 1;
        if self.next == self.capacity {
            self.next = 0;
        }
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    /// Replaces the newest record, keeping both mirrored copies equal; nothing when empty.
    pub fn set_newest(&mut self, value: T) {
        if self.len == 0 {
            return;
        }
        let newest = if self.next == 0 {
            self.capacity - 1
        } else {
            self.next - 1
        };
        self.storage[newest] = value;
        self.storage[newest + self.capacity] = value;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The last [`Self::len`] records, oldest first.
    pub fn as_slice(&self) -> &[T] {
        let end = self.next + self.capacity;
        &self.storage[end - self.len..end]
    }

    /// Applies `f` once to each live record, oldest to newest, keeping both
    /// mirrored copies of every updated record equal.
    pub fn update_live(&mut self, mut f: impl FnMut(&mut T)) {
        let end = self.next + self.capacity;
        let start = end - self.len;
        for i in start..end {
            f(&mut self.storage[i]);
            let value = self.storage[i];
            let mirror = if i < self.capacity {
                i + self.capacity
            } else {
                i - self.capacity
            };
            self.storage[mirror] = value;
        }
    }
}

#[cfg(test)]
mod tests;
