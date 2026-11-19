//! Capacity: counts are clamped honestly, reservations are capped up front but never what may be
//! produced, and a size that cannot be allocated is a [`CapacityError`], never a panic.

use alloc::vec::Vec;

/// The slots could not be allocated: the capacity needs more than `isize::MAX` bytes, or the
/// allocator refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CapacityError;

/// What a growing structure, whose input has no known size, passes as the bound of a
/// caller's count: past it the buffer grows as entries arrive.
pub(crate) const UNBOUNDED_RESERVE: usize = 1 << 16;

/// The most entries any structure reserves up front, whatever its true bound: past it the
/// buffer grows during warm-up only.
pub(crate) const RESERVE_LIMIT: usize = 1 << 24;

/// A new vector with room for exactly `count` entries, or a
/// [`CapacityError`] when that many cannot be allocated: a count
/// that overflows `usize` or `isize::MAX` bytes, or one the allocator refuses. Nothing is
/// computed before the reservation, so an impossible size is known before any work.
pub(crate) fn reserved_vec<T>(count: u128) -> Result<Vec<T>, CapacityError> {
    let count = usize::try_from(count).map_err(|_| CapacityError)?;
    let mut vec = Vec::new();
    vec.try_reserve_exact(count).map_err(|_| CapacityError)?;
    Ok(vec)
}

/// What an infallible entry point does with a [`CapacityError`]: what any failed allocation does.
pub(crate) fn abort_on_capacity(_: CapacityError) -> ! {
    alloc::alloc::handle_alloc_error(core::alloc::Layout::new::<f64>())
}

/// A caller's `count` limited to `bound`, what the site's input can produce: the clamp every
/// batch count and every count-sized reservation goes through.
pub(crate) fn clamp_count(count: usize, bound: usize) -> usize {
    count.min(bound)
}

/// What to reserve for a caller's `count` of entries the site can hold at most `bound` of: the
/// whole of it, so nothing grows after warm-up, up to [`RESERVE_LIMIT`].
pub(crate) fn reserve_for(count: usize, bound: usize) -> usize {
    clamp_count(clamp_count(count, bound), RESERVE_LIMIT)
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::{CapacityError, reserved_vec};

    /// A count that cannot be allocated, by overflowing the address space or `isize::MAX` bytes,
    /// is a capacity error; one that can be is reserved exactly, however large it is next to the
    /// reservation limit.
    #[test]
    fn an_allocation_too_large_to_make_is_a_capacity_error() {
        assert_eq!(reserved_vec::<u64>(u128::MAX).err(), Some(CapacityError));
        assert_eq!(
            reserved_vec::<u64>(usize::MAX as u128 + 1).err(),
            Some(CapacityError)
        );
        assert_eq!(
            reserved_vec::<u64>(usize::MAX as u128).err(),
            Some(CapacityError)
        );
        assert_eq!(
            reserved_vec::<u8>(isize::MAX as u128 + 1).err(),
            Some(CapacityError)
        );
        assert_eq!(
            reserved_vec::<(u64, crate::entry::Entry)>(0)
                .unwrap()
                .capacity(),
            0
        );
        assert_eq!(
            reserved_vec::<(u64, crate::entry::Entry)>(1 << 25)
                .unwrap()
                .capacity(),
            1 << 25
        );
    }
}
