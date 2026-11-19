//! The `Stream` trait every streaming engine implements.

use alloc::vec::Vec;

use crate::entry::Entry;
use crate::index::SampleIndex;

/// A causal left nearest-neighbor matrix profile fed one sample at a time.
pub trait Stream {
    /// `None` until this sample completes a window; then `(row, entry)` for the newest one.
    fn update(&mut self, x: f64) -> Option<(SampleIndex, Entry)> {
        self.update_flagged(x, None)
    }

    /// [`Self::update`] with the caller's flag for the window this sample completes being
    /// constant (`None` for the default rule); ignored while no window completes.
    fn update_flagged(&mut self, x: f64, constant: Option<bool>) -> Option<(SampleIndex, Entry)>;

    /// Folds in every sample of `xs`, in order, appending each completed window's `(row,
    /// entry)` to `out` — exactly what the same samples, fed one at a time to
    /// [`Self::update`], would have appended. Never clears `out`, so a caller who
    /// wants just this chunk's output clears it first; one with capacity reserved for the
    /// chunk sees no allocation here.
    fn update_many(&mut self, xs: &[f64], out: &mut Vec<(SampleIndex, Entry)>) {
        for &x in xs {
            if let Some(row) = self.update(x) {
                out.push(row);
            }
        }
    }
}

#[cfg(test)]
mod tests;
