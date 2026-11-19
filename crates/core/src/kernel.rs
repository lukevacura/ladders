//! The parameters the kernel is monomorphized over.

/// A metric's per-diagonal arithmetic: how two windows are compared along one diagonal of the
/// profile (`start`, `advance`). What derived procedures need of the same metric (validity, the
/// exact pair distance and its order, the reported hit) is `Measure`, which extends this.
pub trait Metric {
    /// What one step of the diagonal reads of each window, computed exactly from the ticks.
    type Step: Copy;
    /// State carried from `(i − 1, j − 1)` to `(i, j)` along one diagonal.
    type Diagonal: Copy;

    /// Exact state for the pair whose windows are `a` and `b` (tick slices of length m).
    fn start(a: &[i64], a_step: &Self::Step, b: &[i64], b_step: &Self::Step) -> Self::Diagonal;

    /// O(1) step from the pair one diagonal position earlier.
    fn advance(previous: Self::Diagonal, a: &Self::Step, b: &Self::Step) -> Self::Diagonal;
}

/// Folds the candidates for one row into that row's result. Built fresh per row, from that
/// row's own window and `m`, rather than from a default state.
pub(crate) trait Reducer {
    type Candidate;
    type Output;

    fn offer(&mut self, candidate: Self::Candidate);
    fn finish(self) -> Self::Output;
}
