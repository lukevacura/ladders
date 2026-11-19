//! The batch's one runner of independent tasks: the seed pre-pass's groups and sorts, and any
//! later pass that fans a list of tasks out, call [`run_tasks`] instead of each choosing between
//! the threaded executor and a plain loop. With the `parallel` feature (and not under loom) the
//! tasks run on [`crate::par::Cpu`]; without it they run in order on the calling thread. Either
//! way the results are in task order, so what a caller computes never depends on the thread count.

use alloc::vec::Vec;
use core::num::NonZeroUsize;

/// Runs `tasks` tasks on up to `threads` threads, results in task order.
pub(super) fn run_tasks<R: Send>(
    threads: NonZeroUsize,
    tasks: usize,
    task: impl Fn(usize) -> R + Sync,
) -> Vec<R> {
    #[cfg(all(feature = "parallel", not(loom)))]
    {
        use crate::par::Executor;
        crate::par::Cpu { threads }.run(tasks, task)
    }
    #[cfg(not(all(feature = "parallel", not(loom))))]
    {
        let _ = threads;
        (0..tasks).map(task).collect()
    }
}
