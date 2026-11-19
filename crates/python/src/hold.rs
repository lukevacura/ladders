//! Hidden test hook: while armed, `left_profile`, `subspace`, `mdl` and `mmotifs` park right after it releases the GIL until
//! Python calls `_hold_release`. Python can only make that call if the GIL really was released,
//! so a held GIL shows up as a bounded wait that ran out, never as a timing measurement.

use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::Duration;

use pyo3::prelude::*;

/// How long a parked call or a waiting caller gives up and reports failure; far above any
/// correct run, which never waits on the clock at all.
const HANG_GUARD: Duration = Duration::from_secs(60);

#[derive(Default)]
struct State {
    armed: bool,
    parked: usize,
    released: bool,
    gave_up: bool,
}

static STATE: Mutex<State> = Mutex::new(State {
    armed: false,
    parked: 0,
    released: false,
    gave_up: false,
});
static CHANGED: Condvar = Condvar::new();

fn state() -> MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Called with the GIL released: parks while armed and not yet released.
pub(crate) fn park_if_armed() {
    let mut guard = state();
    if !guard.armed {
        return;
    }
    guard.parked += 1;
    CHANGED.notify_all();
    let (mut guard, timeout) = CHANGED
        // timing-ok: a hang-guard bound only; correct code never waits on the clock
        .wait_timeout_while(guard, HANG_GUARD, |s| !s.released)
        .unwrap_or_else(|e| e.into_inner());
    guard.gave_up |= timeout.timed_out();
}

/// Hidden: the next `left_profile`, `subspace`, `mdl` or `mmotifs` call parks (with the GIL released) until `_hold_release`.
#[pyfunction]
pub(crate) fn _hold_arm() {
    *state() = State {
        armed: true,
        ..State::default()
    };
}

/// Hidden: blocks (GIL released) until a call is parked; false if none parked in time.
#[pyfunction]
pub(crate) fn _hold_wait_parked(py: Python<'_>) -> bool {
    py.detach(|| {
        let (guard, _) = CHANGED
            // timing-ok: a hang-guard bound only; correct code never waits on the clock
            .wait_timeout_while(state(), HANG_GUARD, |s| s.parked == 0)
            .unwrap_or_else(|e| e.into_inner());
        guard.parked > 0
    })
}

/// Hidden: lets parked calls continue and disarms; true if no call ever gave up waiting.
#[pyfunction]
pub(crate) fn _hold_release() -> bool {
    let mut guard = state();
    guard.released = true;
    guard.armed = false;
    CHANGED.notify_all();
    !guard.gave_up
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_hold_arm, module)?)?;
    module.add_function(wrap_pyfunction!(_hold_wait_parked, module)?)?;
    module.add_function(wrap_pyfunction!(_hold_release, module)?)?;
    Ok(())
}
