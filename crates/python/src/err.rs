//! Turns an out-of-range Python argument into a `ValueError`, before it ever reaches core: core
//! itself never panics on these (`NonZeroUsize`/`Scale` reject them by construction), but a
//! Python caller gets a message instead of an exception from an unrelated type.

use core::num::NonZeroUsize;

use ladde::quantize::{Scale, to_ticks};
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;

/// A window length: STUMPY's own minimum of 3 (below that, a window's variance and every
/// neighbor comparison degenerate), cast to `usize` only once it has passed that floor.
pub(crate) fn window_length(m: i64) -> PyResult<NonZeroUsize> {
    if m < 3 {
        return Err(PyValueError::new_err(format!(
            "m must be at least 3, got {m}"
        )));
    }
    let m =
        usize::try_from(m).map_err(|_| PyValueError::new_err(format!("m is too large: {m}")))?;
    Ok(NonZeroUsize::new(m).expect("m >= 3 is checked above"))
}

/// A bounded stream's trailing window `w`: at least `m`, the shortest span that could ever
/// hold one.
pub(crate) fn window_bound(w: i64, m: NonZeroUsize) -> PyResult<NonZeroUsize> {
    if w < m.get() as i64 {
        return Err(PyValueError::new_err(format!(
            "w must be at least m ({}), got {w}",
            m.get()
        )));
    }
    let w =
        usize::try_from(w).map_err(|_| PyValueError::new_err(format!("w is too large: {w}")))?;
    Ok(NonZeroUsize::new(w).expect("w >= m >= 3 is checked above"))
}

/// A tick's size: positive and finite, same as [`Scale::new`]'s own contract.
pub(crate) fn tick_scale(scale: f64) -> PyResult<Scale> {
    Scale::new(scale).ok_or_else(|| {
        PyValueError::new_err(format!("scale must be positive and finite, got {scale}"))
    })
}

/// A `scale` argument that may be inferred: a tick size, or `"finest"`, the opt-in finer rule.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ScaleArg {
    Tick(f64),
    Finest,
}

impl<'a, 'py> FromPyObject<'a, 'py> for ScaleArg {
    type Error = PyErr;

    fn extract(ob: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if ob.is_instance_of::<pyo3::types::PyString>() {
            return match ob.extract::<String>()?.as_str() {
                "finest" => Ok(Self::Finest),
                other => Err(PyValueError::new_err(format!(
                    "scale must be a number, None or \"finest\", got {other:?}"
                ))),
            };
        }
        Ok(Self::Tick(ob.extract()?))
    }
}

/// How a caller gives a coarser `scale`: as an argument of the call, when starting a stream, or
/// when building a pattern library.
#[derive(Clone, Copy)]
pub(crate) enum Remedy {
    Pass,
    Restart,
    Library,
}

/// Every finite sample of `samples` has a tick at `scale`: one beyond `2**53` ticks would silently
/// become a gap, so it is a `ValueError` naming the coarser scale (`scale` times a power of two)
/// that holds it.
pub(crate) fn countable(samples: &[f64], scale: Scale, remedy: Remedy) -> PyResult<()> {
    match samples
        .iter()
        .find(|x| x.is_finite() && to_ticks(**x, scale).is_none())
    {
        None => Ok(()),
        Some(&x) => Err(uncountable(x, scale, remedy)),
    }
}

fn uncountable(x: f64, scale: Scale, remedy: Remedy) -> PyErr {
    let mut coarser = scale.get();
    while let Some(candidate) = Scale::new(coarser * 2.0) {
        coarser = candidate.get();
        if to_ticks(x, candidate).is_some() {
            break;
        }
    }
    let remedy = match remedy {
        Remedy::Pass => "pass",
        Remedy::Restart => "start the stream with",
        Remedy::Library => "build the library with",
    };
    PyValueError::new_err(format!(
        "scale={:?} cannot count the sample {x:?}: it is beyond 2**53 ticks and would become a \
         gap; STUMPY computes in floats and keeps it; {remedy} scale={coarser:?} to keep it",
        scale.get()
    ))
}

/// A checkpoint that will not restore, as a `ValueError` naming the core variant.
pub(crate) fn checkpoint_error(error: ladde::checkpoint::CheckpointError) -> PyErr {
    PyValueError::new_err(format!("invalid checkpoint: {error:?}"))
}

/// A count that sizes an up-front allocation (`what` names it): at least 1, and no more than
/// `MAX_COUNT`, so a mistyped argument is a message rather than an allocation failure.
pub(crate) fn bounded_count(what: &str, value: i64) -> PyResult<usize> {
    if value < 1 {
        return Err(PyValueError::new_err(format!(
            "{what} must be at least 1, got {value}"
        )));
    }
    if value > MAX_COUNT {
        return Err(PyValueError::new_err(format!(
            "{what} must be at most {MAX_COUNT}, got {value}"
        )));
    }
    Ok(value as usize)
}

const MAX_COUNT: i64 = 1 << 28;

/// A result of `rows` by `columns` cells, no more than `MAX_COUNT` of them, so a huge `columns`
/// is a message rather than an allocation failure.
pub(crate) fn bounded_cells(what: &str, rows: usize, columns: usize) -> PyResult<()> {
    match rows.checked_mul(columns) {
        Some(cells) if cells as u64 <= MAX_COUNT as u64 => Ok(()),
        _ => Err(PyValueError::new_err(format!(
            "{what} gives {rows} x {columns} cells, more than the {MAX_COUNT} allowed"
        ))),
    }
}

/// A length that is only compared against (the regime length, an end zone factor): not
/// negative.
pub(crate) fn non_negative(what: &str, value: i64) -> PyResult<usize> {
    usize::try_from(value)
        .map_err(|_| PyValueError::new_err(format!("{what} must not be negative, got {value}")))
}

/// A multi-dimensional input the core refused, as a `ValueError` saying what is wrong.
pub(crate) fn multi_error(error: ladde::multidim::MultiError) -> PyErr {
    use ladde::multidim::MultiError;
    PyValueError::new_err(match error {
        MultiError::NoDimensions => "t must hold at least one dimension",
        MultiError::UnequalLengths => "every dimension of t must have the same length",
        MultiError::IncludeOutOfRange => "include must hold dimensions of t",
        MultiError::WindowTooLong => "m is larger than the length of t",
        MultiError::BitsOutOfRange => "n_bit must be between 1 and 8",
        MultiError::WindowOutOfRange => "subseq_idx and nn_idx must be starts of windows of t",
        MultiError::WrongPairCount => "subseq_idx and nn_idx must have one entry per dimension",
        MultiError::KOutOfRange => "k must be below the number of dimensions of t",
        MultiError::ProfileShape => {
            "P and I must have one row per dimension of T and one column per window"
        }
        MultiError::CutoffCount => "cutoffs must have one entry per dimension of T",
        MultiError::ScaleCount => "scales must have one entry per dimension of t",
        // The enum is non-exhaustive: a variant this file does not know is named as core names it.
        _ => return PyValueError::new_err(format!("the input was refused: {error:?}")),
    })
}

/// STUMPY's `name` argument, which this library has no way to honor: `NotImplementedError` when it
/// is given (not `None`), so the call never silently runs without it.
pub(crate) fn require_none(name: &str, value: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
    match value {
        Some(value) if !value.is_none() => Err(PyNotImplementedError::new_err(format!(
            "{name} is not available: this library computes every statistic exactly itself"
        ))),
        _ => Ok(()),
    }
}

/// STUMPY's `atol`: not negative and not NaN. STUMPY would add it as it is; a tolerance that
/// shrinks the threshold or makes every comparison false is refused rather than run.
pub(crate) fn tolerance(atol: f64) -> PyResult<f64> {
    if atol.is_nan() || atol < 0.0 {
        return Err(PyValueError::new_err(format!(
            "atol must be a number that is not negative, got {atol}"
        )));
    }
    Ok(atol)
}
