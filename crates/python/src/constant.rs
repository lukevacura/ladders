//! A caller's constant-window predicate, STUMPY's `T_subseq_isconstant`, as the one flag per
//! window the core reads: a boolean array, or a callable `f(a, w)` applied to the series (its
//! non-finite samples as NaN, as STUMPY hands it over) and the window length, returning one
//! boolean per window. Nothing here decides what a window is: wrong lengths, wrong dtypes and a
//! failing callable are the caller's, and raise before any computation starts.

use core::num::NonZeroUsize;
use std::collections::VecDeque;

use ladde::stats::Series;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

/// A series' flags, `None` for the default rule (a window is constant when its samples are equal).
pub(crate) type Flags = Option<Vec<bool>>;

fn windows(len: usize, m: NonZeroUsize) -> usize {
    (len + 1).saturating_sub(m.get())
}

fn nan_for_gaps(series: &[f64]) -> Vec<f64> {
    series
        .iter()
        .map(|&x| if x.is_finite() { x } else { f64::NAN })
        .collect()
}

fn boolean_array<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let array = py.import("numpy")?.call_method1("asarray", (value,))?;
    let kind: String = array.getattr("dtype")?.getattr("kind")?.extract()?;
    if kind != "b" {
        let dtype = array.getattr("dtype")?.str()?;
        return Err(PyValueError::new_err(format!(
            "{name} must hold booleans, got dtype {dtype}"
        )));
    }
    Ok(array)
}

fn one_dimensional(array: &Bound<'_, PyAny>, name: &str, windows: usize) -> PyResult<Vec<bool>> {
    let ndim: usize = array.getattr("ndim")?.extract()?;
    if ndim != 1 {
        return Err(PyValueError::new_err(format!(
            "{name} must be one-dimensional, got {ndim} dimensions"
        )));
    }
    let flags: PyReadonlyArray1<'_, bool> = array.extract()?;
    let flags = flags.as_array().to_vec();
    if flags.len() != windows {
        return Err(PyValueError::new_err(format!(
            "{name} must have one flag per window ({windows}), got {}",
            flags.len()
        )));
    }
    Ok(flags)
}

/// The flags `arg` stands for over `series` at window length `m`: `None` for no predicate, a
/// boolean array of one flag per window, or a callable `f(a, w)` returning one. `name` is the
/// keyword it came in by, for the error.
pub(crate) fn flags_of(
    py: Python<'_>,
    arg: Option<&Bound<'_, PyAny>>,
    series: &[f64],
    m: NonZeroUsize,
    name: &str,
) -> PyResult<Flags> {
    let Some(arg) = arg.filter(|arg| !arg.is_none()) else {
        return Ok(None);
    };
    let windows = windows(series.len(), m);
    let given;
    let value = if arg.is_callable() {
        let a = PyArray1::from_vec(py, nan_for_gaps(series));
        given = arg.call1((a, m.get()))?;
        &given
    } else {
        arg
    };
    let array = boolean_array(py, value, name)?;
    one_dimensional(&array, name, windows).map(Some)
}

/// [`flags_of`] for a set of series sharing one `arg`: `None`; a callable applied to every series;
/// a list or tuple with one entry (`None`, an array, or a callable) for each; or one
/// two-dimensional boolean array, a row per series.
pub(crate) fn flags_of_each(
    py: Python<'_>,
    arg: Option<&Bound<'_, PyAny>>,
    series: &[&[f64]],
    m: NonZeroUsize,
    name: &str,
) -> PyResult<Vec<Flags>> {
    let Some(arg) = arg.filter(|arg| !arg.is_none()) else {
        return Ok(vec![None; series.len()]);
    };
    if arg.is_callable() {
        return series
            .iter()
            .map(|s| flags_of(py, Some(arg), s, m, name))
            .collect();
    }
    if arg.is_instance_of::<PyList>() || arg.is_instance_of::<PyTuple>() {
        let items: Vec<Bound<'_, PyAny>> = arg.try_iter()?.collect::<PyResult<_>>()?;
        if items.len() != series.len() {
            return Err(PyValueError::new_err(format!(
                "{name} must have one entry per series ({}), got {}",
                series.len(),
                items.len()
            )));
        }
        return items
            .iter()
            .zip(series)
            .map(|(item, s)| flags_of(py, Some(item), s, m, name))
            .collect();
    }
    let array = boolean_array(py, arg, name)?;
    let rows: PyReadonlyArray2<'_, bool> = array.extract().map_err(|_| {
        PyValueError::new_err(format!(
            "{name} must be a two-dimensional array of booleans"
        ))
    })?;
    let rows = rows.as_array();
    if rows.nrows() != series.len() {
        return Err(PyValueError::new_err(format!(
            "{name} must have one row per series ({}), got {}",
            series.len(),
            rows.nrows()
        )));
    }
    rows.rows()
        .into_iter()
        .zip(series)
        .map(|(row, s)| {
            let expected = windows(s.len(), m);
            if row.len() != expected {
                return Err(PyValueError::new_err(format!(
                    "{name} must have one flag per window ({expected}) in every row, got {}",
                    row.len()
                )));
            }
            Ok(Some(row.to_vec()))
        })
        .collect()
}

/// `values` as a core series carrying `flags`, the flags checked against `m` once more.
pub(crate) fn series<'a>(
    values: &'a [f64],
    m: NonZeroUsize,
    flags: &'a Flags,
    name: &str,
) -> PyResult<Series<'a>> {
    match flags {
        None => Ok(Series::from(values)),
        Some(flags) => Series::with_constant_windows(values, m, flags).map_err(|e| {
            PyValueError::new_err(format!(
                "{name} must have one flag per window ({}), got {}",
                e.expected, e.found
            ))
        }),
    }
}

/// A streaming caller's predicate (STUMPY's `T_subseq_isconstant_func`): a callable `f(a, w)`
/// applied, as each sample arrives, to the latest `m` samples (non-finite ones as NaN), returning
/// one flag, that of the window those samples are.
pub(crate) struct WindowPredicate {
    function: Py<PyAny>,
    m: NonZeroUsize,
    tail: VecDeque<f64>,
}

impl WindowPredicate {
    pub(crate) fn new(function: &Bound<'_, PyAny>, m: NonZeroUsize, name: &str) -> PyResult<Self> {
        if !function.is_callable() {
            return Err(PyValueError::new_err(format!(
                "{name} must be a callable, got {}",
                function.get_type().name()?
            )));
        }
        Ok(Self {
            function: function.clone().unbind(),
            m,
            tail: VecDeque::new(),
        })
    }

    /// The latest samples, oldest first: what a checkpoint keeps of them.
    pub(crate) fn tail(&self) -> Vec<f64> {
        self.tail.iter().copied().collect()
    }

    /// This predicate with its latest samples set to `tail` (the last `m` of what it was fed).
    pub(crate) fn with_tail(mut self, tail: Vec<f64>) -> PyResult<Self> {
        if tail.len() > self.m.get() {
            return Err(PyValueError::new_err("invalid checkpoint: Inconsistent"));
        }
        self.tail = tail.into();
        Ok(self)
    }

    /// Records `history` as what was fed so far (its last `m` samples), after a warm start whose
    /// flags came from [`flags_of`].
    pub(crate) fn warm(&mut self, history: &[f64]) {
        let from = history.len().saturating_sub(self.m.get());
        self.tail = nan_for_gaps(&history[from..]).into();
    }

    /// Folds in `x`; the flag of the window it completes, `None` while no window is complete.
    pub(crate) fn push(&mut self, py: Python<'_>, x: f64, name: &str) -> PyResult<Option<bool>> {
        let before = self.tail.clone();
        self.tail
            .push_back(if x.is_finite() { x } else { f64::NAN });
        if self.tail.len() > self.m.get() {
            self.tail.pop_front();
        }
        if self.tail.len() < self.m.get() {
            return Ok(None);
        }
        let flag = self.flag_of_tail(py, name);
        if flag.is_err() {
            // The sample is not folded in, so it is not part of what the predicate has seen.
            self.tail = before;
        }
        flag.map(Some)
    }

    fn flag_of_tail(&self, py: Python<'_>, name: &str) -> PyResult<bool> {
        let a = PyArray1::from_vec(py, self.tail());
        let given = self.function.bind(py).call1((a, self.m.get()))?;
        let array = boolean_array(py, &given, name)?;
        Ok(one_dimensional(&array, name, 1)?[0])
    }
}
