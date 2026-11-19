//! `motif_pairs`: the exact best pair of a series ([`ladde::motif::motif_pairs`]) as NumPy
//! arrays.

use numpy::PyArray1;
use numpy::PyArray2;
use numpy::ndarray::Array2;
use pyo3::exceptions::PyMemoryError;
use pyo3::prelude::*;

use crate::batch::{resolve_scale, resolve_threads};
use crate::convert::Floats;
use crate::err::{ScaleArg, window_length};

type MotifPairsResult<'py> = (Bound<'py, PyArray2<i64>>, Bound<'py, PyArray1<f64>>, f64);

/// The best pair of `t`: the pair of highest exact correlation among any two windows outside each
/// other's exclusion zone `ceil(m/4)`, ties to the smaller `(a, b)`, with `a < b`. It is the pair
/// `matrix_profile`'s `P` and `I` hold (the smallest distance, its window and neighbor ordered),
/// found without finishing the profile, so it is faster on a series with a pair clearly nearer
/// than most. Returns `(indices, distances, scale)`: `indices` is an int64 array of shape
/// `(found, 2)`, the pair's two window starts with the earlier first, `distances` a float64 array
/// of shape `(found,)`, the earlier window's distance to the later one, and `scale` the tick size
/// used.
/// `found` is 1, or 0 when the series has no window with a finite distance to another (shorter
/// than a window, or all gaps); nothing is padded. `scale` and `threads` mean what they do in
/// `matrix_profile`, and the result does not depend on `threads`. Releases the GIL while it runs.
#[pyfunction]
#[pyo3(signature = (t, m, *, scale=None, threads=None))]
pub(crate) fn motif_pairs<'py>(
    py: Python<'py>,
    t: Floats<'py>,
    m: i64,
    scale: Option<ScaleArg>,
    threads: Option<&Bound<'py, PyAny>>,
) -> PyResult<MotifPairsResult<'py>> {
    let m = window_length(m)?;
    let threads = resolve_threads(threads)?;
    let series = t.as_array().to_vec();
    let scale = resolve_scale(&series, m, scale)?;
    let found = py
        .detach(|| ladde::motif::motif_pairs(&series[..], m, scale, threads))
        .map_err(|_| PyMemoryError::new_err("the series is too large to allocate"))?;

    let indices: Vec<i64> = found
        .iter()
        .flat_map(|p| [p.a as i64, p.b as i64])
        .collect();
    let distances: Vec<f64> = found.iter().map(|p| p.distance).collect();
    let indices =
        Array2::from_shape_vec((found.len(), 2), indices).expect("two indices for each pair");
    Ok((
        PyArray2::from_owned_array(py, indices),
        PyArray1::from_vec(py, distances),
        scale.get(),
    ))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(motif_pairs, module)?)?;
    Ok(())
}
