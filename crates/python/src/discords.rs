//! `discords`: the top-k batch discords ([`ladde::batch::discords`]) as NumPy arrays.

use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::batch::{resolve_scale, resolve_threads};
use crate::constant::{flags_of, series as flagged};
use crate::convert::{Floats, discord_index_and_distance};
use crate::err::{ScaleArg, window_length};

type DiscordsResult<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i64>>);

/// The `k` discords of `t`: the windows whose exact nearest neighbor, on either side, is
/// farthest, as `(distances, indices)`, farthest first. Unlike sorting `matrix_profile`'s `P`
/// (STUMPY has no `discords`), picks stay pairwise outside the exclusion zone `ceil(m/4)`, a
/// window inside the zone of a farther pick is skipped, and the order is by exact distance, ties
/// to the smaller index. Fewer than `k` come back when the zones leave no room; a window with no
/// neighbor is never listed. `scale` and `threads` mean what they do in `matrix_profile`, and
/// the result does not depend on `threads`. Releases the GIL while it runs.
/// `T_A_subseq_isconstant` is STUMPY's (see `matrix_profile`): which windows are constant.
#[pyfunction]
#[pyo3(signature = (t, m, k=1, scale=None, threads=None, T_A_subseq_isconstant=None))]
#[allow(non_snake_case)]
pub(crate) fn discords<'py>(
    py: Python<'py>,
    t: Floats<'py>,
    m: i64,
    k: i64,
    scale: Option<ScaleArg>,
    threads: Option<&Bound<'py, PyAny>>,
    T_A_subseq_isconstant: Option<&Bound<'py, PyAny>>,
) -> PyResult<DiscordsResult<'py>> {
    let m = window_length(m)?;
    let k = usize::try_from(k).map_err(|_| PyValueError::new_err("k must be non-negative"))?;
    let threads = resolve_threads(threads)?;
    let series = t.as_array().to_vec();
    let scale = resolve_scale(&series, m, scale)?;

    let name = "T_A_subseq_isconstant";
    let flags = flags_of(py, T_A_subseq_isconstant, &series, m, name)?;
    let input = flagged(&series, m, &flags, name)?;
    let found = py.detach(|| ladde::batch::discords(input, m, scale, threads, k));

    let (indices, distances): (Vec<i64>, Vec<f64>) =
        found.into_iter().map(discord_index_and_distance).unzip();
    Ok((
        PyArray1::from_vec(py, distances),
        PyArray1::from_vec(py, indices),
    ))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(discords, module)?)?;
    Ok(())
}
