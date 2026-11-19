//! `atsc`, `allc` and `chains`: time-series chains ([`ladde::chain`]) as NumPy arrays,
//! in STUMPY's shapes.

use ladde::chain::{AllChains, ChainInputError, all_chains, anchored_chain, chains_from_series};
use ladde::index::SampleIndex;
use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::batch::{resolve_scale, resolve_threads};
use crate::convert::{Floats, Ints};
use crate::err::{ScaleArg, window_length};

/// `allc`'s return shape: `(S, C)`, the chains and the longest one.
type AllChainsResult<'py> = (Vec<Bound<'py, PyArray1<i64>>>, Bound<'py, PyArray1<i64>>);

fn neighbors(what: &str, raw: &Ints<'_>) -> PyResult<Vec<SampleIndex>> {
    raw.as_array()
        .iter()
        .map(|&i| match u64::try_from(i) {
            Ok(index) => Ok(SampleIndex::new(index).expect("an i64 is below u64::MAX")),
            Err(_) if i == -1 => Ok(SampleIndex::NONE),
            Err(_) => Err(PyValueError::new_err(format!(
                "{what} entries must be window indices or -1, got {i}"
            ))),
        })
        .collect()
}

fn input_error(error: ChainInputError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn chain_array<'py>(py: Python<'py>, chain: &[usize]) -> Bound<'py, PyArray1<i64>> {
    PyArray1::from_vec(py, chain.iter().map(|&w| w as i64).collect())
}

fn all_chains_arrays<'py>(py: Python<'py>, found: &AllChains) -> AllChainsResult<'py> {
    let chains = found.chains.iter().map(|c| chain_array(py, c)).collect();
    (chains, chain_array(py, &found.longest))
}

/// The anchored time series chain from window `j`: `j`, then each next window while the two
/// point at each other (`IR[k] == k'` and `IL[k'] == k`), as STUMPY's `atsc`. `IL` and `IR` are
/// left and right nearest-neighbor indices (`-1` for none), e.g. `matrix_profile`'s third and
/// fourth outputs. Like those, they must point their own way: `IL[k] < k` and `IR[k] > k`;
/// anything else is a `ValueError`, as is `j` outside the windows.
#[pyfunction]
#[pyo3(signature = (IL, IR, j))]
#[allow(non_snake_case)]
pub(crate) fn atsc<'py>(
    py: Python<'py>,
    IL: Ints<'py>,
    IR: Ints<'py>,
    j: i64,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let (left, right) = (neighbors("IL", &IL)?, neighbors("IR", &IR)?);
    let anchor = usize::try_from(j).map_err(|_| input_error(ChainInputError::OutOfRange))?;
    let chain = anchored_chain(&left, &right, anchor).map_err(input_error)?;
    Ok(chain_array(py, &chain))
}

/// The all-chain set and the longest chain, as STUMPY's `allc`: `(S, C)`. `S` lists every
/// maximal chain, ordered by first window (STUMPY's order is arbitrary), and `C` is the first
/// of the longest ones. Inputs as for `atsc`.
#[pyfunction]
#[pyo3(signature = (IL, IR))]
#[allow(non_snake_case)]
pub(crate) fn allc<'py>(
    py: Python<'py>,
    IL: Ints<'py>,
    IR: Ints<'py>,
) -> PyResult<AllChainsResult<'py>> {
    let (left, right) = (neighbors("IL", &IL)?, neighbors("IR", &IR)?);
    let found = all_chains(&left, &right).map_err(input_error)?;
    Ok(all_chains_arrays(py, &found))
}

/// `allc` over the exact left and right neighbors of `t`'s windows (what `matrix_profile`
/// returns), in one call: `(S, C)`. `scale` and `threads` mean what they do in
/// `matrix_profile`, and the result does not depend on `threads`. Releases the GIL while it runs.
#[pyfunction]
#[pyo3(signature = (t, m, scale=None, threads=None))]
pub(crate) fn chains<'py>(
    py: Python<'py>,
    t: Floats<'py>,
    m: i64,
    scale: Option<ScaleArg>,
    threads: Option<&Bound<'py, PyAny>>,
) -> PyResult<AllChainsResult<'py>> {
    let m = window_length(m)?;
    let threads = resolve_threads(threads)?;
    let series = t.as_array().to_vec();
    let scale = resolve_scale(&series, m, scale)?;
    let found = py.detach(|| chains_from_series(&series, m, scale, threads));
    Ok(all_chains_arrays(py, &found))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(atsc, module)?)?;
    module.add_function(wrap_pyfunction!(allc, module)?)?;
    module.add_function(wrap_pyfunction!(chains, module)?)?;
    Ok(())
}
