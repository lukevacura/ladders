//! `FlossStream`: a stream and a [`Floss`] fed from it inside Rust, reporting the corrected arc
//! curve of its last `width` left neighbors.

use ladde::floss::{Floss, corrected_arc_curve, fluss as fluss_of_index};
use ladde::index::SampleIndex;
use ladde::stream_api::Stream;
use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::constant::WindowPredicate;
use crate::convert::{Floats, Ints};
use crate::engine::Engine;
use crate::err::{
    Remedy, bounded_count, countable, non_negative, tick_scale, window_bound, window_length,
};

/// STUMPY's `fluss` shape: `(cac, regime_locs)`.
pub(crate) type FlussResult<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i64>>);

/// `l`, `n_regimes` and `excl_factor` as `fluss` takes them: `l` at least 1, `n_regimes` at least 1.
pub(crate) fn fluss_counts(
    l: i64,
    n_regimes: i64,
    excl_factor: i64,
) -> PyResult<(usize, usize, usize)> {
    Ok((
        bounded_count("L", l)?,
        usize::try_from(n_regimes)
            .ok()
            .filter(|&n| n >= 1)
            .ok_or_else(|| {
                PyValueError::new_err(format!("n_regimes must be at least 1, got {n_regimes}"))
            })?,
        non_negative("excl_factor", excl_factor)?,
    ))
}

/// `n_regimes` locations' worth of windows: `n_regimes - 1` at most the number of `windows`, the
/// most distinct locations there are. STUMPY fills the rest with repeated locations, which is a
/// `ValueError` here.
pub(crate) fn regimes_fit(n_regimes: usize, windows: usize) -> PyResult<()> {
    if n_regimes - 1 <= windows {
        return Ok(());
    }
    Err(PyValueError::new_err(format!(
        "n_regimes={n_regimes} asks for {} regime locations but there are only {windows} windows; \
         STUMPY fills the rest with repeated locations; pass n_regimes={} to get every location \
         there is",
        n_regimes - 1,
        windows + 1
    )))
}

/// The curve and locations as arrays; a `custom_iac` of the wrong length is a `ValueError`.
pub(crate) fn fluss_arrays<'py>(
    py: Python<'py>,
    result: Option<(Vec<f64>, Vec<usize>)>,
) -> PyResult<FlussResult<'py>> {
    let (curve, locations) =
        result.ok_or_else(|| PyValueError::new_err("custom_iac must have one value per window"))?;
    let locations = locations.into_iter().map(|c| c as i64).collect();
    Ok((
        PyArray1::from_vec(py, curve),
        PyArray1::from_vec(py, locations),
    ))
}

/// Regime boundaries of a left nearest-neighbor stream (streaming FLOSS).
///
/// `m` is the subsequence length (at least 3); `width` how many of the latest windows the
/// curve covers; `scale` the tick size samples are quantized to. With `w` the stream searches
/// only the last `w` samples (at least `m`); without it nothing is forgotten. `l` is the
/// regime length (default `m`, STUMPY's `L`) and `excl_factor` multiplies it into the end
/// zones forced to 1 (STUMPY's 5). `T_subseq_isconstant_func` is STUMPY's: a function `f(a, w)`
/// applied as each sample arrives to the latest `m` samples (non-finite ones as NaN), returning
/// one boolean saying whether that window is constant, in place of "all samples equal". It is
/// called holding the GIL, and an exception it raises propagates and leaves that sample out.
#[pyclass(module = "ladders")]
pub struct FlossStream {
    stream: Engine,
    floss: Floss,
    predicate: Option<WindowPredicate>,
}

/// The keyword the predicate comes in by, for the errors about it.
const PREDICATE: &str = "T_subseq_isconstant_func";

#[pymethods]
impl FlossStream {
    #[new]
    #[pyo3(signature = (m, width, scale, w=None, l=None, excl_factor=5, T_subseq_isconstant_func=None))]
    #[allow(non_snake_case)]
    fn new(
        m: i64,
        width: i64,
        scale: f64,
        w: Option<i64>,
        l: Option<i64>,
        excl_factor: i64,
        T_subseq_isconstant_func: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let m = window_length(m)?;
        let width = core::num::NonZeroUsize::new(bounded_count("width", width)?)
            .expect("width >= 1 is checked above");
        let scale = tick_scale(scale)?;
        let w = w.map(|w| window_bound(w, m)).transpose()?;
        let l = match l {
            None => m.get(),
            Some(l) => non_negative("l", l)?,
        };
        let excl_factor = non_negative("excl_factor", excl_factor)?;
        let predicate = T_subseq_isconstant_func
            .filter(|function| !function.is_none())
            .map(|function| WindowPredicate::new(function, m, PREDICATE))
            .transpose()?;
        let floss = Floss::new(m, width, horizon(m.get(), w), l, excl_factor);
        Ok(Self {
            stream: Engine::new(m, w, scale),
            floss,
            predicate,
        })
    }

    /// Folds in one sample; with a `T_subseq_isconstant_func`, one the function fails on is left
    /// out and its exception raised.
    fn update(&mut self, py: Python<'_>, x: f64) -> PyResult<()> {
        countable(&[x], self.stream.scale(), Remedy::Restart)?;
        let flag = match self.predicate.as_mut() {
            Some(predicate) => predicate.push(py, x, PREDICATE)?,
            None => None,
        };
        if let Some((row, entry)) = self.stream.update_flagged(x, flag) {
            self.floss.update(row, entry);
        }
        Ok(())
    }

    /// Folds in every sample of `xs`, in order: exactly what calling `update` once per sample
    /// would do.
    fn update_many<'py>(&mut self, py: Python<'py>, xs: Floats<'py>) -> PyResult<()> {
        let xs = xs.as_array().to_vec();
        countable(&xs, self.stream.scale(), Remedy::Restart)?;
        if self.predicate.is_some() {
            for x in xs {
                self.update(py, x)?;
            }
            return Ok(());
        }
        py.detach(|| {
            let mut out = Vec::new();
            self.stream.update_many(&xs, &mut out);
            for (row, entry) in out {
                self.floss.update(row, entry);
            }
        });
        Ok(())
    }

    /// The corrected arc curve over the current window: low where a regime changes, 1 in the
    /// end zones. Empty before the first window completes.
    fn curve<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.floss.curve())
    }

    /// `(position, score)` of the lowest curve value outside the end zones, the position local
    /// to the curve; ties go to the smaller position. `None` when there is no interior yet.
    fn min_interior(&self) -> Option<(usize, f64)> {
        self.floss.min_interior()
    }
}

fn horizon(m: usize, w: Option<core::num::NonZeroUsize>) -> usize {
    w.map_or(usize::MAX, |w| w.get() - m)
}

/// Hidden: the corrected arc curve of left indices (`-1` for none) through the same function
/// `FlossStream` is built on, which `ladders.stumpy.floss` mirrors for its right
/// neighbors. `w` is the stream's bound, if any.
#[pyfunction]
#[pyo3(signature = (left_indices, window_start, m, w=None, l=None, excl_factor=5))]
pub fn _corrected_arc_curve(
    left_indices: Ints<'_>,
    window_start: u64,
    m: i64,
    w: Option<i64>,
    l: Option<i64>,
    excl_factor: i64,
) -> PyResult<Vec<f64>> {
    let m = window_length(m)?;
    let w = w.map(|w| window_bound(w, m)).transpose()?;
    let l = match l {
        None => m.get(),
        Some(l) => non_negative("l", l)?,
    };
    let excl_factor = non_negative("excl_factor", excl_factor)?;
    let mut left = Vec::new();
    for &i in left_indices.as_array().iter() {
        left.push(match i {
            -1 => SampleIndex::NONE,
            i => u64::try_from(i)
                .ok()
                .and_then(SampleIndex::new)
                .ok_or_else(|| PyValueError::new_err(format!("bad left index {i}")))?,
        });
    }
    Ok(corrected_arc_curve(
        &left,
        window_start,
        ladde::exclusion::zone(m.get()),
        horizon(m.get(), w),
        l,
        excl_factor,
    ))
}

/// STUMPY's `fluss(I, L, n_regimes, excl_factor, custom_iac)`: the corrected arc curve and the
/// regime locations `(cac, regime_locs)` of a two-sided nearest-neighbor index `I` (`-1` for none,
/// e.g. `matrix_profile`'s second output). `L` is the regime length (at least 1) that sizes the end
/// zones and the exclusion around each pick, `excl_factor` its multiplier (0 turns both off),
/// `n_regimes` the number of regimes to find (`regime_locs` has `n_regimes - 1` positions; more
/// than there are windows raises `ValueError`, where STUMPY pads with repeated locations).
/// An entry of `I` that is neither `-1` nor a window index raises `ValueError`. `custom_iac`, one
/// value per window, replaces the idealized arc curve. With the same `I` and
/// `custom_iac` the result equals STUMPY's.
///
/// The default idealized arc curve is a closed-form expectation, not STUMPY's seeded beta fit, so
/// the default `cac` agrees with STUMPY's in shape but not on bits. It depends on the exclusion
/// zone `ceil(m / 4)` of the profile `I` came from, so `m` (keyword-only, default `L`, as the
/// usual choice of `L` is the window length) names that `m`. To segment a series itself, see
/// `fluss_series`. Breaking change from earlier versions: `fluss(t, m, L, n_regimes, ...)` took the
/// series and is now `fluss_series`.
#[pyfunction]
#[pyo3(signature = (I, L, n_regimes, excl_factor=5, custom_iac=None, *, m=None))]
#[allow(non_snake_case)]
pub fn fluss<'py>(
    py: Python<'py>,
    I: Ints<'py>,
    L: i64,
    n_regimes: i64,
    excl_factor: i64,
    custom_iac: Option<Floats<'py>>,
    m: Option<i64>,
) -> PyResult<FlussResult<'py>> {
    let index = I;
    let l = L;
    let m = window_length(m.unwrap_or(l))?;
    let (l, n_regimes, excl_factor) = fluss_counts(l, n_regimes, excl_factor)?;
    let windows = index.as_array().len();
    regimes_fit(n_regimes, windows)?;
    let indices: Vec<SampleIndex> = index
        .as_array()
        .iter()
        .map(|&i| match usize::try_from(i) {
            Ok(window) if window < windows => {
                Ok(SampleIndex::new(window as u64).expect("a window index is below u64::MAX"))
            }
            _ if i == -1 => Ok(SampleIndex::NONE),
            _ => Err(PyValueError::new_err(format!(
                "I entries must be window indices (below {windows}) or -1, got {i}"
            ))),
        })
        .collect::<PyResult<_>>()?;
    let custom = custom_iac.map(|c| c.as_array().to_vec());
    let result = fluss_of_index(
        &indices,
        ladde::exclusion::zone(m.get()),
        l,
        n_regimes,
        excl_factor,
        custom.as_deref(),
    );
    fluss_arrays(py, result)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<FlossStream>()?;
    module.add_function(wrap_pyfunction!(_corrected_arc_curve, module)?)?;
    module.add_function(wrap_pyfunction!(fluss, module)?)?;
    Ok(())
}
