//! `stimp` and `aamp_stimp`: STUMPY's pan matrix profile classes ([`ladde::pan`]), every
//! row the exact two-sided profile at its window size, z-normalized or not.

use ladde::pan::{PanError, PanProfile, PanTransform};
use ladde::quantize::ScaleChoice;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::aamp::require_euclidean;
use crate::batch::resolve_threads;
use crate::constant::flags_of;
use crate::convert::Floats;
use crate::err::{Remedy, ScaleArg, countable, tick_scale};

/// The pan matrix profile of a series over a range of window sizes, computed one size per
/// `update()` in STUMPY's breadth-first order.
///
/// The window sizes are STUMPY's list: with `max_m` given, `min_m` through `max_m`; with
/// `max_m=None`, `min_m` through `max(min_m + 1, n - (n + 3) // 5 - 1)`, unclamped; then every
/// `step`-th size from the first. Differences from STUMPY, each a `ValueError`: with `max_m`
/// given, a `min_m` above it (STUMPY swaps them), a `min_m` below 3 (STUMPY raises it to 3) and a
/// `max_m` above `n - (n + 3) // 5 - 1` (STUMPY lowers it to that); and where STUMPY builds the
/// object and fails later in `update()` (a `step` below 1, a size below 3 or longer than `t`),
/// this raises here.
/// A negative `step` is an error too (STUMPY accepts it and builds a descending range, which only
/// yields sizes on tiny series).
///
/// `scale` is the tick size used for every size; when `None`, each size infers its own, so rows
/// of a series that is not on a plain grid can use different scales (`scales_` says which). Each
/// row equals the default `matrix_profile(t, m)` call at its size. `threads` means what it does
/// in `matrix_profile` and the result does not depend on it.
///
/// Every row is exact: `percentage` and `pre_scrump` are accepted so STUMPY's calls run, and
/// ignored (STUMPY's default `percentage=0.01` samples the profile, this never does). Each row
/// equals `matrix_profile(t, m, ...)` on bits and STUMPY's `stump` on tie-free series. Holds
/// one `float64` and one `int64` per window per computed size. `update()` releases the GIL.
///
/// With `normalize=False` this is `aamp_stimp` (as STUMPY re-routes it; the object is still a
/// `stimp`), and `p` is its `p`. With the default `normalize=True`, `p` is accepted and ignored,
/// as in STUMPY. `T_subseq_isconstant_func` is STUMPY's: a function `f(a, w)` returning, for
/// the series `a` (non-finite samples as NaN) and the window size `w` of the row being computed,
/// one boolean per window saying whether it is constant, in place of "all samples equal"; it is
/// called by `update()` (holding the GIL), once per row, and an exception it raises propagates
/// and leaves the row not computed.
///
/// `M_` is the window sizes in visiting order, `P_` the raw nearest distances per size in that
/// order (`inf` while not yet computed), `PAN_` the image of `pan()` at STUMPY's defaults. Extra
/// to STUMPY: `PI_`, the nearest-neighbor indices per size (`-1` while not yet computed), and
/// `scales_`, the tick size each size used (`nan` while not yet computed).
#[pyclass(module = "ladders", name = "stimp", subclass)]
pub struct Stimp {
    inner: PanProfile,
    constant: Option<(Py<PyAny>, Vec<f64>)>,
}

/// What core refused, in words; the range errors only come with `max_m` given.
fn pan_error(error: PanError, sizes: &SizeRange, len: usize) -> PyErr {
    let SizeRange { min_m, max_m, .. } = *sizes;
    let max = max_m.unwrap_or_default();
    PyValueError::new_err(match error {
        PanError::ZeroStep => "step must be at least 1".to_owned(),
        PanError::WindowBelowThree if max_m.is_none() => {
            "min_m must be at least 3 when max_m is None".to_owned()
        }
        PanError::WindowBelowThree => {
            format!("min_m={min_m} is below 3; STUMPY raises it to 3; pass min_m=3 to get that")
        }
        PanError::WindowBeyondSeries => "the window sizes must not be longer than t".to_owned(),
        PanError::MinAboveMax => format!(
            "min_m={min_m} is above max_m={max}; STUMPY swaps them; pass min_m={max}, \
             max_m={min_m} to get that"
        ),
        PanError::MaxAboveLargest { largest } => format!(
            "max_m={max} is above {largest}, the largest window size for a series of {len} \
             samples; STUMPY lowers it to {largest}; pass max_m={largest} to get that"
        ),
        // The enum is non-exhaustive: a variant this file does not know is named as core names it.
        _ => format!("the input was refused: {error:?}"),
    })
}

/// A window size as core takes it: a negative one is below 3 like 0, which core refuses.
fn size(value: i64) -> usize {
    usize::try_from(value.max(0)).unwrap_or(usize::MAX)
}

/// Where the window sizes lie, as the caller gave them.
#[derive(Clone, Copy)]
struct SizeRange {
    min_m: i64,
    max_m: Option<i64>,
    step: i64,
}

/// The pan profile of `t` under the metric `nonnorm` picks, with the checks both classes share.
fn make_profile(
    t: &[f64],
    sizes: SizeRange,
    scale: Option<ScaleArg>,
    threads: Option<&Bound<'_, PyAny>>,
    nonnorm: bool,
) -> PyResult<PanProfile> {
    let threads = resolve_threads(threads)?;
    if sizes.step < 1 {
        return Err(pan_error(PanError::ZeroStep, &sizes, t.len()));
    }
    let scale = match scale {
        Some(ScaleArg::Tick(scale)) => {
            let scale = tick_scale(scale)?;
            countable(t, scale, Remedy::Pass)?;
            ScaleChoice::Given(scale)
        }
        Some(ScaleArg::Finest) => ScaleChoice::Finest,
        None => ScaleChoice::Fast,
    };
    let new = if nonnorm {
        PanProfile::new_nonnorm::<ScaleChoice>
    } else {
        PanProfile::new::<ScaleChoice>
    };
    let inner = new(
        t,
        size(sizes.min_m),
        sizes.max_m.map(size),
        size(sizes.step),
        scale,
        threads,
    )
    .map_err(|error| pan_error(error, &sizes, t.len()))?;
    if (nonnorm || !matches!(scale, ScaleChoice::Given(_))) && !t.iter().any(|x| x.is_finite()) {
        return Err(PyValueError::new_err(
            "no scale fits: t has no finite sample at all",
        ));
    }
    Ok(inner)
}

#[pymethods]
impl Stimp {
    #[new]
    #[pyo3(signature = (T, min_m=3, max_m=None, step=1, percentage=0.01, pre_scrump=true, normalize=true, p=2.0, T_subseq_isconstant_func=None, *, scale=None, threads=None))]
    #[allow(clippy::too_many_arguments, unused_variables, non_snake_case)]
    fn new(
        T: Floats<'_>,
        min_m: i64,
        max_m: Option<i64>,
        step: i64,
        percentage: f64,
        pre_scrump: bool,
        normalize: bool,
        p: f64,
        T_subseq_isconstant_func: Option<&Bound<'_, PyAny>>,
        scale: Option<ScaleArg>,
        threads: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let function = T_subseq_isconstant_func.filter(|function| !function.is_none());
        if !normalize {
            require_euclidean(p)?;
            if function.is_some() {
                return Err(PyTypeError::new_err(
                    "T_subseq_isconstant_func does not apply to normalize=False",
                ));
            }
        }
        if let Some(function) = function.filter(|function| !function.is_callable()) {
            return Err(PyValueError::new_err(format!(
                "T_subseq_isconstant_func must be a callable, got {}",
                function.get_type().name()?
            )));
        }
        let sizes = SizeRange { min_m, max_m, step };
        let series = T.as_array().to_vec();
        let inner = make_profile(&series, sizes, scale, threads, !normalize)?;
        let constant = function.map(|function| (function.clone().unbind(), series));
        Ok(Self { inner, constant })
    }

    /// Computes the next window size in breadth-first order; does nothing once all are done.
    fn update(&mut self, py: Python<'_>) -> PyResult<()> {
        let Some((function, series)) = &self.constant else {
            py.detach(|| self.inner.update());
            return Ok(());
        };
        let Some(window) = self.inner.next_window_size() else {
            return Ok(());
        };
        let m = core::num::NonZeroUsize::new(window).expect("window sizes are at least 3");
        let flags = flags_of(
            py,
            Some(function.bind(py)),
            series,
            m,
            "T_subseq_isconstant_func",
        )?;
        let flags = flags.expect("a predicate was given");
        self.inner
            .update_with(Some(&flags))
            .map_err(|e| {
                PyValueError::new_err(format!(
                    "T_subseq_isconstant_func must give one flag per window ({}), got {}",
                    e.expected, e.found
                ))
            })
            .map(drop)
    }

    /// STUMPY's `pan`: the image, one row per window size ascending, one column per sample.
    #[pyo3(signature = (threshold=0.2, normalize=true, contrast=true, binary=true, clip=true))]
    fn pan<'py>(
        &self,
        py: Python<'py>,
        threshold: f64,
        normalize: bool,
        contrast: bool,
        binary: bool,
        clip: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let transform = PanTransform {
            threshold,
            normalize,
            contrast,
            binary,
            clip,
        };
        let image = py.detach(|| self.inner.pan(transform));
        let shape = [self.inner.sizes(), self.inner.len()];
        Ok(PyArray1::from_vec(py, image).reshape(shape)?.into_any())
    }

    #[getter(PAN_)]
    fn pan_default<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.pan(py, 0.2, true, true, true, true)
    }

    #[getter(M_)]
    fn sizes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_vec(py, self.inner.window_sizes().map(|m| m as i64).collect())
    }

    #[getter(scales_)]
    fn scales<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let rows = self.inner.rows();
        let scales = (0..self.inner.sizes())
            .map(|visit| {
                rows.get(visit)
                    .and_then(|row| row.scale)
                    .map_or(f64::NAN, |scale| scale.get())
            })
            .collect();
        PyArray1::from_vec(py, scales)
    }

    #[getter(P_)]
    fn distances<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let rows = self.inner.rows();
        let lists = self.inner.window_sizes().enumerate().map(|(visit, m)| {
            let distances = match rows.get(visit) {
                Some(row) => row.distances.clone(),
                None => vec![f64::INFINITY; self.inner.len() + 1 - m],
            };
            PyArray1::from_vec(py, distances)
        });
        PyList::new(py, lists)
    }

    #[getter(PI_)]
    fn indices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let rows = self.inner.rows();
        let lists = self.inner.window_sizes().enumerate().map(|(visit, m)| {
            let indices = match rows.get(visit) {
                Some(row) => row
                    .indices
                    .iter()
                    .map(|index| index.get().map_or(-1, |i| i as i64))
                    .collect(),
                None => vec![-1; self.inner.len() + 1 - m],
            };
            PyArray1::from_vec(py, indices)
        });
        PyList::new(py, lists)
    }
}

/// STUMPY's `aamp_stimp`: the pan matrix profile of the Euclidean distance on the raw values
/// (every `stimp` method and attribute, `P_` and `PAN_` included). Each row is the exact
/// `aamp(t, m)` at its size, in `t`'s units, so `PAN_` normalizes a distance by
/// `|max(t) - min(t)| * sqrt(m)` over the finite samples, as STUMPY does. `scale` is the tick
/// size every sample is quantized to, per size inferred from `t` as `aamp` infers it when
/// `None`.
///
/// Every row is exact: `percentage` and `pre_scraamp` are accepted so STUMPY's calls run, and
/// ignored (STUMPY's default `percentage=0.01` samples the profile with `scraamp`, this never
/// does). `p` must be 2.0; any other value raises `NotImplementedError` (STUMPY's `aamp_stimp`
/// silently uses 2.0 whatever `p` is given). A `t` with no finite sample raises `ValueError`.
#[pyclass(module = "ladders", name = "aamp_stimp", extends = Stimp)]
pub struct AampStimp;

#[pymethods]
impl AampStimp {
    #[new]
    #[pyo3(signature = (T, min_m=3, max_m=None, step=1, percentage=0.01, pre_scraamp=true, p=2.0, *, scale=None, threads=None))]
    #[allow(clippy::too_many_arguments, unused_variables, non_snake_case)]
    fn new(
        T: Floats<'_>,
        min_m: i64,
        max_m: Option<i64>,
        step: i64,
        percentage: f64,
        pre_scraamp: bool,
        p: f64,
        scale: Option<ScaleArg>,
        threads: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        require_euclidean(p)?;
        let sizes = SizeRange { min_m, max_m, step };
        let inner = make_profile(&T.as_array().to_vec(), sizes, scale, threads, true)?;
        Ok(PyClassInitializer::from(Stimp {
            inner,
            constant: None,
        })
        .add_subclass(Self))
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Stimp>()?;
    module.add_class::<AampStimp>()?;
    Ok(())
}
