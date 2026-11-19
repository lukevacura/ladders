//! `left_profile`: the whole-series batch left matrix profile
//! ([`ladde::batch::left_profile`]), returned as NumPy arrays, releasing the GIL for
//! the computation itself.

use core::num::NonZeroUsize;

use ladde::entry::Entry;
use ladde::quantize::{Scale, ScaleChoice};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::constant::{flags_of, series as flagged};
use crate::convert::{Floats, index_and_distance};
use crate::err::{Remedy, ScaleArg, bounded_cells, countable, tick_scale, window_length};

/// `left_profile`'s own return shape: STUMPY's `(P, I, scale_used)`, 1-D with `k = 1` and
/// `(windows, k)` otherwise.
type LeftProfileResult<'py> = (Bound<'py, PyAny>, Bound<'py, PyAny>, f64);

/// The one place `threads` is resolved: `None` is every core available, an int of at least 1
/// passes through unchanged, anything else (zero, negative, `bool`, `numpy.bool_`, a non-int)
/// is a `ValueError`.
pub(crate) fn resolve_threads(threads: Option<&Bound<'_, PyAny>>) -> PyResult<NonZeroUsize> {
    let Some(threads) = threads else {
        return Ok(ladde::par::threads_available());
    };
    // By type name, so `bool` and `numpy.bool_` are both caught.
    let is_bool = threads
        .get_type()
        .name()
        .is_ok_and(|name| name.to_string().starts_with("bool"));
    if is_bool {
        return Err(PyValueError::new_err("threads must be an int, got a bool"));
    }
    let value = threads
        .extract::<i64>()
        .map_err(|_| PyValueError::new_err("threads must be an int of at least 1 or None"))?;
    usize::try_from(value)
        .ok()
        .and_then(NonZeroUsize::new)
        .ok_or_else(|| PyValueError::new_err(format!("threads must be at least 1, got {value}")))
}

/// Hidden: the thread count `left_profile(..., threads=threads)` would use.
#[pyfunction]
#[pyo3(signature = (threads=None))]
pub(crate) fn _resolve_threads(threads: Option<&Bound<'_, PyAny>>) -> PyResult<usize> {
    resolve_threads(threads).map(NonZeroUsize::get)
}

/// Hidden: the core's own count of available threads.
#[pyfunction]
pub(crate) fn _threads_available() -> usize {
    ladde::par::threads_available().get()
}

/// The tick size `scale` names, which must count every finite sample of `series`, or the one
/// inferred from `series` and `m`: by the default rule when `None`, by the finest one for
/// `"finest"`.
pub(crate) fn resolve_scale(
    series: &[f64],
    m: NonZeroUsize,
    scale: Option<ScaleArg>,
) -> PyResult<Scale> {
    resolve_scale_of(&[series], m, scale)
}

/// [`resolve_scale`] for several series sharing one scale ([`ScaleChoice::resolve_each`]).
pub(crate) fn resolve_scale_of(
    parts: &[&[f64]],
    m: NonZeroUsize,
    scale: Option<ScaleArg>,
) -> PyResult<Scale> {
    let choice = match scale {
        Some(ScaleArg::Tick(scale)) => {
            let scale = tick_scale(scale)?;
            for part in parts {
                countable(part, scale, Remedy::Pass)?;
            }
            return Ok(scale);
        }
        Some(ScaleArg::Finest) => ScaleChoice::Finest,
        None => ScaleChoice::Fast,
    };
    choice.resolve_each(parts, m).ok_or_else(|| {
        PyValueError::new_err("no scale fits: t is shorter than m, or has no finite sample at all")
    })
}

/// The tick size a call given no `scale` would use for `t` at window length `m`: the default
/// rule's, which `SCALE_RULE` numbers, or with `finest=True` the finer opt-in rule's (what
/// `scale="finest"` uses). Pass it back as `scale` to pin a call's results: with a given scale
/// they are the same bits in every release, while an inferred scale changes when `SCALE_RULE`
/// does.
///
/// `t` is one series, or several that share one scale (a list of series, or the rows of a 2-D
/// array). Which to pass depends on the function the scale is for:
/// - a z-normalized function infers each series' scale on its own (`ab_join`'s `scale_a` and
///   `scale_b`, each dimension of `mstump`, each series of `ostinato`): call this once per series;
/// - a non-normalized function gives every series of one call the same scale (`aamp` over two
///   series, the dimensions of `maamp`, the set of `aamp_ostinato`, the series and query of
///   `aamp_match`): pass them together;
/// - `snippets` quantizes at its inner window length (`s`, else `ceil(percentage * m)`): pass
///   that as `m`.
///
/// `ValueError` for `m < 3`, when no series is at least `m` long, or when there is no finite
/// sample at all.
#[pyfunction]
#[pyo3(signature = (t, m, finest=false))]
pub(crate) fn infer_scale(t: &Bound<'_, PyAny>, m: i64, finest: bool) -> PyResult<f64> {
    let m = window_length(m)?;
    let series: Vec<Vec<f64>> = match t.extract::<Floats<'_>>() {
        Ok(one) => vec![one.as_array().to_vec()],
        // Not one series: several, or the error the one series gave.
        Err(not_one) => match t.extract::<Vec<Floats<'_>>>() {
            Ok(many) if !many.is_empty() => many.iter().map(|s| s.as_array().to_vec()).collect(),
            _ => return Err(not_one),
        },
    };
    let parts: Vec<&[f64]> = series.iter().map(Vec::as_slice).collect();
    // No series holds a window, so no call could use a scale (the shared rule alone would read the
    // series end to end with their seams and could find them long enough).
    if parts.iter().all(|part| part.len() < m.get()) {
        return Err(PyValueError::new_err(format!(
            "no scale fits: no series is at least m = {m} samples long"
        )));
    }
    let rule = finest.then_some(ScaleArg::Finest);
    Ok(resolve_scale_of(&parts, m, rule)?.get())
}

/// How `left_profile(t, m, scale, threads=threads)` splits its work, as a dict: `tiles`, the
/// independent tiles it runs, and `wide_tiles`, how many of them needed the exact i64 pass.
/// Informational only: no result depends on it.
#[pyfunction]
#[pyo3(signature = (t, m, scale=None, threads=None))]
pub(crate) fn left_profile_report<'py>(
    py: Python<'py>,
    t: Floats<'py>,
    m: i64,
    scale: Option<ScaleArg>,
    threads: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let m = window_length(m)?;
    let threads = resolve_threads(threads)?;
    let series = t.as_array().to_vec();
    let scale = resolve_scale(&series, m, scale)?;
    let report = py.detach(|| ladde::batch::left_profile_report(&series, m, scale, threads).1);
    let dict = PyDict::new(py);
    dict.set_item("tiles", report.tiles)?;
    dict.set_item("wide_tiles", report.wide_tiles)?;
    Ok(dict)
}

/// The exact causal left nearest-neighbor matrix profile over `t`, computed in one batch pass.
/// `scale`, when given, is the tick size `t` is quantized to; a finite sample beyond 2**53 ticks
/// of it raises `ValueError` (it would otherwise be a gap), naming a coarser scale that holds
/// it, and so does every function taking a `scale`. When `None`, the scale is
/// inferred from `t` and `m`: if every finite sample lies on one grid (for example 0.01, 0.25
/// or 100), the coarsest such grid from a fixed list of decimal steps (5, 2.5, 2 or 1 times a
/// power of ten) and powers of two that the arithmetic also accepts; otherwise the finest power
/// of two at which every window with no non-finite sample, and every run of m + 128 finite
/// samples (a whole run of at least m when shorter), holding no outlier (an isolated sample far
/// from its neighbors' level, at most one window in 16 holding one) has all its ticks within
/// (2**31 - 1) / m of one anchor, the bound of the fast path: about 7 significant digits at
/// m = 64, a bit or two coarser than the finest exact scale. Series that share a scale (a join,
/// the dimensions of a non-normalized profile) each fit this rule on their own. `scale="finest"` keeps the grid
/// rule but falls back instead to the finest power of two under which every window with no
/// non-finite sample stays exact, which runs plain floats on the slower exact path. Either
/// choice depends only on `t` and `m`, and is reported back as `scale_used`; every function
/// that infers a scale takes `"finest"` the same way. Raises `ValueError` when no scale could
/// be chosen: `t` shorter than `m`, or with no finite sample at all.
///
/// Returns `(P, I, scale_used)`: `P` and `I` are STUMPY's own distance and index arrays, one
/// entry per window, oldest first; a window with no earlier neighbor gets index `-1` and
/// distance `inf`.
///
/// `threads` is how many threads share the work: `None` (the default) uses every core
/// available, an int `n >= 1` uses exactly `n` (`threads=1` runs on one thread). Anything else
/// (zero, negative, `bool`, a non-int) raises `ValueError`. The result does not depend on
/// `threads`: `P` and `I` are identical on bits for every value. Releases the GIL while it runs.
///
/// `k` (default 1) is how many nearest earlier neighbors each window keeps; `k < 1` raises
/// `ValueError`. With `k > 1`, `P` and `I` have shape `(windows, k)`, nearest first, ties to the
/// smaller index, padded with `inf` and `-1` where a window has fewer than `k` neighbors; `windows * k`
/// above 2**28 raises `ValueError`. `T_A_subseq_isconstant` is STUMPY's: a boolean array with one
/// flag per window, or a function `f(a, w)` returning one (given `t`, non-finite samples as NaN,
/// and `m`), saying which windows are constant, in place of "all samples equal"; two constant windows are at distance 0
/// and a constant one is at `sqrt(m)` from any other. A window with a non-finite sample has no distance whatever
/// its flag. A wrong length or a non-boolean result is a `ValueError`. The
/// neighbors may overlap each other (as in STUMPY). STUMPY's `stump(T, m, k=k)` ranks over both
/// sides, so it agrees with this only for windows whose `k` nearest neighbors are all earlier;
/// `matrix_profile(..., k=k)` is the two-sided one.
#[pyfunction]
#[pyo3(signature = (t, m, scale=None, threads=None, k=1, T_A_subseq_isconstant=None))]
#[allow(non_snake_case)]
pub(crate) fn left_profile<'py>(
    py: Python<'py>,
    t: Floats<'py>,
    m: i64,
    scale: Option<ScaleArg>,
    threads: Option<&Bound<'py, PyAny>>,
    k: i64,
    T_A_subseq_isconstant: Option<&Bound<'py, PyAny>>,
) -> PyResult<LeftProfileResult<'py>> {
    let m = window_length(m)?;
    let threads = resolve_threads(threads)?;
    let k = resolve_k(k)?;
    let series = t.as_array().to_vec();
    let scale = resolve_scale(&series, m, scale)?;
    let name = "T_A_subseq_isconstant";
    let flags = flags_of(py, T_A_subseq_isconstant, &series, m, name)?;
    let input = flagged(&series, m, &flags, name)?;

    if k == NonZeroUsize::MIN {
        let entries = py.detach(|| {
            #[cfg(feature = "test-hooks")]
            crate::hold::park_if_armed();
            ladde::batch::left_profile_threads(input, m, scale, threads)
        });
        let (indices, distances): (Vec<i64>, Vec<f64>) = entries
            .into_iter()
            .map(|(_, entry)| index_and_distance(entry))
            .unzip();
        return Ok((
            PyArray1::from_vec(py, distances).into_any(),
            PyArray1::from_vec(py, indices).into_any(),
            scale.get(),
        ));
    }

    bounded_cells("k", (series.len() + 1).saturating_sub(m.get()), k.get())?;
    let rows = py.detach(|| ladde::batch::left_topk_profile(input, m, scale, threads, k));
    let (distances, indices) = topk_arrays(py, rows.iter().map(|(_, entries)| entries), k)?;
    Ok((distances, indices, scale.get()))
}

/// A `k` whose neighbors the batch cannot reserve, as the `ValueError` it raises.
pub(crate) fn capacity_error(_: ladde::capacity::CapacityError) -> PyErr {
    PyValueError::new_err("k keeps more neighbors than can be reserved; use a smaller k")
}

/// `k` as a count: at least 1, else `ValueError`.
pub(crate) fn resolve_k(k: i64) -> PyResult<NonZeroUsize> {
    usize::try_from(k)
        .ok()
        .and_then(NonZeroUsize::new)
        .ok_or_else(|| PyValueError::new_err(format!("k must be at least 1, got {k}")))
}

/// STUMPY's top-k `(P, I)` arrays of shape `(windows, k)`, from each window's kept entries,
/// padded with `inf` and `-1` where a window has fewer than `k`.
fn topk_arrays<'a, 'py>(
    py: Python<'py>,
    rows: impl ExactSizeIterator<Item = &'a Vec<Entry>>,
    k: NonZeroUsize,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    topk_arrays_of(py, rows, k, |entry| {
        index_and_distance(entry.unwrap_or(Entry::NONE))
    })
}

/// [`topk_arrays`] for any kind of kept neighbor: `convert` gives the `(index, distance)` of the
/// neighbor in a slot, or of no neighbor (`None`) where a row has fewer than `k`.
pub(crate) fn topk_arrays_of<'a, 'py, T: Copy + 'a>(
    py: Python<'py>,
    rows: impl ExactSizeIterator<Item = &'a Vec<T>>,
    k: NonZeroUsize,
    convert: impl Fn(Option<T>) -> (i64, f64),
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let (windows, width) = (rows.len(), k.get());
    let mut distances = Vec::with_capacity(windows * width);
    let mut indices = Vec::with_capacity(windows * width);
    for entries in rows {
        for slot in 0..width {
            let (index, distance) = convert(entries.get(slot).copied());
            indices.push(index);
            distances.push(distance);
        }
    }
    let shape = [windows, width];
    Ok((
        PyArray1::from_vec(py, distances).reshape(shape)?.into_any(),
        PyArray1::from_vec(py, indices).reshape(shape)?.into_any(),
    ))
}

/// STUMPY's `stump` shape: `(P, I, left_I, right_I)`; `P` and `I` are 1-D with `k = 1` and
/// `(windows, k)` otherwise.
type MatrixProfileResult<'py> = (
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
);

/// The exact two-sided nearest-neighbor matrix profile over `t`, in the shape of STUMPY's
/// `stump`: `(P, I, left_I, right_I)`, one entry per window, oldest first. `I` is the nearer of
/// the left neighbor (`left_I`, earlier windows, as `left_profile` finds it) and the right
/// neighbor (`right_I`, later windows), ties to the smaller index; `P` is its distance. A window
/// with no neighbor on a side gets index `-1` there, and with none at all, distance `inf`.
/// Looks ahead: the right side uses samples after each window, so it is not a causal signal.
///
/// `scale` and `threads` mean what they do in `left_profile`, and the result does not depend on
/// `threads`. Releases the GIL while it runs.
///
/// `k` (default 1) is how many nearest neighbors each window keeps over both sides, as STUMPY's
/// `stump(T, m, k=k)`: with `k > 1`, `P` and `I` have shape `(windows, k)`, nearest first, ties to
/// the smaller index, padded with `inf` and `-1` where a window has fewer than `k` neighbors, while
/// `left_I` and `right_I` stay each side's one nearest. `k < 1` raises `ValueError`, and so does
/// `windows * k` above 2**28.
///
/// `T_A_subseq_isconstant` is STUMPY's (see `left_profile`): which windows are constant.
#[pyfunction(name = "matrix_profile")]
#[pyo3(signature = (t, m, scale=None, threads=None, k=1, T_A_subseq_isconstant=None))]
#[allow(non_snake_case)]
pub(crate) fn two_sided_profile<'py>(
    py: Python<'py>,
    t: Floats<'py>,
    m: i64,
    scale: Option<ScaleArg>,
    threads: Option<&Bound<'py, PyAny>>,
    k: i64,
    T_A_subseq_isconstant: Option<&Bound<'py, PyAny>>,
) -> PyResult<MatrixProfileResult<'py>> {
    let m = window_length(m)?;
    let threads = resolve_threads(threads)?;
    let k = resolve_k(k)?;
    let series = t.as_array().to_vec();
    let scale = resolve_scale(&series, m, scale)?;
    let name = "T_A_subseq_isconstant";
    let flags = flags_of(py, T_A_subseq_isconstant, &series, m, name)?;
    let input = flagged(&series, m, &flags, name)?;

    if k != NonZeroUsize::MIN {
        bounded_cells("k", (series.len() + 1).saturating_sub(m.get()), k.get())?;
        let rows = py
            .detach(|| ladde::batch::two_sided_topk_profile(input, m, scale, threads, k))
            .map_err(capacity_error)?;
        let (distances, indices) =
            topk_arrays(py, rows.iter().map(|(_, sides)| &sides.nearest), k)?;
        let left = rows
            .iter()
            .map(|(_, sides)| index_and_distance(sides.left).0)
            .collect();
        let right = rows
            .iter()
            .map(|(_, sides)| index_and_distance(sides.right).0)
            .collect();
        return Ok((
            distances,
            indices,
            PyArray1::from_vec(py, left),
            PyArray1::from_vec(py, right),
        ));
    }

    let profile = py.detach(|| ladde::batch::two_sided_profile(input, m, scale, threads));

    let mut distances = Vec::with_capacity(profile.len());
    let mut indices = Vec::with_capacity(profile.len());
    let mut left = Vec::with_capacity(profile.len());
    let mut right = Vec::with_capacity(profile.len());
    for (_, sides) in profile {
        let (index, distance) = index_and_distance(sides.nearest);
        indices.push(index);
        distances.push(distance);
        left.push(index_and_distance(sides.left).0);
        right.push(index_and_distance(sides.right).0);
    }
    Ok((
        PyArray1::from_vec(py, distances).into_any(),
        PyArray1::from_vec(py, indices).into_any(),
        PyArray1::from_vec(py, left),
        PyArray1::from_vec(py, right),
    ))
}

/// `ab_join`'s return shape: `(P, I, scale_a_used, scale_b_used)`; `P` and `I` are 1-D with `k = 1`
/// and `(windows, k)` otherwise.
type AbJoinResult<'py> = (Bound<'py, PyAny>, Bound<'py, PyAny>, f64, f64);

/// The exact AB-join of `t_a` against `t_b`: for every window of `t_a`, its nearest window of
/// `t_b` under the z-normalized distance, with no exclusion zone, as STUMPY's
/// `stump(t_a, m, t_b)` ranks it. Returns `(P, I, scale_a_used, scale_b_used)`: `P` and `I` are
/// STUMPY's distance and index columns for the A side, one entry per window of `t_a`, oldest
/// first, `I` indexing the windows of `t_b` (ties to the smaller index; a window with no finite
/// window of `t_b` to meet gets index `-1` and distance `inf`). STUMPY's other two columns (the
/// left and right nearest neighbors) are not returned: they rank earlier and later windows of
/// one series, and across two series there is no earlier or later.
///
/// `scale_a` and `scale_b` are each series' own tick size, inferred per series from it and `m`
/// when `None` (as `left_profile` does) and reported back; distances are comparable across the
/// two because z-normalization divides each series' own scale out. `ValueError` when a scale
/// cannot be chosen (a series shorter than `m` or with no finite sample, and no scale given) or
/// for a bad `threads`, which means what it does in `left_profile`, and the result does not
/// depend on it. Releases the GIL while it runs.
///
/// `k` (default 1) is how many nearest windows of `t_b` each window of `t_a` keeps, as STUMPY's
/// `stump(t_a, m, t_b, k=k)`: with `k > 1`, `P` and `I` have shape `(windows, k)`, nearest first,
/// ties to the smaller index, padded with `inf` and `-1` where fewer than `k` windows of `t_b` are
/// finite. `k < 1` raises `ValueError`, and so does `windows * k` above 2**28.
///
/// `T_A_subseq_isconstant` and `T_B_subseq_isconstant` are STUMPY's (see `left_profile`): which
/// windows of each series are constant.
#[pyfunction]
#[pyo3(signature = (
    t_a, t_b, m, scale_a=None, scale_b=None, threads=None, k=1, T_A_subseq_isconstant=None,
    T_B_subseq_isconstant=None
))]
#[allow(clippy::too_many_arguments, non_snake_case)]
pub(crate) fn ab_join<'py>(
    py: Python<'py>,
    t_a: Floats<'py>,
    t_b: Floats<'py>,
    m: i64,
    scale_a: Option<ScaleArg>,
    scale_b: Option<ScaleArg>,
    threads: Option<&Bound<'py, PyAny>>,
    k: i64,
    T_A_subseq_isconstant: Option<&Bound<'py, PyAny>>,
    T_B_subseq_isconstant: Option<&Bound<'py, PyAny>>,
) -> PyResult<AbJoinResult<'py>> {
    let m = window_length(m)?;
    let threads = resolve_threads(threads)?;
    let k = resolve_k(k)?;
    let (a, b) = (t_a.as_array().to_vec(), t_b.as_array().to_vec());
    let (scale_a, scale_b) = (
        resolve_scale(&a, m, scale_a)?,
        resolve_scale(&b, m, scale_b)?,
    );
    let (name_a, name_b) = ("T_A_subseq_isconstant", "T_B_subseq_isconstant");
    let (flags_a, flags_b) = (
        flags_of(py, T_A_subseq_isconstant, &a, m, name_a)?,
        flags_of(py, T_B_subseq_isconstant, &b, m, name_b)?,
    );
    let (input_a, input_b) = (
        flagged(&a, m, &flags_a, name_a)?,
        flagged(&b, m, &flags_b, name_b)?,
    );
    if k != NonZeroUsize::MIN {
        bounded_cells("k", (a.len() + 1).saturating_sub(m.get()), k.get())?;
        let rows = py
            .detach(|| {
                ladde::batch::ab_join_topk(input_a, input_b, m, scale_a, scale_b, threads, k)
            })
            .map_err(capacity_error)?;
        let (distances, indices) = topk_arrays(py, rows.iter().map(|(_, entries)| entries), k)?;
        return Ok((distances, indices, scale_a.get(), scale_b.get()));
    }
    let entries =
        py.detach(|| ladde::batch::ab_join(input_a, input_b, m, scale_a, scale_b, threads));
    let (indices, distances): (Vec<i64>, Vec<f64>) = entries
        .into_iter()
        .map(|(_, entry)| index_and_distance(entry))
        .unzip();
    Ok((
        PyArray1::from_vec(py, distances).into_any(),
        PyArray1::from_vec(py, indices).into_any(),
        scale_a.get(),
        scale_b.get(),
    ))
}

/// Whole-series segmentation, STUMPY's FLUSS: `(cac, regime_locs)` of the two-sided profile of
/// `t` (see `matrix_profile`). `L` is the regime length (at least 1) that sizes the end zones
/// and the exclusion around each pick, `excl_factor` its multiplier (0 turns both off),
/// `n_regimes` the number of regimes to find (`regime_locs` has `n_regimes - 1` positions; more
/// than there are windows raises `ValueError`, where STUMPY pads with repeated locations).
/// `custom_iac`, one value per window, replaces the idealized arc curve.
///
/// Takes the series, not an index like STUMPY's `fluss(I, L, n_regimes)` (which is `fluss`), so the
/// profile is computed here. The default idealized arc curve is a closed-form expectation, not STUMPY's
/// seeded beta fit, so the default `cac` agrees with STUMPY's in shape but not on bits; with
/// the same `custom_iac` and the same index it equals STUMPY's. Looks ahead, like
/// `matrix_profile`. `scale`, `threads` and `T_A_subseq_isconstant` mean what they do there.
/// Releases the GIL.
#[pyfunction]
#[pyo3(signature = (
    t, m, l, n_regimes, excl_factor=5, custom_iac=None, *, scale=None, threads=None,
    T_A_subseq_isconstant=None
))]
#[allow(clippy::too_many_arguments, non_snake_case)]
pub(crate) fn fluss_series<'py>(
    py: Python<'py>,
    t: Floats<'py>,
    m: i64,
    l: i64,
    n_regimes: i64,
    excl_factor: i64,
    custom_iac: Option<Floats<'py>>,
    scale: Option<ScaleArg>,
    threads: Option<&Bound<'py, PyAny>>,
    T_A_subseq_isconstant: Option<&Bound<'py, PyAny>>,
) -> PyResult<crate::floss::FlussResult<'py>> {
    let m = window_length(m)?;
    let (l, n_regimes, excl_factor) = crate::floss::fluss_counts(l, n_regimes, excl_factor)?;
    let threads = resolve_threads(threads)?;
    let series = t.as_array().to_vec();
    let scale = resolve_scale(&series, m, scale)?;
    crate::floss::regimes_fit(n_regimes, series.len() + 1 - m.get())?;
    let custom = custom_iac.map(|c| c.as_array().to_vec());
    let name = "T_A_subseq_isconstant";
    let flags = flags_of(py, T_A_subseq_isconstant, &series, m, name)?;
    let input = flagged(&series, m, &flags, name)?;
    let result = py.detach(|| {
        ladde::batch::fluss(
            input,
            m,
            scale,
            threads,
            l,
            n_regimes,
            excl_factor,
            custom.as_deref(),
        )
    });
    crate::floss::fluss_arrays(py, result)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(fluss_series, module)?)?;
    module.add_function(wrap_pyfunction!(ab_join, module)?)?;
    module.add_function(wrap_pyfunction!(left_profile, module)?)?;
    module.add_function(wrap_pyfunction!(two_sided_profile, module)?)?;
    module.add_function(wrap_pyfunction!(left_profile_report, module)?)?;
    module.add_function(wrap_pyfunction!(infer_scale, module)?)?;
    module.add("SCALE_RULE", ladde::quantize::SCALE_RULE)?;
    module.add_function(wrap_pyfunction!(_resolve_threads, module)?)?;
    module.add_function(wrap_pyfunction!(_threads_available, module)?)?;
    Ok(())
}
