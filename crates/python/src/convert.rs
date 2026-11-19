//! The one place an [`Entry`] becomes the pair of values Python sees: its neighbor's index (or
//! STUMPY's `-1` sentinel for [`SampleIndex::NONE`]) and its distance, Rust's correctly rounded
//! `sqrt` of the entry's own squared distance. Both native streams, `stumpi` and
//! `left_profile` all go through this, so none of them can drift from the others.

use core::ops::Deref;
use ladde::discord::Discord;
use ladde::entry::Entry;
use ladde::index::SampleIndex;
use ladde::nonnorm::NonNormNeighbor;
use ladde::quantize::Scale;

use numpy::ndarray::{Dimension, Ix1, Ix2};
use numpy::{AllowTypeChange, Element, PyArray1, PyArrayLike, PyReadonlyArray1};
use pyo3::Borrowed;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyInt};

/// An array argument: what NumPy turns into an array of `T` (lists, and arrays of any integer or
/// float dtype, converted), except what the conversion would change: complex input, which NumPy
/// would cast by dropping the imaginary part, is a `TypeError`, and a value `T` cannot hold
/// exactly (see [`Exact`]) is a `ValueError`.
pub(crate) struct Real<'py, T: Element, D: Dimension>(PyArrayLike<'py, T, D, AllowTypeChange>);

impl<'py, T: Element, D: Dimension> Deref for Real<'py, T, D> {
    type Target = PyArrayLike<'py, T, D, AllowTypeChange>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, 'py, T, D> FromPyObject<'a, 'py> for Real<'py, T, D>
where
    T: Exact + 'py,
    D: Dimension + 'py,
    Vec<T>: FromPyObject<'a, 'py>,
{
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Some((view, kind)) = numpy_view(&ob) {
            if kind == "c" {
                return Err(PyTypeError::new_err(
                    "complex input is not supported: pass its .real or .imag, not a cast that drops the imaginary part",
                ));
            }
            T::exactly(&view, &kind)?;
        }
        Ok(Self(PyArrayLike::extract(ob)?))
    }
}

/// `ob` as a NumPy array, with its dtype's kind. An object NumPy cannot read at all is left to the
/// extraction that follows, which reports it.
fn numpy_view<'py>(ob: &Borrowed<'_, 'py, PyAny>) -> Option<(Bound<'py, PyAny>, String)> {
    let view = ob
        .py()
        .import("numpy")
        .ok()?
        .call_method1("asarray", (ob.as_any(),))
        .ok()?;
    let kind = view
        .getattr("dtype")
        .ok()?
        .getattr("kind")
        .ok()?
        .extract()
        .ok()?;
    Some((view, kind))
}

/// An element type whose conversion from NumPy's other dtypes is checked to be exact.
pub(crate) trait Exact: Element + Sized {
    /// `Ok` when every value of the array `view` (of dtype kind `kind`) converts to `Self`
    /// unchanged.
    fn exactly(view: &Bound<'_, PyAny>, kind: &str) -> PyResult<()>;
}

/// The values of the array `view`, flattened, through NumPy's own cast to `dtype`.
fn flat<T: Element + Copy>(view: &Bound<'_, PyAny>, dtype: &str) -> PyResult<Vec<T>> {
    let cast = view
        .call_method0("ravel")?
        .call_method1("astype", (dtype,))?;
    let cast: PyReadonlyArray1<'_, T> = cast.extract()?;
    Ok(cast.as_array().to_vec())
}

/// Whether `value` survives the trip to `f64` and back.
fn float_holds(value: i128) -> bool {
    value as f64 as i128 == value
}

fn rounded(value: impl core::fmt::Display, rounded_to: f64) -> PyErr {
    PyValueError::new_err(format!(
        "an array argument holds the integer {value}, which float64 cannot hold exactly; STUMPY \
         refuses integer arrays and asks for .astype(float), which rounds it to {rounded_to:?}; \
         pass the array as float64 to get that rounding"
    ))
}

impl Exact for f64 {
    fn exactly(view: &Bound<'_, PyAny>, kind: &str) -> PyResult<()> {
        let inexact = match kind {
            "i" => flat::<i64>(view, "int64")?
                .into_iter()
                .find(|&v| !float_holds(v.into()))
                .map(|v| rounded(v, v as f64)),
            "u" => flat::<u64>(view, "uint64")?
                .into_iter()
                .find(|&v| !float_holds(v.into()))
                .map(|v| rounded(v, v as f64)),
            "O" => {
                let mut found = None;
                for item in view.call_method0("ravel")?.try_iter()? {
                    let item = item?;
                    if !item.is_instance_of::<PyInt>() || item.is_instance_of::<PyBool>() {
                        continue;
                    }
                    if item.extract::<i128>().is_ok_and(float_holds) {
                        continue;
                    }
                    found = Some(rounded(item.str()?, item.extract()?));
                    break;
                }
                found
            }
            _ => None,
        };
        inexact.map_or(Ok(()), Err)
    }
}

impl Exact for i64 {
    fn exactly(view: &Bound<'_, PyAny>, kind: &str) -> PyResult<()> {
        const RANGE: core::ops::Range<f64> =
            -9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0;
        let message = match kind {
            "f" => flat::<f64>(view, "float64")?
                .into_iter()
                .find(|&v| !(v.fract() == 0.0 && RANGE.contains(&v)))
                .map(|v| {
                    let cast = match RANGE.contains(&v.trunc()) {
                        true => (v as i64).to_string(),
                        false => "an arbitrary integer".to_owned(),
                    };
                    format!(
                        "an index argument holds {v:?}, which is not an int64 index; NumPy's cast \
                         would make it {cast}; pass the array as int64 to get that"
                    )
                }),
            "u" => flat::<u64>(view, "uint64")?
                .into_iter()
                .find(|&v| i64::try_from(v).is_err())
                .map(|v| {
                    format!(
                        "an index argument holds {v}, above the int64 range; NumPy's cast wraps \
                         it to {}; pass the array as int64 to get that",
                        v as i64
                    )
                }),
            _ => None,
        };
        message.map_or(Ok(()), |message| Err(PyValueError::new_err(message)))
    }
}

pub(crate) type Floats<'py> = Real<'py, f64, Ix1>;
/// [`Floats`] for index arguments.
pub(crate) type Ints<'py> = Real<'py, i64, Ix1>;
/// [`Floats`] for a series per row.
pub(crate) type FloatRows<'py> = Real<'py, f64, Ix2>;
/// [`Ints`] for a row per series.
pub(crate) type IntRows<'py> = Real<'py, i64, Ix2>;

/// An index argument as dimension numbers or window starts: negative entries are the caller's
/// to reject, so they come back as they are.
pub(crate) fn int_vec(values: &Ints<'_>) -> Vec<i64> {
    values.as_array().to_vec()
}

/// `(index, distance)`: `-1` and `inf` for [`Entry::NONE`], since `NONE`'s squared distance is
/// already `f64::INFINITY` and its index already has no `get()` — no separate no-neighbor case
/// to keep in sync with this one.
pub(crate) fn index_and_distance(entry: Entry) -> (i64, f64) {
    let index = entry.index().get().map_or(-1, |i| i as i64);
    let distance = entry.squared_distance().sqrt();
    (index, distance)
}

/// `(index, distance)` of a non-normalized neighbor: `-1` and `inf` for none, else the Euclidean
/// distance in the caller's units. The one place a non-normalized neighbor becomes these values.
pub(crate) fn nonnorm_index_and_distance(neighbor: NonNormNeighbor, scale: Scale) -> (i64, f64) {
    let index = neighbor.index().get().map_or(-1, |i| i as i64);
    (index, neighbor.euclidean(scale))
}

/// A listed discord as `(index, distance)`: its own index and the correctly rounded `sqrt` of
/// its squared distance, the same rounding every entry gets.
pub(crate) fn discord_index_and_distance(discord: Discord) -> (i64, f64) {
    let index = discord.index().get().map_or(-1, |i| i as i64);
    (index, discord.squared_distance().sqrt())
}

/// `update_many`'s own return shape, shared by every native stream class.
pub(crate) type IndexDistanceArrays<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>);

/// Every `(row, entry)` a stream's `update_many` collected, as parallel NumPy arrays in the
/// same order; the row itself is dropped (never part of what crosses into Python).
pub(crate) fn entries_to_arrays(
    py: Python<'_>,
    entries: Vec<(SampleIndex, Entry)>,
) -> IndexDistanceArrays<'_> {
    entry_arrays(py, entries.into_iter().map(|(_, entry)| entry))
}

/// Parallel NumPy arrays of `(index, distance)` for entries in the order given.
pub(crate) fn entry_arrays(
    py: Python<'_>,
    entries: impl ExactSizeIterator<Item = Entry>,
) -> IndexDistanceArrays<'_> {
    let mut indices = Vec::with_capacity(entries.len());
    let mut distances = Vec::with_capacity(entries.len());
    for entry in entries {
        let (index, distance) = index_and_distance(entry);
        indices.push(index);
        distances.push(distance);
    }
    (
        PyArray1::from_vec(py, indices),
        PyArray1::from_vec(py, distances),
    )
}
