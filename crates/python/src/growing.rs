//! `GrowingStream`: the unbounded engine ([`ladde::stream::GrowingStream`]) as seen
//! from Python. Every method here only converts arguments in and [`Entry`] values out; the
//! arithmetic is exactly core's own.

use ladde::entry::Entry;
use ladde::index::SampleIndex;
use ladde::stream::GrowingStream as CoreGrowingStream;
use ladde::stream_api::Stream;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::convert::{Floats, IndexDistanceArrays, entries_to_arrays, index_and_distance};
use crate::err::{Remedy, checkpoint_error, countable, tick_scale, window_length};

/// A causal left nearest-neighbor matrix profile over a stream that never forgets a sample:
/// every window, once completed, keeps its place forever.
///
/// `m` is the subsequence length (at least 3); `scale` is the tick size each sample is
/// quantized to before the arithmetic ever sees it. A finite sample beyond 2**53 ticks of `scale`
/// raises `ValueError` and is not folded in (by `update_many`, none of its samples is), where
/// STUMPY, which works in floats, keeps it.
#[pyclass(module = "ladders")]
pub struct GrowingStream {
    inner: CoreGrowingStream,
}

#[pymethods]
impl GrowingStream {
    #[new]
    fn new(m: i64, scale: f64) -> PyResult<Self> {
        let m = window_length(m)?;
        let scale = tick_scale(scale)?;
        Ok(Self {
            inner: CoreGrowingStream::new(m, scale),
        })
    }

    /// Folds in one sample. Returns `None` until it completes a window; then `(index,
    /// distance)` for that window's nearest earlier neighbor. No neighbor yet: index `-1`,
    /// distance `inf` (STUMPY's own convention for an unmatched row).
    fn update(&mut self, x: f64) -> PyResult<Option<(i64, f64)>> {
        countable(&[x], self.inner.scale(), Remedy::Restart)?;
        Ok(self
            .inner
            .update(x)
            .map(|(_, entry)| index_and_distance(entry)))
    }

    /// Folds in every sample of `xs`, in order. Returns `(indices, distances)`: one pair per
    /// window `xs` completes, in the order they complete — exactly what calling
    /// [`Self::update`] once per sample, in a Python loop, would have returned.
    fn update_many<'py>(
        &mut self,
        py: Python<'py>,
        xs: Floats<'py>,
    ) -> PyResult<IndexDistanceArrays<'py>> {
        let xs = xs.as_array().to_vec();
        countable(&xs, self.inner.scale(), Remedy::Restart)?;
        let mut out: Vec<(SampleIndex, Entry)> = Vec::new();
        py.detach(|| self.inner.update_many(&xs, &mut out));
        Ok(entries_to_arrays(py, out))
    }

    /// The stream's whole state as bytes: `restore` of them continues bit for bit like this
    /// stream would.
    fn checkpoint<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.checkpoint())
    }

    /// A stream rebuilt from `checkpoint()` bytes. Raises `ValueError` for any bytes that are
    /// not a checkpoint of this kind of stream.
    #[staticmethod]
    fn restore(data: &[u8]) -> PyResult<Self> {
        CoreGrowingStream::restore(data)
            .map(|inner| Self { inner })
            .map_err(checkpoint_error)
    }

    /// Pickle support: rebuilds through `restore` from the checkpoint bytes.
    fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyBytes>,))> {
        let restore = py.get_type::<Self>().getattr("restore")?;
        Ok((restore, (self.checkpoint(py),)))
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<GrowingStream>()?;
    Ok(())
}
