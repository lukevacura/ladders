//! `DiscordStream`: a stream and a [`DiscordTracker`] fed from it inside Rust, so the tracker
//! ranks the stream's exact distances — a tracker fed rounded distances from Python could
//! reorder near-ties.

use ladde::discord::DiscordTracker;
use pyo3::prelude::*;

use crate::convert::{Floats, discord_index_and_distance};
use crate::engine::Engine;
use crate::err::{Remedy, bounded_count, countable, tick_scale, window_bound, window_length};

/// The top `k` discords of a left nearest-neighbor stream: the windows whose nearest earlier
/// neighbor is farthest, no two within the exclusion zone of each other.
///
/// `m` is the subsequence length (at least 3); `k` how many discords to list; `scale` the tick
/// size each sample is quantized to. With `w` the stream searches, and the list considers, only
/// the last `w` samples (at least `m`); without it nothing is ever forgotten.
#[pyclass(module = "ladders")]
pub struct DiscordStream {
    stream: Engine,
    tracker: DiscordTracker,
}

#[pymethods]
impl DiscordStream {
    #[new]
    #[pyo3(signature = (m, k, scale, w=None))]
    fn new(m: i64, k: i64, scale: f64, w: Option<i64>) -> PyResult<Self> {
        let m = window_length(m)?;
        let k = bounded_count("k", k)?;
        let scale = tick_scale(scale)?;
        let w = w.map(|w| window_bound(w, m)).transpose()?;
        let tracker = match w {
            None => DiscordTracker::new(m, k),
            Some(w) => DiscordTracker::windowed(m, k, w).expect("w >= m is checked above"),
        };
        Ok(Self {
            stream: Engine::new(m, w, scale),
            tracker,
        })
    }

    /// Folds in one sample.
    fn update(&mut self, x: f64) -> PyResult<()> {
        countable(&[x], self.stream.scale(), Remedy::Restart)?;
        if let Some((row, score)) = self.stream.update_exact(x) {
            self.tracker.update(row, score);
        }
        Ok(())
    }

    /// Folds in every sample of `xs`, in order: exactly what calling `update` once per sample
    /// would do.
    fn update_many<'py>(&mut self, py: Python<'py>, xs: Floats<'py>) -> PyResult<()> {
        let xs = xs.as_array().to_vec();
        countable(&xs, self.stream.scale(), Remedy::Restart)?;
        py.detach(|| {
            for x in xs {
                if let Some((row, score)) = self.stream.update_exact(x) {
                    self.tracker.update(row, score);
                }
            }
        });
        Ok(())
    }

    /// The discords so far as `(index, distance)`, best first (at most `k`): `index` is the
    /// window's first sample, `distance` its distance to its nearest earlier neighbor.
    fn top(&self) -> Vec<(i64, f64)> {
        self.tracker
            .top()
            .iter()
            .map(|&d| discord_index_and_distance(d))
            .collect()
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<DiscordStream>()?;
    Ok(())
}
