//! Python bindings: every entry crossing into Python is bit-identical to the Rust value it
//! came from, built directly on `ladde`'s own stream and batch engines — never a
//! second implementation of either.
//!
//! Empty when built as a test (`--lib`, which cargo honors over `test = false`): a test binary
//! of the bindings would link libpython without an rpath and abort at start; they are tested
//! from Python. Also empty under `--cfg loom`, where the threaded batch entry points it wraps do
//! not exist.

#![cfg(not(any(test, loom)))]

mod aamp;
mod aampi;
mod batch;
mod bounded;
mod chains;
mod constant;
mod convert;
mod discord;
mod discords;
mod engine;
mod err;
mod floss;
mod growing;
#[cfg(feature = "test-hooks")]
mod hold;
mod mdl;
mod mmotifs;
mod motif;
mod motif_pairs;
mod mpdist;
mod multi;
mod ostinato;
mod pan;
mod pattern;
mod scrump;
mod snippets;
mod stumpi;
mod subspace;

use pyo3::prelude::*;

#[pymodule(name = "_native")]
fn ladders(module: &Bound<'_, PyModule>) -> PyResult<()> {
    growing::register(module)?;
    bounded::register(module)?;
    stumpi::register(module)?;
    aampi::register(module)?;
    discord::register(module)?;
    pattern::register(module)?;
    floss::register(module)?;
    batch::register(module)?;
    #[cfg(feature = "test-hooks")]
    hold::register(module)?;
    aamp::register(module)?;
    discords::register(module)?;
    chains::register(module)?;
    motif::register(module)?;
    motif_pairs::register(module)?;
    multi::register(module)?;
    mpdist::register(module)?;
    ostinato::register(module)?;
    scrump::register(module)?;
    snippets::register(module)?;
    pan::register(module)?;
    subspace::register(module)?;
    mdl::register(module)?;
    mmotifs::register(module)?;
    Ok(())
}
