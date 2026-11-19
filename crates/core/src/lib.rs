//! Items hidden from the documentation, and everything behind the `oracle` feature, are not part
//! of the stable API.
//!
//! Given a [`quantize::Scale`], a result is the same bits in every release. A scale the library
//! infers follows a numbered rule, [`quantize::SCALE_RULE`], and a result on an inferred scale
//! changes only when that number does.

#![no_std]

extern crate alloc;
// Lets test files shared with the integration tests name the crate the same way.
#[cfg(test)]
extern crate self as ladde;
#[cfg(any(feature = "std", feature = "parallel", feature = "oracle", test))]
extern crate std;

pub mod adapter;
pub mod batch;
mod bound_paa;
pub mod bounded;
pub mod capacity;
pub mod chain;
pub mod checkpoint;
#[doc(hidden)]
pub mod columns;
#[doc(hidden)]
pub mod cross_term;
pub mod discord;
pub mod entry;
#[doc(hidden)]
pub mod estimate;
pub mod exclusion;
pub mod floss;
#[cfg(any(test, target_arch = "x86_64"))]
mod full_range;
pub mod index;
mod kbest;
#[doc(hidden)]
pub mod kernel;
#[cfg(target_arch = "x86_64")]
#[doc(hidden)]
pub mod kernel_avx2;
#[doc(hidden)]
pub mod kernel_neon;
pub mod mdl;
pub mod motif;
pub mod mpdist;
pub mod multidim;
#[cfg(any(test, feature = "oracle"))]
#[doc(hidden)]
pub mod naive;
#[doc(hidden)]
pub mod narrow;
pub mod nearest;
pub mod nonnorm;
pub mod nonnorm_bounded;
mod nonnorm_lanes;
pub mod nonnorm_stream;
pub mod ostinato;
pub mod pan;
#[cfg(feature = "parallel")]
pub mod par;
pub mod pattern;
pub mod quantize;
mod right;
mod right_nonnorm;
mod ring;
// Public only where its forcing hook exists; elsewhere it holds the stubs the engines call.
#[cfg(all(any(test, feature = "oracle"), not(any(loom, shuttle))))]
#[doc(hidden)]
pub mod route;
#[cfg(not(all(any(test, feature = "oracle"), not(any(loom, shuttle)))))]
mod route;
#[doc(hidden)]
pub mod row;
pub mod sampler;
#[doc(hidden)]
pub mod sanitize;
pub mod scrump;
pub mod snippets;
#[doc(hidden)]
pub mod spill;
pub mod stats;
pub mod stream;
pub mod stream_api;
pub mod subspace;
#[doc(hidden)]
pub mod sync;
#[doc(hidden)]
pub mod wide;
#[doc(hidden)]
pub mod znorm;
