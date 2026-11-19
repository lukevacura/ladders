//! The stream behind a class that feeds a consumer (a discord tracker, FLOSS): growing when no
//! window `w` is given, bounded to the last `w` samples when it is. One place decides which,
//! so the consumers and their pool or horizon always agree with it.

use core::num::NonZeroUsize;

use ladde::bounded::BoundedStream;
use ladde::entry::Entry;
use ladde::index::SampleIndex;
use ladde::nearest::ExactDistance;
use ladde::quantize::Scale;
use ladde::stream::GrowingStream;
use ladde::stream_api::Stream;

pub(crate) enum Engine {
    Growing(GrowingStream),
    Bounded(BoundedStream),
}

impl Engine {
    /// `w` (already checked to be at least `m`) bounds the search; `None` keeps everything.
    pub(crate) fn new(m: NonZeroUsize, w: Option<NonZeroUsize>, scale: Scale) -> Self {
        match w {
            None => Self::Growing(GrowingStream::new(m, scale)),
            Some(w) => Self::Bounded(
                BoundedStream::new(m, w, scale).expect("w >= m is checked by the caller"),
            ),
        }
    }
}

impl Engine {
    /// The tick size each sample is quantized to.
    pub(crate) fn scale(&self) -> Scale {
        match self {
            Self::Growing(stream) => stream.scale(),
            Self::Bounded(stream) => stream.scale(),
        }
    }

    /// [`Stream::update`], keeping the exact correlation the entry was ranked by.
    pub(crate) fn update_exact(&mut self, x: f64) -> Option<(SampleIndex, Option<ExactDistance>)> {
        match self {
            Self::Growing(stream) => stream.update_exact(x),
            Self::Bounded(stream) => stream.update_exact(x),
        }
    }
}

impl Stream for Engine {
    fn update_flagged(&mut self, x: f64, constant: Option<bool>) -> Option<(SampleIndex, Entry)> {
        match self {
            Self::Growing(stream) => stream.update_flagged(x, constant),
            Self::Bounded(stream) => stream.update_flagged(x, constant),
        }
    }

    fn update_many(&mut self, xs: &[f64], out: &mut Vec<(SampleIndex, Entry)>) {
        match self {
            Self::Growing(stream) => stream.update_many(xs, out),
            Self::Bounded(stream) => stream.update_many(xs, out),
        }
    }
}
