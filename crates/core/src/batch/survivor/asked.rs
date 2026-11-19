//! What the seed pre-pass asks a bound, and what it is handed to ask with: the producer it asks
//! ([`Asked`] and its impl for the PAA producer), the value that producer answers with
//! ([`Bound`]), and the [`Survey`] of one block of rows. The seed pre-pass reads a producer only
//! through [`Asked`], so a producer that bounds more finely adds an impl here and no seed code of
//! its own.

use core::marker::PhantomData;
use core::ops::Range;

use crate::batch::pass::Shape;
use crate::bound_paa::{Coefficients, Paa, PaaSides, Span};
use crate::estimate::Producer;

/// A block's value from the one-sided producer `P`.
pub(in crate::batch) struct Bound<P> {
    value: f64,
    producer: PhantomData<P>,
}

impl<P: Producer> Bound<P> {
    fn new(value: f64) -> Self {
        const {
            assert!(
                P::SIDES.bounds_above(),
                "the filter reads upper bounds only"
            )
        };
        Self {
            value,
            producer: PhantomData,
        }
    }

    /// The value as the producer computed it, without the slack: what the seed pre-pass ranks
    /// blocks by, where only the order matters.
    pub(in crate::batch) fn ranked(&self) -> f64 {
        self.value
    }
}

/// A producer the seed pre-pass asks, part by part, for the value its block over a part is judged
/// by, after reading once per block what does not depend on the part ([`Asked::Rows`]).
pub(in crate::batch) trait Asked: Producer + Sized {
    /// What the producer reads of the block's rows, once per block.
    type Rows;
    /// The block's rows as the producer reads them, `None` when it bounds nothing there.
    fn rows(survey: &Survey) -> Option<Self::Rows>;
    fn value(rows: &Self::Rows, survey: &Survey, part: &Range<usize>) -> Bound<Self>;
    /// [`Self::value`] over a sub-block, with the box of exactly its own columns where the
    /// producer keeps boxes; the same as [`Self::value`] for a producer that reads no boxes.
    #[inline(always)]
    fn fine_value(rows: &Self::Rows, survey: &Survey, part: &Range<usize>) -> Bound<Self> {
        Self::value(rows, survey, part)
    }
}

impl<const K: usize> Asked for Paa<K> {
    /// The box of the block's valid rows' coefficients.
    type Rows = Span<K>;
    #[inline]
    fn rows(survey: &Survey) -> Option<Span<K>> {
        survey.paa.rows.rows_span(survey.rows.clone())
    }
    /// [`crate::bound_paa::Coefficients::value`] between the rows' box and the box of the
    /// columns the part's pairs fall in.
    #[inline]
    fn value(rows: &Span<K>, survey: &Survey, part: &Range<usize>) -> Bound<Self> {
        let sides = survey.paa;
        Bound::new(
            sides
                .cols
                .value(rows, &sides.cols.columns_span(survey.columns(part))),
        )
    }
    /// The value over a sub-block: the rows' box against the box of the sub-block's own
    /// columns, window by window.
    #[inline]
    fn fine_value(rows: &Span<K>, survey: &Survey, part: &Range<usize>) -> Bound<Self> {
        let sides = survey.paa;
        Bound::new(
            sides
                .cols
                .value(rows, &sides.cols.window_span(survey.columns(part))),
        )
    }
}

/// What the seed pre-pass is handed for one block of rows of a self-join's triangle: the rows,
/// the exclusion zone that sets the slots' columns, and the PAA coefficients of the rows' and the
/// columns' column sets.
pub(in crate::batch) struct Survey<'a> {
    rows: Range<usize>,
    shape: Shape,
    paa: PaaSides<'a>,
}

impl<'a> Survey<'a> {
    /// The block of `rows` of a self-join over windows with PAA coefficients `paa`: the triangle's
    /// slots, the whole series as columns.
    pub(in crate::batch) fn of_seeds(
        rows: Range<usize>,
        exclusion: usize,
        paa: &'a Coefficients,
    ) -> Self {
        Self {
            rows,
            shape: Shape::Triangle { exclusion },
            paa: PaaSides::of(paa),
        }
    }

    /// The columns the block's pairs over `part` fall in.
    fn columns(&self, part: &Range<usize>) -> Range<usize> {
        let (first, last) = (self.rows.start, self.rows.end - 1);
        let column = |row: usize, slot: usize| (row as i64 + self.shape.offset(slot)) as usize;
        column(first, part.end - 1)..column(last, part.start) + 1
    }
}
