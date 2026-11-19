use alloc::vec::Vec;
use core::num::NonZeroUsize;

use super::{InnerLength, SnippetsError, runs, snippets};
use crate::mpdist::Selection;
use crate::quantize::Scale;

fn nz(x: usize) -> NonZeroUsize {
    NonZeroUsize::new(x).unwrap()
}

#[test]
fn inner_length_is_stumpys_ceiling_or_the_given_length_never_clamped() {
    let s = |inner: InnerLength, m| inner.resolve(m).map(NonZeroUsize::get);
    assert_eq!(s(InnerLength::Share(1.0), 10), Ok(10));
    assert_eq!(s(InnerLength::Share(0.25), 10), Ok(3));
    assert_eq!(s(InnerLength::Share(0.3), 10), Ok(3));
    assert_eq!(
        s(InnerLength::Share(9.0), 10),
        Err(SnippetsError::ShareOutOfRange)
    );
    assert_eq!(
        s(InnerLength::Share(0.0001), 10),
        Err(SnippetsError::InnerWindowTooShort)
    );
    assert_eq!(
        s(InnerLength::Length(2), 10),
        Err(SnippetsError::InnerWindowTooShort)
    );
    assert_eq!(s(InnerLength::Length(3), 10), Ok(3));
    assert_eq!(
        s(InnerLength::Length(10), 2),
        Err(SnippetsError::InnerWindowAboveSegment)
    );
    assert_eq!(
        s(InnerLength::Length(2), 2),
        Err(SnippetsError::InnerWindowTooShort)
    );
    assert_eq!(
        s(InnerLength::Share(0.0), 10),
        Err(SnippetsError::EmptyInnerWindow)
    );
    assert_eq!(
        s(InnerLength::Share(-2.0), 10),
        Err(SnippetsError::ShareOutOfRange)
    );
    assert_eq!(
        s(InnerLength::Share(f64::NAN), 10),
        Err(SnippetsError::NotANumber)
    );
    assert_eq!(s(InnerLength::Length(4), 10), Ok(4));
    assert_eq!(
        s(InnerLength::Length(40), 10),
        Err(SnippetsError::InnerWindowAboveSegment)
    );
    assert_eq!(s(InnerLength::Length(10), 10), Ok(10));
    assert_eq!(
        s(InnerLength::Length(11), 10),
        Err(SnippetsError::InnerWindowAboveSegment)
    );
    assert_eq!(
        s(InnerLength::Length(0), 10),
        Err(SnippetsError::EmptyInnerWindow)
    );
}

#[test]
fn runs_are_the_half_open_stretches_of_true() {
    let all = |mask: &[bool]| runs(mask).collect::<Vec<_>>();
    assert_eq!(all(&[]), [] as [(usize, usize); 0]);
    assert_eq!(all(&[false, false]), [] as [(usize, usize); 0]);
    assert_eq!(all(&[true]), [(0, 1)]);
    assert_eq!(all(&[true, true, false, true]), [(0, 2), (3, 4)]);
    assert_eq!(all(&[false, true, true]), [(1, 3)]);
}

#[test]
fn the_stumpy_documentation_example() {
    let t = [584.0, -11.0, 23.0, 79.0, 1001.0, 0.0, -19.0];
    let got = snippets(
        &t,
        nz(3),
        nz(2),
        Scale::new(1.0).unwrap(),
        nz(1),
        InnerLength::default(),
        Selection::default(),
    )
    .unwrap();
    assert_eq!(got.indices, [0, 3]);
    assert_eq!(got.snippets, [584.0, -11.0, 23.0, 79.0, 1001.0, 0.0]);
    assert_eq!(got.fractions, [0.6, 0.4]);
    assert_eq!(
        got.regimes,
        [[0, 0, 1], [0, 2, 3], [0, 4, 5], [1, 1, 2], [1, 3, 4]]
    );
    assert!((got.areas[0] - 9.3441034).abs() < 1e-6);
    assert!((got.areas[1] - 5.81050512).abs() < 1e-6);
    assert_eq!(got.profile_len, 5);
}
