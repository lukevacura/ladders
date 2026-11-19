use super::Entry;
use crate::index::SampleIndex;

#[test]
fn none_is_the_absence_of_a_neighbor() {
    assert_eq!(Entry::NONE.squared_distance(), f64::INFINITY);
    assert!(Entry::NONE.index().is_none());
}

#[test]
fn nan_is_unrepresentable() {
    let index = SampleIndex::new(0).unwrap();
    assert!(Entry::new(f64::NAN, 1.0, index).is_none());
    assert!(Entry::new(1.0, f64::NAN, index).is_none());
}
