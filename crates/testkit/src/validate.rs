//! Checks for a left profile's invariants, each naming the row and values that violate it.
//!
//! These take `ladde::entry::Entry`, a different type from the `Entry` core's own
//! unit tests build against `crate::`, so they are reachable only from `crates/core/tests/`.

use ladde::entry::Entry;

/// A row that fails a check, naming the check and the values involved.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Violation {
    ExclusionZone {
        row: u64,
        neighbor: u64,
        zone: u64,
    },
    Sentinel {
        row: u64,
        neighbor: Option<u64>,
        squared_distance: f64,
    },
    DistanceBounds {
        row: u64,
        squared_distance: f64,
        bound: f64,
    },
    Causal {
        row: u64,
        neighbor: u64,
    },
    CompleteRows {
        row: u64,
    },
}

/// No neighbor lies within `e = ceil(m/4)` of its row.
pub fn exclusion_zone(profile: &[(u64, Entry)], m: usize) -> Result<(), Violation> {
    let zone = ladde::exclusion::zone(m) as u64;
    for &(row, entry) in profile {
        if let Some(neighbor) = entry.index().get()
            && row.abs_diff(neighbor) <= zone
        {
            return Err(Violation::ExclusionZone {
                row,
                neighbor,
                zone,
            });
        }
    }
    Ok(())
}

/// A row's index is `NONE` exactly when its squared distance is `+inf`.
pub fn sentinel(profile: &[(u64, Entry)]) -> Result<(), Violation> {
    for &(row, entry) in profile {
        let neighbor = entry.index().get();
        let squared_distance = entry.squared_distance();
        if neighbor.is_none() != (squared_distance == f64::INFINITY) {
            return Err(Violation::Sentinel {
                row,
                neighbor,
                squared_distance,
            });
        }
    }
    Ok(())
}

/// Every squared distance other than `+inf` lies in `[0, 4m]`.
pub fn distance_bounds(profile: &[(u64, Entry)], m: usize) -> Result<(), Violation> {
    let bound = 4.0 * m as f64;
    for &(row, entry) in profile {
        let squared_distance = entry.squared_distance();
        if squared_distance != f64::INFINITY && !(0.0..=bound).contains(&squared_distance) {
            return Err(Violation::DistanceBounds {
                row,
                squared_distance,
                bound,
            });
        }
    }
    Ok(())
}

/// Every neighbor index is less than its row's index.
pub fn causal(profile: &[(u64, Entry)]) -> Result<(), Violation> {
    for &(row, entry) in profile {
        if let Some(neighbor) = entry.index().get()
            && neighbor >= row
        {
            return Err(Violation::Causal { row, neighbor });
        }
    }
    Ok(())
}

/// One entry per window start `0..=n − m`, in increasing order, with no duplicates.
pub fn complete_rows(profile: &[(u64, Entry)], n: usize, m: usize) -> Result<(), Violation> {
    let rows = n
        .checked_sub(m)
        .map_or(0, |span| (span as u64).saturating_add(1));
    let mut next = 0u64;
    for &(row, _) in profile {
        if next < rows && row == next {
            next += 1;
        } else if row < next {
            return Err(Violation::CompleteRows { row });
        } else {
            return Err(Violation::CompleteRows { row: next });
        }
    }
    if next == rows {
        Ok(())
    } else {
        Err(Violation::CompleteRows { row: next })
    }
}

/// All four checks above; stops at the first violation found.
pub fn left_profile(profile: &[(u64, Entry)], m: usize) -> Result<(), Violation> {
    exclusion_zone(profile, m)?;
    sentinel(profile)?;
    distance_bounds(profile, m)?;
    causal(profile)
}

#[cfg(test)]
mod tests {
    use super::{
        Violation, causal, complete_rows, distance_bounds, exclusion_zone, left_profile, sentinel,
    };
    use ladde::entry::Entry;
    use ladde::index::SampleIndex;

    fn entry(squared_distance: f64, neighbor: Option<u64>) -> Entry {
        let index = match neighbor {
            Some(i) => SampleIndex::new(i).unwrap(),
            None => SampleIndex::NONE,
        };
        Entry::new(0.0, squared_distance, index).unwrap()
    }

    #[test]
    fn exclusion_zone_accepts_neighbor_outside_zone() {
        let m = 5; // zone = 2
        let profile = [(0, entry(f64::INFINITY, None)), (3, entry(1.0, Some(0)))];
        assert_eq!(exclusion_zone(&profile, m), Ok(()));
    }

    #[test]
    fn exclusion_zone_rejects_neighbor_at_boundary() {
        let m = 5; // zone = ceil(5/4) = 2
        let profile = [(2, entry(1.0, Some(0)))];
        assert_eq!(
            exclusion_zone(&profile, m),
            Err(Violation::ExclusionZone {
                row: 2,
                neighbor: 0,
                zone: 2
            })
        );
    }

    #[test]
    fn exclusion_zone_accepts_neighbor_just_outside_zone() {
        let m = 5; // zone = 2
        let profile = [(3, entry(1.0, Some(0)))];
        assert_eq!(exclusion_zone(&profile, m), Ok(()));
    }

    #[test]
    fn sentinel_accepts_consistent_entries() {
        let profile = [(0, entry(f64::INFINITY, None)), (5, entry(2.0, Some(1)))];
        assert_eq!(sentinel(&profile), Ok(()));
    }

    #[test]
    fn sentinel_rejects_none_index_with_finite_distance() {
        let profile = [(4, entry(2.0, None))];
        assert_eq!(
            sentinel(&profile),
            Err(Violation::Sentinel {
                row: 4,
                neighbor: None,
                squared_distance: 2.0
            })
        );
    }

    #[test]
    fn sentinel_rejects_finite_index_with_infinite_distance() {
        let profile = [(4, entry(f64::INFINITY, Some(1)))];
        assert_eq!(
            sentinel(&profile),
            Err(Violation::Sentinel {
                row: 4,
                neighbor: Some(1),
                squared_distance: f64::INFINITY
            })
        );
    }

    #[test]
    fn distance_bounds_accepts_within_range_and_infinite() {
        let m = 3; // bound = 12
        let profile = [(0, entry(5.0, None)), (1, entry(f64::INFINITY, None))];
        assert_eq!(distance_bounds(&profile, m), Ok(()));
    }

    #[test]
    fn distance_bounds_rejects_negative_infinity() {
        let entry = Entry::new(0.0, f64::NEG_INFINITY, SampleIndex::new(0).unwrap()).unwrap();
        assert!(distance_bounds(&[(10, entry)], 4).is_err());
    }

    #[test]
    fn distance_bounds_accepts_exactly_4m() {
        let m = 3; // bound = 12
        let profile = [(0, entry(12.0, None))];
        assert_eq!(distance_bounds(&profile, m), Ok(()));
    }

    #[test]
    fn distance_bounds_rejects_above_4m() {
        let m = 3; // bound = 12
        let profile = [(0, entry(12.000_001, None))];
        assert_eq!(
            distance_bounds(&profile, m),
            Err(Violation::DistanceBounds {
                row: 0,
                squared_distance: 12.000_001,
                bound: 12.0
            })
        );
    }

    #[test]
    fn distance_bounds_rejects_negative() {
        let m = 3; // bound = 12
        let profile = [(2, entry(-0.5, None))];
        assert_eq!(
            distance_bounds(&profile, m),
            Err(Violation::DistanceBounds {
                row: 2,
                squared_distance: -0.5,
                bound: 12.0
            })
        );
    }

    #[test]
    fn causal_accepts_earlier_neighbor() {
        let profile = [(5, entry(1.0, Some(3)))];
        assert_eq!(causal(&profile), Ok(()));
    }

    #[test]
    fn causal_rejects_neighbor_equal_to_row() {
        let profile = [(5, entry(1.0, Some(5)))];
        assert_eq!(
            causal(&profile),
            Err(Violation::Causal {
                row: 5,
                neighbor: 5
            })
        );
    }

    #[test]
    fn causal_rejects_neighbor_after_row() {
        let profile = [(5, entry(1.0, Some(6)))];
        assert_eq!(
            causal(&profile),
            Err(Violation::Causal {
                row: 5,
                neighbor: 6
            })
        );
    }

    #[test]
    fn complete_rows_accepts_full_profile() {
        let profile = [
            (0, entry(0.0, None)),
            (1, entry(0.0, None)),
            (2, entry(0.0, None)),
            (3, entry(0.0, None)),
        ];
        assert_eq!(complete_rows(&profile, 7, 4), Ok(())); // n-m = 3, rows 0..=3
    }

    #[test]
    fn complete_rows_accepts_single_row_when_n_equals_m() {
        let profile = [(0, entry(0.0, None))];
        assert_eq!(complete_rows(&profile, 4, 4), Ok(())); // n-m = 0, one row
    }

    #[test]
    fn complete_rows_accepts_empty_profile_when_n_below_m() {
        let profile: [(u64, Entry); 0] = [];
        assert_eq!(complete_rows(&profile, 2, 4), Ok(()));
    }

    #[test]
    fn complete_rows_rejects_any_row_when_n_below_m() {
        let profile = [(0, entry(0.0, None))];
        assert_eq!(
            complete_rows(&profile, 2, 4),
            Err(Violation::CompleteRows { row: 0 })
        );
    }

    #[test]
    fn complete_rows_rejects_missing_middle_row() {
        let profile = [
            (0, entry(0.0, None)),
            (1, entry(0.0, None)),
            (3, entry(0.0, None)),
        ];
        assert_eq!(
            complete_rows(&profile, 7, 4), // n-m = 3, rows 0..=3
            Err(Violation::CompleteRows { row: 2 })
        );
    }

    #[test]
    fn complete_rows_rejects_missing_last_row() {
        let profile = [
            (0, entry(0.0, None)),
            (1, entry(0.0, None)),
            (2, entry(0.0, None)),
        ];
        assert_eq!(
            complete_rows(&profile, 7, 4), // n-m = 3, rows 0..=3
            Err(Violation::CompleteRows { row: 3 })
        );
    }

    #[test]
    fn complete_rows_rejects_duplicated_row() {
        let profile = [
            (0, entry(0.0, None)),
            (1, entry(0.0, None)),
            (1, entry(0.0, None)),
        ];
        assert_eq!(
            complete_rows(&profile, 6, 4), // n-m = 2, rows 0..=2
            Err(Violation::CompleteRows { row: 1 })
        );
    }

    #[test]
    fn complete_rows_rejects_swapped_rows() {
        let profile = [
            (1, entry(0.0, None)),
            (0, entry(0.0, None)),
            (2, entry(0.0, None)),
        ];
        assert_eq!(
            complete_rows(&profile, 6, 4), // n-m = 2, rows 0..=2
            Err(Violation::CompleteRows { row: 0 })
        );
    }

    #[test]
    fn complete_rows_rejects_a_row_beyond_the_last_expected() {
        let profile = [
            (0, entry(0.0, None)),
            (1, entry(0.0, None)),
            (2, entry(0.0, None)),
            (3, entry(0.0, None)),
            (4, entry(0.0, None)),
        ];
        assert_eq!(
            complete_rows(&profile, 6, 4), // n-m = 2, rows 0..=2
            Err(Violation::CompleteRows { row: 3 })
        );
    }

    #[test]
    fn complete_rows_does_not_panic_when_the_row_count_overflows() {
        let profile: [(u64, Entry); 0] = [];
        assert_eq!(
            complete_rows(&profile, usize::MAX, 0),
            Err(Violation::CompleteRows { row: 0 })
        );
    }

    #[test]
    fn left_profile_accepts_valid_profile() {
        let m = 4; // zone = 1, bound = 16
        let profile = [
            (0, entry(f64::INFINITY, None)),
            (1, entry(f64::INFINITY, None)),
            (4, entry(9.0, Some(0))),
        ];
        assert_eq!(left_profile(&profile, m), Ok(()));
    }

    #[test]
    fn left_profile_reports_whichever_check_fails() {
        let m = 4; // zone = 1, bound = 16; only the causal check fails here
        let profile = [(3, entry(9.0, Some(5)))];
        assert_eq!(
            left_profile(&profile, m),
            Err(Violation::Causal {
                row: 3,
                neighbor: 5
            })
        );
    }
}
