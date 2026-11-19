"""`infer_scale` and `SCALE_RULE`: the scale a call with no `scale` would use is available before
the call, equals the scale the call reports, and given back pins the call's bits."""

import math

import numpy as np
import pytest
import ladders as mp
from ladders import SCALE_RULE, aamp, ab_join, infer_scale, left_profile, snippets

from conftest import random_walk

M = 8

# An irrational factor puts a series on no decimal or binary grid.
OFF_GRID = math.pi


def same_bits(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


SERIES = {
    "integers": random_walk(300, 1),
    "cents": random_walk(300, 2) / 100.0,
    "off_grid": random_walk(300, 3) * OFF_GRID,
    "off_grid_wide": random_walk(300, 4) * OFF_GRID * 1e6,
}


def test_the_rule_has_a_number():
    assert type(SCALE_RULE) is int
    assert SCALE_RULE == 1
    assert mp.SCALE_RULE == 1


@pytest.mark.parametrize("name", SERIES)
@pytest.mark.parametrize("finest", [False, True])
def test_it_is_the_scale_the_call_reports(name, finest):
    series = SERIES[name]
    inferred = infer_scale(series, M, finest=finest)
    assert type(inferred) is float
    reported = left_profile(series, M, "finest" if finest else None).scale
    assert inferred == reported
    if not finest:
        assert infer_scale(series, M) == inferred


def test_grid_data_gets_its_grid_and_off_grid_data_a_power_of_two():
    assert infer_scale(SERIES["integers"], M) == 1.0
    assert infer_scale(SERIES["cents"], M) == 0.01
    off = infer_scale(SERIES["off_grid"], M)
    assert math.log2(off) == round(math.log2(off))
    assert infer_scale(SERIES["off_grid"], M, finest=True) < off


@pytest.mark.parametrize("name", SERIES)
def test_giving_it_back_pins_the_bits(name):
    series = SERIES[name]
    inferred, pinned = left_profile(series, M), left_profile(series, M, infer_scale(series, M))
    assert same_bits(inferred.P, pinned.P) and same_bits(inferred.I, pinned.I)


def test_a_z_normalized_join_infers_each_series_on_its_own():
    a, b = SERIES["off_grid"], SERIES["off_grid_wide"]
    joined = ab_join(a, b, M)
    assert joined.scale_a == infer_scale(a, M)
    assert joined.scale_b == infer_scale(b, M)
    assert joined.scale_a != joined.scale_b


def test_several_series_share_one_scale():
    a, b = SERIES["off_grid"], SERIES["off_grid_wide"][:200]
    shared = infer_scale([a, b], M)
    assert shared == infer_scale((a, b), M)
    assert shared == max(infer_scale(a, M), infer_scale(b, M))
    assert shared != infer_scale(a, M)
    # The non-normalized join infers this scale: given it, the same bits.
    for inferred, pinned in zip(aamp(a, M, b), aamp(a, M, b, scale=shared)):
        assert same_bits(inferred, pinned)
    # Given one series' own scale it is a different profile.
    assert not same_bits(aamp(a, M, b).P, aamp(a, M, b, scale=infer_scale(a, M)).P)


def test_the_rows_of_a_2d_array_are_several_series():
    rows = np.stack([SERIES["off_grid"], SERIES["off_grid_wide"]])
    assert infer_scale(rows, M) == infer_scale([rows[0], rows[1]], M)
    assert infer_scale([SERIES["off_grid"]], M) == infer_scale(SERIES["off_grid"], M)
    assert infer_scale(list(SERIES["integers"]), M) == 1.0


def test_snippets_quantizes_at_its_inner_window():
    series, m, s = SERIES["off_grid"], 32, 8
    assert infer_scale(series, s) != infer_scale(series, m)
    inferred = snippets(series, m, 2, s=s)
    pinned = snippets(series, m, 2, s=s, scale=infer_scale(series, s))
    for left, right in zip(inferred, pinned):
        assert same_bits(left, right)


@pytest.mark.parametrize(
    "series, m, error",
    [
        (SERIES["integers"], 2, ValueError),
        (SERIES["integers"][:5], M, ValueError),
        (np.full(50, np.nan), M, ValueError),
        ([np.full(50, np.nan), np.full(50, np.inf)], M, ValueError),
        ([SERIES["integers"][:3], SERIES["cents"][:3]], M, ValueError),
        (SERIES["integers"].astype(complex), M, TypeError),
        (3.5, M, TypeError),
        ("series", M, ValueError),
        ([], M, ValueError),
    ],
)
def test_what_has_no_scale_is_an_error(series, m, error):
    with pytest.raises(error):
        infer_scale(series, m)
