"""A huge count (`k`, `max_matches`, `max_motifs`) returns the result a count equal to the number of
windows does, or a `ValueError` where the result's shape would be too large; an `n_regimes` asking
for more locations than there are windows is a `ValueError`."""

import numpy as np
import pytest
from ladders import (
    DiscordStream,
    discords,
    fluss,
    fluss_series,
    left_profile,
    matrix_profile,
    motifs,
)

from conftest import profile_for, random_walk

M = 20
HUGE = 2**62


def same(a, b):
    return all(np.array_equal(x, y, equal_nan=True) for x, y in zip(a, b))


def test_discords_with_a_huge_k_equals_k_at_the_window_count():
    series = random_walk(300, seed=1)
    windows = len(series) - M + 1
    assert same(discords(series, M, k=HUGE), discords(series, M, k=windows))


def test_left_profile_with_a_huge_k_is_a_value_error():
    with pytest.raises(ValueError):
        left_profile(random_walk(120, seed=2), M, 1e-6, k=HUGE)


def test_fluss_with_more_regimes_than_windows_is_an_error_naming_the_window_count():
    series = random_walk(300, seed=3)
    windows = len(series) - M + 1
    for n_regimes in [windows + 2, 2**62]:
        with pytest.raises(ValueError, match=f"n_regimes={n_regimes} asks for") as error:
            fluss_series(series, M, 10, n_regimes)
        assert f"pass n_regimes={windows + 1} to get every location" in str(error.value)
    cac, locations = fluss_series(series, M, 10, windows + 1)
    assert len(locations) == windows
    index = matrix_profile(series, M)[1]
    with pytest.raises(ValueError, match=f"pass n_regimes={windows + 1}"):
        fluss(index, 10, windows + 2, m=M)
    assert len(fluss(index, 10, windows + 1, m=M)[1]) == windows
    with pytest.raises(ValueError, match="pass n_regimes=1"):
        fluss(np.array([], dtype=np.int64), 10, 2, m=M)


def test_motifs_with_huge_counts_equal_windows_counts():
    series = random_walk(300, seed=4)
    windows = len(series) - M + 1
    assert same(
        motifs(series, profile_for(series, M), scale=1e-6, max_matches=HUGE, max_motifs=HUGE),
        motifs(series, profile_for(series, M), scale=1e-6, max_matches=windows, max_motifs=windows),
    )


def test_a_discord_stream_with_the_largest_k_runs():
    for w in (None, 100):
        stream = DiscordStream(M, 2**28, 1e-6, w)
        for x in random_walk(200, seed=5):
            stream.update(float(x))
