"""`fluss_series(t, m, L, n_regimes, ...)` and `fluss(I, L, n_regimes, ...)`: STUMPY's whole-series segmentation `(cac, regime_locs)`.

Given the same index and the same idealized arc curve, the corrected arc curve equals STUMPY's
on bits and the regime locations are equal. The default idealized arc curve is a closed form,
not STUMPY's seeded beta fit, so the default curve is only compared in shape.
"""

import ladders as mp
import numpy as np
import pytest
import stumpy

from conftest import random_walk


def u64(x):
    return np.asarray(x).view(np.uint64)


def noisy(parts, seed, noise=0.05):
    rng = np.random.default_rng(seed)
    t = np.concatenate(parts)
    return t + noise * rng.normal(size=len(t))


def sine(n, period, amp=1.0):
    return amp * np.sin(2 * np.pi * np.arange(n) / period)


def saw(n, period):
    return 2 * ((np.arange(n) % period) / period) - 1


def square(n, period):
    return np.sign(np.sin(2 * np.pi * np.arange(n) / period))


SEGMENTED = {
    "sine_saw_square": lambda: noisy([sine(500, 40), saw(500, 25), square(500, 60)], seed=1),
    "two_sines": lambda: noisy([sine(700, 30), sine(800, 55, 1.5)], seed=2),
    "four_regimes": lambda: noisy(
        [sine(400, 20), saw(400, 33), square(400, 50), sine(400, 70, 2.0)], seed=3
    ),
    "walk": lambda: np.cumsum(np.random.default_rng(4).normal(size=1500)),
}

SETTINGS = [
    # (m, L, n_regimes, excl_factor)
    (30, 30, 3, 5),
    (30, 40, 2, 5),
    (20, 20, 4, 3),
    (25, 25, 3, 0),
]


def parabola(width):
    """A custom idealized arc curve with zeros in it, so the zero guard is exercised."""
    c = np.arange(width)
    return np.minimum(c - 3.0, width - c - 3.0).clip(min=0.0)


def stumpy_fluss(index, length, n_regimes, excl_factor, iac):
    cac, locs = stumpy.fluss(
        index.copy(), length, n_regimes, excl_factor, None if iac is None else iac.copy()
    )
    return cac, locs


@pytest.mark.parametrize("name", SEGMENTED)
@pytest.mark.parametrize("m, length, n_regimes, excl_factor", SETTINGS)
def test_from_stumpys_own_index_equals_stumpy_with_the_same_ideal(
    name, m, length, n_regimes, excl_factor
):
    series = SEGMENTED[name]()
    index = stumpy.stump(series, m)[:, 1].astype(np.int64)
    iac = parabola(len(index))

    want_cac, want_locs = stumpy_fluss(index, length, n_regimes, excl_factor, iac)
    cac, locs = mp.fluss(index, length, n_regimes, excl_factor, iac, m=m)

    assert np.array_equal(u64(cac), u64(want_cac))
    assert np.array_equal(locs, want_locs)
    assert locs.dtype == np.int64 and len(locs) == n_regimes - 1


@pytest.mark.parametrize("seed", range(6))
def test_arbitrary_indices_including_none_and_self_equal_stumpy(seed):
    rng = np.random.default_rng(seed)
    width = 600
    index = rng.integers(-1, width, size=width)
    index[rng.random(width) < 0.2] = -1
    iac = parabola(width) + 1.0
    want_cac, want_locs = stumpy_fluss(index, 20, 4, 5, iac)
    cac, locs = mp.fluss(index, 20, 4, 5, iac, m=20)
    assert np.array_equal(u64(cac), u64(want_cac))
    assert np.array_equal(locs, want_locs)


@pytest.mark.parametrize("name", ["sine_saw_square", "two_sines", "four_regimes"])
def test_default_ideal_finds_stumpys_regimes_and_stays_near_its_curve(name):
    series = SEGMENTED[name]()
    m, length, n_regimes = 30, 30, 3 if name != "two_sines" else 2
    index = stumpy.stump(series, m)[:, 1].astype(np.int64)
    want_cac, want_locs = stumpy.fluss(index, length, n_regimes)
    cac, locs = mp.fluss(index, length, n_regimes, m=m)
    assert sorted(locs) == sorted(want_locs)
    assert np.abs(cac - want_cac).max() < 0.1


def tie_free(n, seed):
    return np.cumsum(np.random.default_rng(seed).normal(size=n))


@pytest.mark.parametrize("name", ["sine_saw_square", "two_sines", "walk"])
def test_end_to_end_equals_stumpys_pipeline_with_the_same_ideal(name):
    series = SEGMENTED[name]()
    m, length, n_regimes = 30, 30, 3
    stumpy_index = stumpy.stump(series, m)[:, 1].astype(np.int64)
    ours_index = mp.matrix_profile(series, m, 1e-6)[1]
    assert np.array_equal(ours_index, stumpy_index)

    iac = parabola(len(stumpy_index)) + 1.0
    want_cac, want_locs = stumpy_fluss(stumpy_index, length, n_regimes, 5, iac)
    cac, locs = mp.fluss_series(series, m, length, n_regimes, scale=1e-6, custom_iac=iac)
    assert np.array_equal(u64(cac), u64(want_cac))
    assert np.array_equal(locs, want_locs)


@pytest.mark.parametrize("threads", [1, 2, 7, None])
def test_thread_count_changes_no_bit(threads):
    series = SEGMENTED["four_regimes"]()
    base_cac, base_locs = mp.fluss_series(series, 30, 30, 4, threads=1)
    cac, locs = mp.fluss_series(series, 30, 30, 4, threads=threads)
    assert np.array_equal(u64(cac), u64(base_cac))
    assert np.array_equal(locs, base_locs)


@pytest.mark.parametrize("seed", range(4))
def test_default_curve_is_a_probability_with_forced_ends(seed):
    series = random_walk(700, seed)
    cac, locs = mp.fluss_series(series, 20, 20, 3, scale=1.0)
    assert len(cac) == len(series) - 20 + 1
    assert np.all((cac >= 0.0) & (cac <= 1.0))
    assert np.all(cac[:100] == 1.0) and np.all(cac[-100:] == 1.0)


def test_signature_and_arguments():
    series = SEGMENTED["two_sines"]()
    cac, locs = mp.fluss_series(series, 30, 30, 1)
    assert len(locs) == 0
    with pytest.raises(ValueError):
        mp.fluss_series(series, 30, 0, 2)
    with pytest.raises(ValueError):
        mp.fluss_series(series, 30, 30, 0)
    with pytest.raises(ValueError):
        mp.fluss_series(series, 30, 30, 2, excl_factor=-1)
    with pytest.raises(ValueError):
        mp.fluss_series(series, 30, 30, 2, custom_iac=np.ones(5))
    with pytest.raises(ValueError):
        mp.fluss_series(series, 2, 30, 2)
