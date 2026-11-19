"""`left_profile(..., threads=)`: the thread count never changes a bit, the GIL is released
while the batch runs, and a bad count is a `ValueError`. Nothing here depends on timing; the
rule that threads run in parallel is tested in the core crate by ordering.

The GIL test needs a wheel built with the hidden hooks: `maturin develop --features test-hooks`
(it skips otherwise)."""

import os
import threading

import numpy as np
import pytest
import ladders
from ladders import GrowingStream, left_profile

from conftest import (
    assert_mostly_finite,
    flat_run_series,
    random_walk,
    walk_with_gaps,
)

M = 50
THREADS = [1, 2, 3, 8, None, np.int64(4)]

SERIES = {
    "random_walk": (lambda: random_walk(4000, seed=30), 1.0),
    # 1e8 offset on a 0.01 grid, as a market price would be.
    "market_level": (lambda: 1e8 + 0.01 * random_walk(4000, seed=31), 0.01),
    "flat_run": (lambda: flat_run_series(4000, seed=32, run_at=300, run_len=80), 1.0),
    "with_gaps": (lambda: walk_with_gaps(4000, seed=33, gap_every=301, leading_gap=5), 1.0),
}
FINITE_THRESHOLD = {"random_walk": 0.9, "market_level": 0.9, "flat_run": 0.9, "with_gaps": 0.2}


def u64(x):
    return np.asarray(x).view(np.uint64)


@pytest.mark.parametrize("name", SERIES)
def test_every_thread_count_equals_one_thread_and_the_stream_on_bits(name):
    make, scale = SERIES[name]
    series = make()
    p1, i1, s1 = left_profile(series, M, scale, threads=1)
    assert s1 == scale
    assert_mostly_finite(i1, M, threshold=FINITE_THRESHOLD[name])
    assert np.isfinite(p1).any()

    # The Rust single-threaded engine through an independent path.
    stream_i, stream_p = GrowingStream(M, scale).update_many(series)
    assert np.array_equal(stream_i, i1)
    assert np.array_equal(u64(stream_p), u64(p1))

    for threads in THREADS:
        p, i, s = left_profile(series, M, scale, threads=threads)
        assert s == s1
        assert np.array_equal(i, i1), threads
        assert np.array_equal(u64(p), u64(p1)), threads


def test_scale_none_is_the_same_for_every_thread_count():
    series = SERIES["market_level"][0]()
    p1, i1, s1 = left_profile(series, M, threads=1)
    for threads in THREADS:
        p, i, s = left_profile(series, M, threads=threads)
        assert s == s1
        assert np.array_equal(i, i1)
        assert np.array_equal(u64(p), u64(p1))


def test_threads_is_the_fourth_positional_argument():
    series = random_walk(200, seed=34)
    p, i, _ = left_profile(series, M, 1.0, 2)
    p1, i1, _ = left_profile(series, M, 1.0, 1)
    assert np.array_equal(u64(p), u64(p1)) and np.array_equal(i, i1)


def test_more_threads_than_work_is_fine():
    series = random_walk(60, seed=35)
    p, i, _ = left_profile(series, M, 1.0, threads=10**9)
    p1, i1, _ = left_profile(series, M, 1.0, threads=1)
    assert np.array_equal(u64(p), u64(p1)) and np.array_equal(i, i1)


@pytest.mark.skipif(
    not hasattr(ladders, "_hold_arm"),
    reason="needs the test hooks: build the wheel with `maturin develop --features test-hooks`",
)
def test_the_gil_is_released_while_a_call_runs():
    """The call parks inside the batch engine until this thread tells it to go on, which this
    thread can only do if the call released the GIL. A held GIL makes the call give up after a
    long hang guard and `_hold_release` report it; nothing here depends on speed."""
    series = random_walk(400, seed=36)
    result = []
    ladders._hold_arm()
    worker = threading.Thread(
        target=lambda: result.append(left_profile(series, 50, 1.0, threads=1)), daemon=True
    )
    worker.start()
    try:
        parked = ladders._hold_wait_parked()
    finally:
        released_in_time = ladders._hold_release()
    # timing-ok: a hang-guard bound only; correct code never waits on the clock
    worker.join(timeout=120)
    assert not worker.is_alive()
    assert parked and released_in_time
    assert len(result) == 1


@pytest.mark.parametrize("bad", [0, -1, -(10**30), 10**30, 1.5, 2.0, "2", True, False, [2], np.float64(2), np.True_, np.False_])
def test_bad_threads_raise_value_error(bad):
    series = random_walk(100, seed=38)
    with pytest.raises(ValueError):
        left_profile(series, M, 1.0, threads=bad)


def test_threads_resolution_is_exact():
    """`None` is every core the core crate reports; an int is passed through untouched."""
    resolve = ladders._resolve_threads
    available = ladders._threads_available()
    assert available >= 1
    cores = os.cpu_count()  # timing-ok: upper bound check only; the value is not compared for equality
    if cores:
        assert available <= cores
    assert resolve() == available
    assert resolve(None) == available
    for k in (1, 2, 3, 4, 8, 10**9):
        assert resolve(k) == k
    assert resolve(np.int64(4)) == 4
    for bad in (0, -1, True, np.True_, 1.5, "2"):
        with pytest.raises(ValueError):
            resolve(bad)


def test_tile_report_counts_wide_tiles_only_where_the_range_needs_them():
    narrow = random_walk(4000, seed=40)
    report = ladders.left_profile_report(narrow, M, 1.0, threads=1)
    assert report["tiles"] >= 1 and report["wide_tiles"] == 0

    # Ticks spread over a whole narrow range everywhere: no tile can avoid the exact pass. (A few
    # far-off ticks would not do: they are outliers, which the cold route takes off the tiles.)
    spread = np.random.default_rng(40).uniform(-1.5e9, 1.5e9, narrow.size)
    report = ladders.left_profile_report(spread, M, 1.0, threads=1)
    assert report["tiles"] >= 1 and report["wide_tiles"] == report["tiles"]
    assert ladders.left_profile_report(spread, M, None, threads=1)["tiles"] >= 1

    for bad in (0, True):
        with pytest.raises(ValueError):
            ladders.left_profile_report(narrow, M, 1.0, threads=bad)
