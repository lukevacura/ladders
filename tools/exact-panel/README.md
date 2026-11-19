# exact-panel: three engines against exact answers

This is the comparison behind "Against exact answers" in [BENCHMARKS.md](../../BENCHMARKS.md). It
runs `ladders`, STUMPY and SCAMP on small seeded series and classifies every row against nearest
neighbors computed in exact arithmetic by [`../exact-ref`](../exact-ref), which shares no code
with any of the three.

## Run it

```sh
PY_OURS=/path/to/python PY_STUMPY=/path/to/python PY_SCAMP=/path/to/python ./run.sh
```

Each variable is a Python interpreter; they can be the same one. Leave `PY_SCAMP` unset to skip
SCAMP.

| Interpreter | Needs |
|---|---|
| `PY_OURS` | numpy and a `ladders` wheel (see the top-level README) |
| `PY_STUMPY` | numpy and `stumpy==1.14.1` |
| `PY_SCAMP` | numpy and `pyscamp==4.0.3` |

It takes a minute or two on 8 cores and writes to `out/`: `classify.md` (every cell),
`breakdown.md` (the summary below) and `classify.json`.

## What it runs

- **Series:** 512 and 1024 samples, four kinds, generated from fixed seeds by `make_data.py`:
  a random walk on a 0.01 grid, a random walk in plain doubles, a tie-heavy series, and a grid
  walk with flat stretches, gaps and 10 far-off misprints.
- **Workloads:** the two-sided self-join and the AB-join, at m = 16 and 64.
- **Calls:** `stumpy.stump`, and `pyscamp.selfjoin` / `abjoin` with `precision="double"`, each on
  one thread. `ladders` runs at its default scale and at `scale="finest"`.
- **Truth:** the doubles exactly as passed. Every finite double is a dyadic rational, so the
  samples are exact integers at a power-of-two scale and no rounding enters.

The truth follows STUMPY's conventions (see `../exact-ref`): the exclusion zone is
|i − j| ≤ ⌈m/4⌉, a constant window has defined distances, and a window with a gap has no answer.

## Reading the result

`breakdown.py` splits the rows that disagree with the truth by cause:

| Column | Meaning |
|---|---|
| farther | The chosen neighbor's true distance exceeds the nearest by more than 1e-6 (relative), on ordinary windows. An arithmetic error. |
| near-tie | Another neighbor, within 1e-12 (relative) of the nearest. |
| zone edge | The chosen neighbor is inside the exclusion zone. A difference of convention. |
| constant | The row or the true neighbor is a constant window. A difference of convention. |

The recorded run (x86_64, 2026-11-18):

| engine | rows | farther | near-tie | zone edge | constant |
|---|---|---|---|---|---|
| ladders, default scale | 23328 | 0 | 6 | 0 | 0 |
| ladders, finest scale | 23328 | 0 | 6 | 0 | 0 |
| STUMPY | 23328 | 11 | 7 | 0 | 0 |
| SCAMP | 23328 | 831 | 7 | 64 | 561 |

A rerun on Apple silicon without SCAMP gave the same `ladders` and STUMPY rows.
