# Benchmarks

Measured once, on one machine type: an AWS `c6a.4xlarge` (AMD EPYC 7R13, AVX2, generic build), on
2026-11-18, against STUMPY 1.14.1 and SCAMP (pyscamp 4.0.3, double precision).

NOTE: Bare in mind this was measured on a CPU, not a GPU!

## Speed

Each row is one dataset at one length `n` and window `m`. A figure is the other library's time
divided by this library's, so above 1× means this library was faster. A dash means that
combination was not timed.

**Two-sided profile**

| Dataset | n | m | vs STUMPY, 1 thread | vs STUMPY, 8 threads | vs SCAMP, 1 thread | vs SCAMP, 8 threads |
|---|---|---|---|---|---|---|
| Random walk on a 0.01 grid | 2^14 | 64 | 4.62× | 5.55× | 1.22× | 1.15× |
| Random walk on a 0.01 grid | 2^16 | 16 | 5.13× | 5.35× | 1.48× | 1.57× |
| Random walk on a 0.01 grid | 2^16 | 64 | 5.19× | 5.32× | 1.37× | 1.44× |
| Random walk on a 0.01 grid | 2^16 | 256 | 5.17× | 5.59× | 1.32× | 1.28× |
| Random walk on a 0.01 grid | 2^16 | 1024 | 10.4× | 11.0× | 2.37× | 2.07× |
| Random walk on a 0.01 grid | 2^17 | 64 | 5.48× | 5.30× | 1.47× | 1.52× |
| Random walk on a 0.01 grid | 2^20 | 64 | – | 16.1× | – | 1.63× |
| Random walk on a 0.01 grid | 2^20 | 2048 | – | 87.3× | – | 7.44× |
| Random walk, plain doubles | 2^14 | 64 | 3.94× | 4.70× | 1.04× | 0.97× |
| Random walk, plain doubles | 2^16 | 16 | 4.56× | 4.62× | 1.32× | 1.36× |
| Random walk, plain doubles | 2^16 | 64 | 4.52× | 4.64× | 1.19× | 1.26× |
| Random walk, plain doubles | 2^16 | 256 | 4.63× | 4.97× | 1.18× | 1.14× |
| Random walk, plain doubles | 2^16 | 1024 | 8.09× | 8.88× | 1.84× | 1.70× |
| Random walk, plain doubles | 2^17 | 64 | 4.48× | 4.48× | 1.20× | 1.29× |
| Random walk, plain doubles | 2^20 | 64 | – | 14.1× | – | 1.42× |
| Random walk, plain doubles | 2^20 | 2048 | – | 55.1× | – | 4.68× |
| Tie-heavy (runs and repeats of three values) | 2^14 | 64 | 5.60× | 6.09× | 1.50× | 1.32× |
| Tie-heavy (runs and repeats of three values) | 2^16 | 64 | 7.15× | 6.85× | 1.90× | 1.87× |
| Tie-heavy (runs and repeats of three values) | 2^17 | 64 | 7.90× | 7.43× | 2.13× | 2.09× |
| Grid walk, flat stretches, gaps, 1 misprint | 2^14 | 64 | 4.78× | 5.74× | 1.36× | 1.25× |
| Grid walk, flat stretches, gaps, 1 misprint | 2^16 | 64 | 5.58× | 5.66× | 1.58× | 1.60× |
| Grid walk, flat stretches, gaps, 1 misprint | 2^17 | 64 | 5.61× | 5.61× | 1.63× | 1.66× |
| Grid walk, flat stretches, gaps, 10 misprints | 2^14 | 64 | 4.76× | 5.77× | 1.36× | 1.25× |
| Grid walk, flat stretches, gaps, 10 misprints | 2^16 | 64 | 5.61× | 5.65× | 1.60× | 1.60× |
| Grid walk, flat stretches, gaps, 10 misprints | 2^17 | 64 | 5.61× | 5.66× | 1.63× | 1.66× |
| Grid walk, flat stretches, gaps, 30 misprints | 2^14 | 64 | 4.33× | 5.45× | 1.24× | 1.19× |
| Grid walk, flat stretches, gaps, 30 misprints | 2^16 | 64 | 5.04× | 5.16× | 1.44× | 1.46× |
| Grid walk, flat stretches, gaps, 30 misprints | 2^17 | 64 | 5.07× | 5.10× | 1.47× | 1.52× |
| Grid walk, flat stretches, gaps, 100 misprints | 2^14 | 64 | 4.25× | 5.20× | 1.21× | 1.13× |
| Grid walk, flat stretches, gaps, 100 misprints | 2^16 | 64 | 4.97× | 5.14× | 1.42× | 1.45× |
| Grid walk, flat stretches, gaps, 100 misprints | 2^17 | 64 | 5.08× | 5.16× | 1.45× | 1.52× |
| BTC/USDT trades (Binance, 2024-03-12) | 2^17 | 64 | 5.25× | 5.41× | 1.46× | 1.55× |
| BTC/USDT trades (Binance, 2024-03-12) | 2^17 | 256 | 5.18× | 5.65× | 1.35× | 1.38× |
| BTC/USDT trades (Binance, 2024-03-12) | 2^20 | 64 | – | 16.6× | – | 1.72× |
| BTC/USDT trades (Binance, 2024-03-12) | 2^20 | 256 | – | 16.7× | – | 1.57× |
| EOG_one_hour_400_Hz (MOMP archive) | 2^20 | 256 | – | 15.0× | – | 1.35× |
| CinC_Challenge (MOMP archive) | 2^20 | 256 | – | 17.0× | – | 1.50× |
| stator_winding (MOMP archive) | 2^20 | 256 | – | 15.1× | – | 1.42× |
| house (MOMP archive) | 2^20 | 256 | – | 15.0× | – | 1.35× |
| WindTurbine (MOMP archive) | 2^20 | 256 | – | 15.2× | – | 1.36× |
| water (MOMP archive) | 2^20 | 256 | – | 18.2× | – | 1.70× |
| HAR_Ambient_Sensor_Data (MOMP archive) | 2^20 | 256 | – | 15.1× | – | 1.37× |
| BlackLeggedKittiwake (MOMP archive) | 2^20 | 256 | – | 15.1× | – | 1.34× |

**Two-sided profile, full-length real series**

| Dataset | n | m | vs STUMPY, 8 threads | vs SCAMP, 8 threads | vs SCAMP, 16 threads |
|---|---|---|---|---|---|
| EOG_one_hour_400_Hz (MOMP archive) | 1,439,997 | 8000 | – | 1.14× | 1.12× |
| stator_winding (MOMP archive) | 1,330,816 | 2048 | – | 5.78× | 5.82× |
| water (MOMP archive) | 2,100,777 | 2048 | – | 17.6× | 17.5× |
| BlackLeggedKittiwake (MOMP archive) | 1,288,330 | 2048 | – | 13.9× | 13.4× |
| Challenge2009Respiration500HZ | 1,799,997 | 2048 | – | 19.5× | 18.9× |
| swtAttack7 | 449,919 | 16384 | 22.7× | 1.67× | 2.16× |

**AB-join**

| Dataset | n | m | vs STUMPY, 1 thread | vs STUMPY, 8 threads | vs SCAMP, 1 thread | vs SCAMP, 8 threads |
|---|---|---|---|---|---|---|
| Random walk on a 0.01 grid | 2^16 | 64 | 4.49× | 4.99× | 1.03× | 1.15× |
| Random walk on a 0.01 grid | 2^16 | 1024 | 17.1× | 14.6× | 3.57× | 3.01× |
| Random walk on a 0.01 grid | 2^17 | 64 | 4.56× | 5.48× | 1.04× | 1.12× |
| Random walk on a 0.01 grid | 2^17 | 1024 | 12.6× | 9.65× | 2.84× | 1.97× |
| Random walk, plain doubles | 2^16 | 64 | 4.23× | 4.28× | 0.97× | 0.96× |
| Random walk, plain doubles | 2^16 | 1024 | 12.0× | 13.2× | 2.49× | 2.68× |
| Random walk, plain doubles | 2^17 | 64 | 4.29× | 4.80× | 0.98× | 0.98× |
| Random walk, plain doubles | 2^17 | 1024 | 10.2× | 10.0× | 2.36× | 2.09× |
| Tie-heavy (runs and repeats of three values) | 2^16 | 64 | 6.53× | 5.93× | 1.52× | 1.40× |
| Tie-heavy (runs and repeats of three values) | 2^16 | 1024 | 4.54× | 4.48× | 1.01× | 1.21× |
| Tie-heavy (runs and repeats of three values) | 2^17 | 64 | 5.67× | 5.90× | 1.48× | 1.26× |
| Tie-heavy (runs and repeats of three values) | 2^17 | 1024 | 4.85× | 5.14× | 1.28× | 1.34× |
| BTC/USDT trades (Binance, 2024-03-12) | 2^16 | 64 | 4.90× | 4.84× | 1.18× | 1.17× |
| BTC/USDT trades (Binance, 2024-03-12) | 2^16 | 1024 | 29.8× | 19.5× | 6.09× | 3.66× |
| BTC/USDT trades (Binance, 2024-03-12) | 2^17 | 64 | 4.69× | 5.36× | 1.12× | 1.13× |
| BTC/USDT trades (Binance, 2024-03-12) | 2^17 | 1024 | 13.3× | 10.2× | 2.97× | 2.02× |

**Non-normalized profile (`aamp`)**

| Dataset | n | m | vs STUMPY, 1 thread | vs STUMPY, 8 threads |
|---|---|---|---|---|
| Random walk on a 0.01 grid | 2^14 | 64 | 38.8× | 41.5× |
| Random walk on a 0.01 grid | 2^16 | 16 | 39.8× | 41.3× |
| Random walk on a 0.01 grid | 2^16 | 256 | 39.5× | 46.0× |
| Random walk on a 0.01 grid | 2^16 | 1024 | 36.5× | 48.7× |
| Random walk on a 0.01 grid | 2^17 | 64 | 40.2× | 41.7× |
| Random walk, plain doubles | 2^14 | 64 | 39.0× | 40.5× |
| Random walk, plain doubles | 2^17 | 64 | 40.4× | 43.0× |
| Tie-heavy (runs and repeats of three values) | 2^14 | 64 | 33.9× | 33.0× |
| Tie-heavy (runs and repeats of three values) | 2^17 | 64 | 35.6× | 35.5× |
| Grid walk, flat stretches, gaps, 10 misprints | 2^14 | 64 | 39.5× | 43.1× |
| Grid walk, flat stretches, gaps, 10 misprints | 2^17 | 64 | 40.3× | 42.6× |
| BTC/USDT trades (Binance, 2024-03-12) | 2^17 | 64 | 40.2× | 46.8× |

**Top-k**

| Dataset | n | m | k | vs STUMPY, 1 thread | vs STUMPY, 8 threads |
|---|---|---|---|---|---|
| Random walk on a 0.01 grid | 2^14 | 64 | 1 | 4.60× | 5.54× |
| Random walk on a 0.01 grid | 2^14 | 64 | 4 | 2.97× | 3.66× |
| Random walk on a 0.01 grid | 2^14 | 64 | 16 | 1.87× | 3.49× |
| Random walk on a 0.01 grid | 2^16 | 64 | 1 | 5.27× | 5.41× |
| Random walk on a 0.01 grid | 2^16 | 64 | 4 | 4.12× | 3.92× |
| Random walk on a 0.01 grid | 2^16 | 64 | 16 | 3.16× | 8.72× |
| Random walk, plain doubles | 2^14 | 64 | 1 | 3.93× | 4.70× |
| Random walk, plain doubles | 2^14 | 64 | 4 | 2.70× | 3.53× |
| Random walk, plain doubles | 2^14 | 64 | 16 | 1.86× | 3.28× |
| Random walk, plain doubles | 2^16 | 64 | 1 | 4.51× | 4.63× |
| Random walk, plain doubles | 2^16 | 64 | 4 | 3.71× | 3.58× |
| Random walk, plain doubles | 2^16 | 64 | 16 | 2.95× | 8.40× |
| BTC/USDT trades (Binance, 2024-03-12) | 2^17 | 64 | 1 | 5.37× | 5.30× |
| BTC/USDT trades (Binance, 2024-03-12) | 2^17 | 64 | 4 | 4.40× | 5.35× |
| BTC/USDT trades (Binance, 2024-03-12) | 2^17 | 64 | 16 | 4.11× | 16.1× |

**Multi-dimensional (`mstump`, 3 dimensions)**

| Dataset | n | m | vs STUMPY, 1 thread | vs STUMPY, 8 threads |
|---|---|---|---|---|
| Random walk on a 0.01 grid | 2^13 | 64 | 2.57× | 14.8× |
| Random walk on a 0.01 grid | 2^15 | 64 | 2.62× | 18.0× |
| Random walk, plain doubles | 2^13 | 64 | 2.57× | 14.9× |
| Random walk, plain doubles | 2^15 | 64 | 2.65× | 18.2× |

**Growing stream, per update**

| Dataset | n | m | w | vs STUMPY, 1 thread |
|---|---|---|---|---|
| Random walk on a 0.01 grid | 2^14 | 64 | 1,000 | 53.2× |
| Random walk on a 0.01 grid | 2^14 | 64 | 10,000 | 26.0× |
| Random walk, plain doubles | 2^14 | 64 | 1,000 | 32.1× |
| Random walk, plain doubles | 2^14 | 64 | 10,000 | 12.0× |
| Grid walk, flat stretches, gaps, 30 misprints | 2^14 | 64 | 1,000 | 65.5× |
| Grid walk, flat stretches, gaps, 30 misprints | 2^14 | 64 | 10,000 | 26.2× |
| BTC/USDT trades (Binance, 2024-03-12) | 2^14 | 64 | 1,000 | 66.7× |
| BTC/USDT trades (Binance, 2024-03-12) | 2^14 | 64 | 10,000 | 26.0× |

**Bounded stream, per update**

| Dataset | n | m | w | vs STUMPY, 1 thread |
|---|---|---|---|---|
| Random walk on a 0.01 grid | 2^14 | 64 | 1,000 | 148× |
| Random walk on a 0.01 grid | 2^14 | 64 | 10,000 | 28.2× |
| Random walk, plain doubles | 2^14 | 64 | 1,000 | 39.2× |
| Random walk, plain doubles | 2^14 | 64 | 10,000 | 5.42× |
| Grid walk, flat stretches, gaps, 30 misprints | 2^14 | 64 | 1,000 | 147× |
| Grid walk, flat stretches, gaps, 30 misprints | 2^14 | 64 | 10,000 | 27.2× |
| BTC/USDT trades (Binance, 2024-03-12) | 2^14 | 64 | 1,000 | 145× |
| BTC/USDT trades (Binance, 2024-03-12) | 2^14 | 64 | 10,000 | 26.8× |

**Bounded stream, non-normalized, per update**

| Dataset | n | m | w | vs STUMPY, 1 thread |
|---|---|---|---|---|
| Random walk on a 0.01 grid | 2^14 | 64 | 1,000 | 35.5× |
| Random walk on a 0.01 grid | 2^14 | 64 | 10,000 | 14.2× |
| Random walk, plain doubles | 2^14 | 64 | 1,000 | 35.0× |
| Random walk, plain doubles | 2^14 | 64 | 10,000 | 13.0× |
| Grid walk, flat stretches, gaps, 30 misprints | 2^14 | 64 | 1,000 | 33.2× |
| Grid walk, flat stretches, gaps, 30 misprints | 2^14 | 64 | 10,000 | 13.4× |
| BTC/USDT trades (Binance, 2024-03-12) | 2^14 | 64 | 1,000 | 42.7× |
| BTC/USDT trades (Binance, 2024-03-12) | 2^14 | 64 | 10,000 | 16.9× |

## How to read the speed figures

- **Coverage.** There are 202 cells (a row at one thread count) and 509 timings. The synthetic
  series are seeded random walks and a tie-heavy series; the "misprint" walks add flat stretches,
  gaps and far-off values. The real series are from the
  [MOMP paper](https://sites.google.com/view/momp2024)'s archive, two further recordings
  (`swtAttack7` and `Challenge2009Respiration500HZ`), and a day of Binance BTC/USDT trades.
- **Against SCAMP it is about even in places:** the plain-double walk at n = 2^14 on 8 threads,
  AB-joins of plain doubles at m = 64 (0.96× to 0.98×), and the 1.44-million-sample series at
  m = 8000. The large ratios are long windows on long series, where the skip rule applies.
- **The comparison is CPU only.** SCAMP's GPU path was not measured.
- **Method.** All three engines were timed in the same jobs on the same host. A figure is a ratio
  of medians: 5 rounds below n = 2^16 and 3 from there up, after one discarded warm-up call.
  STUMPY was timed for one round at n = 2^20 and on the full-length series. A stream is one pass,
  compared per update; `w` is the number of samples it starts with, and keeps when bounded. 33 of the 509 timings had rounds more than 3% apart after a re-run.
- **This library was timed through its Python bindings** (`crates/python`), which wrap the
  core. The AB-join rows were timed on the build this snapshot is cut from, in a later
  job on the same machine type, and the other rows on the build just before it.

## Against exact answers

Each engine's nearest neighbors were compared with neighbors computed in exact arithmetic from the
doubles as passed, on the same host: series of 512 and 1024 samples, m = 16 and 64, self-join and
AB-join, 23,328 rows per engine, 22,062 of them with an answer.

| Engine | Farther neighbor | Near-tie, other neighbor | Differs by convention |
|---|---|---|---|
| This library | 0 | 6 | 0 |
| STUMPY | 11 | 7 | 0 |
| SCAMP | 831 | 7 | 625 |

- **Farther neighbor** is an arithmetic error: the chosen neighbor's true distance exceeds the
  nearest by more than 1e-6 (relative), on ordinary windows. All of STUMPY's 11 and SCAMP's 831
  are on one kind of series, the grid walk with gaps and far-off misprints (values up to 1000
  times the level). 294 of SCAMP's 831 are within a window length of a gap. On the plain grid and
  plain-double walks, neither library has any.
- **Near-tie** is another neighbor within 1e-12 (relative) of the nearest. This library's six are
  within 2.4e-13.