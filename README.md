[![gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/candaraz/gta6-twitch-ccu-forecast-01-v4/rank.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/candaraz/gta6-twitch-ccu-forecast-01-v4/reward.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)

# GTA VI launch-day Twitch forecast — v4

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day (2026-11-19 UTC) after the first: for hours 2–24,
five quantiles (5/25/50/75/95%) of that hour's concurrent-viewer count.
Built for [gta6-twitch-ccu-forecast-01](https://make.fun/c/gta6-twitch-ccu-forecast-01)
on [make.fun](https://make.fun).

## Model

A one-factor shape model on rolling-midnight donors, fitted fresh from each
case's own comparables at run time (no hardcoded curves, no memorised
practice answers):

- **Shape factor.** In log space relative to hour 1, the release-day curve
  is dominated by one factor (88% of variance across 41 past launches): the
  evening amplitude *A*, the mean log-ratio over hours 17–22. Within
  rolling-midnight launches, `r_h = a_h + b_h·A` fits with mean residual SD
  ~0.20, and ~0.06–0.11 in the evening hours.
- **Amplitude prediction.** *A* is predicted by a small ridge regression on
  three checkpoint-observable features: ln(hour-1 print), the pre-day peak
  ratio, and the late-ramp slope over the last 6 pre-day hours. The
  ln(hour-1) slope is negative — the scale trap: a huge hour-1 print means
  most of the world already unlocked. The prediction is clamped to the donor
  amplitude range so an unprecedented hour-1 print cannot extrapolate
  catastrophically.
- **Regime-aware donors.** The target's own notes (or, failing that, its
  observable pre-day ramp) decide whether it is a rolling-midnight launch;
  rolling targets use only rolling-midnight donors, classified from each
  comparable's unlock text with a behavioral fallback. Early-access and
  simultaneous-global launches are excluded from the rolling pool.
- **Honest spreads.** Each hour's quantiles come from inner leave-one-out
  residuals over the donor pool — hour-specific, asymmetric, and untuned.

## Validation (self-built; no local grade exists for this competition)

Ground truth for the 4 practice cases was fetched independently from
SullyGnome's public month JSONs (Average viewers) and cross-checked against
each case's own `history_hourly` overlap (all anchors and ramp values exact).
Scoring uses an independent pinball implementation verified against the
RULES.txt worked example (0.112545) and the house baseline (0.387097).

| Check | v4 | Prior best |
|---|---|---|
| Practice-4 mean pinball loss | **0.1598** | 0.180 (v1, server-graded) |
| Practice-4 per case (Wolverine / RDR2 / GoW / SM2) | 0.188 / 0.151 / 0.114 / 0.186 | 0.245 / 0.143 / 0.128 / 0.202 |
| Leave-one-out, 41 launches | 0.386 | 0.565 (v1) |
| Leave-one-out, console-rolling subset (10 folds) | 0.149 | 0.159 (v3) |
| Practice-4 coverage (90% band) | 0.913 | — |
| Determinism | two runs byte-identical | — |
| Runtime | ~0.11 s/case (limit 8 s) | — |

## Files

- `main.py` — the whole entry: Python 3.13, standard library only,
  deterministic, no network. Reads every `*.in` in `/input` (or
  `$MAKEFUN_INPUT_DIR`), writes `*.out` to `/output` (or
  `$MAKEFUN_OUTPUT_DIR`). Any model-path failure falls back to the house
  persistence forecast so the output is always valid.
- `LICENSE` — MIT (from the kit starter).

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
