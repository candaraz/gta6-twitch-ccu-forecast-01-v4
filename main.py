"""gta6-twitch-ccu-forecast-01 entry (v4): a one-factor shape model on
rolling-midnight donors, with regime-aware donor selection, bounded
evening-amplitude prediction, and inner-leave-one-out residual spreads.

Python 3.13, standard library only, deterministic, no network. Run as
`python3 -P -s /bot/main.py`; reads every <id>.in in /input (or
$MAKEFUN_INPUT_DIR) and writes <id>.out to /output (or $MAKEFUN_OUTPUT_DIR).

How it forecasts
----------------
The release-day curve is modelled in log space relative to hour 1's print:
r_h = ln(y_h) - ln(y_1) for hours 2..24. Across the 41 past launches in the
input, that curve is dominated by a single factor (88% of variance): the
evening amplitude A, the mean of r over hours 17..22. Within rolling-midnight
launches, r_h = a_h + b_h * A fits with mean residual SD ~0.20, and ~0.06-0.11
in the evening hours -- so the whole forecast turns on predicting A well.

A is predicted by a small ridge regression on three checkpoint-observable
features, fitted on the donor pool: ln(hour-1 print), the pre-day peak ratio
(ln of the highest hourly count in the 24h before release day, over hour 1),
and the late-ramp slope (log change per hour over the last 6 pre-day hours).
The ln(hour-1) slope is negative -- the scale trap: a huge hour-1 print means
most of the world already unlocked, so the evening plateau sits near hour 1
rather than far above it. The prediction is clamped to the donor amplitude
range (plus a small margin) so an unprecedented hour-1 print cannot
extrapolate catastrophically in either direction.

Donor selection is regime-aware. The target's own notes (and, failing that,
its observable pre-day ramp) decide whether it is a rolling-midnight launch;
rolling targets use only rolling-midnight donors, classified from each
comparable's unlock text with a behavioral fallback (substantial hour-1 print
relative to the day peak plus a rising pre-day ramp). Early-access-contaminated
and simultaneous-global launches are excluded from the rolling pool: their
pre-day features are distorted and their release-day shapes are different
families. Non-rolling targets use all donors.

Spreads are honest: for each donor in the pool, the amplitude + shape fit is
re-run without it, the donor is forecast, and the residual is recorded. Each
hour's five quantiles are the forecast plus that hour's empirical residual
quantiles from those inner leave-one-out runs -- hour-specific and asymmetric,
reflecting what the model actually gets wrong on donors. No spread tuning.

Robustness: any failure in the model path (missing fields, empty donor pool,
singular fits) falls back to the house persistence forecast, so the entry
always writes a valid output file. Quantiles are finite, above 0, at most 1e9,
non-decreasing, and the file is far below the 65,536-byte limit.
"""

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
HOURS = list(range(2, 25))
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
SIGMA = 0.15

AMP_FEATURES = ["lnh1", "prepeak", "slope6"]
AMP_LAM = 1.0


# ---------------------------------------------------------------------------
# small numeric helpers

def solve_linear(mat, vec):
    """Solve mat x = vec by Gaussian elimination with partial pivoting."""
    n = len(vec)
    aug = [row[:] + [vec[i]] for i, row in enumerate(mat)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            raise ValueError("singular matrix")
        aug[col], aug[piv] = aug[piv], aug[col]
        pv = aug[col][col]
        for r in range(n):
            if r != col:
                f = aug[r][col] / pv
                if f:
                    for c in range(col, n + 1):
                        aug[r][c] -= f * aug[col][c]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def ridge_fit(X, y, lam):
    """Ridge regression with unpenalised intercept. X rows include a leading
    1. Falls back to intercept-only on singularity."""
    p = len(X[0])
    xtx = [[0.0] * p for _ in range(p)]
    xty = [0.0] * p
    for row, yi in zip(X, y):
        for i in range(p):
            xty[i] += row[i] * yi
            for j in range(p):
                xtx[i][j] += row[i] * row[j]
    for i in range(1, p):
        xtx[i][i] += lam
    try:
        return solve_linear(xtx, xty)
    except ValueError:
        mean_y = sum(y) / len(y)
        return [mean_y] + [0.0] * (p - 1)


def quantile(sorted_vals, tau):
    """Linear-interpolation quantile."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def round2(x):
    """x rounded to 2 decimals, half up (as JavaScript Math.round rounds it).
    Whole numbers come back as ints."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


# ---------------------------------------------------------------------------
# time helpers

def parse_ts(ts):
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# feature extraction

def extract_curve(points, day_start):
    """Align hourly points to day_start. Returns (day[24], pre[336]) with
    None where no point exists. Minute reads bucket into their hour."""
    by_from = {}
    for p in points or []:
        v = p.get("viewers")
        key = p.get("from") or p.get("at")
        if v is not None and v > 0 and key:
            by_from[key[:13] + ":00:00Z"] = float(v)
    day = []
    for h in range(24):
        key = (day_start + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ")
        day.append(by_from.get(key))
    pre = []
    for h in range(-336, 0):
        key = (day_start + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ")
        pre.append(by_from.get(key))
    return day, pre


def features_from_pre(pre, p1):
    """Checkpoint-observable features from the 336 pre-day hours. Only the
    last 24h are used, so cases with short histories (the final, whose
    recorder started two days before release day) work unchanged."""
    pre24 = [v for v in pre[-24:] if v]
    prepeak = max(pre24) if pre24 else p1

    last6 = [(i, v) for i, v in enumerate(pre[-6:]) if v]
    if len(last6) >= 2:
        slope6 = (math.log(last6[-1][1]) - math.log(last6[0][1])) / max(
            1, last6[-1][0] - last6[0][0])
    else:
        slope6 = 0.0
    last24 = [(i, v) for i, v in enumerate(pre[-24:]) if v]
    if len(last24) >= 2:
        ramp24 = (math.log(last24[-1][1]) - math.log(last24[0][1])) / max(
            1, last24[-1][0] - last24[0][0])
    else:
        ramp24 = 0.0
    return {
        "lnh1": math.log(p1),
        "prepeak": math.log(prepeak / p1),
        "slope6": slope6,
        "ramp24": ramp24,
    }


def target_features(day_start, history_hourly, history, known_viewers):
    pts = history_hourly if history_hourly else history
    _, pre = extract_curve(pts or [], day_start)
    p1 = max(1.0, float(known_viewers))
    return features_from_pre(pre, p1)


# ---------------------------------------------------------------------------
# donor classification

_ROLLING_WORDS = ("midnight local", "rolling", "local midnight",
                  "midnight, rolling", "coD norm", "sony midnight local",
                  "midnight local expected")
_SIMULT_WORDS = ("single global", "simultaneous global", "single simultaneous",
                 "simultaneous across platforms",
                 "standard edition simultaneous")


def classify_donor(comp, day, pre):
    """'rolling', 'simultaneous', or 'other'. Unlock text first, behavioral
    fallback for unverified texts. Early-access-contaminated and weak-tier
    launches are never 'rolling'."""
    text = (comp.get("unlock") or "").lower()
    if "early-access-contaminated" in text or "tier: weak" in text:
        return "other"
    rolling_text = any(w in text for w in _ROLLING_WORDS)
    simult_text = any(w in text for w in _SIMULT_WORDS)
    if rolling_text and not simult_text:
        return "rolling"
    if simult_text and not rolling_text:
        return "simultaneous"
    h1 = day[0] if day and day[0] else None
    peak = max([v for v in day if v], default=None)
    if h1 is None or peak is None:
        return "other"
    pre24 = [v for v in pre[-24:] if v]
    if not pre24:
        return "other"
    last = [(i, v) for i, v in enumerate(pre[-24:]) if v]
    ramp = ((math.log(last[-1][1]) - math.log(last[0][1]))
            / max(1, last[-1][0] - last[0][0])) if len(last) >= 2 else 0.0
    if h1 / peak >= 0.05 and ramp >= 0.08 and max(pre24) / h1 >= 0.5:
        return "rolling"
    return "other"


def target_is_rolling(case, feats):
    """Decide whether the TARGET launch is rolling-midnight, from its own
    case data only: notes text first, behavioral fallback second."""
    notes = ""
    if isinstance(case.get("notes"), str):
        notes = case["notes"].lower()
    elif isinstance(case.get("notes"), dict):
        notes = " ".join(str(v) for v in case["notes"].values()).lower()
    rolling_text = any(w in notes for w in _ROLLING_WORDS)
    simult_text = any(w in notes for w in _SIMULT_WORDS)
    if rolling_text and not simult_text:
        return True
    if simult_text and not rolling_text:
        return False
    if "early-access" in notes or "early access" in notes:
        return False
    return (feats["ramp24"] >= 0.08 and feats["prepeak"] >= -0.7
            and feats["lnh1"] >= 9.5)


def build_donor(comp):
    """Build a donor record from a comparable dict, or None if the
    release-day curve is incomplete."""
    try:
        ds = parse_ts(comp["day_start"])
    except Exception:
        return None
    day, pre = extract_curve(comp.get("hourly") or [], ds)
    if day[0] is None or any(v is None for v in day):
        return None
    p1 = float(day[0])
    ratios = [math.log(float(day[h]) / p1) for h in range(1, 24)]
    feats = features_from_pre(pre, p1)
    amp = sum(ratios[15:21]) / 6.0
    return {
        "name": comp.get("name", "?"),
        "ratios": ratios,
        "amp": amp,
        "feats": feats,
        "class": classify_donor(comp, day, pre),
    }


# ---------------------------------------------------------------------------
# the model

def _standardise(donors, keys):
    mus, sds = {}, {}
    for k in keys:
        vals = [d["feats"][k] for d in donors]
        mu = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals)) or 1.0
        mus[k], sds[k] = mu, sd
    return mus, sds


def fit_amplitude(donors, keys=AMP_FEATURES, lam=AMP_LAM):
    mus, sds = _standardise(donors, keys)
    X = [[1.0] + [(d["feats"][k] - mus[k]) / sds[k] for k in keys]
         for d in donors]
    y = [d["amp"] for d in donors]
    return ridge_fit(X, y, lam), mus, sds


def predict_amplitude(fit, feats, keys=AMP_FEATURES):
    beta, mus, sds = fit
    x = [1.0] + [(feats[k] - mus[k]) / sds[k] for k in keys]
    return sum(b * xi for b, xi in zip(beta, x))


def fit_shape(donors):
    """Per-hour OLS: r_h = a_h + b_h * A."""
    shape = []
    for j in range(23):
        X = [[1.0, d["amp"]] for d in donors]
        y = [d["ratios"][j] for d in donors]
        beta = ridge_fit(X, y, 1e-6)
        shape.append((beta[0], beta[1]))
    return shape


def inner_loo_residuals(donors):
    """For each donor, refit amplitude + shape on the others and record the
    residual vector. Returns 23 lists of residuals (one per hour)."""
    resid = [[] for _ in range(23)]
    for i, d in enumerate(donors):
        others = donors[:i] + donors[i + 1:]
        if len(others) < 5:
            break
        afit = fit_amplitude(others)
        amps = [x["amp"] for x in others]
        a_hat = clamp(predict_amplitude(afit, d["feats"]),
                      min(amps) - 0.2, max(amps) + 0.2)
        shape = fit_shape(others)
        for j in range(23):
            a, b = shape[j]
            resid[j].append(d["ratios"][j] - (a + b * a_hat))
    return resid


def predict_case(case, donors):
    """Full prediction for one case dict. donors: donor records built from
    the case's comparables."""
    ds = parse_ts(case["day_start"])
    known = case["known"][0]
    p1 = max(1.0, float(known["viewers"]))
    feats = target_features(ds, case.get("history_hourly"),
                            case.get("history"), p1)

    if target_is_rolling(case, feats):
        pool = [d for d in donors if d["class"] == "rolling"]
    else:
        pool = list(donors)
    if len(pool) < 8:
        pool = list(donors)
    if not pool:
        return {"hours": [{"hour": h, "q": persistence_quantiles(p1, h)}
                          for h in HOURS]}

    afit = fit_amplitude(pool)
    amps = [d["amp"] for d in pool]
    a_hat = clamp(predict_amplitude(afit, feats),
                  min(amps) - 0.2, max(amps) + 0.2)
    shape = fit_shape(pool)
    resid = inner_loo_residuals(pool)

    hours = []
    for j, h in enumerate(HOURS):
        a, b = shape[j]
        pred = a + b * a_hat
        r = sorted(resid[j]) if resid[j] else [0.0]
        qs = []
        for tau in LEVELS:
            try:
                q = p1 * math.exp(pred + quantile(r, tau))
            except OverflowError:
                q = 1e9
            if not math.isfinite(q):
                q = 1e9
            qs.append(round2(clamp(q, 1.0, 1e9)))
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        hours.append({"hour": h, "q": qs})
    return {"hours": hours}


def persistence_quantiles(p1, h):
    return [round2(clamp(p1 * math.exp(z * SIGMA * math.sqrt(h - 1)),
                         1.0, 1e9)) for z in Z]


def fallback_forecast(case):
    """House persistence forecast, used when the model path fails for any
    reason. Never raises for a dict-like case."""
    p1 = 1.0
    try:
        known = case.get("known") or []
        if known:
            p1 = max(1.0, float(known[0].get("viewers") or 1.0))
        else:
            hh = case.get("history_hourly") or []
            if hh:
                p1 = max(1.0, float(hh[-1].get("viewers") or 1.0))
    except Exception:
        p1 = 1.0
    hours = []
    try:
        wanted = [row["hour"] for row in case.get("forecast") or []]
    except Exception:
        wanted = []
    if sorted(wanted) != HOURS:
        wanted = HOURS
    for h in wanted:
        hours.append({"hour": h, "q": persistence_quantiles(p1, h)})
    return {"hours": hours}


def forecast_case(case):
    try:
        donors = []
        for comp in case.get("comparables") or []:
            try:
                d = build_donor(comp)
            except Exception:
                d = None
            if d is not None:
                donors.append(d)
        result = predict_case(case, donors)
        # Validate shape before accepting: exactly 23 rows, hours 2..24.
        rows = result["hours"]
        if (len(rows) != 23 or sorted(r["hour"] for r in rows) != HOURS
                or any(len(r["q"]) != 5 for r in rows)):
            raise ValueError("bad forecast shape")
        return result
    except Exception:
        return fallback_forecast(case)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            case = {}
        forecast = forecast_case(case) if isinstance(case, dict) else fallback_forecast({})
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
