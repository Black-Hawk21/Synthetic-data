"""Sampling primitives shared by the normal-activity engine and the patterns.

Everything here takes an explicit `rng` (numpy Generator) so runs are reproducible.
"""
from __future__ import annotations

import numpy as np

SECONDS_PER_DAY = 86400

# Channel routing by amount (INR). Order matters: first match wins.
CHANNEL_BANDS = [
    (2_000, ["UPI", "UPI", "UPI", "CARD"]),
    (100_000, ["UPI", "IMPS", "IMPS", "CARD", "NEFT"]),
    (2_000_000, ["NEFT", "IMPS", "NEFT", "RTGS"]),
    (float("inf"), ["RTGS", "RTGS", "NEFT", "WIRE"]),
]


def lognormal_amount(rng, median, sigma, size=None, floor=10.0, clip_sigma=3.0):
    """Log-normal amounts parameterised by median (not mean) and log-sigma.

    Draws are clipped at +/- clip_sigma so a single account cannot emit an
    absurd outlier that dominates every downstream aggregate.
    """
    z = np.clip(rng.standard_normal(size), -clip_sigma, clip_sigma)
    draws = np.asarray(median) * np.exp(np.asarray(sigma) * z)
    return np.maximum(draws, floor)


def humanise_amounts(rng, amounts, p_round=0.35):
    """Humans transfer round numbers a lot of the time. Mimic that.

    Applied to BOTH normal and laundering traffic - if only one side gets
    round numbers the label becomes trivially learnable.
    """
    amounts = np.asarray(amounts, dtype=float)
    out = np.round(amounts, 2)
    mask = rng.random(out.shape) < p_round
    if mask.any():
        step = np.where(out[mask] > 50_000, 1000.0, np.where(out[mask] > 5_000, 100.0, 10.0))
        out[mask] = np.maximum(np.round(out[mask] / step) * step, step)
    return out


def split_amount(rng, total, n_parts, evenness=0.5, floor=100.0):
    """Split `total` into `n_parts` positive parts.

    evenness=1.0 -> nearly equal parts (easy to spot),
    evenness=0.0 -> highly skewed Dirichlet parts (looks organic).
    """
    n_parts = max(int(n_parts), 1)
    alpha = np.full(n_parts, 0.6 + 12.0 * float(evenness))
    parts = rng.dirichlet(alpha) * float(total)
    parts = np.maximum(parts, floor)
    return parts * (float(total) / parts.sum())


def structured_amounts(rng, total, threshold, center_frac=0.80, sigma=0.25,
                       floor=500.0, max_parts=20000):
    """Split `total` into transfers deliberately kept under a reporting threshold.

    Amounts are drawn log-normally around `center_frac * threshold` rather than
    uniformly across a band: a uniform band leaves a rectangular fingerprint in
    the amount histogram that a model can memorise instead of learning the
    behaviour. Callers should also jitter `center_frac` per episode, so that
    different operators sit at different distances below the threshold.
    """
    target = max(threshold * float(center_frac), floor * 2)
    ceiling = threshold * 0.995
    parts, remaining = [], float(total)
    while remaining > floor and len(parts) < max_parts:
        z = np.clip(rng.standard_normal(), -2.5, 2.5)
        draw = min(target * np.exp(sigma * z), ceiling, remaining)
        parts.append(max(draw, floor))
        remaining -= draw
    return np.array(parts) if parts else np.array([float(total)])


def hour_weights(night_ratio, business_hours=False):
    """A 24-slot probability vector for time-of-day."""
    w = np.zeros(24)
    if business_hours:
        w[9:19] = 1.0
        w[7:9] = 0.35
        w[19:22] = 0.30
    else:
        w[7:10] = 0.7
        w[10:13] = 1.0
        w[13:18] = 0.9
        w[18:23] = 1.2
    night = np.zeros(24)
    night[0:6] = 1.0
    night[23] = 0.6
    w = w / w.sum()
    night = night / night.sum()
    nr = float(np.clip(night_ratio, 0.0, 0.6))
    return (1 - nr) * w + nr * night


def sample_timestamps(rng, start_ts, days, size, hour_w, day_w=None):
    """Sample epoch-second timestamps inside the simulation window."""
    if day_w is None:
        day_idx = rng.integers(0, days, size)
    else:
        cdf = np.cumsum(day_w) / np.sum(day_w)
        day_idx = np.searchsorted(cdf, rng.random(size))
    hcdf = np.cumsum(hour_w)
    hour = np.searchsorted(hcdf, rng.random(size) * hcdf[-1])
    sec = rng.integers(0, 3600, size)
    return start_ts + day_idx * SECONDS_PER_DAY + hour * 3600 + sec


def weekday_weights(start_ts, days):
    """Fewer retail transactions on Sundays, a bump on Fri/Sat."""
    dow = ((np.arange(days) + _weekday_of(start_ts)) % 7)
    base = np.array([1.0, 1.0, 1.0, 1.05, 1.2, 1.25, 0.7])  # Mon..Sun
    return base[dow]


def _weekday_of(epoch_seconds):
    import datetime as _dt
    return _dt.datetime.utcfromtimestamp(epoch_seconds).weekday()


def pick_channel(rng, amounts, cross_border=None):
    amounts = np.asarray(amounts, dtype=float)
    out = np.empty(amounts.shape, dtype=object)
    for hi, options in CHANNEL_BANDS:
        mask = amounts < hi
        mask &= out == None  # noqa: E711 - object array null check
        if mask.any():
            out[mask] = rng.choice(options, size=int(mask.sum()))
    out[out == None] = "NEFT"  # noqa: E711
    if cross_border is not None:
        out = np.where(np.asarray(cross_border), "SWIFT", out)
    return out.astype(object)


def business_hour_offset(rng, size, low_h, high_h):
    """A positive time delta in seconds, drawn log-uniformly between two hour bounds."""
    lo = max(float(low_h), 1e-4) * 3600.0
    hi = max(float(high_h) * 3600.0, lo * 1.01)
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size))
