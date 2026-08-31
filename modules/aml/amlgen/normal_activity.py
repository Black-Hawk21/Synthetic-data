"""Generates the legitimate baseline: salaries, recurring bills and organic payments.

The baseline is the hard part of a laundering dataset. If the normal population
is too clean, any injected pattern is trivially separable and your detector's
scores are meaningless.
"""
from __future__ import annotations

import numpy as np

from . import distributions as D

SECONDS_PER_DAY = 86400
_HOUR_BINS = 16


def _hour_cdf_table() -> np.ndarray:
    """Pre-compute hour distributions on a night-ratio grid (avoids per-account work)."""
    table = np.zeros((2, _HOUR_BINS, 24))
    for b, business in enumerate([False, True]):
        for j in range(_HOUR_BINS):
            nr = 0.6 * j / (_HOUR_BINS - 1)
            table[b, j] = np.cumsum(D.hour_weights(nr, business_hours=business))
    return table


_HOUR_CDF = _hour_cdf_table()


def generate_normal_activity(cfg, rng, accounts, preferred, ledger, start_ts, days):
    _generate_salaries(cfg, rng, accounts, ledger, start_ts, days)
    _generate_organic(cfg, rng, accounts, preferred, ledger, start_ts, days)


def _generate_salaries(cfg, rng, accounts, ledger, start_ts, days):
    if not cfg["normal_activity"].get("salary_enabled", True):
        return
    salaried = accounts.index[accounts["employer_idx"] >= 0].to_numpy()
    if salaried.size == 0:
        return
    employers = accounts["employer_idx"].to_numpy()[salaried].astype(np.int32)
    base = accounts["salary_amount"].to_numpy()[salaried]
    countries = accounts["country"].to_numpy()
    n_months = max(days // 30, 1)

    senders, receivers, stamps, amounts = [], [], [], []
    for m in range(n_months):
        day = accounts["salary_day"].to_numpy()[salaried] + m * 30
        keep = day < days
        if not keep.any():
            continue
        amt = base[keep] * rng.uniform(0.97, 1.03, int(keep.sum()))
        ts = (start_ts + day[keep] * SECONDS_PER_DAY
              + rng.integers(9 * 3600, 20 * 3600, int(keep.sum())))
        senders.append(employers[keep])
        receivers.append(salaried[keep])
        stamps.append(ts)
        amounts.append(amt)

    if not senders:
        return
    s = np.concatenate(senders); r = np.concatenate(receivers)
    ts = np.concatenate(stamps); amt = D.humanise_amounts(rng, np.concatenate(amounts), 0.55)
    ledger.add_bulk(s, r, ts, amt, np.full(len(s), "NEFT", dtype=object),
                    countries[s], countries[r], pattern="salary", is_laundering=0)


def _generate_organic(cfg, rng, accounts, preferred, ledger, start_ts, days):
    n = len(accounts)
    p_pref = float(cfg["normal_activity"]["p_preferred_counterparty"])
    rate = accounts["baseline_out_per_day"].to_numpy()
    median = accounts["baseline_amount_median"].to_numpy()
    sigma = accounts["baseline_amount_sigma"].to_numpy()
    night = accounts["night_ratio"].to_numpy()
    biz = accounts["business_hours"].to_numpy()
    countries = accounts["country"].to_numpy()

    pop_cdf = np.cumsum(accounts["popularity"].to_numpy())
    pop_cdf /= pop_cdf[-1]
    day_w = D.weekday_weights(start_ts, days)
    day_cdf = np.cumsum(day_w) / day_w.sum()

    night_bin = np.clip((night / 0.6 * (_HOUR_BINS - 1)).round().astype(int), 0, _HOUR_BINS - 1)
    counts = rng.poisson(np.maximum(rate * days, 0.0))

    for i in range(n):
        k = int(counts[i])
        if k == 0:
            continue
        # counterparties
        pool = preferred[i]
        use_pref = rng.random(k) < p_pref
        rec = np.empty(k, dtype=np.int32)
        n_pref = int(use_pref.sum())
        if n_pref:
            rec[use_pref] = pool[rng.integers(0, pool.size, n_pref)]
        n_rand = k - n_pref
        if n_rand:
            rec[~use_pref] = np.searchsorted(pop_cdf, rng.random(n_rand)).astype(np.int32)
        rec[rec == i] = (i + 1) % n

        # timing
        hcdf = _HOUR_CDF[int(biz[i]), night_bin[i]]
        day_idx = np.searchsorted(day_cdf, rng.random(k))
        hour = np.searchsorted(hcdf, rng.random(k))
        ts = (start_ts + day_idx * SECONDS_PER_DAY + hour * 3600
              + rng.integers(0, 3600, k))

        amt = D.humanise_amounts(rng, D.lognormal_amount(rng, median[i], sigma[i], k))
        cb = countries[i] != countries[rec]
        ch = D.pick_channel(rng, amt, cb)
        ledger.add_bulk(np.full(k, i, dtype=np.int32), rec, ts, amt, ch,
                        np.full(k, countries[i], dtype=object), countries[rec],
                        pattern="normal", is_laundering=0)
