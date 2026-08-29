"""Builds the account population and its social/commercial affinity structure."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .entities import ARCHETYPES, CITIES, FOREIGN_COUNTRIES


def build_population(cfg: dict, rng: np.random.Generator, start_ts: int):
    pop = cfg["population"]
    n = int(pop["n_accounts"])
    mix = pop["archetype_mix"]
    names = list(mix.keys())
    probs = np.array([mix[k] for k in names], dtype=float)
    probs /= probs.sum()

    kind = rng.choice(names, size=n, p=probs)

    def rng_range(attr, default=(0.0, 1.0)):
        lo = np.array([getattr(ARCHETYPES[k], attr, default)[0] for k in kind])
        hi = np.array([getattr(ARCHETYPES[k], attr, default)[1] for k in kind])
        return lo + rng.random(n) * (hi - lo)

    out_per_day = rng_range("out_per_day")
    jitter = float(cfg["normal_activity"]["activity_jitter"])
    out_per_day *= np.exp(rng.normal(0.0, jitter, n))

    amount_median = rng_range("amount_median")
    amount_sigma = rng_range("amount_sigma")
    night_ratio = rng_range("night_ratio")
    monthly_income = rng_range("monthly_income")

    is_business = np.array([ARCHETYPES[k].is_business for k in kind])
    business_hours = np.array([ARCHETYPES[k].business_hours for k in kind])
    popularity = np.array([ARCHETYPES[k].popularity for k in kind], dtype=float)
    popularity *= np.exp(rng.normal(0.0, 0.6, n))          # heavy tail of "hub" receivers

    city = rng.choice(CITIES, size=n)
    country = np.full(n, "IN", dtype=object)
    foreign = rng.random(n) < float(pop["foreign_share"])
    country[foreign] = rng.choice(FOREIGN_COUNTRIES, size=int(foreign.sum()))

    # Dormant accounts: opened, KYC'd, then barely used. Prime mule material.
    dormant = rng.random(n) < float(pop["dormant_share"])
    out_per_day = np.where(dormant, out_per_day * 0.03, out_per_day)

    age_days = rng.integers(30, 4000, n)
    open_ts = start_ts - age_days * 86400
    kyc_level = rng.choice(["full", "full", "full", "min"], size=n)
    kyc_level = np.where(dormant & (rng.random(n) < 0.5), "min", kyc_level)

    accounts = pd.DataFrame({
        "account_idx": np.arange(n, dtype=np.int32),
        "account_id": [f"ACC{i:06d}" for i in range(n)],
        "archetype": kind,
        "is_business": is_business,
        "business_hours": business_hours,
        "city": city,
        "country": country,
        "kyc_level": kyc_level,
        "open_ts": open_ts,
        "account_age_days": age_days,
        "dormant": dormant,
        "baseline_out_per_day": out_per_day.round(4),
        "baseline_amount_median": amount_median.round(2),
        "baseline_amount_sigma": amount_sigma.round(3),
        "night_ratio": night_ratio.round(4),
        "monthly_income": monthly_income.round(2),
        "popularity": popularity.round(4),
    })

    employers = _assign_employers(accounts, rng)
    accounts["employer_idx"] = employers
    accounts["salary_amount"] = np.where(
        employers >= 0, (accounts["monthly_income"] * rng.uniform(0.75, 1.0, n)).round(2), 0.0)
    accounts["salary_day"] = np.where(employers >= 0, rng.integers(1, 8, n), 0)

    preferred = _build_affinity(accounts, cfg, rng)
    return accounts, preferred


def _assign_employers(accounts: pd.DataFrame, rng) -> np.ndarray:
    n = len(accounts)
    employers = np.full(n, -1, dtype=np.int64)
    employer_pool = accounts.index[accounts["archetype"].isin(
        ["large_business", "small_business", "merchant"])].to_numpy()
    if employer_pool.size == 0:
        return employers
    salaried = accounts.index[accounts["archetype"] == "salary"].to_numpy()
    weights = np.where(accounts.loc[employer_pool, "archetype"] == "large_business", 6.0, 1.0)
    weights = weights / weights.sum()
    employers[salaried] = rng.choice(employer_pool, size=salaried.size, p=weights)
    return employers


def _build_affinity(accounts: pd.DataFrame, cfg: dict, rng) -> list:
    """Each account gets a stable set of preferred counterparties.

    Preference is popularity-weighted with a same-city bonus, which produces
    community structure in the graph. Real payment networks are not uniform.
    """
    n = len(accounts)
    lo, hi = cfg["normal_activity"]["n_preferred_range"]
    city_bias = float(cfg["normal_activity"]["same_city_bias"])
    pop_w = accounts["popularity"].to_numpy()
    city_codes = pd.factorize(accounts["city"])[0]
    n_cities = city_codes.max() + 1

    global_cdf = np.cumsum(pop_w)
    global_cdf /= global_cdf[-1]

    city_members, city_cdfs = [], []
    for c in range(n_cities):
        idx = np.flatnonzero(city_codes == c)
        w = np.cumsum(pop_w[idx])
        city_members.append(idx)
        city_cdfs.append(w / w[-1])

    k_all = rng.integers(int(lo), int(hi) + 1, n)
    preferred = []
    p_local = city_bias / (city_bias + 1.0)
    for i in range(n):
        k = int(k_all[i])
        local_k = int(rng.binomial(k, p_local))
        c = city_codes[i]
        loc = city_members[c][np.searchsorted(city_cdfs[c], rng.random(local_k))]
        glo = np.searchsorted(global_cdf, rng.random(k - local_k))
        cand = np.unique(np.concatenate([loc, glo]).astype(np.int32))
        cand = cand[cand != i]
        if cand.size == 0:
            cand = np.array([(i + 1) % n], dtype=np.int32)
        preferred.append(cand)
    return preferred
