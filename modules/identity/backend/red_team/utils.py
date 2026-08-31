"""Shared helpers for attack `mutate()` implementations."""
from __future__ import annotations

import uuid

import numpy as np
import pandas as pd


def blend(easy: float, hard: float, difficulty: float) -> float:
    """Interpolate a target statistic between its "easy to catch" value (low
    difficulty) and its "hard to catch / near-legit" value (high difficulty)."""
    return easy + (hard - easy) * difficulty


def perturb_toward(
    rng: np.random.Generator, series: pd.Series, target_easy: float,
    target_hard: float, difficulty: float, noise: float = 0.06,
) -> pd.Series:
    """Push a [0,1]-bounded feature toward a difficulty-dependent target,
    with per-row noise so the attack population still overlaps the
    legitimate one (section 13)."""
    target = blend(target_easy, target_hard, difficulty)
    n = len(series)
    vals = rng.normal(target, noise + 0.02 * (1 - difficulty), size=n)
    return pd.Series(np.clip(vals, 0, 1), index=series.index)


def shrink_days_toward(
    rng: np.random.Generator, n: int, target_easy_days: float,
    target_hard_days: float, difficulty: float,
) -> np.ndarray:
    target = blend(target_easy_days, target_hard_days, difficulty)
    vals = rng.lognormal(mean=np.log(max(target, 1)), sigma=0.4, size=n)
    return np.clip(vals, 0, 365 * 20).round().astype(int)


def shared_pool(rng: np.random.Generator, n: int, cluster_size_range: tuple[int, int], prefix: str) -> np.ndarray:
    """Assign `n` rows into clusters of `cluster_size_range`, each cluster
    sharing one synthetic identifier -- used to simulate infrastructure
    reuse (shared device/IP/phone/email/address) within an attack batch."""
    assigned = []
    i = 0
    while i < n:
        size = int(rng.integers(cluster_size_range[0], cluster_size_range[1] + 1))
        size = min(size, n - i)
        shared_val = f"{prefix}_{uuid.uuid4().hex[:12]}"
        assigned.extend([shared_val] * size)
        i += size
    rng.shuffle(assigned)
    return np.array(assigned[:n])


def tight_timestamps(rng: np.random.Generator, n: int, window_minutes: float) -> list[str]:
    from datetime import datetime, timedelta
    now = datetime.utcnow()
    start = now - timedelta(minutes=window_minutes)
    offsets = rng.uniform(0, window_minutes * 60, size=n)
    return [(start + timedelta(seconds=float(s))).isoformat() for s in sorted(offsets)]
