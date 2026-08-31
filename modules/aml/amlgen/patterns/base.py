"""Shared machinery for injecting episodes into the simulated ecosystem.

An *episode* is the unit of ground truth: a set of transactions belonging to one
coordinated behaviour, with a type, a time window, participating accounts and
their roles. Labelling individual transactions in isolation loses the structure
that makes laundering detectable in the first place.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .. import distributions as D
from ..entities import MULE_POOL, SHELL_POOL, SOURCE_POOL

SECONDS_PER_DAY = 86400
Txn = Tuple[int, int, int, float]  # sender_idx, receiver_idx, epoch_seconds, amount


class SimContext:
    """Everything a pattern needs to place transactions in the world."""

    def __init__(self, cfg, rng, accounts, preferred, ledger, episodes, start_ts, days):
        self.cfg = cfg
        self.rng = rng
        self.accounts = accounts
        self.preferred = preferred
        self.ledger = ledger
        self.episodes = episodes
        self.start_ts = int(start_ts)
        self.days = int(days)
        self.end_ts = self.start_ts + self.days * SECONDS_PER_DAY
        self.d = float(cfg["laundering"]["difficulty"])
        self.threshold = float(cfg["simulation"]["reporting_threshold"])

        self.countries = accounts["country"].to_numpy()
        self.income = accounts["monthly_income"].to_numpy()
        self.median_amt = accounts["baseline_amount_median"].to_numpy()
        self.sigma = accounts["baseline_amount_sigma"].to_numpy()
        self.activity = accounts["baseline_out_per_day"].to_numpy()
        self.business_hours = accounts["business_hours"].to_numpy()
        self.night = accounts["night_ratio"].to_numpy()
        arche = accounts["archetype"].to_numpy()
        self._pools = {name: np.flatnonzero(arche == name) for name in np.unique(arche)}
        self._dormant = np.flatnonzero(accounts["dormant"].to_numpy())
        self.used_as_mule = np.zeros(len(accounts), dtype=bool)
        self._build_network_pools(cfg, rng, accounts)

    def _build_network_pools(self, cfg, rng, accounts):
        """The criminal network is a bounded subset of the population.

        Drawing mules from *everyone* would label a third of the dataset as
        suspicious and destroy the class balance. Bounded pools also reproduce a
        real signal: mules get reused across episodes.
        """
        net = cfg["laundering"].get("network", {})
        dormant_mask = accounts["dormant"].to_numpy()

        retail = self.pool(MULE_POOL)
        n_mule = max(20, int(len(retail) * float(net.get("mule_pool_share", 0.15))))
        # A careless recruiter grabs dormant accounts (cheap, but they light up the
        # moment they move money). A careful one recruits accounts that already
        # transact enough to absorb the flow. `difficulty` slides between the two.
        w = self._activity_weight(retail)
        w = w * np.where(dormant_mask[retail], 4.0 ** (1.0 - self.d), 1.0)
        self.mule_pool = rng.choice(retail, size=min(n_mule, retail.size),
                                    replace=False, p=w / w.sum())

        biz = np.unique(self.pool(SHELL_POOL))
        n_shell = max(10, int(len(biz) * float(net.get("shell_pool_share", 0.12))))
        self.shell_pool = rng.choice(biz, size=min(n_shell, biz.size), replace=False)

        src = np.unique(self.pool(SOURCE_POOL))
        n_src = max(10, int(len(src) * float(net.get("source_pool_share", 0.10))))
        ws = self.income[src]
        self.source_pool = rng.choice(src, size=min(n_src, src.size), replace=False,
                                      p=ws / ws.sum())
        self.dormant_mules = np.intersect1d(self.mule_pool, self._dormant)

    def _activity_weight(self, idx):
        """Interpolates between preferring quiet accounts and preferring busy ones."""
        act = self.activity[idx] + 0.05
        return (1.0 / act) ** (1.0 - self.d) * act ** self.d

    def _from_pool(self, pool, k, exclude=(), bias="quiet", replace=False):
        cand = np.setdiff1d(pool, np.asarray(list(exclude), dtype=int)) if exclude else pool
        if cand.size == 0:
            return np.array([], dtype=int)
        if bias == "quiet":
            w = self._activity_weight(cand)
        elif bias == "income":
            w = self.income[cand]
        else:
            w = np.ones(cand.size)
        k = k if replace else min(k, cand.size)
        return self.rng.choice(cand, size=max(k, 0), replace=replace, p=w / w.sum())

    def mules(self, k, exclude=(), replace=False):
        return [int(x) for x in self._from_pool(self.mule_pool, k, exclude, "quiet", replace)]

    def shells(self, k, exclude=(), replace=False):
        return [int(x) for x in self._from_pool(self.shell_pool, k, exclude, None, replace)]

    def sources(self, k, exclude=(), replace=False):
        return [int(x) for x in self._from_pool(self.source_pool, k, exclude, "income", replace)]

    # ---------------------------------------------------------------- helpers
    def lerp(self, easy, hard):
        """Interpolate a parameter between its blatant and its subtle setting."""
        return easy + (hard - easy) * self.d

    def pool(self, names: Sequence[str]) -> np.ndarray:
        parts = [self._pools[n] for n in names if n in self._pools]
        return np.concatenate(parts) if parts else np.arange(len(self.accounts))

    def sample(self, names, k, exclude=(), bias=None, dormant=False, replace=False):
        """Pick k account indices from the given archetype pools.

        bias: None | 'income' | 'quiet' | 'active'
        """
        cand = self._dormant if dormant else self.pool(names)
        if exclude:
            cand = np.setdiff1d(cand, np.asarray(list(exclude), dtype=cand.dtype))
        if cand.size == 0:
            cand = np.setdiff1d(np.arange(len(self.accounts)), np.asarray(list(exclude)))
        if bias == "income":
            w = self.income[cand]
        elif bias == "quiet":
            w = 1.0 / (self.activity[cand] + 0.05)
        elif bias == "active":
            w = self.activity[cand] + 0.05
        else:
            w = np.ones(cand.size)
        w = w / w.sum()
        k = min(k, cand.size) if not replace else k
        return self.rng.choice(cand, size=k, replace=replace, p=w)

    def window(self, duration_seconds) -> int:
        """Random episode start such that the whole episode fits in the window."""
        span = max(self.end_ts - self.start_ts - int(duration_seconds) - 3600, 3600)
        return self.start_ts + int(self.rng.integers(0, span))

    def plausible_hour(self, ts, idx):
        """Nudge a timestamp toward hours the account normally transacts in.

        At difficulty 0 laundering happens at 3am; at difficulty 1 it hides
        inside the account's own routine.
        """
        if self.rng.random() > self.d:
            return int(ts)
        day0 = (int(ts) // SECONDS_PER_DAY) * SECONDS_PER_DAY
        if self.business_hours[idx]:
            hour = int(self.rng.integers(9, 19))
        else:
            hour = int(self.rng.choice(np.arange(7, 23)))
        return day0 + hour * 3600 + int(self.rng.integers(0, 3600))

    def throughput(self, idx) -> float:
        """Roughly what this account moves in a month under its own baseline."""
        per_txn = self.median_amt[idx] * np.exp(self.sigma[idx] ** 2 / 2)
        return float(max(self.activity[idx] * 30.0 * per_txn, per_txn * 3.0))

    def episode_total(self, idx, scale=1.0, floor=25_000.0, minimum=None,
                      ceiling=8e7) -> float:
        """Total value moved by an episode.

        Anchored on the population-wide amount distribution rather than on the
        source's income, then capped by what the source plausibly moves. This
        keeps laundering amounts *overlapping* with legitimate ones - if they
        sat in their own range, `amount > X` would solve the whole dataset.
        """
        # Careful operators move less per episode and run more of them, so the
        # deviation from each participant's own baseline stays small.
        base = float(D.lognormal_amount(self.rng, 1_200_000 * scale * self.lerp(1.6, 0.4),
                                        1.15, clip_sigma=2.2))
        cap = self.throughput(idx) * self.rng.uniform(0.3, 3.0) * self.lerp(1.5, 0.5)
        total = min(base, cap)
        if minimum is not None:
            total = max(total, float(minimum))
        return float(np.clip(total, floor, ceiling))

    def hop_delay(self, n=1):
        """Seconds between consecutive hops, controlled by difficulty."""
        lo = self.lerp(0.02, 6.0)      # hours
        hi = self.lerp(0.5, 96.0)
        return D.business_hour_offset(self.rng, n, lo, hi)

    def retention(self, n=1):
        """Fraction of funds forwarded at each hop (the rest is the 'cut')."""
        lo = self.lerp(0.97, 0.80)
        hi = self.lerp(0.99, 0.995)
        return self.rng.uniform(lo, hi, n)

    # ---------------------------------------------------------------- commit
    def commit(self, episode_id, pattern, family, is_laundering,
               txns: List[Txn], members: Dict[str, Sequence[int]]):
        if not txns:
            return
        s = np.array([t[0] for t in txns], dtype=np.int32)
        r = np.array([t[1] for t in txns], dtype=np.int32)
        ts = np.clip(np.array([t[2] for t in txns], dtype=np.int64),
                     self.start_ts, self.end_ts - 1)
        amt = D.humanise_amounts(self.rng, np.array([t[3] for t in txns], dtype=float),
                                 p_round=self.lerp(0.55, 0.30))
        keep = (s != r) & (amt > 0)
        s, r, ts, amt = s[keep], r[keep], ts[keep], amt[keep]
        if s.size == 0:
            return
        cb = self.countries[s] != self.countries[r]
        ch = D.pick_channel(self.rng, amt, cb)
        self.ledger.add_bulk(s, r, ts, amt, ch, self.countries[s], self.countries[r],
                             episode_id=episode_id, pattern=pattern,
                             is_laundering=int(is_laundering))
        self.episodes.record(episode_id, pattern, family, is_laundering,
                             int(ts.min()), int(ts.max()), float(amt.sum()),
                             int(s.size), self.d, members)
        if is_laundering:
            for role, group in members.items():
                if role in ("mule", "intermediary", "transit"):
                    self.used_as_mule[np.asarray(list(group), dtype=int)] = True


class Pattern:
    """Base class for every injectable behaviour. Subclasses implement `run`.

    Class attributes (not dataclass fields, so subclasses can simply override):
      name           unique pattern identifier, used as the label value
      family         'laundering' or 'benign_lookalike'
      is_laundering  1 or 0
    """
    name = "pattern"
    family = "laundering"
    is_laundering = 1

    def run(self, ctx: SimContext, episode_id: str) -> None:  # pragma: no cover
        raise NotImplementedError

    def __call__(self, ctx: SimContext, episode_id: str) -> None:
        self.run(ctx, episode_id)
