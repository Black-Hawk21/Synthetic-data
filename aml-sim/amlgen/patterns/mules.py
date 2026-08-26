"""Mule-network and dormant-account patterns."""
from __future__ import annotations

import numpy as np

from .. import distributions as D
from ..entities import MULE_POOL, SHELL_POOL, SOURCE_POOL
from .base import Pattern, SimContext


class MuleNetwork(Pattern):
    """Two layers of mules between a source and a beneficiary.

    Each mule looks unremarkable on its own; only the network is suspicious.
    """
    name = "mule_network"

    def run(self, ctx: SimContext, episode_id: str) -> None:
        source = ctx.sources(1)
        if not source:
            return
        source = source[0]
        n1 = int(round(ctx.lerp(8, 3))) + int(ctx.rng.integers(0, 3))
        layer1 = ctx.mules(n1, exclude=[source])
        n2 = max(2, int(round(ctx.lerp(10, 4))))
        layer2 = ctx.mules(n2, exclude=[source, *layer1])
        dest_pick = ctx.shells(1, exclude=[source, *layer1, *layer2])
        if not layer1 or not layer2 or not dest_pick:
            return
        dest = dest_pick[0]

        total = ctx.episode_total(source, scale=1.5)
        first = D.split_amount(ctx.rng, total, len(layer1), evenness=ctx.lerp(0.95, 0.25))
        ts0 = ctx.window(24 * 14 * 3600)
        txns = []
        for mule, amt in zip(layer1, first):
            t1 = ctx.plausible_hour(ts0 + int(ctx.rng.random() * ctx.lerp(2, 72) * 3600), mule)
            txns.append((source, mule, t1, float(amt)))
            fwd = float(amt * ctx.retention(1)[0])
            picks = ctx.rng.choice(layer2, size=min(len(layer2), int(ctx.rng.integers(1, 3))),
                                   replace=False)
            parts = D.split_amount(ctx.rng, fwd, len(picks), evenness=ctx.lerp(0.9, 0.3))
            for m2, p in zip(picks, parts):
                t2 = t1 + int(ctx.hop_delay(1)[0])
                txns.append((mule, int(m2), ctx.plausible_hour(t2, int(m2)), float(p)))
                t3 = t2 + int(ctx.hop_delay(1)[0])
                txns.append((int(m2), dest, ctx.plausible_hour(t3, dest),
                             float(p * ctx.retention(1)[0])))
        ctx.commit(episode_id, self.name, "laundering", 1, txns,
                   {"source": [source], "mule": layer1 + layer2, "beneficiary": [dest]})


class DormantActivation(Pattern):
    """A long-idle account suddenly cycles large sums in and straight back out."""
    name = "dormant_activation"

    def run(self, ctx: SimContext, episode_id: str) -> None:
        pool = ctx.dormant_mules if ctx.dormant_mules.size else ctx.mule_pool
        acct = int(ctx.rng.choice(pool))
        source = ctx.sources(1, exclude=[acct])
        dest = ctx.shells(1, exclude=[acct])
        if not source or not dest:
            return
        source, dest = source[0], dest[0]
        bursts = int(round(ctx.lerp(5, 2))) + int(ctx.rng.integers(0, 3))
        # A blatant activation dumps everything on day one; a careful one ramps up.
        ramp = np.linspace(ctx.lerp(1.0, 0.25), 1.0, bursts)
        base = ctx.episode_total(source, scale=0.6) / max(bursts, 1)
        ts = ctx.window(bursts * 48 * 3600)
        txns = []
        for b in range(bursts):
            amt = base * float(ramp[b]) * float(ctx.rng.uniform(0.8, 1.2))
            t_in = ctx.plausible_hour(ts, acct)
            txns.append((source, acct, t_in, amt))
            t_out = t_in + int(D.business_hour_offset(ctx.rng, 1, ctx.lerp(0.05, 3.0),
                                                      ctx.lerp(6.0, 48.0))[0])
            txns.append((acct, dest, t_out, float(amt * ctx.retention(1)[0])))
            ts = t_in + int(ctx.rng.uniform(0.5, 2.5) * 86400)
        ctx.commit(episode_id, self.name, "laundering", 1, txns,
                   {"source": [source], "transit": [acct], "destination": [dest]})
