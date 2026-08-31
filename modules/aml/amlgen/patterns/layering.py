"""Placement/layering patterns: chains, cycles and pass-through accounts."""
from __future__ import annotations

import numpy as np

from .. import distributions as D
from ..entities import MULE_POOL, SHELL_POOL, SOURCE_POOL
from .base import Pattern, SimContext


class LayeringChain(Pattern):
    """A -> B -> C -> D: value walks through a chain, shedding a cut at each hop."""
    name = "layering_chain"

    def run(self, ctx: SimContext, episode_id: str) -> None:
        n_hops = int(round(ctx.lerp(6, 2))) + int(ctx.rng.integers(0, 2))
        source = ctx.sources(1)
        if not source:
            return
        source = source[0]
        mids = ctx.mules(n_hops - 1, exclude=[source]) + ctx.shells(1, exclude=[source])
        ctx.rng.shuffle(mids)
        dest = ctx.shells(1, exclude=[source, *mids])
        if not mids or not dest:
            return
        path = [source, *mids, dest[0]]
        dest = dest[0]

        total = ctx.episode_total(source, scale=1.0)
        ts = ctx.window(n_hops * 96 * 3600)
        txns = []
        amount = total
        for a, b in zip(path[:-1], path[1:]):
            ts = int(ts + ctx.hop_delay(1)[0])
            # Under difficulty, a hop is paid out as several smaller transfers.
            n_parts = 1 if ctx.rng.random() > ctx.d else int(ctx.rng.integers(1, 4))
            parts = D.split_amount(ctx.rng, amount, n_parts, evenness=ctx.lerp(0.9, 0.25))
            for j, p in enumerate(parts):
                t = ctx.plausible_hour(ts + j * int(ctx.hop_delay(1)[0] * 0.3), b)
                txns.append((a, b, t, float(p)))
            amount *= float(ctx.retention(1)[0])

        ctx.commit(episode_id, self.name, "laundering", 1, txns,
                   {"source": [source], "intermediary": [int(m) for m in mids],
                    "destination": [dest]})


class CircularFlow(Pattern):
    """A -> B -> C -> A: value returns to origin, often through 'invoices'."""
    name = "circular_flow"

    def run(self, ctx: SimContext, episode_id: str) -> None:
        k = int(ctx.rng.integers(3, 7))
        ring = ctx.shells(k)
        if len(ring) < 3:
            return
        laps = 1 if ctx.rng.random() > 0.35 else 2
        total = ctx.episode_total(ring[0], scale=1.0)
        ts = ctx.window(len(ring) * laps * 96 * 3600)
        txns, amount = [], total
        for _ in range(laps):
            for i in range(len(ring)):
                a, b = ring[i], ring[(i + 1) % len(ring)]
                ts = int(ts + ctx.hop_delay(1)[0])
                txns.append((a, b, ctx.plausible_hour(ts, b), float(amount)))
                amount *= float(ctx.retention(1)[0])
        ctx.commit(episode_id, self.name, "laundering", 1, txns,
                   {"source": [ring[0]], "intermediary": ring[1:]})


class RapidPassThrough(Pattern):
    """An account behaves as a transit node: money in, money straight back out."""
    name = "rapid_pass_through"

    def run(self, ctx: SimContext, episode_id: str) -> None:
        transit = ctx.mules(1)
        if not transit:
            return
        transit = transit[0]
        rounds = int(round(ctx.lerp(6, 3))) + int(ctx.rng.integers(0, 3))
        sources = ctx.sources(rounds, exclude=[transit], replace=True)
        n_dest = max(1, int(round(ctx.lerp(1, 3))))
        dests = ctx.shells(n_dest, exclude=[transit, *sources])
        if not sources or not dests:
            return

        ts = ctx.window(rounds * 72 * 3600)
        txns = []
        for i in range(rounds):
            inflow = ctx.episode_total(sources[i], scale=0.35)
            t_in = ctx.plausible_hour(int(ts), transit)
            txns.append((sources[i], transit, t_in, inflow))
            keep = float(ctx.retention(1)[0])
            parts = D.split_amount(ctx.rng, inflow * keep, len(dests),
                                   evenness=ctx.lerp(0.9, 0.3))
            for dest, part in zip(dests, parts):
                # The signature of this pattern is the holding time, not the amount.
                t_out = t_in + int(D.business_hour_offset(
                    ctx.rng, 1, ctx.lerp(0.002, 2.0), ctx.lerp(0.1, 30.0))[0])
                txns.append((transit, dest, t_out, float(part)))
            ts = t_in + int(ctx.hop_delay(1)[0] * ctx.rng.uniform(1.0, 4.0))
        ctx.commit(episode_id, self.name, "laundering", 1, txns,
                   {"source": sources, "transit": [transit], "destination": dests})
