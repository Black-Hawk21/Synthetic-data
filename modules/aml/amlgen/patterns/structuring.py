"""Threshold-avoidance and dispersal patterns: smurfing, fan-out, fan-in."""
from __future__ import annotations

import numpy as np

from .. import distributions as D
from ..entities import MULE_POOL, SHELL_POOL, SOURCE_POOL
from .base import Pattern, SimContext


class Smurfing(Pattern):
    """One large sum broken into many transfers kept under the reporting threshold."""
    name = "smurfing"

    def run(self, ctx: SimContext, episode_id: str) -> None:
        collector = ctx.shells(1)
        if not collector:
            return
        collector = collector[0]
        n_smurfs = max(1, int(round(ctx.lerp(1, 6))) + int(ctx.rng.integers(0, 3)))
        smurfs = ctx.mules(n_smurfs, exclude=[collector])
        if not smurfs:
            return
        total = ctx.episode_total(collector, scale=4.0, minimum=ctx.threshold * 10)
        # Blatant: amounts hug the threshold and land within a single day.
        # Subtle: amounts sit far below it and spread across weeks and senders.
        # Each operator picks their own comfort distance below the threshold.
        center = float(ctx.lerp(0.88, 0.18) * ctx.rng.uniform(0.75, 1.25))
        parts = D.structured_amounts(ctx.rng, total, ctx.threshold,
                                     center_frac=center,
                                     sigma=float(ctx.lerp(0.12, 0.45)))
        span_h = ctx.lerp(6, 24 * 21)
        ts0 = ctx.window(int(span_h * 3600))
        txns = []
        for p in parts:
            sender = int(ctx.rng.choice(smurfs))
            t = ts0 + int(ctx.rng.random() * span_h * 3600)
            txns.append((sender, collector, ctx.plausible_hour(t, sender), float(p)))
        ctx.commit(episode_id, self.name, "laundering", 1, txns,
                   {"mule": smurfs, "collector": [collector]})


class FanOut(Pattern):
    """One account disperses value to many receivers."""
    name = "fan_out"

    def run(self, ctx: SimContext, episode_id: str) -> None:
        source = ctx.sources(1) or ctx.shells(1)
        if not source:
            return
        source = source[0]
        k = int(round(ctx.lerp(60, 12))) + int(ctx.rng.integers(0, 6))
        receivers = ctx.mules(k, exclude=[source])
        if not receivers:
            return
        total = ctx.episode_total(source, scale=1.2, minimum=ctx.threshold * 0.3)
        parts = D.split_amount(ctx.rng, total, len(receivers), evenness=ctx.lerp(0.97, 0.15))
        span_h = ctx.lerp(1.5, 24 * 10)
        ts0 = ctx.window(int(span_h * 3600))
        txns = []
        for rcv, p in zip(receivers, parts):
            t = ts0 + int(ctx.rng.random() * span_h * 3600)
            txns.append((source, rcv, ctx.plausible_hour(t, source), float(p)))
        ctx.commit(episode_id, self.name, "laundering", 1, txns,
                   {"source": [source], "mule": receivers})


class FanIn(Pattern):
    """Many accounts funnel value into one collector."""
    name = "fan_in"

    def run(self, ctx: SimContext, episode_id: str) -> None:
        collector = ctx.shells(1)
        if not collector:
            return
        collector = collector[0]
        k = int(round(ctx.lerp(50, 10))) + int(ctx.rng.integers(0, 6))
        senders = ctx.mules(k, exclude=[collector])
        if not senders:
            return
        total = ctx.episode_total(collector, scale=1.2, minimum=ctx.threshold * 0.3)
        parts = D.split_amount(ctx.rng, total, len(senders), evenness=ctx.lerp(0.97, 0.15))
        span_h = ctx.lerp(2.0, 24 * 12)
        ts0 = ctx.window(int(span_h * 3600))
        txns = []
        for snd, p in zip(senders, parts):
            t = ts0 + int(ctx.rng.random() * span_h * 3600)
            txns.append((snd, collector, ctx.plausible_hour(t, snd), float(p)))
        # Subtle variants sweep the collected balance onward instead of parking it.
        if ctx.rng.random() < ctx.d:
            picked = ctx.shells(1, exclude=[collector, *senders])
            dest = picked[0] if picked else collector
            t_out = ts0 + int(span_h * 3600) + int(ctx.hop_delay(1)[0])
            txns.append((collector, dest, t_out, float(total * ctx.retention(1)[0])))
        ctx.commit(episode_id, self.name, "laundering", 1, txns,
                   {"mule": senders, "collector": [collector]})
