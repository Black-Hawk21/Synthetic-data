"""Benign episodes that are structurally similar to the laundering patterns.

Without these, a detector achieves high recall by learning shortcuts such as
"fast in-and-out = laundering" or "many small transfers = structuring", and the
score collapses on real data. Every laundering pattern here has a legitimate twin.
"""
from __future__ import annotations

import numpy as np

from .. import distributions as D
from ..entities import MULE_POOL, SHELL_POOL, SOURCE_POOL
from .base import Pattern, SimContext

FAMILY = "benign_lookalike"


class SupplierPassThrough(Pattern):
    """Twin of rapid_pass_through: customer settles, business pays its supplier."""
    name = "supplier_passthrough"
    family = FAMILY
    is_laundering = 0

    def run(self, ctx: SimContext, episode_id: str) -> None:
        biz = int(ctx.sample(["small_business", "merchant"], 1, bias="active")[0])
        n_cust = int(ctx.rng.integers(2, 7))
        customers = [int(x) for x in ctx.sample(["salary", "household", "small_business"],
                                                n_cust, exclude=[biz])]
        suppliers = [int(x) for x in ctx.sample(["large_business", "small_business"],
                                                int(ctx.rng.integers(1, 3)),
                                                exclude=[biz, *customers])]
        ts = ctx.window(10 * 86400)
        txns, pot = [], 0.0
        for cust in customers:
            amt = float(D.lognormal_amount(ctx.rng, ctx.median_amt[biz] * 6, 0.8))
            t = ctx.plausible_hour(ts + int(ctx.rng.random() * 6 * 3600), biz)
            txns.append((cust, biz, t, amt))
            pot += amt
        for sup in suppliers:
            # Same holding-time signature as laundering, entirely legitimate cause.
            t_out = ts + int(D.business_hour_offset(ctx.rng, 1, 0.05, 30.0)[0])
            txns.append((biz, sup, ctx.plausible_hour(t_out, biz),
                         float(pot * ctx.rng.uniform(0.55, 0.95) / len(suppliers))))
        ctx.commit(episode_id, self.name, FAMILY, 0, txns,
                   {"counterparty": customers, "transit": [biz], "beneficiary": suppliers})


class PayrollFanOut(Pattern):
    """Twin of fan_out: an employer pays salaries to many staff on the same day."""
    name = "payroll_fanout"
    family = FAMILY
    is_laundering = 0

    def run(self, ctx: SimContext, episode_id: str) -> None:
        emp = int(ctx.sample(["large_business", "small_business"], 1, bias="income")[0])
        k = int(ctx.rng.integers(25, 90))
        staff = [int(x) for x in ctx.sample(["salary", "freelancer"], k, exclude=[emp])]
        if not staff:
            return
        parts = D.lognormal_amount(ctx.rng, 55_000, 0.65, len(staff), clip_sigma=2.5)
        day = ctx.window(2 * 86400)
        txns = [(emp, s, ctx.plausible_hour(day + int(ctx.rng.integers(9 * 3600, 20 * 3600)), emp),
                 float(p)) for s, p in zip(staff, parts)]
        ctx.commit(episode_id, self.name, FAMILY, 0, txns,
                   {"source": [emp], "beneficiary": staff})


class MarketplaceFanIn(Pattern):
    """Twin of fan_in: a busy merchant collects from many buyers in one day."""
    name = "marketplace_fanin"
    family = FAMILY
    is_laundering = 0

    def run(self, ctx: SimContext, episode_id: str) -> None:
        merch = int(ctx.sample(["merchant"], 1, bias="active")[0])
        k = int(ctx.rng.integers(30, 140))
        buyers = [int(x) for x in ctx.sample(["salary", "student", "household", "freelancer"],
                                             k, exclude=[merch])]
        if not buyers:
            return
        ts = ctx.window(2 * 86400)
        amts = D.lognormal_amount(ctx.rng, ctx.median_amt[merch], 0.9, len(buyers))
        txns = [(b, merch, ctx.plausible_hour(ts + int(ctx.rng.random() * 20 * 3600), b), float(a))
                for b, a in zip(buyers, amts)]
        ctx.commit(episode_id, self.name, FAMILY, 0, txns,
                   {"counterparty": buyers, "collector": [merch]})


class TreasuryCycle(Pattern):
    """Twin of circular_flow: a group sweeps cash between its own entities."""
    name = "treasury_cycle"
    family = FAMILY
    is_laundering = 0

    def run(self, ctx: SimContext, episode_id: str) -> None:
        k = int(ctx.rng.integers(3, 6))
        ring = [int(x) for x in ctx.sample(["large_business", "small_business", "investment"],
                                           k, bias="income")]
        if len(ring) < 3:
            return
        amount = ctx.episode_total(ring[0], scale=1.0)
        ts = ctx.window(len(ring) * 5 * 86400)
        txns = []
        for i in range(len(ring)):
            a, b = ring[i], ring[(i + 1) % len(ring)]
            ts += int(ctx.rng.uniform(0.5, 4.0) * 86400)
            txns.append((a, b, ctx.plausible_hour(ts, b), float(amount)))
            amount *= float(ctx.rng.uniform(0.9, 1.0))
        ctx.commit(episode_id, self.name, FAMILY, 0, txns,
                   {"source": [ring[0]], "counterparty": ring[1:]})


class InstallmentSplit(Pattern):
    """Twin of smurfing: one buyer pays a large invoice in scheduled instalments."""
    name = "installment_split"
    family = FAMILY
    is_laundering = 0

    def run(self, ctx: SimContext, episode_id: str) -> None:
        buyer = int(ctx.sample(["salary", "small_business", "household"], 1, bias="income")[0])
        seller = int(ctx.sample(["merchant", "large_business", "small_business"], 1,
                                exclude=[buyer], bias="active")[0])
        n = int(ctx.rng.integers(4, 13))
        # Deliberately lands in the same amount band as smurfing.
        per = ctx.threshold * ctx.rng.uniform(0.35, 1.05)
        total = per * n
        ts = ctx.window(n * 8 * 86400)
        txns = []
        for i in range(n):
            t = ts + i * int(ctx.rng.uniform(5, 9) * 86400)
            txns.append((buyer, seller, ctx.plausible_hour(t, buyer),
                         float(per * ctx.rng.uniform(0.98, 1.02))))
        ctx.commit(episode_id, self.name, FAMILY, 0, txns,
                   {"source": [buyer], "beneficiary": [seller]})
