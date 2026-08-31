"""Pattern registry + the injection driver."""
from __future__ import annotations

from typing import Dict

from .base import Pattern, SimContext
from .layering import CircularFlow, LayeringChain, RapidPassThrough
from .lookalikes import (InstallmentSplit, MarketplaceFanIn, PayrollFanOut,
                         SupplierPassThrough, TreasuryCycle)
from .mules import DormantActivation, MuleNetwork
from .structuring import FanIn, FanOut, Smurfing

LAUNDERING: Dict[str, Pattern] = {p.name: p for p in [
    LayeringChain(), CircularFlow(), RapidPassThrough(),
    Smurfing(), FanOut(), FanIn(), MuleNetwork(), DormantActivation(),
]}

BENIGN: Dict[str, Pattern] = {p.name: p for p in [
    SupplierPassThrough(), PayrollFanOut(), MarketplaceFanIn(),
    TreasuryCycle(), InstallmentSplit(),
]}

ALL: Dict[str, Pattern] = {**LAUNDERING, **BENIGN}


def _scale(cfg: dict, n_accounts: int) -> float:
    """Episode counts are written for a reference population size.

    Without this, shrinking the population for a quick test silently multiplies
    the prevalence of laundering and every metric shifts for the wrong reason.
    """
    if not cfg["laundering"].get("scale_with_population", True):
        return 1.0
    ref = float(cfg["laundering"].get("population_reference", 10000))
    return max(n_accounts / ref, 0.02)


def inject_episodes(ctx: SimContext, cfg: dict, verbose: bool = True) -> None:
    """Inject every configured episode, interleaved so no ordering artefact leaks."""
    k = _scale(cfg, len(ctx.accounts))
    plan = []
    for name, count in cfg["laundering"]["episodes"].items():
        if name not in LAUNDERING:
            raise KeyError(f"unknown laundering pattern: {name}")
        plan += [LAUNDERING[name]] * max(int(round(count * k)), 1)
    if cfg["benign_lookalikes"].get("enabled", True):
        for name, count in cfg["benign_lookalikes"]["episodes"].items():
            if name not in BENIGN:
                raise KeyError(f"unknown benign pattern: {name}")
            plan += [BENIGN[name]] * max(int(round(count * k)), 1)

    order = ctx.rng.permutation(len(plan))
    for n, i in enumerate(order):
        pattern = plan[i]
        prefix = "L" if pattern.is_laundering else "B"
        pattern(ctx, ctx.episodes.new_id(prefix))
    if verbose:
        n_laundering = sum(1 for p in plan if p.is_laundering)
        print(f"  injected {len(plan)} episodes "
              f"({n_laundering} laundering, {len(plan) - n_laundering} benign lookalikes)")
