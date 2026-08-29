"""End-to-end simulation driver: population -> normal traffic -> injected episodes."""
from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .ledger import EpisodeStore, Ledger
from .normal_activity import generate_normal_activity
from .patterns import SimContext, inject_episodes
from .population import build_population


@dataclass
class SimulationResult:
    accounts: pd.DataFrame
    transactions: pd.DataFrame
    episodes: pd.DataFrame
    episode_members: pd.DataFrame
    config: dict


def _to_epoch(date_str: str) -> int:
    return int(dt.datetime.strptime(date_str, "%Y-%m-%d")
               .replace(tzinfo=dt.timezone.utc).timestamp())


def simulate(cfg: dict, verbose: bool = True) -> SimulationResult:
    rng = np.random.default_rng(int(cfg["seed"]))
    start_ts = _to_epoch(cfg["simulation"]["start_date"])
    days = int(cfg["simulation"]["days"])
    t0 = time.time()

    if verbose:
        print(f"[1/4] population: {cfg['population']['n_accounts']} accounts")
    accounts, preferred = build_population(cfg, rng, start_ts)

    ledger, episodes = Ledger(), EpisodeStore()
    if verbose:
        print(f"[2/4] normal activity over {days} days")
    generate_normal_activity(cfg, rng, accounts, preferred, ledger, start_ts, days)
    n_normal = len(ledger)
    if verbose:
        print(f"      {n_normal:,} legitimate transactions")

    if verbose:
        print(f"[3/4] injecting episodes (difficulty={cfg['laundering']['difficulty']})")
    ctx = SimContext(cfg, rng, accounts, preferred, ledger, episodes, start_ts, days)
    inject_episodes(ctx, cfg, verbose=verbose)

    if verbose:
        print("[4/4] assembling tables")
    txns = ledger.to_frame(accounts)
    ep_df, mem_df = episodes.to_frames(accounts)

    accounts = _attach_account_labels(accounts, mem_df, ctx)
    if verbose:
        n_l = int(txns["is_laundering"].sum())
        print(f"      {len(txns):,} transactions total | {n_l:,} laundering "
              f"({100 * n_l / len(txns):.3f}%) | {len(ep_df):,} episodes "
              f"| {time.time() - t0:.1f}s")
    return SimulationResult(accounts, txns, ep_df, mem_df, cfg)


def _attach_account_labels(accounts: pd.DataFrame, members: pd.DataFrame,
                           ctx: SimContext) -> pd.DataFrame:
    """Account-level ground truth derived from episode membership."""
    accounts = accounts.copy()
    accounts["is_laundering"] = 0
    accounts["laundering_role"] = ""
    accounts["laundering_patterns"] = ""
    accounts["n_laundering_episodes"] = 0
    accounts["in_benign_lookalike"] = 0
    if members is None or members.empty:
        return accounts

    idx = pd.Series(accounts.index.values, index=accounts["account_id"].values)
    bad = members[members["is_laundering"] == 1]
    if not bad.empty:
        grp = bad.groupby("account_id")
        roles = grp["role"].agg(lambda s: "|".join(sorted(set(s))))
        pats = grp["pattern"].agg(lambda s: "|".join(sorted(set(s))))
        counts = grp.size()
        pos = idx.reindex(roles.index).to_numpy()
        accounts.loc[pos, "is_laundering"] = 1
        accounts.loc[pos, "laundering_role"] = roles.to_numpy()
        accounts.loc[pos, "laundering_patterns"] = pats.to_numpy()
        accounts.loc[pos, "n_laundering_episodes"] = counts.to_numpy()
    good = members[members["is_laundering"] == 0]
    if not good.empty:
        pos = idx.reindex(good["account_id"].unique()).to_numpy()
        accounts.loc[pos, "in_benign_lookalike"] = 1
    return accounts
