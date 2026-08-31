"""Account-level behavioural features derived from the transaction table.

These are the features that let a model learn *money-movement behaviour* rather
than "big number = bad". Everything is computed from the transaction table only,
so the same code runs on real data with the same schema.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import epoch_seconds

HOUR = 3600
DAY = 86400


def _events(txns: pd.DataFrame) -> pd.DataFrame:
    """One row per (account, transaction) - each transaction appears twice."""
    ts = epoch_seconds(txns["timestamp"])
    out = pd.DataFrame({
        "account": txns["sender"].astype(str),
        "counterparty": txns["receiver"].astype(str),
        "ts": ts, "amount": txns["amount"].to_numpy(), "direction": -1,
    })
    inn = pd.DataFrame({
        "account": txns["receiver"].astype(str),
        "counterparty": txns["sender"].astype(str),
        "ts": ts, "amount": txns["amount"].to_numpy(), "direction": 1,
    })
    ev = pd.concat([out, inn], ignore_index=True)
    ev["account"] = ev["account"].astype("category")
    return ev.sort_values(["account", "ts"], kind="mergesort").reset_index(drop=True)


def _flow_features(ev: pd.DataFrame) -> pd.DataFrame:
    inn = ev[ev["direction"] == 1]
    out = ev[ev["direction"] == -1]
    f = pd.DataFrame(index=ev["account"].cat.categories)
    f.index.name = "account_id"

    for tag, side in (("in", inn), ("out", out)):
        g = side.groupby("account", observed=True)["amount"]
        f[f"n_{tag}"] = g.size()
        f[f"amt_{tag}_total"] = g.sum()
        f[f"amt_{tag}_mean"] = g.mean()
        f[f"amt_{tag}_median"] = g.median()
        f[f"amt_{tag}_max"] = g.max()
        f[f"amt_{tag}_std"] = g.std()
    f = f.fillna(0.0)

    f["n_txns"] = f["n_in"] + f["n_out"]
    f["amt_total"] = f["amt_in_total"] + f["amt_out_total"]
    f["inflow_outflow_ratio"] = f["amt_in_total"] / (f["amt_out_total"] + 1.0)
    f["net_flow"] = f["amt_in_total"] - f["amt_out_total"]
    # A transit account retains almost nothing of what it receives.
    f["retention_ratio"] = f["net_flow"] / (f["amt_in_total"] + 1.0)
    f["in_out_txn_ratio"] = f["n_in"] / (f["n_out"] + 1.0)
    f["amt_cv_out"] = f["amt_out_std"] / (f["amt_out_mean"] + 1.0)
    return f


def _counterparty_features(ev: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    for tag, d in (("senders", 1), ("receivers", -1)):
        side = ev[ev["direction"] == d]
        g = side.groupby("account", observed=True)["counterparty"]
        f[f"unique_{tag}"] = g.nunique().reindex(f.index).fillna(0)

    # Herfindahl concentration: 1.0 = a single counterparty, ~0 = fully dispersed.
    pair = (ev.groupby(["account", "counterparty"], observed=True)["amount"]
              .sum().reset_index())
    tot = pair.groupby("account", observed=True)["amount"].transform("sum")
    pair["share2"] = (pair["amount"] / tot) ** 2
    f["counterparty_hhi"] = (pair.groupby("account", observed=True)["share2"].sum()
                             .reindex(f.index).fillna(0))
    cnt = ev.groupby(["account", "counterparty"], observed=True).size().reset_index(name="k")
    f["one_shot_counterparty_ratio"] = (
        cnt.assign(one=(cnt["k"] == 1).astype(float))
           .groupby("account", observed=True)["one"].mean().reindex(f.index).fillna(0))
    f["counterparty_per_txn"] = (f["unique_senders"] + f["unique_receivers"]) / (f["n_txns"] + 1)
    f["degree_ratio"] = f["unique_senders"] / (f["unique_receivers"] + 1.0)
    return f


def _velocity_features(ev: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    """Max transactions/value inside rolling 1h and 24h windows, per account.

    Groups are separated in a shifted timeline so one vectorised searchsorted
    handles all accounts at once instead of a Python loop over 10k groups.
    """
    codes = ev["account"].cat.codes.to_numpy()
    ts = ev["ts"].to_numpy()
    amt = ev["amount"].to_numpy()
    offset = (ts.max() - ts.min() + 10 * DAY)
    shifted = codes.astype(np.int64) * offset + ts
    idx_all = np.arange(len(ts))
    cum = np.concatenate([[0.0], np.cumsum(amt)])

    for label, window in (("1h", HOUR), ("24h", DAY)):
        start = np.searchsorted(shifted, shifted - window, side="left")
        counts = idx_all - start + 1
        sums = cum[idx_all + 1] - cum[start]
        f[f"max_txns_{label}"] = (pd.Series(counts, index=ev["account"].to_numpy())
                                  .groupby(level=0, observed=True).max().reindex(f.index).fillna(0))
        f[f"max_amount_{label}"] = (pd.Series(sums, index=ev["account"].to_numpy())
                                    .groupby(level=0, observed=True).max().reindex(f.index).fillna(0))

    gap = np.diff(ts)
    same = np.diff(codes) == 0
    gaps = pd.DataFrame({"account": ev["account"].to_numpy()[1:][same],
                         "gap": gap[same].astype(float)})
    g = gaps.groupby("account", observed=True)["gap"]
    mean_gap, std_gap = g.mean().reindex(f.index), g.std().reindex(f.index)
    f["mean_inter_txn_seconds"] = mean_gap.fillna(0)
    # Burstiness in [-1, 1]: 1 = tightly clustered bursts, <0 = regular metronome.
    f["burstiness"] = ((std_gap - mean_gap) / (std_gap + mean_gap + 1e-9)).fillna(0)
    f["min_inter_txn_seconds"] = g.min().reindex(f.index).fillna(0)
    return f


def _temporal_features(ev: pd.DataFrame, f: pd.DataFrame, threshold: float) -> pd.DataFrame:
    hour = (ev["ts"].to_numpy() // HOUR) % 24
    dow = ((ev["ts"].to_numpy() // DAY) + 4) % 7  # 1970-01-01 was a Thursday
    amt = ev["amount"].to_numpy()
    acc = ev["account"].to_numpy()
    aux = pd.DataFrame({
        "account": acc,
        "night": ((hour < 6) | (hour >= 23)).astype(float),
        "weekend": (dow >= 5).astype(float),
        "round_1k": (np.mod(amt, 1000) == 0).astype(float),
        "near_threshold": ((amt >= 0.70 * threshold) & (amt < threshold)).astype(float),
        "above_threshold": (amt >= threshold).astype(float),
    })
    g = aux.groupby("account", observed=True).mean(numeric_only=True).reindex(f.index).fillna(0)
    f["night_ratio"] = g["night"]
    f["weekend_ratio"] = g["weekend"]
    f["round_amount_ratio"] = g["round_1k"]
    f["near_threshold_ratio"] = g["near_threshold"]
    f["above_threshold_ratio"] = g["above_threshold"]
    span = ev.groupby("account", observed=True)["ts"].agg(["min", "max"])
    f["active_span_days"] = ((span["max"] - span["min"]) / DAY).reindex(f.index).fillna(0)
    f["txns_per_active_day"] = f["n_txns"] / (f["active_span_days"] + 1.0)
    return f


def _passthrough_features(ev: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    """How fast money leaves after it arrives - the core layering signal."""
    inn = ev[ev["direction"] == 1][["account", "ts", "amount"]].rename(
        columns={"ts": "in_ts", "amount": "in_amount"}).sort_values("in_ts")
    out = ev[ev["direction"] == -1][["account", "ts", "amount"]].sort_values("ts")
    if inn.empty or out.empty:
        for c in ["median_holding_seconds", "min_holding_seconds",
                  "outflow_within_1h_ratio", "outflow_within_24h_ratio", "passthrough_ratio"]:
            f[c] = 0.0
        return f
    inn["account"] = inn["account"].astype(str)
    out["account"] = out["account"].astype(str)
    m = pd.merge_asof(out, inn, left_on="ts", right_on="in_ts", by="account",
                      direction="backward")
    m = m.dropna(subset=["in_ts"])
    m["hold"] = m["ts"] - m["in_ts"]
    g = m.groupby("account", observed=True)
    f["median_holding_seconds"] = g["hold"].median().reindex(f.index).fillna(-1)
    f["min_holding_seconds"] = g["hold"].min().reindex(f.index).fillna(-1)
    f["outflow_within_1h_ratio"] = (m.assign(x=(m["hold"] <= HOUR).astype(float))
                                    .groupby("account", observed=True)["x"].mean()
                                    .reindex(f.index).fillna(0))
    f["outflow_within_24h_ratio"] = (m.assign(x=(m["hold"] <= DAY).astype(float))
                                     .groupby("account", observed=True)["x"].mean()
                                     .reindex(f.index).fillna(0))
    # How closely each outflow mirrors the inflow that preceded it.
    m["mirror"] = np.minimum(m["amount"] / (m["in_amount"] + 1.0), 5.0)
    f["passthrough_ratio"] = g["mirror"].median().reindex(f.index).fillna(0)
    return f


def _deviation_features(ev: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    """Behaviour against the account's own history, not against the population."""
    lo, hi = ev["ts"].min(), ev["ts"].max()
    mid = lo + (hi - lo) / 2
    first = ev[ev["ts"] < mid].groupby("account", observed=True)["amount"].agg(["size", "sum"])
    second = ev[ev["ts"] >= mid].groupby("account", observed=True)["amount"].agg(["size", "sum"])
    first = first.reindex(f.index).fillna(0)
    second = second.reindex(f.index).fillna(0)
    f["volume_growth_ratio"] = (second["sum"] + 1.0) / (first["sum"] + 1.0)
    f["count_growth_ratio"] = (second["size"] + 1.0) / (first["size"] + 1.0)

    cp_first = ev[ev["ts"] < mid].groupby("account", observed=True)["counterparty"].nunique()
    cp_second = ev[ev["ts"] >= mid].groupby("account", observed=True)["counterparty"].nunique()
    f["counterparty_growth_ratio"] = ((cp_second.reindex(f.index).fillna(0) + 1.0)
                                      / (cp_first.reindex(f.index).fillna(0) + 1.0))
    # A dormant account waking up shows a huge idle gap followed by dense activity.
    f["dormancy_wakeup_score"] = np.log1p(f["max_txns_24h"]) * np.log1p(
        f["mean_inter_txn_seconds"] / DAY)
    return f


def build_account_features(txns: pd.DataFrame, threshold: float = 200_000.0) -> pd.DataFrame:
    ev = _events(txns)
    f = _flow_features(ev)
    f = _counterparty_features(ev, f)
    f = _velocity_features(ev, f)
    f = _temporal_features(ev, f, threshold)
    f = _passthrough_features(ev, f)
    f = _deviation_features(ev, f)
    return f.replace([np.inf, -np.inf], 0.0).fillna(0.0).reset_index()
