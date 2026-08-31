"""Scoring that is actually useful for AML, plus the Red-Team feedback report.

Global accuracy hides everything that matters here. What matters is recall
*per laundering typology* at a realistic alert budget, and which episodes were
missed entirely - those are the inputs to the next generator iteration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)


def alert_threshold(scores: np.ndarray, alert_rate: float = 0.02) -> float:
    """Investigators can only review so many accounts - score at that budget."""
    return float(np.quantile(scores, 1.0 - alert_rate))


def account_report(features: pd.DataFrame, scores: np.ndarray,
                   alert_rate: float = 0.02) -> dict:
    y = features["is_laundering"].to_numpy().astype(int)
    thr = alert_threshold(scores, alert_rate)
    pred = (scores >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    lookalike_fp = int(((pred == 1) & (y == 0)
                        & (features.get("in_benign_lookalike", 0) == 1)).sum())
    return {
        "n_accounts": int(len(y)),
        "n_laundering": int(y.sum()),
        "alert_rate": alert_rate,
        "alerts": int(pred.sum()),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / max(precision + recall, 1e-9), 4),
        "pr_auc": round(float(average_precision_score(y, scores)), 4),
        "roc_auc": round(float(roc_auc_score(y, scores)), 4),
        "false_positives": fp,
        "false_positives_on_benign_lookalikes": lookalike_fp,
        "lookalike_share_of_fp": round(lookalike_fp / max(fp, 1), 4),
    }


def recall_by_pattern(features: pd.DataFrame, scores: np.ndarray,
                      alert_rate: float = 0.02) -> pd.DataFrame:
    """Per-typology recall. An average hides a typology you cannot see at all."""
    thr = alert_threshold(scores, alert_rate)
    df = features[["account_id", "is_laundering", "laundering_patterns"]].copy()
    df["alerted"] = (scores >= thr).astype(int)
    pos = df[df["is_laundering"] == 1].copy()
    if pos.empty:
        return pd.DataFrame(columns=["pattern", "n_accounts", "recall"])
    pos["pattern"] = pos["laundering_patterns"].astype(str).str.split("|")
    exploded = pos.explode("pattern")
    out = (exploded.groupby("pattern")
           .agg(n_accounts=("alerted", "size"), recall=("alerted", "mean"))
           .reset_index().sort_values("recall"))
    out["recall"] = out["recall"].round(4)
    return out


def episode_detection(episodes: pd.DataFrame, members: pd.DataFrame,
                      features: pd.DataFrame, scores: np.ndarray,
                      alert_rate: float = 0.02) -> pd.DataFrame:
    """An episode counts as caught if any participating account is alerted.

    This is closer to how an AML team actually works: one good alert opens the
    whole network.
    """
    thr = alert_threshold(scores, alert_rate)
    alerted = set(features.loc[scores >= thr, "account_id"].astype(str))
    laundering = episodes[episodes["is_laundering"] == 1]
    mem = members[members["episode_id"].isin(set(laundering["episode_id"]))]
    hit = (mem.assign(a=mem["account_id"].astype(str).isin(alerted))
              .groupby("episode_id")["a"].max())
    df = laundering.merge(hit.rename("detected"), on="episode_id", how="left")
    df["detected"] = df["detected"].fillna(False).astype(int)
    return (df.groupby("pattern")
              .agg(n_episodes=("detected", "size"),
                   episode_recall=("detected", "mean"),
                   median_amount=("total_amount", "median"))
              .round(4).reset_index().sort_values("episode_recall"))


def missed_episodes(episodes: pd.DataFrame, members: pd.DataFrame,
                    features: pd.DataFrame, scores: np.ndarray,
                    alert_rate: float = 0.02, top_n: int = 25) -> pd.DataFrame:
    """The false negatives the Red Team should study before the next iteration."""
    thr = alert_threshold(scores, alert_rate)
    alerted = set(features.loc[scores >= thr, "account_id"].astype(str))
    laundering = episodes[episodes["is_laundering"] == 1]
    mem = members[members["episode_id"].isin(set(laundering["episode_id"]))]
    hit = (mem.assign(a=mem["account_id"].astype(str).isin(alerted))
              .groupby("episode_id")["a"].max())
    missed = hit[~hit.astype(bool)].index
    cols = ["episode_id", "pattern", "total_amount", "n_transactions",
            "n_accounts", "duration_hours"]
    return (laundering[laundering["episode_id"].isin(missed)][cols]
            .sort_values("total_amount", ascending=False).head(top_n))


def pr_curve(features: pd.DataFrame, scores: np.ndarray):
    y = features["is_laundering"].to_numpy().astype(int)
    precision, recall, thresholds = precision_recall_curve(y, scores)
    return pd.DataFrame({"precision": precision[:-1], "recall": recall[:-1],
                         "threshold": thresholds})


def print_report(report: dict, by_pattern: pd.DataFrame,
                 by_episode: pd.DataFrame | None = None) -> None:
    print("\n--- account-level ---")
    for k, v in report.items():
        print(f"  {k:38s} {v}")
    print("\n--- recall by laundering typology (account level) ---")
    print(by_pattern.to_string(index=False))
    if by_episode is not None and len(by_episode):
        print("\n--- episode-level detection ---")
        print(by_episode.to_string(index=False))
