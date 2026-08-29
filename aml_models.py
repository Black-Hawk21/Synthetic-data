"""Shared feature engineering and model-bundle I/O for the AML detectors.

This module is imported by both `train_models.py` and `predict.py` so that the
features built at inference time are guaranteed to match the ones the model was
trained on. Drop it into the `aml-sim/` directory next to `run.py`.

Two detectors are defined:

  * account-level  — one row per account, built from `account_features.csv`
                     (the repo's own modelling table, leak-safe columns only)
  * transaction-level — one row per transaction, built from `transactions.csv`
                     plus behavioural context of the sender and receiver
                     accounts joined in from `account_features.csv`

Ground-truth columns (`is_laundering`, `episode_id`, `pattern`, roles) are
never used as features — only as labels during training / evaluation.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Leak-safe column handling for the account table
# ---------------------------------------------------------------------------
# Mirrors amlgen.features.build; falls back to a copy so predict.py also works
# outside the repo tree.
try:  # pragma: no cover
    from amlgen.features.build import LABEL_COLUMNS, SIMULATOR_ONLY
except Exception:  # pragma: no cover
    LABEL_COLUMNS = ["is_laundering", "laundering_role", "laundering_patterns",
                     "n_laundering_episodes", "in_benign_lookalike"]
    SIMULATOR_ONLY = ["account_idx", "dormant", "baseline_out_per_day",
                      "baseline_amount_median", "baseline_amount_sigma",
                      "night_ratio", "monthly_income", "popularity",
                      "employer_idx", "salary_amount", "salary_day", "open_ts",
                      "is_business", "business_hours"]

REPORTING_THRESHOLD = 200_000.0  # INR 2 lakh, matches config.yaml

# Compact behavioural context joined onto each transaction for the
# sender and the receiver. Chosen from the strongest signals in the repo's
# own permutation-importance output plus the pattern-specific ratios.
ACCOUNT_CONTEXT_COLS = [
    "amt_out_std", "amt_cv_out", "amt_in_median", "amt_out_median",
    "amt_out_max", "amt_out_total", "n_in", "n_out",
    "inflow_outflow_ratio", "retention_ratio", "passthrough_ratio",
    "counterparty_hhi", "one_shot_counterparty_ratio", "unique_senders",
    "unique_receivers", "max_txns_24h", "burstiness",
    "median_holding_seconds", "outflow_within_24h_ratio",
    "round_amount_ratio", "near_threshold_ratio", "dormancy_wakeup_score",
    "volume_growth_ratio", "g_pagerank", "g_clustering", "g_relay_score",
    "g_fan_out_ratio", "g_in_cycle", "g_scc_size", "g_core_number",
    "account_age_days",
]


# ---------------------------------------------------------------------------
# Account-level features
# ---------------------------------------------------------------------------
def account_feature_matrix(features: pd.DataFrame,
                           columns: list[str] | None = None) -> pd.DataFrame:
    """Leak-safe account design matrix.

    Same drop-list as amlgen's `model_matrix`, but tolerant of label columns
    being absent (real inference) and able to align to a saved column list.
    """
    drop = set(LABEL_COLUMNS) | set(SIMULATOR_ONLY) | {"account_id"}
    X = features.drop(columns=[c for c in drop if c in features.columns])
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    X = pd.get_dummies(X, columns=cat, drop_first=True)
    if columns is not None:                      # align to training schema
        X = X.reindex(columns=columns, fill_value=0.0)
    return X.astype(np.float32)


# ---------------------------------------------------------------------------
# Transaction-level features
# ---------------------------------------------------------------------------
def transaction_feature_matrix(txns: pd.DataFrame, features: pd.DataFrame,
                               columns: list[str] | None = None) -> pd.DataFrame:
    """One row per transaction: intrinsic fields + sender/receiver context.

    Uses only observable columns — never `episode_id`, `pattern` or
    `is_laundering`.
    """
    ts = pd.to_datetime(txns["timestamp"])
    amount = txns["amount"].astype(np.float32)

    X = pd.DataFrame(index=txns.index)
    X["amount"] = amount
    X["log_amount"] = np.log1p(amount)
    X["hour"] = ts.dt.hour.astype(np.int8)
    X["dow"] = ts.dt.dayofweek.astype(np.int8)
    X["is_night"] = ((ts.dt.hour < 6) | (ts.dt.hour >= 23)).astype(np.int8)
    X["is_weekend"] = (ts.dt.dayofweek >= 5).astype(np.int8)
    X["is_round_amount"] = (amount % 1000 == 0).astype(np.int8)
    X["near_threshold"] = ((amount >= 0.8 * REPORTING_THRESHOLD)
                           & (amount < REPORTING_THRESHOLD)).astype(np.int8)
    X["above_threshold"] = (amount >= REPORTING_THRESHOLD).astype(np.int8)
    X["cross_border"] = txns["cross_border"].astype(np.int8)

    X = pd.concat([X, pd.get_dummies(txns["channel"], prefix="ch")], axis=1)

    ctx_cols = [c for c in ACCOUNT_CONTEXT_COLS if c in features.columns]
    ctx = features.set_index("account_id")[ctx_cols].astype(np.float32)
    snd = ctx.reindex(txns["sender"].values).add_prefix("snd_")
    rcv = ctx.reindex(txns["receiver"].values).add_prefix("rcv_")
    snd.index = X.index
    rcv.index = X.index

    X = pd.concat([X, snd, rcv], axis=1).fillna(0.0)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X.astype(np.float32)


# ---------------------------------------------------------------------------
# Bundle I/O
# ---------------------------------------------------------------------------
def save_bundle(path: str | Path, model, columns: list[str],
                threshold: float, meta: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "columns": columns,
                 "threshold": float(threshold), "meta": meta or {}}, path)


def load_bundle(path: str | Path) -> dict:
    return joblib.load(path)


def threshold_at_alert_rate(scores: np.ndarray, alert_rate: float) -> float:
    """Score cutoff that flags the top `alert_rate` fraction (analyst budget)."""
    alert_rate = min(max(alert_rate, 1e-6), 1.0)
    return float(np.quantile(scores, 1.0 - alert_rate))
