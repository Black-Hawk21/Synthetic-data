"""Assembles the modelling table: behavioural + graph features + labels."""
from __future__ import annotations

import pandas as pd

from ..graphs import build_edge_table
from .account_features import build_account_features
from .graph_features import build_graph_features

# Columns that leak ground truth and must never be fed to a model.
LABEL_COLUMNS = ["is_laundering", "laundering_role", "laundering_patterns",
                 "n_laundering_episodes", "in_benign_lookalike"]
# Columns that describe the simulator's own state rather than observable behaviour.
SIMULATOR_ONLY = ["account_idx", "dormant", "baseline_out_per_day",
                  "baseline_amount_median", "baseline_amount_sigma", "night_ratio",
                  "monthly_income", "popularity", "employer_idx", "salary_amount",
                  "salary_day", "open_ts", "is_business", "business_hours"]


def build_feature_table(txns: pd.DataFrame, accounts: pd.DataFrame,
                        threshold: float = 200_000.0,
                        edges: pd.DataFrame | None = None) -> pd.DataFrame:
    if edges is None:
        edges = build_edge_table(txns)
    behavioural = build_account_features(txns, threshold=threshold)
    topological = build_graph_features(edges, accounts)

    kyc = accounts[["account_id", "archetype", "country", "city", "kyc_level",
                    "account_age_days"] + [c for c in LABEL_COLUMNS if c in accounts.columns]]
    df = (kyc.merge(behavioural, on="account_id", how="left")
             .merge(topological, on="account_id", how="left"))
    return df.fillna(0.0)


def model_matrix(features: pd.DataFrame, target: str = "is_laundering",
                 one_hot: bool = True):
    """Split the feature table into X, y with every leaking column removed."""
    drop = set(LABEL_COLUMNS) | set(SIMULATOR_ONLY) | {"account_id"}
    y = features[target].astype(int).to_numpy()
    X = features.drop(columns=[c for c in drop if c in features.columns])
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if one_hot and cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    else:
        X = X.drop(columns=cat_cols)
    return X.astype(float), y
