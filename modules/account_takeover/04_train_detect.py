"""
04_train_detect.py

BLUE TEAM layer, part 2: detection models.

Two models, combined:
  - Supervised XGBoost on our injected ATO labels. Fast, interpretable
    (feature importances), catches the attack patterns we know about.
  - Unsupervised Isolation Forest on the same numeric features, trained on
    ALL events (label-blind). Catches anomalies that don't match our
    injected pattern exactly -- this is the "novel attack" safety net,
    since a red team can never enumerate every real-world variant.

Final risk score = weighted blend of the two, 0-100.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve, average_precision_score
)
import xgboost as xgb

import config as cfg

FEATURE_COLS = [
    "device_new_for_account",
    "ip_new_for_account",
    "country_new_for_account",
    "is_first_event_for_account",
    "failed_logins_trailing_10min",
    "seconds_since_last_event",
    "amount_zscore_vs_history",
    "device_distinct_accounts_24h",
    "ip_distinct_accounts_24h",
    "is_login_event",
    "is_txn_event",
    "hour_of_day",
    "spending_deviation_score",
    "velocity_score",
    "geo_anomaly_score",
]


def prep_matrix(df):
    X = df[FEATURE_COLS].copy()
    # Login events have no transaction to compute amount/spend/velocity/geo
    # scores from. Filling those with 0 would create an artificial "missing
    # value" fingerprint that trivially separates event types (we hit this
    # exact leakage bug in an earlier version). Instead impute with the
    # population median from real transaction rows -- i.e. "assume typical"
    # rather than "assume zero" -- so the model has to find real signal in
    # the *other* features (novelty, velocity, fan-out) for login events.
    for col in ["amount_zscore_vs_history", "spending_deviation_score", "velocity_score", "geo_anomaly_score"]:
        median = X[col].median()
        X[col] = X[col].fillna(median)
    X = X.fillna(0.0)
    X["seconds_since_last_event"] = X["seconds_since_last_event"].clip(lower=0)
    return X


def train_supervised(X_train, y_train, X_test, y_test):
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="aucpr",
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        random_state=cfg.RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    print("\n--- Supervised XGBoost (held-out test set) ---")
    print(classification_report(y_test, (proba > 0.5).astype(int), digits=3))
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.4f}")
    print(f"PR-AUC : {average_precision_score(y_test, proba):.4f}")

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nTop feature importances:")
    print(importances.head(8).to_string())
    return model


def train_unsupervised(X_all):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    iso = IsolationForest(
        n_estimators=300,
        contamination=0.08,   # rough prior: assume ~8% of events are anomalous
        random_state=cfg.RANDOM_SEED,
    )
    iso.fit(X_scaled)
    return iso, scaler


def main():
    print("Loading features...")
    df = pd.read_csv(cfg.FEATURES_CSV, parse_dates=["event_time"])
    X = prep_matrix(df)
    y = df["label_ato"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=cfg.RANDOM_SEED
    )

    xgb_model = train_supervised(X_train, y_train, X_test, y_test)
    iso_model, scaler = train_unsupervised(X)

    # blended risk score across ALL rows, for downstream mitigation demo
    xgb_proba_all = xgb_model.predict_proba(X)[:, 1]
    iso_raw = -iso_model.score_samples(scaler.transform(X))  # higher = more anomalous
    iso_norm = (iso_raw - iso_raw.min()) / (iso_raw.max() - iso_raw.min() + 1e-9)

    blended = 0.7 * xgb_proba_all + 0.3 * iso_norm
    df["risk_xgb"] = xgb_proba_all
    df["risk_iforest"] = iso_norm
    df[cfg.RISK_SCORE_COL] = (blended * 100).round(1)

    df.to_csv(cfg.SCORED_EVENTS_CSV, index=False)
    print(f"\nSaved scored events -> {cfg.SCORED_EVENTS_CSV}")

    xgb_model.save_model(cfg.MODEL_PATH)
    joblib.dump(iso_model, cfg.IFOREST_PATH)
    joblib.dump(scaler, cfg.SCALER_PATH)
    print("Saved models ->", cfg.MODEL_PATH, cfg.IFOREST_PATH, cfg.SCALER_PATH)


if __name__ == "__main__":
    main()
