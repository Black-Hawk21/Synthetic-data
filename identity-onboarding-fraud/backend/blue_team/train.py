"""Training pipeline: Logistic Regression baseline -> Random Forest ->
XGBoost final model, with explicit class-imbalance handling, stratified
splits, and full model versioning (section 20/25).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from backend.blue_team.evaluate import compute_metrics, per_attack_type_recall
from backend.blue_team.preprocessing import FeatureSchema, build_feature_matrix, fit_scaler
from backend.config import settings

logger = logging.getLogger(__name__)


def _next_version(models_dir: Path) -> int:
    existing = [int(p.name.replace("model_v", "")) for p in models_dir.glob("model_v*") if p.name.replace("model_v", "").isdigit()]
    return (max(existing) + 1) if existing else 1


def train_and_evaluate(
    df: pd.DataFrame,
    models_dir: Path | None = None,
    seed: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.15,
) -> dict:
    """Full train -> compare -> select -> save pipeline. Returns a summary
    dict with metrics for all three models plus the version that was saved.

    Splits are STRATIFIED on the label and graph features are assumed to
    already be computed on the full dataset *before* the split (they are
    purely structural, not label-derived, so this does not leak the label --
    section 20)."""
    models_dir = models_dir or settings.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    # Manual (Live KYC Form) submissions have UNKNOWN ground truth -- they
    # must never be trained on or counted in metrics, only ever scored
    # on-demand via predict.score_applicant.
    if "source" in df.columns:
        df = df[df["source"] != "manual"]
    df = df.reset_index(drop=True)
    y = df["is_fraud"].astype(int).values

    X, schema = build_feature_matrix(df)
    scaler = fit_scaler(X)

    idx = np.arange(len(df))
    idx_train, idx_temp = train_test_split(idx, test_size=(test_size + val_size), random_state=seed, stratify=y)
    y_temp = y[idx_temp]
    rel_test = test_size / (test_size + val_size)
    idx_val, idx_test = train_test_split(idx_temp, test_size=rel_test, random_state=seed, stratify=y_temp)

    X_train, y_train = X.iloc[idx_train], y[idx_train]
    X_val, y_val = X.iloc[idx_val], y[idx_val]
    X_test, y_test = X.iloc[idx_test], y[idx_test]
    df_test = df.iloc[idx_test]

    X_train_s = scaler.transform(X_train.values)
    X_val_s = scaler.transform(X_val.values)
    X_test_s = scaler.transform(X_test.values)

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = (n_neg / max(n_pos, 1))

    results = {}

    logger.info("Training LogisticRegression baseline")
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    lr.fit(X_train_s, y_train)
    lr_prob = lr.predict_proba(X_test_s)[:, 1]
    results["logistic_regression"] = {
        "model": lr, "metrics": compute_metrics(y_test, lr_prob),
    }

    logger.info("Training RandomForest")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=3,
        class_weight="balanced", random_state=seed, n_jobs=-1,
    )
    rf.fit(X_train.values, y_train)
    rf_prob = rf.predict_proba(X_test.values)[:, 1]
    results["random_forest"] = {
        "model": rf, "metrics": compute_metrics(y_test, rf_prob),
    }

    logger.info("Training XGBoost (final model)")
    xgb = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.06,
        subsample=0.85, colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
        random_state=seed, n_jobs=-1,
    )
    xgb.fit(X_train.values, y_train, eval_set=[(X_val.values, y_val)], verbose=False)
    xgb_prob = xgb.predict_proba(X_test.values)[:, 1]
    xgb_metrics = compute_metrics(y_test, xgb_prob)
    results["xgboost"] = {"model": xgb, "metrics": xgb_metrics}

    per_attack = per_attack_type_recall(df_test, xgb_prob)

    version = _next_version(models_dir)
    version_dir = models_dir / f"model_v{version}"
    version_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(xgb, version_dir / "model.pkl")
    joblib.dump(scaler, version_dir / "scaler.pkl")
    schema.save(str(version_dir / "feature_schema.json"))

    comparison_metrics = {name: r["metrics"] for name, r in results.items()}
    metrics_payload = {
        "version": version,
        "trained_at": datetime.utcnow().isoformat(),
        "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
        "n_total": len(df), "class_balance_train": {"fraud": n_pos, "legit": n_neg},
        "model_comparison": comparison_metrics,
        "final_model": "xgboost",
        "final_model_metrics": xgb_metrics,
        "per_attack_type_recall": per_attack,
    }
    with open(version_dir / "metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    metadata = {
        "version": version,
        "trained_at": metrics_payload["trained_at"],
        "algorithm": "XGBoost (XGBClassifier)",
        "n_training_samples": len(X_train),
        "attack_types_present": sorted(df["attack_type"].unique().tolist()),
        "scale_pos_weight": scale_pos_weight,
        "random_seed": seed,
        "feature_count": len(schema.model_feature_order),
    }
    with open(version_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Update "current" pointer used by predict.py
    with open(models_dir / "current_version.json", "w") as f:
        json.dump({"current_version": version}, f, indent=2)

    logger.info("Saved model_v%s to %s", version, version_dir)
    return {
        "version": version,
        "version_dir": str(version_dir),
        "comparison_metrics": comparison_metrics,
        "final_model_metrics": xgb_metrics,
        "per_attack_type_recall": per_attack,
        "held_out_test_df": df_test,
        "held_out_test_prob": xgb_prob,
    }
