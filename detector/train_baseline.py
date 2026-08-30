"""
Baseline blue-team detector: TF-IDF features + Logistic Regression.

Deliberately simple and fast -- this is the "explainable baseline" the
report compares a stronger model against, not the final word. Trains on
our own generated data (train/test split), then separately evaluates
against the holdout (public, real-world) data to check generalization.

Usage:
    cd detector
    python train_baseline.py
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)

sys.path.insert(0, os.path.dirname(__file__))
from dataset_utils import load_all, summarize

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
EVAL_DIR = os.path.join(os.path.dirname(__file__), "..", "eval")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "baseline_model.joblib")
VECTORIZER_PATH = os.path.join(ARTIFACTS_DIR, "baseline_vectorizer.joblib")
METRICS_PATH = os.path.join(EVAL_DIR, "baseline_metrics.json")


def compute_breakdown(df: pd.DataFrame, y_true, y_pred, y_prob, group_col: str) -> dict:
    """Precision/recall/F1 broken down by a grouping column (e.g. attack_subtype).
    Only meaningful for rows where the group is a real category, and we compute
    per-group recall on the FRAUD class specifically (since that's what "did we
    catch this kind of attack" means for a subtype breakdown)."""
    out = {}
    tmp = df.copy()
    tmp["_pred"] = y_pred
    tmp["_true"] = y_true
    for group_val, sub in tmp.groupby(group_col):
        if (sub["_true"] == 1).sum() == 0:
            # group has no fraud examples (e.g. "legit" subtype) -- report FPR instead
            fp_rate = float((sub["_pred"] == 1).mean())
            out[str(group_val)] = {"n": int(len(sub)), "false_positive_rate": round(fp_rate, 4)}
        else:
            fraud_sub = sub[sub["_true"] == 1]
            recall = float((fraud_sub["_pred"] == 1).mean())
            out[str(group_val)] = {"n": int(len(sub)), "recall_on_fraud": round(recall, 4)}
    return out


def evaluate_split(name: str, df: pd.DataFrame, vectorizer, model) -> dict:
    if df.empty:
        return {"note": f"{name} set is empty, skipped"}

    X = vectorizer.transform(df["text"].fillna(""))
    y_true = df["label"].values
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "n": int(len(df)),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }
    # AUC needs both classes present
    if len(set(y_true)) > 1:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    metrics["false_positive_rate_on_legit"] = round(float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0, 4)

    if "attack_subtype" in df.columns:
        metrics["breakdown_by_subtype"] = compute_breakdown(df, y_true, y_pred, y_prob, "attack_subtype")
    if "difficulty_tier" in df.columns:
        metrics["breakdown_by_difficulty"] = compute_breakdown(df, y_true, y_pred, y_prob, "difficulty_tier")

    print(f"\n=== {name} ===")
    print(f"  n={metrics['n']}  precision={metrics['precision']}  recall={metrics['recall']}  "
          f"f1={metrics['f1']}  auc={metrics.get('roc_auc', 'n/a')}")
    print(f"  false positive rate on legit: {metrics['false_positive_rate_on_legit']}")
    if "breakdown_by_subtype" in metrics:
        print(f"  by subtype: {json.dumps(metrics['breakdown_by_subtype'], indent=2)}")

    return metrics


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    train_pool_df, holdout_df = load_all()
    summarize(train_pool_df, "train pool")
    summarize(holdout_df, "holdout")

    if train_pool_df.empty:
        print("\nNo generated data found in data/generated/*.jsonl -- run generate_static.py "
              "(and optionally generate_conversational.py) first.", file=sys.stderr)
        sys.exit(1)

    if train_pool_df["label"].nunique() < 2:
        print("\nTraining data only has one class -- check your generator output "
              "(are fraud AND legit samples both present?).", file=sys.stderr)
        sys.exit(1)

    train_df, test_df = train_test_split(
        train_pool_df, test_size=0.2, random_state=42, stratify=train_pool_df["label"],
    )
    print(f"\nSplit: {len(train_df)} train / {len(test_df)} test")

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_df["text"].fillna(""))
    y_train = train_df["label"].values

    model = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved vectorizer to {VECTORIZER_PATH}")

    all_metrics = {
        "test_split": evaluate_split("Test split (held-out from our own generated data)",
                                      test_df, vectorizer, model),
    }
    if not holdout_df.empty:
        all_metrics["public_holdout"] = evaluate_split(
            "Public holdout (real-world data, never trained on -- generalization check)",
            holdout_df, vectorizer, model)
    else:
        print("\n[note] no public holdout data found -- run prepare_holdout.py to add one "
              "for a generalization check.")

    # top features (explainability) -- what the baseline actually keys on
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]
    top_fraud_idx = np.argsort(coefs)[-15:][::-1]
    top_legit_idx = np.argsort(coefs)[:15]
    all_metrics["top_fraud_indicator_terms"] = feature_names[top_fraud_idx].tolist()
    all_metrics["top_legit_indicator_terms"] = feature_names[top_legit_idx].tolist()

    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nWrote full metrics report to {METRICS_PATH}")

    print("\nTop terms pushing toward FRAUD classification:")
    print("  " + ", ".join(all_metrics["top_fraud_indicator_terms"]))
    print("Top terms pushing toward LEGIT classification:")
    print("  " + ", ".join(all_metrics["top_legit_indicator_terms"]))


if __name__ == "__main__":
    main()
