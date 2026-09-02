"""Evaluation metrics -- deliberately goes beyond accuracy (section 24)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = float("nan")
    pr_auc = average_precision_score(y_true, y_prob)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "threshold": threshold,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4) if roc_auc == roc_auc else None,
        "pr_auc": round(float(pr_auc), 4),
        "false_positive_rate": round(float(fpr), 4),
        "false_negative_rate": round(float(fnr), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_samples": int(len(y_true)),
        "n_fraud": int(y_true.sum()),
    }


def per_attack_type_recall(df: pd.DataFrame, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Recall broken out by attack_type (section 15)."""
    out = {}
    d = df.copy()
    d["_prob"] = y_prob
    d["_pred"] = (d["_prob"] >= threshold).astype(int)
    for attack_type, sub in d[d["is_fraud"] == 1].groupby("attack_type"):
        recall = float((sub["_pred"] == 1).mean())
        out[attack_type] = {"recall": round(recall, 4), "n_samples": int(len(sub))}
    return out
