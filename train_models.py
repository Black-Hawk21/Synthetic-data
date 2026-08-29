"""Train the account-level and transaction-level AML detectors and save them.

Usage (from inside `aml-sim/`, after `python run.py all` or `run.py features`):

    python train_models.py                       # reads data/, writes models/
    python train_models.py --data data --models models
    python train_models.py --alert-rate-mult 2.0

Outputs:
    models/account_model.joblib      — flags suspicious ACCOUNTS
    models/transaction_model.joblib  — flags suspicious TRANSACTIONS

Both files are self-contained bundles: {model, columns, threshold, meta}.
Load them with `predict.py` or with `aml_models.load_bundle(...)`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from aml_models import (account_feature_matrix, save_bundle,
                        threshold_at_alert_rate, transaction_feature_matrix)


def _report(name: str, y_true, scores, thr: float) -> None:
    flag = scores >= thr
    tp = int(((flag == 1) & (y_true == 1)).sum())
    precision = tp / max(int(flag.sum()), 1)
    recall = tp / max(int(y_true.sum()), 1)
    print(f"[{name}] PR-AUC {average_precision_score(y_true, scores):.4f} | "
          f"ROC-AUC {roc_auc_score(y_true, scores):.4f} | "
          f"@threshold {thr:.4f}: precision {precision:.3f}, "
          f"recall {recall:.3f}, alerts {int(flag.sum()):,}")


def train_account_model(data: Path, models: Path, alert_mult: float,
                        seed: int) -> None:
    feats = pd.read_csv(data / "account_features.csv")
    y = feats["is_laundering"].astype(int).to_numpy()
    X = account_feature_matrix(feats)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                          stratify=y, random_state=seed)
    model = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.08, l2_regularization=1.0,
        early_stopping=True, class_weight="balanced",
        random_state=seed).fit(Xtr, ytr)
    scores = model.predict_proba(Xte)[:, 1]
    thr = threshold_at_alert_rate(scores, alert_mult * y.mean())
    _report("account model / holdout", yte, scores, thr)

    save_bundle(models / "account_model.joblib", model, list(X.columns), thr,
                meta={"level": "account", "target": "is_laundering",
                      "base_rate": float(y.mean()),
                      "alert_rate": float(alert_mult * y.mean())})
    print(f"  -> saved {models / 'account_model.joblib'}")


def train_transaction_model(data: Path, models: Path, alert_mult: float,
                            seed: int) -> None:
    txns = pd.read_csv(data / "transactions.csv")
    feats = pd.read_csv(data / "account_features.csv")
    y = txns["is_laundering"].astype(int).to_numpy()
    X = transaction_feature_matrix(txns, feats)

    # Group split by laundering episode so one episode never straddles the
    # train/test boundary (each normal transaction is its own group).
    groups = txns["episode_id"].fillna("").astype(str)
    groups = np.where(groups == "", "T_" + txns["txn_id"].astype(str), groups)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3,
                                    random_state=seed).split(X, y, groups))
    model = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.08, l2_regularization=1.0,
        early_stopping=True, class_weight="balanced",
        random_state=seed).fit(X.iloc[tr], y[tr])
    scores = model.predict_proba(X.iloc[te])[:, 1]
    thr = threshold_at_alert_rate(scores, alert_mult * y.mean())
    _report("transaction model / holdout", y[te], scores, thr)

    save_bundle(models / "transaction_model.joblib", model, list(X.columns),
                thr, meta={"level": "transaction", "target": "is_laundering",
                           "base_rate": float(y.mean()),
                           "alert_rate": float(alert_mult * y.mean())})
    print(f"  -> saved {models / 'transaction_model.joblib'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data", help="dataset directory")
    ap.add_argument("--models", default="models", help="output directory")
    ap.add_argument("--alert-rate-mult", type=float, default=2.0,
                    help="alert budget as a multiple of the base rate "
                         "(repo convention: 2x)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data, models = Path(args.data), Path(args.models)
    models.mkdir(parents=True, exist_ok=True)
    train_account_model(data, models, args.alert_rate_mult, args.seed)
    train_transaction_model(data, models, args.alert_rate_mult, args.seed)


if __name__ == "__main__":
    main()
