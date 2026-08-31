"""Load the saved AML detectors and classify accounts + transactions.

Usage (from inside `aml-sim/`):

    python predict.py                              # score data/, write predictions/
    python predict.py --data data --models models --out predictions
    python predict.py --accounts-only
    python predict.py --transactions-only

Reads:
    <models>/account_model.joblib, <models>/transaction_model.joblib
    <data>/account_features.csv, <data>/transactions.csv

Writes:
    <out>/account_predictions.csv      account_id, laundering_score, is_flagged
    <out>/transaction_predictions.csv  txn_id, sender, receiver, amount,
                                       laundering_score, is_flagged

If ground-truth labels are present in the data, evaluation metrics are printed;
otherwise the scripts simply score. `is_flagged` uses the operating threshold
stored in the bundle (an analyst-budget cutoff), so treat `laundering_score`
as the ranking signal and the flag as one reasonable cutoff.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from aml_models import (account_feature_matrix, load_bundle,
                        transaction_feature_matrix)


def _maybe_report(name: str, df: pd.DataFrame, label_col: str) -> None:
    if label_col not in df.columns:
        return
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, s, f = df[label_col].astype(int), df["laundering_score"], df["is_flagged"]
    tp = int(((f == 1) & (y == 1)).sum())
    precision = tp / max(int(f.sum()), 1)
    recall = tp / max(int(y.sum()), 1)
    print(f"[{name}] PR-AUC {average_precision_score(y, s):.4f} | "
          f"ROC-AUC {roc_auc_score(y, s):.4f} | flagged {int(f.sum()):,} "
          f"({f.mean():.2%}) | precision {precision:.3f} | recall {recall:.3f}")


def score_accounts(data: Path, models: Path, out: Path) -> pd.DataFrame:
    bundle = load_bundle(models / "account_model.joblib")
    feats = pd.read_csv(data / "account_features.csv")
    X = account_feature_matrix(feats, columns=bundle["columns"])
    scores = bundle["model"].predict_proba(X)[:, 1]

    result = pd.DataFrame({
        "account_id": feats["account_id"],
        "laundering_score": scores.round(6),
        "is_flagged": (scores >= bundle["threshold"]).astype(int),
    })
    if "is_laundering" in feats.columns:
        result["is_laundering"] = feats["is_laundering"].astype(int)
    result = result.sort_values("laundering_score", ascending=False)
    result.to_csv(out / "account_predictions.csv", index=False)
    _maybe_report("accounts", result, "is_laundering")
    print(f"  -> {out / 'account_predictions.csv'}  ({len(result):,} rows)")
    return result


def score_transactions(data: Path, models: Path, out: Path) -> pd.DataFrame:
    bundle = load_bundle(models / "transaction_model.joblib")
    txns = pd.read_csv(data / "transactions.csv")
    feats = pd.read_csv(data / "account_features.csv")
    X = transaction_feature_matrix(txns, feats, columns=bundle["columns"])
    scores = bundle["model"].predict_proba(X)[:, 1]

    result = pd.DataFrame({
        "txn_id": txns["txn_id"],
        "sender": txns["sender"],
        "receiver": txns["receiver"],
        "amount": txns["amount"],
        "laundering_score": scores.round(6),
        "is_flagged": (scores >= bundle["threshold"]).astype(int),
    })
    if "is_laundering" in txns.columns:
        result["is_laundering"] = txns["is_laundering"].astype(int)
    result = result.sort_values("laundering_score", ascending=False)
    result.to_csv(out / "transaction_predictions.csv", index=False)
    _maybe_report("transactions", result, "is_laundering")
    print(f"  -> {out / 'transaction_predictions.csv'}  ({len(result):,} rows)")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="predictions")
    ap.add_argument("--accounts-only", action="store_true")
    ap.add_argument("--transactions-only", action="store_true")
    args = ap.parse_args()

    data, models, out = Path(args.data), Path(args.models), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.transactions_only:
        score_accounts(data, models, out)
    if not args.accounts_only:
        score_transactions(data, models, out)


if __name__ == "__main__":
    main()
