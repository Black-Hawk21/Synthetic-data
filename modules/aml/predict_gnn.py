"""Load the trained GNN detectors and classify accounts + transactions.

Usage (from inside `aml-sim/`):

    python predict_gnn.py                            # score data/, write predictions/
    python predict_gnn.py --data data --out predictions
    python predict_gnn.py --accounts-only

Writes:
    <out>/gnn_account_predictions.csv
    <out>/gnn_transaction_predictions.csv

Both carry `laundering_score` (the ranking signal) and `is_flagged` (the score
thresholded at the analyst budget stored in the bundle). Metrics are printed
when ground-truth labels are present in the data.

The account graph is scored full-batch — the whole graph goes through the
network in one pass, so every account's score reflects its 3-hop neighbourhood.
Transactions are scored in chunks to keep memory flat on large ledgers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aml_models import account_feature_matrix, transaction_feature_matrix
from gnn_model import (AMLGNN, TransactionHead, build_graph_tensors, load_gnn,
                       normalize_features)


def _report(name: str, df: pd.DataFrame) -> None:
    if "is_laundering" not in df.columns:
        return
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, s, f = df["is_laundering"].astype(int), df["laundering_score"], df["is_flagged"]
    tp = int(((f == 1) & (y == 1)).sum())
    print(f"[{name}] PR-AUC {average_precision_score(y, s):.4f} | "
          f"ROC-AUC {roc_auc_score(y, s):.4f} | flagged {int(f.sum()):,} "
          f"({f.mean():.2%}) | precision {tp / max(int(f.sum()), 1):.3f} | "
          f"recall {tp / max(int(y.sum()), 1):.3f}")


def _node_embeddings(data: Path, models: Path):
    """Run the account GNN once; return (accounts, embeddings, logits, bundle)."""
    bundle = load_gnn(models / "gnn_account_model.pt")
    feats = pd.read_csv(data / "account_features.csv")
    edges = pd.read_csv(data / "edges.csv")

    Xdf = account_feature_matrix(feats, columns=bundle.get("columns"))
    Xn, _ = normalize_features(Xdf.to_numpy(np.float32), bundle["norm_stats"])
    x = torch.from_numpy(Xn.copy())
    edge_index, edge_attr = build_graph_tensors(edges, feats["account_id"])

    model = AMLGNN(bundle["in_dim"], hidden=bundle["hidden"],
                   layers=bundle["layers"], dropout=0.0)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    with torch.no_grad():
        emb = model.embed(x, edge_index, edge_attr)
        scores = torch.sigmoid(model.head(emb).squeeze(-1)).numpy()
    return feats, emb, scores, bundle


def score_accounts(data: Path, models: Path, out: Path, cache: dict) -> None:
    feats, emb, scores, bundle = cache["node"]
    result = pd.DataFrame({
        "account_id": feats["account_id"],
        "laundering_score": scores.round(6),
        "is_flagged": (scores >= bundle["threshold"]).astype(int),
    })
    if "is_laundering" in feats.columns:
        result["is_laundering"] = feats["is_laundering"].astype(int)
    result = result.sort_values("laundering_score", ascending=False)
    result.to_csv(out / "gnn_account_predictions.csv", index=False)
    _report("gnn/accounts", result)
    print(f"  -> {out / 'gnn_account_predictions.csv'}  ({len(result):,} rows)")


def score_transactions(data: Path, models: Path, out: Path, cache: dict) -> None:
    feats, emb, _, _ = cache["node"]
    bundle = load_gnn(models / "gnn_transaction_model.pt")
    emb = (emb - emb.mean(0)) / emb.std(0).clamp(min=1e-6)
    index = {a: i for i, a in enumerate(feats["account_id"])}

    txns = pd.read_csv(data / "transactions.csv")
    Xt = transaction_feature_matrix(txns, feats, columns=bundle["columns"])
    Xt = torch.from_numpy(np.nan_to_num(Xt.to_numpy(np.float32), posinf=0, neginf=0))
    Xt = (Xt - Xt.mean(0)) / Xt.std(0).clamp(min=1e-6)
    src = torch.from_numpy(txns["sender"].map(index).fillna(0).to_numpy(np.int64))
    dst = torch.from_numpy(txns["receiver"].map(index).fillna(0).to_numpy(np.int64))

    head = TransactionHead(bundle["txn_dim"], bundle["emb_dim"],
                           hidden=bundle["hidden"], dropout=0.0)
    head.load_state_dict(bundle["state_dict"])
    head.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(txns), 65536):
            sl = slice(i, i + 65536)
            chunks.append(torch.sigmoid(
                head(Xt[sl], emb[src[sl]], emb[dst[sl]])).numpy())
    scores = np.concatenate(chunks)

    result = pd.DataFrame({
        "txn_id": txns["txn_id"], "sender": txns["sender"],
        "receiver": txns["receiver"], "amount": txns["amount"],
        "laundering_score": scores.round(6),
        "is_flagged": (scores >= bundle["threshold"]).astype(int),
    })
    if "is_laundering" in txns.columns:
        result["is_laundering"] = txns["is_laundering"].astype(int)
    result = result.sort_values("laundering_score", ascending=False)
    result.to_csv(out / "gnn_transaction_predictions.csv", index=False)
    _report("gnn/transactions", result)
    print(f"  -> {out / 'gnn_transaction_predictions.csv'}  ({len(result):,} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="predictions")
    ap.add_argument("--accounts-only", action="store_true")
    ap.add_argument("--transactions-only", action="store_true")
    ap.add_argument("--threads", type=int,
                    default=max(1, (__import__("os").cpu_count() or 2)))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    data, models, out = Path(args.data), Path(args.models), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = {"node": _node_embeddings(data, models)}
    if not args.transactions_only:
        score_accounts(data, models, out, cache)
    if not args.accounts_only:
        score_transactions(data, models, out, cache)


if __name__ == "__main__":
    main()
