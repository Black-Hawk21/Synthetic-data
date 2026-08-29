"""Evaluate every detector on identical held-out splits and print the comparison.

Usage (from inside `aml-sim/`):

    python compare_models.py
    python compare_models.py --data data --models models --out evaluation

This is the honest scoreboard: the tabular and GNN models are scored on exactly
the same test rows (70/30 stratified for accounts, GroupShuffleSplit on
`episode_id` for transactions, same seed), so the numbers are directly
comparable. Scoring the full dataset instead would include training rows and
flatter every model.

Also prints per-typology recall, which is the metric that actually matters here
— an average across typologies hides one you cannot see at all.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from aml_models import (account_feature_matrix, load_bundle,
                        threshold_at_alert_rate, transaction_feature_matrix)


def _row(name: str, y, s, alert_rate: float) -> dict:
    thr = threshold_at_alert_rate(s, alert_rate)
    flag = s >= thr
    tp = int((flag & (y == 1)).sum())
    return {"model": name,
            "PR-AUC": round(average_precision_score(y, s), 4),
            "ROC-AUC": round(roc_auc_score(y, s), 4),
            "precision": round(tp / max(int(flag.sum()), 1), 3),
            "recall": round(tp / max(int(y.sum()), 1), 3),
            "alerts": int(flag.sum())}


# ---------------------------------------------------------------------------
def compare_accounts(data: Path, models: Path, alert_mult: float, seed: int):
    import torch
    from gnn_model import AMLGNN, build_graph_tensors, load_gnn, normalize_features

    feats = pd.read_csv(data / "account_features.csv")
    y = feats["is_laundering"].astype(int).to_numpy()
    idx = np.arange(len(y))
    _, test = train_test_split(idx, test_size=0.3, stratify=y, random_state=seed)
    alert_rate = alert_mult * y.mean()

    rows, scores = [], {}

    tab = load_bundle(models / "account_model.joblib")
    X = account_feature_matrix(feats, columns=tab["columns"])
    s = tab["model"].predict_proba(X)[:, 1]
    scores["GradientBoosting (tabular)"] = s
    rows.append(_row("GradientBoosting (tabular)", y[test], s[test], alert_rate))

    b = load_gnn(models / "gnn_account_model.pt")
    Xdf = account_feature_matrix(feats, columns=b.get("columns"))
    Xn, _ = normalize_features(Xdf.to_numpy(np.float32), b["norm_stats"])
    edge_index, edge_attr = build_graph_tensors(
        pd.read_csv(data / "edges.csv"), feats["account_id"])
    net = AMLGNN(b["in_dim"], hidden=b["hidden"], layers=b["layers"], dropout=0.0)
    net.load_state_dict(b["state_dict"])
    net.eval()
    with torch.no_grad():
        s = torch.sigmoid(net(torch.from_numpy(Xn.copy()), edge_index,
                              edge_attr)).numpy()
    scores["AML-GNN (graph)"] = s
    rows.append(_row("AML-GNN (graph)", y[test], s[test], alert_rate))

    return pd.DataFrame(rows), feats.iloc[test].reset_index(drop=True), \
        {k: v[test] for k, v in scores.items()}, alert_rate


def compare_transactions(data: Path, models: Path, alert_mult: float, seed: int):
    import torch
    from gnn_model import (AMLGNN, TransactionHead, build_graph_tensors,
                           load_gnn, normalize_features)

    txns = pd.read_csv(data / "transactions.csv")
    feats = pd.read_csv(data / "account_features.csv")
    y = txns["is_laundering"].astype(int).to_numpy()
    groups = txns["episode_id"].fillna("").astype(str)
    groups = np.where(groups == "", "T_" + txns["txn_id"].astype(str), groups)
    _, test = next(GroupShuffleSplit(n_splits=1, test_size=0.3,
                                     random_state=seed).split(txns, y, groups))
    alert_rate = alert_mult * y.mean()
    rows = []

    tab = load_bundle(models / "transaction_model.joblib")
    Xt = transaction_feature_matrix(txns, feats, columns=tab["columns"])
    s = tab["model"].predict_proba(Xt.iloc[test])[:, 1]
    rows.append(_row("GradientBoosting (tabular)", y[test], s, alert_rate))
    del Xt

    tb = load_gnn(models / "gnn_transaction_model.pt")
    nb = load_gnn(models / "gnn_account_model.pt")
    Xdf = account_feature_matrix(feats, columns=nb.get("columns"))
    Xn, _ = normalize_features(Xdf.to_numpy(np.float32), nb["norm_stats"])
    edge_index, edge_attr = build_graph_tensors(
        pd.read_csv(data / "edges.csv"), feats["account_id"])
    net = AMLGNN(nb["in_dim"], hidden=nb["hidden"], layers=nb["layers"], dropout=0.0)
    net.load_state_dict(nb["state_dict"])
    net.eval()
    with torch.no_grad():
        emb = net.embed(torch.from_numpy(Xn.copy()), edge_index, edge_attr)
    emb = (emb - emb.mean(0)) / emb.std(0).clamp(min=1e-6)

    index = {a: i for i, a in enumerate(feats["account_id"])}
    Xt = transaction_feature_matrix(txns, feats, columns=tb["columns"])
    Xt = torch.from_numpy(np.nan_to_num(Xt.to_numpy(np.float32), posinf=0, neginf=0))
    Xt = (Xt - Xt.mean(0)) / Xt.std(0).clamp(min=1e-6)
    src = torch.from_numpy(txns["sender"].map(index).fillna(0).to_numpy(np.int64))
    dst = torch.from_numpy(txns["receiver"].map(index).fillna(0).to_numpy(np.int64))
    head = TransactionHead(tb["txn_dim"], tb["emb_dim"], hidden=tb["hidden"],
                           dropout=0.0)
    head.load_state_dict(tb["state_dict"])
    head.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(test), 65536):
            b = torch.from_numpy(test[i:i + 65536])
            out.append(torch.sigmoid(head(Xt[b], emb[src[b]], emb[dst[b]])).numpy())
    rows.append(_row("GNN + transaction head", y[test], np.concatenate(out),
                     alert_rate))
    return pd.DataFrame(rows)


def per_typology(test_feats: pd.DataFrame, scores: dict, alert_rate: float
                 ) -> pd.DataFrame:
    """Recall per laundering typology at a fixed alert budget."""
    from amlgen.evaluation.metrics import recall_by_pattern
    table = None
    for name, s in scores.items():
        r = recall_by_pattern(test_feats, s, alert_rate=alert_rate)
        r = r.rename(columns={"recall": name})
        table = r if table is None else table.merge(
            r[["pattern", name]], on="pattern", how="outer")
    cols = list(scores)
    table["delta"] = (table[cols[-1]] - table[cols[0]]).round(4)
    return table.sort_values("delta", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="evaluation")
    ap.add_argument("--alert-rate-mult", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--accounts-only", action="store_true")
    args = ap.parse_args()

    data, models, out = Path(args.data), Path(args.models), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    acc, test_feats, test_scores, alert_rate = compare_accounts(
        data, models, args.alert_rate_mult, args.seed)
    print("\n=== ACCOUNT LEVEL (held-out test split) ===")
    print(acc.to_string(index=False))
    acc.to_csv(out / "model_comparison_accounts.csv", index=False)

    typ = per_typology(test_feats, test_scores, alert_rate)
    print("\n=== RECALL BY TYPOLOGY (accounts, alert budget = "
          f"{args.alert_rate_mult:g}x base rate) ===")
    print(typ.to_string(index=False))
    typ.to_csv(out / "recall_by_typology.csv", index=False)

    if not args.accounts_only:
        txn = compare_transactions(data, models, args.alert_rate_mult, args.seed)
        print("\n=== TRANSACTION LEVEL (held-out test split) ===")
        print(txn.to_string(index=False))
        txn.to_csv(out / "model_comparison_transactions.csv", index=False)

    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    main()
