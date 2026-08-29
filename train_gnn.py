"""Train the GNN detectors on the transaction graph.

Two stages:

  stage 1 (account)      full-batch node classification over the account graph
  stage 2 (transaction)  a head over the frozen GNN embeddings of the sender
                         and receiver, plus the transaction's own attributes

Usage (from inside `aml-sim/`, after `python run.py all`):

    python train_gnn.py                     # both stages, default 250 epochs
    python train_gnn.py --stage account
    python train_gnn.py --epochs 400 --hidden 128
    python train_gnn.py --max-seconds 240   # stop early, checkpoint, resume later

Training is resumable. Every run writes `models/gnn_<stage>_checkpoint.pt`; if
`--max-seconds` fires or you interrupt it, run the same command again and it
picks up where it stopped. On a CPU-only laptop the account stage takes roughly
10-20 minutes at the default settings, so this matters.

Outputs:
    models/gnn_account_model.pt
    models/gnn_transaction_model.pt
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from aml_models import threshold_at_alert_rate, transaction_feature_matrix
from aml_models import account_feature_matrix
from gnn_model import (AMLGNN, TransactionHead, build_graph_tensors, load_gnn,
                       normalize_features, save_gnn)


# ---------------------------------------------------------------------------
def load_graph(data: Path, columns: list[str] | None = None,
               stats: tuple | None = None):
    feats = pd.read_csv(data / "account_features.csv")
    edges = pd.read_csv(data / "edges.csv")
    Xdf = account_feature_matrix(feats, columns=columns)
    Xn, stats = normalize_features(Xdf.to_numpy(np.float32), stats)
    edge_index, edge_attr = build_graph_tensors(edges, feats["account_id"])
    return (feats, torch.from_numpy(Xn.copy()), edge_index, edge_attr, stats,
            list(Xdf.columns))


def splits(y: np.ndarray, seed: int):
    """Same 70/30 test split as the tabular baseline, with val carved from train."""
    idx = np.arange(len(y))
    train_all, test = train_test_split(idx, test_size=0.3, stratify=y,
                                       random_state=seed)
    train, val = train_test_split(train_all, test_size=0.2,
                                  stratify=y[train_all], random_state=seed)
    return train, val, test


def _metrics(y, s):
    return average_precision_score(y, s), roc_auc_score(y, s)


# ---------------------------------------------------------------------------
def train_account(data: Path, models: Path, args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)

    feats, x, edge_index, edge_attr, stats, columns = load_graph(data)
    y = feats["is_laundering"].astype(int).to_numpy()
    yt = torch.from_numpy(y).float()
    train, val, test = splits(y, args.seed)
    tr, va, te = (torch.from_numpy(a) for a in (train, val, test))
    print(f"[gnn/account] {x.shape[0]:,} nodes, {edge_index.shape[1]:,} edges, "
          f"{x.shape[1]} features, {y.sum():,} laundering")

    model = AMLGNN(x.size(1), hidden=args.hidden, layers=args.layers,
                   dropout=args.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    pos_weight = torch.tensor((y[train] == 0).sum() / max((y[train] == 1).sum(), 1),
                              dtype=torch.float32)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    ckpt_path = models / "gnn_account_checkpoint.pt"
    start, best_val, best_state = 1, -1.0, None
    if args.resume and ckpt_path.exists():
        ck = load_gnn(ckpt_path)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, best_val, best_state = ck["epoch"] + 1, ck["best_val"], ck["best_state"]
        print(f"[gnn/account] resumed at epoch {start} (best val PR-AUC {best_val:.4f})")

    t0 = time.time()
    for epoch in range(start, args.epochs + 1):
        model.train()
        opt.zero_grad()
        # DropEdge: a graph-native regulariser, and it halves epoch cost.
        if args.drop_edge > 0:
            keep = torch.rand(edge_index.size(1)) >= args.drop_edge
            ei, ea = edge_index[:, keep], edge_attr[keep]
        else:
            ei, ea = edge_index, edge_attr
        loss = lossf(model(x, ei, ea)[tr], yt[tr])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        sched.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                scores = torch.sigmoid(model(x, edge_index, edge_attr)).numpy()
            vp, _ = _metrics(y[val], scores[val])
            tp, _ = _metrics(y[test], scores[test])
            if vp > best_val:
                best_val = vp
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  epoch {epoch:4d} | loss {loss.item():.4f} | "
                  f"val PR-AUC {vp:.4f} | test PR-AUC {tp:.4f} | "
                  f"{time.time() - t0:.0f}s")
            save_gnn(ckpt_path, {"model": model.state_dict(), "opt": opt.state_dict(),
                                 "sched": sched.state_dict(), "epoch": epoch,
                                 "best_val": best_val, "best_state": best_state})
        if args.max_seconds and time.time() - t0 > args.max_seconds:
            print(f"  time budget reached at epoch {epoch}; "
                  f"checkpoint saved, rerun to continue")
            return

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(x, edge_index, edge_attr)).numpy()
    pr, roc = _metrics(y[test], scores[test])
    thr = threshold_at_alert_rate(scores[test], args.alert_rate_mult * y.mean())
    flag = scores[test] >= thr
    tp_ = int((flag & (y[test] == 1)).sum())
    print(f"[gnn/account] TEST PR-AUC {pr:.4f} | ROC-AUC {roc:.4f} | "
          f"precision {tp_ / max(flag.sum(), 1):.3f} | "
          f"recall {tp_ / max(y[test].sum(), 1):.3f}")

    save_gnn(models / "gnn_account_model.pt", {
        "state_dict": best_state, "in_dim": x.size(1), "hidden": args.hidden,
        "layers": args.layers, "dropout": args.dropout, "norm_stats": stats,
        "columns": columns,
        "threshold": float(thr), "test_index": test, "seed": args.seed,
        "metrics": {"pr_auc": float(pr), "roc_auc": float(roc)}})
    print(f"  -> saved {models / 'gnn_account_model.pt'}")


# ---------------------------------------------------------------------------
def _embeddings(data: Path, models: Path, args):
    bundle = load_gnn(models / "gnn_account_model.pt")
    feats, x, edge_index, edge_attr, _, _ = load_graph(
        data, columns=bundle.get("columns"), stats=bundle.get("norm_stats"))
    model = AMLGNN(bundle["in_dim"], hidden=bundle["hidden"],
                   layers=bundle["layers"], dropout=0.0)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    with torch.no_grad():
        emb = model.embed(x, edge_index, edge_attr)
    return feats, emb, bundle


def train_transaction(data: Path, models: Path, args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)

    feats, emb, node_bundle = _embeddings(data, models, args)
    emb = (emb - emb.mean(0)) / emb.std(0).clamp(min=1e-6)
    index = {a: i for i, a in enumerate(feats["account_id"])}

    txns = pd.read_csv(data / "transactions.csv")
    y = txns["is_laundering"].astype(int).to_numpy()
    Xt = transaction_feature_matrix(txns, feats)
    columns = list(Xt.columns)
    Xt = torch.from_numpy(np.nan_to_num(Xt.to_numpy(np.float32), posinf=0, neginf=0))
    Xt = (Xt - Xt.mean(0)) / Xt.std(0).clamp(min=1e-6)
    src = torch.from_numpy(txns["sender"].map(index).to_numpy(np.int64))
    dst = torch.from_numpy(txns["receiver"].map(index).to_numpy(np.int64))

    # Group split on episode_id so no episode straddles train/test.
    groups = txns["episode_id"].fillna("").astype(str)
    groups = np.where(groups == "", "T_" + txns["txn_id"].astype(str), groups)
    from sklearn.model_selection import GroupShuffleSplit
    tr_all, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3,
                                        random_state=args.seed
                                        ).split(Xt, y, groups))
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
                                    random_state=args.seed
                                    ).split(Xt[tr_all], y[tr_all], groups[tr_all]))
    tr, va = tr_all[tr], tr_all[va]

    # Negative subsampling keeps the epoch affordable on CPU; the positives are
    # all kept, and evaluation still runs on the untouched test set.
    rng = np.random.default_rng(args.seed)
    pos = tr[y[tr] == 1]
    neg = tr[y[tr] == 0]
    neg = rng.choice(neg, size=min(len(neg), args.neg_per_pos * len(pos)),
                     replace=False)
    fit_idx = np.concatenate([pos, neg])
    rng.shuffle(fit_idx)
    print(f"[gnn/transaction] {len(txns):,} transactions, {y.sum():,} laundering | "
          f"training on {len(fit_idx):,} rows ({len(pos):,} positive)")

    head = TransactionHead(Xt.size(1), emb.size(1), hidden=args.hidden * 2,
                           dropout=args.dropout)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    pos_weight = torch.tensor(len(neg) / max(len(pos), 1), dtype=torch.float32)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def score(idx: np.ndarray, batch: int = 65536) -> np.ndarray:
        head.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), batch):
                b = torch.from_numpy(idx[i:i + batch])
                out.append(torch.sigmoid(head(Xt[b], emb[src[b]], emb[dst[b]])).numpy())
        return np.concatenate(out)

    ckpt_path = models / "gnn_transaction_checkpoint.pt"
    start, best_val, best_state = 1, -1.0, None
    if args.resume and ckpt_path.exists():
        ck = load_gnn(ckpt_path)
        head.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start, best_val, best_state = ck["epoch"] + 1, ck["best_val"], ck["best_state"]
        print(f"[gnn/transaction] resumed at epoch {start} "
              f"(best val PR-AUC {best_val:.4f})")

    t0 = time.time()
    for epoch in range(start, args.txn_epochs + 1):
        head.train()
        perm = rng.permutation(len(fit_idx))
        running = 0.0
        for i in range(0, len(perm), args.batch):
            b = torch.from_numpy(fit_idx[perm[i:i + args.batch]])
            opt.zero_grad()
            loss = lossf(head(Xt[b], emb[src[b]], emb[dst[b]]), torch.from_numpy(
                y[b.numpy()].astype(np.float32)))
            loss.backward()
            opt.step()
            running += loss.item()
        vp, _ = _metrics(y[va], score(va))
        if vp > best_val:
            best_val = vp
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        print(f"  epoch {epoch:3d} | loss {running / max(1, len(perm) // args.batch):.4f}"
              f" | val PR-AUC {vp:.4f} | {time.time() - t0:.0f}s")
        save_gnn(ckpt_path, {"model": head.state_dict(), "opt": opt.state_dict(),
                             "epoch": epoch, "best_val": best_val,
                             "best_state": best_state})
        if args.max_seconds and time.time() - t0 > args.max_seconds:
            print("  time budget reached; checkpoint saved, rerun to continue")
            return

    head.load_state_dict(best_state)
    s_te = score(te)
    pr, roc = _metrics(y[te], s_te)
    thr = threshold_at_alert_rate(s_te, args.alert_rate_mult * y.mean())
    flag = s_te >= thr
    tp_ = int((flag & (y[te] == 1)).sum())
    print(f"[gnn/transaction] TEST PR-AUC {pr:.4f} | ROC-AUC {roc:.4f} | "
          f"precision {tp_ / max(flag.sum(), 1):.3f} | "
          f"recall {tp_ / max(y[te].sum(), 1):.3f}")

    save_gnn(models / "gnn_transaction_model.pt", {
        "state_dict": best_state, "txn_dim": Xt.size(1), "emb_dim": emb.size(1),
        "hidden": args.hidden * 2, "dropout": args.dropout, "columns": columns,
        "threshold": float(thr), "seed": args.seed,
        "metrics": {"pr_auc": float(pr), "roc_auc": float(roc)}})
    print(f"  -> saved {models / 'gnn_transaction_model.pt'}")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--models", default="models")
    ap.add_argument("--stage", choices=["both", "account", "transaction"],
                    default="both")
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--txn-epochs", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--drop-edge", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--neg-per-pos", type=int, default=40)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--alert-rate-mult", type=float, default=2.0)
    ap.add_argument("--threads", type=int, default=max(1, (__import__("os").cpu_count() or 2)))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="stop and checkpoint after this many seconds (0 = off)")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()

    data, models = Path(args.data), Path(args.models)
    models.mkdir(parents=True, exist_ok=True)
    if args.stage in ("both", "account"):
        train_account(data, models, args)
    if args.stage in ("both", "transaction"):
        train_transaction(data, models, args)


if __name__ == "__main__":
    main()
