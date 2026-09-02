"""Score a generated dataset with the GNN and write the replay file the web view plays.

Run this once after `python run.py all`. It does the expensive part — a full-graph
pass of the account GNN, then the transaction head over every row of the ledger —
and flattens the result into `web/stream.json`: one chronological array of
transactions, each already carrying its laundering score.

    python aml-live/build_stream.py                       # scores data/, writes web/stream.json
    python aml-live/build_stream.py --data sample_data    # the small bundled run
    python aml-live/build_stream.py --max-events 30000    # trim a large ledger

Why precompute: the account GNN aggregates over a 3-hop neighbourhood, so a node's
embedding depends on edges that may not have arrived yet at any given point in a
replay. Scoring the finished graph once and revealing each score at its
transaction's own timestamp is both faster and more honest than pretending the
model re-runs per row. The web view never sees a score before its transaction
arrives.

Output schema (arrays, not objects — this file gets big):

    meta      thresholds, counts, model metrics, time span
    accounts  parallel arrays: id, archetype, city, kyc, score, flagged
    events    [t_offset_sec, sender_idx, receiver_idx, amount, channel_idx,
               score, pattern_idx, is_laundering, cross_border]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def score_with_gnn(data: Path, models: Path, threads: int):
    """Return (account_frame, account_scores, txn_scores, acct_thr, txn_thr)."""
    import torch

    from aml_models import account_feature_matrix, transaction_feature_matrix
    from gnn_model import (AMLGNN, TransactionHead, build_graph_tensors,
                           load_gnn, normalize_features)

    torch.set_num_threads(threads)

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
        acct_scores = torch.sigmoid(model.head(emb).squeeze(-1)).numpy()
    print(f"  accounts scored: {len(acct_scores):,}")

    tb = load_gnn(models / "gnn_transaction_model.pt")
    embn = (emb - emb.mean(0)) / emb.std(0).clamp(min=1e-6)
    index = {a: i for i, a in enumerate(feats["account_id"])}

    txns = pd.read_csv(data / "transactions.csv")
    Xt = transaction_feature_matrix(txns, feats, columns=tb["columns"])
    Xt = torch.from_numpy(np.nan_to_num(Xt.to_numpy(np.float32),
                                        posinf=0, neginf=0).copy())
    Xt = (Xt - Xt.mean(0)) / Xt.std(0).clamp(min=1e-6)
    src = torch.from_numpy(txns["sender"].map(index).fillna(0).to_numpy(np.int64))
    dst = torch.from_numpy(txns["receiver"].map(index).fillna(0).to_numpy(np.int64))

    head = TransactionHead(tb["txn_dim"], tb["emb_dim"], hidden=tb["hidden"],
                           dropout=0.0)
    head.load_state_dict(tb["state_dict"])
    head.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(txns), 65536):
            sl = slice(i, i + 65536)
            out.append(torch.sigmoid(head(Xt[sl], embn[src[sl]],
                                         embn[dst[sl]])).numpy())
    txn_scores = np.concatenate(out)
    print(f"  transactions scored: {len(txn_scores):,}")

    return (feats, acct_scores, txns, txn_scores,
            float(bundle["threshold"]), float(tb["threshold"]))


def metrics(y: np.ndarray, s: np.ndarray, thr: float) -> dict:
    """PR-AUC / ROC-AUC / precision / recall at the bundled threshold."""
    if y is None or y.sum() == 0:
        return {}
    from sklearn.metrics import average_precision_score, roc_auc_score
    flagged = s >= thr
    tp = int((flagged & (y == 1)).sum())
    return {
        "pr_auc": round(float(average_precision_score(y, s)), 4),
        "roc_auc": round(float(roc_auc_score(y, s)), 4),
        "precision": round(tp / max(int(flagged.sum()), 1), 4),
        "recall": round(tp / max(int(y.sum()), 1), 4),
        "flag_rate": round(float(flagged.mean()), 5),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data", help="dataset directory")
    ap.add_argument("--models", default="models", help="trained model bundles")
    ap.add_argument("--out", default=None, help="output json (default web/stream.json)")
    ap.add_argument("--max-events", type=int, default=60000,
                    help="keep the first N transactions chronologically")
    ap.add_argument("--threads", type=int,
                    default=max(1, (__import__("os").cpu_count() or 2)))
    args = ap.parse_args()

    data = Path(args.data)
    if not data.is_absolute():
        data = REPO / data
    models = Path(args.models)
    if not models.is_absolute():
        models = REPO / models
    out = Path(args.out) if args.out else Path(__file__).resolve().parent / "web" / "stream.json"

    for need in ("transactions.csv", "accounts.csv", "account_features.csv", "edges.csv"):
        if not (data / need).exists():
            raise SystemExit(f"missing {data / need} — run `python run.py all` first")

    print(f"scoring {data} with the GNN")
    feats, acct_scores, txns, txn_scores, acct_thr, txn_thr = score_with_gnn(
        data, models, args.threads)

    # ---- chronological order is what the replay needs -------------------
    txns["timestamp"] = pd.to_datetime(txns["timestamp"])
    txns["_score"] = txn_scores
    txns = txns.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if args.max_events and len(txns) > args.max_events:
        txns = txns.iloc[:args.max_events].copy()
        print(f"  trimmed to first {args.max_events:,} transactions")

    t0 = txns["timestamp"].iloc[0]
    offs = ((txns["timestamp"] - t0).dt.total_seconds()).to_numpy(np.int64)

    # ---- only ship accounts that actually appear ------------------------
    used = pd.unique(pd.concat([txns["sender"], txns["receiver"]], ignore_index=True))
    used_set = set(used)
    order = {a: i for i, a in enumerate(used)}

    accounts = pd.read_csv(data / "accounts.csv")
    accounts = accounts[accounts["account_id"].isin(used_set)]
    ascore = dict(zip(feats["account_id"], acct_scores))
    accounts = accounts.set_index("account_id").reindex(used)

    archetypes = sorted(accounts["archetype"].dropna().unique().tolist())
    arch_idx = {a: i for i, a in enumerate(archetypes)}
    cities = sorted(accounts["city"].dropna().unique().tolist())
    city_idx = {c: i for i, c in enumerate(cities)}
    kyc_levels = sorted(accounts["kyc_level"].dropna().unique().tolist())
    kyc_idx = {k: i for i, k in enumerate(kyc_levels)}

    channels = sorted(txns["channel"].dropna().unique().tolist())
    chan_idx = {c: i for i, c in enumerate(channels)}
    patterns = sorted(txns["pattern"].fillna("normal").unique().tolist())
    pat_idx = {p: i for i, p in enumerate(patterns)}

    a_scores = np.array([ascore.get(a, 0.0) for a in used], dtype=np.float64)

    # ---- risk tiers -----------------------------------------------------
    # Alert threshold comes from the trained bundle. The watch tier is the band
    # just below it: scores high enough to notice, not high enough to queue.
    alert_rate = float((txn_scores >= txn_thr).mean())
    watch_thr = float(np.quantile(txn_scores, max(0.0, 1.0 - 3.0 * alert_rate)))
    watch_thr = min(watch_thr, txn_thr * 0.999)

    truth = (txns["is_laundering"].to_numpy(np.int64)
             if "is_laundering" in txns.columns else None)

    # ---- risk bar scale -------------------------------------------------
    # Raw scores are violently bimodal — the median row sits at 1e-4 and a
    # flagged row at 0.999 — so a linear bar is all-empty or all-full and shows
    # nothing in between. Map log(score) instead, pinned so the alert threshold
    # lands at exactly 500/1000. The midpoint of every bar is the alert line,
    # whatever threshold this particular run was calibrated to.
    floor = 1e-6
    ls = np.log10(np.maximum(txns["_score"].to_numpy(np.float64), floor))
    lt, lf = np.log10(txn_thr), np.log10(floor)
    below = 500.0 * (ls - lf) / max(lt - lf, 1e-9)
    above = 500.0 + 500.0 * (ls - lt) / max(0.0 - lt, 1e-9)
    risk = np.clip(np.where(ls >= lt, above, below), 0, 1000).astype(np.int64)

    pat_family = {}
    ep_path = data / "episodes.csv"
    if ep_path.exists():
        eps = pd.read_csv(ep_path)
        for pat, fam in eps.drop_duplicates("pattern")[["pattern", "family"]].values:
            pat_family[pat] = ("laundering" if fam == "laundering"
                               else "lookalike")
    families = [pat_family.get(p, "normal") for p in patterns]

    # txn_id is "T000123456" — ship the integer and rebuild the label in the UI
    tids = txns["txn_id"].str.extract(r"(\d+)", expand=False).fillna("0").astype(np.int64)

    events = [
        [int(o), int(ti), int(order[s]), int(order[r]), round(float(a), 2),
         int(chan_idx[c]), round(float(sc), 6), int(rk), int(pat_idx[p]),
         int(t), int(x)]
        for o, ti, s, r, a, c, sc, rk, p, t, x in zip(
            offs, tids, txns["sender"], txns["receiver"], txns["amount"],
            txns["channel"], txns["_score"], risk,
            txns["pattern"].fillna("normal"),
            truth if truth is not None else np.zeros(len(txns), np.int64),
            txns["cross_border"].fillna(0).to_numpy(np.int64))
    ]

    payload = {
        "meta": {
            "source": str(data.name),
            "generated_from": "gnn_account_model.pt + gnn_transaction_model.pt",
            "start": t0.isoformat(),
            "end": txns["timestamp"].iloc[-1].isoformat(),
            "span_seconds": int(offs[-1]),
            "n_events": len(events),
            "n_accounts": len(used),
            "alert_threshold": round(txn_thr, 6),
            "watch_threshold": round(watch_thr, 6),
            "account_threshold": round(acct_thr, 6),
            "has_labels": truth is not None,
            "txn_metrics": metrics(truth, txns["_score"].to_numpy(), txn_thr),
            "archetypes": archetypes,
            "cities": cities,
            "kyc_levels": kyc_levels,
            "channels": channels,
            "patterns": patterns,
            "families": families,
            "fields": ["t", "tid", "src", "dst", "amount", "channel",
                       "score", "risk", "pattern", "truth", "cross_border"],
        },
        "accounts": {
            "id": used.tolist(),
            "archetype": [arch_idx.get(v, 0) for v in accounts["archetype"].fillna(archetypes[0])],
            "city": [city_idx.get(v, 0) for v in accounts["city"].fillna(cities[0])],
            "kyc": [kyc_idx.get(v, 0) for v in accounts["kyc_level"].fillna(kyc_levels[0])],
            "business": [int(bool(v)) for v in accounts["is_business"].fillna(False)],
            "score": [round(float(v), 4) for v in a_scores],
            "flagged": [int(v >= acct_thr) for v in a_scores],
        },
        "events": events,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        json.dump(payload, fh, separators=(",", ":"))

    mb = out.stat().st_size / 1e6
    m = payload["meta"]["txn_metrics"]
    print(f"\nwrote {out}  ({mb:.1f} MB)")
    print(f"  {len(events):,} transactions | {len(used):,} accounts | "
          f"span {offs[-1] / 86400:.1f} days")
    print(f"  alert threshold {txn_thr:.4f} -> "
          f"{int((txns['_score'] >= txn_thr).sum()):,} flagged")
    if m:
        print(f"  PR-AUC {m['pr_auc']} | ROC-AUC {m['roc_auc']} | "
              f"precision {m['precision']} | recall {m['recall']}")
    print("\nnow run:  python aml-live/serve.py")


if __name__ == "__main__":
    main()
