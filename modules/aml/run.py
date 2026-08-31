#!/usr/bin/env python3
"""Command-line entry point for the synthetic AML dataset generator.

    python run.py all                        # generate -> features -> graph -> train
    python run.py generate --accounts 20000 --days 120 --difficulty 0.8
    python run.py features
    python run.py graph --episode L000007
    python run.py train
    python run.py redteam --levels 0,0.25,0.5,0.75,1.0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from amlgen.config import load_config
from amlgen.evaluation.metrics import (account_report, episode_detection,
                                       missed_episodes, print_report,
                                       recall_by_pattern)
from amlgen.export import write_manifest, write_tables
from amlgen.features import build_feature_table
from amlgen.graphs import build_edge_table, build_graph, write_graphml
from amlgen.models.baseline import feature_importance, train_detector
from amlgen.simulate import simulate

ROOT = Path(__file__).resolve().parent


def _cfg(args) -> dict:
    return load_config(
        args.config,
        **{"population.n_accounts": args.accounts,
           "simulation.days": args.days,
           "laundering.difficulty": args.difficulty,
           "seed": args.seed,
           "output.dir": args.out})


def _load(data_dir: Path, name: str) -> pd.DataFrame:
    for ext, reader in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
        p = data_dir / f"{name}{ext}"
        if p.exists():
            return reader(p)
    raise FileNotFoundError(f"{name} not found in {data_dir} - run `python run.py generate` first")


# --------------------------------------------------------------------- steps
def cmd_generate(args) -> dict:
    cfg = _cfg(args)
    out = Path(cfg["output"]["dir"])
    res = simulate(cfg, verbose=True)

    print("      building transaction graph")
    edges = build_edge_table(res.transactions)
    tables = {
        "transactions": res.transactions,
        "accounts": res.accounts,
        "episodes": res.episodes,
        "episode_members": res.episode_members,
        "edges": edges,
    }
    print("      writing tables")
    write_tables(tables, out, formats=tuple(cfg["output"]["formats"]))
    if cfg["output"].get("write_graphml", True):
        G = build_graph(edges, res.accounts)
        write_graphml(G, out / "transaction_graph.graphml")
        print(f"      {out / 'transaction_graph.graphml'}  "
              f"({G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges)")
    write_manifest(cfg, tables, out)
    return {"config": cfg, **tables}


def cmd_features(args) -> pd.DataFrame:
    cfg = _cfg(args)
    data = Path(cfg["output"]["dir"])
    txns = _load(data, "transactions")
    txns["timestamp"] = pd.to_datetime(txns["timestamp"])
    accounts = _load(data, "accounts")
    print(f"[features] {len(txns):,} transactions, {len(accounts):,} accounts")
    edges = build_edge_table(txns)
    feats = build_feature_table(txns, accounts, cfg["simulation"]["reporting_threshold"], edges)
    write_tables({"account_features": feats}, data, formats=tuple(cfg["output"]["formats"]))
    return feats


def cmd_graph(args) -> None:
    cfg = _cfg(args)
    data = Path(cfg["output"]["dir"])
    txns = _load(data, "transactions")
    txns["timestamp"] = pd.to_datetime(txns["timestamp"])
    members = _load(data, "episode_members")
    from amlgen.viz import plot_amount_overlap, plot_episode
    figs = data / "figures"
    episodes = _load(data, "episodes")
    episode = args.episode
    if episode is None:
        laundering = episodes[episodes["is_laundering"] == 1]
        episode = laundering.sort_values("n_transactions").iloc[len(laundering) // 2]["episode_id"]
    row = episodes[episodes["episode_id"] == episode]
    label = f"({row.iloc[0]['pattern']})" if len(row) else ""
    print(f"[graph] plotting episode {episode} {label}")
    print("       ", plot_episode(txns, members, episode, figs / f"episode_{episode}.png",
                                  subtitle=label))
    print("       ", plot_amount_overlap(txns, figs / "amount_overlap.png"))


def _alert_rate(args, feats: pd.DataFrame) -> float:
    if args.alert_rate is not None:
        return args.alert_rate
    base = float(feats["is_laundering"].mean())
    return float(min(max(2.0 * base, 0.01), 0.25))


def cmd_train(args) -> dict:
    cfg = _cfg(args)
    data = Path(cfg["output"]["dir"])
    try:
        feats = _load(data, "account_features")
    except FileNotFoundError:
        feats = cmd_features(args)
    alert_rate = _alert_rate(args, feats)
    print(f"[train] {len(feats):,} accounts, "
          f"{int(feats['is_laundering'].sum()):,} laundering")
    fit = train_detector(feats, seed=cfg["seed"] % 2 ** 31)
    test, scores = fit["test_features"], fit["test_scores"]
    report = account_report(test, scores, alert_rate=alert_rate)
    by_pattern = recall_by_pattern(test, scores, alert_rate=alert_rate)

    episodes = _load(data, "episodes")
    members = _load(data, "episode_members")
    by_episode = episode_detection(episodes, members, test, scores, alert_rate)
    print_report(report, by_pattern, by_episode)

    imp = feature_importance(fit)
    print("\n--- top features ---")
    print(imp.head(15).to_string(index=False))
    missed = missed_episodes(episodes, members, test, scores, alert_rate)
    if len(missed):
        print("\n--- largest missed episodes (Red Team: study these) ---")
        print(missed.head(10).to_string(index=False))

    from amlgen.viz import plot_holding_times, plot_pattern_recall
    figs = data / "figures"
    plot_pattern_recall(by_pattern, figs / "recall_by_pattern.png")
    plot_holding_times(feats, figs / "holding_times.png")
    write_tables({"evaluation_by_pattern": by_pattern,
                  "evaluation_by_episode": by_episode,
                  "feature_importance": imp,
                  "missed_episodes": missed}, data / "evaluation", formats=("csv",))
    (data / "evaluation").mkdir(parents=True, exist_ok=True)
    (data / "evaluation" / "report.json").write_text(json.dumps(report, indent=2))
    return report


def cmd_redteam(args) -> None:
    """Sweep the difficulty knob and watch recall degrade - the core experiment."""
    levels = [float(x) for x in args.levels.split(",")]
    rows = []
    for level in levels:
        print(f"\n{'=' * 70}\nDIFFICULTY {level}\n{'=' * 70}")
        args.difficulty = level
        args.out = str(Path(args.out_root) / f"difficulty_{level:g}")
        cmd_generate(args)
        cmd_features(args)
        report = cmd_train(args)
        rows.append({"difficulty": level, **report})
    df = pd.DataFrame(rows)
    out = Path(args.out_root) / "redteam_sweep.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n=== Red Team sweep ===\n{df[['difficulty', 'pr_auc', 'recall', 'precision']].to_string(index=False)}")
    print(f"\nwritten: {out}")


def cmd_all(args) -> None:
    cmd_generate(args)
    cmd_features(args)
    cmd_graph(args)
    cmd_train(args)


# ---------------------------------------------------------------------- main
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["all", "generate", "features", "graph",
                                       "train", "redteam"])
    p.add_argument("--config", default=str(ROOT / "config.yaml"))
    p.add_argument("--out", default=None, help="output directory (default: from config)")
    p.add_argument("--accounts", type=int, default=None)
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--difficulty", type=float, default=None, help="0 = blatant, 1 = subtle")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--episode", default=None, help="episode id to plot")
    p.add_argument("--alert-rate", type=float, default=None, dest="alert_rate",
                   help="share of accounts an analyst can review "
                        "(default: 2x the dataset's positive rate)")
    p.add_argument("--levels", default="0,0.25,0.5,0.75,1.0", help="redteam difficulty sweep")
    p.add_argument("--out-root", default="data/redteam", dest="out_root")
    args = p.parse_args(argv)
    if args.out is None:
        args.out = load_config(args.config)["output"]["dir"]
    {"all": cmd_all, "generate": cmd_generate, "features": cmd_features,
     "graph": cmd_graph, "train": cmd_train, "redteam": cmd_redteam}[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
