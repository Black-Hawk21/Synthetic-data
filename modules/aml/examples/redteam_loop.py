#!/usr/bin/env python3
"""The Red Team / Blue Team loop as an editable script.

Sweeps the difficulty knob, trains the baseline at each level, and reports how
recall degrades per typology. Run from the project root:

    python examples/redteam_loop.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amlgen.config import load_config
from amlgen.evaluation.metrics import account_report, recall_by_pattern
from amlgen.features import build_feature_table
from amlgen.models.baseline import train_detector
from amlgen.simulate import simulate

LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
ACCOUNTS = 6000
DAYS = 60


def run_level(difficulty: float) -> tuple[dict, pd.DataFrame]:
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    cfg["laundering"]["difficulty"] = difficulty
    cfg["population"]["n_accounts"] = ACCOUNTS
    cfg["simulation"]["days"] = DAYS

    res = simulate(cfg, verbose=False)
    feats = build_feature_table(res.transactions, res.accounts,
                               cfg["simulation"]["reporting_threshold"])
    fit = train_detector(feats, seed=0)
    test, scores = fit["test_features"], fit["test_scores"]
    alert_rate = min(max(2.0 * float(feats["is_laundering"].mean()), 0.01), 0.25)
    report = account_report(test, scores, alert_rate=alert_rate)
    by_pattern = recall_by_pattern(test, scores, alert_rate=alert_rate)
    return report, by_pattern


def main() -> None:
    summary, per_pattern = [], {}
    for level in LEVELS:
        print(f"difficulty {level} ...", flush=True)
        report, by_pattern = run_level(level)
        summary.append({"difficulty": level, "pr_auc": report["pr_auc"],
                        "recall": report["recall"], "precision": report["precision"],
                        "fp_on_lookalikes": report["lookalike_share_of_fp"]})
        per_pattern[level] = by_pattern.set_index("pattern")["recall"]

    print("\n=== overall ===")
    print(pd.DataFrame(summary).to_string(index=False))
    print("\n=== recall by typology across difficulty ===")
    print(pd.DataFrame(per_pattern).round(3).to_string())
    print("\nTypologies whose recall collapses fastest are where the Blue Team's "
          "features are weakest - start there.")


if __name__ == "__main__":
    main()
