#!/usr/bin/env python3
"""Evaluate the current model version on a dataset, with a per-attack-type
breakdown (section 15).

Usage:
    python scripts/evaluate_model.py --data data/synthetic/dataset.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from backend.blue_team.evaluate import compute_metrics, per_attack_type_recall
from backend.blue_team.predict import ModelBundle, score_dataframe
from backend.config import settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=str(settings.synthetic_dir / "dataset.csv"))
    parser.add_argument("--version", type=int, default=None)
    args = parser.parse_args()

    bundle = ModelBundle.load(version=args.version)
    if bundle is None:
        print("No trained model found. Run scripts/train_model.py first.")
        sys.exit(1)

    df = pd.read_csv(args.data, low_memory=False)
    probs = score_dataframe(df, bundle)
    metrics = compute_metrics(df["is_fraud"].values, probs)
    per_attack = per_attack_type_recall(df, probs)

    print(f"Model version: {bundle.version}\n")
    print(json.dumps(metrics, indent=2))
    print("\nPer-attack-type recall:")
    for attack_type, stats in sorted(per_attack.items(), key=lambda kv: -kv[1]["recall"]):
        print(f"  {attack_type:35s} recall={stats['recall']:.2%}  n={stats['n_samples']}")


if __name__ == "__main__":
    main()
