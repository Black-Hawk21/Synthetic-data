#!/usr/bin/env python3
"""Standalone closed-loop CLI (no server required): generate an initial
dataset, train, then run N adversarial feedback iterations, printing a
before/after recall/precision/F1 table (section 14/25/31).

Usage:
    python scripts/run_closed_loop.py --n 20000 --iterations 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from backend.blue_team.predict import ModelBundle
from backend.data.generator import generate_legitimate_applicants
from backend.feedback.engine import run_closed_loop_iteration
from backend.graph.features import calculate_graph_features
from backend.red_team.registry import get_attack, list_attacks, load_all


def build_initial_dataset(n: int, seed: int) -> pd.DataFrame:
    load_all()
    rng = np.random.default_rng(seed)
    legit = generate_legitimate_applicants(n, seed=seed)
    batches = [legit]
    n_per_type = max(20, int(n * 0.2) // len(list_attacks()))
    for attack_type in list_attacks():
        batches.append(get_attack(attack_type).generate(n_per_type, difficulty=float(rng.uniform(0.2, 0.7)), seed=int(rng.integers(0, 2_000_000_000))))
    return calculate_graph_features(pd.concat(batches, ignore_index=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20_000, help="Number of legitimate applicants to seed with")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--n-per-type", type=int, default=200, help="New hard-attack records generated per iteration")
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Building initial dataset (n={args.n:,})...")
    df = build_initial_dataset(args.n, args.seed)
    print(f"Dataset: {len(df):,} rows, fraud rate {df['is_fraud'].mean():.2%}\n")

    print("=== Iteration 1: initial training ===")
    result = run_closed_loop_iteration(df, None, seed=args.seed)
    bundle = ModelBundle.load(version=result["model_version"])
    print(f"model_v{result['model_version']}  recall={result['after_metrics']['recall']:.2%}  "
          f"precision={result['after_metrics']['precision']:.2%}  f1={result['after_metrics']['f1']:.2%}\n")

    rows = [("1", None, result["after_metrics"])]
    for i in range(2, args.iterations + 1):
        print(f"=== Iteration {i}: evaluate -> find weaknesses -> harden -> retrain ===")
        result = run_closed_loop_iteration(df, bundle, n_per_type=args.n_per_type, use_llm=args.use_llm, seed=args.seed + i)
        if result.get("weakness_report"):
            print(result["weakness_report"]["summary"])
        print(f"before recall={result['before_metrics']['recall']:.2%}  ->  after recall={result['after_metrics']['recall']:.2%}")
        print(f"model_v{result['model_version']}  new hard attacks: {result['new_attack_count']}\n")
        rows.append((str(i), result["before_metrics"], result["after_metrics"]))
        df = result["augmented_df"]
        bundle = ModelBundle.load(version=result["model_version"])

    print("\n=== Summary ===")
    print(f"{'Iter':<6}{'Before Recall':<16}{'After Recall':<16}{'After Precision':<18}{'After F1':<10}")
    for it, before, after in rows:
        b = f"{before['recall']:.2%}" if before else "--"
        print(f"{it:<6}{b:<16}{after['recall']:.2%}{'':<8}{after['precision']:.2%}{'':<10}{after['f1']:.2%}")


if __name__ == "__main__":
    main()
