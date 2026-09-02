#!/usr/bin/env python3
"""Generate a synthetic onboarding dataset (legit population + a spread of
attacks across all registered types/difficulties) and write it to
data/synthetic/dataset.csv.

Usage:
    python scripts/generate_data.py --n 100000
    python scripts/generate_data.py --n 10000 --attack-fraction 0.25
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from backend.config import settings
from backend.data.generator import generate_legitimate_applicants
from backend.graph.features import calculate_graph_features
from backend.red_team.registry import get_attack, list_attacks, load_all


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic onboarding dataset")
    parser.add_argument("--n", type=int, default=100_000, help="Total number of legitimate applicants to generate")
    parser.add_argument("--attack-fraction", type=float, default=0.25, help="Fraction of --n to generate as fraud, spread across all attack types/difficulties")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=str(settings.synthetic_dir / "dataset.csv"))
    args = parser.parse_args()

    load_all()
    rng = np.random.default_rng(args.seed)

    print(f"[1/3] Generating {args.n:,} legitimate applicants...")
    legit = generate_legitimate_applicants(args.n, seed=args.seed)

    n_fraud_total = int(args.n * args.attack_fraction)
    attack_types = list_attacks()
    n_per_type = max(10, n_fraud_total // (len(attack_types) * 3))  # 3 difficulty tiers
    print(f"[2/3] Generating ~{n_fraud_total:,} fraud records across {len(attack_types)} attack types x 3 difficulty tiers ({n_per_type} each)...")

    batches = [legit]
    for attack_type in attack_types:
        for difficulty in (0.25, 0.55, 0.85):
            batch = get_attack(attack_type).generate(n_per_type, difficulty=difficulty, seed=int(rng.integers(0, 2_000_000_000)))
            batches.append(batch)

    full = pd.concat(batches, ignore_index=True)
    print(f"[3/3] Computing identity graph features over {len(full):,} rows...")
    full = calculate_graph_features(full)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out_path, index=False)

    print(f"\nDone. {len(full):,} rows written to {out_path}")
    print(f"Fraud rate: {full['is_fraud'].mean():.2%}")
    print(full["attack_type"].value_counts())


if __name__ == "__main__":
    main()
