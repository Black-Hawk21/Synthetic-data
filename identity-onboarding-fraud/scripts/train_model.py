#!/usr/bin/env python3
"""Train the blue-team detector on a generated dataset.

Usage:
    python scripts/train_model.py --data data/synthetic/dataset.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from backend.blue_team.train import train_and_evaluate
from backend.config import settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=str(settings.synthetic_dir / "dataset.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.data, low_memory=False)
    print(f"Loaded {len(df):,} rows from {args.data}")

    result = train_and_evaluate(df, seed=args.seed)
    print(f"\nSaved model_v{result['version']} to {result['version_dir']}")
    print(json.dumps(result["final_model_metrics"], indent=2))


if __name__ == "__main__":
    main()
