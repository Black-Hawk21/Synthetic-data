#!/usr/bin/env python3
"""Standalone red-team CLI: generate a single attack batch and print its
summary (no server required).

Usage:
    python scripts/run_red_team.py --attack-type FRAUD_RING --difficulty 0.7 --n 500
    python scripts/run_red_team.py --list
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.red_team.registry import all_attack_meta, get_attack, load_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack-type", type=str, default="FRAUD_RING")
    parser.add_argument("--difficulty", type=float, default=0.5)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--list", action="store_true", help="List all attack types and exit")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    load_all()

    if args.list:
        for meta in all_attack_meta():
            print(f"{meta['attack_type']:35s} [{meta['severity']:8s}] {meta['description']}")
        return

    strategy = get_attack(args.attack_type)
    batch = strategy.generate(args.n, difficulty=args.difficulty, seed=args.seed)
    print(f"Generated {len(batch)} records for {args.attack_type} (difficulty={args.difficulty})")
    print(strategy.description())
    if args.out:
        batch.to_csv(args.out, index=False)
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
