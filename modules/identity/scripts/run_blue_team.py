#!/usr/bin/env python3
"""Convenience wrapper: train + evaluate in one command (section 15).

Usage:
    python scripts/run_blue_team.py --data data/synthetic/dataset.csv
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    args = sys.argv[1:]
    subprocess.run([sys.executable, str(ROOT / "train_model.py"), *args], check=True)
    subprocess.run([sys.executable, str(ROOT / "evaluate_model.py"), *args], check=True)


if __name__ == "__main__":
    main()
