"""
Loads generated + holdout JSONL files into a single pandas DataFrame for
training and evaluation.

Keeps the merge logic in one place so train_baseline.py and evaluate.py
(and any teammate's detector script) all see the same data consistently.
"""

import glob
import json
import os

import pandas as pd

GENERATED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "generated")

# Files considered "holdout" -- real-world/public data the detector should
# never be trained on, only evaluated against for generalization.
HOLDOUT_FILENAMES = {"holdout_sms.jsonl", "holdout_email.jsonl"}


def _load_jsonl(path: str) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_all(generated_dir: str = GENERATED_DIR):
    """Returns (train_pool_df, holdout_df). train_pool_df is everything NOT
    tagged as a holdout file (our own generated data, to be train/test split
    downstream); holdout_df is the public real-world data kept fully separate."""
    train_rows, holdout_rows = [], []

    for path in sorted(glob.glob(os.path.join(generated_dir, "*.jsonl"))):
        fname = os.path.basename(path)
        rows = _load_jsonl(path)
        if fname in HOLDOUT_FILENAMES:
            holdout_rows.extend(rows)
        else:
            train_rows.extend(rows)

    train_df = pd.DataFrame(train_rows) if train_rows else pd.DataFrame(
        columns=["id", "text", "label", "channel", "attack_subtype", "difficulty_tier"])
    holdout_df = pd.DataFrame(holdout_rows) if holdout_rows else pd.DataFrame(
        columns=["id", "text", "label", "channel", "attack_subtype", "difficulty_tier"])

    for df in (train_df, holdout_df):
        if "label" in df.columns:
            df["label"] = df["label"].astype(int)

    return train_df, holdout_df


def summarize(df: pd.DataFrame, name: str = "dataset"):
    if df.empty:
        print(f"{name}: empty")
        return
    print(f"{name}: {len(df)} rows | "
          f"fraud={int((df['label']==1).sum())} legit={int((df['label']==0).sum())}")
    if "attack_subtype" in df.columns:
        print(f"  by subtype: {df['attack_subtype'].value_counts().to_dict()}")
    if "difficulty_tier" in df.columns:
        print(f"  by difficulty: {df['difficulty_tier'].value_counts().to_dict()}")
    if "channel" in df.columns:
        print(f"  by channel: {df['channel'].value_counts().to_dict()}")


if __name__ == "__main__":
    train_df, holdout_df = load_all()
    summarize(train_df, "train pool (our generated data)")
    print()
    summarize(holdout_df, "holdout (public real-world data)")
