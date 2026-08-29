"""Writes the dataset tables to disk."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_tables(tables: dict, out_dir, formats=("csv",), verbose=True) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, df in tables.items():
        if df is None or len(df) == 0:
            continue
        for fmt in formats:
            path = out / f"{name}.{fmt}"
            if fmt == "csv":
                df.to_csv(path, index=False)
            elif fmt == "parquet":
                df.to_parquet(path, index=False)
            else:
                raise ValueError(f"unsupported format: {fmt}")
            written[f"{name}.{fmt}"] = path
            if verbose:
                size = path.stat().st_size / 1e6
                print(f"      {path}  ({len(df):,} rows, {size:.1f} MB)")
    return written


def write_manifest(cfg: dict, tables: dict, out_dir) -> Path:
    """A run manifest so every dataset version is traceable to its parameters."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": cfg,
        "row_counts": {k: int(len(v)) for k, v in tables.items() if v is not None},
    }
    txns = tables.get("transactions")
    if txns is not None and len(txns):
        manifest["label_balance"] = {
            "laundering_transactions": int(txns["is_laundering"].sum()),
            "total_transactions": int(len(txns)),
            "positive_rate": round(float(txns["is_laundering"].mean()), 6),
        }
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path
