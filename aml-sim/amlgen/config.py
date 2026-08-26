"""Configuration loading and validation."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULTS: Dict[str, Any] = {
    "seed": 42,
    "simulation": {
        "start_date": "2025-01-01",
        "days": 90,
        "currency": "INR",
        "reporting_threshold": 200_000,
    },
    "population": {
        "n_accounts": 10000,
        "foreign_share": 0.02,
        "dormant_share": 0.04,
        "archetype_mix": {
            "salary": 0.40, "student": 0.10, "freelancer": 0.08, "household": 0.10,
            "small_business": 0.18, "merchant": 0.08, "large_business": 0.04,
            "investment": 0.02,
        },
    },
    "normal_activity": {
        "p_preferred_counterparty": 0.72,
        "n_preferred_range": [3, 25],
        "same_city_bias": 3.0,
        "salary_enabled": True,
        "activity_jitter": 0.35,
    },
    "laundering": {
        "difficulty": 0.5,
        "scale_with_population": True,
        "population_reference": 10000,
        "network": {"mule_pool_share": 0.10, "shell_pool_share": 0.08,
                    "source_pool_share": 0.06},
        "episodes": {
            "layering_chain": 50, "circular_flow": 25, "rapid_pass_through": 40,
            "smurfing": 22, "fan_out": 30, "fan_in": 30, "mule_network": 20,
            "dormant_activation": 30,
        },
    },
    "benign_lookalikes": {
        "enabled": True,
        "episodes": {
            "supplier_passthrough": 140, "payroll_fanout": 60, "marketplace_fanin": 60,
            "treasury_cycle": 40, "installment_split": 90,
        },
    },
    "output": {"dir": "data", "formats": ["csv"], "write_graphml": True},
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None, **overrides) -> dict:
    """Load config.yaml (if present) on top of DEFAULTS, then apply kwargs.

    Keyword overrides use dotted paths, e.g. load_config(path, **{"laundering.difficulty": 0.9}).
    """
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        p = Path(path)
        if p.exists():
            with p.open("r", encoding="utf-8") as fh:
                cfg = _deep_merge(cfg, yaml.safe_load(fh) or {})
    for dotted, value in overrides.items():
        if value is None:
            continue
        node = cfg
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    _validate(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    mix = cfg["population"]["archetype_mix"]
    total = sum(mix.values())
    if abs(total - 1.0) > 1e-6:
        for key in mix:
            mix[key] /= total
    d = cfg["laundering"]["difficulty"]
    if not 0.0 <= d <= 1.0:
        raise ValueError("laundering.difficulty must be in [0, 1]")
    if cfg["simulation"]["days"] < 7:
        raise ValueError("simulation.days must be >= 7 for temporal patterns to make sense")
