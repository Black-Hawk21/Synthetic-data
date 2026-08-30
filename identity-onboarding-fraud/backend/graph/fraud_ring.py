"""High-level fraud-ring summary utilities built on top of graph.features."""
from __future__ import annotations

import pandas as pd

from backend.graph.features import detect_suspicious_clusters


def summarize_fraud_rings(df: pd.DataFrame, min_size: int = 3) -> dict:
    clusters = detect_suspicious_clusters(df, min_size=min_size)
    total_flagged_identities = sum(c["size"] for c in clusters)
    return {
        "num_clusters": len(clusters),
        "total_identities_in_clusters": total_flagged_identities,
        "clusters": clusters[:50],
    }
