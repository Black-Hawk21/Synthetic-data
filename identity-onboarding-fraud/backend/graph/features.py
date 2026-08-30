"""Structural graph features computed WITHOUT touching the fraud label
(section 20: "Graph features must be computed without using the target
label")."""
from __future__ import annotations

import networkx as nx
import pandas as pd

from backend.graph.builder import build_graph, find_connected_components

ATTR_COLS = {
    "phone": "phone", "email": "email", "address": "address",
    "device_id": "device_id", "ip_id": "ip_id",
}


def calculate_graph_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the identity graph from `df` and annotate reuse / graph
    features back onto a copy of the dataframe."""
    df = df.copy()
    graph = build_graph(df)

    # Reuse counts: simple groupby value_counts is equivalent to node degree
    # for attribute nodes, but computing via groupby is faster than walking
    # the graph per-row for large N.
    for col, out_prefix in [
        ("phone", "phone_reuse_count"), ("email", "email_reuse_count"),
        ("address", "address_reuse_count"), ("device_id", "device_reuse_count"),
        ("ip_id", "ip_reuse_count"),
    ]:
        counts = df[col].value_counts()
        df[out_prefix] = df[col].map(counts).fillna(1).astype(int) - 1  # "reuse" = others sharing it

    df["identity_reuse_count"] = df["phone_reuse_count"]  # bureau-identity proxy
    df["identities_from_ip"] = df["ip_id"].map(df["ip_id"].value_counts()).astype(int)
    df["applications_from_ip"] = df["identities_from_ip"]
    df["device_identity_count"] = df["device_id"].map(df["device_id"].value_counts()).astype(int)
    df["face_reuse_count"] = df["face_hash"].map(df["face_hash"].value_counts()).fillna(1).astype(int) - 1

    # Connected component size (graph-theoretic identity cluster)
    components = find_connected_components(graph)
    comp_size_by_applicant: dict[str, int] = {}
    for comp in components:
        size = sum(1 for node in comp if node.startswith("person:"))
        for node in comp:
            if node.startswith("person:"):
                comp_size_by_applicant[node.split(":", 1)[1]] = size
    df["connected_component_size"] = df["applicant_id"].map(comp_size_by_applicant).fillna(1).astype(int)
    df["identity_cluster_size"] = df["connected_component_size"]

    df["shared_device_count"] = df["device_reuse_count"]
    df["shared_ip_count"] = df["ip_reuse_count"]
    df["shared_phone_count"] = df["phone_reuse_count"]
    df["shared_address_count"] = df["address_reuse_count"]

    df["graph_degree"] = df["applicant_id"].apply(
        lambda aid: graph.degree[f"person:{aid}"] if f"person:{aid}" in graph else 0
    )

    df["suspicious_cluster_score"] = _suspicious_cluster_score(df)
    return df


def _suspicious_cluster_score(df: pd.DataFrame) -> pd.Series:
    """Transparent, documented formula (section 22/23 style transparency):

    score = 0.35 * norm(cluster_size) + 0.25 * norm(device_reuse)
          + 0.20 * norm(ip_reuse) + 0.20 * (num_distinct_reused_signals / 4)

    where norm(x) = min(1, x / 20). Purely structural -- no label used.
    """
    def norm(s: pd.Series, cap: float = 20.0) -> pd.Series:
        return (s.clip(lower=0) / cap).clip(upper=1.0)

    distinct_signals = (
        (df["device_reuse_count"] > 0).astype(int)
        + (df["ip_reuse_count"] > 0).astype(int)
        + (df["phone_reuse_count"] > 0).astype(int)
        + (df["email_reuse_count"] > 0).astype(int)
    ) / 4.0

    score = (
        0.35 * norm(df["connected_component_size"] - 1)
        + 0.25 * norm(df["device_reuse_count"])
        + 0.20 * norm(df["ip_reuse_count"])
        + 0.20 * distinct_signals
    )
    return score.clip(0, 1)


def detect_suspicious_clusters(df: pd.DataFrame, min_size: int = 3, score_threshold: float = 0.35) -> list[dict]:
    """Group applicants by connected component and flag clusters that look
    like coordinated infrastructure (used for the fraud-ring view)."""
    graph = build_graph(df)
    components = find_connected_components(graph)
    clusters = []
    for comp in components:
        applicant_ids = [n.split(":", 1)[1] for n in comp if n.startswith("person:")]
        if len(applicant_ids) < min_size:
            continue
        sub = df[df["applicant_id"].isin(applicant_ids)]
        avg_score = float(sub["suspicious_cluster_score"].mean()) if len(sub) else 0.0
        if avg_score < score_threshold:
            continue
        shared_devices = sub["device_id"].nunique()
        shared_ips = sub["ip_id"].nunique()
        clusters.append({
            "cluster_id": f"cluster_{hash(frozenset(applicant_ids)) & 0xffffffff:x}",
            "size": len(applicant_ids),
            "applicant_ids": applicant_ids[:100],
            "avg_suspicious_score": round(avg_score, 4),
            "distinct_devices": int(shared_devices),
            "distinct_ips": int(shared_ips),
            "fraud_rate": float(sub["is_fraud"].mean()) if "is_fraud" in sub else None,
        })
    clusters.sort(key=lambda c: c["avg_suspicious_score"], reverse=True)
    return clusters
