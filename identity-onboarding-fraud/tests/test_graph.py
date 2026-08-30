from __future__ import annotations

import pandas as pd

from backend.graph.builder import build_graph, find_connected_components, find_shared_attributes
from backend.graph.features import calculate_graph_features, detect_suspicious_clusters
from backend.red_team.registry import get_attack


def test_build_graph_creates_person_nodes(small_legit_df):
    graph = build_graph(small_legit_df.head(20))
    person_nodes = [n for n in graph.nodes if n.startswith("person:")]
    assert len(person_nodes) == 20


def test_shared_device_creates_edge_between_applicants():
    strategy = get_attack("DEVICE_REUSE")
    df = strategy.generate(20, difficulty=0.05, seed=1)
    graph = build_graph(df)
    device_nodes = [n for n, d in graph.nodes(data=True) if d.get("kind") == "device"]
    assert any(graph.degree[n] > 1 for n in device_nodes), "expected at least one device shared by >1 applicant"


def test_calculate_graph_features_adds_reuse_counts(small_legit_df):
    out = calculate_graph_features(small_legit_df)
    for col in ["device_reuse_count", "ip_reuse_count", "connected_component_size", "suspicious_cluster_score"]:
        assert col in out.columns
    assert out["suspicious_cluster_score"].between(0, 1).all()


def test_graph_features_do_not_use_label(small_legit_df):
    """Graph features must be computable even with the label column dropped
    (section 20: no label leakage)."""
    df_no_label = small_legit_df.drop(columns=["is_fraud"])
    out = calculate_graph_features(df_no_label)
    assert "connected_component_size" in out.columns


def test_fraud_ring_produces_larger_clusters_than_legit(small_legit_df):
    ring = get_attack("FRAUD_RING").generate(60, difficulty=0.2, seed=1)
    combined = pd.concat([small_legit_df, ring], ignore_index=True)
    combined = calculate_graph_features(combined)
    fraud_avg = combined[combined.is_fraud == 1]["suspicious_cluster_score"].mean()
    legit_avg = combined[combined.is_fraud == 0]["suspicious_cluster_score"].mean()
    assert fraud_avg > legit_avg


def test_find_shared_attributes_for_known_applicant(small_legit_df):
    graph = build_graph(small_legit_df)
    aid = small_legit_df.iloc[0]["applicant_id"]
    info = find_shared_attributes(graph, aid)
    assert info["found"] is True
    assert len(info["shared_attributes"]) > 0


def test_find_shared_attributes_unknown_applicant(small_legit_df):
    graph = build_graph(small_legit_df)
    info = find_shared_attributes(graph, "NOT_A_REAL_ID")
    assert info["found"] is False


def test_detect_suspicious_clusters_returns_list(small_legit_df):
    ring = get_attack("FRAUD_RING").generate(60, difficulty=0.2, seed=2)
    combined = calculate_graph_features(pd.concat([small_legit_df, ring], ignore_index=True))
    clusters = detect_suspicious_clusters(combined, min_size=3)
    assert isinstance(clusters, list)
    if clusters:
        assert "avg_suspicious_score" in clusters[0]
