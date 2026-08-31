from __future__ import annotations

from backend.graph.builder import find_shared_attributes
from backend.graph.fraud_ring import summarize_fraud_rings
from backend.services.detection_service import NoDatasetError
from backend.services.state import state


def get_applicant_graph(applicant_id: str) -> dict:
    df = state.get_dataset()
    if df is None:
        raise NoDatasetError("No dataset yet -- call POST /api/generate-applicants first.")
    from backend.graph.builder import build_graph
    graph = build_graph(df)
    info = find_shared_attributes(graph, applicant_id)
    if info["found"]:
        row = df[df["applicant_id"] == applicant_id].iloc[0]
        info["risk_score"] = float(row.get("suspicious_cluster_score", 0)) * 100
        info["is_fraud"] = int(row.get("is_fraud", 0))
        info["attack_type"] = row.get("attack_type")
        info["connected_component_size"] = int(row.get("connected_component_size", 1))
    return info


def get_fraud_rings(min_size: int = 3) -> dict:
    df = state.get_dataset()
    if df is None:
        raise NoDatasetError("No dataset yet -- call POST /api/generate-applicants first.")
    return summarize_fraud_rings(df, min_size=min_size)
