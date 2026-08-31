"""Identity graph construction (NetworkX -- no Neo4j required, section 9).

Nodes: person, phone, email, address, device, ip, document.
Edges: person -> {phone, email, address, device, ip, document}.

This graph is built purely from structural attributes (who shares what with
whom). It never looks at the fraud label, which is required so that the
resulting graph features can be used as leakage-free ML inputs (section 20).
"""
from __future__ import annotations

import networkx as nx
import pandas as pd

NODE_KIND_PREFIX = {
    "phone": "phone", "email": "email", "address": "address",
    "device_id": "device", "ip_id": "ip", "document_number": "document",
}


def _node_id(kind: str, value: str) -> str:
    return f"{kind}:{value}"


def add_identity(graph: nx.Graph, row: pd.Series) -> None:
    """Add one applicant and their shared attribute edges to the graph."""
    person_node = f"person:{row['applicant_id']}"
    graph.add_node(person_node, kind="person", applicant_id=row["applicant_id"])

    for col, kind in NODE_KIND_PREFIX.items():
        value = row.get(col)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        attr_node = _node_id(kind, str(value))
        if attr_node not in graph:
            graph.add_node(attr_node, kind=kind, value=str(value))
        graph.add_edge(person_node, attr_node, relation=kind)


def build_graph(df: pd.DataFrame) -> nx.Graph:
    """Build the full identity graph for a dataset of applicants."""
    graph = nx.Graph()
    for _, row in df.iterrows():
        add_identity(graph, row)
    return graph


def find_shared_attributes(graph: nx.Graph, applicant_id: str) -> dict:
    """Return, for one applicant, every attribute node they touch and every
    OTHER person who shares that attribute -- used by GET /api/graph/{id}."""
    person_node = f"person:{applicant_id}"
    if person_node not in graph:
        return {"applicant_id": applicant_id, "found": False, "shared_attributes": []}

    shared = []
    for attr_node in graph.neighbors(person_node):
        kind = graph.nodes[attr_node].get("kind")
        co_applicants = [
            graph.nodes[p]["applicant_id"]
            for p in graph.neighbors(attr_node)
            if p != person_node
        ]
        shared.append({
            "attribute_type": kind,
            "value": graph.nodes[attr_node].get("value"),
            "shared_with_count": len(co_applicants),
            "shared_with": co_applicants[:25],
        })
    return {"applicant_id": applicant_id, "found": True, "shared_attributes": shared}


def find_connected_components(graph: nx.Graph) -> list[set]:
    return list(nx.connected_components(graph))
