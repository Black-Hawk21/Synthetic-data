"""Graph-topology features.

Chains, rings and fan structures are properties of the *network*, so a purely
row-wise feature set cannot see them. These are computed on the aggregated
account-to-account graph.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def build_graph_features(edges: pd.DataFrame, accounts: pd.DataFrame,
                         max_degree_for_local: int = 400) -> pd.DataFrame:
    ids = accounts["account_id"].astype(str).to_numpy()
    G = nx.from_pandas_edgelist(edges, "sender", "receiver",
                                edge_attr=["n_txns", "total_amount"],
                                create_using=nx.DiGraph)
    G.add_nodes_from(ids)

    f = pd.DataFrame(index=pd.Index(ids, name="account_id"))
    in_deg = pd.Series(dict(G.in_degree())).reindex(ids).fillna(0)
    out_deg = pd.Series(dict(G.out_degree())).reindex(ids).fillna(0)
    f["g_in_degree"] = in_deg.to_numpy()
    f["g_out_degree"] = out_deg.to_numpy()
    f["g_degree"] = f["g_in_degree"] + f["g_out_degree"]
    f["g_degree_ratio"] = f["g_in_degree"] / (f["g_out_degree"] + 1.0)
    # A pure relay has one way in and one way out.
    f["g_relay_score"] = 1.0 / (1.0 + np.abs(f["g_in_degree"] - f["g_out_degree"]))

    f["g_pagerank"] = pd.Series(nx.pagerank(G, alpha=0.85, weight="total_amount")
                                ).reindex(ids).fillna(0).to_numpy()
    f["g_core_number"] = pd.Series(nx.core_number(nx.Graph(G))).reindex(ids).fillna(0).to_numpy()

    # Strongly connected components: the direct fingerprint of circular flows.
    scc_size, scc_id = {}, {}
    for i, comp in enumerate(nx.strongly_connected_components(G)):
        for node in comp:
            scc_size[node] = len(comp)
            scc_id[node] = i
    f["g_scc_size"] = pd.Series(scc_size).reindex(ids).fillna(1).to_numpy()
    f["g_in_cycle"] = (f["g_scc_size"] > 1).astype(int)

    # Reciprocity: legitimate pairs pay each other both ways; layering rarely does.
    recip = {n: 0 for n in ids}
    for u, v in G.edges():
        if G.has_edge(v, u):
            recip[u] = recip.get(u, 0) + 1
    f["g_reciprocal_edges"] = pd.Series(recip).reindex(ids).fillna(0).to_numpy()
    f["g_reciprocity"] = f["g_reciprocal_edges"] / (f["g_out_degree"] + 1.0)

    # Local clustering on a hub-trimmed undirected view (exact version is O(d^2)
    # and a single merchant hub with 5k neighbours dominates the runtime).
    U = nx.Graph(G)
    hubs = [n for n, d in U.degree() if d > max_degree_for_local]
    U.remove_nodes_from(hubs)
    clus = nx.clustering(U)
    f["g_clustering"] = pd.Series(clus).reindex(ids).fillna(0.0).to_numpy()
    f["g_is_hub"] = pd.Series({n: 1 for n in hubs}).reindex(ids).fillna(0).to_numpy()

    f["g_two_hop_out"] = _two_hop_counts(G, ids, direction="out")
    f["g_two_hop_in"] = _two_hop_counts(G, ids, direction="in")
    f["g_fan_out_ratio"] = f["g_two_hop_out"] / (f["g_out_degree"] + 1.0)
    return f.reset_index()


def _two_hop_counts(G: nx.DiGraph, ids, direction="out", cap=2000) -> np.ndarray:
    """Size of the 2-hop reachable set, capped so hubs don't explode the cost."""
    step = G.successors if direction == "out" else G.predecessors
    out = np.zeros(len(ids), dtype=float)
    for i, node in enumerate(ids):
        if node not in G:
            continue
        first = list(step(node))
        if len(first) > cap:
            out[i] = float(cap)
            continue
        seen = set(first)
        for nb in first:
            seen.update(step(nb))
            if len(seen) > cap:
                break
        seen.discard(node)
        out[i] = float(min(len(seen), cap))
    return out
