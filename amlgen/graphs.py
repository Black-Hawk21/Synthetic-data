"""Transaction graph construction and export.

Two views of the same data:
  * the multigraph of raw transactions (kept as the transaction table)
  * the aggregated account-to-account graph, which is what graph algorithms run on
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from .utils import epoch_seconds


def build_edge_table(txns: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw transactions into weighted directed edges."""
    df = txns.copy()
    df["sender"] = df["sender"].astype(str)
    df["receiver"] = df["receiver"].astype(str)
    df["_ts"] = epoch_seconds(df["timestamp"])
    g = df.groupby(["sender", "receiver"], observed=True, sort=False)
    edges = g.agg(
        n_txns=("amount", "size"),
        total_amount=("amount", "sum"),
        mean_amount=("amount", "mean"),
        max_amount=("amount", "max"),
        first_ts=("_ts", "min"),
        last_ts=("_ts", "max"),
        n_laundering_txns=("is_laundering", "sum"),
    ).reset_index()
    edges["is_laundering_edge"] = (edges["n_laundering_txns"] > 0).astype(np.int8)
    edges["lifespan_days"] = ((edges["last_ts"] - edges["first_ts"]) / 86400).round(3)
    edges["first_time"] = pd.to_datetime(edges["first_ts"], unit="s")
    edges["last_time"] = pd.to_datetime(edges["last_ts"], unit="s")
    edges["total_amount"] = edges["total_amount"].round(2)
    edges["mean_amount"] = edges["mean_amount"].round(2)
    return edges.drop(columns=["first_ts", "last_ts"])


def build_graph(edges: pd.DataFrame, accounts: pd.DataFrame | None = None) -> nx.DiGraph:
    """Aggregated directed graph, with account attributes attached to nodes."""
    G = nx.from_pandas_edgelist(
        edges, source="sender", target="receiver",
        edge_attr=["n_txns", "total_amount", "mean_amount", "max_amount",
                   "lifespan_days", "is_laundering_edge"],
        create_using=nx.DiGraph)
    if accounts is not None:
        keep = ["archetype", "city", "country", "kyc_level", "account_age_days",
                "is_laundering", "laundering_role", "in_benign_lookalike"]
        keep = [c for c in keep if c in accounts.columns]
        attrs = accounts.set_index("account_id")[keep]
        for col in keep:
            values = attrs[col]
            if values.dtype == bool:
                values = values.astype(int)
            nx.set_node_attributes(G, values.to_dict(), col)
    return G


def episode_subgraph(txns: pd.DataFrame, episode_id: str, context_hops: int = 1,
                     max_context_edges: int = 300, seed: int = 0) -> nx.DiGraph:
    """One episode's flows plus a sample of the participants' legitimate traffic.

    The context is sampled: a mule may also be a customer of a merchant with
    thousands of edges, which would bury the episode in grey.
    """
    core = txns[txns["episode_id"] == episode_id]
    if core.empty:
        return nx.DiGraph()
    nodes = set(core["sender"].astype(str)) | set(core["receiver"].astype(str))
    core_edges = build_edge_table(core)

    if context_hops > 0:
        s = txns["sender"].astype(str)
        r = txns["receiver"].astype(str)
        ctx = txns[(s.isin(nodes) | r.isin(nodes)) & (txns["episode_id"] != episode_id)]
        ctx_edges = build_edge_table(ctx)
        if len(ctx_edges) > max_context_edges:
            ctx_edges = ctx_edges.sample(max_context_edges, random_state=seed)
        edges = pd.concat([core_edges, ctx_edges], ignore_index=True)
    else:
        edges = core_edges

    G = build_graph(edges)
    core_pairs = set(zip(core_edges["sender"], core_edges["receiver"]))
    nx.set_edge_attributes(
        G, {(u, v): int((u, v) in core_pairs) for u, v in G.edges}, "in_episode_edge")
    nx.set_node_attributes(G, {n: int(n in nodes) for n in G.nodes}, "in_episode")
    return G


def write_graphml(G: nx.DiGraph, path) -> None:
    """GraphML only accepts primitives - coerce everything first."""
    H = G.copy()
    for _, data in H.nodes(data=True):
        for k, v in list(data.items()):
            if not isinstance(v, (int, float, str, bool)) or isinstance(v, np.generic):
                data[k] = "" if v is None else str(v)
            elif isinstance(v, np.generic):
                data[k] = v.item()
    for _, _, data in H.edges(data=True):
        for k, v in list(data.items()):
            if isinstance(v, np.generic):
                data[k] = v.item()
    nx.write_graphml(H, str(path))
