"""A directed, edge-aware GNN for money-laundering detection.

Pure PyTorch — no torch-geometric, no torch-scatter. The only extra dependency
over the repo's `requirements.txt` is `torch` itself, so this installs cleanly
on CPU-only machines.

Why this architecture rather than a stock GCN/GraphSAGE:

  * **Direction matters.** Laundering is about where money *goes*. Every layer
    aggregates over incoming and outgoing edges with separate weights, so an
    account that receives from many and pays one (fan-in collector) is
    represented differently from the reverse (fan-out distributor).
  * **Edges carry money, not just adjacency.** Each message is conditioned on
    the edge attributes — log volume, transaction count, mean/max amount,
    lifespan, and the share of the sender's outflow and the receiver's inflow
    that this edge represents. A GCN that only sees "there is an edge" cannot
    tell a salary credit from a pass-through hop.
  * **Mean *and* max aggregation.** Mean describes routine behaviour; max
    catches the single anomalous counterparty that carries the episode.
  * **Jumping knowledge.** The classification head sees the raw features and
    every layer's output concatenated, so 1-hop, 2-hop and 3-hop views stay
    separable and the model can fall back to pure tabular signal where the
    graph is uninformative. This is what makes the GNN a strict improvement on
    the tabular baseline rather than a gamble.

Three hops is deliberate: `layering_chain` and `mule_network` are multi-hop
typologies, and `circular_flow` needs enough range to come back around.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

EDGE_ATTR_DIM = 8


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_graph_tensors(edges: pd.DataFrame, account_ids: pd.Series
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (edge_index [2, E], edge_attr [E, EDGE_ATTR_DIM]).

    Ground-truth edge columns (`is_laundering_edge`, `n_laundering_txns`) are
    never read here.
    """
    index = {a: i for i, a in enumerate(account_ids)}
    src = edges["sender"].map(index)
    dst = edges["receiver"].map(index)
    keep = src.notna() & dst.notna()
    edges = edges.loc[keep]
    src = src[keep].to_numpy(np.int64)
    dst = dst[keep].to_numpy(np.int64)

    total = edges["total_amount"].to_numpy(np.float64)
    n = len(index)
    out_total = np.bincount(src, weights=total, minlength=n)
    in_total = np.bincount(dst, weights=total, minlength=n)

    attr = np.column_stack([
        np.log1p(total),
        np.log1p(edges["n_txns"].to_numpy(np.float64)),
        np.log1p(edges["mean_amount"].to_numpy(np.float64)),
        np.log1p(edges["max_amount"].to_numpy(np.float64)),
        np.log1p(edges["lifespan_days"].to_numpy(np.float64)),
        total / np.maximum(out_total[src], 1.0),   # share of sender's outflow
        total / np.maximum(in_total[dst], 1.0),    # share of receiver's inflow
        (edges["max_amount"].to_numpy(np.float64)
         / np.maximum(edges["mean_amount"].to_numpy(np.float64), 1.0)),
    ]).astype(np.float32)
    attr[:, :5] /= np.maximum(attr[:, :5].std(0, keepdims=True), 1e-6)

    edge_index = torch.from_numpy(np.stack([src, dst]))
    return edge_index, torch.from_numpy(attr)


def normalize_features(X: np.ndarray, stats: tuple | None = None):
    """Signed-log squash for heavy tails, then standardize.

    Money data is heavy-tailed enough that a plain z-score leaves most of the
    mass in a spike near zero; the log first is what makes the scale usable.
    """
    Xs = np.sign(X) * np.log1p(np.abs(X))
    if stats is None:
        mu, sd = Xs.mean(0), Xs.std(0)
        sd[sd < 1e-6] = 1.0
        stats = (mu, sd)
    mu, sd = stats
    return ((Xs - mu) / sd).astype(np.float32), stats


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------
def _aggregate(msg: torch.Tensor, index: torch.Tensor, n: int
               ) -> torch.Tensor:
    """Mean and max of `msg` grouped by destination `index`. -> [n, 2*d]"""
    d = msg.size(1)
    total = torch.zeros(n, d, device=msg.device, dtype=msg.dtype)
    total.index_add_(0, index, msg)
    count = torch.zeros(n, 1, device=msg.device, dtype=msg.dtype)
    count.index_add_(0, index, torch.ones(msg.size(0), 1, dtype=msg.dtype,
                                          device=msg.device))
    mean = total / count.clamp(min=1.0)

    peak = torch.full((n, d), -1e9, device=msg.device, dtype=msg.dtype)
    peak = peak.scatter_reduce(0, index.unsqueeze(1).expand(-1, d), msg,
                               reduce="amax", include_self=True)
    peak = torch.where(count > 0, peak, torch.zeros_like(peak))
    return torch.cat([mean, peak], dim=1)


class DirectedEdgeConv(nn.Module):
    """One hop of directed, edge-conditioned message passing.

    Messages are formed as `W_node · h_src + W_edge · e`, which lets the whole
    layer be computed with two small dense matmuls (N and E rows) plus gathers,
    instead of materialising a concatenated [E, d_node + d_edge] tensor. On a
    400k-edge graph that is the difference between seconds and minutes per
    epoch on CPU.
    """

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = EDGE_ATTR_DIM,
                 dropout: float = 0.2):
        super().__init__()
        self.lin_out_node = nn.Linear(in_dim, out_dim)    # along edge direction
        self.lin_out_edge = nn.Linear(edge_dim, out_dim)
        self.lin_in_node = nn.Linear(in_dim, out_dim)     # against direction
        self.lin_in_edge = nn.Linear(edge_dim, out_dim)
        self.combine = nn.Linear(in_dim + 4 * out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]
        n = h.size(0)
        # money arriving at dst, from the perspective of the receiver
        m_in = self.lin_out_node(h)[src] + self.lin_out_edge(edge_attr)
        agg_in = _aggregate(m_in, dst, n)
        # money leaving src, from the perspective of the sender
        m_out = self.lin_in_node(h)[dst] + self.lin_in_edge(edge_attr)
        agg_out = _aggregate(m_out, src, n)

        z = self.combine(torch.cat([h, agg_in, agg_out], dim=1))
        return self.dropout(F.relu(self.norm(z)))


class AMLGNN(nn.Module):
    """Node (account) classifier: 3 directed hops + jumping-knowledge head."""

    def __init__(self, in_dim: int, hidden: int = 96, layers: int = 3,
                 dropout: float = 0.2, edge_dim: int = EDGE_ATTR_DIM):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden),
                                   nn.ReLU(), nn.Dropout(dropout))
        self.convs = nn.ModuleList([
            DirectedEdgeConv(hidden, hidden, edge_dim, dropout)
            for _ in range(layers)])
        self.embed_dim = hidden * (layers + 1)
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1))

    def embed(self, x, edge_index, edge_attr) -> torch.Tensor:
        h = self.input(x)
        views = [h]
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr) + h      # residual
            views.append(h)
        return torch.cat(views, dim=1)                  # jumping knowledge

    def forward(self, x, edge_index, edge_attr) -> torch.Tensor:
        return self.head(self.embed(x, edge_index, edge_attr)).squeeze(-1)


class TransactionHead(nn.Module):
    """Transaction classifier over GNN node embeddings + transaction fields.

    Each transaction is represented by its own attributes plus the sender's and
    receiver's graph embeddings and their interaction (difference and product),
    which is what lets the model reason about *the pair* rather than two
    independent accounts.
    """

    def __init__(self, txn_dim: int, emb_dim: int, hidden: int = 128,
                 dropout: float = 0.2):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(emb_dim, 64), nn.ReLU())
        self.net = nn.Sequential(
            nn.Linear(txn_dim + 4 * 64, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1))

    def forward(self, txn_x, emb_src, emb_dst) -> torch.Tensor:
        a, b = self.project(emb_src), self.project(emb_dst)
        z = torch.cat([txn_x, a, b, a - b, a * b], dim=1)
        return self.net(z).squeeze(-1)


# ---------------------------------------------------------------------------
# Bundle I/O
# ---------------------------------------------------------------------------
def save_gnn(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_gnn(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)
