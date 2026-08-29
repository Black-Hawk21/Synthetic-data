"""Plots for inspecting a generated dataset. Matplotlib only, no seaborn."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .graphs import episode_subgraph

ROLE_COLOURS = {
    "source": "#d62728", "intermediary": "#ff7f0e", "mule": "#ff7f0e",
    "transit": "#9467bd", "collector": "#1f77b4", "destination": "#2ca02c",
    "beneficiary": "#2ca02c", "counterparty": "#8c8c8c",
}


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_episode(txns: pd.DataFrame, members: pd.DataFrame, episode_id: str,
                 out_path, context_hops: int = 1, subtitle: str = ""):
    """Draw one episode's money flow, with its surrounding legitimate traffic."""
    import networkx as nx
    plt = _mpl()
    G = episode_subgraph(txns, episode_id, context_hops=context_hops)
    if G.number_of_nodes() == 0:
        raise ValueError(f"no transactions found for episode {episode_id}")
    roles = dict(zip(members.loc[members["episode_id"] == episode_id, "account_id"].astype(str),
                     members.loc[members["episode_id"] == episode_id, "role"]))

    core = [n for n in G.nodes if G.nodes[n].get("in_episode")]
    pos = nx.spring_layout(G, seed=7, k=1.6 / np.sqrt(max(G.number_of_nodes(), 2)))
    fig, ax = plt.subplots(figsize=(12, 9))
    outside = [n for n in G.nodes if not G.nodes[n].get("in_episode")]
    nx.draw_networkx_nodes(G, pos, nodelist=outside, node_size=18,
                           node_color="#dddddd", ax=ax)
    nx.draw_networkx_nodes(
        G, pos, nodelist=core, node_size=260,
        node_color=[ROLE_COLOURS.get(roles.get(n, "counterparty"), "#8c8c8c") for n in core],
        edgecolors="black", linewidths=0.6, ax=ax)

    ep_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("in_episode_edge")]
    bg_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get("in_episode_edge")]
    nx.draw_networkx_edges(G, pos, edgelist=bg_edges, edge_color="#cccccc", width=0.5,
                           arrows=False, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=ep_edges, edge_color="#c0392b", width=1.8,
                           arrowsize=14, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={n: n[-4:] for n in core}, font_size=7, ax=ax)

    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=r)
               for r, c in ROLE_COLOURS.items() if r in set(roles.values())]
    ax.legend(handles=handles, loc="upper left", fontsize=8, frameon=False)
    ax.set_title(f"Episode {episode_id} {subtitle}\nred = injected flow, "
                 f"grey = the same accounts' legitimate traffic (sampled)")
    ax.axis("off")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_amount_overlap(txns: pd.DataFrame, out_path):
    """Sanity check: laundering amounts must overlap legitimate ones.

    If these two histograms separate cleanly, the dataset is broken - a single
    threshold on `amount` would solve it. Bars show the share of each class's
    transactions falling in the bin (not a density: on log bins a density is
    divided by linear bin width and the shape becomes unreadable).
    """
    plt = _mpl()
    legit = txns.loc[txns["is_laundering"] == 0, "amount"].to_numpy()
    bad = txns.loc[txns["is_laundering"] == 1, "amount"].to_numpy()
    if len(bad) == 0:
        raise ValueError("no laundering transactions to plot")
    bins = np.logspace(1, np.log10(max(legit.max(), bad.max(), 10)), 55)
    fig, ax = plt.subplots(figsize=(9, 5))
    for x, label, colour in ((legit, "legitimate", "#4c78a8"),
                             (bad, "laundering", "#c0392b")):
        ax.hist(x, bins=bins, weights=np.ones(len(x)) / len(x), alpha=0.6,
                label=f"{label} (median {np.median(x):,.0f})", color=colour)
    ax.set_xscale("log")
    ax.set_xlabel("transaction amount (log scale)")
    ax.set_ylabel("share of that class's transactions")
    ax.set_title("Amount distributions must overlap")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_pattern_recall(by_pattern: pd.DataFrame, out_path, title="Recall by typology"):
    plt = _mpl()
    d = by_pattern.sort_values("recall")
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(d) + 2))
    ax.barh(d["pattern"], d["recall"], color="#4c78a8")
    ax.set_xlim(0, 1)
    ax.set_xlabel("recall at the alert budget")
    ax.set_title(title)
    for y, (v, n) in enumerate(zip(d["recall"], d.iloc[:, 1])):
        ax.text(min(v + 0.02, 0.95), y, f"{v:.2f}  (n={n})", va="center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_holding_times(features: pd.DataFrame, out_path):
    """Holding time is the signal the simulator is built around - check it exists."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, mask, colour in [("legitimate", features["is_laundering"] == 0, "#4c78a8"),
                                ("laundering", features["is_laundering"] == 1, "#c0392b")]:
        v = features.loc[mask, "median_holding_seconds"]
        v = v[v > 0] / 3600.0
        if len(v):
            ax.hist(v, bins=np.logspace(-2, 3, 50), density=True, alpha=0.6,
                    label=label, color=colour)
    ax.set_xscale("log")
    ax.set_xlabel("median holding time (hours, log scale)")
    ax.set_ylabel("density")
    ax.set_title("Time between receiving and forwarding funds")
    ax.legend(frameon=False)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
