"""AML / synthetic transaction data -- a read-only view over modules/aml.

The AML project is CLI-only: `python run.py all` simulates a financial
ecosystem, injects 8 laundering typologies alongside 5 benign lookalikes, and
`compare_models.py` scores four detectors against it. This page presents what
that pipeline already committed under modules/aml/sample_data/ rather than
re-running any of it, so it is safe to open during a live demo.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

AML_DIR = Path(__file__).resolve().parent.parent.parent / "modules" / "aml"
SAMPLE = AML_DIR / "sample_data"
EVAL = SAMPLE / "evaluation"
FIGURES = SAMPLE / "figures"

# Categorical slots 1 and 2 of the reference palette. Validated as a pair:
# CVD dE 24.7, normal-vision dE 33.6, both >= 3:1 on a light surface.
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
TEXT_SECONDARY = "#52514e"
GRID = "rgba(82,81,78,0.16)"


def _layout(fig: go.Figure, height: int, x_title: str = "") -> go.Figure:
    """Recessive axes, no chart junk, text in ink rather than series colour."""
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=56, t=8, b=32),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_SECONDARY, size=13),
        xaxis_title=x_title,
        bargap=0.28,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, showline=False)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", zeroline=False, showline=False)
    return fig


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


st.title("💸 AML / Synthetic Transaction Data")
st.caption(
    "The dataset the other modules are built to be measured like: a simulated "
    "financial ecosystem with 8 laundering typologies injected alongside 5 "
    "*benign lookalikes* that exist specifically to generate hard negatives. "
    "Ground truth is the episode, not the row."
)

manifest = load_json(SAMPLE / "manifest.json")
report = load_json(EVAL / "report.json")

if manifest is None and report is None:
    st.warning(
        f"No pre-generated run found under `{SAMPLE.relative_to(AML_DIR.parent.parent)}`. "
        "Regenerate it with `cd modules/aml && python run.py all`."
    )
    st.stop()

tab_data, tab_quality, tab_models, tab_figures = st.tabs(
    ["Dataset", "Detection quality", "Tabular vs GNN", "Figures"]
)

# --------------------------------------------------------------------------
with tab_data:
    if manifest:
        rows = manifest["row_counts"]
        balance = manifest["label_balance"]
        cfg = manifest["config"]

        a, b, c, d = st.columns(4)
        a.metric("Transactions", f"{rows['transactions']:,}")
        b.metric("Accounts", f"{rows['accounts']:,}")
        c.metric("Episodes", f"{rows['episodes']:,}")
        d.metric("Laundering rate", f"{balance['positive_rate'] * 100:.2f}%")
        st.caption(
            f"{balance['laundering_transactions']:,} laundering transactions out of "
            f"{balance['total_transactions']:,} — deliberately imbalanced, because "
            "real AML data is."
        )

        st.subheader("Episode mix")
        st.caption(
            "The red-team dial is `laundering.difficulty` "
            f"(this run: **{cfg['laundering']['difficulty']}**, seed "
            f"`{cfg['simulation']['seed'] if 'seed' in cfg.get('simulation', {}) else cfg['seed']}`). "
            "The benign lookalikes are the point: a payroll fan-out and a "
            "laundering fan-out have the same shape, so a detector that keys on "
            "shape alone fails here."
        )

        laundering = cfg["laundering"]["episodes"]
        lookalikes = cfg.get("benign_lookalikes", {}).get("episodes", {})
        mix = pd.DataFrame(
            [{"pattern": k, "episodes": v, "family": "laundering"} for k, v in laundering.items()]
            + [{"pattern": k, "episodes": v, "family": "benign lookalike"} for k, v in lookalikes.items()]
        ).sort_values("episodes")

        fig = go.Figure()
        for family, colour in (("laundering", SERIES_2), ("benign lookalike", SERIES_1)):
            part = mix[mix["family"] == family]
            fig.add_bar(
                y=part["pattern"],
                x=part["episodes"],
                name=family,
                orientation="h",
                marker=dict(color=colour, cornerradius=4),
                text=part["episodes"],
                textposition="outside",
                textfont=dict(color=TEXT_SECONDARY),
                hovertemplate="%{y}: %{x} episodes<extra>" + family + "</extra>",
            )
        fig.update_layout(legend=dict(orientation="h", y=1.06, x=0, title=None))
        st.plotly_chart(_layout(fig, 460, "episodes configured"), width="stretch")

    episodes = load_csv(SAMPLE / "episodes.csv")
    if episodes is not None:
        with st.expander(f"Episode ledger ({len(episodes):,} rows)"):
            st.dataframe(episodes, width="stretch", hide_index=True)

# --------------------------------------------------------------------------
with tab_quality:
    if report:
        a, b, c, d = st.columns(4)
        a.metric("PR-AUC", f"{report['pr_auc']:.4f}")
        b.metric("ROC-AUC", f"{report['roc_auc']:.4f}")
        c.metric("Precision", f"{report['precision']:.3f}")
        d.metric("Recall", f"{report['recall']:.3f}")
        st.caption(
            f"{report['alerts']} alerts on {report['n_accounts']} held-out accounts "
            f"({report['alert_rate'] * 100:.1f}% alert rate), {report['n_laundering']} of which "
            "actually launder."
        )
        st.info(
            f"**{report['lookalike_share_of_fp'] * 100:.0f}% of the false positives "
            f"({report['false_positives_on_benign_lookalikes']} of {report['false_positives']}) "
            "land on benign lookalikes.** That is the dataset working as designed — the "
            "hard negatives are the ones a shape-based detector trips over, and chasing "
            "them away would mean the simulation had stopped being realistic."
        )

    by_pattern = load_csv(EVAL / "evaluation_by_pattern.csv")
    if by_pattern is not None:
        st.subheader("Account-level recall by typology")
        ranked = by_pattern.sort_values("recall")
        fig = go.Figure(
            go.Bar(
                y=ranked["pattern"],
                x=ranked["recall"],
                orientation="h",
                marker=dict(color=SERIES_1, cornerradius=4),
                text=[f"{v:.0%}" for v in ranked["recall"]],
                textposition="outside",
                textfont=dict(color=TEXT_SECONDARY),
                customdata=ranked["n_accounts"],
                hovertemplate="%{y}<br>recall %{x:.1%} over %{customdata} accounts<extra></extra>",
            )
        )
        fig.update_xaxes(range=[0, 1.12], tickformat=".0%")
        st.plotly_chart(_layout(fig, 380, "recall"), width="stretch")
        st.caption(
            "`rapid_pass_through` is the weak spot — money moves through in hours, so "
            "the account-level features have little history to key on."
        )

    importance = load_csv(EVAL / "feature_importance.csv")
    if importance is not None:
        st.subheader("What the tabular model actually uses")
        top = importance.nlargest(10, "importance").sort_values("importance")
        fig = go.Figure(
            go.Bar(
                y=top["feature"],
                x=top["importance"],
                orientation="h",
                marker=dict(color=SERIES_1, cornerradius=4),
                text=[f"{v:.3f}" for v in top["importance"]],
                textposition="outside",
                textfont=dict(color=TEXT_SECONDARY),
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            )
        )
        st.plotly_chart(_layout(fig, 400, "permutation importance"), width="stretch")

    by_episode = load_csv(EVAL / "evaluation_by_episode.csv")
    if by_episode is not None:
        with st.expander("Episode-level recall (the metric that counts)"):
            st.dataframe(by_episode, width="stretch", hide_index=True)

# --------------------------------------------------------------------------
with tab_models:
    st.subheader("Graph structure beats flat features")
    st.caption(
        "Four detectors on identical held-out splits (from `MODELS_README.md`; "
        "reproduce with `cd modules/aml && python compare_models.py`). The GNN wins "
        "on every typology, not on an average that hides a failure."
    )

    comparison = pd.DataFrame(
        [
            {"level": "Account PR-AUC", "Tabular": 0.6447, "AML-GNN": 0.8845},
            {"level": "Account ROC-AUC", "Tabular": 0.8894, "AML-GNN": 0.9674},
            {"level": "Transaction PR-AUC", "Tabular": 0.7943, "AML-GNN": 0.9122},
            {"level": "Transaction ROC-AUC", "Tabular": 0.9976, "AML-GNN": 0.9992},
        ]
    )

    fig = go.Figure()
    for name, colour in (("Tabular", SERIES_1), ("AML-GNN", SERIES_2)):
        fig.add_bar(
            y=comparison["level"],
            x=comparison[name],
            name=name,
            orientation="h",
            marker=dict(color=colour, cornerradius=4),
            text=[f"{v:.4f}" for v in comparison[name]],
            textposition="outside",
            textfont=dict(color=TEXT_SECONDARY),
            hovertemplate="%{y}<br>" + name + " %{x:.4f}<extra></extra>",
        )
    fig.update_xaxes(range=[0, 1.16])
    fig.update_layout(legend=dict(orientation="h", y=1.08, x=0, title=None))
    st.plotly_chart(_layout(fig, 400, "score"), width="stretch")

    st.dataframe(comparison, width="stretch", hide_index=True)

    st.markdown(
        """
Two caveats the module's own README insists on, worth repeating to judges:

- `predict_gnn.py` reports roughly **0.95** account PR-AUC because it scores the
  whole dataset, training rows included. **0.8845 is the honest number.**
- The GNN's advantage is **not unconditional**. On a deliberately small run
  (3k accounts, 60 epochs) the tabular model wins, 0.654 to 0.644. The graph
  signal needs enough graph to be worth having.
"""
    )

# --------------------------------------------------------------------------
with tab_figures:
    captions = {
        "recall_by_pattern.png": "Recall per typology — the same story as the bar chart above, as the pipeline drew it.",
        "amount_overlap.png": "Laundering vs legitimate amount distributions. The overlap is the whole difficulty.",
        "holding_times.png": "How long value rests in an account before moving on.",
        "episode_L000019.png": "One laundering episode as a subgraph.",
    }
    available = [p for p in (FIGURES / n for n in captions) if p.exists()]
    if not available:
        st.info("No figures generated yet — `cd modules/aml && python run.py all`.")
    for path in available:
        st.image(str(path), caption=captions[path.name], width="stretch")

st.divider()
st.caption(
    "Reproduce everything on this page: `cd modules/aml && python run.py all` "
    "(~25 s), then `python train_models.py` and `python compare_models.py`. "
    "Training the GNNs is `python train_gnn.py` and takes roughly 30 minutes on a "
    "single core, which is why the weights are committed."
)
