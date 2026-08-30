"""
app.py - Streamlit frontend for the ATO Defense Lab.

Run with:  streamlit run app.py

Reads whatever scoring output exists (prefers the full pipeline output,
falls back to the bundled sample) -- it doesn't retrain or re-score
anything itself, it's a viewer/demo layer on top of artifacts already
produced by 02_baseline_and_attack.py -> 03_features.py -> 04_train_detect.py
-> 05_mitigate_and_demo.py.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config as cfg

st.set_page_config(page_title="Account Takeover", layout="wide")

FULL_PATH = Path(cfg.SCORED_EVENTS_CSV)
SAMPLE_PATH = Path(cfg.OUT_DIR) / "sample" / "scored_events_sample.csv"


@st.cache_data(show_spinner="Loading scored events...")
def load_data():
    if FULL_PATH.exists():
        df = pd.read_csv(FULL_PATH, parse_dates=["event_time"])
        source = f"Full pipeline output ({FULL_PATH.name})"
    elif SAMPLE_PATH.exists():
        df = pd.read_csv(SAMPLE_PATH, parse_dates=["event_time"])
        source = f"Bundled sample ({SAMPLE_PATH.name}) — run the pipeline scripts for the full dataset"
    else:
        return None, None
    if "action" not in df.columns:
        from importlib import import_module
        mitigate = import_module("05_mitigate_and_demo")
        df["action"] = df[cfg.RISK_SCORE_COL].apply(mitigate.assign_action)
    return df, source


df, source = load_data()

st.title("Authentication & Account Takeover Module")
#st.caption("Mastercard Innovation Challenge 2026 — red team / blue team demo")

if df is None:
    st.error(
        "No scored data found. Run the pipeline first:\n\n"
        "```\npython3 02_baseline_and_attack.py\npython3 03_features.py\n"
        "python3 04_train_detect.py\npython3 05_mitigate_and_demo.py\n```"
    )
    st.stop()

#st.sidebar.success(f"Data source: {source}")
st.sidebar.metric("Total events", f"{len(df):,}")
page = st.sidebar.radio(
    "View",
    ["Overview", "Botnet Ring", "Attack Episode Playback", "Explore Events"],
)

ACTION_COLORS = {"allow": "#2ca02c", "step_up_auth": "#ff9800", "block": "#d62728"}


# ---------------------------------------------------------------- Overview
if page == "Overview":
    total_ato = int(df["label_ato"].sum())
    blocked_ato = int(((df["action"] == "block") & (df["label_ato"] == 1)).sum())
    missed_ato = int(((df["action"] == "allow") & (df["label_ato"] == 1)).sum())
    fp_blocked = int(((df["action"] == "block") & (df["label_ato"] == 0)).sum())
    total_blocked = int((df["action"] == "block").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Injected ATO events", f"{total_ato:,}")
    c2.metric("Blocked outright", f"{blocked_ato:,}", f"{100*blocked_ato/max(total_ato,1):.1f}% of attacks")
    c3.metric("Missed (allowed)", f"{missed_ato:,}")
    c4.metric("False-positive blocks", f"{fp_blocked:,}", f"{100*fp_blocked/max(total_blocked,1):.1f}% of all blocks")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk score distribution")
        fig = px.histogram(
            df, x=cfg.RISK_SCORE_COL, color=df["label_ato"].map({0: "benign", 1: "ATO"}),
            nbins=50, barmode="overlay", opacity=0.7,
            color_discrete_map={"benign": "#1f77b4", "ATO": "#d62728"},
            labels={"color": "ground truth"},
        )
        fig.update_layout(height=380, xaxis_title="ATO risk score (0-100)")
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("Action bands")
        action_counts = df["action"].value_counts().reindex(["allow", "step_up_auth", "block"]).fillna(0)
        fig2 = px.bar(
            x=action_counts.index, y=action_counts.values,
            color=action_counts.index, color_discrete_map=ACTION_COLORS,
            labels={"x": "action", "y": "events"},
        )
        fig2.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig2, width='stretch')

    st.subheader("Action vs. ground truth")
    tbl = pd.crosstab(df["action"], df["label_ato"].map({0: "benign", 1: "ATO"}))
    st.dataframe(tbl, width='stretch')


# --------------------------------------------------------------- Botnet Ring
elif page == "Botnet Ring":
    st.subheader("Shared attacker infrastructure")
    st.caption(
        "Devices/IPs touching many distinct accounts in a short window are the "
        "signature of automated, GenAI-driven credential-stuffing bots reusing "
        "infrastructure across victims."
    )
    login_events = df[df["event_type"].isin(["failed_login", "ato_login"])]
    top_devices = (
        login_events.groupby("device_id")[cfg.COL_ACCOUNT]
        .nunique().sort_values(ascending=False).head(15)
    )
    fig = px.bar(
        x=top_devices.values, y=top_devices.index, orientation="h",
        labels={"x": "distinct accounts touched", "y": "device_id"},
    )
    fig.update_layout(height=450, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, width='stretch')

    st.subheader("Look into a device")
    picked = st.selectbox("Device", top_devices.index)
    victims = login_events[login_events["device_id"] == picked][cfg.COL_ACCOUNT].unique()
    st.write(f"**{picked}** touched **{len(victims)}** distinct accounts:")
    st.dataframe(pd.DataFrame({cfg.COL_ACCOUNT: victims}), width='stretch', height=250)


# ------------------------------------------------------- Attack Episode Playback
elif page == "Attack Episode Playback":
    st.subheader("Watch an attack unfold")
    episode_ids = sorted(df["attack_episode_id"].dropna().unique())
    if not episode_ids:
        st.warning("No attack episodes in this dataset.")
    else:
        eid = st.selectbox("Attack episode", episode_ids)
        acc = df[df["attack_episode_id"] == eid][cfg.COL_ACCOUNT].iloc[0]
        timeline = df[df[cfg.COL_ACCOUNT] == acc].sort_values("event_time").reset_index(drop=True)
        st.caption(f"Account **{acc}** — full timeline, {len(timeline)} events (attack + surrounding normal activity)")

        step = st.slider("Step through time", 1, len(timeline), min(10, len(timeline)))
        visible = timeline.iloc[:step]
        current = visible.iloc[-1]

        gc1, gc2 = st.columns([1, 2])
        with gc1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=current[cfg.RISK_SCORE_COL],
                title={"text": f"Risk score — {current['event_type']}"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": ACTION_COLORS.get(current["action"], "gray")},
                    "steps": [
                        {"range": [0, 30], "color": "#e8f5e9"},
                        {"range": [30, 70], "color": "#fff3e0"},
                        {"range": [70, 100], "color": "#ffebee"},
                    ],
                },
            ))
            fig.update_layout(height=280, margin=dict(t=40, b=0))
            st.plotly_chart(fig, width='stretch')
            st.metric("Action taken", current["action"])

        with gc2:
            line = px.line(
                visible, x="event_time", y=cfg.RISK_SCORE_COL, markers=True,
                color=visible["label_ato"].map({0: "benign", 1: "ATO"}),
                color_discrete_map={"benign": "#1f77b4", "ATO": "#d62728"},
            )
            line.update_layout(height=280, showlegend=True)
            st.plotly_chart(line, width='stretch')

        st.dataframe(
            visible[["event_time", "event_type", "device_id", "ip_address", "amount",
                     "device_new_for_account", "ip_new_for_account",
                     "failed_logins_trailing_10min", cfg.RISK_SCORE_COL, "action"]],
            width='stretch', height=300,
        )


# --------------------------------------------------------------- Explore Events
elif page == "Explore Events":
    st.subheader("Filter and search all events")
    colf1, colf2, colf3 = st.columns(3)
    with colf1:
        event_types = st.multiselect(
            "Event type", sorted(df["event_type"].unique()), default=list(df["event_type"].unique())
        )
    with colf2:
        actions = st.multiselect(
            "Action", sorted(df["action"].unique()), default=list(df["action"].unique())
        )
    with colf3:
        min_risk, max_risk = st.slider("Risk score range", 0, 100, (0, 100))

    acc_filter = st.text_input("Account ID contains (optional)")

    filtered = df[
        df["event_type"].isin(event_types)
        & df["action"].isin(actions)
        & df[cfg.RISK_SCORE_COL].between(min_risk, max_risk)
    ]
    if acc_filter:
        filtered = filtered[filtered[cfg.COL_ACCOUNT].str.contains(acc_filter, case=False, na=False)]

    st.caption(f"{len(filtered):,} events match")
    st.dataframe(
        filtered.sort_values(cfg.RISK_SCORE_COL, ascending=False)
        [["event_time", cfg.COL_ACCOUNT, "event_type", "device_id", "amount",
          "label_ato", cfg.RISK_SCORE_COL, "action"]],
        width='stretch', height=500,
    )
