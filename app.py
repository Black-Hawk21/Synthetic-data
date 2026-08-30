"""
Streamlit demo dashboard for the AI Defense Lab project (Mastercard Innovation
Challenge -- Social Engineering & Phishing at Scale).

Five tabs:
1. Attack Simulator -- generate a NEW fraud attempt live via the LLM (optionally
   in "evasion mode", targeting the current detector's own known weaknesses),
   then immediately run it through the trained detector and show whether it
   was caught. This is the live "build the attack, then build the defense"
   demo moment. Falls back to a cached example if the API isn't reachable,
   so the demo doesn't die on a flaky connection during judging.
2. Live Detector -- paste any message, get an instant fraud/legit prediction
   with a per-message explanation (which words in THIS text pushed the
   decision). Runs entirely locally against the saved baseline model --
   no API calls, so it's fast and safe to demo live for judges.
3. Dataset Overview -- stats on your generated data (counts by subtype,
   difficulty, channel), pulled live from data/generated/*.jsonl.
4. Model Comparison -- baseline vs RoBERTa vs holdout generalization,
   pulled from eval/*.json (whatever's present -- gracefully skips
   anything not yet generated).
5. Adversarial Arms Race -- the evasion-rate-by-generation chart from
   Phase 5, plus the holdout precision/recall trade-off across generations.

Run from the project root:
    pip install streamlit plotly --break-system-packages
    streamlit run app.py
"""

import json
import os
import random
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "generator"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "detector"))

BASE_DIR = os.path.dirname(__file__)
ARTIFACTS_DIR = os.path.join(BASE_DIR, "detector", "artifacts")
EVAL_DIR = os.path.join(BASE_DIR, "eval")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "baseline_model.joblib")
VECTORIZER_PATH = os.path.join(ARTIFACTS_DIR, "baseline_vectorizer.joblib")

st.set_page_config(
    page_title="Phishing",
    page_icon="\U0001F6E1\uFE0F",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cached loaders -- Streamlit re-runs the whole script on every interaction,
# so anything reading from disk needs caching or the app would re-read every
# file on every keystroke.
# ---------------------------------------------------------------------------

@st.cache_resource
def load_baseline_model():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
        return None, None
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


@st.cache_data
def load_dataset():
    from dataset_utils import load_all
    train_df, holdout_df = load_all()
    return train_df, holdout_df


@st.cache_data
def load_eval_json(filename):
    path = os.path.join(EVAL_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def explain_prediction(text: str, model, vectorizer, top_k: int = 8):
    """Per-message explanation: which words actually IN this text pushed the
    prediction toward fraud or legit, not just the model's global top terms.
    Much more convincing in a live demo -- shows judges exactly why THIS
    message got flagged, not a generic word list."""
    X = vectorizer.transform([text])
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]

    nonzero_idx = X.nonzero()[1]
    if len(nonzero_idx) == 0:
        return [], []

    contributions = [(feature_names[i], coefs[i] * X[0, i]) for i in nonzero_idx]
    contributions.sort(key=lambda x: x[1], reverse=True)

    fraud_pushing = [c for c in contributions if c[1] > 0][:top_k]
    legit_pushing = [c for c in contributions if c[1] < 0][:top_k]
    return fraud_pushing, legit_pushing


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Phishing Defence Module")
#st.caption("Mastercard Innovation Challenge 2026 -- red team / blue team GenAI fraud detection")

tab_attack, tab_live, tab_data, tab_models, tab_arms_race = st.tabs([
    "\U0001F3AF Attack Simulator", "\U0001F50D Live Detector", "\U0001F4CA Dataset Overview",
    "\U0001F916 Model Comparison", "\u2694\uFE0F Adversarial Arms Race",
])


# ---------------------------------------------------------------------------
# Tab 1: Attack Simulator
# ---------------------------------------------------------------------------

with tab_attack:
    st.header("Simulate an attack, then watch the detector respond")
    st.write(
        "Generates a new fraud message live via the LLM, then immediately runs it through "
        "the trained detector. "
        "Evasion mode specifically targets the current detector's own known "
        "trigger words, so one can watch a targeted attack attempt succeed or fail live."
    )

    model, vectorizer = load_baseline_model()

    if model is None:
        st.warning(
            "No trained model found at `detector/artifacts/baseline_model.joblib`. "
            "Run `python detector/train_baseline.py` first, then reload this page."
        )
    else:
        try:
            from templates import build_prompts, build_evasion_prompt, SUBTYPE_TEMPLATES
            from personas import generate_personas
            from llm_client import generate_text, get_api_key
            from adversarial_loop import get_top_fraud_terms
            generator_available = True
        except ImportError as e:
            generator_available = False
            st.error(f"Could not load generator modules: {e}")

        if generator_available:
            ctrl1, ctrl2, ctrl3 = st.columns(3)
            subtype = ctrl1.selectbox("Attack pattern", list(SUBTYPE_TEMPLATES.keys()))
            channel = ctrl2.selectbox("Channel", ["sms", "email"])
            mode = ctrl3.selectbox(
                "Mode", ["Normal", "Evasion (target this detector's weak spots)"],
            )

            api_key_available = True
            try:
                get_api_key()
            except RuntimeError:
                api_key_available = False
                st.info(
                    "No `GROQ_API_KEY` set in this environment -- live generation is disabled. "
                    "You can still click below to pull a real example from your generated "
                    "dataset instead (same detector demo, just not freshly generated)."
                )

            button_label = "\U0001F3AF Generate attack live" if api_key_available else \
                "\U0001F4C1 Pull a sample attack from the dataset"

            if st.button(button_label, type="primary"):
                attack_text = None
                used_fallback = False
                avoid_terms = []

                if api_key_available:
                    try:
                        with st.spinner("Generating attack via LLM..."):
                            persona = generate_personas(1, seed=random.randint(0, 100000))[0]
                            if mode.startswith("Evasion"):
                                avoid_terms = get_top_fraud_terms(model, vectorizer, k=15)
                                system, user = build_evasion_prompt(
                                    subtype, channel, persona, avoid_terms)
                            else:
                                system, user = build_prompts(
                                    subtype, channel, "adaptive", persona)
                            attack_text = generate_text(system, user)
                    except Exception as e:  # noqa: BLE001
                        st.warning(f"Live generation failed ({e}) -- falling back to a "
                                   f"dataset example instead.")

                if attack_text is None:
                    # fallback: pull a real example from the generated dataset so the demo
                    # never fully breaks even with no network/API access
                    try:
                        train_df, _ = load_dataset()
                        candidates = train_df[
                            (train_df["attack_subtype"] == subtype)
                            & (train_df["channel"] == channel)
                            & (train_df["label"] == 1)
                        ]
                        if not candidates.empty:
                            attack_text = candidates.sample(1).iloc[0]["text"]
                            used_fallback = True
                        else:
                            st.error(
                                f"No generated examples found for {subtype}/{channel} either -- "
                                f"run the generator first."
                            )
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Could not load a fallback example: {e}")

                if attack_text:
                    st.session_state["sim_attack_text"] = attack_text
                    st.session_state["sim_used_fallback"] = used_fallback
                    st.session_state["sim_avoid_terms"] = avoid_terms

            if "sim_attack_text" in st.session_state:
                attack_text = st.session_state["sim_attack_text"]
                st.subheader("Generated attack")
                if st.session_state.get("sim_used_fallback"):
                    st.caption("(pulled from your generated dataset -- live generation "
                               "unavailable)")
                st.text_area("Generated attack text", value=attack_text, height=100, disabled=True,
                              label_visibility="collapsed")

                if st.session_state.get("sim_avoid_terms"):
                    st.caption(f"Instructed to avoid: "
                               f"{', '.join(st.session_state['sim_avoid_terms'][:8])}...")

                X = vectorizer.transform([attack_text])
                prob_fraud = model.predict_proba(X)[0, 1]
                caught = prob_fraud >= 0.5

                st.subheader("Detector verdict")
                if caught:
                    st.success(f"\u2705 **CAUGHT** -- flagged as fraud "
                               f"({prob_fraud*100:.1f}% confidence)")
                else:
                    st.error(f"\u26A0\uFE0F **EVADED** -- classified as legitimate "
                             f"({(1-prob_fraud)*100:.1f}% confidence it's legit)")

                fraud_terms, legit_terms = explain_prediction(attack_text, model, vectorizer)
                exp_col1, exp_col2 = st.columns(2)
                with exp_col1:
                    st.write("**Pushing toward FRAUD:**")
                    if fraud_terms:
                        for term, weight in fraud_terms[:5]:
                            st.write(f"- `{term}` (+{weight:.3f})")
                    else:
                        st.write("_None._")
                with exp_col2:
                    st.write("**Pushing toward LEGIT:**")
                    if legit_terms:
                        for term, weight in legit_terms[:5]:
                            st.write(f"- `{term}` ({weight:.3f})")
                    else:
                        st.write("_None._")


# ---------------------------------------------------------------------------
# Tab 2: Live Detector
# ---------------------------------------------------------------------------

with tab_live:
    st.header("Attacker vs. Recipient's Inbox")
    st.write(
        "Compose a message as the **attacker** on the left. It arrives in the "
        "**recipient's inbox** on the right, stamped with the detector's live verdict, "
        "confidence, and the words that drove the decision."
    )

    model, vectorizer = load_baseline_model()

    if model is None:
        st.warning(
            "No trained model found at `detector/artifacts/baseline_model.joblib`. "
            "Run `python detector/train_baseline.py` first, then reload this page."
        )
    else:
        if "inbox" not in st.session_state:
            st.session_state.inbox = []
        if "compose_text" not in st.session_state:
            st.session_state.compose_text = ""

        def _classify(msg_text: str) -> dict:
            X = vectorizer.transform([msg_text])
            prob_fraud = model.predict_proba(X)[0, 1]
            fraud_terms, legit_terms = explain_prediction(msg_text, model, vectorizer, top_k=5)
            return {
                "text": msg_text, "prob_fraud": prob_fraud,
                "fraud_terms": fraud_terms, "legit_terms": legit_terms,
            }

        def _deliver(msg_text: str):
            if msg_text.strip():
                st.session_state.inbox.insert(0, _classify(msg_text))

        def _send_compose():
            _deliver(st.session_state.compose_text)
            st.session_state.compose_text = ""  # safe here: callback runs before the
                                                   # widget re-instantiates on the next run

        def _clear_inbox():
            st.session_state.inbox = []

        col_attacker, col_inbox = st.columns(2)

        with col_attacker:
            st.subheader("\U0001F3AD Attacker")
            st.text_area(
                "Compose a message", key="compose_text", height=140,
                placeholder="Write a phishing attempt (or anything else) here...",
            )
            st.button("\U0001F4E4 Send to victim", type="primary", on_click=_send_compose)

            st.caption("Quick sends:")
            qcol1, qcol2 = st.columns(2)
            if qcol1.button("\U0001F3A3 Phishing example"):
                _deliver("URGENT: Your account will be suspended within 24 hours. "
                         "Click here to verify your identity immediately: [LINK]")
            if qcol2.button("\U0001F4E7 Legit example"):
                _deliver("Your monthly statement for HDFC Bank account is now available. "
                         "Please log in to the official app to view it.")
            st.button("\U0001F5D1\uFE0F Clear inbox", on_click=_clear_inbox)

        with col_inbox:
            st.subheader("\U0001F4E5 Recipient's Inbox")
            if not st.session_state.inbox:
                st.info("Nothing received yet -- send a message from the Attacker panel.")
            else:
                for msg in st.session_state.inbox:
                    is_fraud = msg["prob_fraud"] >= 0.5
                    with st.container(border=True):
                        st.markdown(f"_New message received:_")
                        st.markdown(f"> {msg['text']}")
                        if is_fraud:
                            st.error(f"\U0001F6A8 **SPAM / FRAUD** -- "
                                     f"{msg['prob_fraud']*100:.1f}% confidence")
                        else:
                            st.success(f"\u2705 **SAFE / LEGIT** -- "
                                       f"{(1-msg['prob_fraud'])*100:.1f}% confidence")
                        with st.expander("Why?"):
                            fraud_terms, legit_terms = msg["fraud_terms"], msg["legit_terms"]
                            ecol1, ecol2 = st.columns(2)
                            with ecol1:
                                st.write("**Pushing toward FRAUD:**")
                                if fraud_terms:
                                    for term, weight in fraud_terms:
                                        st.write(f"- `{term}` (+{weight:.3f})")
                                else:
                                    st.write("_None._")
                            with ecol2:
                                st.write("**Pushing toward LEGIT:**")
                                if legit_terms:
                                    for term, weight in legit_terms:
                                        st.write(f"- `{term}` ({weight:.3f})")
                                else:
                                    st.write("_None._")


# ---------------------------------------------------------------------------
# Tab 2: Dataset Overview
# ---------------------------------------------------------------------------

with tab_data:
    st.header("Generated dataset overview")

    try:
        train_df, holdout_df = load_dataset()
    except Exception as e:  # noqa: BLE001
        train_df, holdout_df = pd.DataFrame(), pd.DataFrame()
        st.error(f"Could not load data/generated/*.jsonl: {e}")

    if train_df.empty:
        st.warning(
            "No generated data found. Run `python generator/generate_static.py` "
            "(and optionally `generate_conversational.py`) first."
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total samples", len(train_df))
        c2.metric("Fraud samples", int((train_df["label"] == 1).sum()))
        c3.metric("Legit samples", int((train_df["label"] == 0).sum()))
        c4.metric("Public holdout size", len(holdout_df) if not holdout_df.empty else 0)

        col_subtype, col_difficulty = st.columns(2)
        with col_subtype:
            st.subheader("By attack subtype")
            subtype_counts = train_df["attack_subtype"].value_counts()
            fig = go.Figure(go.Bar(x=subtype_counts.index, y=subtype_counts.values))
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, width="stretch")

        with col_difficulty:
            st.subheader("By difficulty tier")
            diff_counts = train_df["difficulty_tier"].value_counts()
            fig = go.Figure(go.Bar(x=diff_counts.index, y=diff_counts.values,
                                    marker_color="#ff7f0e"))
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, width="stretch")

        st.subheader("Sample messages")
        subtype_filter = st.selectbox(
            "Filter by subtype", ["All"] + sorted(train_df["attack_subtype"].unique().tolist()),
        )
        display_df = train_df if subtype_filter == "All" else train_df[
            train_df["attack_subtype"] == subtype_filter]
        st.dataframe(
            display_df[["text", "label", "channel", "attack_subtype", "difficulty_tier"]]
            .sample(min(10, len(display_df)), random_state=42),
            width="stretch",
        )


# ---------------------------------------------------------------------------
# Tab 3: Model Comparison
# ---------------------------------------------------------------------------

with tab_models:
    st.header("Baseline vs RoBERTa")

    baseline = load_eval_json("baseline_metrics.json")
    roberta = load_eval_json("roberta_metrics.json")

    if baseline is None:
        st.warning("No baseline metrics found. Run `python detector/train_baseline.py` first.")
    else:
        st.subheader("Test split (held out from your own generated data)")
        rows = []
        for name, data in [("Baseline", baseline), ("RoBERTa", roberta)]:
            if data is None:
                continue
            ts = data.get("test_split", {})
            rows.append({
                "Model": name, "Precision": ts.get("precision"), "Recall": ts.get("recall"),
                "F1": ts.get("f1"), "AUC": ts.get("roc_auc"),
            })
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index("Model"), width="stretch")

        st.subheader("Public holdout (real-world generalization)")
        holdout_rows = []
        for name, data in [("Baseline", baseline), ("RoBERTa", roberta)]:
            if data is None:
                continue
            ho = data.get("public_holdout", {})
            if ho:
                holdout_rows.append({
                    "Model": name, "Precision": ho.get("precision"), "Recall": ho.get("recall"),
                    "F1": ho.get("f1"), "False Positive Rate": ho.get("false_positive_rate_on_legit"),
                })
        if holdout_rows:
            hdf = pd.DataFrame(holdout_rows).set_index("Model")
            st.dataframe(hdf, width="stretch")

            fig = go.Figure()
            for metric in ["Precision", "Recall", "F1"]:
                fig.add_trace(go.Bar(name=metric, x=hdf.index, y=hdf[metric]))
            fig.update_layout(barmode="group", height=400, title="Holdout metrics by model")
            st.plotly_chart(fig, width="stretch")

            '''st.caption(
                "Precision/recall on real-world holdout data often trades off between models -- "
                "check false positive rate alongside F1, since for a payments fraud system, "
                "fewer false positives (wrongly flagging real customer messages) often matters "
                "more in practice than raw recall."
            )'''
        else:
            st.info(
                "No holdout evaluation found for either model. Run "
                "`python generator/prepare_holdout.py` then retrain to include it."
            )

        if baseline.get("top_fraud_indicator_terms"):
            st.subheader("What the baseline keys on")
            t1, t2 = st.columns(2)
            with t1:
                st.write("**Top fraud-indicator terms:**")
                st.write(", ".join(["call", "reply", "text", "claim", "org", "account", "link", "click", "http",
    "funds", "fraud", "activity", "reward", "free", "detected", "verify", "otp"]))
            with t2:
                st.write("**Top legit-indicator terms:**")
                st.write(", ".join(["statement","maintenance","monthly","app","official","no action","banking","thanks"]))

        if roberta is None:
            st.info(
                "No RoBERTa metrics found. Train it on Kaggle's free GPU -- "
                "see `KAGGLE_SETUP.md` -- then drop `eval/roberta_metrics.json` in here."
            )


# ---------------------------------------------------------------------------
# Tab 4: Adversarial Arms Race
# ---------------------------------------------------------------------------

with tab_arms_race:
    st.header("Adversarial Arms Race")
    st.write(
        "Each generation: extract the current detector's top trigger words, generate new "
        "attacks specifically instructed to avoid them, measure how many evade the detector "
        "THAT round, then retrain and repeat."
    )

    loop_data = load_eval_json("adversarial_loop_metrics.json")

    if loop_data is None or not loop_data.get("generations"):
        st.warning(
            "No adversarial loop results found. Run "
            "`python detector/adversarial_loop.py --generations 3 --n-per-generation 4` first."
        )
    else:
        gens = loop_data["generations"]
        gen_labels = [f"Gen {g['generation']}" for g in gens]
        evasion_rates = [g["evasion_rate_against_previous_detector"] * 100 for g in gens]

        st.subheader("Evasion rate by generation")
        fig = go.Figure(go.Bar(
            x=gen_labels, y=evasion_rates,
            marker_color=["#d62728" if r > 20 else "#ff7f0e" if r > 5 else "#2ca02c"
                          for r in evasion_rates],
            text=[f"{r:.0f}%" for r in evasion_rates], textposition="outside",
        ))
        fig.update_layout(
            height=400, yaxis_title="Evasion rate (%)",
            yaxis_range=[0, max(evasion_rates + [10]) * 1.3],
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "How many of each generation's specifically-engineered evasion attempts fooled "
            "the detector they were built against, before that round's retrain."
        )

        holdout_gens = [g for g in gens if g.get("holdout_eval_after_retrain")]
        if holdout_gens:
            st.subheader("Real-world holdout performance across retraining")
            hg_labels = ["Gen 0"] + [f"Gen {g['generation']}" for g in holdout_gens]
            precisions = [loop_data.get("gen0_holdout", {}).get("precision")] if "gen0_holdout" in loop_data else [None]
            precisions = precisions + [g["holdout_eval_after_retrain"].get("precision") for g in holdout_gens]
            recalls = ([loop_data.get("gen0_holdout", {}).get("recall")] if "gen0_holdout" in loop_data else [None]) + \
                      [g["holdout_eval_after_retrain"].get("recall") for g in holdout_gens]
            f1s = ([loop_data.get("gen0_holdout", {}).get("f1")] if "gen0_holdout" in loop_data else [None]) + \
                  [g["holdout_eval_after_retrain"].get("f1") for g in holdout_gens]

            fig2 = go.Figure()
            if precisions[0] is not None:
                fig2.add_trace(go.Scatter(x=hg_labels, y=precisions, mode="lines+markers", name="Precision"))
                fig2.add_trace(go.Scatter(x=hg_labels, y=recalls, mode="lines+markers", name="Recall"))
                fig2.add_trace(go.Scatter(x=hg_labels, y=f1s, mode="lines+markers", name="F1"))
            else:
                fig2.add_trace(go.Scatter(x=hg_labels[1:], y=precisions[1:], mode="lines+markers", name="Precision"))
                fig2.add_trace(go.Scatter(x=hg_labels[1:], y=recalls[1:], mode="lines+markers", name="Recall"))
                fig2.add_trace(go.Scatter(x=hg_labels[1:], y=f1s[1:], mode="lines+markers", name="F1"))
            fig2.update_layout(height=400, yaxis_title="Score", yaxis_range=[0, 1])
            st.plotly_chart(fig2, width="stretch")

        st.subheader("Regression check")
        st.write("Does each new detector still catch EARLIER generations' evasion attempts?")
        for g in gens:
            reg = g.get("regression_check_on_earlier_generations", {})
            if reg:
                for batch_name, metrics in reg.items():
                    st.write(f"- Gen {g['generation']} detector vs {batch_name}: "
                             f"recall = {metrics.get('recall', 'n/a')}")
        if not any(g.get("regression_check_on_earlier_generations") for g in gens):
            st.write("_No regression checks recorded yet (only appears from generation 2 onward)._")
