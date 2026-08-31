"""Landing page: what the five modules are and how they fit together."""

from __future__ import annotations

import streamlit as st

from shell import services
from shell.loader import MODULES

st.title("🛡️ AI Defense Lab")
st.caption(
    "Mastercard Innovation Challenge 2026 — five red-team vs blue-team modules "
    "for GenAI-era payment fraud, built independently and merged into one app."
)

st.markdown(
    """
Every module follows the same shape, because that is the brief: **build the
attack, then build the defence, then measure whether the defence held.** Pick
one from the sidebar.
"""
)

MODULE_CARDS = [
    (
        "🧾 Chargeback (Aegis)",
        "chargeback",
        "Reason code 4853 — 'defective merchandise'. A red-team agent holds a real "
        "turn-by-turn chat with a jailbreakable support bot and submits two AI-generated "
        "damage photos; the blue team screens it with a prompt-injection sanitiser, a "
        "forensic vision inspector, deterministic EXIF/C2PA metadata forensics, and a "
        "supervisor that never sees the raw transcript.",
    ),
    (
        "🔐 Account Takeover",
        "account_takeover",
        "Credential-stuffing burst → breach login → rapid cash-out, injected into real "
        "transaction data with a shared attacker botnet so the attack has a realistic "
        "infrastructure signature. Blue team scores every event with XGBoost blended with "
        "an Isolation Forest and turns the score into an action, not just a label.",
    ),
    (
        "🎣 Phishing",
        "phishing",
        "LLM-generated phishing across 5 subtypes × 2 channels × 3 difficulty tiers, plus "
        "multi-turn vishing. The arms race is the point: extract the detector's own top "
        "fraud terms, generate attacks that avoid exactly those words, measure the evasion "
        "rate, retrain, repeat.",
    ),
    (
        "🪪 Identity & Onboarding",
        "identity",
        "20 synthetic-identity attack classes against a synthetic applicant population, a "
        "detector with SHAP explanations, a NetworkX identity graph for fraud rings, and a "
        "feedback engine that mines false negatives into harder attacks and retrains.",
    ),
    (
        "💸 AML / Synthetic Data",
        "aml",
        "The dataset the rest are measured like: 8 laundering typologies injected alongside "
        "5 benign lookalikes that exist to generate hard negatives. Four detectors compared "
        "on identical splits — the graph model wins on every typology.",
    ),
]

for title, directory, blurb in MODULE_CARDS:
    with st.container(border=True):
        st.markdown(f"**{title}** &nbsp;·&nbsp; `modules/{directory}/`")
        st.markdown(blurb)

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Runs offline")
    st.markdown(
        """
No module needs the network or a GPU on its demo path. Every API key is
optional and every provider has a working fallback:

- **Chargeback** without `ANTHROPIC_API_KEY` uses a deterministic mock provider
  — real chat flow, PIL-drawn evidence, zero cost.
- **Phishing** without `GROQ_API_KEY` samples a real message from its generated
  dataset instead of writing a new one live.
- **Identity** without a local Ollama daemon falls back to rule-based discovery.
- **Account Takeover** and **AML** need no configuration at all.

Copy `.env.example` to `.env` only if you want the live paths.
"""
    )

with right:
    st.subheader("Service status")
    st.caption(
        "Four modules run inside this Streamlit process. Identity is FastAPI + React "
        "and runs alongside it."
    )
    if services.missing_backend_package() is not None:
        st.error(
            "Identity's `backend/data/` source package is missing — see the "
            "Identity & Onboarding page."
        )
    for service in services.SERVICES:
        up = service.is_up()
        st.markdown(
            f"{'🟢' if up else '⚪'} **{service.name}** — "
            f"{'running at ' + service.url if up else 'stopped'}"
        )

    missing_modules = [d for _, d, _ in MODULE_CARDS if not (MODULES / d).is_dir()]
    if missing_modules:
        st.warning(f"Missing module directories: {', '.join(missing_modules)}")
