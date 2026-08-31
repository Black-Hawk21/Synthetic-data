"""AI Defense Lab -- the unified demo shell.

Five hackathon modules, each originally its own root-level application, are
presented here as pages of one Streamlit app:

    modules/chargeback        Streamlit, hosted in-process
    modules/account_takeover  Streamlit, hosted in-process
    modules/phishing          Streamlit, hosted in-process
    modules/aml               CLI pipeline, presented read-only by shell/pages/aml.py
    modules/identity          FastAPI + React, run alongside and embedded

Run it with:

    streamlit run app.py

This file owns the only st.set_page_config() call in the process. The three
hosted Streamlit modules make their own call, which shell/loader.py suppresses
for the duration of their run so that their app.py files need no edits at all
-- they still run standalone from their own directory, exactly as their
authors wrote them.
"""

import sys
from pathlib import Path

import streamlit as st

# Ensure `import shell...` resolves when Streamlit runs this file directly.
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

st.set_page_config(
    page_title="AI Defense Lab",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

navigation = st.navigation(
    [
        st.Page("shell/pages/overview.py", title="Overview", icon="🛡️", default=True),
        st.Page("shell/pages/chargeback.py", title="Chargeback (Aegis)", icon="🧾"),
        st.Page("shell/pages/account_takeover.py", title="Account Takeover", icon="🔐"),
        st.Page("shell/pages/phishing.py", title="Phishing", icon="🎣"),
        st.Page("shell/pages/identity.py", title="Identity & Onboarding", icon="🪪"),
        st.Page("shell/pages/aml.py", title="AML / Synthetic Data", icon="💸"),
    ]
)
navigation.run()
