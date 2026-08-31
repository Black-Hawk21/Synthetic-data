"""Identity & Onboarding fraud -- the one module that is not a Streamlit app.

It is a FastAPI backend plus a React/Vite frontend, so this page reports on
those two processes, offers to start them, and embeds the running UI.
"""

from __future__ import annotations

import streamlit as st

from shell import services

st.title("🪪 Identity & Onboarding Fraud")
st.caption(
    "Closed-loop red team vs blue team for KYC onboarding: 20 synthetic-identity "
    "attack classes, a LogReg → RandomForest → XGBoost detector with SHAP "
    "explanations, a NetworkX identity graph for fraud rings, and a feedback "
    "engine that mines false negatives into harder attacks and retrains."
)

missing = services.missing_backend_package()
if missing is not None:
    st.error(
        "**The backend cannot start: `backend/data/` is missing.**\n\n"
        "This is the module's hand-written source package — `generator.py` "
        "(`generate_legitimate_applicants`) and `schemas.py` "
        "(`ALL_FEATURE_GROUPS`, `CATEGORICAL_MODEL_FEATURES`, `DROP_FROM_MODEL`, "
        "`ATTACK_TYPES`). Twelve files import it at module scope, including "
        "`backend/red_team/base.py`, which `backend/main.py` pulls in at startup.\n\n"
        "It is **not** the generated data directory — that is "
        "`modules/identity/data/`, which `backend/config.py` creates on import. "
        "And `scripts/generate_data.py` cannot bootstrap it: line 22 of that "
        "script imports `backend.data.generator` itself.\n\n"
        "It was never committed because the root `.gitignore` carried a bare "
        "`data/` rule, which matches at any depth. That rule is fixed now, so "
        "dropping the two files back in will commit them — recover them from "
        "the module author's working tree."
    )

status = {service.name: service.is_up() for service in services.SERVICES}
cols = st.columns(len(services.SERVICES))
for col, service in zip(cols, services.SERVICES):
    up = status[service.name]
    col.metric(service.name, "running" if up else "stopped", service.url)
    col.caption(service.note)

everything_up = all(status.values())

if everything_up:
    st.success("Both services are up — the live UI is embedded below.")
    st.components.v1.iframe(services.FRONTEND.url, height=900, scrolling=True)
else:
    if not services.frontend_deps_installed():
        st.warning(
            "The frontend has no `node_modules/` yet. Run "
            "`cd modules/identity/frontend && npm install` once before starting it."
        )

    if st.button("Start the identity services", type="primary"):
        for service in services.SERVICES:
            if not status[service.name]:
                try:
                    services.start(service)
                except (OSError, FileNotFoundError) as exc:
                    st.error(f"Could not start {service.name}: {exc}")
        st.info("Launching — give them a few seconds, then rerun this page.")

    st.markdown(
        """
Or start them by hand, in two terminals:

```bash
# terminal 1 -- cwd must be modules/identity: every import is `from backend.X`
cd modules/identity && uvicorn backend.main:app --reload --port 8000

# terminal 2
cd modules/identity/frontend && npm run dev
```

Then reload this page. The UI is also reachable directly at
<http://localhost:5173>.

**Do not use this module's `docker-compose.yml` for the demo.** Its frontend
image serves the static build with `serve` and no proxy, so the relative
`/api` calls 404; and it bind-mounts `identity_fraud.db`, a gitignored file
that does not exist, so Docker creates a directory there and SQLite fails.
`npm run dev` is the working path.
"""
    )
