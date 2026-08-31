# AI Defense Lab

**Mastercard Innovation Challenge 2026.** Five red-team vs blue-team modules for
GenAI-era payment fraud, built independently on five branches and merged into one
application.

Every module follows the same shape, because that is the brief: **build the attack,
then build the defence, then measure whether the defence held.**

| Module | Sub-topic | Directory | Stack |
|---|---|---|---|
| 🧾 Chargeback (Aegis) | Chargeback fraud, reason code 4853 | `modules/chargeback/` | Streamlit |
| 🔐 Account Takeover | Authentication & account takeover | `modules/account_takeover/` | Streamlit |
| 🎣 Phishing | Social engineering & phishing at scale | `modules/phishing/` | Streamlit |
| 🪪 Identity & Onboarding | Identity & onboarding (KYC) fraud | `modules/identity/` | FastAPI + React |
| 💸 AML / Synthetic Data | Synthetic laundering data + GNN detectors | `modules/aml/` | CLI + a read-only page |

Each module keeps its own `README.md`, `requirements.txt` and `.gitignore`. Read
those for the depth — this file only covers running the whole thing.

## Quick start

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py          # http://localhost:8501
```

That is the entire demo for four of the five modules. **No network, no API keys and
no GPU are needed on any demo path** — every provider has a working offline
fallback, so nothing here dies on a flaky conference connection.

Copy `.env.example` to `.env` only if you want the live paths (a real Anthropic
model driving the chargeback simulation, live Groq generation in the phishing
attack simulator). Every key in it is optional.

### The identity module

It is FastAPI + React, so it runs alongside the Streamlit app rather than inside
it. Its page embeds the running UI and can start both processes for you, or:

```bash
python -m venv modules/identity/.venv
modules/identity/.venv/bin/pip install -r modules/identity/requirements.txt
cd modules/identity/frontend && npm install && cd ../../..

# terminal 1 -- cwd must be modules/identity: every import is `from backend.X`
cd modules/identity && uvicorn backend.main:app --reload --port 8000
# terminal 2
cd modules/identity/frontend && npm run dev          # http://localhost:5173
```

It gets **its own virtualenv on purpose**: it pins `scikit-learn==1.5.2` and
`numpy==1.26.4`, which cannot coexist with the pins the other modules need.

> **Known gap:** `modules/identity/backend/data/` is missing — the package holding
> `generator.py` and `schemas.py`. Twelve files import it at module scope, so the
> backend will not start until it is restored. It is not the module's generated
> data directory (that is `modules/identity/data/`, created automatically), and
> `scripts/generate_data.py` cannot produce it because that script imports it too.
> It was lost to a bare `data/` rule in the root `.gitignore`, which matches at any
> depth. That rule is fixed, so dropping the two files back in will commit them.

## How the merge works

Three of the five branches each shipped a *different* root `app.py`, all Streamlit,
all calling `st.set_page_config()` — which Streamlit permits once per process. One
of them (`ChatBot-Fraud-Detection`) was an orphan branch with no common ancestor at
all, so `git merge` refused it outright.

Rather than rewrite anyone's code, each branch was grafted into its own
subdirectory with `git subtree add`, which preserves every contributor's commits
and authorship:

```bash
git log --oneline --graph          # the subtree merges, with each branch as a parent
git log -- modules/phishing        # that contributor's history, intact
```

`app.py` at the root is a thin shell: it makes the single `set_page_config` call and
builds the sidebar with `st.navigation`. `shell/loader.py` then runs each module's
own `app.py` **verbatim** — it puts the module's directory on `sys.path` (which is
what `streamlit run` would have done) and suppresses the module's own
`set_page_config` for the duration. So no module file was edited, every module still
runs standalone from its own directory, and the team can keep pulling from their
branches:

```bash
cd modules/account_takeover && streamlit run app.py
```

The one piece of genuinely new UI is `shell/pages/aml.py`, a read-only view over the
artifacts the AML pipeline already committed.

## Layout

```
app.py                  shell entry point -- the only st.set_page_config()
shell/
  loader.py             runs a module's app.py in the context it expects
  services.py           port probes + launcher for the identity processes
  pages/                one page per module
modules/
  chargeback/           each module's tree, exactly as its branch had it
  account_takeover/
  phishing/
  identity/
  aml/
requirements.txt        shell + the four in-process modules
.env.example            every module's env vars, all optional
```

## Tests

```bash
cd modules/chargeback && python -m pytest tests/ -v   # offline, no keys needed
cd modules/identity   && pytest tests/ -q             # blocked on backend/data/
```
