# AI Defense Lab

**Mastercard Innovation Challenge @ GFF 2026** — *Build the attack, then build the defense.*

Five red-team vs blue-team systems for GenAI-era payment fraud, each covering a
different attack surface of the payment lifecycle, merged into one web application.

| Module | Attack surface | Directory | Headline result |
|---|---|---|---|
| 🧾 **Chargeback (Aegis)** | Dispute fraud, MCC reason 4853 | `modules/chargeback/` | Supervisor never sees raw transcript |
| 🔐 **Account Takeover** | Authentication & ATO | `modules/account_takeover/` | 0.845 precision / 0.990 recall |
| 🎣 **Phishing** | Social engineering at scale | `modules/phishing/` | F1 0.9808, 0.36% FPR on real mail |
| 🪪 **Identity & Onboarding** | KYC / synthetic identity | `modules/identity/` | XGBoost F1 0.900, PR-AUC 0.948 |
| 💸 **AML / Synthetic Data** | Transaction laundering | `modules/aml/` | PR-AUC 0.6447 → 0.8845 with graph |

**Submission artifacts:** the prototype is `app.py`; the Solution Walkthrough is
[`docs/AI_Defense_Lab_Solution_Walkthrough.docx`](docs/AI_Defense_Lab_Solution_Walkthrough.docx);
the demo run sheet is [`docs/DEMO.md`](docs/DEMO.md).

---

## Setup

### Prerequisites

- **Python 3.12+** (developed on 3.12.3)
- **Node 20+ and npm** — only for the Identity module's React frontend
- ~1 GB disk for the virtualenv; the repo itself is ~250 MB because every
  module's trained models and evaluation artifacts are committed

### Install and run — 2 commands

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

Open **http://localhost:8501**. That is the complete demo for four of the five
modules.

> **It runs fully offline.** No network, no API keys, no GPU is needed on any
> demo path. Every LLM provider has a deterministic fallback, so nothing here
> depends on conference wifi.

### Optional — enable the live LLM paths

Only two things change behaviour, and both degrade gracefully without a key:

```bash
cp .env.example .env      # then fill in what you want
```

| Key | Enables | Without it |
|---|---|---|
| `ANTHROPIC_API_KEY` | Chargeback runs its red-team/blue-team chat against real models | Deterministic mock provider — real chat flow, PIL-drawn evidence, zero cost |
| `GROQ_API_KEY` | Phishing generates a brand-new attack live in the Attack Simulator | Samples a real message from the generated dataset |

For chargeback you can also paste a key straight into the page sidebar during a
demo — it is session-only and never written to disk.

### Optional — the Identity module

It is FastAPI + React, so it runs as two processes alongside the Streamlit app.
Its page in the shell embeds the running UI and can start both for you.

```bash
# 1. its own virtualenv -- it pins scikit-learn==1.5.2, which cannot coexist
#    with the pins the other modules need
python -m venv modules/identity/.venv
modules/identity/.venv/bin/pip install -r modules/identity/requirements.txt

# 2. frontend dependencies (one time)
cd modules/identity/frontend && npm install && cd ../../..

# 3. two terminals
cd modules/identity && uvicorn backend.main:app --reload --port 8000
cd modules/identity/frontend && npm run dev            # http://localhost:5173
```

> ⚠️ **Known gap.** `modules/identity/backend/data/` is missing — the package
> holding `generator.py` and `schemas.py`. Twelve files import it at module
> scope, so the backend will not start until it is restored.
>
> This is **not** the module's generated data directory (that is
> `modules/identity/data/`, which `backend/config.py` creates automatically),
> and `scripts/generate_data.py` cannot produce it because line 22 of that
> script imports `backend.data.generator` itself.
>
> It was lost to a bare `data/` rule in the root `.gitignore`, which matches at
> any depth. That rule is now anchored, so dropping the two files back in will
> commit them normally.

### Run a single module standalone

Every module still works exactly as its author built it — no module file was
edited during the merge:

```bash
cd modules/chargeback       && streamlit run app.py
cd modules/account_takeover && streamlit run app.py
cd modules/phishing         && streamlit run app.py
```

### Verify the install

```bash
cd modules/chargeback && python -m pytest tests/ -v     # 35 tests, offline
```

---

## Reproducing the results

Nothing in the walkthrough document is hand-typed — regenerate it and it re-reads
every number from the repository:

```bash
.venv/bin/python docs/generate_walkthrough.py
```

| To rebuild | Command | Time |
|---|---|---|
| AML dataset | `cd modules/aml && python run.py all` | ~25 s |
| AML tabular models | `cd modules/aml && python train_models.py` | ~2 min |
| AML GNNs | `cd modules/aml && python train_gnn.py` | ~30 min, 1 core |
| All four AML detectors compared | `cd modules/aml && python compare_models.py` | ~3 min |
| ATO pipeline | `cd modules/account_takeover && python 02_baseline_and_attack.py && python 03_features.py && python 04_train_detect.py && python 05_mitigate_and_demo.py` | ~5 min |
| Phishing detector | `cd modules/phishing/detector && python train_baseline.py` | ~1 min |
| Phishing adversarial loop | `cd modules/phishing/detector && python adversarial_loop.py --generations 3` | needs `GROQ_API_KEY` |

Seeds are fixed throughout (AML `20260822`, ATO `42`, identity `42`), so a rerun
reproduces the reported numbers rather than something close to them.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'streamlit'` | You are using system Python. Use `.venv/bin/streamlit`, or activate the venv first. |
| `InconsistentVersionWarning: ... from version 1.7.2` | Expected and harmless. Models were pickled with scikit-learn 1.7.2/1.8.0; the pinned 1.8.0 reads both. |
| Account Takeover page is slow the first time | It reads a 50 MB scored-events CSV. Cached after the first visit (~0.4 s thereafter). |
| `set_page_config() can only be called once` | A module page bypassed `shell/loader.py`. Each page wrapper must call `run_module_app`. |
| Identity page shows both services stopped | Start them by hand (above). Do **not** use that module's `docker-compose.yml` — its frontend image has no `/api` proxy and its DB bind-mount is broken. |
| Chargeback shows `MOCK` and you wanted live | Paste a key into the page sidebar; no restart needed. |
| A live generation stalls mid-demo | It has already fallen back to cached data. Keep going — the tab still works. |

---

## How the five branches were merged

Three of the five branches each shipped a *different* root `app.py`, all
Streamlit, all calling `st.set_page_config()` — which Streamlit permits once per
process. One (`ChatBot-Fraud-Detection`) was an orphan branch with no common
ancestor at all, so `git merge` refused it outright.

Rather than rewrite anyone's code, each branch was grafted into its own
subdirectory with `git subtree add`, which preserves every contributor's commits
and authorship:

```bash
git log --oneline --graph        # subtree merges, each branch still a parent
git log -- modules/phishing      # that contributor's history, intact
```

Root `app.py` is a thin shell: it makes the single `set_page_config` call and
builds the sidebar with `st.navigation`. `shell/loader.py` then runs each
module's own `app.py` **verbatim** — it puts the module's directory on
`sys.path` (what `streamlit run` would have done) and suppresses the module's
own `set_page_config` for the duration. So the team can keep pulling from their
branches, and each module still runs standalone.

```
app.py                  shell entry point -- the only st.set_page_config()
shell/
  loader.py             runs a module's app.py in the context it expects
  services.py           port probes + launcher for the identity processes
  pages/                one page per module
modules/                each module's tree, exactly as its branch had it
docs/
  generate_walkthrough.py   regenerates the submission .docx from live data
  DEMO.md                   run sheet for presenting
```

Each module keeps its own `README.md` — with full methodology, caveats and
results — plus its own `requirements.txt` and test suite. Read those for depth.
