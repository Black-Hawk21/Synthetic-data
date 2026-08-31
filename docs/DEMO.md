# Demo run sheet

## Before the room

1. **Clone and build once, not on the day.** The repo carries every module's
   committed artifacts (~250 MB working tree), so a cold clone is slow.
   ```bash
   python -m venv .venv && .venv/bin/pip install -r requirements.txt
   ```
2. **Decide live or offline.** Everything works with no `.env` at all. If you want
   the live paths, copy `.env.example` to `.env` and fill in `ANTHROPIC_API_KEY`
   (chargeback) and/or `GROQ_API_KEY` (phishing). A chargeback key can also be
   pasted into the page sidebar mid-demo — session-only, never written to disk.
3. **Identity module**, only if you are showing it:
   ```bash
   python -m venv modules/identity/.venv
   modules/identity/.venv/bin/pip install -r modules/identity/requirements.txt
   cd modules/identity/frontend && npm install
   ```
   Start both processes before you present — the page can launch them, but not
   instantly. **This module is currently blocked**; see the note in the root README.
4. **Open it once and click every page** before the room fills. The first visit to
   Account Takeover reads a 50 MB CSV; after that it is cached.

## Running it

```bash
.venv/bin/streamlit run app.py      # http://localhost:8501
```

## Suggested order — about 8 minutes

The narrative is the same in every module, so lead with it once and let the rest
land fast: *build the attack, build the defence, measure whether it held.*

**1 · Overview (30 s).** Five sub-topics, one shell, five branches merged with
history intact. Say the offline line here: no network, no keys, no GPU.

**2 · Chargeback / Aegis (2.5 min).** The strongest live moment.
- *Live Simulation* → pick a tactic, **Run Next Round**. The red team chats with
  the support bot and submits two angle photos; the blue team rejects; the red team
  mutates and escalates to img2img conditioning. Let one full round play.
- Call out that the supervisor **never sees the raw transcript** — only sanitised
  structured signals. That is the prompt-injection defence.
- *Batch Evaluation* → **Run Batch** for the ROC curve, if there is time.

**3 · Account Takeover (1.5 min).**
- *Overview* → the action bands. Emphasise it emits `allow` / `step_up_auth` /
  `block`, not a label — that is the product surface.
- *Botnet Ring* → the shared-device fan-out. This is the signal that only exists
  because the attack reuses infrastructure, like the real thing.

**4 · Phishing (1.5 min).**
- *Live Detector* → paste anything, get an instant per-word explanation. Fully
  local, safe to let a judge type into.
- *Adversarial Arms Race* → the evasion rate by generation. This is the headline:
  attacks written specifically to avoid the detector's own top terms, then the
  detector retrained on them.

**5 · AML / Synthetic Data (1.5 min).**
- *Dataset* → the benign lookalikes. A payroll fan-out and a laundering fan-out
  have the same shape, so shape alone cannot decide.
- *Detection quality* → **57% of false positives land on those lookalikes.** That
  is the dataset working as designed, and it is the most interesting slide here.
- *Tabular vs GNN* → 0.6447 → 0.8845 account PR-AUC, winning on every typology.

**6 · Identity & Onboarding (1 min).** The closed loop: mine false negatives into
harder attacks, retrain, measure before/after recall on an untouched holdout.

## If something goes wrong

| Symptom | Fix |
|---|---|
| A module's live generation stalls | It has already fallen back to cached data — keep going, the tab still works. |
| Chargeback shows `MOCK` and you wanted live | Paste a key into the page sidebar; no restart needed. |
| Account Takeover feels slow on first load | It is reading the 50 MB scored-events CSV. It is cached after the first visit. |
| Identity page shows both services stopped | Start them by hand (see the page). Do **not** reach for `docker-compose.yml` — its frontend image has no `/api` proxy and its DB bind-mount is broken. |
| `set_page_config() can only be called once` | A module's `set_page_config` escaped `shell/loader.py`. Check the page wrapper calls `run_module_app`. |
