# AI Defense Lab — Authentication & Account Takeover (ATO)

Mastercard Innovation Challenge 2026 — sub-topic: **Authentication & Account Takeover**.

## What this is

An end-to-end red team / blue team pipeline for GenAI-era account-takeover fraud:

1. **Red team** (`02_baseline_and_attack.py`) — simulates the ATO kill chain
   (credential-stuffing burst → breach login → rapid cash-out) at scale, on
   top of real transaction data, using a shared "botnet" identity pool across
   victims so the attack has a realistic infrastructure-reuse signature.
2. **Blue team, features** (`03_features.py`) — leak-safe, point-in-time
   behavioral features: device/IP/geo novelty, login velocity, spend
   deviation, and cross-account device/IP fan-out (who else has this
   device/IP touched recently — the graph-lite signal for shared attacker
   infrastructure).
3. **Blue team, detection** (`04_train_detect.py`) — XGBoost (supervised, on
   our injected labels) blended with an Isolation Forest (unsupervised,
   label-blind) so the system also has a shot at attack patterns it wasn't
   explicitly trained on.
4. **Blue team, mitigation** (`05_mitigate_and_demo.py`) — turns the risk
   score into an action (`allow` / `step_up_auth` / `block`), reports
   precision/recall by action band, surfaces the shared-device botnet ring,
   and prints full attack-episode timelines for the demo.

## Why this dataset, and an important caveat

We used your Kaggle dataset (`financial_fraud_detection_dataset.csv`,
5M rows). Before building on it, we checked whether `device_used`,
`ip_address`, and `location` were persistent per account — that's the
entire premise of ATO detection (a "new device for this account" signal
needs the account to *have* a usual device). They aren't: every
transaction, fraud or not, has an essentially random device/IP/location.
So those three raw columns are **not used** for ATO logic.

What we do use from the real data: `amount`, `timestamp`, `transaction_type`,
`merchant_category`, `payment_channel`, and the dataset's own precomputed
`spending_deviation_score` / `velocity_score` / `geo_anomaly_score`, all
keyed per `sender_account`. We layer our **own** internally-consistent
synthetic identity/session model on top (home device, home IP prefix, home
country per account; a shared attacker-device pool for the injected
attacks). This is normal practice for ATO research — no public transaction
dataset ships with labeled login sessions — but it means the device/IP/geo
*features* are demonstrating the detection logic correctly, not detecting
something latent in Mastercard-scale real traffic. Say this plainly if asked
in Q&A; it's a defensible modeling choice, not a hidden shortcut.

We also hit and fixed a real leakage bug worth mentioning in your writeup:
our first pass gave every synthetic event a `NaN` (→ 0) for the dataset's
precomputed scores, which perfectly fingerprinted "synthetic" vs. "real
row" and produced a suspicious 1.000 accuracy. Fix: added benign login
events (including occasional benign failed logins, e.g. mistyped
passwords) so `event_type` isn't a perfect label proxy, and gave cash-out
events sampled elevated-but-overlapping scores instead of a sentinel value.
Final held-out numbers: **0.845 precision / 0.990 recall** on the attack
class, ROC-AUC 0.9998 — strong, but with a believable error profile rather
than a suspicious perfect one.

## Pipeline results (this run)

- 20,000 accounts sampled from the 5M-row dataset (accounts with ≥4
  transactions, to have enough history for a baseline).
- 1,000 accounts (5%) had an ATO episode injected.
- 270,635 total events; 13,425 labeled ATO events.
- Action bands: **12,913 / 13,425 attack events blocked outright**, 493
  sent to step-up auth, only 19 waved through as `allow`. Only 39 benign
  events wrongly blocked (0.3% of all blocks).

## How to run

```bash
pip install -r requirements.txt
python3 02_baseline_and_attack.py   # red team: builds artifacts/events.csv
python3 03_features.py              # blue team: builds artifacts/features.csv
python3 04_train_detect.py          # blue team: trains models, builds artifacts/scored_events.csv
python3 05_mitigate_and_demo.py     # mitigation + demo walkthrough (prints to console)
```

Run them in that order — each script reads the previous one's output. The
sampled Kaggle data is already included at `data/sampled_transactions.csv`
(20k accounts, ~127k transactions), so this runs out of the box with no
extra downloads. Full run time is a few minutes on a laptop.

To rerun against different/more data: replace `data/sampled_transactions.csv`
with your own CSV (same column names — see `config.py`), or point
`RAW_CSV` in `config.py` at a different path.

## Frontend (Streamlit dashboard)

```bash
streamlit run app.py
```

Four views:
- **Overview** — KPIs (blocked/missed/false-positive counts), risk score
  distribution, action-band breakdown.
- **Botnet Ring** — shared attacker devices ranked by distinct victims
  touched, with drill-down into which accounts each one hit.
- **Attack Episode Playback** — pick an injected attack episode, step
  through its timeline with a slider, watch a live risk gauge and a
  risk-over-time line chart update as the kill chain unfolds. This is the
  best tab for a live demo.
- **Explore Events** — filterable table across all events (by type, action,
  risk range, account).

It reads whatever's in `artifacts/scored_events.csv` if you've run the
pipeline, and falls back to the bundled `artifacts/sample/` data otherwise
— it doesn't retrain anything itself, it's a viewer on top of the pipeline
output.

## Natural next steps (good for your "future work" slide)

- Replace the rule-based attacker-persona generator with a handful of LLM
  calls to author varied attack *strategy configs* (velocity, target
  selection, geo pattern) — cheap (tens of calls, not per-row generation)
  and directly answers the "GenAI-generated attack" framing.
- Add the device/IP bipartite graph as an actual `networkx` graph with
  community detection instead of the current rolling fan-out count, for a
  more visual "ring" demo.
- Wire `05_mitigate_and_demo.py`'s action decision into a live step-up-auth
  simulation (e.g. an OTP challenge mock) for the live demo.
