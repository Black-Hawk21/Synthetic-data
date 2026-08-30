# AI Defense Lab — Social Engineering & Phishing at Scale

Mastercard Innovation Challenge 2026 — Red team (attack generator) + Blue team (detector)
for GenAI-powered social engineering / phishing targeting payments.

## Structure

```
project/
├── app.py                    # Streamlit demo dashboard (live detector + charts)
├── data/
│   ├── raw/                 # public datasets (SMS spam collection, manually-added phishing email CSVs)
│   └── generated/           # synthetic output + normalized holdout data (jsonl)
├── generator/
│   ├── schema.py            # canonical sample schema
│   ├── personas.py          # Faker-based synthetic victim persona generator
│   ├── templates.py         # attack subtype prompt templates
│   ├── llm_client.py        # Groq API wrapper (rate limiting, refusal detection/retry)
│   ├── generate_static.py   # batch generates email/SMS phishing + legit samples
│   ├── generate_conversational.py  # multi-turn vishing/chat attacker vs victim-sim
│   └── prepare_holdout.py   # downloads/normalizes public real-world data as a holdout set
├── detector/
│   ├── dataset_utils.py     # loads + merges data/generated/*.jsonl (train pool vs holdout)
│   ├── train_baseline.py    # TF-IDF + Logistic Regression baseline detector
│   └── artifacts/           # saved model + vectorizer (created on first run)
└── eval/                    # baseline_metrics.json (created on first run)
```

## Setup

Uses Groq's free, OpenAI-compatible API (open-weight models -- `openai/gpt-oss-120b` by default --
running on Groq's LPU hardware, no credit card needed).

```bash
pip install requests faker --break-system-packages
```

1. Sign in at https://console.groq.com
2. Go to "API Keys" -> create a new key
3. `export GROQ_API_KEY=gsk_...`

Free tier is roughly 25-30 requests/minute and ~1,000/day (check
console.groq.com/settings/limits for current numbers) -- `generator/llm_client.py`
client-side rate-limits itself to stay under that, so batch generation will just
run a bit slowly rather than erroring out. To swap the model (e.g. to a smaller/faster
one if you're burning through daily quota), change `MODEL` at the top of `llm_client.py`
to any model id from https://console.groq.com/docs/models.

## Run generation

```bash
cd generator
python generate_static.py --n-per-cell 2         # sanity check first: 5 subtypes x 2 channels x 3 difficulties x 2 (fraud+legit) = 120 API calls
python generate_static.py --n-per-cell 10        # scale up once output looks good: ~600 calls
python generate_conversational.py --n-dialogues 20 --max-turns 5  # up to ~200 calls (20 dialogues x ~5 turns x 2 speakers)
```

`--n-per-cell` multiplies fast: it's `n * subtypes(5) * channels(2) * difficulties(3) * 2` (fraud+legit)
total API calls. Always start at `--n-per-cell 2` to check output quality/prompt correctness
before scaling up and burning API credits.

Output lands in `data/generated/phishing_synthetic.jsonl` and
`data/generated/conversational_synthetic.jsonl`, both following the schema in `schema.py`.

**Quality control, important:** `gpt-oss-120b` sometimes refuses "write a phishing message"
requests outright, even with the red-team/defensive framing in the system prompt. `llm_client.py`
detects these refusals and retries automatically (with a reinforced prompt) rather than saving
the refusal text as if it were a real labeled sample -- if you see a run finish with the exact
error text `model refused: ...` in the error log, that's this working as intended, not a bug.
`generate_static.py` also drops exact-duplicate outputs and reports near-duplicate pairs at the
end of a run, since a small persona pool can otherwise produce repetitive text.

## Public real-world data (train + holdout split)

```bash
python prepare_holdout.py --train-fraction 0.7
```

Downloads the classic SMS Spam Collection dataset (real-world spam/ham SMS), and processes any
phishing-email CSV you've placed in `data/raw/` (e.g. Kaggle's "Phishing Email Dataset" or
Nazario/Enron) -- then **splits each source 70/30** (stratified by label, fixed seed so it's
reproducible): 70% goes into `data/generated/real_world_train.jsonl` and joins the training pool
alongside your synthetic data, 30% stays in `holdout_sms.jsonl`/`holdout_email.jsonl` as a
genalization check the detector never trains on.

**Why split instead of using 100% of it as holdout (which is what earlier versions of this
script did):** if you train on the exact same data you use to measure real-world generalization,
the holdout stops meaning anything -- you lose any honest way to tell whether the detector
generalizes to real text it hasn't seen. Splitting lets you get the real benefit of real training
data (better real-world recall) without destroying the benchmark that proves it's working.
`--train-fraction` defaults to 0.7; keep it well under 1.0 so a meaningful holdout remains.

## Detector (baseline)

```bash
pip install scikit-learn pandas joblib numpy --break-system-packages
cd detector
python train_baseline.py
```

Trains a TF-IDF + Logistic Regression classifier on an 80/20 split of everything in
`data/generated/` (excluding holdout files), then separately evaluates it against any holdout
data present. Writes:

- `detector/artifacts/baseline_model.joblib`, `baseline_vectorizer.joblib` -- the trained model
- `eval/baseline_metrics.json` -- precision/recall/F1/AUC overall, broken down by
  `attack_subtype` and `difficulty_tier`, plus false-positive rate on legit messages and the
  top TF-IDF terms driving fraud vs. legit predictions (useful for the "explainability" part
  of the report)

**Read the holdout numbers carefully, not just the test-split numbers.** A model can look
perfect on our own generated test split (same writing style it trained on) while badly
generalizing to real-world text -- e.g. in an early run, a baseline trained on ~15 fraud
examples scored 1.0 F1 on its own test split but only caught 0.1% of real-world spam in the
SMS holdout. That gap is itself a legitimate finding for the report (and the fix is more/more
diverse training data, not a better classifier) -- don't just report the test-split numbers
without the holdout comparison alongside them.

## Detector (stronger model — RoBERTa)

```bash
cd detector
python train_transformer.py --epochs 5 --batch-size 16
```

Fine-tunes `roberta-base` (or pass `--model-name distilbert-base-uncased` for a lighter/faster
alternative). **Needs a GPU** -- see `KAGGLE_SETUP.md` for running this on Kaggle's free GPU
notebooks, which is the easiest path if you don't have local GPU access. Not installed by the
main `requirements.txt` since `torch`/`transformers` are heavy and only needed for this script.

This isn't meant to replace the baseline -- it's meant to catch what the baseline can't. TF-IDF
+ Logistic Regression is already near-perfect on obvious ("naive"/"moderate" difficulty)
phishing text; RoBERTa's value case is specifically the "adaptive" difficulty tier (messages
written to avoid obvious trigger words). Compare `eval/roberta_metrics.json` against
`eval/baseline_metrics.json` -- specifically each one's `breakdown_by_difficulty` -> `adaptive`
recall -- for the actual finding to put in the report, not just an overall accuracy number.

**Small-data warning:** fine-tuning a 125M-parameter model on a few hundred examples can overfit
and underperform the baseline. If that happens, it's a real result worth reporting (motivates
"why we needed more synthetic data"), not a failed experiment to hide.

## Adversarial loop (Phase 5 — the "arms race" demo)

```bash
cd detector
python adversarial_loop.py --generations 3 --n-per-generation 4
```

The core demo for judges: trains a detector, extracts exactly which words it's keying on (its
`top_fraud_indicator_terms`), has the generator write NEW attacks specifically instructed to
avoid those words while keeping the same fraudulent intent, measures what fraction evade the
CURRENT detector (the **evasion rate** — your attack potency metric), retrains on those
evasions, and repeats. `--n-per-generation` is per fraud subtype (5 subtypes), so `4` means 20
fraud + 20 matched legit samples generated per round — keep this modest, each generation is a
full round of live LLM calls.

Uses the TF-IDF+LR baseline throughout, not RoBERTa — retraining every generation needs to be
fast, and baseline retrains in seconds. This is a separate, complementary story to the
baseline-vs-RoBERTa comparison, not a replacement for it.

Writes `eval/adversarial_loop_metrics.json` with, per generation: the evasion rate against the
detector it faced, a regression check (does the new detector still catch EARLIER generations'
evasions, or did it forget them while learning the new pattern?), and holdout performance after
each retrain. The evasion-rate-by-generation numbers are the headline chart for your demo —
expect it to start non-trivial and trend down as the detector adapts each round; if it's already
near-zero on generation 1, your existing training data or holdout results already discussed some of these patterns and the loop won't show much movement — try a higher
`--generations` count or check `top_terms_faced` per generation to see if the vocabulary is
actually shifting round to round.

## Demo dashboard (Streamlit)

```bash
pip install streamlit plotly --break-system-packages
streamlit run app.py
```

Run from the project root (not from inside `generator/` or `detector/`). Opens a browser dashboard
with five tabs:

- **Attack Simulator** — the live "build the attack, then build the defense" demo. Generates a
  NEW fraud message via the LLM (pick the attack pattern, channel, and optionally **evasion
  mode**, which targets the current detector's own known weak spots — same mechanism as the
  Phase 5 loop), then immediately runs it through the trained detector and shows CAUGHT or
  EVADED. Falls back to pulling a real example from your generated dataset if no `GROQ_API_KEY`
  is set or the API call fails, so the demo never breaks even without live network access —
  important for judged presentations where you can't rely on connectivity.
- **Live Detector** — two-panel attacker/recipient simulation: compose a message on the
  **Attacker** side, send it, and it arrives in the **Recipient's Inbox** on the other side
  stamped with the detector's live verdict, confidence, and an expandable breakdown of which
  words drove the decision. Runs entirely against the saved baseline model locally — no API
  calls, so it's fast and safe to demo live for judges without depending on network/Groq
  availability.
- **Dataset Overview** — live stats on everything in `data/generated/` (counts by subtype,
  difficulty, channel) plus a sample of actual generated messages.
- **Model Comparison** — baseline vs RoBERTa metrics side by side, pulled from `eval/*.json`.
  Gracefully shows a "not yet generated" message for whichever file doesn't exist yet, rather
  than crashing — so it's safe to run before every phase is finished.
- **Adversarial Arms Race** — the evasion-rate-by-generation chart from Phase 5, plus the holdout
  precision/recall trade-off across retraining generations.

Requires at minimum `detector/artifacts/baseline_model.joblib` (from `train_baseline.py`) for the
Attack Simulator and Live Detector tabs, and `data/generated/*.jsonl` for the Dataset Overview
tab and the Attack Simulator's fallback path — the other two tabs work with whatever `eval/*.json`
files happen to exist, including none. `GROQ_API_KEY` is only needed for LIVE generation in the
Attack Simulator tab; everything else runs fully offline.

## Notes for teammates

- Every sample (yours and theirs) should eventually conform to the same top-level
  fields (`id`, `text`, `label`, `channel`, `source_topic`) so we can merge datasets
  across sub-topics before the shared detector/eval stage. See `schema.py`.
- Keep `attack_subtype` / `difficulty_tier` populated — that's what lets the final
  report break down detector performance instead of reporting one flat accuracy number.
- All personas and message content are synthetic (no real PII, no real people/brands
  used adversarially) — this is red-teaming for defense purposes only.
