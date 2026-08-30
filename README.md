# Aegis — AI Defense Lab for Payment Security

Built for the Mastercard Innovation Challenge @ GFF 2026. An end-to-end Red Team vs
Blue Team simulation for chargeback fraud (Mastercard reason code 4853 — defective
merchandise), covering all three challenge pillars:

- **Identify**: `aegis/attacks/taxonomy.py` — a catalog of social-engineering tactics
  crossed with image-forgery techniques, including a two-angle-photo requirement
  designed to be resistant to naive single-shot diffusion fakes.
- **Generate**: `aegis/redteam/` — an agent that carries on a real turn-by-turn chat
  (via `aegis/chat_prompts.py` and `aegis/support_bot.py`) with an actual customer
  support bot, generates two angle photos of the "damaged" item, then mutates its
  approach (including escalating from naive independent generation to img2img
  conditioning) based on why the Blue Team rejected it.
- **Defend**: `aegis/blueteam/` — a prompt-injection sanitizer, a multi-signal
  forensic vision inspector, deterministic (non-LLM) image metadata forensics, and
  a dual-LLM supervisor that never sees the raw chat transcript, only sanitized
  structured signals.

The UI has five tabs: **Live Simulation** (the interactive Red-vs-Blue chat demo),
**Try It Yourself** (chat and upload your own photos against the real pipeline),
**Batch Evaluation** (precision/recall/F1/AUC), **Attack Taxonomy** (the catalog),
and **Example Gallery** — real, pre-generated Red Team vs Blue Team rounds saved to
`data/fraud_samples.json`, so there's always something to show even if the live
network/API is unavailable at demo time. Regenerate it any time with
`python -m scripts.generate_fraud_samples`.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

That's it — the app runs entirely on a deterministic **mock provider** by default,
with **zero API keys, zero network calls, and zero cost**. Every tab is fully
functional out of the box; the sidebar's "Bring your own key" field (below) is the
only way to switch to live mode.

## The Live Simulation chat is a real conversation

The customer (Red Team) and the support bot don't share one canned transcript —
they actually take turns: the customer opens the dispute, the bot asks for two
photos of the damage from different angles, the customer replies and attaches
them, the bot acknowledges and forwards the claim. Each message is a separate
generation call rendered live in the UI (`aegis/orchestrator.py`'s
`start_interactive`/`continue_interactive`, with an `on_message` callback `app.py`
uses to stream chat bubbles as they're produced). In mock mode these are canned
per-turn lines; with a live key, both sides are genuinely dynamic.

The support bot (`aegis/support_bot.py`) is deliberately **not** part of the
trusted Blue Team boundary — it's the vulnerable surface a real attacker targets.
Even a fully jailbroken bot saying "refund approved!" changes nothing: the
sanitizer and supervisor never read anything the bot says, only their own
classification of the transcript and the vision inspector's findings.

**Try It Yourself** puts you on the other side of that same boundary: you type as
the customer and upload two real photos, and the exact same sanitizer / vision
inspector / supervisor pipeline runs against your input live, with detection
signals updating turn by turn. Vision analysis kicks off automatically the moment
both photos are uploaded (no extra click needed) so the AI's recommendation is
never silently out of date with what's on screen. The human always makes the final
call independently, via Accept / Escalate / Ask for more evidence / Deny.

## Bring your own key (optional)

The app's sidebar has an "Anthropic API key" field. Paste a key there to use live
Claude for that browser session only — it's held in Streamlit's session state, never
written to `.env` or disk, and never shared across sessions. Leave it blank and the
app runs the zero-cost mock provider, no key required at all. This is on top of (not
instead of) the `.env`-based key described below, which is the simpler option for
running the app yourself long-term.

```bash
cp .env.example .env
```

```bash
ANTHROPIC_API_KEY=sk-ant-...   # console.anthropic.com, pay-as-you-go, $5 minimum top-up
IMAGE_PROVIDER=pollinations    # free, no key needed -- or "stability"/"gemini" with their own key
```

One Anthropic key covers both roles: **Claude Haiku 4.5** drives every chat turn
(customer and support bot) since it's the cheapest capable text model and the
interactive chat makes several small calls per round, and **Claude Sonnet 4.5**
handles the Blue Team's vision/forensic inspection. No code changes are needed to
switch — `aegis/providers/factory.py` picks the live provider automatically once a
key is present (from `.env` or the sidebar), and falls back to the mock provider
otherwise.

## Detection signals (Blue Team)

**Sanitizer** (`aegis/blueteam/sanitizer.py`) screens the chat transcript first:
`injection_detected` (a definite instruction-override attempt — short-circuits
straight to REJECT) and, independently, a `manipulation_risk_score` with a
plain-English `reason` for softer manipulative tone (fabricated urgency, invented
authority, policy-citation pressure) that falls short of an outright injection but
should still move the needle.

**Vision inspector** (`aegis/blueteam/vision_inspector.py`) pipelines the two
submitted photos through several *independent* model calls rather than one combined
prompt — deliberately, after finding that a single mega-prompt let conclusions leak
between logically separate judgments (e.g. justifying a low fraud score by citing
good cross-image consistency, which is backwards reasoning for a question that
should be judged from one photo's own pixels):

- **Artifact check** (per photo, run in parallel): does this image on its own show
  signs of AI generation or editing — lighting/shadow physics, sensor-noise realism,
  edge artifacts at the damage boundary.
- **Identity check**: exhaustive, feature-by-feature same-physical-unit comparison
  (color/finish, logo placement, texture, pre-existing wear, screws/seams) — drives
  `detail_consistency_score`, built to catch a sophisticated forgery that nails the
  overall gestalt but slips on one micro-detail.
- **Holistic comparison check**: does the same damage plausibly appear from two
  angles, and does it match the claimed defect — drives `semantic_match` and one
  vote on cross-angle consistency.
- **Localization + geometric check**: a model call asked only to *measure* (name two
  shared landmarks and a damage bounding box per photo, no verdict at all), feeding
  a deterministic Python calculation of landmark-relative position/extent drift
  between the two photos. This exists because extensive testing showed VLMs are
  reliably bad at judging fine-grained cross-angle geometric consistency directly —
  moving the actual judgment into arithmetic works around that instead of trying to
  prompt around it.
  - **Shape-match check**: a focused re-ask on a cropped-and-scaled composite of
  just the two damage regions, isolating the comparison from the rest of the photo.
- `angle_consistency_score` is the **minimum** of the holistic/geometric/shape-match
  scores (any one flagging inconsistency isn't diluted by the other two looking
  clean), and `consistency_disagreement` (how far apart the three landed) feeds a
  technique-agnostic safety-net override: high disagreement between independent
  methods is itself grounds to escalate to a human, even when no single score is
  extreme.

All vision prompts share an explicitly adversarial "investigator" persona rather
than a neutral-critic framing — added after observing the model default to a
charitable reading of ambiguous evidence (explaining a mismatch away instead of
flagging it), which is the wrong prior for a role whose job is catching submissions
specifically designed to survive a charitable read.

**Metadata forensics** (`aegis/blueteam/metadata_forensics.py`) — deterministic,
non-LLM checks on the raw uploaded file bytes: real EXIF camera data, and structurally
-parsed (not naive substring-matched) C2PA content-credential markers and IPTC
`DigitalSourceType` AI-declaration markers. Only ever run in the **Try It Yourself**
tab on real uploads — Red Team/Batch Evaluation images are all synthetically rendered
by this same codebase and would be flagged identically for a meaningless reason.

**Supervisor** (`aegis/blueteam/supervisor.py`) — the dual-LLM defense boundary. It
only ever sees the sanitizer's flags, the vision inspector's structured output, and
order metadata, never the raw chat transcript, so a jailbreak of the customer-facing
bot can't reach the logic holding refund authority. A weighted `fraud_confidence`
score drives a three-band decision (APPROVE / ESCALATE-to-human / REJECT, not a
binary cutoff) so ambiguous cases route to review instead of auto-denial, plus a
generalized `STRONG_SIGNAL_OVERRIDES` list that floors the decision at ESCALATE
whenever any one signal is independently suspicious enough on its own — regardless
of what the blended average says.

## Project layout

```
app.py                  Streamlit UI: 5 tabs (see above)
aegis/
  config.py             env loading, round cap, decision thresholds, scoring weights
  schemas.py             pydantic models shared across the pipeline
  state.py                thin st.session_state wrapper (UI-only)
  providers/              TextProvider / ImageProvider / VisionProvider + mock/Anthropic/Pollinations/Stability/Gemini
  attacks/taxonomy.py     tactic x technique catalog
  attacks/products.py     concrete product/damage catalog grounding image-gen prompts
  chat_prompts.py         turn-by-turn prompt builders shared by customer + support bot
  support_bot.py          the (untrusted, jailbreakable) customer-facing chat bot
  redteam/                customer chat turns + two-angle image generation, mutation on rejection
  blueteam/               sanitizer, vision inspector, metadata forensics, supervisor decision logic
  manual_session.py       pure logic for the Try It Yourself tab (session-history reduction)
  orchestrator.py         round loop -- both one-shot (start/continue_round) and
                          interactive turn-by-turn (start_interactive/continue_interactive)
  metrics.py              batch evaluation: confusion matrix, precision/recall/F1, ROC/AUC
data/legit_samples.json  canned genuine-claim fixtures for false-positive testing
data/fraud_samples.json  Example Gallery fixtures (see scripts/ below)
scripts/generate_legit_samples.py   regenerates the legit fixture file above
scripts/generate_fraud_samples.py   regenerates the fraud gallery fixtures
scripts/build_sony_example.py       builds a gallery entry from a real reference photo pair
scripts/build_live_demo_example.py  adds a live, 2-round adaptive-attacker gallery entry
tests/                   35 tests covering the full pipeline, no network required
```

## Running tests

```bash
pytest tests/ -v
```

## Notes on the detection design

The Blue Team's `supervisor.py` deliberately never sees the raw chat transcript —
only the sanitizer's injection flags and the vision inspector's structured JSON.
That separation means a prompt injection that fully jailbreaks the customer-facing
bot still can't reach the logic holding refund authority. The three-band decision
(APPROVE / ESCALATE / REJECT, not a binary) keeps ambiguous cases routed to human
review instead of auto-denied, which is what keeps the false-positive rate on
genuine claims low.
