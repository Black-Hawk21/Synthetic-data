# Synthetic Money-Laundering Dataset Generator

Generates a synthetic financial ecosystem, injects money-laundering typologies into it,
turns the result into a transaction **graph**, and flattens that graph into **modelling
tables** you can train a detector on.

The design goal is *not* a big CSV. It is a dataset where laundering is genuinely hard to
separate from legitimate behaviour, so that a model has to learn money-movement behaviour
instead of a threshold on `amount`.

```
population of accounts  ->  legitimate traffic  ->  injected episodes
                                                          |
                          transaction graph  <-------------
                                   |
                    account features + labels  ->  detector  ->  per-typology recall
                                                                        |
                                                    Red Team reads the misses and
                                                    raises `difficulty` -----------+
```

---

## Quick start

```bash
pip install -r requirements.txt

python run.py all                      # full pipeline, writes to data/
python run.py all --accounts 4000 --days 45     # smaller/faster run
```

Roughly 25 seconds and ~1.1M transactions at the default 10,000 accounts / 90 days.

Individual stages:

```bash
python run.py generate --accounts 20000 --days 120 --difficulty 0.8
python run.py features                 # behavioural + graph features
python run.py graph --episode L000012  # draw one episode's money flow
python run.py train                    # baseline detector + per-typology recall
python run.py redteam --levels 0,0.25,0.5,0.75,1.0
```

---

## What lands in `data/`

| File | Grain | What it is |
|---|---|---|
| `transactions.csv` | one transaction | the raw ledger — sender, receiver, amount, timestamp, channel, countries |
| `accounts.csv` | one account | archetype, KYC, age, plus account-level ground truth |
| `episodes.csv` | one episode | a coordinated behaviour: type, window, total amount, participant count |
| `episode_members.csv` | account × episode | which account played which role in which episode |
| `edges.csv` | account pair | the aggregated transaction graph as a weighted edge list |
| `transaction_graph.graphml` | — | the same graph for Gephi / networkx / PyG |
| `account_features.csv` | one account | ~70 engineered features + labels — **this is the modelling table** |
| `evaluation/` | — | baseline scores, per-typology recall, missed episodes, feature importance |
| `figures/` | — | episode plots, amount-overlap check, recall chart |
| `manifest.json` | — | the exact config that produced this dataset |

Full column-by-column reference: [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md).

---

## The design decisions that matter

**Episodes, not flagged rows.** The unit of ground truth is a *laundering episode* — a set
of transactions belonging to one coordinated behaviour, with a start, an end, participants
and roles. Labelling transaction #18372 in isolation throws away the structure that makes
laundering detectable, and makes evaluation meaningless: catching 1 of 40 transactions in
a chain is not the same as catching the chain.

**Eight laundering typologies, not one `fraud=1` class.**

| Pattern | Shape | What it stresses |
|---|---|---|
| `layering_chain` | A → B → C → D | multi-hop path tracing |
| `circular_flow` | A → B → C → A | cycle / SCC detection |
| `rapid_pass_through` | in, then straight out | holding time |
| `smurfing` | many transfers under a threshold | aggregate vs per-transaction view |
| `fan_out` | one → many | out-degree bursts |
| `fan_in` | many → one | in-degree bursts |
| `mule_network` | two mule layers | multi-hop network structure |
| `dormant_activation` | idle account wakes up | behavioural deviation over time |

**Every typology has a legitimate twin.** A dataset with only suspicious fan-outs teaches a
model that fan-out means crime. So the generator also injects `payroll_fanout`,
`marketplace_fanin`, `supplier_passthrough`, `treasury_cycle` and `installment_split` —
structurally similar, labelled clean. They are the hard negatives, and in the default run
they account for **~60–80% of the baseline model's false positives**, which is exactly what
they are for.

**Amounts overlap on purpose.** Episode totals are anchored on the population-wide amount
distribution and capped by what the source account plausibly moves, not drawn from some
"big number" range. In the default run the laundering median transaction is ₹38k against a
legitimate median of ₹5.5k — elevated, but sitting well inside the legitimate spread.
`figures/amount_overlap.png` is the check; if those two histograms ever separate cleanly,
the dataset is broken.

**Laundering is run by a bounded network.** Mules, shells and sources are drawn from
restricted pools (`laundering.network.*`) rather than the whole population. This keeps
suspicious accounts rare and reproduces a real signal: mules get reused across episodes.

**One difficulty knob.** `laundering.difficulty` ∈ [0, 1] moves every pattern between its
blatant and its subtle form at once — hop counts, holding times, amount splitting,
threshold-hugging, time-of-day alignment, number of participants.

```
difficulty 0.0                          difficulty 1.0
5-hop chain, 30 seconds apart           2-hop chain, spread over days
smurfs hug 98% of the threshold         amounts scattered at 5-30% of it
fan-out to 110 accounts, equal splits   fan-out to 14, Dirichlet splits
transfers at 3am                        transfers inside the account's own routine
```

---

## The Red Team / Blue Team loop

This is the part worth building a hackathon around. Generate, train, find the misses, make
those typologies harder, regenerate.

```bash
python run.py redteam --levels 0,0.25,0.5,0.75,1.0
```

Each level gets its own dataset in `data/redteam/difficulty_<x>/`, and the sweep summary
lands in `data/redteam/redteam_sweep.csv`. What you should see is PR-AUC falling as
difficulty rises — and, more interestingly, *different typologies* failing at different
points. `evaluation/missed_episodes.csv` lists the false negatives, largest first: those
are the Red Team's next targets.

`python run.py train` prints recall per typology at a realistic alert budget (by default,
an analyst queue of twice the base rate), because an average across typologies hides one
you cannot see at all.

---

## Layout

```
aml-sim/
├── run.py                        # CLI: all | generate | features | graph | train | redteam
├── config.yaml                   # every knob
├── amlgen/
│   ├── config.py                 # defaults + YAML merge + validation
│   ├── entities.py               # the 8 account archetypes and their parameter ranges
│   ├── distributions.py          # amount/time/channel sampling primitives
│   ├── population.py             # accounts, employers, counterparty affinity graph
│   ├── normal_activity.py        # salaries + organic traffic (the baseline)
│   ├── ledger.py                 # transaction and episode stores
│   ├── simulate.py               # end-to-end driver
│   ├── graphs.py                 # edge aggregation, graph build, GraphML, subgraphs
│   ├── export.py                 # table writers + run manifest
│   ├── viz.py                    # episode plots and dataset sanity plots
│   ├── patterns/
│   │   ├── base.py               # SimContext: pools, timing, difficulty, commit
│   │   ├── layering.py           # layering_chain, circular_flow, rapid_pass_through
│   │   ├── structuring.py        # smurfing, fan_out, fan_in
│   │   ├── mules.py              # mule_network, dormant_activation
│   │   ├── lookalikes.py         # the five benign twins
│   │   └── registry.py           # registry + injection driver
│   ├── features/
│   │   ├── account_features.py   # velocity, flow, counterparty, temporal, pass-through
│   │   ├── graph_features.py     # degree, PageRank, SCC, k-core, clustering, 2-hop
│   │   └── build.py              # modelling table + leak-safe X/y split
│   ├── models/baseline.py        # HistGradientBoosting baseline + permutation importance
│   └── evaluation/metrics.py     # per-typology recall, episode recall, missed episodes
└── examples/redteam_loop.py      # the loop as a script you can edit
```

---

## Adding your own typology

Subclass `Pattern`, use the context helpers, register it. That is the whole contract.

```python
# amlgen/patterns/mine.py
from .base import Pattern

class TradeInvoicing(Pattern):
    name = "trade_invoicing"

    def run(self, ctx, episode_id):
        exporter = ctx.shells(1)[0]
        importer = ctx.sources(1, exclude=[exporter])[0]
        total = ctx.episode_total(importer, scale=2.0)
        ts = ctx.window(30 * 86400)
        txns = [(importer, exporter, ctx.plausible_hour(ts, importer), total)]
        ctx.commit(episode_id, self.name, "laundering", 1, txns,
                   {"source": [importer], "beneficiary": [exporter]})
```

Then add it to `LAUNDERING` in `patterns/registry.py` and to `config.yaml`. Useful helpers
on `ctx`: `mules(k)`, `shells(k)`, `sources(k)` (bounded network pools), `episode_total()`,
`hop_delay()`, `retention()`, `plausible_hour()`, `window()`, and `lerp(easy, hard)` to make
your pattern respond to the difficulty knob.

If your new typology has a plausible legitimate twin, add that too — in `lookalikes.py`,
with `is_laundering = 0`. It is worth more than another laundering pattern.

---

## Modelling notes

`account_features.csv` carries the labels alongside the features. Use the helper rather
than dropping columns by hand:

```python
import pandas as pd
from amlgen.features import model_matrix

features = pd.read_csv("data/account_features.csv")
X, y = model_matrix(features)      # drops labels and simulator-internal columns
```

`model_matrix` removes both the label columns and the simulator's own state
(`baseline_out_per_day`, `monthly_income`, `dormant`, …). Those exist for analysis and
would leak badly — a real bank does not know an account's ground-truth activity parameters.
In the raw tables, `episode_id` and `pattern` on `transactions.csv` are ground truth too.

The baseline is deliberately a plain gradient-boosted tree on account-level features. Two
obvious directions from there: sequence models over each account's transaction stream, and
GNNs over `transaction_graph.graphml` — the graph is already there, and the typologies that
the tabular baseline handles worst (`circular_flow`, `rapid_pass_through`) are exactly the
ones with the most structure to exploit.

---

## Limitations, stated plainly

Parameters are *plausible*, not calibrated against real bank data — internally coherent
distributions, INR-denominated, with no attempt to match any real institution's profile.
There is no cash, no cheques, no FX conversion, no account opening or closing mid-window,
and no seasonality beyond a weekday effect. Laundering is injected on top of the baseline
rather than displacing it, so participants carry slightly more volume than they otherwise
would. Treat results here as a measure of *relative* detector quality across typologies and
difficulty levels, not as an estimate of real-world performance.

---

## What's in `sample_data/`

A small pre-generated run (1,500 accounts, 30 days, difficulty 0.5) so you can look at the
schema and the figures before running anything. It is a *scaled-down* dataset — episode
counts scale with population, so the class balance matches the default run, but the graph
is far sparser. Regenerate at full size with `python run.py all`.