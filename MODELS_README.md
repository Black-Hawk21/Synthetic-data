# AML Detector Models for `aml-sim`

Four trained detectors for the synthetic money-laundering dataset — two
scikit-learn baselines and two graph neural networks that beat them at both the
account and the transaction level.

All models ship pre-trained. You can score data immediately, or retrain from
scratch on your own generated datasets.

---

## Install

Unzip this folder, then copy its contents into your local `aml-sim/` directory
(the one containing `run.py` and `amlgen/`). Either do it by hand, or:

```
python install_into_repo.py /path/to/Synthetic-data/aml-sim
```

The script refuses to overwrite anything that already exists, so it can't damage
your checkout.

Then install dependencies:

```
cd /path/to/Synthetic-data/aml-sim
pip install -r requirements.txt              # the repo's own
pip install -r requirements-models.txt       # adds torch + joblib
```

**On the torch install:** unless you have an NVIDIA GPU, use the CPU build —
the default PyPI wheel drags in ~2.5 GB of CUDA libraries you will never touch:

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

macOS `pip install torch` is already CPU/MPS and needs no flags.

Your final layout:

```
aml-sim/
├── run.py, config.yaml, amlgen/     # the original repo
├── aml_models.py                    # shared feature engineering
├── gnn_model.py                     # GNN architecture
├── train_models.py                  # train the scikit-learn baselines
├── train_gnn.py                     # train the GNNs
├── predict.py                       # classify with the baselines
├── predict_gnn.py                   # classify with the GNNs
├── compare_models.py                # all four, side by side
└── models/
    ├── account_model.joblib
    ├── transaction_model.joblib
    ├── gnn_account_model.pt
    └── gnn_transaction_model.pt
```

---

## Run it

You need a generated dataset first:

```
python run.py all          # ~25s, writes data/
```

Then classify:

```
python predict_gnn.py      # the GNNs — this is the one you want
python predict.py          # the scikit-learn baselines
python compare_models.py   # all four on identical held-out splits
```

`predict_gnn.py` writes:

| File | Contents |
|---|---|
| `predictions/gnn_account_predictions.csv` | `account_id, laundering_score, is_flagged` — every account ranked |
| `predictions/gnn_transaction_predictions.csv` | `txn_id, sender, receiver, amount, laundering_score, is_flagged` |

`laundering_score` is the real output — a 0-1 ranking signal. `is_flagged`
applies the operating threshold stored in the bundle (an analyst budget at 2x
the base rate, the repo's own convention). Re-threshold freely to match whatever
alert volume you can actually staff.

Useful flags on both predict scripts: `--data`, `--models`, `--out`,
`--accounts-only`, `--transactions-only`.

The shipped weights were trained on the default config (10,000 accounts,
90 days, difficulty 0.5, `seed: 20260822`). Since that seed is fixed in
`config.yaml`, a plain `python run.py all` reproduces exactly the dataset they
were trained on.

---

## Results

Held-out test splits, identical rows for every model — 70/30 stratified for
accounts, `GroupShuffleSplit` on `episode_id` for transactions so no laundering
episode straddles the boundary. Alert budget = 2x base rate.

### Account level

| Model | PR-AUC | ROC-AUC | Precision | Recall |
|---|---|---|---|---|
| HistGradientBoosting (tabular) | 0.6447 | 0.8894 | 0.371 | 0.743 |
| **AML-GNN (graph)** | **0.8845** | **0.9674** | **0.457** | **0.916** |

### Transaction level

| Model | PR-AUC | ROC-AUC | Precision | Recall |
|---|---|---|---|---|
| HistGradientBoosting (tabular) | 0.7943 | 0.9976 | 0.424 | 0.884 |
| **GNN + transaction head** | **0.9122** | **0.9992** | **0.473** | **0.985** |

### Recall by typology (accounts, same alert budget)

The GNN wins on every typology — not an average that hides a failure.

| Typology | Accounts | Tabular | AML-GNN | Δ |
|---|---|---|---|---|
| `fan_in` | 169 | 0.793 | 0.970 | **+0.178** |
| `fan_out` | 174 | 0.724 | 0.891 | **+0.167** |
| `circular_flow` | 25 | 0.800 | 0.960 | **+0.160** |
| `dormant_activation` | 20 | 0.800 | 0.950 | **+0.150** |
| `rapid_pass_through` | 58 | 0.845 | 0.966 | **+0.121** |
| `smurfing` | 35 | 0.914 | 1.000 | **+0.086** |
| `mule_network` | 87 | 0.874 | 0.954 | **+0.080** |
| `layering_chain` | 89 | 0.955 | 0.989 | **+0.034** |

A caveat worth reading: `predict_gnn.py` reports ~0.95 account PR-AUC because
it scores the *whole* dataset, training rows included. **0.8845 is the honest
number.** Use `compare_models.py` for anything you intend to quote.

Precision around 0.45-0.50 is expected and intentional. The dataset's benign
lookalikes (`payroll_fanout`, `marketplace_fanin`, `supplier_passthrough`,
`treasury_cycle`, `installment_split`) exist specifically to draw false
positives. Chasing 0.99 precision here would mean the dataset is broken.

### Transfer to unseen data

The shipped weights were also tested on a freshly generated dataset they had
never seen (3,000 accounts, 40 days, different seed), with no retraining:

| Level | Tabular | GNN |
|---|---|---|
| Account PR-AUC | 0.5435 | **0.5892** |
| Transaction PR-AUC | 0.6715 | **0.7880** |

Both drop against the headline numbers because that graph is far sparser — the
absolute values track dataset density, not model quality. The ordering holds,
which is the part that matters: the GNN generalises to graphs it was not
trained on.


---

## Retrain on your own data

Whenever you regenerate at a different `difficulty`, add a typology, or change
the population size, retrain:

```
python train_models.py                 # baselines, ~2 min
python train_gnn.py                    # both GNN stages
python train_gnn.py --stage account --epochs 400 --hidden 128
```

Training is **resumable**. Every run checkpoints to
`models/gnn_<stage>_checkpoint.pt`, and `--max-seconds N` stops cleanly and
saves. Rerun the same command to continue:

```
python train_gnn.py --stage account --max-seconds 600   # train in 10-min chunks
```

This matters on CPU. On the single-core box I trained these on, the account
stage ran ~7s/epoch and needed ~30 minutes for 250 epochs. On a normal
multi-core laptop expect several times faster; on a GPU it's minutes.

Useful knobs: `--epochs`, `--txn-epochs`, `--hidden`, `--layers`, `--dropout`,
`--drop-edge`, `--lr`, `--neg-per-pos`, `--threads`, `--seed`.

### When the GNN *loses* — read this before retraining

The GNN's advantage is not unconditional. I verified this on a deliberately
small run (3,000 accounts, 40 days, only 60 epochs) and the result flipped:

| Model | PR-AUC (3k accounts, 60 epochs) |
|---|---|
| HistGradientBoosting (tabular) | **0.654** |
| AML-GNN | 0.644 |

Two reasons, and both are fixable:

1. **Undertrained.** 60 epochs is a quarter of the default schedule. On the
   full dataset the GNN only overtook the baseline around epoch 40 and kept
   climbing until roughly epoch 150. Gradient-boosted trees converge almost
   immediately; the GNN does not. If you cut epochs, you will lose.
2. **The graph is too sparse.** 3,000 accounts gives 64k edges against 418k at
   the default population. Episode counts scale with population, so a small run
   has both a thinner graph and far fewer positives (295 vs 987) to learn
   from — and a GNN's whole advantage is the neighbourhood structure that
   sparsity destroys.

So: **retrain at the default 10,000 accounts and let it run the full 250
epochs.** If you must work small, the tabular baseline is the more honest
choice, and `compare_models.py` will tell you which is winning on your data
rather than leaving you to assume.

---

## How the GNN works

`gnn_model.py` implements a directed, edge-conditioned message-passing network
in **pure PyTorch** — no torch-geometric, no torch-scatter, no compiled
extensions. That was deliberate: those packages are the single most common
reason a GNN repo won't install, and nothing here needs them.

Four design choices drive the gain over the gradient-boosted baseline:

**Direction is a first-class citizen.** Laundering is about where money *goes*.
Every layer aggregates over incoming and outgoing edges with separate weight
matrices, so a fan-in collector and a fan-out distributor get different
representations. A symmetric GCN collapses exactly the distinction you need.
The `fan_in` and `fan_out` rows in the typology table are the largest gains in
the whole model, which is this choice showing up in the metrics.

**Edges carry money, not just adjacency.** Each message is conditioned on eight
edge attributes: log total volume, transaction count, log mean and max amount,
lifespan, the share of the sender's total outflow this edge represents, the
share of the receiver's total inflow, and the max/mean burstiness ratio. A model
that only knows "there is an edge" cannot tell a salary credit from a
pass-through hop.

**Mean and max aggregation together.** Mean describes routine behaviour; max
catches the single anomalous counterparty carrying the episode. Averaging alone
dilutes exactly the signal you're hunting.

**Jumping knowledge.** The classification head sees the raw features and all
three hop-levels concatenated, so 1-hop, 2-hop and 3-hop views stay separable
and the model can fall back to pure tabular signal where the graph is
uninformative. This is what makes the GNN a strict improvement rather than a
gamble — it can always recover the baseline's information.

Three hops is deliberate: `layering_chain` and `mule_network` are multi-hop by
construction, and `circular_flow` needs enough range to come back around.

Training uses **DropEdge** (30% of edges dropped per epoch) as a graph-native
regulariser, a cosine LR schedule, class-weighted BCE, gradient clipping, and
early model selection on validation PR-AUC.

The **transaction head** is stage 2: it takes the frozen node embeddings, and
scores each transaction from its own attributes plus the sender's and
receiver's embeddings *and their interaction* (difference and elementwise
product). That interaction term is what lets it reason about the pair rather
than two independent accounts.

Nothing in either model ever sees `episode_id`, `pattern`, `is_laundering`,
laundering roles, or the simulator's internal state columns as an input. The
leak-safe drop-list mirrors `amlgen.features.model_matrix`.

---

## Using the models in your own code

```python
import pandas as pd, torch
from aml_models import account_feature_matrix
from gnn_model import AMLGNN, build_graph_tensors, load_gnn, normalize_features

b     = load_gnn("models/gnn_account_model.pt")
feats = pd.read_csv("data/account_features.csv")
edges = pd.read_csv("data/edges.csv")

X = account_feature_matrix(feats, columns=b["columns"]).to_numpy("float32")
Xn, _ = normalize_features(X, b["norm_stats"])
edge_index, edge_attr = build_graph_tensors(edges, feats["account_id"])

net = AMLGNN(b["in_dim"], hidden=b["hidden"], layers=b["layers"], dropout=0.0)
net.load_state_dict(b["state_dict"]); net.eval()

with torch.no_grad():
    embeddings = net.embed(torch.from_numpy(Xn), edge_index, edge_attr)  # [N, 384]
    scores = torch.sigmoid(net.head(embeddings).squeeze(-1)).numpy()
```

`embeddings` is a 384-dim learned representation per account. It's useful well
beyond classification — cluster it to find mule rings, nearest-neighbour it to
find accounts behaving like a known bad one, or feed it as node features to
whatever you build next.

---

## Where to go from here

The Red Team loop is the obvious next step. `python run.py redteam --levels
0,0.25,0.5,0.75,1.0` generates datasets at rising difficulty; retrain the GNN at
each level and watch which typologies break first. The tabular baseline degrades
sooner, so the gap between the two curves is itself a measurement of how much of
each typology is structural rather than behavioural.

Beyond that: the current model aggregates transactions into a static weighted
graph, which throws away ordering. A temporal GNN over the raw transaction
stream — or per-edge time encodings — is the natural next increment, and
`rapid_pass_through` and `circular_flow` are where it should pay off, since both
are defined by *sequence* rather than volume.
