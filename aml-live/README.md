# aml-live

A live view over the detector. Transactions replay in order, the GNN's score
arrives with each one, and anything at or above the alert threshold goes to the
queue. Two ways to watch it: a streaming ledger, and a network where money
travels along the edges and alerting accounts pull into visible clusters.

```
python aml-live/build_stream.py --data sample_data    # score a run -> web/stream.json
python aml-live/serve.py                              # http://127.0.0.1:8000
```

`build_stream.py` needs `torch` (same as `predict_gnn.py`). `serve.py` needs
nothing beyond the standard library.

---

## Where it goes

Drop the `aml-live/` folder in the repo root, next to `run.py`:

```
aml-sim/
├── run.py
├── gnn_model.py            # imported by build_stream.py
├── aml_models.py           # imported by build_stream.py
├── models/                 # gnn_account_model.pt, gnn_transaction_model.pt
└── aml-live/
    ├── build_stream.py
    ├── serve.py
    ├── README.md
    └── web/
        ├── index.html      # the whole view, no build step, no dependencies
        └── stream.json     # written by build_stream.py
```

The included `stream.json` is the bundled `sample_data` run — 54,270
transactions, 1,500 accounts, 30 days — so the page works before you generate
anything. Rebuild it against your own run:

```
python run.py all
python aml-live/build_stream.py                  # defaults to data/
python aml-live/build_stream.py --max-events 40000
```

`--max-events` keeps the first N transactions chronologically; the default of
60,000 lands around 3 MB. A full 10,000-account run is 1.1M transactions, which
is more than a browser wants to hold in one array and far more than anyone will
watch, so it gets trimmed.

---

## Why the scores are precomputed

The account GNN aggregates over a 3-hop neighbourhood. An account's embedding
therefore depends on edges that arrive later in wall-clock order — scoring
"transaction 12,000 using only the first 12,000 transactions" would be a
different model than the one in `models/`, with different numbers.

So `build_stream.py` scores the finished graph once, exactly as `predict_gnn.py`
does, and the page reveals each score at its own transaction's timestamp. The
metrics printed on load are the real ones for that run. Nothing in the view
reads a score before its transaction has arrived.

On the bundled sample run that is PR-AUC 0.7155, ROC-AUC 0.9890, recall 84.6%
at a 2.9% flag rate — the same figures `predict_gnn.py --data sample_data`
prints.

---

## Reading the screen

**Three tiers, everywhere.** Teal is cleared, amber is the watchlist band just
under the threshold, rose is a queued alert. Nothing else uses those colours.

**The risk meter's midpoint is the alert line.** Scores are violently bimodal —
median 1e-4, threshold 1.2e-3, flagged rows at 0.999 — so a linear bar is either
empty or full and shows nothing in between. The meter plots log(score) pinned so
the threshold sits at exactly half. Past the tick means queued, whatever
threshold your run was calibrated to.

**Clusters are model signal, not labels.** A halo is a connected component over
*alerting* edges — accounts the detector linked, with no reference to
`episode_id`. That is the point of the network view: a chain, a cycle and a
fan-out are different shapes, and at 1 hour/sec you can watch one assemble.

**Ground truth is a toggle.** Press `G` and the generator's labels come in, in
violet, which nothing else uses. True laundering edges underlay the network, so
a violet path with no rose on it is a visible miss. In the alert queue each card
gains its real pattern — and the benign twins (`payroll_fanout`,
`supplier_passthrough`, `installment_split`, `marketplace_fanin`) tag amber
rather than "false positive", since they are hard negatives doing their job.
They are 12.7% of false positives on the sample run.

**Detection by typology** fills in as episodes arrive: caught over seen, per
laundering pattern, live. An average across typologies hides the one you cannot
see at all, so there isn't one.

---

## Controls

| | |
|---|---|
| `Space` | play / pause |
| `1` `2` | ledger / network |
| `N` | skip to the next alert |
| `G` | ground truth overlay |
| `R` | restart |

Speed runs from 5 minutes to 24 hours of simulated time per real second. At
1 hour/sec the sample run takes about 12 minutes and the ledger is readable at
roughly 21 transactions/sec; at 24 hours/sec the whole month goes past in half a
minute. The network's retention window scales with the speed setting, otherwise
the canvas silts up at the fast end.

The filter (`All traffic` / `Watchlist and above` / `Alerts only`) controls what
reaches the ledger. When the feed outruns the DOM, cleared rows collapse into a
`· 312 cleared` summary line — alerts and watchlist rows are never sampled out.
Clicking an account or an alert card focuses that node in the network.

---

## Notes

- The page needs an HTTP origin because it fetches `stream.json`; opening
  `index.html` off the filesystem trips the browser's `file://` rules. If you
  would rather not run the server, the error state accepts a `stream.json`
  dropped onto it.
- Fonts come from Google Fonts. Offline, it falls back to system faces.
- `prefers-reduced-motion` turns off the particles and the halo pulse.
- Verified in Chrome from 390px to 1920px wide.
- If you retrain, rebuild `stream.json` — the alert threshold ships inside the
  model bundle and the page reads it from there, so the meter and the queue
  follow your calibration automatically.
