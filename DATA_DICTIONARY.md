# Data dictionary

All money amounts are INR. All timestamps are UTC.
Columns marked **GT** are ground truth — never feed them to a model.

---

## `transactions.csv` — one row per transaction

| Column | Type | Notes |
|---|---|---|
| `txn_id` | str | `T000000000`, assigned after global sort by timestamp |
| `timestamp` | datetime | second resolution |
| `sender` | str | account id |
| `receiver` | str | account id |
| `amount` | float | rounded to 2dp; a share of amounts are rounded to human-looking figures in both classes |
| `channel` | str | `UPI`, `IMPS`, `NEFT`, `RTGS`, `CARD`, `SWIFT`, `WIRE` — derived from amount |
| `sender_country` / `receiver_country` | str | ISO-2 |
| `cross_border` | 0/1 | derived |
| `episode_id` | str | **GT** — empty for background traffic |
| `pattern` | str | **GT** — `normal`, `salary`, or a typology name |
| `is_laundering` | 0/1 | **GT** |

Rows are sorted by timestamp with laundering and legitimate traffic interleaved, so row
order carries no signal.

---

## `accounts.csv` — one row per account

**Observable** (a bank would have these): `account_id`, `archetype`, `city`, `country`,
`kyc_level` (`full`/`min`), `account_age_days`.

**Simulator state** — internally used to generate behaviour. Excluded by `model_matrix()`;
useful for analysis, fatal as features: `account_idx`, `is_business`, `business_hours`,
`open_ts`, `dormant`, `baseline_out_per_day`, `baseline_amount_median`,
`baseline_amount_sigma`, `night_ratio`, `monthly_income`, `popularity`, `employer_idx`,
`salary_amount`, `salary_day`.

**Ground truth:**

| Column | Notes |
|---|---|
| `is_laundering` | **GT** — participated in ≥1 laundering episode |
| `laundering_role` | **GT** — pipe-joined roles, e.g. `mule\|transit` |
| `laundering_patterns` | **GT** — pipe-joined typologies |
| `n_laundering_episodes` | **GT** — reuse count; mules serving several networks |
| `in_benign_lookalike` | **GT** — took part in a benign twin episode (hard negative) |

Archetypes: `salary`, `student`, `freelancer`, `household`, `small_business`, `merchant`,
`large_business`, `investment`.

---

## `episodes.csv` — one row per coordinated behaviour

`episode_id` (`L…` laundering, `B…` benign), `pattern`, `family`
(`laundering` / `benign_lookalike`), `is_laundering`, `start_time`, `end_time`,
`duration_hours`, `total_amount`, `n_transactions`, `n_accounts`, `difficulty`
(the setting in force when it was generated).

## `episode_members.csv` — one row per (episode, account)

`episode_id`, `account_id`, `role`, `pattern`, `family`, `is_laundering`.

Roles: `source`, `intermediary`, `mule`, `transit`, `collector`, `destination`,
`beneficiary`, `counterparty`.

## `edges.csv` — the aggregated graph

`sender`, `receiver`, `n_txns`, `total_amount`, `mean_amount`, `max_amount`,
`first_time`, `last_time`, `lifespan_days`, `n_laundering_txns` (**GT**),
`is_laundering_edge` (**GT**).

---

## `account_features.csv` — the modelling table

Identity and KYC: `account_id`, `archetype`, `country`, `city`, `kyc_level`,
`account_age_days`.

**Flow** — `n_in`, `n_out`, `n_txns`, `amt_in_*`, `amt_out_*` (total/mean/median/max/std),
`amt_total`, `net_flow`, `inflow_outflow_ratio`, `in_out_txn_ratio`, `amt_cv_out`,
`retention_ratio` (share of inflow kept — near zero for a transit account).

**Counterparty** — `unique_senders`, `unique_receivers`, `degree_ratio`,
`counterparty_hhi` (1.0 = a single counterparty, ~0 = fully dispersed),
`one_shot_counterparty_ratio`, `counterparty_per_txn`.

**Velocity** — `max_txns_1h`, `max_txns_24h`, `max_amount_1h`, `max_amount_24h`,
`mean_inter_txn_seconds`, `min_inter_txn_seconds`,
`burstiness` (∈[-1,1]; 1 = tight bursts, <0 = metronome-regular).

**Temporal** — `night_ratio`, `weekend_ratio`, `active_span_days`, `txns_per_active_day`,
`round_amount_ratio`, `near_threshold_ratio` (70–100% of the reporting threshold),
`above_threshold_ratio`.

**Pass-through** — `median_holding_seconds`, `min_holding_seconds` (−1 = never received
before sending), `outflow_within_1h_ratio`, `outflow_within_24h_ratio`,
`passthrough_ratio` (how closely each outflow mirrors the inflow before it).

**Deviation from own history** — `volume_growth_ratio`, `count_growth_ratio`,
`counterparty_growth_ratio` (second half of the window vs first), `dormancy_wakeup_score`.

**Graph** — `g_in_degree`, `g_out_degree`, `g_degree`, `g_degree_ratio`,
`g_relay_score` (1.0 when in-degree equals out-degree), `g_pagerank`, `g_core_number`,
`g_scc_size`, `g_in_cycle`, `g_reciprocal_edges`, `g_reciprocity`, `g_clustering`,
`g_is_hub`, `g_two_hop_in`, `g_two_hop_out`, `g_fan_out_ratio`.

Plus the five ground-truth columns carried over from `accounts.csv`.

---

## `evaluation/`

| File | Contents |
|---|---|
| `report.json` | precision, recall, F1, PR-AUC, ROC-AUC at the alert budget, and the share of false positives landing on benign lookalikes |
| `evaluation_by_pattern.csv` | account-level recall per typology |
| `evaluation_by_episode.csv` | episode-level recall — an episode counts as caught if any participant is alerted |
| `missed_episodes.csv` | the false negatives, largest first — Red Team input |
| `feature_importance.csv` | permutation importance on the test split |
