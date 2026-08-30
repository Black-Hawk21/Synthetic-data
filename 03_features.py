"""
03_features.py

BLUE TEAM layer, part 1: feature engineering.

For every event (chronological, per account) we compute features using only
information available *up to and including* that event -- no lookahead, so
this is safe to use as-is for real-time scoring later.

Feature groups:
  - Identity novelty: has this device / IP-prefix / country ever been seen
    for this account before now?
  - Login velocity: failed logins in the trailing 10-minute window; time
    since last event.
  - Behavioral deviation: transaction amount vs. this account's running
    mean/std (only normal_txn history counts toward the baseline); flag for
    account's first-ever high-risk cash-out channel.
  - Cross-account fan-out: how many *distinct accounts* has this
    device_id / ip_address touched in the trailing 24h (bot/shared-infra
    signal) -- this is the graph-lite feature.
  - Passthrough of the dataset's own precomputed scores where present.
"""

import numpy as np
import pandas as pd

import config as cfg

TRAILING_LOGIN_WINDOW = pd.Timedelta(minutes=10)
TRAILING_FANOUT_WINDOW = pd.Timedelta(hours=24)


def compute_account_features(events: pd.DataFrame) -> pd.DataFrame:
    events = events.sort_values([cfg.COL_ACCOUNT, "event_time"]).reset_index(drop=True)

    seen_devices, seen_ips, seen_countries = {}, {}, {}
    amount_hist = {}  # account -> list of normal_txn amounts seen so far
    fail_times = {}   # account -> list of failed_login timestamps seen so far
    last_event_time = {}

    device_new, ip_new, country_new = [], [], []
    failed_login_trailing = []
    seconds_since_last = []
    amount_z = []
    is_first_event = []

    for row in events.itertuples(index=False):
        acc = getattr(row, cfg.COL_ACCOUNT)
        t = row.event_time
        dev = row.device_id
        ip_prefix = ".".join(str(row.ip_address).split(".")[:2])
        country = row.country

        devs = seen_devices.setdefault(acc, set())
        ips = seen_ips.setdefault(acc, set())
        countries = seen_countries.setdefault(acc, set())

        device_new.append(int(dev not in devs))
        ip_new.append(int(ip_prefix not in ips))
        country_new.append(int(country not in countries))
        is_first_event.append(int(len(devs) == 0))

        # trailing failed-login velocity (before adding current event)
        flist = fail_times.setdefault(acc, [])
        flist[:] = [ft for ft in flist if t - ft <= TRAILING_LOGIN_WINDOW]
        failed_login_trailing.append(len(flist))
        if row.event_type == "failed_login":
            flist.append(t)

        # time since last event for this account
        prev_t = last_event_time.get(acc)
        seconds_since_last.append((t - prev_t).total_seconds() if prev_t is not None else -1.0)
        last_event_time[acc] = t

        # amount z-score vs running normal-txn history (leak-safe: history is prior-only)
        hist = amount_hist.setdefault(acc, [])
        if len(hist) >= 3:
            mu, sigma = np.mean(hist), np.std(hist) + 1e-6
            amt = row.amount if not pd.isna(row.amount) else np.nan
            amount_z.append((amt - mu) / sigma if not np.isnan(amt) else np.nan)
        else:
            amount_z.append(np.nan)

        if row.event_type == "normal_txn" and not pd.isna(row.amount):
            hist.append(row.amount)

        # update seen sets AFTER computing novelty for this event
        devs.add(dev)
        ips.add(ip_prefix)
        countries.add(country)

    events["device_new_for_account"] = device_new
    events["ip_new_for_account"] = ip_new
    events["country_new_for_account"] = country_new
    events["is_first_event_for_account"] = is_first_event
    events["failed_logins_trailing_10min"] = failed_login_trailing
    events["seconds_since_last_event"] = seconds_since_last
    events["amount_zscore_vs_history"] = amount_z
    return events


def compute_fanout_features(events: pd.DataFrame) -> pd.DataFrame:
    """For each event, how many distinct accounts has this device/IP touched
    in the trailing 24h? High fan-out = shared/bot infrastructure."""
    events = events.sort_values("event_time").reset_index(drop=True)

    device_hist = {}  # device_id -> list of (time, account)
    ip_hist = {}       # ip_prefix -> list of (time, account)

    device_fanout, ip_fanout = [], []

    for row in events.itertuples(index=False):
        t = row.event_time
        acc = getattr(row, cfg.COL_ACCOUNT)
        dev = row.device_id
        ip_prefix = ".".join(str(row.ip_address).split(".")[:2])

        dlist = device_hist.setdefault(dev, [])
        dlist[:] = [(tt, aa) for tt, aa in dlist if t - tt <= TRAILING_FANOUT_WINDOW]
        device_fanout.append(len({aa for _, aa in dlist} | {acc}))
        dlist.append((t, acc))

        ilist = ip_hist.setdefault(ip_prefix, [])
        ilist[:] = [(tt, aa) for tt, aa in ilist if t - tt <= TRAILING_FANOUT_WINDOW]
        ip_fanout.append(len({aa for _, aa in ilist} | {acc}))
        ilist.append((t, acc))

    events["device_distinct_accounts_24h"] = device_fanout
    events["ip_distinct_accounts_24h"] = ip_fanout
    return events


def main():
    print("Loading events...")
    events = pd.read_csv(cfg.EVENTS_CSV, parse_dates=["event_time"])

    print("Computing per-account novelty/velocity/deviation features...")
    events = compute_account_features(events)

    print("Computing cross-account device/IP fan-out features...")
    events = compute_fanout_features(events)

    # a couple of light derived/cleanup fields for modeling convenience
    events["is_login_event"] = events["event_type"].isin(
        ["failed_login", "ato_login", "normal_login"]
    ).astype(int)
    events["is_txn_event"] = events["event_type"].isin(
        ["normal_txn", "ato_txn"]
    ).astype(int)
    events["hour_of_day"] = events["event_time"].dt.hour

    events.to_csv(cfg.FEATURES_CSV, index=False)
    print(f"Saved -> {cfg.FEATURES_CSV}  ({len(events)} rows)")
    print(events["label_ato"].value_counts())


if __name__ == "__main__":
    main()
