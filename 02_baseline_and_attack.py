"""
02_baseline_and_attack.py

RED TEAM layer.

The Kaggle data gives us real amounts/timestamps/merchant categories per
account, but no persistent device/IP/location per account (verified: those
columns are effectively random per row). So we:

  1. Assign every account a synthetic "home" identity: 1-2 home devices,
     a home country, and a home IP prefix. This is OUR ground truth for
     "what normal looks like" for that account.
  2. Treat every real transaction as happening inside a session from the
     account's home identity (with small natural variation, e.g. a second
     home device or occasional travel) -> event_type = normal_txn.
  3. For a random subset of accounts, inject an ATO attack episode at a
     random point in their timeline:
       - a burst of failed_login events from a shared "attacker device/IP"
         pool (botnet reuse across victims -> graph signal for detection)
       - one successful ato_login from that attacker identity
       - a short burst of ato_txn cash-out transactions, amount elevated
         vs. the account's own historical baseline, skewed toward
         transfer/withdrawal and faster payment rails.

Output: artifacts/events.csv - one row per event (login or transaction),
chronologically orderable per account, with a ground-truth `label_ato` and
`attack_episode_id` for supervised training / evaluation.
"""

import numpy as np
import pandas as pd

import config as cfg

rng = np.random.default_rng(cfg.RANDOM_SEED)

COUNTRIES = ["US", "UK", "DE", "SG", "AE", "JP", "AU", "CA"]
ATTACK_FRACTION = 0.05          # share of accounts that suffer an ATO episode
N_ATTACKER_IDENTITIES = 60      # shared pool -> botnet/device-reuse signal


def load_real_transactions():
    df = pd.read_csv(cfg.RAW_CSV, parse_dates=[cfg.COL_TS])
    df = df.sort_values([cfg.COL_ACCOUNT, cfg.COL_TS]).reset_index(drop=True)
    return df


def assign_home_identities(accounts):
    n = len(accounts)
    home_device = np.array([f"DEV-{i:06d}" for i in rng.choice(2_000_000, size=n, replace=False)])
    # ~15% of users regularly use a second device (phone + laptop)
    has_second_device = rng.random(n) < 0.15
    second_device = np.array([f"DEV-{i:06d}" for i in rng.choice(2_000_000, size=n, replace=False)])
    home_country = rng.choice(COUNTRIES, size=n)
    home_ip_prefix = np.array([f"{rng.integers(1,223)}.{rng.integers(0,255)}" for _ in range(n)])
    return pd.DataFrame({
        cfg.COL_ACCOUNT: accounts,
        "home_device": home_device,
        "second_device": np.where(has_second_device, second_device, None),
        "home_country": home_country,
        "home_ip_prefix": home_ip_prefix,
    })


def make_attacker_pool():
    """A small reused pool of attacker devices/IPs/geos -> shared infra across victims."""
    devices = [f"BOT-{i:05d}" for i in range(N_ATTACKER_IDENTITIES)]
    ip_prefixes = [f"{rng.integers(1,223)}.{rng.integers(0,255)}" for _ in range(N_ATTACKER_IDENTITIES)]
    countries = rng.choice(COUNTRIES, size=N_ATTACKER_IDENTITIES).tolist()
    return pd.DataFrame({"device": devices, "ip_prefix": ip_prefixes, "country": countries})


def normal_events_for_account(acc_txns, identity):
    """Turn an account's real transactions into paired normal_login + normal_txn
    events (with an occasional benign failed_login, e.g. a mistyped password)
    from the account's own home identity. Keeping benign examples of every
    event_type is what stops event_type alone from leaking the label."""
    n = len(acc_txns)
    has_second = identity["second_device"] is not None and not (
        isinstance(identity["second_device"], float) and np.isnan(identity["second_device"])
    )
    use_second = (rng.random(n) < 0.3) if has_second else np.zeros(n, dtype=bool)
    device = [
        identity["second_device"] if use_second[j] else identity["home_device"]
        for j in range(n)
    ]
    # occasional benign travel: different country, still same device
    traveling = rng.random(n) < 0.04
    travel_country = rng.choice(COUNTRIES, size=n)
    country = [travel_country[j] if traveling[j] else identity["home_country"] for j in range(n)]
    ip = [
        identity["home_ip_prefix"] + f".{rng.integers(0,255)}.{rng.integers(0,255)}"
        for _ in range(n)
    ]
    ts = pd.to_datetime(acc_txns[cfg.COL_TS].values)

    rows = []
    for j in range(n):
        # benign mistyped-password moment ~3% of sessions, same home device
        if rng.random() < 0.03:
            rows.append({
                "event_time": ts[j] - pd.Timedelta(seconds=int(rng.integers(20, 90))),
                cfg.COL_ACCOUNT: identity[cfg.COL_ACCOUNT],
                "event_type": "failed_login",
                "device_id": device[j], "ip_address": ip[j], "country": country[j],
                "amount": np.nan, "transaction_type": None, "merchant_category": None,
                "payment_channel": None, "spending_deviation_score": np.nan,
                "velocity_score": np.nan, "geo_anomaly_score": np.nan,
                "source_is_fraud": False, "label_ato": 0, "attack_episode_id": None,
            })
        # the successful login that precedes the transaction
        rows.append({
            "event_time": ts[j] - pd.Timedelta(seconds=int(rng.integers(5, 20))),
            cfg.COL_ACCOUNT: identity[cfg.COL_ACCOUNT],
            "event_type": "normal_login",
            "device_id": device[j], "ip_address": ip[j], "country": country[j],
            "amount": np.nan, "transaction_type": None, "merchant_category": None,
            "payment_channel": None, "spending_deviation_score": np.nan,
            "velocity_score": np.nan, "geo_anomaly_score": np.nan,
            "source_is_fraud": False, "label_ato": 0, "attack_episode_id": None,
        })
        # the real transaction itself
        rows.append({
            "event_time": ts[j],
            cfg.COL_ACCOUNT: identity[cfg.COL_ACCOUNT],
            "event_type": "normal_txn",
            "device_id": device[j], "ip_address": ip[j], "country": country[j],
            "amount": acc_txns[cfg.COL_AMOUNT].values[j],
            "transaction_type": acc_txns[cfg.COL_TYPE].values[j],
            "merchant_category": acc_txns[cfg.COL_MERCHANT_CAT].values[j],
            "payment_channel": acc_txns[cfg.COL_CHANNEL].values[j],
            "spending_deviation_score": acc_txns[cfg.COL_SPEND_DEV].values[j],
            "velocity_score": acc_txns[cfg.COL_VELOCITY].values[j],
            "geo_anomaly_score": acc_txns[cfg.COL_GEO_ANOM].values[j],
            "source_is_fraud": acc_txns[cfg.COL_IS_FRAUD].values[j],
            "label_ato": 0, "attack_episode_id": None,
        })
    return pd.DataFrame(rows)


def inject_ato_episode(acc_txns, identity, attacker, episode_id, baseline_amount):
    """Build the failed_login -> ato_login -> ato_txn burst for one attacked account."""
    # pick an attack time somewhere within the account's observed activity window
    t_min, t_max = acc_txns[cfg.COL_TS].min(), acc_txns[cfg.COL_TS].max()
    span = (t_max - t_min).total_seconds()
    attack_start = t_min + pd.Timedelta(seconds=rng.uniform(0, max(span, 3600)))

    events = []

    # 1) credential-stuffing burst: several failed logins in a tight window
    n_fail = int(rng.integers(5, 16))
    fail_offsets = np.sort(rng.uniform(0, 240, size=n_fail))  # within 4 minutes
    for off in fail_offsets:
        events.append({
            "event_time": attack_start + pd.Timedelta(seconds=off),
            cfg.COL_ACCOUNT: identity[cfg.COL_ACCOUNT],
            "event_type": "failed_login",
            "device_id": attacker["device"],
            "ip_address": attacker["ip_prefix"] + f".{rng.integers(0,255)}.{rng.integers(0,255)}",
            "country": attacker["country"],
            "amount": np.nan,
            "transaction_type": None,
            "merchant_category": None,
            "payment_channel": None,
            "spending_deviation_score": np.nan,
            "velocity_score": np.nan,
            "geo_anomaly_score": np.nan,
            "source_is_fraud": False,
            "label_ato": 1,
            "attack_episode_id": episode_id,
        })

    # 2) breach: one successful login from the attacker identity
    breach_time = attack_start + pd.Timedelta(seconds=fail_offsets[-1] + rng.uniform(5, 30))
    events.append({
        "event_time": breach_time,
        cfg.COL_ACCOUNT: identity[cfg.COL_ACCOUNT],
        "event_type": "ato_login",
        "device_id": attacker["device"],
        "ip_address": attacker["ip_prefix"] + f".{rng.integers(0,255)}.{rng.integers(0,255)}",
        "country": attacker["country"],
        "amount": np.nan,
        "transaction_type": None,
        "merchant_category": None,
        "payment_channel": None,
        "spending_deviation_score": np.nan,
        "velocity_score": np.nan,
        "geo_anomaly_score": np.nan,
        "source_is_fraud": False,
        "label_ato": 1,
        "attack_episode_id": episode_id,
    })

    # 3) cash-out burst: 1-4 elevated transactions shortly after breach
    n_cashout = int(rng.integers(1, 5))
    cashout_channels = ["wire_transfer", "UPI"]
    cashout_types = ["transfer", "withdrawal"]
    for i in range(n_cashout):
        t = breach_time + pd.Timedelta(seconds=rng.uniform(30, 600) * (i + 1))
        amt = baseline_amount * rng.uniform(3.5, 12.0) + rng.uniform(50, 500)
        # Elevated-but-overlapping synthetic risk scores rather than a sentinel:
        # sampled to skew high, on the SAME scale as the dataset's own scores,
        # so the model has to learn a real (noisy) distributional shift rather
        # than an artificial "missing value" fingerprint.
        events.append({
            "event_time": t,
            cfg.COL_ACCOUNT: identity[cfg.COL_ACCOUNT],
            "event_type": "ato_txn",
            "device_id": attacker["device"],
            "ip_address": attacker["ip_prefix"] + f".{rng.integers(0,255)}.{rng.integers(0,255)}",
            "country": attacker["country"],
            "amount": round(float(amt), 2),
            "transaction_type": rng.choice(cashout_types),
            "merchant_category": "other",
            "payment_channel": rng.choice(cashout_channels),
            "spending_deviation_score": round(float(rng.normal(2.6, 1.0)), 3),
            "velocity_score": int(np.clip(rng.normal(17, 4), 0, 20)),
            "geo_anomaly_score": round(float(np.clip(rng.normal(0.85, 0.12), 0, 1)), 3),
            "source_is_fraud": False,
            "label_ato": 1,
            "attack_episode_id": episode_id,
        })

    return pd.DataFrame(events)


def main():
    print("Loading real transactions...")
    df = load_real_transactions()
    accounts = df[cfg.COL_ACCOUNT].unique()
    print(f"{len(df)} transactions across {len(accounts)} accounts")

    identities = assign_home_identities(accounts).set_index(cfg.COL_ACCOUNT)
    attacker_pool = make_attacker_pool()

    attacked_accounts = set(
        rng.choice(accounts, size=int(len(accounts) * ATTACK_FRACTION), replace=False)
    )
    print(f"Injecting ATO episodes into {len(attacked_accounts)} accounts")

    all_events = []
    episode_id = 0
    grouped = df.groupby(cfg.COL_ACCOUNT, sort=False)

    for i, (acc, acc_txns) in enumerate(grouped):
        identity = identities.loc[acc].to_dict()
        identity[cfg.COL_ACCOUNT] = acc
        normal = normal_events_for_account(acc_txns, identity)
        all_events.append(normal)

        if acc in attacked_accounts:
            attacker = attacker_pool.iloc[rng.integers(0, N_ATTACKER_IDENTITIES)].to_dict()
            baseline_amount = float(acc_txns[cfg.COL_AMOUNT].mean())
            episode_id += 1
            attack_ev = inject_ato_episode(acc_txns, identity, attacker, episode_id, baseline_amount)
            all_events.append(attack_ev)

        if i % 4000 == 0:
            print(f"  processed {i}/{len(accounts)} accounts")

    events = pd.concat(all_events, ignore_index=True)
    events = events.sort_values([cfg.COL_ACCOUNT, "event_time"]).reset_index(drop=True)

    print(f"Total events: {len(events)}  |  ATO-labeled events: {events['label_ato'].sum()}")
    events.to_csv(cfg.EVENTS_CSV, index=False)
    print(f"Saved -> {cfg.EVENTS_CSV}")


if __name__ == "__main__":
    main()
