"""
05_mitigate_and_demo.py

BLUE TEAM layer, part 3: mitigation + demo narrative.

Turns the blended risk score into an action (this is the "defend" part of
the brief -- not just a label) and prints:
  1. Action-band performance vs. ground truth.
  2. The shared-device/IP "botnet ring" our fan-out feature would have
     caught (several attacked accounts, one attacker device).
  3. A full timeline walkthrough for a couple of real attack episodes:
     benign activity -> credential-stuffing burst -> breach -> cash-out,
     each event's action decision.
"""

import pandas as pd

import config as cfg

ALLOW_MAX = 30
STEP_UP_MAX = 70


def assign_action(score):
    if score < ALLOW_MAX:
        return "allow"
    if score < STEP_UP_MAX:
        return "step_up_auth"
    return "block"


def band_report(df):
    print("=== Action bands vs. ground truth ===")
    tbl = pd.crosstab(df["action"], df["label_ato"], rownames=["action"], colnames=["is_ato"])
    print(tbl)
    print()
    caught = df[(df["action"] == "block") & (df["label_ato"] == 1)]
    missed = df[(df["action"] == "allow") & (df["label_ato"] == 1)]
    print(f"ATO events blocked outright: {len(caught)} / {df['label_ato'].sum()}")
    print(f"ATO events waved through as 'allow' (missed): {len(missed)}")
    fp_blocked = df[(df["action"] == "block") & (df["label_ato"] == 0)]
    print(f"Benign events wrongly blocked (false positives): {len(fp_blocked)} "
          f"({100*len(fp_blocked)/max((df['action']=='block').sum(),1):.1f}% of all blocks)")


def botnet_ring_demo(df):
    print("\n=== Shared-infrastructure ('botnet ring') check ===")
    top = (
        df[df["event_type"].isin(["failed_login", "ato_login"])]
        .groupby("device_id")[cfg.COL_ACCOUNT]
        .nunique()
        .sort_values(ascending=False)
        .head(5)
    )
    print("Attacker devices ranked by number of distinct victim accounts touched:")
    print(top.to_string())


def episode_walkthrough(df, n_episodes=2):
    print(f"\n=== Sample attack episode walkthrough(s) ===")
    episode_ids = df["attack_episode_id"].dropna().unique()[:n_episodes]
    cols = ["event_time", "event_type", "device_id", "ip_address", "amount",
            "device_new_for_account", "ip_new_for_account",
            "failed_logins_trailing_10min", "ato_risk_score", "action"]
    for eid in episode_ids:
        acc = df[df["attack_episode_id"] == eid][cfg.COL_ACCOUNT].iloc[0]
        window = df[df[cfg.COL_ACCOUNT] == acc].sort_values("event_time")
        print(f"\n--- Account {acc} | episode {eid} ---")
        print(window[cols].to_string(index=False))


def main():
    df = pd.read_csv(cfg.SCORED_EVENTS_CSV, parse_dates=["event_time"])
    df["action"] = df[cfg.RISK_SCORE_COL].apply(assign_action)
    df.to_csv(cfg.SCORED_EVENTS_CSV, index=False)

    band_report(df)
    botnet_ring_demo(df)
    episode_walkthrough(df)


if __name__ == "__main__":
    main()
