"""Shared config for the ATO Defense Lab pipeline."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_CSV = str(BASE_DIR / "data" / "sampled_transactions.csv")

# Real columns we trust from the Kaggle data (content/behavior signal)
COL_TXN_ID = "transaction_id"
COL_TS = "timestamp"
COL_ACCOUNT = "sender_account"
COL_RECEIVER = "receiver_account"
COL_AMOUNT = "amount"
COL_TYPE = "transaction_type"
COL_MERCHANT_CAT = "merchant_category"
COL_CHANNEL = "payment_channel"
COL_IS_FRAUD = "is_fraud"
COL_SPEND_DEV = "spending_deviation_score"
COL_VELOCITY = "velocity_score"
COL_GEO_ANOM = "geo_anomaly_score"

# NOTE: we deliberately IGNORE the dataset's own device_used / ip_address /
# location / device_hash columns for ATO purposes. We verified they carry no
# per-account persistence (every row has an essentially random device/IP/geo
# regardless of account or fraud label), so they can't support a "new device
# for this account" signal. We replace them with our own internally
# consistent synthetic session layer (built in 02_baseline_and_attack.py).

OUT_DIR = str(BASE_DIR / "artifacts")
EVENTS_CSV = f"{OUT_DIR}/events.csv"
FEATURES_CSV = f"{OUT_DIR}/features.csv"
MODEL_PATH = f"{OUT_DIR}/xgb_ato_model.json"
IFOREST_PATH = f"{OUT_DIR}/iforest.joblib"
SCALER_PATH = f"{OUT_DIR}/scaler.joblib"

RANDOM_SEED = 42

RISK_SCORE_COL = "ato_risk_score"
ACTION_COL = "action"
SCORED_EVENTS_CSV = f"{OUT_DIR}/scored_events.csv"
