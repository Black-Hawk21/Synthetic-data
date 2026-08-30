"""
Downloads a public, real-world SMS spam/ham dataset and normalizes it, SPLIT
into two portions:
  - a HOLDOUT portion (data the detector never trains on -- used purely to
    check it generalizes beyond our own synthetic generator's writing style)
  - a TRAIN portion (real-world examples added into the training pool
    alongside the synthetic data, to genuinely improve real-world recall)

Splitting rather than sending 100% of this data to holdout matters: if you
train on the exact same data you use to measure generalization, you no
longer have any honest way to tell whether the detector generalizes to
real-world text it hasn't seen -- the holdout stops meaning anything. The
split is stratified by label (fraud/legit) and uses a fixed random seed, so
rerunning this script doesn't quietly reshuffle which examples land where.

Source: the classic UCI SMS Spam Collection (Almeida & Hidalgo), mirrored
on GitHub. Verified reachable as of Aug 2026 -- if the mirror ever moves,
swap SMS_URL below for another mirror (search "SMS Spam Collection csv
github raw").

NOTE on email data: most phishing-email datasets worth using (Kaggle's
"Phishing Email Dataset", Nazario corpus, etc.) require a Kaggle login and
can't be fetched by a script without credentials. Rather than silently
skip this, prepare_email_holdout() below gives you the exact manual steps
-- download the CSV yourself, drop it in data/raw/, then this script will
detect and normalize it into the same schema automatically.

Usage:
    python prepare_holdout.py --train-fraction 0.7
"""

import argparse
import csv
import io
import json
import os
import random
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from schema import Sample

# Real phishing email corpora (Nazario especially) sometimes pack an entire
# raw email -- headers, HTML, occasionally base64 attachment text -- into a
# single CSV field, which blows past Python's default 128KB-per-field limit
# with "field larger than field limit (131072)". Raise it. On Windows, the C
# long backing this is 32-bit even in 64-bit Python, so sys.maxsize itself
# can raise OverflowError -- back off until a value is accepted.
_maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(_maxInt)
        break
    except OverflowError:
        _maxInt = int(_maxInt / 10)

SMS_URL = "https://raw.githubusercontent.com/mohitgupta-1O1/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_SMS = os.path.join(os.path.dirname(__file__), "..", "data", "generated", "holdout_sms.jsonl")
OUT_EMAIL = os.path.join(os.path.dirname(__file__), "..", "data", "generated", "holdout_email.jsonl")
# NOT named "holdout_*" on purpose -- dataset_utils.py only excludes files matching that
# pattern from the training pool, so this one gets picked up as ordinary training data
# automatically, no changes needed elsewhere.
OUT_TRAIN_PORTION = os.path.join(
    os.path.dirname(__file__), "..", "data", "generated", "real_world_train.jsonl")

# Common column-name variants across the phishing-email CSVs people tend to
# download from Kaggle -- prepare_email_holdout() tries these in order.
EMAIL_TEXT_COLUMNS = ["text", "email_text", "body", "Email Text", "Message"]
EMAIL_LABEL_COLUMNS = ["label", "Email Type", "Label", "type", "spam", "Spam"]
PHISHING_LABEL_VALUES = {"phishing", "phishing email", "1", "spam", "fraud"}


def _stratified_split(samples: list, train_fraction: float, seed: int = 42):
    """Splits samples into (train_portion, holdout_portion), stratified by
    label so both portions keep roughly the same fraud/legit ratio as the
    original data. Fixed seed -> reproducible across reruns."""
    rng = random.Random(seed)
    fraud = [s for s in samples if s.label == 1]
    legit = [s for s in samples if s.label == 0]
    rng.shuffle(fraud)
    rng.shuffle(legit)

    def _split_one(group):
        cut = int(len(group) * train_fraction)
        return group[:cut], group[cut:]

    fraud_train, fraud_holdout = _split_one(fraud)
    legit_train, legit_holdout = _split_one(legit)
    return fraud_train + legit_train, fraud_holdout + legit_holdout


def _write_jsonl(samples: list, path: str, mode: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode) as f:
        for s in samples:
            f.write(json.dumps(s.to_dict()) + "\n")


def prepare_sms_holdout(train_fraction: float):
    print(f"Downloading SMS data from {SMS_URL} ...")
    try:
        with urllib.request.urlopen(SMS_URL, timeout=20) as resp:
            raw = resp.read().decode("latin-1")  # this dataset has non-UTF8 bytes
    except Exception as e:  # noqa: BLE001
        print(f"  [error] could not download SMS data: {e}", file=sys.stderr)
        print("  Skipping -- you can manually place a v1,v2 CSV at "
              f"{os.path.join(RAW_DIR, 'sms_spam.csv')} and rerun.")
        return 0, 0

    os.makedirs(RAW_DIR, exist_ok=True)
    raw_path = os.path.join(RAW_DIR, "sms_spam.csv")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(raw)

    reader = csv.reader(io.StringIO(raw))
    header = next(reader)
    samples = []
    for row in reader:
        if len(row) < 2:
            continue
        label_raw, text = row[0].strip().lower(), row[1].strip()
        if not text:
            continue
        label = 1 if label_raw == "spam" else 0
        s = Sample(
            text=text,
            label=label,
            channel="sms",
            attack_subtype="legit" if label == 0 else "public_spam",
            difficulty_tier="naive",
            source_topic="public_holdout",
            generation_model="public_dataset:sms_spam_collection",
        )
        s.validate()
        samples.append(s)

    train_portion, holdout_portion = _stratified_split(samples, train_fraction)
    _write_jsonl(holdout_portion, OUT_SMS, "w")
    _write_jsonl(train_portion, OUT_TRAIN_PORTION, "a")

    print(f"  SMS: {len(train_portion)} added to training pool, "
          f"{len(holdout_portion)} held out "
          f"({sum(1 for s in holdout_portion if s.label==1)} spam / "
          f"{sum(1 for s in holdout_portion if s.label==0)} ham in holdout)")
    return len(train_portion), len(holdout_portion)


def _extract_text(row: dict, fieldnames: list) -> str:
    """Some phishing-email datasets (e.g. the 'Phish No More' Kaggle compilation
    of Enron/Ling/CEAS/Nazario/Nigerian-Fraud/SpamAssassin) split subject and
    body into separate columns instead of one text column. Subject lines often
    carry real phishing signal ("URGENT: Verify Your Account") so combine them
    rather than silently dropping the subject."""
    if "subject" in fieldnames and "body" in fieldnames:
        subject = (row.get("subject") or "").strip()
        body = (row.get("body") or "").strip()
        return f"Subject: {subject}\n{body}".strip() if subject else body
    text_col = next((c for c in EMAIL_TEXT_COLUMNS if c in fieldnames), None)
    return (row.get(text_col) or "").strip() if text_col else ""


def prepare_email_holdout(train_fraction: float):
    """Looks for a manually-downloaded phishing email CSV in data/raw/ and
    normalizes it if found. Otherwise prints instructions and skips."""
    candidates = [f for f in os.listdir(RAW_DIR)] if os.path.isdir(RAW_DIR) else []
    csv_candidates = [f for f in candidates if f.lower().endswith(".csv") and f != "sms_spam.csv"]

    if not csv_candidates:
        print("\nNo email dataset found in data/raw/.")
        print("To add one (optional but recommended for the report):")
        print("  1. Download a phishing email CSV, e.g. Kaggle's 'Phishing Email Dataset'")
        print("     (https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)")
        print(f"  2. Place the .csv file in {RAW_DIR}/")
        print("  3. Re-run this script -- it will auto-detect text/label columns (and combine "
              "subject+body if the dataset splits them) and normalize it.")
        return 0, 0

    total_train, total_holdout = 0, 0
    for fname in csv_candidates:
        path = os.path.join(RAW_DIR, fname)
        print(f"Found {fname} -- attempting to normalize...")
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                has_text_col = any(c in fieldnames for c in EMAIL_TEXT_COLUMNS)
                has_subject_body = "subject" in fieldnames and "body" in fieldnames
                label_col = next((c for c in EMAIL_LABEL_COLUMNS if c in fieldnames), None)
                if not (has_text_col or has_subject_body) or not label_col:
                    print(f"  [skip] couldn't find recognizable text/label columns in {fname} "
                          f"(saw columns: {fieldnames}). Rename columns to one of "
                          f"{EMAIL_TEXT_COLUMNS} (or provide 'subject'+'body') / "
                          f"{EMAIL_LABEL_COLUMNS} and rerun.")
                    continue

                samples = []
                for row in reader:
                    text = _extract_text(row, fieldnames)
                    label_raw = (row.get(label_col) or "").strip().lower()
                    if not text:
                        continue
                    label = 1 if label_raw in PHISHING_LABEL_VALUES else 0
                    s = Sample(
                        text=text,
                        label=label,
                        channel="email",
                        attack_subtype="legit" if label == 0 else "public_spam",
                        difficulty_tier="naive",
                        source_topic="public_holdout",
                        generation_model=f"public_dataset:{fname}",
                    )
                    s.validate()
                    samples.append(s)

            train_portion, holdout_portion = _stratified_split(samples, train_fraction)
            holdout_mode = "a" if total_holdout > 0 else "w"
            _write_jsonl(holdout_portion, OUT_EMAIL, holdout_mode)
            _write_jsonl(train_portion, OUT_TRAIN_PORTION, "a")
            print(f"  {fname}: {len(train_portion)} added to training pool, "
                  f"{len(holdout_portion)} held out")
            total_train += len(train_portion)
            total_holdout += len(holdout_portion)
        except Exception as e:  # noqa: BLE001
            print(f"  [error] failed to parse {fname}: {e}", file=sys.stderr)

    return total_train, total_holdout


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-fraction", type=float, default=0.7,
                         help="fraction of each real-world source added to the training pool "
                              "(the rest stays held out for generalization checks). Default "
                              "0.7 -- keep this well under 1.0, sending 100%% of real data to "
                              "training leaves nothing to measure real-world generalization "
                              "against.")
    args = parser.parse_args()
    if not 0 <= args.train_fraction < 1:
        print("--train-fraction must be in [0, 1) -- some data must stay held out.", file=sys.stderr)
        sys.exit(1)

    # fresh file each run -- both source functions below append to it
    os.makedirs(os.path.dirname(OUT_TRAIN_PORTION), exist_ok=True)
    open(OUT_TRAIN_PORTION, "w").close()

    sms_train, sms_holdout = prepare_sms_holdout(args.train_fraction)
    email_train, email_holdout = prepare_email_holdout(args.train_fraction)

    total_train = sms_train + email_train
    total_holdout = sms_holdout + email_holdout
    print(f"\nTotal: {total_train} real-world samples added to training pool "
          f"({OUT_TRAIN_PORTION}), {total_holdout} held out for generalization checks.")
    print(f"(split: {args.train_fraction*100:.0f}% train / {(1-args.train_fraction)*100:.0f}% holdout)")
