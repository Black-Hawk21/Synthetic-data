"""
Phase 5: the adversarial "arms race" loop.

The core demo: train a detector, extract exactly which words/phrases it's
keying on, have the generator write NEW attacks specifically instructed to
avoid those words while keeping the same fraudulent intent, measure how many
evade the CURRENT detector (the "evasion rate" -- this is your attack potency
metric), retrain on those evasions, and repeat. Each generation's evasion
rate against the detector THAT GENERATION FACED is the headline number for
your report/demo chart.

This uses the TF-IDF + Logistic Regression baseline throughout (not RoBERTa)
-- retraining needs to happen every generation, and baseline retrains in
seconds, making an iterative loop actually feasible in a hackathon timeframe.
RoBERTa is a separate, one-shot "stronger model" story (see train_transformer.py),
not part of this loop.

Requires GROQ_API_KEY (same as the other generator scripts) since it calls
the LLM to generate each round's evasion attempts.

Usage:
    cd detector
    python adversarial_loop.py --generations 3 --n-per-generation 20
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "generator"))

from dataset_utils import load_all, summarize
from schema import Sample
from personas import generate_personas
from templates import build_evasion_prompt, SUBTYPE_TEMPLATES, build_legit_prompts, LEGIT_TEMPLATES
from llm_client import generate_text, MODEL

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
GENERATED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "generated")
EVAL_DIR = os.path.join(os.path.dirname(__file__), "..", "eval")
LOOP_METRICS_PATH = os.path.join(EVAL_DIR, "adversarial_loop_metrics.json")

FRAUD_SUBTYPES = [s for s in SUBTYPE_TEMPLATES.keys()]
CHANNELS = ["email", "sms"]


def fit_model(train_df: pd.DataFrame):
    """Same fit logic as train_baseline.py -- kept identical so a 'generation 0'
    detector here is directly comparable to your existing baseline_metrics.json."""
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    X_train = vectorizer.fit_transform(train_df["text"].fillna(""))
    y_train = train_df["label"].values
    model = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    model.fit(X_train, y_train)
    return model, vectorizer


def get_top_fraud_terms(model, vectorizer, k=15) -> list:
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]
    top_idx = np.argsort(coefs)[-k:][::-1]
    return feature_names[top_idx].tolist()


def evaluate_on(model, vectorizer, df: pd.DataFrame) -> dict:
    """Quick precision/recall/f1 for a batch -- used both for the 'evasion rate'
    check (recall on a fresh adversarial batch) and any other ad-hoc eval here."""
    if df.empty:
        return {"n": 0}
    X = vectorizer.transform(df["text"].fillna(""))
    y_true = df["label"].values
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "n": int(len(df)),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def generate_evasion_batch(avoid_terms: list, n_per_subtype: int, generation: int,
                            workers: int = 3) -> list:
    """Generates fraud samples instructed to avoid the given trigger terms,
    plus a matched number of ordinary legit samples to keep the batch balanced.
    Returns a list of Sample objects (not yet written to disk).

    Runs concurrently (like generate_static.py) rather than one call at a time
    -- at ~8-10 calls/minute (Groq's real TPM cap for this model), a fully
    sequential batch of ~40 calls takes 4-5+ minutes with zero visibility in
    between, which looks identical to a hang even when it's working fine."""
    personas = generate_personas(max(20, n_per_subtype * len(FRAUD_SUBTYPES)), seed=100 + generation)
    persona_i = 0

    def _gen_fraud(subtype, channel, persona):
        system, user = build_evasion_prompt(subtype, channel, persona, avoid_terms)
        text = generate_text(system, user)
        s = Sample(text=text, label=1, channel=channel, attack_subtype=subtype,
                   difficulty_tier="adversarial", generation_model=MODEL)
        s.validate()
        return s

    def _gen_legit(kind, channel, persona):
        system, user = build_legit_prompts(kind, channel, persona)
        text = generate_text(system, user)
        s = Sample(text=text, label=0, channel=channel, attack_subtype="legit",
                   difficulty_tier="naive", generation_model=MODEL)
        s.validate()
        return s

    jobs = []
    for subtype in FRAUD_SUBTYPES:
        for i in range(n_per_subtype):
            persona = personas[persona_i % len(personas)]
            persona_i += 1
            channel = CHANNELS[i % len(CHANNELS)]
            jobs.append(("fraud", subtype, channel, persona))

    legit_kinds = list(LEGIT_TEMPLATES.keys())
    n_legit = len(jobs)  # match fraud count for balance
    for i in range(n_legit):
        persona = personas[i % len(personas)]
        kind = legit_kinds[i % len(legit_kinds)]
        channel = CHANNELS[i % len(CHANNELS)]
        jobs.append(("legit", kind, channel, persona))

    print(f"  Generating {len(jobs)} samples ({len(jobs)-n_legit} evasion fraud + {n_legit} legit) "
          f"with {workers} parallel workers (avoiding: "
          f"{', '.join(avoid_terms[:8])}{'...' if len(avoid_terms) > 8 else ''})")

    samples = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for job in jobs:
            kind, arg2, channel, persona = job
            if kind == "fraud":
                fut = executor.submit(_gen_fraud, arg2, channel, persona)
            else:
                fut = executor.submit(_gen_legit, arg2, channel, persona)
            futures[fut] = job

        for i, fut in enumerate(as_completed(futures), 1):
            job = futures[fut]
            try:
                samples.append(fut.result())
            except Exception as e:  # noqa: BLE001
                print(f"    [error] {job[0]} sample ({job[1]}/{job[2]}) failed: {e}", file=sys.stderr)
            if i % 5 == 0 or i == len(jobs):
                print(f"    ...{i}/{len(jobs)} done")

    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=3,
                         help="number of attack/retrain rounds")
    parser.add_argument("--n-per-generation", type=int, default=4,
                         help="evasion samples generated PER FRAUD SUBTYPE per generation "
                              "(5 subtypes, so total fraud samples per generation = 5x this, "
                              "plus a matched number of legit samples)")
    args = parser.parse_args()

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs(GENERATED_DIR, exist_ok=True)

    print("Loading current training pool (everything in data/generated/ except holdout files)...")
    train_pool_df, holdout_df = load_all()
    summarize(train_pool_df, "starting train pool")
    if train_pool_df.empty or train_pool_df["label"].nunique() < 2:
        print("\nNo usable data found -- run generate_static.py first.", file=sys.stderr)
        sys.exit(1)

    print("\nTraining generation-0 detector (your current baseline, for reference)...")
    model, vectorizer = fit_model(train_pool_df)
    gen0_holdout_eval = evaluate_on(model, vectorizer, holdout_df) if not holdout_df.empty else {}
    print(f"  gen-0 holdout: {gen0_holdout_eval}")

    loop_log = {"generations": [], "holdout_available": not holdout_df.empty}
    all_evasion_batches = []  # keep every generation's batch to re-check for "forgetting" later

    for gen in range(1, args.generations + 1):
        print(f"\n{'='*60}\nGeneration {gen}/{args.generations}\n{'='*60}")

        top_terms = get_top_fraud_terms(model, vectorizer, k=15)
        print(f"Current detector's top fraud-indicator terms: {top_terms}")

        batch = generate_evasion_batch(top_terms, args.n_per_generation, gen)
        batch_df = pd.DataFrame([s.to_dict() for s in batch])
        if batch_df.empty or (batch_df["label"] == 1).sum() == 0:
            print(f"  [warning] generation {gen} produced no usable fraud samples, skipping "
                  f"this round's retrain -- check GROQ_API_KEY / rate limits.")
            continue

        # THE headline metric: how well did this batch evade the detector it was built against?
        pre_retrain_eval = evaluate_on(model, vectorizer, batch_df[batch_df["label"] == 1])
        evasion_rate = round(1 - pre_retrain_eval.get("recall", 0), 4)
        print(f"\n  >>> Evasion rate against generation-{gen-1} detector: {evasion_rate} "
              f"(caught {pre_retrain_eval.get('recall', 0)*100:.1f}% of this batch's fraud attempts)")

        # persist this generation's batch -- lands in data/generated/, so it's automatically
        # picked up by train_baseline.py / train_transformer.py on any future run too
        gen_path = os.path.join(GENERATED_DIR, f"adversarial_gen{gen}.jsonl")
        with open(gen_path, "w") as f:
            for s in batch:
                f.write(json.dumps(s.to_dict()) + "\n")
        print(f"  Wrote {len(batch)} samples to {gen_path}")
        all_evasion_batches.append(batch_df)

        # retrain on everything accumulated so far
        train_pool_df = pd.concat([train_pool_df, batch_df], ignore_index=True)
        model, vectorizer = fit_model(train_pool_df)

        post_retrain_eval = evaluate_on(model, vectorizer, batch_df[batch_df["label"] == 1])
        print(f"  After retraining: recall on THIS SAME batch = {post_retrain_eval.get('recall', 0)} "
              f"(expected to be high -- it just trained on this data; the real test is next "
              f"generation's evasion rate against this new detector)")

        # check for regression: does the new detector still catch EARLIER generations' evasions?
        regression_check = {}
        for prior_gen_idx, prior_batch in enumerate(all_evasion_batches[:-1], start=1):
            prior_fraud = prior_batch[prior_batch["label"] == 1]
            regression_check[f"gen{prior_gen_idx}_batch"] = evaluate_on(model, vectorizer, prior_fraud)

        holdout_eval = evaluate_on(model, vectorizer, holdout_df) if not holdout_df.empty else {}
        if holdout_eval:
            print(f"  Holdout (real-world data) after this generation: {holdout_eval}")

        loop_log["generations"].append({
            "generation": gen,
            "top_terms_faced": top_terms,
            "batch_size": len(batch),
            "evasion_rate_against_previous_detector": evasion_rate,
            "recall_before_retrain": pre_retrain_eval,
            "recall_after_retrain_same_batch": post_retrain_eval,
            "regression_check_on_earlier_generations": regression_check,
            "holdout_eval_after_retrain": holdout_eval,
        })

        joblib.dump(model, os.path.join(ARTIFACTS_DIR, f"adversarial_gen{gen}_model.joblib"))
        joblib.dump(vectorizer, os.path.join(ARTIFACTS_DIR, f"adversarial_gen{gen}_vectorizer.joblib"))

    with open(LOOP_METRICS_PATH, "w") as f:
        json.dump(loop_log, f, indent=2)
    print(f"\n{'='*60}\nWrote full loop report to {LOOP_METRICS_PATH}")

    print("\nEvasion rate by generation (the headline chart for your demo):")
    for g in loop_log["generations"]:
        print(f"  Generation {g['generation']}: {g['evasion_rate_against_previous_detector']*100:.1f}% "
              f"evasion rate against the detector it faced")


if __name__ == "__main__":
    main()
