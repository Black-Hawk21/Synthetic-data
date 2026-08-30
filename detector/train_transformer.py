"""
Stronger blue-team detector: fine-tuned RoBERTa (roberta-base).

Designed to run on Kaggle's free GPU notebooks (T4/P100), where torch +
transformers come pre-installed. Mirrors train_baseline.py's data loading
and breakdown-metrics logic so the two reports are directly comparable.

WHY THIS EXISTS ALONGSIDE THE BASELINE (not instead of it):
TF-IDF + Logistic Regression is already near-perfect on "naive"/"moderate"
difficulty attacks (obvious keywords: "click the link", "verify", "OTP").
Where it's expected to struggle is "adaptive" difficulty -- messages
specifically written to avoid obvious trigger words. RoBERTa's contextual
understanding should show its value THERE. Compare
eval/baseline_metrics.json vs eval/roberta_metrics.json, specifically the
breakdown_by_difficulty sections, for the actual story to put in the report.

WARNING on data size: with a small dataset (a few hundred to low thousands
of examples), fine-tuning a 125M-parameter model can overfit and end up
WORSE than the baseline. If your data/generated/ directory only has a few
hundred rows, run generate_static.py with a higher --n-per-cell first, or
at least don't be surprised if this script's own held-out test metrics
look shakier than the baseline's -- report that honestly, it's a real
finding, not a bug.

Kaggle setup:
    1. New Notebook -> Settings -> Accelerator -> GPU T4 x2 (or P100)
    2. Settings -> Internet -> On (needed to download roberta-base weights
       from Hugging Face on first run)
    3. Upload this whole project/ folder (or just data/generated/ +
       detector/ + generator/schema.py, dataset_utils.py needs schema.py's
       sibling files to not be imported, so simplest is uploading the
       whole repo as a Kaggle Dataset and adding it to the notebook)
    4. !pip install -q transformers  # torch is already preinstalled on Kaggle
    5. Run: !python detector/train_transformer.py

Usage (also fine on any machine with a GPU, not just Kaggle):
    cd detector
    python train_transformer.py --epochs 5 --batch-size 16
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from dataset_utils import load_all, summarize

try:
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, EarlyStoppingCallback,
    )
except ImportError as e:
    print(f"[fatal] missing dependency: {e}\n"
          f"On Kaggle, run: !pip install -q transformers  (torch is preinstalled)\n"
          f"Elsewhere: pip install torch transformers --break-system-packages",
          file=sys.stderr)
    sys.exit(1)

MODEL_NAME = "roberta-base"
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts", "roberta")
EVAL_DIR = os.path.join(os.path.dirname(__file__), "..", "eval")
METRICS_PATH = os.path.join(EVAL_DIR, "roberta_metrics.json")
MAX_LENGTH = 256  # phishing emails/SMS are short; 256 tokens covers almost all of them


class TextDataset(Dataset):
    """Plain torch Dataset -- avoids pulling in the separate `datasets`
    package just for this. Tokenizes lazily per-item, which is fine at
    hackathon data scale (low thousands of rows at most)."""

    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def compute_breakdown(df: pd.DataFrame, y_true, y_pred, group_col: str) -> dict:
    """Same logic as train_baseline.py's compute_breakdown, duplicated here
    (rather than imported) so this script stays runnable standalone on a
    fresh Kaggle notebook where you may not have copied over every file."""
    out = {}
    tmp = df.copy()
    tmp["_pred"] = y_pred
    tmp["_true"] = y_true
    for group_val, sub in tmp.groupby(group_col):
        if (sub["_true"] == 1).sum() == 0:
            fp_rate = float((sub["_pred"] == 1).mean())
            out[str(group_val)] = {"n": int(len(sub)), "false_positive_rate": round(fp_rate, 4)}
        else:
            fraud_sub = sub[sub["_true"] == 1]
            recall = float((fraud_sub["_pred"] == 1).mean())
            out[str(group_val)] = {"n": int(len(sub)), "recall_on_fraud": round(recall, 4)}
    return out


def evaluate_df(name: str, df: pd.DataFrame, model, tokenizer, device, batch_size=32) -> dict:
    if df.empty:
        return {"note": f"{name} set is empty, skipped"}

    model.eval()
    all_preds, all_probs = [], []
    texts = df["text"].fillna("").tolist()

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, truncation=True, max_length=MAX_LENGTH,
                             padding=True, return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())

    y_true = df["label"].values
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    metrics = {
        "n": int(len(df)),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }
    if len(set(y_true)) > 1:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    metrics["false_positive_rate_on_legit"] = round(float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0, 4)

    if "attack_subtype" in df.columns:
        metrics["breakdown_by_subtype"] = compute_breakdown(df, y_true, y_pred, "attack_subtype")
    if "difficulty_tier" in df.columns:
        metrics["breakdown_by_difficulty"] = compute_breakdown(df, y_true, y_pred, "difficulty_tier")

    print(f"\n=== {name} ===")
    print(f"  n={metrics['n']}  precision={metrics['precision']}  recall={metrics['recall']}  "
          f"f1={metrics['f1']}  auc={metrics.get('roc_auc', 'n/a')}")
    print(f"  false positive rate on legit: {metrics['false_positive_rate_on_legit']}")
    if "breakdown_by_difficulty" in metrics:
        print(f"  by difficulty: {json.dumps(metrics['breakdown_by_difficulty'], indent=2)}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--model-name", type=str, default=MODEL_NAME,
                         help="swap to 'distilbert-base-uncased' for a lighter/faster alternative")
    args = parser.parse_args()

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("[warning] no GPU detected -- fine-tuning roberta-base on CPU will be very slow. "
              "On Kaggle: Settings -> Accelerator -> GPU T4 x2.")

    train_pool_df, holdout_df = load_all()
    summarize(train_pool_df, "train pool")
    summarize(holdout_df, "holdout")

    if train_pool_df.empty or train_pool_df["label"].nunique() < 2:
        print("\nNo usable generated data found -- run generate_static.py first.", file=sys.stderr)
        sys.exit(1)

    if len(train_pool_df) < 200:
        print(f"\n[warning] only {len(train_pool_df)} rows in the training pool. Fine-tuning "
              f"{args.model_name} on this little data risks overfitting and may underperform "
              f"the TF-IDF baseline. Consider raising --n-per-cell in generate_static.py first. "
              f"Continuing anyway...")

    train_df, temp_df = train_test_split(
        train_pool_df, test_size=0.3, random_state=42, stratify=train_pool_df["label"])
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])
    print(f"\nSplit: {len(train_df)} train / {len(val_df)} val / {len(test_df)} test")

    print(f"Loading tokenizer + model: {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    model.to(device)

    train_ds = TextDataset(train_df["text"], train_df["label"], tokenizer)
    val_ds = TextDataset(val_df["text"], val_df["label"], tokenizer)

    training_args = TrainingArguments(
        output_dir=os.path.join(ARTIFACTS_DIR, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=10,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\nTraining...")
    trainer.train()

    final_model_dir = os.path.join(ARTIFACTS_DIR, "final")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"\nSaved fine-tuned model to {final_model_dir}")

    all_metrics = {
        "model_name": args.model_name,
        "test_split": evaluate_df("Test split (held-out from our own generated data)",
                                   test_df, model, tokenizer, device),
    }
    if not holdout_df.empty:
        all_metrics["public_holdout"] = evaluate_df(
            "Public holdout (real-world data, generalization check)",
            holdout_df, model, tokenizer, device)
    else:
        print("\n[note] no public holdout data found -- run prepare_holdout.py for a "
              "generalization check.")

    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nWrote full metrics report to {METRICS_PATH}")

    baseline_path = os.path.join(EVAL_DIR, "baseline_metrics.json")
    if os.path.exists(baseline_path):
        print(f"\nCompare against {baseline_path} -- specifically the breakdown_by_difficulty "
              f"sections -- to see whether RoBERTa actually catches more 'adaptive' tier attacks "
              f"than the TF-IDF baseline. That comparison is the point of having both models.")


if __name__ == "__main__":
    main()
