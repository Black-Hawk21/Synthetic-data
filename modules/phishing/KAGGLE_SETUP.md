# Kaggle setup for detector/train_transformer.py

Only needed if you're running the RoBERTa fine-tuning script. The rest of the
project (generator, baseline detector) doesn't need any of this.

## 1. Get your data onto Kaggle

Easiest path: zip up `data/generated/` (your synthetic phishing/legit data) and
`data/raw/` (if you added an email CSV) after running the generator locally,
then create a new Kaggle Dataset from that zip and attach it to your notebook.
Also upload `detector/`, `generator/schema.py`, and `generator/dataset_utils.py`
(only schema.py is actually imported by the training script's data loading --
the rest of generator/ isn't needed on Kaggle).

Alternatively, if your uploaded spam.csv (or similar) is the raw Kaggle SMS
Spam Collection format (`v1,v2` columns, ham/spam labels) -- that's exactly
what `generator/prepare_holdout.py` already normalizes automatically. Run
that locally first rather than re-writing the parsing logic on Kaggle.

## 2. Notebook settings

- Settings -> Accelerator -> **GPU T4 x2** (or P100 if available) -- do this
  BEFORE running anything, changing it later restarts the kernel
- Settings -> Internet -> **On** -- needed on first run to download
  `roberta-base`'s pretrained weights from Hugging Face (cached after that)

## 3. Install

```python
!pip install -q transformers
# torch is preinstalled on Kaggle notebooks -- do NOT pip install torch there,
# it'll pull a version without the right CUDA build
```

## 4. Run

```python
!cd detector && python train_transformer.py --epochs 5 --batch-size 16
```

If you hit a CUDA out-of-memory error, drop `--batch-size` to 8, or switch to
the lighter model: `--model-name distilbert-base-uncased`.

## 5. Compare against the baseline

`eval/roberta_metrics.json` and `eval/baseline_metrics.json` use the identical
structure (same breakdown_by_subtype / breakdown_by_difficulty keys), so you
can diff them directly. The interesting comparison for your report is
`breakdown_by_difficulty` -> `adaptive` -> `recall_on_fraud` in both files --
that's where RoBERTa should show an edge the keyword-based baseline can't.
