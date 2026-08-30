from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy.orm import Session

from backend.blue_team.predict import ModelBundle, decision_from_score, score_applicant, score_dataframe
from backend.blue_team.train import train_and_evaluate
from backend.config import settings
from backend.services import persistence
from backend.services.state import state

logger = logging.getLogger(__name__)


class NoModelError(Exception):
    pass


class NoDatasetError(Exception):
    pass


def _require_dataset() -> pd.DataFrame:
    df = state.get_dataset()
    if df is None or len(df) == 0:
        raise NoDatasetError("No dataset yet -- call POST /api/generate-applicants first.")
    return df


def _require_model() -> ModelBundle:
    bundle = ModelBundle.load()
    if bundle is None:
        raise NoModelError("No trained model yet -- call POST /api/run-blue-team first.")
    return bundle


def run_blue_team(session: Session, seed: int = 42) -> dict:
    df = _require_dataset()
    if df["is_fraud"].nunique() < 2:
        raise NoDatasetError("Need both legitimate and fraud examples before training -- generate some attacks first.")

    result = train_and_evaluate(df, seed=seed)
    persistence.save_model_version(session, result["version"], result["final_model_metrics"], {"comparison": result["comparison_metrics"]})
    persistence.save_training_run(session, result["version"], len(df), {
        "comparison_metrics": result["comparison_metrics"],
        "per_attack_type_recall": result["per_attack_type_recall"],
    })
    return {
        "model_version": result["version"],
        "comparison_metrics": result["comparison_metrics"],
        "final_model_metrics": result["final_model_metrics"],
        "per_attack_type_recall": result["per_attack_type_recall"],
        "n_trained_on": len(df),
    }


def score_applicant_by_id(session: Session, applicant_id: str, top_n: int = 5) -> dict:
    df = _require_dataset()
    bundle = _require_model()
    matches = df[df["applicant_id"] == applicant_id]
    if matches.empty:
        raise ValueError(f"Unknown applicant_id: {applicant_id}")
    row = matches.iloc[0]
    result = score_applicant(row, bundle, explain=True, top_n=top_n)
    result["ground_truth"] = {"is_fraud": int(row["is_fraud"]), "attack_type": row["attack_type"]}
    persistence.save_prediction(
        session, applicant_id, bundle.version, result["fraud_probability"], result["risk_score"],
        result["decision"], int(row["is_fraud"]), result["risk_factors"],
    )
    return result


def run_detection(session: Session, threshold: float | None = None) -> dict:
    df = _require_dataset()
    bundle = _require_model()
    threshold = threshold if threshold is not None else settings.review_threshold  # BLOCK-vs-not, common "detected" bar

    # Manual submissions have unknown ground truth -- exclude from
    # aggregate/labeled metrics (they're still individually scoreable).
    if "source" in df.columns:
        df = df[df["source"] != "manual"].reset_index(drop=True)

    probs = score_dataframe(df, bundle)
    predicted = (probs >= threshold).astype(int)
    y = df["is_fraud"].values

    from backend.blue_team.evaluate import compute_metrics, per_attack_type_recall
    metrics = compute_metrics(y, probs, threshold=threshold)
    per_attack = per_attack_type_recall(df, probs, threshold=threshold)

    return {
        "model_version": bundle.version,
        "threshold": threshold,
        "metrics": metrics,
        "per_attack_type_recall": per_attack,
        "detected": int(((predicted == 1) & (y == 1)).sum()),
        "total_fraud": int(y.sum()),
    }


def attack_results_summary(session: Session) -> dict:
    """Per-attack-type generation + detection summary for the Attack Lab
    page -- real numbers from the current dataset + current model."""
    df = _require_dataset()
    bundle = ModelBundle.load()

    from backend.red_team.registry import list_attacks, get_attack
    out = []
    probs = score_dataframe(df, bundle) if bundle else None
    if probs is not None:
        df = df.copy()
        df["_prob"] = probs

    for attack_type in list_attacks():
        sub = df[df["attack_type"] == attack_type]
        if sub.empty:
            continue
        strategy = get_attack(attack_type)
        detection_rate = None
        if probs is not None:
            detection_rate = float((sub["_prob"] >= settings.review_threshold).mean())
        out.append({
            "attack_type": attack_type,
            "description": strategy.description(),
            "n_generated": int(len(sub)),
            "avg_difficulty": float(sub["difficulty"].mean()),
            "detection_rate": detection_rate,
        })
    return {"attacks": out, "model_version": bundle.version if bundle else None}
