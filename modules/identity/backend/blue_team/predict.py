"""Inference layer: loads the current model version and scores applicants.
Never puts ML logic in route handlers (section 17) -- routes call into this
module via services/scoring_service.py.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.blue_team.explain import explain_instance
from backend.blue_team.preprocessing import FeatureSchema, transform
from backend.config import settings

logger = logging.getLogger(__name__)


def decision_from_score(score: float, approve_th: float | None = None, review_th: float | None = None) -> str:
    approve_th = approve_th if approve_th is not None else settings.approve_threshold
    review_th = review_th if review_th is not None else settings.review_threshold
    if score < approve_th:
        return "APPROVE"
    if score < review_th:
        return "REVIEW"
    return "BLOCK"


@dataclass
class ModelBundle:
    version: int
    model: object
    scaler: object
    schema: FeatureSchema
    metadata: dict

    @classmethod
    def load(cls, models_dir: Path | None = None, version: int | None = None) -> "ModelBundle | None":
        models_dir = models_dir or settings.models_dir
        if version is None:
            pointer = models_dir / "current_version.json"
            if not pointer.exists():
                return None
            with open(pointer) as f:
                version = json.load(f)["current_version"]

        version_dir = models_dir / f"model_v{version}"
        model_path = version_dir / "model.pkl"
        if not model_path.exists():
            return None

        model = joblib.load(model_path)
        scaler = joblib.load(version_dir / "scaler.pkl") if (version_dir / "scaler.pkl").exists() else None
        schema = FeatureSchema.load(str(version_dir / "feature_schema.json"))
        metadata = {}
        meta_path = version_dir / "model_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
        return cls(version=version, model=model, scaler=scaler, schema=schema, metadata=metadata)


def score_dataframe(df: pd.DataFrame, bundle: ModelBundle) -> np.ndarray:
    X = transform(df, bundle.schema)
    proba = bundle.model.predict_proba(X.values)[:, 1]
    return proba


def score_applicant(row: pd.Series, bundle: ModelBundle, explain: bool = True, top_n: int = 5) -> dict:
    df_row = pd.DataFrame([row])
    X = transform(df_row, bundle.schema)
    proba = float(bundle.model.predict_proba(X.values)[:, 1][0])
    risk_score = round(proba * 100, 2)
    decision = decision_from_score(proba)

    result = {
        "applicant_id": row.get("applicant_id"),
        "fraud_probability": round(proba, 4),
        "risk_score": risk_score,
        "decision": decision,
        "model_version": bundle.version,
    }
    if explain:
        result["risk_factors"] = explain_instance(
            bundle.model, X.values[0], bundle.schema.model_feature_order, row, top_n=top_n,
        )
    return result
