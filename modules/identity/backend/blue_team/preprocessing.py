"""Turns a raw applicant dataframe (as produced by data.generator +
red_team.* + graph.features) into a clean numeric matrix for scikit-learn /
XGBoost, with a fitted, reusable transformer for inference-time consistency.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from backend.data.schemas import (
    ALL_FEATURE_GROUPS, CATEGORICAL_MODEL_FEATURES, DROP_FROM_MODEL,
)


@dataclass
class FeatureSchema:
    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    categorical_values: dict[str, list[str]] = field(default_factory=dict)
    model_feature_order: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "categorical_values": self.categorical_values,
            "model_feature_order": self.model_feature_order,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureSchema":
        return cls(**d)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FeatureSchema":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def _model_input_columns(df: pd.DataFrame) -> list[str]:
    all_feature_cols = sum(ALL_FEATURE_GROUPS.values(), [])
    return [c for c in all_feature_cols if c in df.columns and c not in DROP_FROM_MODEL]


def fit_schema(df: pd.DataFrame) -> FeatureSchema:
    cols = _model_input_columns(df)
    categorical = [c for c in cols if c in CATEGORICAL_MODEL_FEATURES]
    numeric = [c for c in cols if c not in categorical]
    cat_values = {c: sorted(df[c].astype(str).unique().tolist()) for c in categorical}

    model_feature_order = list(numeric)
    for c in categorical:
        model_feature_order += [f"{c}__{v}" for v in cat_values[c]]

    return FeatureSchema(
        numeric_features=numeric, categorical_features=categorical,
        categorical_values=cat_values, model_feature_order=model_feature_order,
    )


def transform(df: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    """Deterministic transform: numeric passthrough + one-hot encode using
    the *fitted* category vocabulary (unseen categories -> all-zero row,
    matching production behavior for unseen values)."""
    out = pd.DataFrame(index=df.index)
    for c in schema.numeric_features:
        out[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    for c in schema.categorical_features:
        vals = df[c].astype(str)
        for v in schema.categorical_values[c]:
            out[f"{c}__{v}"] = (vals == v).astype(int)

    # Ensure exact column order / presence for the model
    for col in schema.model_feature_order:
        if col not in out.columns:
            out[col] = 0
    return out[schema.model_feature_order]


def build_feature_matrix(df: pd.DataFrame, schema: FeatureSchema | None = None) -> tuple[pd.DataFrame, FeatureSchema]:
    if schema is None:
        schema = fit_schema(df)
    X = transform(df, schema)
    return X, schema


def fit_scaler(X: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X.values)
    return scaler
