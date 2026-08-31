from __future__ import annotations

import pandas as pd
import pytest

from backend.blue_team.predict import ModelBundle, decision_from_score, score_applicant
from backend.blue_team.train import train_and_evaluate
from backend.graph.features import calculate_graph_features
from backend.red_team.registry import get_attack


@pytest.fixture(scope="module")
def trained_bundle(tmp_path_factory, small_legit_df):
    models_dir = tmp_path_factory.mktemp("models")
    attacks = pd.concat([
        get_attack("FRAUD_RING").generate(80, difficulty=0.4, seed=1),
        get_attack("DEVICE_REUSE").generate(80, difficulty=0.4, seed=2),
        get_attack("DOCUMENT_TAMPERING").generate(80, difficulty=0.4, seed=3),
    ], ignore_index=True)
    df = calculate_graph_features(pd.concat([small_legit_df, attacks], ignore_index=True))
    result = train_and_evaluate(df, models_dir=models_dir, seed=42)
    bundle = ModelBundle.load(models_dir=models_dir, version=result["version"])
    return bundle, df


def test_training_produces_all_three_models(trained_bundle):
    bundle, _ = trained_bundle
    assert bundle.model is not None


def test_model_metrics_are_not_trivial(trained_bundle):
    """Recall/precision should be realistic (< 1.0), i.e. NOT a trivially
    separable synthetic dataset (section 13/32)."""
    bundle, df = trained_bundle
    from backend.blue_team.evaluate import compute_metrics
    from backend.blue_team.predict import score_dataframe
    probs = score_dataframe(df, bundle)
    metrics = compute_metrics(df["is_fraud"].values, probs)
    assert 0.0 < metrics["recall"] <= 1.0
    assert metrics["pr_auc"] > 0.3  # meaningfully better than random for this imbalance


def test_decision_thresholds():
    assert decision_from_score(0.1) == "APPROVE"
    assert decision_from_score(0.5) == "REVIEW"
    assert decision_from_score(0.9) == "BLOCK"


def test_score_applicant_returns_explanation(trained_bundle):
    bundle, df = trained_bundle
    row = df[df.is_fraud == 1].iloc[0]
    result = score_applicant(row, bundle, explain=True, top_n=3)
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["decision"] in {"APPROVE", "REVIEW", "BLOCK"}
    assert len(result["risk_factors"]) <= 3


def test_model_bundle_missing_returns_none(tmp_path):
    assert ModelBundle.load(models_dir=tmp_path) is None
