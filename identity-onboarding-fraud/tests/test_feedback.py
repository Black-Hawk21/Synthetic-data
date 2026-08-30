from __future__ import annotations

import pandas as pd
import pytest

from backend.blue_team.predict import ModelBundle
from backend.blue_team.train import train_and_evaluate
from backend.feedback.engine import (
    analyze_detector_weaknesses, evaluate_attacks, find_false_negatives,
    generate_harder_attacks, run_closed_loop_iteration,
)
from backend.graph.features import calculate_graph_features
from backend.red_team.registry import get_attack


@pytest.fixture(scope="module")
def seeded_dataset_and_bundle(tmp_path_factory, small_legit_df):
    models_dir = tmp_path_factory.mktemp("fb_models")
    attacks = pd.concat([
        get_attack("FRAUD_RING").generate(60, difficulty=0.3, seed=1),
        get_attack("SYNTHETIC_IDENTITY").generate(60, difficulty=0.3, seed=2),
    ], ignore_index=True)
    df = calculate_graph_features(pd.concat([small_legit_df, attacks], ignore_index=True))
    result = train_and_evaluate(df, models_dir=models_dir, seed=42)
    bundle = ModelBundle.load(models_dir=models_dir, version=result["version"])
    return df, bundle, models_dir


def test_evaluate_attacks_adds_predictions(seeded_dataset_and_bundle):
    df, bundle, _ = seeded_dataset_and_bundle
    scored = evaluate_attacks(df, bundle)
    assert "fraud_probability" in scored.columns
    assert "predicted_fraud" in scored.columns


def test_find_false_negatives_only_returns_missed_fraud(seeded_dataset_and_bundle):
    df, bundle, _ = seeded_dataset_and_bundle
    scored = evaluate_attacks(df, bundle)
    fn = find_false_negatives(scored)
    assert (fn["is_fraud"] == 1).all()
    assert (fn["predicted_fraud"] == 0).all()


def test_analyze_weaknesses_returns_summary(seeded_dataset_and_bundle):
    df, bundle, _ = seeded_dataset_and_bundle
    scored = evaluate_attacks(df, bundle)
    report = analyze_detector_weaknesses(scored)
    assert "summary" in report
    assert isinstance(report["recall_by_attack_type"], dict)


def test_generate_harder_attacks_uses_fallback_without_llm(seeded_dataset_and_bundle):
    df, bundle, _ = seeded_dataset_and_bundle
    scored = evaluate_attacks(df, bundle)
    report = analyze_detector_weaknesses(scored)
    harder = generate_harder_attacks(report, n_per_type=20, use_llm=False, seed=1)
    assert len(harder) > 0
    assert (harder["is_fraud"] == 1).all()
    assert (harder["difficulty"] >= 0.5).all()


def test_closed_loop_iteration_improves_or_reports_metrics(seeded_dataset_and_bundle, monkeypatch, tmp_path):
    df, bundle, models_dir = seeded_dataset_and_bundle
    monkeypatch.setattr("backend.config.settings.models_dir", models_dir)
    result = run_closed_loop_iteration(df, bundle, n_per_type=20, use_llm=False, seed=7)
    assert "before_metrics" in result and "after_metrics" in result
    assert 0.0 <= result["after_metrics"]["recall"] <= 1.0
    assert result["new_attack_count"] > 0


def test_closed_loop_first_iteration_with_no_bundle(small_legit_df, tmp_path, monkeypatch):
    monkeypatch.setattr("backend.config.settings.models_dir", tmp_path)
    attacks = get_attack("DEVICE_REUSE").generate(40, difficulty=0.4, seed=5)
    df = calculate_graph_features(pd.concat([small_legit_df, attacks], ignore_index=True))
    result = run_closed_loop_iteration(df, None, seed=1)
    assert result["before_metrics"] is None
    assert result["after_metrics"]["recall"] >= 0
