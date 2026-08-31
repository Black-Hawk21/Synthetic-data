from __future__ import annotations

import pytest

from backend.data.schemas import ATTACK_TYPES
from backend.red_team.registry import get_attack, list_attacks


def test_all_20_attack_types_registered():
    registered = set(list_attacks())
    assert set(ATTACK_TYPES) == registered, f"missing: {set(ATTACK_TYPES) - registered}, extra: {registered - set(ATTACK_TYPES)}"


@pytest.mark.parametrize("attack_type", ATTACK_TYPES)
def test_attack_generates_labeled_fraud_records(attack_type):
    strategy = get_attack(attack_type)
    df = strategy.generate(30, difficulty=0.5, seed=1)
    assert len(df) == 30
    assert (df["is_fraud"] == 1).all()
    assert (df["attack_type"] == attack_type).all()
    assert df["attack_id"].nunique() == 30


@pytest.mark.parametrize("attack_type", ATTACK_TYPES)
def test_attack_has_description_and_severity(attack_type):
    strategy = get_attack(attack_type)
    assert isinstance(strategy.description(), str) and len(strategy.description()) > 10
    assert strategy.severity(0.9) in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert len(strategy.features_affected) > 0


def test_difficulty_changes_output_distribution():
    """Sanity check that difficulty actually perturbs the generated
    distribution (not a no-op parameter)."""
    strategy = get_attack("DOCUMENT_TAMPERING")
    easy = strategy.generate(200, difficulty=0.05, seed=1)
    hard = strategy.generate(200, difficulty=0.95, seed=1)
    assert easy["document_tamper_score"].mean() != hard["document_tamper_score"].mean()


def test_device_reuse_attack_actually_creates_shared_devices():
    strategy = get_attack("DEVICE_REUSE")
    df = strategy.generate(100, difficulty=0.1, seed=1)  # low difficulty = heavy reuse
    assert df["device_id"].nunique() < len(df)


def test_unknown_attack_type_raises():
    with pytest.raises(KeyError):
        get_attack("NOT_A_REAL_ATTACK")
