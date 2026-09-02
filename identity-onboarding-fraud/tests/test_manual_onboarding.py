from __future__ import annotations

import io

import pandas as pd
import pytest
from PIL import Image

from backend.blue_team.train import train_and_evaluate
from backend.document.forensics import analyze_document_image
from backend.document.generator import generate_document_image, random_synthetic_fields
from backend.graph.features import calculate_graph_features
from backend.red_team.registry import get_attack
from backend.services import manual_onboarding_service
from backend.services.state import state


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def synthetic_doc_bytes():
    from faker import Faker
    fake = Faker()
    fields = random_synthetic_fields(fake, seed=1)
    return _png_bytes(generate_document_image(fields))


def test_analyze_document_image_returns_bounded_scores(synthetic_doc_bytes):
    result = analyze_document_image(synthetic_doc_bytes)
    assert 0.0 <= result["document_quality"] <= 1.0
    assert 0.0 <= result["document_tamper_score"] <= 1.0
    assert 0.0 <= result["document_authenticity_score"] <= 1.0


def test_analyze_document_image_handles_garbage_bytes():
    result = analyze_document_image(b"not an image")
    assert result.get("error") == "could_not_decode_image"


@pytest.fixture(autouse=True)
def _reset_state(tmp_path, monkeypatch):
    # Isolate global dataset + model state per test.
    state.dataset = None
    monkeypatch.setattr("backend.config.settings.models_dir", tmp_path)
    yield
    state.dataset = None


@pytest.fixture
def trained_bundle_for_manual(small_legit_df, tmp_path):
    attacks = get_attack("FRAUD_RING").generate(60, difficulty=0.4, seed=1)
    df = calculate_graph_features(pd.concat([small_legit_df, attacks], ignore_index=True))
    state.set_dataset(df)
    train_and_evaluate(df, models_dir=tmp_path, seed=42)


def test_submit_manual_application_without_files(db_session, trained_bundle_for_manual):
    result = manual_onboarding_service.submit_manual_application(
        db_session,
        name="Jane Tester", date_of_birth="1992-06-01", address="1 Test Way",
        phone="5550001111", email="jane.tester@example.com",
        document_image_bytes=None, selfie_image_bytes=None,
        telemetry={"session_duration_sec": 40, "automation_score": 0.1},
        device_fingerprint={"device_id": "unit_test_device", "os": "Linux", "browser": "Firefox"},
        client_ip="203.0.113.5",
    )
    assert result["applicant"]["source"] == "manual"
    assert result["applicant"]["is_fraud"] == 0
    assert result["verification"] is not None
    assert result["verification"]["decision"] in {"APPROVE", "REVIEW", "BLOCK"}


def test_submit_manual_application_with_document(db_session, trained_bundle_for_manual, synthetic_doc_bytes):
    result = manual_onboarding_service.submit_manual_application(
        db_session,
        name="Someone Else", date_of_birth="1988-01-01", address="2 Test Way",
        phone="5550002222", email="someone.else@example.com",
        document_image_bytes=synthetic_doc_bytes, selfie_image_bytes=None,
        telemetry={}, device_fingerprint={"device_id": "unit_test_device_2"},
        client_ip="203.0.113.6",
    )
    applicant = result["applicant"]
    # Name typed does NOT match the random name baked into the sample doc ->
    # consistency score should reflect a real mismatch, not a neutral default.
    assert applicant["name_match_score"] < 0.9


def test_manual_submissions_excluded_from_training(small_legit_df, db_session, tmp_path):
    attacks = get_attack("FRAUD_RING").generate(40, difficulty=0.4, seed=2)
    df = calculate_graph_features(pd.concat([small_legit_df, attacks], ignore_index=True))
    state.set_dataset(df)
    train_and_evaluate(df, models_dir=tmp_path, seed=42)

    manual_onboarding_service.submit_manual_application(
        db_session,
        name="Repeat Visitor", date_of_birth="1980-01-01", address="3 Test Way",
        phone="5550003333", email="repeat@example.com",
        document_image_bytes=None, selfie_image_bytes=None,
        telemetry={}, device_fingerprint={"device_id": "unit_test_device_3"},
        client_ip="203.0.113.7",
    )
    combined = state.get_dataset()
    assert (combined["source"] == "manual").sum() == 1

    result = train_and_evaluate(combined, models_dir=tmp_path, seed=1)
    # The manual row must never appear in what got trained/tested on.
    assert "manual" not in result["held_out_test_df"]["source"].values


def test_device_and_ip_reuse_detected_across_manual_submissions(db_session, trained_bundle_for_manual):
    common_fp = {"device_id": "shared_device_xyz"}
    manual_onboarding_service.submit_manual_application(
        db_session, name="Person One", date_of_birth="1991-01-01", address="A",
        phone="5551110000", email="one@example.com", document_image_bytes=None,
        selfie_image_bytes=None, telemetry={}, device_fingerprint=common_fp, client_ip="198.51.100.1",
    )
    result2 = manual_onboarding_service.submit_manual_application(
        db_session, name="Person Two", date_of_birth="1992-01-01", address="B",
        phone="5552220000", email="two@example.com", document_image_bytes=None,
        selfie_image_bytes=None, telemetry={}, device_fingerprint=common_fp, client_ip="198.51.100.1",
    )
    assert result2["applicant"]["device_reuse_count"] >= 1
    assert result2["applicant"]["ip_reuse_count"] >= 1
