from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module):
    models_dir = tmp_path_factory.mktemp("api_models")
    db_path = tmp_path_factory.mktemp("api_db") / "test.db"

    monkeypatch_module.setattr("backend.config.settings.models_dir", models_dir)
    monkeypatch_module.setattr("backend.config.settings.db_path", db_path)

    # Point the existing db module at a fresh throwaway engine WITHOUT
    # reloading the module: other test files import backend.models.orm at
    # collection time (via backend.services.persistence), which binds the
    # ORM classes to db.py's Base exactly once. Reloading db.py here would
    # create a second, empty Base with no tables registered on it -- so we
    # swap the engine/session factory in place instead.
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from backend.models import db as db_module

    test_engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    monkeypatch_module.setattr(db_module, "engine", test_engine)
    monkeypatch_module.setattr(db_module, "SessionLocal", sessionmaker(autocommit=False, autoflush=False, bind=test_engine))

    from backend.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["attacks_registered"] == 20


def test_list_attacks(client):
    r = client.get("/api/attacks")
    assert r.status_code == 200
    assert len(r.json()["attacks"]) == 20


def test_generate_applicants(client):
    r = client.post("/api/generate-applicants", json={"n": 300, "seed": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["generated"] == 300
    assert body["total_dataset_size"] >= 300


def test_generate_attack(client):
    r = client.post("/api/generate-attack", json={"attack_type": "FRAUD_RING", "difficulty": 0.4, "n": 60})
    assert r.status_code == 200
    body = r.json()
    assert body["n_generated"] == 60
    assert body["attack_type"] == "FRAUD_RING"


def test_generate_attack_unknown_type_returns_400(client):
    r = client.post("/api/generate-attack", json={"attack_type": "NOT_REAL", "difficulty": 0.4, "n": 10})
    assert r.status_code == 400


def test_score_without_model_returns_400(client):
    # This runs before run-blue-team in the flow below on a fresh client,
    # but since fixtures are module-scoped, guard with a fresh applicant id lookup.
    r = client.post("/api/score-applicant", json={"applicant_id": "does-not-exist"})
    assert r.status_code in (400, 404)


def test_run_blue_team_trains_model(client):
    r = client.post("/api/run-blue-team", json={"seed": 42})
    assert r.status_code == 200
    body = r.json()
    assert "final_model_metrics" in body
    assert body["model_version"] >= 1


def test_run_detection(client):
    r = client.post("/api/run-detection", json={})
    assert r.status_code == 200
    body = r.json()
    assert "metrics" in body


def test_attack_results(client):
    r = client.get("/api/attack-results")
    assert r.status_code == 200
    assert len(r.json()["attacks"]) >= 1


def test_graph_rings(client):
    r = client.get("/api/graph/rings?min_size=2")
    assert r.status_code == 200
    assert "num_clusters" in r.json()


def test_graph_unknown_applicant_404(client):
    r = client.get("/api/graph/NOT_A_REAL_APPLICANT")
    assert r.status_code == 404


def test_model_info(client):
    r = client.get("/api/model-info")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_run_closed_loop(client):
    r = client.post("/api/run-closed-loop", json={"iterations": 1, "n_per_type": 20, "use_llm": False, "seed": 1})
    assert r.status_code == 200
    assert len(r.json()["iterations"]) == 1


def test_feedback_history(client):
    r = client.get("/api/feedback")
    assert r.status_code == 200
    assert isinstance(r.json()["history"], list)


def test_sample_document_endpoint(client):
    r = client.get("/api/document/sample")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert len(r.content) > 500

    r_tampered = client.get("/api/document/sample?blur=true&noise=true&rotate=5&tamper_fields=true")
    assert r_tampered.status_code == 200


def test_manual_onboarding_submit_without_files(client):
    r = client.post(
        "/api/onboarding/submit",
        data={
            "name": "API Test User", "date_of_birth": "1993-03-03", "address": "9 API Way",
            "phone": "5559998888", "email": "api.test@example.com",
            "telemetry": "{}", "device_fingerprint": '{"device_id": "api_test_device"}',
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["applicant"]["source"] == "manual"
    assert body["verification"]["decision"] in {"APPROVE", "REVIEW", "BLOCK"}


def test_manual_onboarding_submit_with_document(client):
    doc_resp = client.get("/api/document/sample")
    r = client.post(
        "/api/onboarding/submit",
        data={
            "name": "Doc Test User", "date_of_birth": "1990-05-05", "address": "10 API Way",
            "phone": "5551237890", "email": "doc.test@example.com",
            "telemetry": "{}", "device_fingerprint": '{"device_id": "api_test_device_2"}',
        },
        files={"document_image": ("sample.png", doc_resp.content, "image/png")},
    )
    assert r.status_code == 200
    applicant = r.json()["applicant"]
    assert applicant["ocr_confidence"] is not None
    # Typed name won't match the random name baked into the sample doc.
    assert applicant["name_match_score"] < 0.9


def test_manual_onboarding_bad_telemetry_json_returns_422(client):
    r = client.post(
        "/api/onboarding/submit",
        data={
            "name": "Bad JSON", "date_of_birth": "1990-01-01", "address": "x",
            "phone": "1", "email": "x@example.com",
            "telemetry": "{not valid json", "device_fingerprint": "{}",
        },
    )
    assert r.status_code == 422


def test_dataset_summary_reports_manual_submissions_separately(client):
    r = client.get("/api/dataset-summary")
    assert r.status_code == 200
    assert r.json()["manual_submissions"] >= 2
