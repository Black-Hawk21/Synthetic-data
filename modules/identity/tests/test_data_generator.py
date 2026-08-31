from __future__ import annotations

from backend.data.generator import generate_legitimate_applicants
from backend.data.schemas import ALL_FEATURE_GROUPS


def test_generates_requested_row_count():
    df = generate_legitimate_applicants(250, seed=1)
    assert len(df) == 250


def test_all_applicants_unique():
    df = generate_legitimate_applicants(300, seed=2)
    assert df["applicant_id"].nunique() == len(df)


def test_all_legit_are_unlabeled_as_fraud():
    df = generate_legitimate_applicants(200, seed=3)
    assert (df["is_fraud"] == 0).all()
    assert (df["attack_type"] == "NONE").all()


def test_feature_groups_present():
    df = generate_legitimate_applicants(100, seed=4)
    for group, cols in ALL_FEATURE_GROUPS.items():
        for col in cols:
            assert col in df.columns, f"missing {col} from group {group}"


def test_scores_are_bounded_0_1():
    df = generate_legitimate_applicants(400, seed=5)
    bounded_cols = [
        "document_quality", "ocr_confidence", "face_similarity_score",
        "liveness_score", "deepfake_probability",
    ]
    for col in bounded_cols:
        assert df[col].between(0, 1).all(), col


def test_reproducible_with_same_seed():
    a = generate_legitimate_applicants(50, seed=99)
    b = generate_legitimate_applicants(50, seed=99)
    assert a["name"].tolist() == b["name"].tolist()
    assert (a["face_similarity_score"] == b["face_similarity_score"]).all()


def test_configurable_scale():
    for n in [10, 1000]:
        df = generate_legitimate_applicants(n, seed=1)
        assert len(df) == n
