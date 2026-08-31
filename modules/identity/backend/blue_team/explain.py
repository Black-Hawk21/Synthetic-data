"""Explainability layer (section 8). Prefers SHAP TreeExplainer; falls back
to a feature-importance x deviation proxy if SHAP is unavailable or errors,
so the API never breaks because an optional dependency is missing."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import shap
    _SHAP_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _SHAP_AVAILABLE = False


def _human_readable(feature: str, raw_row: pd.Series) -> str:
    templates = {
        "device_reuse_count": lambda r: f"Device shared with {int(r.get('device_reuse_count', 0))} other identities",
        "device_identity_count": lambda r: f"Device associated with {int(r.get('device_identity_count', 0))} identities total",
        "ip_reuse_count": lambda r: f"IP address shared with {int(r.get('ip_reuse_count', 0))} other identities",
        "identities_from_ip": lambda r: f"IP address associated with {int(r.get('identities_from_ip', 0))} applications",
        "applications_from_ip": lambda r: f"IP address associated with {int(r.get('applications_from_ip', 0))} applications",
        "phone_reuse_count": lambda r: f"Phone number shared with {int(r.get('phone_reuse_count', 0))} other identities",
        "email_reuse_count": lambda r: f"Email shared with {int(r.get('email_reuse_count', 0))} other identities",
        "address_reuse_count": lambda r: f"Address shared with {int(r.get('address_reuse_count', 0))} other identities",
        "face_similarity_score": lambda r: f"Face/document similarity is low ({float(r.get('face_similarity_score', 0)):.2f})",
        "liveness_score": lambda r: f"Liveness score is low ({float(r.get('liveness_score', 0)):.2f}) -- possible spoofing",
        "deepfake_probability": lambda r: f"Deepfake probability is elevated ({float(r.get('deepfake_probability', 0)):.2f})",
        "face_quality_score": lambda r: f"Face image quality is unusual ({float(r.get('face_quality_score', 0)):.2f})",
        "face_reuse_count": lambda r: f"Face signature reused across {int(r.get('face_reuse_count', 0))} applications",
        "document_tamper_score": lambda r: f"Document tamper score is high ({float(r.get('document_tamper_score', 0)):.2f})",
        "document_authenticity_score": lambda r: f"Document authenticity score is low ({float(r.get('document_authenticity_score', 0)):.2f})",
        "doc_field_consistency": lambda r: f"Document field consistency is low ({float(r.get('doc_field_consistency', 0)):.2f})",
        "name_match_score": lambda r: f"Name does not consistently match across application and document ({float(r.get('name_match_score', 0)):.2f})",
        "dob_match_score": lambda r: f"Date of birth does not consistently match ({float(r.get('dob_match_score', 0)):.2f})",
        "document_number_consistency": lambda r: f"Document number consistency is low ({float(r.get('document_number_consistency', 0)):.2f})",
        "ocr_confidence": lambda r: f"OCR confidence on document is low ({float(r.get('ocr_confidence', 0)):.2f})",
        "identity_age_days": lambda r: f"Identity/contact details are only {int(r.get('identity_age_days', 0))} days old",
        "phone_age_days": lambda r: f"Phone number is only {int(r.get('phone_age_days', 0))} days old",
        "email_age_days": lambda r: f"Email address is only {int(r.get('email_age_days', 0))} days old",
        "application_velocity": lambda r: f"Application velocity is unusually high ({float(r.get('application_velocity', 0)):.2f}/hr)",
        "automation_score": lambda r: f"Form-filling behavior looks automated (score {float(r.get('automation_score', 0)):.2f})",
        "typing_variance": lambda r: f"Typing rhythm is unnaturally consistent (variance {float(r.get('typing_variance', 0)):.2f})",
        "mouse_entropy": lambda r: f"Mouse-movement entropy is low ({float(r.get('mouse_entropy', 0)):.2f}) -- possible scripted input",
        "suspicious_cluster_score": lambda r: f"Applicant belongs to a suspicious identity cluster (score {float(r.get('suspicious_cluster_score', 0)):.2f})",
        "connected_component_size": lambda r: f"Applicant is graph-connected to {max(int(r.get('connected_component_size', 1)) - 1, 0)} other identities via shared infrastructure",
        "vpn_proxy_probability": lambda r: f"Network shows elevated VPN/proxy probability ({float(r.get('vpn_proxy_probability', 0)):.2f})",
        "geo_consistency": lambda r: f"IP geolocation is inconsistent with stated address ({float(r.get('geo_consistency', 0)):.2f})",
        "previous_application_count": lambda r: f"Applicant has {int(r.get('previous_application_count', 0))} prior applications on file",
    }
    base = feature.split("__")[0]
    if base in templates:
        try:
            return templates[base](raw_row)
        except Exception:
            pass
    return f"{base.replace('_', ' ')} = {raw_row.get(base, 'n/a')}"


def explain_instance(model, X_row: np.ndarray, feature_names: list[str], raw_row: pd.Series, top_n: int = 5) -> list[dict]:
    """Returns top_n risk factors as [{feature, contribution, description}]
    sorted by how much they pushed the score toward FRAUD."""
    contributions: np.ndarray | None = None

    if _SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_row.reshape(1, -1))
            contributions = np.array(sv).reshape(-1)
        except Exception as e:  # pragma: no cover
            logger.warning("SHAP explanation failed, falling back: %s", e)
            contributions = None

    if contributions is None:
        # Fallback: global feature_importances_ weighted by how far this
        # instance's (roughly standardized) value sits from a neutral 0,
        # signed toward higher values = more suspicious for count/score
        # features. This is a coarse but dependency-free approximation.
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            importances = np.ones(len(feature_names)) / len(feature_names)
        contributions = importances * (X_row - np.median(X_row))

    order = np.argsort(-contributions)
    results = []
    for i in order:
        if contributions[i] <= 0:
            continue
        feat = feature_names[i]
        results.append({
            "feature": feat,
            "contribution": round(float(contributions[i]), 5),
            "description": _human_readable(feat, raw_row),
        })
        if len(results) >= top_n:
            break
    return results
