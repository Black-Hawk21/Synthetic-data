"""Backs the Live KYC Form: a REAL manual onboarding submission (typed
fields + an uploaded document photo + a captured selfie + real client-side
behavioral telemetry + a client-side device fingerprint + the server-
observed IP). This is the "actual product" path referenced in the README --
everything else in the app (Red Team / Onboarding Simulator) is Faker-
generated for bulk demo/training purposes; THIS path processes real bytes
a real person just submitted, using the real (optional) OCR, forensics and
OpenCV pipelines with honest fallbacks when those aren't installed.

No biometric data is stored beyond this process's in-memory dataset /
local SQLite file -- see README "Ethics & Synthetic Data". Nothing here
ever leaves the machine this backend runs on.
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import date, datetime

import numpy as np
from sqlalchemy.orm import Session

from backend.biometric.opencv_adapter import get_opencv_adapter
from backend.blue_team.predict import ModelBundle, score_applicant
from backend.data.generator import generate_legitimate_applicants
from backend.document import ocr as ocr_module
from backend.document.forensics import analyze_document_image
from backend.document.validator import compute_consistency
from backend.graph.features import calculate_graph_features
from backend.services import persistence
from backend.services.state import state

logger = logging.getLogger(__name__)

PRIVATE_IP_PREFIXES = ("127.", "10.", "192.168.", "::1", "localhost")


def _age_from_dob(dob_str: str) -> int:
    try:
        dob = date.fromisoformat(dob_str)
        today = date.today()
        return max(0, today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day)))
    except Exception:
        return 30


def _network_context(client_ip: str) -> dict:
    is_private = any(client_ip.startswith(p) for p in PRIVATE_IP_PREFIXES)
    return {
        "asn_category": "CORPORATE" if is_private else "RESIDENTIAL_ISP",
        "vpn_proxy_probability": 0.05 if is_private else 0.12,
        "geo_consistency": 0.8,
    }


def submit_manual_application(
    session: Session,
    *,
    name: str, date_of_birth: str, address: str, phone: str, email: str,
    document_type: str = "NATIONAL_ID", document_number: str | None = None,
    document_image_bytes: bytes | None = None,
    selfie_image_bytes: bytes | None = None,
    telemetry: dict, device_fingerprint: dict, client_ip: str,
    top_n_factors: int = 6,
) -> dict:
    # Start from a fully-populated synthetic scaffold so every column the
    # model/graph expect exists, then overwrite every field we have a REAL
    # value for. This guarantees schema parity with the bulk-generated data
    # without duplicating ~50 field definitions here.
    scaffold = generate_legitimate_applicants(1, seed=int(uuid.uuid4().int % (2**31))).iloc[0].to_dict()

    notes = []
    applicant_id = f"APP_MANUAL_{uuid.uuid4().hex[:14]}"

    scaffold.update({
        "applicant_id": applicant_id,
        "name": name,
        "age": _age_from_dob(date_of_birth),
        "date_of_birth": date_of_birth,
        "address": address,
        "phone": phone,
        "email": email,
        # A first-time real submission has no bureau history to check
        # against in this demo -- these are honestly "unknown/new", not
        # sampled. A legitimate person will correctly show up as "new"
        # here; that's realistic, not a bug.
        "identity_age_days": 0, "phone_age_days": 0, "email_age_days": 0,
        "address_age_days": 0,
        "document_type": document_type,
        "document_number": document_number or f"MANUAL{uuid.uuid4().hex[:8].upper()}",
        "source": "manual",
        "attack_id": None, "is_fraud": 0, "attack_type": "MANUAL_SUBMISSION", "difficulty": 0.0,
        "created_at": datetime.utcnow().isoformat(),
    })
    notes.append("identity_age_days/phone_age_days/email_age_days/address_age_days are 0 because this demo has no external bureau to check a first-time submission against.")

    # ---- Device / network: REAL identifiers so the identity graph can
    # genuinely detect reuse across repeat submissions from this browser/IP.
    scaffold["device_id"] = f"dev_manual_{device_fingerprint.get('device_id', uuid.uuid4().hex[:12])}"
    scaffold["device_age_days"] = int(device_fingerprint.get("device_age_days", 0))
    scaffold["os"] = device_fingerprint.get("os", scaffold["os"])
    scaffold["browser"] = device_fingerprint.get("browser", scaffold["browser"])
    scaffold["screen_resolution"] = device_fingerprint.get("screen_resolution", scaffold["screen_resolution"])
    scaffold["timezone"] = device_fingerprint.get("timezone", scaffold["timezone"])
    scaffold["language"] = device_fingerprint.get("language", scaffold["language"])
    scaffold["browser_fingerprint_hash"] = "fp_" + hashlib.sha256(
        f"{scaffold['os']}|{scaffold['browser']}|{scaffold['screen_resolution']}|{scaffold['timezone']}|{scaffold['language']}".encode()
    ).hexdigest()[:12]

    scaffold["ip_id"] = f"ip_{client_ip}"
    scaffold.update(_network_context(client_ip))

    # ---- Document: OCR + forensics on the ACTUAL uploaded bytes.
    if document_image_bytes:
        ocr_result = ocr_module.run_ocr(_bytes_to_image(document_image_bytes))
        forensics = analyze_document_image(document_image_bytes)
        consistency = compute_consistency(
            ocr_result.get("fields") or {},
            {"name": name, "date_of_birth": date_of_birth, "document_number": scaffold["document_number"], "address": address},
        )
        scaffold["ocr_confidence"] = round(float(ocr_result.get("confidence", 0.0)), 3)
        scaffold["document_quality"] = forensics.get("document_quality", scaffold["document_quality"])
        scaffold["document_tamper_score"] = forensics.get("document_tamper_score", scaffold["document_tamper_score"])
        scaffold["document_authenticity_score"] = forensics.get("document_authenticity_score", scaffold["document_authenticity_score"])
        scaffold["name_match_score"] = consistency["name_match_score"]
        scaffold["dob_match_score"] = consistency["dob_match_score"]
        scaffold["document_number_consistency"] = consistency["document_number_consistency"]
        scaffold["doc_field_consistency"] = consistency["doc_field_consistency"]
        scaffold["document_age_days"] = 500
        scaffold["expiry_valid"] = 1
        if ocr_result.get("engine") == "simulated_fallback":
            notes.append(f"No OCR engine installed ({ocr_result.get('note')}) -- document match scores are neutral placeholders, not derived from the image text. Install requirements-optional.txt for real OCR.")
        else:
            notes.append(f"Document processed with real OCR ({ocr_result.get('engine')}) + image forensics.")
    else:
        notes.append("No document image uploaded -- document fields use neutral defaults.")
        for k in ["ocr_confidence", "document_quality", "document_tamper_score",
                  "document_authenticity_score", "name_match_score", "dob_match_score",
                  "document_number_consistency", "doc_field_consistency"]:
            scaffold[k] = 0.5

    # ---- Biometric: real OpenCV heuristics on the actual captured photos,
    # honest synthetic fallback if OpenCV isn't installed or no face found.
    adapter = get_opencv_adapter()
    if selfie_image_bytes and adapter.is_available():
        selfie_signal = adapter.analyze_selfie(selfie_image_bytes)
        scaffold["face_quality_score"] = selfie_signal["face_quality_score"]
        scaffold["liveness_score"] = selfie_signal["liveness_score"]
        scaffold["deepfake_probability"] = selfie_signal["deepfake_probability"]
        scaffold["face_hash"] = "face_" + hashlib.sha256(selfie_image_bytes).hexdigest()[:16]
        if not selfie_signal.get("face_detected", True):
            notes.append(f"No face detected in the selfie ({selfie_signal.get('note')}) -- biometric scores reflect that.")
        else:
            notes.append("Selfie processed with a real (heuristic) OpenCV signal: face detection + sharpness/texture-based liveness proxy. Not a production face-matching model -- see biometric/opencv_adapter.py.")

        if document_image_bytes:
            similarity = adapter.compare_faces(document_image_bytes, selfie_image_bytes)
            scaffold["face_similarity_score"] = similarity["face_similarity_score"]
            if similarity.get("note"):
                notes.append(similarity["note"])
        else:
            scaffold["face_similarity_score"] = 0.5
    elif selfie_image_bytes and not adapter.is_available():
        scaffold["face_hash"] = "face_" + hashlib.sha256(selfie_image_bytes).hexdigest()[:16]
        notes.append("OpenCV not installed -- biometric scores are neutral placeholders. `pip install opencv-python-headless` for real face-detection heuristics.")
        for k in ["face_similarity_score", "liveness_score", "deepfake_probability", "face_quality_score"]:
            scaffold[k] = 0.6
    else:
        notes.append("No selfie captured -- biometric fields use neutral defaults.")
        scaffold["face_hash"] = f"face_{uuid.uuid4().hex[:16]}"
        for k in ["face_similarity_score", "liveness_score", "deepfake_probability", "face_quality_score"]:
            scaffold[k] = 0.5

    # ---- Behavior: REAL client-captured telemetry.
    for key in ["session_duration_sec", "form_completion_time_sec", "typing_speed_cps",
                "typing_variance", "mouse_entropy", "num_corrections",
                "avg_time_between_fields_sec", "automation_score"]:
        if key in telemetry and telemetry[key] is not None:
            scaffold[key] = float(telemetry[key])

    # application_velocity computed server-side from real prior submissions
    # sharing this device or IP -- not client-reported (a scripted client
    # could lie about its own timing, it can't erase the dataset history).
    existing = state.get_dataset()
    prior_shared = 0
    if existing is not None and len(existing):
        prior_shared = int((
            (existing["device_id"] == scaffold["device_id"]) | (existing["ip_id"] == scaffold["ip_id"])
        ).sum())
        scaffold["previous_application_count"] = int((
            (existing["phone"] == phone) | (existing["email"] == email)
        ).sum())
    scaffold["application_velocity"] = float(min(prior_shared, 10))

    # ---- Persist + fold into the live dataset so the identity graph and
    # future reuse checks see this real submission too.
    import pandas as pd
    row_df = pd.DataFrame([scaffold])
    full = state.append_dataset(row_df)
    full = calculate_graph_features(full)
    state.set_dataset(full)
    persistence.save_applicants(session, row_df)

    final_row = full[full["applicant_id"] == applicant_id].iloc[0]

    result = {"applicant": final_row.to_dict(), "notes": notes}

    bundle = ModelBundle.load()
    if bundle is not None:
        verification = score_applicant(final_row, bundle, explain=True, top_n=top_n_factors)
        result["verification"] = verification
        persistence.save_prediction(
            session, applicant_id, bundle.version, verification["fraud_probability"],
            verification["risk_score"], verification["decision"], None, verification["risk_factors"],
        )
    else:
        result["verification"] = None
        result["notes"].append("No trained model yet -- run Blue Team training first to get a live risk score.")

    return result


def _bytes_to_image(image_bytes: bytes):
    import io
    from PIL import Image
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")
