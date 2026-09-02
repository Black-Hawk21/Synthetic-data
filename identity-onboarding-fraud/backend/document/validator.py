"""Compares OCR-extracted document fields against applicant-submitted
fields and produces consistency scores (section 10, step 5)."""
from __future__ import annotations

from difflib import SequenceMatcher


def _similarity(a: str | None, b: str | None) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()


def compute_consistency(extracted: dict, submitted: dict) -> dict:
    """extracted: OCR fields (name, date_of_birth, document_number, address).
    submitted: applicant-entered fields with the same keys."""
    name_score = _similarity(extracted.get("name"), submitted.get("name"))
    dob_score = _similarity(extracted.get("date_of_birth"), submitted.get("date_of_birth"))
    doc_num_score = _similarity(extracted.get("document_number"), submitted.get("document_number"))
    address_score = _similarity(extracted.get("address"), submitted.get("address"))

    overall = (name_score + dob_score + doc_num_score + address_score) / 4.0
    return {
        "name_match_score": round(name_score, 3),
        "dob_match_score": round(dob_score, 3),
        "document_number_consistency": round(doc_num_score, 3),
        "address_match_score": round(address_score, 3),
        "doc_field_consistency": round(overall, 3),
    }
