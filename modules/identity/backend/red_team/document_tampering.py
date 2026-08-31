from __future__ import annotations

from backend.red_team.base import AttackStrategy
from backend.red_team.registry import register_attack
from backend.red_team.utils import blend, perturb_toward


@register_attack
class DocumentTamperingAttack(AttackStrategy):
    """The document image itself has been visually edited (splicing,
    cloning, re-compression artifacts) -- low authenticity, high tamper
    score."""

    attack_type = "DOCUMENT_TAMPERING"
    features_affected = ["document_tamper_score", "document_authenticity_score", "ocr_confidence"]
    summary = "Document image shows visual tampering artifacts (splicing, re-compression, cloned regions)."

    def mutate(self, df, rng, difficulty):
        df["document_tamper_score"] = perturb_toward(rng, df["document_tamper_score"], 0.88, 0.40, difficulty)
        df["document_authenticity_score"] = perturb_toward(rng, df["document_authenticity_score"], 0.10, 0.55, difficulty)
        df["ocr_confidence"] = perturb_toward(rng, df["ocr_confidence"], 0.35, 0.75, difficulty)
        return df


@register_attack
class DocumentFieldManipulationAttack(AttackStrategy):
    """Individual text fields on an otherwise-genuine document template
    were edited (name/DOB/address overwritten) without visible image
    tampering -- authenticity stays high, but field consistency breaks."""

    attack_type = "DOCUMENT_FIELD_MANIPULATION"
    features_affected = ["doc_field_consistency", "name_match_score", "dob_match_score"]
    summary = "Specific text fields (name, DOB, address) on the document were edited while the image itself looks clean."

    def mutate(self, df, rng, difficulty):
        df["doc_field_consistency"] = perturb_toward(rng, df["doc_field_consistency"], 0.20, 0.68, difficulty)
        df["name_match_score"] = perturb_toward(rng, df["name_match_score"], 0.25, 0.70, difficulty)
        df["dob_match_score"] = perturb_toward(rng, df["dob_match_score"], 0.30, 0.72, difficulty)
        # authenticity/tamper stay near-legit -- that's what makes this attack distinct
        df["document_authenticity_score"] = perturb_toward(rng, df["document_authenticity_score"], 0.55, 0.85, difficulty)
        return df
