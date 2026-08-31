from __future__ import annotations

import numpy as np
import pandas as pd

from backend.red_team.base import AttackStrategy
from backend.red_team.registry import register_attack
from backend.red_team.utils import blend, perturb_toward, shrink_days_toward


@register_attack
class SyntheticIdentityAttack(AttackStrategy):
    """Fabricated identity assembled from real-looking but non-existent
    attribute combinations (Frankenstein identity). Individually each field
    passes a plausibility check; the tell is that identity/phone/email/
    address all came into existence almost simultaneously."""

    attack_type = "SYNTHETIC_IDENTITY"
    features_affected = [
        "identity_age_days", "phone_age_days", "email_age_days",
        "address_age_days", "previous_application_count",
    ]
    summary = "Individually plausible identity attributes fabricated together; tell-tale sign is near-simultaneous 'birth' of identity, phone, email and address."

    def mutate(self, df, rng, difficulty):
        n = len(df)
        # Easy: everything born in the last few days. Hard: only some
        # signals are new, spread over a few months (subtler correlation).
        df["identity_age_days"] = shrink_days_toward(rng, n, 3, 120, difficulty)
        correlated_noise = rng.normal(0, blend(2, 45, difficulty), size=n)
        df["phone_age_days"] = np.clip(df["identity_age_days"] + correlated_noise, 0, None).astype(int)
        df["email_age_days"] = np.clip(df["identity_age_days"] + rng.normal(0, blend(2, 40, difficulty), size=n), 0, None).astype(int)
        df["address_age_days"] = np.clip(df["identity_age_days"] * rng.uniform(0.8, blend(1.2, 3.5, difficulty), size=n), 0, None).astype(int)
        df["previous_application_count"] = rng.poisson(blend(0.05, 0.5, difficulty), size=n)
        return df


@register_attack
class IdentityAttributeInconsistencyAttack(AttackStrategy):
    """Application data and submitted document disagree on core identity
    fields (name spelling, DOB, ID number) beyond normal typo tolerance."""

    attack_type = "IDENTITY_ATTRIBUTE_INCONSISTENCY"
    features_affected = ["name_match_score", "dob_match_score", "document_number_consistency"]
    summary = "Applicant-submitted identity attributes disagree with the document's attributes beyond normal typo tolerance."

    def mutate(self, df, rng, difficulty):
        n = len(df)
        df["name_match_score"] = perturb_toward(rng, df["name_match_score"], 0.15, 0.72, difficulty)
        df["dob_match_score"] = perturb_toward(rng, df["dob_match_score"], 0.20, 0.70, difficulty)
        df["document_number_consistency"] = perturb_toward(rng, df["document_number_consistency"], 0.25, 0.75, difficulty)
        return df


@register_attack
class MultiSignalSyntheticIdentityAttack(AttackStrategy):
    """The hardest catch-all pattern: every individual signal is only
    mildly off, but identity, device and graph signals are mildly off
    *together* -- requires multi-signal reasoning to catch."""

    attack_type = "MULTI_SIGNAL_SYNTHETIC_IDENTITY"
    features_affected = [
        "identity_age_days", "device_reuse_count", "document_authenticity_score",
        "face_similarity_score", "application_velocity",
    ]
    summary = "Combination of many individually-mild anomalies across identity, document, biometric, device and behavior signals -- no single feature is conclusive."

    def mutate(self, df, rng, difficulty):
        from backend.red_team.utils import shared_pool
        n = len(df)
        df["identity_age_days"] = shrink_days_toward(rng, n, 15, 180, difficulty)
        df["document_authenticity_score"] = perturb_toward(rng, df["document_authenticity_score"], 0.45, 0.80, difficulty)
        df["face_similarity_score"] = perturb_toward(rng, df["face_similarity_score"], 0.45, 0.82, difficulty)
        df["application_velocity"] = np.clip(df["application_velocity"] + blend(0.6, 0.15, difficulty), 0, None)
        # A minority share device infra -- mild, not overwhelming, at high difficulty
        share_frac = blend(0.5, 0.15, difficulty)
        mask = rng.random(n) < share_frac
        if mask.sum() > 0:
            df.loc[mask, "device_id"] = shared_pool(rng, int(mask.sum()), (3, 8), "msyn_dev")
        return df
