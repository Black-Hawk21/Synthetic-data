from __future__ import annotations

import numpy as np

from backend.red_team.base import AttackStrategy
from backend.red_team.registry import register_attack
from backend.red_team.utils import blend, shared_pool


@register_attack
class DuplicateIdentityAttack(AttackStrategy):
    """The same underlying identity (name + DOB + document number) is
    submitted more than once under near-identical applications."""

    attack_type = "DUPLICATE_IDENTITY"
    features_affected = ["document_number_consistency", "previous_application_count", "identity_reuse_count"]
    summary = "The same document number / name+DOB combination is submitted across multiple 'different' applications."

    def mutate(self, df, rng, difficulty):
        n = len(df)
        cluster_range = (2, 3) if difficulty > 0.6 else (3, 6)
        doc_pool = shared_pool(rng, n, cluster_range, "dupdoc")
        df["document_number"] = doc_pool
        df["previous_application_count"] = np.clip(
            df["previous_application_count"] + rng.integers(1, int(blend(6, 2, difficulty)) + 1, size=n), 0, None
        )
        df["document_number_consistency"] = np.clip(
            df["document_number_consistency"] * blend(0.5, 0.95, difficulty), 0, 1
        )
        return df


@register_attack
class IdentityReuseAttack(AttackStrategy):
    """Stolen/leaked contact credentials (phone + email pair) reused across
    otherwise-distinct-looking synthetic applicants."""

    attack_type = "IDENTITY_REUSE"
    features_affected = ["phone_reuse_count", "email_reuse_count", "identity_age_days"]
    summary = "A stolen phone+email credential pair is reused to onboard multiple distinct-looking synthetic applicants."

    def mutate(self, df, rng, difficulty):
        n = len(df)
        cluster_range = (2, 4) if difficulty > 0.6 else (3, 9)
        phone_pool = shared_pool(rng, n, cluster_range, "reuseph")
        email_pool = shared_pool(rng, n, cluster_range, "reuseem")
        share_frac = blend(0.85, 0.35, difficulty)
        mask = rng.random(n) < share_frac
        df.loc[mask, "phone"] = phone_pool[mask]
        df.loc[mask, "email"] = email_pool[mask]
        df["identity_age_days"] = np.clip(
            df["identity_age_days"] * blend(0.3, 0.8, difficulty), 0, None
        ).astype(int)
        return df
