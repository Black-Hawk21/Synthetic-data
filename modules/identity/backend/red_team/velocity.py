from __future__ import annotations

import numpy as np

from backend.red_team.base import AttackStrategy
from backend.red_team.registry import register_attack
from backend.red_team.utils import blend, perturb_toward, tight_timestamps


@register_attack
class RapidMultiAccountCreationAttack(AttackStrategy):
    """A burst of applications from related identities within a very short
    time window -- classic account-farming velocity signal."""

    attack_type = "RAPID_MULTI_ACCOUNT_CREATION"
    features_affected = ["application_velocity", "created_at", "previous_application_count"]
    summary = "A burst of related applications is submitted within an unusually short time window."

    def mutate(self, df, rng, difficulty):
        n = len(df)
        window_minutes = blend(12, 240, difficulty)
        df["created_at"] = tight_timestamps(rng, n, window_minutes)
        df["application_velocity"] = np.clip(
            rng.normal(blend(4.5, 1.2, difficulty), 0.8, size=n), 0.05, None
        )
        df["previous_application_count"] = rng.poisson(blend(3, 0.6, difficulty), size=n)
        return df


@register_attack
class BotLikeOnboardingAttack(AttackStrategy):
    """Form-filling behavior is scripted/automated rather than a human
    typing: unnaturally fast and unnaturally *consistent* (low variance)."""

    attack_type = "BOT_LIKE_ONBOARDING"
    features_affected = ["automation_score", "typing_variance", "mouse_entropy", "form_completion_time_sec"]
    summary = "Form-filling telemetry (timing, mouse movement, corrections) looks scripted rather than human."

    def mutate(self, df, rng, difficulty):
        n = len(df)
        df["automation_score"] = perturb_toward(rng, df["automation_score"], 0.92, 0.55, difficulty)
        df["typing_variance"] = perturb_toward(rng, df["typing_variance"], 0.05, 0.30, difficulty)
        df["mouse_entropy"] = perturb_toward(rng, df["mouse_entropy"], 0.08, 0.35, difficulty)
        df["form_completion_time_sec"] = np.clip(
            df["form_completion_time_sec"] * blend(0.15, 0.55, difficulty), 3, None
        )
        df["num_corrections"] = rng.poisson(blend(0.1, 1.0, difficulty), size=n)
        return df
