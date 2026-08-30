from __future__ import annotations

from backend.red_team.base import AttackStrategy
from backend.red_team.registry import register_attack
from backend.red_team.utils import perturb_toward


@register_attack
class LivenessSpoofAttack(AttackStrategy):
    """Presentation attack against the liveness check: printed photo, replay
    video, or a mask held up to the camera. Face similarity can remain high
    (correct photo) while liveness fails."""

    attack_type = "LIVENESS_SPOOF"
    features_affected = ["liveness_score", "face_similarity_score", "face_quality_score"]
    summary = "Presentation attack (printed photo / screen replay / mask) defeats the liveness check while the face photo itself matches."

    def mutate(self, df, rng, difficulty):
        df["liveness_score"] = perturb_toward(rng, df["liveness_score"], 0.08, 0.45, difficulty)
        df["face_similarity_score"] = perturb_toward(rng, df["face_similarity_score"], 0.80, 0.88, difficulty)
        df["face_quality_score"] = perturb_toward(rng, df["face_quality_score"], 0.40, 0.68, difficulty)
        return df
