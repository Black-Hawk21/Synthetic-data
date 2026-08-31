from __future__ import annotations

from backend.red_team.base import AttackStrategy
from backend.red_team.registry import register_attack
from backend.red_team.utils import perturb_toward


@register_attack
class FaceMismatchAttack(AttackStrategy):
    """The submitted selfie is a real, live person -- just not the person on
    the document (impersonation / photo swap)."""

    attack_type = "FACE_DOCUMENT_MISMATCH"
    features_affected = ["face_similarity_score", "liveness_score"]
    summary = "Selfie passes liveness (it's a real live person) but does not match the identity on the document."

    def mutate(self, df, rng, difficulty):
        df["face_similarity_score"] = perturb_toward(rng, df["face_similarity_score"], 0.10, 0.48, difficulty)
        df["liveness_score"] = perturb_toward(rng, df["liveness_score"], 0.85, 0.90, difficulty)  # stays high
        return df


@register_attack
class AIGeneratedFaceAttack(AttackStrategy):
    """The 'selfie' is an AI-generated face (e.g. GAN/diffusion output,
    thispersondoesnotexist-style) rather than a photo of a real person."""

    attack_type = "AI_GENERATED_FACE"
    features_affected = ["deepfake_probability", "face_quality_score", "liveness_score"]
    summary = "Selfie is a fully AI-generated (non-existent-person) face rather than a photo of a real applicant."

    def mutate(self, df, rng, difficulty):
        df["deepfake_probability"] = perturb_toward(rng, df["deepfake_probability"], 0.90, 0.55, difficulty)
        # AI faces often look *unnaturally* high quality/symmetric at low difficulty,
        # which a naive "high quality = good" heuristic would miss -- that's the point.
        df["face_quality_score"] = perturb_toward(rng, df["face_quality_score"], 0.93, 0.80, difficulty)
        df["liveness_score"] = perturb_toward(rng, df["liveness_score"], 0.35, 0.70, difficulty)
        return df


@register_attack
class DeepfakeLikeSelfieAttack(AttackStrategy):
    """A real base photo run through a face-swap/deepfake pipeline to defeat
    face-match without controlling the underlying identity."""

    attack_type = "DEEPFAKE_LIKE_SELFIE"
    features_affected = ["face_similarity_score", "liveness_score", "deepfake_probability"]
    summary = "Face-swapped/deepfake video-selfie used to defeat face-match against a stolen document photo."

    def mutate(self, df, rng, difficulty):
        # Similarity can look decent (that's the point of a face swap) but variable
        df["face_similarity_score"] = perturb_toward(rng, df["face_similarity_score"], 0.55, 0.75, difficulty, noise=0.15)
        df["liveness_score"] = perturb_toward(rng, df["liveness_score"], 0.30, 0.62, difficulty)
        df["deepfake_probability"] = perturb_toward(rng, df["deepfake_probability"], 0.80, 0.50, difficulty)
        return df
