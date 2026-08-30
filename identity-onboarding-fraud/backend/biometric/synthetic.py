"""Default biometric adapter: NEVER requires a real face. Produces
attack-conditional numerical biometric signals (section 12).

This is what the bulk data generator uses for all 10k-500k rows, and what
the Onboarding Simulator UI calls for a single "RUN VERIFICATION" demo.
"""
from __future__ import annotations

import numpy as np

from backend.biometric.adapter import FaceModelAdapter

# Distribution centers per section 12's worked examples. (easy_target, hard_target)
# for each of the four biometric signals, keyed by attack type. "difficulty"
# interpolates between them exactly like backend.red_team.utils.blend.
_PROFILES: dict[str, dict[str, tuple[float, float]]] = {
    "NONE": {
        "face_similarity_score": (0.90, 0.90), "liveness_score": (0.92, 0.92),
        "deepfake_probability": (0.04, 0.04), "face_quality_score": (0.85, 0.85),
    },
    "FACE_DOCUMENT_MISMATCH": {
        "face_similarity_score": (0.10, 0.48), "liveness_score": (0.85, 0.90),
        "deepfake_probability": (0.05, 0.10), "face_quality_score": (0.80, 0.85),
    },
    "LIVENESS_SPOOF": {
        "face_similarity_score": (0.80, 0.88), "liveness_score": (0.08, 0.45),
        "deepfake_probability": (0.10, 0.20), "face_quality_score": (0.40, 0.68),
    },
    "AI_GENERATED_FACE": {
        "face_similarity_score": (0.30, 0.60), "liveness_score": (0.35, 0.70),
        "deepfake_probability": (0.90, 0.55), "face_quality_score": (0.93, 0.80),
    },
    "DEEPFAKE_LIKE_SELFIE": {
        "face_similarity_score": (0.55, 0.75), "liveness_score": (0.30, 0.62),
        "deepfake_probability": (0.80, 0.50), "face_quality_score": (0.55, 0.70),
    },
}


def generate_synthetic_face_signal(attack_type: str | None = None, difficulty: float = 0.5, seed: int | None = None) -> dict:
    """Standalone function required by section 12. Returns a dict of the
    four biometric features, distribution conditioned on `attack_type`."""
    rng = np.random.default_rng(seed)
    profile = _PROFILES.get(attack_type or "NONE", _PROFILES["NONE"])
    out = {}
    for feature, (easy, hard) in profile.items():
        target = easy + (hard - easy) * float(np.clip(difficulty, 0, 1))
        noise = 0.05 if attack_type in (None, "NONE") else 0.08
        out[feature] = float(np.clip(rng.normal(target, noise), 0, 1))
    out["face_reuse_count"] = 0
    return out


class SyntheticFeatureAdapter(FaceModelAdapter):
    name = "synthetic"

    def compute_signal(self, attack_type: str | None = None, difficulty: float = 0.5, seed: int | None = None) -> dict:
        return generate_synthetic_face_signal(attack_type, difficulty, seed)

    def is_available(self) -> bool:
        return True
