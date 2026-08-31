"""FaceModelAdapter interface (section 11). Nothing in the rest of the
codebase imports InsightFace or SyntheticFeatureAdapter directly -- everyone
goes through `get_face_adapter()` so the model backing "face verification"
can be swapped without touching callers."""
from __future__ import annotations

from abc import ABC, abstractmethod


class FaceModelAdapter(ABC):
    name: str = "base"

    @abstractmethod
    def compute_signal(self, attack_type: str | None = None, difficulty: float = 0.5, seed: int | None = None) -> dict:
        """Return {face_similarity_score, liveness_score, deepfake_probability,
        face_quality_score} for one synthetic verification event."""
        raise NotImplementedError

    def is_available(self) -> bool:
        return True


def get_face_adapter(prefer: str = "synthetic") -> FaceModelAdapter:
    """Factory with graceful fallback (section 11 architecture:
    FaceModelAdapter -> InsightFaceAdapter -> SyntheticFeatureAdapter).
    Default is ALWAYS SyntheticFeatureAdapter unless the caller explicitly
    asks for 'insightface' AND it is installed."""
    if prefer == "insightface":
        try:
            from backend.biometric.optional_insightface import InsightFaceAdapter
            adapter = InsightFaceAdapter()
            if adapter.is_available():
                return adapter
        except Exception:
            pass  # fall through to synthetic

    from backend.biometric.synthetic import SyntheticFeatureAdapter
    return SyntheticFeatureAdapter()
