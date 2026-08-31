"""OPTIONAL real-embedding adapter. Only activates if the `insightface` +
`onnxruntime` packages are installed (they are NOT in requirements.txt on
purpose -- see requirements-optional.txt). This module must be importable
even when insightface is missing; is_available() reports the truth and the
factory in adapter.py falls back to SyntheticFeatureAdapter automatically.

To use real embeddings: pip install insightface onnxruntime, download the
buffalo_l model pack, and pass images through compute_signal(image_a,
image_b). No real user faces are required -- this can run against
publicly-available research face datasets or locally generated synthetic
faces (e.g. StyleGAN samples) for demo purposes only.
"""
from __future__ import annotations

import logging

import numpy as np

from backend.biometric.adapter import FaceModelAdapter

logger = logging.getLogger(__name__)

try:
    import insightface  # type: ignore
    _INSIGHTFACE_AVAILABLE = True
except Exception:
    _INSIGHTFACE_AVAILABLE = False


class InsightFaceAdapter(FaceModelAdapter):
    name = "insightface"

    def __init__(self):
        self._model = None
        if _INSIGHTFACE_AVAILABLE:
            try:
                from insightface.app import FaceAnalysis  # type: ignore
                self._model = FaceAnalysis(name="buffalo_l")
                self._model.prepare(ctx_id=-1)  # CPU
            except Exception as e:  # pragma: no cover
                logger.warning("InsightFace present but failed to initialize: %s", e)
                self._model = None

    def is_available(self) -> bool:
        return _INSIGHTFACE_AVAILABLE and self._model is not None

    def compute_similarity(self, image_a: np.ndarray, image_b: np.ndarray) -> float:
        if not self.is_available():
            raise RuntimeError("InsightFace not available")
        faces_a = self._model.get(image_a)
        faces_b = self._model.get(image_b)
        if not faces_a or not faces_b:
            return 0.0
        emb_a = faces_a[0].normed_embedding
        emb_b = faces_b[0].normed_embedding
        return float(np.clip((np.dot(emb_a, emb_b) + 1) / 2, 0, 1))

    def compute_signal(self, attack_type: str | None = None, difficulty: float = 0.5, seed: int | None = None) -> dict:
        # Real embeddings need actual images (compute_similarity above);
        # this method exists to satisfy the FaceModelAdapter interface for
        # bulk/simulated calls and defers to the synthetic profile so the
        # rest of the pipeline keeps working uniformly.
        from backend.biometric.synthetic import generate_synthetic_face_signal
        return generate_synthetic_face_signal(attack_type, difficulty, seed)
