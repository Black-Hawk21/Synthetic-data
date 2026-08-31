"""OPTIONAL real-image biometric heuristics for the Live KYC Form (manual
onboarding path), using OpenCV's bundled Haar cascade face detector -- no
model download required, works entirely offline/CPU.

Honesty note: this is NOT a production face-recognition or liveness-
detection model. It is a set of transparent, explainable heuristics (face
presence, image sharpness, color/texture variance, coarse histogram
similarity) that give a REAL signal computed from the actual uploaded
photo instead of a sampled distribution. For production-grade face
matching, wire in `backend/biometric/optional_insightface.py` instead (see
its docstring) -- this adapter's `compare_faces` is intentionally simple so
it runs with zero setup.

If `opencv-python-headless` is not installed, `is_available()` returns
False and callers fall back to `SyntheticFeatureAdapter` automatically, so
nothing breaks.
"""
from __future__ import annotations

import io
import logging

import numpy as np
from PIL import Image

from backend.biometric.adapter import FaceModelAdapter

logger = logging.getLogger(__name__)

try:
    import cv2  # type: ignore
    _CV2_AVAILABLE = True
except Exception:
    _CV2_AVAILABLE = False


def _load_gray_cv(image_bytes: bytes, max_side: int = 700) -> "np.ndarray | None":
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        img.thumbnail((max_side, max_side))
        return np.asarray(img)
    except Exception:
        return None


class OpenCVHeuristicAdapter(FaceModelAdapter):
    name = "opencv_heuristic"

    def __init__(self):
        self._cascade = None
        if _CV2_AVAILABLE:
            try:
                path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self._cascade = cv2.CascadeClassifier(path)
                if self._cascade.empty():
                    self._cascade = None
            except Exception as e:  # pragma: no cover
                logger.warning("OpenCV cascade failed to load: %s", e)
                self._cascade = None

    def is_available(self) -> bool:
        return _CV2_AVAILABLE and self._cascade is not None

    def compute_signal(self, attack_type: str | None = None, difficulty: float = 0.5, seed: int | None = None) -> dict:
        # Bulk/simulated calls still go through the synthetic profile table
        # -- this adapter's value is analyze_selfie()/compare_faces() below,
        # which need actual image bytes.
        from backend.biometric.synthetic import generate_synthetic_face_signal
        return generate_synthetic_face_signal(attack_type, difficulty, seed)

    def detect_face(self, image_bytes: bytes) -> dict:
        if not self.is_available():
            return {"found": False, "reason": "opencv_unavailable"}
        gray = _load_gray_cv(image_bytes)
        if gray is None:
            return {"found": False, "reason": "could_not_decode_image"}
        faces = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            return {"found": False, "reason": "no_face_detected"}
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # largest face
        return {"found": True, "bbox": [int(x), int(y), int(w), int(h)], "image_shape": gray.shape}

    def analyze_selfie(self, image_bytes: bytes) -> dict:
        """Returns face_quality_score, liveness_score, deepfake_probability
        (the last stays a low, honestly-labeled placeholder -- see module
        docstring) computed from the real uploaded/captured photo."""
        detection = self.detect_face(image_bytes)
        if not detection["found"]:
            return {
                "face_detected": False, "face_quality_score": 0.15,
                "liveness_score": 0.2, "deepfake_probability": 0.3,
                "note": detection.get("reason"),
            }

        gray = _load_gray_cv(image_bytes)
        x, y, w, h = detection["bbox"]
        face_crop = gray[y:y + h, x:x + w].astype(np.float32)

        # Sharpness of the FACE region specifically (blur/liveness proxy).
        sharpness = float(cv2.Laplacian(face_crop, cv2.CV_32F).var()) if face_crop.size else 0.0
        quality = float(np.clip(np.log1p(sharpness) / 7.5, 0, 1))

        # Texture/contrast variance -- printed photos held up to a camera
        # tend to be flatter (lower local contrast) than a genuine live
        # capture with real skin texture and lighting variation.
        contrast = float(face_crop.std())
        liveness = float(np.clip(0.25 + 0.65 * quality + 0.10 * np.clip(contrast / 55, 0, 1), 0, 1))

        return {
            "face_detected": True,
            "face_quality_score": round(quality, 3),
            "liveness_score": round(liveness, 3),
            "deepfake_probability": round(float(np.clip(0.08 + 0.15 * (1 - quality), 0, 1)), 3),
            "face_bbox": detection["bbox"],
        }

    def compare_faces(self, document_image_bytes: bytes, selfie_image_bytes: bytes) -> dict:
        """Coarse face_similarity_score proxy via grayscale histogram
        correlation of the two detected face crops. This is NOT real face
        recognition (no embeddings, no identity verification guarantee) --
        it's a cheap, transparent, zero-download signal so the demo is
        driven by the actual two photos instead of a sampled number."""
        doc = self.detect_face(document_image_bytes)
        selfie = self.detect_face(selfie_image_bytes)
        if not doc["found"] or not selfie["found"]:
            return {"face_similarity_score": 0.35, "note": "face_not_detected_in_one_or_both_images"}

        gray_doc = _load_gray_cv(document_image_bytes)
        gray_selfie = _load_gray_cv(selfie_image_bytes)
        dx, dy, dw, dh = doc["bbox"]
        sx, sy, sw, sh = selfie["bbox"]
        crop_doc = cv2.resize(gray_doc[dy:dy + dh, dx:dx + dw], (128, 128))
        crop_selfie = cv2.resize(gray_selfie[sy:sy + sh, sx:sx + sw], (128, 128))

        hist_doc = cv2.calcHist([crop_doc], [0], None, [64], [0, 256])
        hist_selfie = cv2.calcHist([crop_selfie], [0], None, [64], [0, 256])
        cv2.normalize(hist_doc, hist_doc)
        cv2.normalize(hist_selfie, hist_selfie)
        correlation = float(cv2.compareHist(hist_doc, hist_selfie, cv2.HISTCMP_CORREL))
        similarity = float(np.clip((correlation + 1) / 2, 0, 1))
        return {"face_similarity_score": round(similarity, 3)}


_adapter_singleton: OpenCVHeuristicAdapter | None = None


def get_opencv_adapter() -> OpenCVHeuristicAdapter:
    global _adapter_singleton
    if _adapter_singleton is None:
        _adapter_singleton = OpenCVHeuristicAdapter()
    return _adapter_singleton
