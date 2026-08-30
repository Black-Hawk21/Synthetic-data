"""Lightweight, dependency-free (PIL + numpy only) image forensics for a
REAL uploaded document photo -- used by the Live KYC Form (manual
onboarding) path. These are honest heuristics, not a trained tamper-
detection model: they estimate sharpness and block-level noise
consistency, which is what naive splicing/re-compression tends to disturb.
Documented here so nobody mistakes this for a production forensics engine.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image


def _to_gray_array(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    # Cap size for speed -- forensics signal doesn't need full resolution.
    img.thumbnail((900, 900))
    return np.asarray(img, dtype=np.float32)


def _laplacian_variance(gray: np.ndarray) -> float:
    """Classic blur-detection metric: variance of the Laplacian. A crisp,
    in-focus photo has high-variance edges; a blurry/re-scanned one doesn't.
    Implemented with a manual 3x3 kernel convolution (numpy only)."""
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    # Valid-mode convolution via stacked shifted views (fast, no scipy).
    acc = np.zeros((h - 2, w - 2), dtype=np.float32)
    for (dy, dx), k in np.ndenumerate(kernel):
        if k == 0:
            continue
        acc += k * gray[dy:dy + h - 2, dx:dx + w - 2]
    return float(acc.var())


def _block_noise_inconsistency(gray: np.ndarray, block: int = 24) -> float:
    """Splits the image into blocks and measures the coefficient of
    variation of each block's local high-frequency energy. Genuine photos
    have fairly uniform micro-texture; a spliced/edited region often
    stands out as a block (or cluster of blocks) with a very different
    noise signature than its neighbors."""
    h, w = gray.shape
    energies = []
    for y in range(0, h - block, block):
        for x in range(0, w - block, block):
            patch = gray[y:y + block, x:x + block]
            energies.append(float(np.abs(np.diff(patch, axis=0)).mean() + np.abs(np.diff(patch, axis=1)).mean()))
    if len(energies) < 4:
        return 0.0
    energies = np.array(energies)
    mean = energies.mean()
    if mean < 1e-6:
        return 0.0
    cv = energies.std() / mean  # coefficient of variation
    return float(np.clip(cv / 2.5, 0, 1))  # empirically-scaled to ~[0,1]


def analyze_document_image(image_bytes: bytes) -> dict:
    """Returns document_quality, document_tamper_score,
    document_authenticity_score -- all derived from the ACTUAL uploaded
    bytes, not sampled from a distribution."""
    try:
        gray = _to_gray_array(image_bytes)
    except Exception:
        return {
            "document_quality": 0.3, "document_tamper_score": 0.5,
            "document_authenticity_score": 0.5, "error": "could_not_decode_image",
        }

    sharpness = _laplacian_variance(gray)
    # Normalize sharpness to [0,1] with a soft knee -- tuned so a typical
    # phone-camera photo of a printed page lands in the 0.6-0.9 range and a
    # heavily blurred/rescanned one lands low.
    quality = float(np.clip(np.log1p(sharpness) / 8.0, 0, 1))

    tamper = _block_noise_inconsistency(gray)
    resolution_penalty = 0.15 if (gray.shape[0] < 300 or gray.shape[1] < 300) else 0.0
    tamper = float(np.clip(tamper + resolution_penalty, 0, 1))

    authenticity = float(np.clip(0.92 - 0.6 * tamper - 0.15 * (1 - quality), 0, 1))

    return {
        "document_quality": round(quality, 3),
        "document_tamper_score": round(tamper, 3),
        "document_authenticity_score": round(authenticity, 3),
        "sharpness_raw": round(sharpness, 2),
        "resolution": f"{gray.shape[1]}x{gray.shape[0]}",
    }
