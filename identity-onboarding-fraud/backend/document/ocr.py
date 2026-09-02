"""OCR extraction. Preferred: PaddleOCR. Fallback: Tesseract. Fallback of
fallback: a deterministic simulated-OCR pass so the pipeline NEVER hard-
fails just because no OCR engine is installed (section 10/29)."""
from __future__ import annotations

import logging
import re
import shutil

from PIL import Image

logger = logging.getLogger(__name__)

_TESSERACT_BIN = shutil.which("tesseract")

try:
    import pytesseract  # type: ignore
    _PYTESSERACT_AVAILABLE = _TESSERACT_BIN is not None
except Exception:
    _PYTESSERACT_AVAILABLE = False

try:
    from paddleocr import PaddleOCR  # type: ignore
    _PADDLE_AVAILABLE = True
except Exception:
    _PADDLE_AVAILABLE = False


def ocr_engine_status() -> dict:
    return {
        "paddleocr_available": _PADDLE_AVAILABLE,
        "pytesseract_available": _PYTESSERACT_AVAILABLE,
        "active_engine": "paddleocr" if _PADDLE_AVAILABLE else ("tesseract" if _PYTESSERACT_AVAILABLE else "simulated_fallback"),
    }


_FIELD_PATTERNS = {
    "document_type": r"Document Type:\s*(.+)",
    "name": r"Full Name:\s*(.+)",
    "date_of_birth": r"Date of Birth:\s*(.+)",
    "document_number": r"Document No\.:\s*(.+)",
    "address": r"Address:\s*(.+)",
}


def _extract_fields_from_text(text: str) -> dict:
    fields = {}
    for key, pattern in _FIELD_PATTERNS.items():
        m = re.search(pattern, text)
        fields[key] = m.group(1).strip() if m else None
    return fields


def run_ocr(image: Image.Image) -> dict:
    """Returns {raw_text, fields, confidence, engine}."""
    if _PADDLE_AVAILABLE:
        try:
            import numpy as np
            ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            result = ocr.ocr(np.array(image), cls=True)
            lines = [line[1][0] for block in result for line in block] if result else []
            text = "\n".join(lines)
            confs = [line[1][1] for block in result for line in block] if result else []
            confidence = sum(confs) / len(confs) if confs else 0.5
            return {"raw_text": text, "fields": _extract_fields_from_text(text), "confidence": confidence, "engine": "paddleocr"}
        except Exception as e:
            logger.warning("PaddleOCR failed, falling back: %s", e)

    if _PYTESSERACT_AVAILABLE:
        try:
            text = pytesseract.image_to_string(image)
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            confs = [int(c) for c in data.get("conf", []) if str(c).isdigit() and int(c) >= 0]
            confidence = (sum(confs) / len(confs) / 100.0) if confs else 0.5
            return {"raw_text": text, "fields": _extract_fields_from_text(text), "confidence": confidence, "engine": "tesseract"}
        except Exception as e:
            logger.warning("Tesseract failed, falling back: %s", e)

    return _simulated_ocr_fallback()


def _simulated_ocr_fallback() -> dict:
    """No OCR engine installed -- returns a clearly-labeled simulated
    result so downstream consistency-scoring code has a stable shape to
    work with even in a minimal environment."""
    return {
        "raw_text": None,
        "fields": {k: None for k in _FIELD_PATTERNS},
        "confidence": 0.0,
        "engine": "simulated_fallback",
        "note": "No OCR engine installed (paddleocr/tesseract). Install requirements-optional.txt for real OCR.",
    }
