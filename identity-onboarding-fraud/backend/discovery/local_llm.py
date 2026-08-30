"""OPTIONAL local LLM client (Ollama). Used ONLY for attack ideation /
explanation text -- never for the numerical fraud-detection model itself
(section 6). If Ollama is not running, every call here fails fast and
attack_discovery.py falls back to discovery/fallback.py."""
from __future__ import annotations

import json
import logging

import httpx

from backend.config import settings

logger = logging.getLogger(__name__)

DISCOVERY_PROMPT_TEMPLATE = """You are a payments fraud red-team analyst. Given these weak spots in a \
fraud detector (attack_type -> recall), propose {n} NEW attack hypotheses as a \
JSON array. Each item must have exactly these keys: attack_type (reuse one of \
the existing types below or propose a MULTI_SIGNAL_SYNTHETIC_IDENTITY-style \
combination), reason, target_weakness, features_to_manipulate (list of \
strings), difficulty (0-1 float, higher = harder to detect).

Existing attack types: {attack_types}
Weak spots (attack_type: recall): {weak_spots}

Respond with ONLY the JSON array, no prose.
"""


def is_ollama_available(timeout: float = 1.5) -> bool:
    try:
        r = httpx.get(f"{settings.ollama_host}/api/tags", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def generate_attack_hypotheses_via_llm(weak_spots: dict, attack_types: list[str], n: int = 5, timeout: float = 20.0) -> list[dict] | None:
    """Returns a list of hypothesis dicts, or None if Ollama is unavailable
    / the response could not be parsed as JSON."""
    if not is_ollama_available():
        return None

    prompt = DISCOVERY_PROMPT_TEMPLATE.format(n=n, attack_types=attack_types, weak_spots=weak_spots)
    try:
        r = httpx.post(
            f"{settings.ollama_host}/api/generate",
            json={"model": settings.ollama_model, "prompt": prompt, "stream": False, "format": "json"},
            timeout=timeout,
        )
        r.raise_for_status()
        raw = r.json().get("response", "")
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            parsed = parsed.get("hypotheses") or parsed.get("attacks") or [parsed]
        if isinstance(parsed, list):
            return parsed
        return None
    except Exception as e:
        logger.info("Ollama attack discovery unavailable/failed (%s); using rule-based fallback.", e)
        return None
