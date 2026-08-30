import json

import anthropic

from aegis import config


def _extract_json(text: str) -> str:
    """Claude often wraps JSON in ```json ... ``` fences, appends trailing prose after
    the object, or (for prompts that ask for reasoning before the final answer, e.g.
    vision inspection) puts prose BEFORE the object too. Try every '{' as a possible
    start and keep whichever successful parse leaves the LEAST text unconsumed --
    this correctly prefers the outermost/complete object over a nested fragment (a
    finding inside the "findings" list is itself valid JSON on its own, so naively
    taking the first -- or last -- brace that merely parses successfully can return
    just that fragment instead of the real top-level result)."""
    best_obj = None
    best_leftover = None
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, end = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        leftover = len(text) - (start + end)
        if best_leftover is None or leftover < best_leftover:
            best_obj, best_leftover = obj, leftover
    if best_obj is None:
        return text
    return json.dumps(best_obj)

_SCHEMA_DESCRIPTIONS = {
    "sanitizer": (
        'Respond with ONLY valid JSON matching: '
        '{"injection_detected": bool, "manipulation_risk_score": float 0-1, '
        '"reason": string (one sentence explaining why), '
        '"flagged_phrases": [string], "cleaned_transcript": string}'
    ),
    "vision_artifact": (
        'Respond with ONLY valid JSON matching: {"artifact_score": float 0-1 (HIGHER = MORE likely '
        'synthetic/manipulated; 0.0 = ordinary unedited photo with no forensic red flags, 1.0 = '
        'strong signs of AI generation or digital editing), '
        '"findings": [{"type": string, "confidence": float, "description": string}]}'
    ),
    "vision_identity": (
        'Respond with ONLY valid JSON matching: {"detail_consistency_score": float 0-1 (HIGHER = '
        'MORE consistent -- more likely the exact same physical unit in both photos, checked '
        'feature by feature), "findings": [{"type": string, "confidence": float, "description": '
        'string}] (every description MUST name the specific feature/surface it is about, e.g. '
        'start with "Left ear cup: ..." or "Logo placement: ...", not a generic category label)}'
    ),
    "vision_comparison": (
        'Work through the numbered steps in prose first, then at the very end of your '
        'response, on its own, output ONLY the final JSON object with no commentary after it, '
        'matching: '
        '{"angle_consistency_score": float 0-1 (HIGHER = MORE consistent/genuine cross-angle '
        'damage plausibility; near-identical noise between two supposedly independent shots is '
        'itself suspicious and should lower this score), "semantic_match": bool, '
        '"findings": [{"type": string, "confidence": float, "description": string}]}'
    ),
    "vision_localization": (
        'Respond with ONLY valid JSON matching: {"image_1": {"landmark_a": {"name": string, '
        '"x": float 0-1, "y": float 0-1}, "landmark_b": {"name": string, "x": float 0-1, "y": '
        'float 0-1}, "damage_bbox": {"x_min": float 0-1, "y_min": float 0-1, "x_max": float 0-1, '
        '"y_max": float 0-1}}, "image_2": {same structure as image_1}} -- x/y are fractions of '
        'that photo\'s own width/height, (0,0) = top-left. landmark_a and landmark_b MUST refer '
        'to the same two physical features in both images, named identically in both (e.g. both '
        'called "logo" and "hinge"), since they will be matched by name programmatically. No '
        'findings or verdict -- this call is measurement only.'
    ),
    "vision_shape_match": (
        'Respond with ONLY valid JSON matching: {"shape_match_score": float 0-1 (HIGHER = MORE '
        'consistent -- the two cropped damage regions plausibly show the same physical damage), '
        '"findings": [{"type": string, "confidence": float, "description": string}]}'
    ),
    "supervisor_explanation": (
        'Respond with ONLY valid JSON matching: {"explanation": string}'
    ),
}

# Per-schema-hint token budget, sized to how much reasoning each focused call actually
# needs -- the identity check's exhaustive feature-by-feature scan and the comparison
# check's explicit describe-both-then-compare reasoning both run long, while the
# single-image artifact check needs much less.
_MAX_TOKENS_BY_HINT = {
    "vision_artifact": 512,
    "vision_identity": 3072,
    "vision_comparison": 2048,
    "vision_localization": 768,
    "vision_shape_match": 768,
}
_DEFAULT_VISION_MAX_TOKENS = 1024


class AnthropicProvider:
    """Implements TextProvider and VisionProvider using the Anthropic API."""

    def __init__(self, api_key: str | None = None) -> None:
        # api_key lets a caller (the Streamlit sidebar's bring-your-own-key field)
        # override the .env-configured key for just this instance, without touching
        # global config or writing anything to disk.
        self._client = anthropic.Anthropic(api_key=api_key or config.ANTHROPIC_API_KEY)

    def generate_text(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=config.ANTHROPIC_TEXT_MODEL,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if block.type == "text")

    def generate_json(self, system: str, user: str, schema_hint: str) -> str:
        instruction = _SCHEMA_DESCRIPTIONS.get(schema_hint, "Respond with ONLY valid JSON.")
        response = self._client.messages.create(
            model=config.ANTHROPIC_TEXT_MODEL,
            max_tokens=1024,
            system=f"{system}\n\n{instruction}",
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return _extract_json(text)

    def inspect_images(self, images_b64: list[str], prompt: str, schema_hint: str = "vision_comparison") -> str:
        instruction = _SCHEMA_DESCRIPTIONS.get(schema_hint, "Respond with ONLY valid JSON.")
        content = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": img_b64},
            }
            for img_b64 in images_b64
        ]
        content.append({"type": "text", "text": f"{prompt}\n\n{instruction}"})
        response = self._client.messages.create(
            model=config.ANTHROPIC_VISION_MODEL,
            max_tokens=_MAX_TOKENS_BY_HINT.get(schema_hint, _DEFAULT_VISION_MAX_TOKENS),
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return _extract_json(text)
