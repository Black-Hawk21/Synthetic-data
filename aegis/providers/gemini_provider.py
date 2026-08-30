import base64

import requests

from aegis import config

_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_TIMEOUT_SECONDS = 60


class GeminiProvider:
    """Optional image provider using Google Gemini's native image-output models.
    Implements ImageProvider via a direct REST call (no SDK dependency, consistent
    with stability_provider.py). Supports img2img-style conditioning by passing the
    reference image back in as an additional input part alongside the text prompt.

    Gemini's image-generation model ids change fairly often (the same way
    Anthropic's did between this project's Claude 3.5 and Claude 5 generations) --
    if generation starts 404ing, check
    https://ai.google.dev/gemini-api/docs/image-generation for the current model id
    and set GEMINI_IMAGE_MODEL in .env accordingly.
    """

    def generate_image(self, prompt: str, reference_b64: str | None = None) -> str:
        parts = [{"text": prompt}]
        if reference_b64:
            parts.insert(0, {"inlineData": {"mimeType": "image/png", "data": reference_b64}})

        url = _ENDPOINT.format(model=config.GEMINI_IMAGE_MODEL)
        response = requests.post(
            url,
            params={"key": config.GEMINI_API_KEY},
            json={
                "contents": [{"parts": parts}],
                "generationConfig": {"responseModalities": ["IMAGE"]},
            },
            timeout=_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()

        try:
            response_parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError) as error:
            raise ValueError(f"Unexpected Gemini response shape: {data}") from error

        for part in response_parts:
            if "inlineData" in part:
                return part["inlineData"]["data"]
        raise ValueError(f"Gemini response contained no image data: {data}")
