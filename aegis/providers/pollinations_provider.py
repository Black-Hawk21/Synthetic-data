import base64
import hashlib
import io
import time
import urllib.parse

import requests
from PIL import Image

_BASE_URL = "https://image.pollinations.ai/prompt/{prompt}"
# Pollinations is a free, best-effort public service -- it's noticeably slower and
# more failure-prone under repeated automated calls (e.g. Batch Evaluation's ~48
# back-to-back requests) than a paid API. A longer timeout plus a few retries with
# backoff absorbs that instead of failing the whole run on one transient hiccup.
_TIMEOUT_SECONDS = 60
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 3


class PollinationsProvider:
    """Free, no-signup image generation. Implements ImageProvider.

    Pollinations' public API is text-to-image only (no real img2img endpoint),
    so "img2img conditioning" is approximated by deriving a stable seed from
    the reference image's bytes: reusing the same seed across both angle
    prompts nudges the model toward a visually consistent result, which is
    the property we actually need for the two-angle attack simulation.
    """

    def generate_image(self, prompt: str, reference_b64: str | None = None) -> str:
        if reference_b64:
            seed = int(hashlib.sha256(reference_b64.encode()).hexdigest(), 16) % (2**31)
        else:
            seed = int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % (2**31)

        # safe="" is required: quote()'s default leaves "/" unescaped, and prompt text
        # containing a literal "/" (e.g. "damage/defect") then splits into extra URL
        # path segments, 404ing against Pollinations' /prompt/{prompt} route.
        url = _BASE_URL.format(prompt=urllib.parse.quote(prompt, safe=""))
        params = {"width": 512, "height": 512, "seed": seed, "nologo": "true"}

        last_error: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = requests.get(url, params=params, timeout=_TIMEOUT_SECONDS)
                response.raise_for_status()
                # Pollinations actually returns JPEG bytes despite no format being
                # requested; ImageSubmission.format and every downstream consumer
                # (Anthropic's vision API included) assume PNG, so normalize here
                # rather than let a mismatched media_type break the vision call.
                image = Image.open(io.BytesIO(response.content)).convert("RGB")
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode()
            except (requests.exceptions.RequestException, OSError) as error:
                # OSError also catches PIL's UnidentifiedImageError, for the rare case
                # Pollinations returns a 200 with a non-image body under load.
                last_error = error
                if attempt < _MAX_ATTEMPTS - 1:
                    time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
        raise last_error
