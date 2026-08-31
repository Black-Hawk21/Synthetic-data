import base64

import requests

from aegis import config

_ENDPOINT = "https://api.stability.ai/v2beta/stable-image/generate/core"


class StabilityProvider:
    """Optional paid image provider (~$0.02-0.08/image). Implements ImageProvider.

    Supports true image-to-image conditioning via the "image-to-image" mode,
    unlike the free Pollinations fallback -- use this if higher-fidelity
    img2img consistency is worth the per-image cost for the demo.
    """

    def generate_image(self, prompt: str, reference_b64: str | None = None) -> str:
        headers = {
            "Authorization": f"Bearer {config.STABILITY_API_KEY}",
            "Accept": "image/*",
        }
        data = {"prompt": prompt, "output_format": "png"}
        files = {}

        if reference_b64:
            data["mode"] = "image-to-image"
            data["strength"] = 0.55
            files["image"] = ("reference.png", base64.b64decode(reference_b64), "image/png")
        else:
            files["none"] = ""

        response = requests.post(_ENDPOINT, headers=headers, data=data, files=files, timeout=60)
        response.raise_for_status()
        return base64.b64encode(response.content).decode()
