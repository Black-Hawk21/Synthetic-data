"""One-off generator for data/legit_samples.json -- genuine (non-fraud) damage
claims used by Batch Evaluation to measure the false-positive rate. Images are
rendered with the mock provider's legit style (no fraud marker); re-run this
any time you want a fresh/larger fixture set.

Usage: python -m scripts.generate_legit_samples
"""

import base64
import io
import json
from pathlib import Path

from aegis.providers.mock_provider import render_legit_image

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "legit_samples.json"

_LEGIT_CHATS = [
    "Customer: Hi, my order arrived with a cracked case, here are two photos of the damage.\n"
    "Customer: Could I get a refund or replacement please?",
    "Customer: The item stopped working after one use and there's visible damage to the housing.\n"
    "Customer: I've attached photos from the front and side.",
    "Customer: Unfortunately this arrived broken during shipping, photos attached.\n"
    "Customer: Happy to send it back if needed.",
    "Customer: There's a defect on the item that wasn't mentioned in the listing, see attached photos.\n"
    "Customer: Let me know what you need from me to process this.",
]

_CASES = [
    {"order_value_usd": 39.99, "days_since_delivery": 4, "is_first_time_customer": False, "shipping_billing_mismatch": False},
    {"order_value_usd": 89.50, "days_since_delivery": 6, "is_first_time_customer": True, "shipping_billing_mismatch": False},
    {"order_value_usd": 24.00, "days_since_delivery": 2, "is_first_time_customer": False, "shipping_billing_mismatch": False},
    {"order_value_usd": 175.00, "days_since_delivery": 10, "is_first_time_customer": False, "shipping_billing_mismatch": False},
    {"order_value_usd": 55.00, "days_since_delivery": 1, "is_first_time_customer": True, "shipping_billing_mismatch": False},
    {"order_value_usd": 12.99, "days_since_delivery": 5, "is_first_time_customer": False, "shipping_billing_mismatch": False},
    {"order_value_usd": 210.00, "days_since_delivery": 8, "is_first_time_customer": False, "shipping_billing_mismatch": False},
    {"order_value_usd": 65.25, "days_since_delivery": 3, "is_first_time_customer": False, "shipping_billing_mismatch": False},
    {"order_value_usd": 30.00, "days_since_delivery": 7, "is_first_time_customer": True, "shipping_billing_mismatch": False},
    {"order_value_usd": 48.75, "days_since_delivery": 2, "is_first_time_customer": False, "shipping_billing_mismatch": False},
]


def _image_to_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def build_cases() -> list[dict]:
    cases = []
    for i, order in enumerate(_CASES):
        seed_key = f"legit-case-{i}"
        front = render_legit_image(seed_key, angle_seed=i * 2)
        side = render_legit_image(seed_key, angle_seed=i * 2 + 1)
        cases.append(
            {
                "case_id": f"legit-{i:03d}",
                "chat_transcript": _LEGIT_CHATS[i % len(_LEGIT_CHATS)],
                "claimed_reason": "Defective merchandise (reason code 4853)",
                "order_metadata": order,
                "images": [
                    {
                        "angle": "front",
                        "data_b64": _image_to_b64(front),
                        "format": "png",
                        "generation_strategy": "genuine",
                    },
                    {
                        "angle": "45deg_side",
                        "data_b64": _image_to_b64(side),
                        "format": "png",
                        "generation_strategy": "genuine",
                    },
                ],
            }
        )
    return cases


if __name__ == "__main__":
    cases = build_cases()
    OUTPUT_PATH.write_text(json.dumps(cases, indent=2))
    print(f"Wrote {len(cases)} legit cases to {OUTPUT_PATH}")
