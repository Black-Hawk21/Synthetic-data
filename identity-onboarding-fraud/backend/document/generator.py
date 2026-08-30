"""Synthetic document image generator (section 3 & 10).

Produces a CLEARLY LABELED fictional document -- never a realistic
counterfeit of any real government ID format. Every image is watermarked
"SYNTHETIC IDENTITY DOCUMENT -- FOR RESEARCH" and uses fictional field
layout. Used by the Onboarding Simulator page for the single-applicant OCR
demo; bulk dataset generation never renders images (numeric features only,
section 30).
"""
from __future__ import annotations

import io
import random
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFilter, ImageFont


@dataclass
class SyntheticDocumentFields:
    name: str
    date_of_birth: str
    document_number: str
    address: str
    document_type: str = "NATIONAL_ID"


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def generate_document_image(fields: SyntheticDocumentFields, width: int = 850, height: int = 540) -> Image.Image:
    img = Image.new("RGB", (width, height), color=(240, 244, 248))
    draw = ImageDraw.Draw(img)

    # Border + header
    draw.rectangle([8, 8, width - 8, height - 8], outline=(30, 60, 110), width=4)
    draw.rectangle([8, 8, width - 8, 64], fill=(30, 60, 110))
    draw.text((24, 20), "SYNTHETIC IDENTITY DOCUMENT -- FOR RESEARCH", fill=(255, 255, 255), font=_font(22))

    # Fictional photo placeholder (never a real face)
    draw.rectangle([30, 90, 220, 300], outline=(120, 120, 120), width=2, fill=(210, 214, 220))
    draw.line([30, 90, 220, 300], fill=(180, 184, 190), width=2)
    draw.line([220, 90, 30, 300], fill=(180, 184, 190), width=2)
    draw.text((45, 185), "SYNTHETIC\nPHOTO", fill=(90, 90, 90), font=_font(16))

    labels = [
        ("Document Type", fields.document_type),
        ("Full Name", fields.name),
        ("Date of Birth", fields.date_of_birth),
        ("Document No.", fields.document_number),
        ("Address", fields.address),
    ]
    y = 100
    for label, value in labels:
        draw.text((250, y), f"{label}:", fill=(60, 60, 60), font=_font(16))
        draw.text((250, y + 22), str(value), fill=(15, 15, 15), font=_font(20))
        y += 62

    draw.text((24, height - 40), "This is a fictional, machine-generated test document. Not a real government ID.", fill=(140, 30, 30), font=_font(13))
    return img


def apply_perturbations(
    img: Image.Image, blur: bool = False, noise: bool = False,
    rotate_degrees: float = 0.0, jpeg_artifact: bool = False,
) -> Image.Image:
    """Controlled visual perturbations used to benchmark the OCR/tamper
    pipeline (section 10)."""
    out = img
    if rotate_degrees:
        out = out.rotate(rotate_degrees, expand=True, fillcolor=(255, 255, 255))
    if blur:
        out = out.filter(ImageFilter.GaussianBlur(radius=2.2))
    if noise:
        import numpy as np
        arr = np.array(out).astype(np.int16)
        noise_arr = np.random.default_rng(0).integers(-25, 25, size=arr.shape)
        arr = (arr + noise_arr).clip(0, 255).astype("uint8")
        out = Image.fromarray(arr)
    if jpeg_artifact:
        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=25)
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
    return out


def random_synthetic_fields(fake, seed: int | None = None) -> SyntheticDocumentFields:
    rng = random.Random(seed)
    return SyntheticDocumentFields(
        name=fake.name(),
        date_of_birth=fake.date_of_birth(minimum_age=18, maximum_age=80).isoformat(),
        document_number=f"SYN{rng.randint(10**7, 10**8 - 1)}",
        address=fake.address().replace("\n", ", "),
        document_type=rng.choice(["NATIONAL_ID", "PASSPORT", "DRIVERS_LICENSE"]),
    )
