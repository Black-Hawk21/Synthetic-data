"""A small catalog of concrete, visually-groundable products with a specific damage
description. A vague prompt like "product photo showing damage" gives a weak free
image model nothing to anchor on, and it improvises an unrelated subject entirely
(observed in practice: it produced a portrait of a person's face). A concrete noun
plus a concrete defect reliably keeps generation on-subject.
"""

import hashlib

PRODUCTS: list[tuple[str, str]] = [
    ("wireless headphones", "a visible crack across one ear cup with the foam padding exposed"),
    ("smartphone", "a shattered screen with spiderweb cracks radiating from one corner"),
    ("blender", "a long crack down the side of the clear plastic pitcher"),
    ("laptop computer", "a dented corner and a crack across the bottom case"),
    ("coffee maker", "a cracked glass carafe with a chip missing from the rim"),
    ("running shoes", "the sole visibly separating from the upper at the toe"),
    ("smartwatch", "a shattered watch face with spiderweb cracks across the glass"),
    ("backpack", "a torn seam along the bottom and a broken zipper pull"),
]


def pick_product(seed_key: str) -> tuple[str, str]:
    """Deterministic per-attack pick, so both angle photos of the same round -- and
    every mutation/resubmission of the same attack -- show the same product. Cross-angle
    product consistency is exactly the signal the two-angle requirement is designed to
    test, so it must hold regardless of which technique generated the images."""
    index = int(hashlib.sha256(seed_key.encode()).hexdigest(), 16) % len(PRODUCTS)
    return PRODUCTS[index]
