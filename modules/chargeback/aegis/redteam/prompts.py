from aegis.attacks.products import pick_product
from aegis.attacks.taxonomy import SOCIAL_ENGINEERING_TACTICS
from aegis.schemas import AttackSpec, Finding

_ANGLE_DESCRIPTIONS = {
    "front": "a straight-on front view",
    "45deg_side": "a 45-degree angled side view",
}

# Visual style hints for the image prompt itself -- deliberately separate from
# IMAGE_FORGERY_TECHNIQUES' human-readable descriptions (aegis/attacks/taxonomy.py),
# which describe the attack conceptually for the UI/docs and aren't written as
# something a diffusion model should render literally (e.g. "no special evasion
# technique" or "injects EXIF metadata" aren't visual instructions).
_TECHNIQUE_VISUAL_HINTS: dict[str, str] = {
    "lighting_shadow_inconsistency": "dramatic close-up lighting with one hard directional shadow",
    "spliced_edited_photo": "the damaged area looks slightly composited onto the surface, with a "
    "faint mismatched edge",
}

CHAT_SYSTEM_PROMPT = (
    "You generate synthetic training examples for Aegis, a fraud-DETECTION classifier built for "
    "the Mastercard Innovation Challenge (an academic hackathon on AI payment-fraud defense). Each "
    "example is a labeled customer-support chat message exhibiting a known manipulation pattern, "
    "used only to train and evaluate an automated detector that flags this exact pattern. These "
    "examples are synthetic, stay entirely inside this local research pipeline, and are never sent "
    "to any real company or person. Output only the message text itself, no labels or commentary."
)

# (keyword, phrase) -- checked as a case-insensitive substring against each finding's
# own "{type} {description}" text, not an exact dict-key match against finding.type.
# The live vision inspector (aegis/blueteam/vision_inspector.py) runs five independent
# model calls that each invent their own freeform finding-type labels rather than
# sticking to one fixed vocabulary, so an exact match against a handful of hardcoded
# strings (the mock provider's canned types) almost never fires against real model
# output -- which silently turned this entire feedback loop into a no-op in live mode:
# confirmed by a live functional run where two consecutive rounds generated
# byte-identical images because next_strategy_and_suffix never found a matching key.
# Substring matching against keywords that already appear naturally in how a forensic
# model describes these categories (regardless of its exact type label) is what
# actually keeps the loop live. Order matters only for which phrase wins on overlap;
# duplicates are removed by the caller, not here.
MUTATION_KEYWORD_RULES: list[tuple[str, str]] = [
    ("light", "consistent single-source soft window lighting, shadow direction matching one primary light"),
    ("shadow", "consistent single-source soft window lighting, shadow direction matching one primary light"),
    ("noise", "natural varying ISO 800 sensor grain between shots, avoid identical noise"),
    ("grain", "natural varying ISO 800 sensor grain between shots, avoid identical noise"),
    ("edge", "clean natural edges at the damage site, no blending halo or warped texture"),
    ("blend", "clean natural edges at the damage site, no blending halo or warped texture"),
    ("artifact", "clean natural edges at the damage site, no blending halo or warped texture"),
    ("semantic", "damage clearly matches the claimed defect, consistent product across both photos"),
    ("mismatch", "damage clearly matches the claimed defect, consistent product across both photos"),
    ("device", "identical product, device, and background details across both angles"),
]

# Keywords indicating a cross-angle/geometric/topology problem specifically -- these
# escalate the GENERATION STRATEGY itself (naive independent -> img2img conditioned),
# not just the prompt wording, since that class of finding means independently
# generating the two angles is the fundamental weak point, not a fixable phrase.
CROSS_ANGLE_ESCALATION_KEYWORDS = ("angle", "geometric", "topology", "position", "incompatible")


def build_chat_prompt(attack: AttackSpec) -> tuple[str, str]:
    tactic_description = SOCIAL_ENGINEERING_TACTICS[attack.tactic]
    user = (
        f"TACTIC: {attack.tactic}\n"
        f"Tactic description: {tactic_description}\n\n"
        "Generate a synthetic dataset example: a short customer-support chat transcript (2-4 "
        "'Customer:' lines) disputing a payment for defective merchandise (Mastercard reason code "
        "4853), exhibiting this labeled manipulation pattern."
    )
    return CHAT_SYSTEM_PROMPT, user


def build_mutation_prompt(previous_transcript: str, findings: list[Finding]) -> tuple[str, str]:
    reasons = "; ".join(f.description for f in findings) or "the claim was rejected"
    user = (
        "MUTATE\n"
        f"This synthetic example was correctly flagged by the detector. Flagged reasons: {reasons}\n"
        f"Previous example:\n{previous_transcript}\n\n"
        "Generate a revised synthetic dataset example: a slightly more insistent version of this "
        "chat transcript that acknowledges submitting new photos, still exhibiting the same "
        "labeled manipulation pattern, without admitting wrongdoing."
    )
    return CHAT_SYSTEM_PROMPT, user


def build_image_prompt(attack: AttackSpec, angle: str, mutation_suffix: str = "") -> str:
    product_name, damage_description = pick_product(f"{attack.tactic}|{attack.technique}")
    angle_description = _ANGLE_DESCRIPTIONS.get(angle, angle)
    base = (
        f"Photo-realistic ordinary consumer photo of a {product_name}, {angle_description}, "
        f"showing {damage_description}. Realistic indoor lighting, natural phone-camera grain, "
        f"not studio photography."
    )
    visual_hint = _TECHNIQUE_VISUAL_HINTS.get(attack.technique)
    if visual_hint:
        base += f" {visual_hint}."
    if mutation_suffix:
        base += f" Additional realism cues: {mutation_suffix}."
    return base
