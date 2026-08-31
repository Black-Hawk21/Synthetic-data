import os

from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
# Haiku for text: the interactive chat now makes several small calls per round (one per
# turn), and Haiku is much cheaper per token than Sonnet while being plenty capable for
# short support-chat dialogue. Vision stays on Sonnet for stronger forensic reasoning on
# the six-signal damage-photo inspection that drives the judged detection metrics --
# Haiku 4.5 does support image input, but Sonnet is the better tradeoff for this specific
# judgment call, not a technical requirement.
ANTHROPIC_TEXT_MODEL = os.getenv("ANTHROPIC_TEXT_MODEL", "claude-haiku-4-5-20251001")
ANTHROPIC_VISION_MODEL = os.getenv("ANTHROPIC_VISION_MODEL", "claude-sonnet-4-5-20250929")

IMAGE_PROVIDER = os.getenv("IMAGE_PROVIDER", "mock").strip().lower()
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY", "").strip()
# Not used by the running app by default (pollinations is the free runtime default) --
# useful as a one-off swap (IMAGE_PROVIDER=gemini) when regenerating the curated demo
# gallery in scripts/generate_fraud_samples.py with higher-fidelity images.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")

ROUND_CAP = _env_int("ROUND_CAP", 5)

# Supervisor decision thresholds -- exposed here so the Batch Evaluation UI
# tab and the orchestrator share one source of truth.
APPROVE_BELOW = 0.4
REJECT_ABOVE = 0.6

FRAUD_CONFIDENCE_WEIGHTS = {
    "artifact_score": 0.25,
    "angle_inconsistency": 0.15,  # applied to (1 - angle_consistency_score)
    "detail_inconsistency": 0.20,  # applied to (1 - detail_consistency_score)
    "semantic_mismatch": 0.10,  # applied to (1 - semantic_match)
    "policy_risk": 0.10,
    "chat_risk": 0.20,  # applied to sanitizer_result.manipulation_risk_score
}

# A sophisticated attack can nail any ONE signal category (e.g. img2img conditioning
# specifically defeats angle_consistency_score) while still leaving a strong tell in
# another -- averaging every signal together lets that one strong tell get diluted
# into an overall APPROVE. Each rule below floors the decision at ESCALATE whenever
# its signal is this suspicious on its own, regardless of what the blended score says,
# matching how real fraud-ops policy treats one strong red flag as grounds for manual
# review even when the aggregate risk score looks moderate. (field, "above"/"below", threshold)
STRONG_SIGNAL_OVERRIDES: list[tuple[str, str, float]] = [
    ("artifact_score", "above", 0.7),
    ("detail_consistency_score", "below", 0.3),
    # Safety net for the known VLM blind spot on cross-angle geometric consistency
    # (see vision_inspector.py's localization/shape-match checks): even when the
    # combined angle_consistency_score itself isn't extreme, if the three independent
    # consistency methods that fed into it substantially disagree with each other,
    # that disagreement is itself worth a human look -- deliberately technique-agnostic
    # (it never inspects which attack technique produced the images, only how much the
    # methods disagree), so it doesn't leak Red Team ground truth into the decision.
    ("consistency_disagreement", "above", 0.4),
]

# Additive, not part of the weights above -- only ever applied when a real
# ImageMetadataReport is supplied (Try It Yourself tab, real uploaded evidence), so
# it never touches the Red Team/Batch Evaluation pipeline's calibration. Modest by
# design: metadata absence alone shouldn't swing a decision, see metadata_forensics.py.
METADATA_RISK_WEIGHT = 0.15


def has_live_text_provider() -> bool:
    return bool(ANTHROPIC_API_KEY)


def has_live_image_provider() -> bool:
    if IMAGE_PROVIDER == "pollinations":
        return True
    if IMAGE_PROVIDER == "stability":
        return bool(STABILITY_API_KEY)
    if IMAGE_PROVIDER == "gemini":
        return bool(GEMINI_API_KEY)
    return False
