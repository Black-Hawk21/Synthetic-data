from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Decision(str, Enum):
    APPROVE = "APPROVE"
    ESCALATE = "ESCALATE"
    REJECT = "REJECT"


class ChatMessage(BaseModel):
    role: Literal["customer", "support_bot"]
    content: str


class ImageSubmission(BaseModel):
    angle: str  # e.g. "front", "45deg_side"
    data_b64: str  # base64-encoded image bytes
    format: str = "png"
    generation_strategy: str = "naive_independent"  # or "img2img_conditioned"


class OrderMetadata(BaseModel):
    order_value_usd: float = 49.99
    days_since_delivery: int = 2
    is_first_time_customer: bool = False
    shipping_billing_mismatch: bool = False


class DisputePayload(BaseModel):
    chat_transcript: str
    # Populated only for interactively-generated rounds (Live Simulation tab), so the
    # UI can render real chat bubbles; empty for batch-eval payloads, which only ever
    # need the flattened chat_transcript string.
    chat_messages: list[ChatMessage] = Field(default_factory=list)
    images: list[ImageSubmission] = Field(min_length=2, max_length=2)
    claimed_reason: str = "Defective merchandise (reason code 4853)"
    order_metadata: OrderMetadata = Field(default_factory=OrderMetadata)


class Finding(BaseModel):
    type: str
    confidence: float
    description: str


class SanitizerResult(BaseModel):
    injection_detected: bool
    # 0-1, higher = more manipulative/pressuring tone (urgency bluffing, fabricated
    # authority, policy-citation pressure) -- independent of injection_detected, which
    # stays reserved for a definite instruction-override attempt. This is what lets a
    # customer's tone alone move fraud_confidence even when nothing crosses the hard
    # injection line, and what the live "chat risk" panel displays turn by turn.
    manipulation_risk_score: float = 0.0
    # One-sentence plain-English explanation of *why* this score/verdict was given --
    # the "flagged phrases" alone say what, this says why it's a problem (or why it
    # isn't, for a calm/unremarkable message).
    reason: str = ""
    flagged_phrases: list[str] = Field(default_factory=list)
    cleaned_transcript: str


class VisionInspectionResult(BaseModel):
    artifact_score: float  # 0-1, higher = more likely synthetic/manipulated
    # 0-1, higher = more consistent (more likely genuine). The final, hardest-to-fool
    # verdict -- computed as the MINIMUM of three independent methods (holistic
    # full-image reasoning, geometric_consistency_score, and shape_match_score below),
    # so none of the three can get diluted/averaged away by the other two looking clean.
    angle_consistency_score: float
    # 0-1, higher = more consistent. Deterministic, not model judgment: computed in
    # Python from landmark-relative damage coordinates the model is only asked to
    # *measure*, not judge -- existing to work around VLMs being reliably bad at
    # judging cross-angle 3D geometric/topological consistency directly (tested
    # extensively; see vision_inspector.py for the full rationale).
    geometric_consistency_score: float = 1.0
    # 0-1, higher = more consistent. A model judgment, but on a cropped-and-scaled
    # side-by-side composite of just the two damage regions rather than the full
    # photos -- isolating the comparison from background/pose distractions the
    # holistic full-image check has to reason around.
    shape_match_score: float = 1.0
    # 0-1, how far apart the three methods above landed from each other. High
    # disagreement between independent methods is itself worth escalating on, even
    # when the combined score alone wouldn't be -- see config.STRONG_SIGNAL_OVERRIDES.
    consistency_disagreement: float = 0.0
    # 0-1, higher = more consistent. A stricter, more literal check than
    # angle_consistency_score: exact color/finish, logo/label placement, material
    # texture, and existing wear marks matching precisely between the two photos --
    # designed to catch the subtle slip-ups a sophisticated (e.g. img2img-conditioned)
    # attack still makes even when the overall pose/lighting looks consistent.
    detail_consistency_score: float
    semantic_match: bool
    findings: list[Finding] = Field(default_factory=list)


class ImageMetadataReport(BaseModel):
    """Deterministic, non-LLM metadata forensics on the raw uploaded file bytes --
    only meaningful for real evidence (the Try It Yourself tab). Red Team-generated
    and legit-fixture images are synthetically rendered by this same codebase and
    have no real-world metadata to check, so this is never computed for that path.
    Absence of camera EXIF is suggestive, not conclusive: ordinary processing
    (screenshots, some messaging apps) strips it from genuine photos too."""

    has_camera_exif: bool
    camera_make: str | None = None
    camera_model: str | None = None
    capture_timestamp: str | None = None
    has_c2pa_marker: bool = False
    # IPTC DigitalSourceType in XMP -- a plain, non-cryptographic "made/edited with AI"
    # declaration some generators (Gemini included) embed, distinct from C2PA's signed
    # manifest and from Google's separate SynthID watermark (undecodable by us).
    has_ai_digital_source_marker: bool = False
    notes: list[str] = Field(default_factory=list)


class SupervisorResult(BaseModel):
    decision: Decision
    fraud_confidence: float
    findings: list[Finding] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class AttackSpec(BaseModel):
    tactic: str
    technique: str


class RoundRecord(BaseModel):
    round_number: int
    attack: AttackSpec
    payload: DisputePayload
    sanitizer_result: SanitizerResult
    vision_result: VisionInspectionResult | None = None
    supervisor_result: SupervisorResult
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class BatchCaseResult(BaseModel):
    case_id: str
    is_fraud_ground_truth: bool
    attack: AttackSpec | None = None
    supervisor_result: SupervisorResult
