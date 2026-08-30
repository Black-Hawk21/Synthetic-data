from aegis import config
from aegis.blueteam.metadata_forensics import metadata_risk_score
from aegis.schemas import (
    Decision,
    ImageMetadataReport,
    OrderMetadata,
    SanitizerResult,
    SupervisorResult,
    VisionInspectionResult,
)


def _policy_risk_factor(order_metadata: OrderMetadata) -> float:
    risk = 0.0
    if order_metadata.days_since_delivery <= 1:
        risk += 0.4
    elif order_metadata.days_since_delivery <= 3:
        risk += 0.15
    if order_metadata.order_value_usd > 150:
        risk += 0.25
    if order_metadata.is_first_time_customer:
        risk += 0.2
    if order_metadata.shipping_billing_mismatch:
        risk += 0.35
    return min(risk, 1.0)


def _fired_overrides(vision_result: VisionInspectionResult) -> list[str]:
    """Checks each rule in config.STRONG_SIGNAL_OVERRIDES against vision_result and
    returns human-readable descriptions for every rule that fired."""
    fired = []
    for field, direction, threshold in config.STRONG_SIGNAL_OVERRIDES:
        value = getattr(vision_result, field)
        if (direction == "above" and value >= threshold) or (direction == "below" and value <= threshold):
            comparator = "is at least" if direction == "above" else "is at most"
            fired.append(
                f"{field} {value:.2f} {comparator} {threshold} -- independently suspicious despite an "
                "otherwise consistent submission, routed to review rather than approved."
            )
    return fired


def decide(
    sanitizer_result: SanitizerResult,
    vision_result: VisionInspectionResult | None,
    order_metadata: OrderMetadata,
    metadata_report: ImageMetadataReport | None = None,
) -> SupervisorResult:
    """The dual-LLM defense boundary. This function only ever sees sanitizer flags,
    vision-inspector JSON, and order metadata -- never the raw chat transcript. Even
    a perfect jailbreak of the customer-facing bot can't reach the logic that holds
    refund authority.

    metadata_report is optional and additive: it's only ever populated for real
    uploaded evidence (the Try It Yourself tab), never for the Red Team/Batch
    Evaluation pipeline, whose images are all synthetically rendered by this same
    codebase and have no real-world metadata to check. Left as None there, it simply
    doesn't contribute -- the base formula's calibration is unchanged."""
    if sanitizer_result.injection_detected:
        phrases = ", ".join(sanitizer_result.flagged_phrases) or "manipulation attempt"
        return SupervisorResult(
            decision=Decision.REJECT,
            fraud_confidence=1.0,
            findings=[],
            reasons=[f"policy_violation: manipulation_attempt ({phrases})"],
        )

    weights = config.FRAUD_CONFIDENCE_WEIGHTS
    policy_risk = _policy_risk_factor(order_metadata)
    fraud_confidence = (
        weights["artifact_score"] * vision_result.artifact_score
        + weights["angle_inconsistency"] * (1 - vision_result.angle_consistency_score)
        + weights["detail_inconsistency"] * (1 - vision_result.detail_consistency_score)
        + weights["semantic_mismatch"] * (0.0 if vision_result.semantic_match else 1.0)
        + weights["policy_risk"] * policy_risk
        + weights["chat_risk"] * sanitizer_result.manipulation_risk_score
    )
    if metadata_report is not None:
        fraud_confidence += config.METADATA_RISK_WEIGHT * metadata_risk_score(metadata_report)
    fraud_confidence = min(max(fraud_confidence, 0.0), 1.0)

    if fraud_confidence < config.APPROVE_BELOW:
        decision = Decision.APPROVE
    elif fraud_confidence <= config.REJECT_ABOVE:
        decision = Decision.ESCALATE
    else:
        decision = Decision.REJECT

    override_reasons = _fired_overrides(vision_result)
    if override_reasons and decision == Decision.APPROVE:
        decision = Decision.ESCALATE

    reasons = [f.description for f in vision_result.findings] if decision != Decision.APPROVE else []
    if override_reasons and decision == Decision.ESCALATE and fraud_confidence < config.APPROVE_BELOW:
        reasons = override_reasons + reasons
    if metadata_report is not None and decision != Decision.APPROVE:
        reasons = metadata_report.notes + reasons
    return SupervisorResult(
        decision=decision,
        fraud_confidence=round(fraud_confidence, 3),
        findings=vision_result.findings,
        reasons=reasons,
    )
