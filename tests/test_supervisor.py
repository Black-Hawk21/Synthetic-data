from aegis.blueteam.supervisor import decide
from aegis.schemas import Decision, ImageMetadataReport, OrderMetadata, SanitizerResult, VisionInspectionResult

_NO_INJECTION = SanitizerResult(injection_detected=False, flagged_phrases=[], cleaned_transcript="")
_DEFAULT_ORDER = OrderMetadata()


def test_strong_artifact_score_escalates_even_when_other_signals_look_clean():
    # Mirrors the real Sony ULT WEAR img2img example: a sophisticated attack nails
    # cross-angle and detail consistency, but the vision model still flags a high
    # artifact_score. The blended weighted average alone would land on APPROVE --
    # the single-strong-signal override exists specifically so this doesn't get
    # silently averaged away.
    vision_result = VisionInspectionResult(
        artifact_score=0.75,
        angle_consistency_score=0.85,
        detail_consistency_score=0.8,
        semantic_match=True,
        findings=[],
    )
    result = decide(_NO_INJECTION, vision_result, _DEFAULT_ORDER)
    assert result.decision == Decision.ESCALATE
    assert any("artifact_score" in reason for reason in result.reasons)


def test_weak_detail_consistency_escalates_even_when_other_signals_look_clean():
    # Symmetric case: overall pose/lighting (angle_consistency) and artifact_score
    # both look clean, but fine details (logo placement, texture) don't match --
    # the kind of subtle slip-up detail_consistency_score exists to catch.
    vision_result = VisionInspectionResult(
        artifact_score=0.15,
        angle_consistency_score=0.85,
        detail_consistency_score=0.2,
        semantic_match=True,
        findings=[],
    )
    result = decide(_NO_INJECTION, vision_result, _DEFAULT_ORDER)
    assert result.decision == Decision.ESCALATE
    assert any("detail_consistency_score" in reason for reason in result.reasons)


def test_low_risk_case_still_approves_without_any_override_firing():
    vision_result = VisionInspectionResult(
        artifact_score=0.1,
        angle_consistency_score=0.9,
        detail_consistency_score=0.9,
        semantic_match=True,
        findings=[],
    )
    result = decide(_NO_INJECTION, vision_result, _DEFAULT_ORDER)
    assert result.decision == Decision.APPROVE


def test_override_does_not_downgrade_an_already_rejected_case():
    vision_result = VisionInspectionResult(
        artifact_score=0.9,
        angle_consistency_score=0.1,
        detail_consistency_score=0.1,
        semantic_match=False,
        findings=[],
    )
    result = decide(_NO_INJECTION, vision_result, _DEFAULT_ORDER)
    assert result.decision == Decision.REJECT


def test_manipulative_chat_tone_alone_can_push_a_borderline_case_past_approve():
    # Same clean vision signals as the approve case above, but a high
    # manipulation_risk_score from the sanitizer (bad client behavior short of an
    # outright injection attempt) should still be able to move the needle -- the
    # chat-risk weight exists so tone isn't ignored just because it never crosses
    # the hard injection line.
    clean_vision = VisionInspectionResult(
        artifact_score=0.1,
        angle_consistency_score=0.9,
        detail_consistency_score=0.9,
        semantic_match=True,
        findings=[],
    )
    calm_sanitizer = SanitizerResult(
        injection_detected=False, manipulation_risk_score=0.0, flagged_phrases=[], cleaned_transcript=""
    )
    manipulative_sanitizer = SanitizerResult(
        injection_detected=False, manipulation_risk_score=0.95, flagged_phrases=[], cleaned_transcript=""
    )
    calm_result = decide(calm_sanitizer, clean_vision, _DEFAULT_ORDER)
    manipulative_result = decide(manipulative_sanitizer, clean_vision, _DEFAULT_ORDER)
    assert manipulative_result.fraud_confidence > calm_result.fraud_confidence


def test_metadata_report_is_ignored_when_not_supplied():
    # The Red Team/Batch Evaluation pipeline never passes a metadata_report (its
    # images are all synthetically rendered) -- omitting it must reproduce the exact
    # same calibration as before this signal existed.
    vision_result = VisionInspectionResult(
        artifact_score=0.1, angle_consistency_score=0.9, detail_consistency_score=0.9, semantic_match=True, findings=[]
    )
    without_metadata = decide(_NO_INJECTION, vision_result, _DEFAULT_ORDER)
    with_none_explicit = decide(_NO_INJECTION, vision_result, _DEFAULT_ORDER, None)
    assert without_metadata.fraud_confidence == with_none_explicit.fraud_confidence


def test_missing_camera_exif_raises_fraud_confidence_but_present_metadata_does_not():
    vision_result = VisionInspectionResult(
        artifact_score=0.1, angle_consistency_score=0.9, detail_consistency_score=0.9, semantic_match=True, findings=[]
    )
    no_metadata = decide(_NO_INJECTION, vision_result, _DEFAULT_ORDER)
    missing_exif = decide(_NO_INJECTION, vision_result, _DEFAULT_ORDER, ImageMetadataReport(has_camera_exif=False))
    has_exif = decide(_NO_INJECTION, vision_result, _DEFAULT_ORDER, ImageMetadataReport(has_camera_exif=True))
    assert missing_exif.fraud_confidence > no_metadata.fraud_confidence
    assert has_exif.fraud_confidence < missing_exif.fraud_confidence


def test_c2pa_marker_is_the_strongest_metadata_signal():
    vision_result = VisionInspectionResult(
        artifact_score=0.1, angle_consistency_score=0.9, detail_consistency_score=0.9, semantic_match=True, findings=[]
    )
    missing_exif = decide(_NO_INJECTION, vision_result, _DEFAULT_ORDER, ImageMetadataReport(has_camera_exif=False))
    c2pa_found = decide(
        _NO_INJECTION, vision_result, _DEFAULT_ORDER, ImageMetadataReport(has_camera_exif=False, has_c2pa_marker=True)
    )
    assert c2pa_found.fraud_confidence > missing_exif.fraud_confidence
