from aegis.manual_session import combine_sanitizer_history
from aegis.schemas import SanitizerResult


def _sanitizer(injection: bool, risk: float, phrases: list[str] | None = None, reason: str = "") -> SanitizerResult:
    return SanitizerResult(
        injection_detected=injection,
        manipulation_risk_score=risk,
        reason=reason,
        flagged_phrases=phrases or [],
        cleaned_transcript="",
    )


def test_empty_history_is_low_risk():
    result = combine_sanitizer_history([])
    assert result.injection_detected is False
    assert result.manipulation_risk_score == 0.0


def test_injection_ever_detected_stays_flagged_even_after_a_calm_turn():
    history = [_sanitizer(True, 0.95, ["ignore previous instructions"]), _sanitizer(False, 0.1)]
    result = combine_sanitizer_history(history)
    assert result.injection_detected is True


def test_risk_shown_is_the_worst_seen_so_far():
    history = [
        _sanitizer(False, 0.2, reason="calm"),
        _sanitizer(False, 0.6, ["compliance"], reason="fabricated authority"),
        _sanitizer(False, 0.3, reason="calm"),
    ]
    result = combine_sanitizer_history(history)
    assert result.manipulation_risk_score == 0.6
    assert "compliance" in result.flagged_phrases
    assert result.reason == "fabricated authority"
