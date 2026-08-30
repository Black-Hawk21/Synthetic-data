"""Pure logic for the interactive "Try It Yourself" tab (app.py) -- a human types as
the customer and uploads real images, instead of the Red Team generating both sides.
Kept separate and Streamlit-free so it's unit-testable like the rest of the pipeline;
app.py only wires this to st.session_state and widgets.
"""

from aegis.schemas import SanitizerResult


def combine_sanitizer_history(history: list[SanitizerResult]) -> SanitizerResult:
    """Reduces a per-turn sanitizer history into one representative result, for both
    the live 'chat risk' display and feeding the supervisor once images are in. Any
    injection ever detected stays flagged (a security violation doesn't un-happen
    because a later message was polite), and the risk shown is the worst seen so far."""
    if not history:
        return SanitizerResult(
            injection_detected=False, manipulation_risk_score=0.0, reason="", flagged_phrases=[], cleaned_transcript=""
        )
    worst = max(history, key=lambda s: s.manipulation_risk_score)
    all_phrases = sorted({phrase for result in history for phrase in result.flagged_phrases})
    return SanitizerResult(
        injection_detected=any(result.injection_detected for result in history),
        manipulation_risk_score=worst.manipulation_risk_score,
        reason=worst.reason,
        flagged_phrases=all_phrases,
        cleaned_transcript=history[-1].cleaned_transcript,
    )
