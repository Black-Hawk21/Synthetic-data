"""Thin wrapper over st.session_state. Only imported by app.py -- orchestrator.py
and the blueteam/redteam modules stay Streamlit-free so they're unit-testable."""

import streamlit as st

from aegis.schemas import (
    BatchCaseResult,
    ChatMessage,
    ImageMetadataReport,
    ImageSubmission,
    RoundRecord,
    SanitizerResult,
    VisionInspectionResult,
)

_ROUNDS_KEY = "aegis_rounds"
_BATCH_KEY = "aegis_batch_results"
_MANUAL_HISTORY_KEY = "aegis_manual_history"
_MANUAL_SANITIZER_HISTORY_KEY = "aegis_manual_sanitizer_history"
_MANUAL_IMAGES_KEY = "aegis_manual_images"
_MANUAL_VISION_KEY = "aegis_manual_vision"
_MANUAL_METADATA_KEY = "aegis_manual_metadata"
_MANUAL_ACTION_KEY = "aegis_manual_action"


def get_rounds() -> list[RoundRecord]:
    return st.session_state.setdefault(_ROUNDS_KEY, [])


def add_round(record: RoundRecord) -> None:
    get_rounds().append(record)


def reset_rounds() -> None:
    st.session_state[_ROUNDS_KEY] = []


def get_batch_results() -> list[BatchCaseResult]:
    return st.session_state.setdefault(_BATCH_KEY, [])


def set_batch_results(results: list[BatchCaseResult]) -> None:
    st.session_state[_BATCH_KEY] = results


# ---- Try It Yourself tab (manual/interactive session) ----------------------

def get_manual_history() -> list[ChatMessage]:
    return st.session_state.setdefault(_MANUAL_HISTORY_KEY, [])


def add_manual_message(message: ChatMessage) -> None:
    get_manual_history().append(message)


def get_manual_sanitizer_history() -> list[SanitizerResult]:
    return st.session_state.setdefault(_MANUAL_SANITIZER_HISTORY_KEY, [])


def add_manual_sanitizer_result(result: SanitizerResult) -> None:
    get_manual_sanitizer_history().append(result)


def get_manual_images() -> list[ImageSubmission] | None:
    return st.session_state.get(_MANUAL_IMAGES_KEY)


def set_manual_images(images: list[ImageSubmission] | None) -> None:
    st.session_state[_MANUAL_IMAGES_KEY] = images


def get_manual_vision_result() -> VisionInspectionResult | None:
    return st.session_state.get(_MANUAL_VISION_KEY)


def set_manual_vision_result(result: VisionInspectionResult | None) -> None:
    st.session_state[_MANUAL_VISION_KEY] = result


def get_manual_metadata() -> ImageMetadataReport | None:
    return st.session_state.get(_MANUAL_METADATA_KEY)


def set_manual_metadata(report: ImageMetadataReport | None) -> None:
    st.session_state[_MANUAL_METADATA_KEY] = report


def get_manual_action() -> str | None:
    return st.session_state.get(_MANUAL_ACTION_KEY)


def set_manual_action(action: str | None) -> None:
    st.session_state[_MANUAL_ACTION_KEY] = action


def reset_manual_session() -> None:
    st.session_state[_MANUAL_HISTORY_KEY] = []
    st.session_state[_MANUAL_SANITIZER_HISTORY_KEY] = []
    st.session_state[_MANUAL_IMAGES_KEY] = None
    st.session_state[_MANUAL_VISION_KEY] = None
    st.session_state[_MANUAL_METADATA_KEY] = None
    st.session_state[_MANUAL_ACTION_KEY] = None
