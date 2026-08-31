import base64
import io
import random
import re
import time

import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from aegis import config, state
from aegis.attacks import taxonomy
from aegis.blueteam.metadata_forensics import combine_metadata_reports, inspect_metadata
from aegis.blueteam.sanitizer import sanitize
from aegis.blueteam.supervisor import decide
from aegis.blueteam.vision_inspector import inspect
from aegis.chat_prompts import render_transcript
from aegis.examples import load_example_gallery
from aegis.manual_session import combine_sanitizer_history
from aegis.metrics import auc, confusion_at_threshold, export_batch_results, precision_recall_f1, roc_curve, run_batch
from aegis.orchestrator import Orchestrator
from aegis.providers.factory import get_image_provider, get_text_provider, get_vision_provider, is_live_mode
from aegis.redteam.agent import RedTeamAgent
from aegis.schemas import AttackSpec, ChatMessage, Decision, ImageSubmission, OrderMetadata, RoundRecord
from aegis.support_bot import SupportBotAgent

# Brief pause after each live chat bubble so the turn-by-turn exchange reads like a
# real conversation instead of the whole thing appearing at once.
_CHAT_TURN_DELAY_SECONDS = 0.6

# Reserved status palette -- used consistently everywhere a decision/confusion
# outcome is shown, never repurposed for anything else. Always paired with an
# icon + text label so meaning never rests on color alone.
STATUS = {
    Decision.APPROVE: ("#1B9E77", "✅"),
    Decision.ESCALATE: ("#D95F02", "⚠️"),
    Decision.REJECT: ("#D7191C", "❌"),
}

st.set_page_config(page_title="Aegis — AI Defense Lab", layout="wide")


@st.cache_resource
def _build_orchestrator(session_api_key: str | None) -> tuple[Orchestrator, RedTeamAgent]:
    # Takes the key as an argument (rather than reading a global) so st.cache_resource
    # keys its cache per distinct api_key value -- entering a key in the sidebar builds
    # a fresh live-mode orchestrator instead of reusing whatever was cached first.
    text_provider = get_text_provider(session_api_key)
    vision_provider = get_vision_provider(session_api_key)
    image_provider = get_image_provider()
    red_team = RedTeamAgent(text_provider, image_provider)
    support_bot = SupportBotAgent(text_provider)
    orchestrator = Orchestrator(
        red_team, text_provider, vision_provider, round_cap=config.ROUND_CAP, support_bot=support_bot
    )
    return orchestrator, red_team


with st.sidebar:
    st.markdown("### 🔑 Bring your own key")
    st.caption(
        "Optional. Paste an Anthropic API key to use live Claude for this session only -- "
        "it's never written to disk or shared. Leave blank to use the built-in zero-cost "
        "mock provider."
    )
    session_api_key = st.text_input("Anthropic API key", type="password", key="session_api_key") or None

orchestrator, red_team = _build_orchestrator(session_api_key)

st.title("🛡️ Aegis — AI Defense Lab for Payment Security")
mode_label = (
    "LIVE (Anthropic API)" if is_live_mode(session_api_key) else "MOCK (offline, zero-cost, zero API keys needed)"
)
st.caption(f"Provider mode: **{mode_label}** · Round cap: {config.ROUND_CAP}")

tab_live, tab_manual, tab_batch, tab_taxonomy, tab_gallery = st.tabs(
    ["Live Simulation", "Try It Yourself", "Batch Evaluation", "Attack Taxonomy", "Example Gallery"]
)


def _decision_badge(decision: Decision) -> str:
    color, icon = STATUS[decision]
    return f"<span style='color:{color}; font-weight:600'>{icon} {decision.value}</span>"


def _show_image(data_b64: str, caption: str) -> None:
    st.image(base64.b64decode(data_b64), caption=caption, width="stretch")


def _png_b64_from_bytes(raw_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _render_chat_bubble(message: ChatMessage) -> None:
    with st.chat_message("user" if message.role == "customer" else "assistant"):
        st.write(message.content)


def _make_live_chat_callback(placeholder):
    """Returns an on_message callback that renders each chat turn into `placeholder`
    the moment it's actually generated (real API call in between, not a replay), so
    the conversation visibly builds up live before the permanent round card -- with
    Blue Team analysis alongside it -- takes over below."""
    box = placeholder.container()

    def on_message(message: ChatMessage) -> None:
        with box:
            _render_chat_bubble(message)
        time.sleep(_CHAT_TURN_DELAY_SECONDS)

    return on_message


_IMAGE_FINDING_PREFIX = re.compile(r"^Image (\d+): (.*)$", re.DOTALL)


def _split_per_image_findings(findings, num_images: int):
    """Artifact-check findings are labeled "Image N: ..." (see vision_inspector.py) since
    each runs against exactly one photo -- pull those out so they can be shown directly
    under the photo they're about instead of in an unlabeled list a viewer has to
    mentally match back to a specific image. Identity/comparison findings are inherently
    about both images together and stay in the shared list."""
    per_image: dict[int, list[tuple[object, str]]] = {i: [] for i in range(1, num_images + 1)}
    shared = []
    for finding in findings:
        match = _IMAGE_FINDING_PREFIX.match(finding.description)
        index = int(match.group(1)) if match else None
        if match and index in per_image:
            per_image[index].append((finding, match.group(2)))
        else:
            shared.append(finding)
    return per_image, shared


def _finding_confidence(findings, finding_type: str) -> float | None:
    return next((f.confidence for f in findings if f.type == finding_type), None)


def _render_evidence(record: RoundRecord) -> None:
    """The two submitted photos plus whatever the Blue Team concluded from them,
    rendered together so it's visually obvious which score/finding belongs to which
    image -- this is the part of a round a demo viewer actually needs to see clearly."""
    st.markdown("**📸 Evidence under review**")
    st.caption(f"Image strategy: `{record.payload.images[0].generation_strategy}`")

    if record.sanitizer_result.injection_detected:
        img_cols = st.columns(2)
        for col, img in zip(img_cols, record.payload.images):
            with col:
                _show_image(img.data_b64, caption=img.angle)
        st.info(
            "Vision inspection was skipped for these photos -- the chat already tripped an "
            "automatic reject, so the dual-LLM boundary never forwarded the images for analysis."
        )
        return

    if not record.vision_result:
        img_cols = st.columns(2)
        for col, img in zip(img_cols, record.payload.images):
            with col:
                _show_image(img.data_b64, caption=img.angle)
        return

    v = record.vision_result
    per_image, shared_findings = _split_per_image_findings(v.findings, len(record.payload.images))
    img_cols = st.columns(2)
    for index, (col, img) in enumerate(zip(img_cols, record.payload.images), 1):
        with col:
            _show_image(img.data_b64, caption=img.angle)
            for finding, description in per_image.get(index, []):
                st.caption(f"`{finding.type}` ({finding.confidence:.2f}) — {description}")

    vcol1, vcol2, vcol3 = st.columns(3)
    vcol1.metric("Artifact", f"{v.artifact_score:.2f}")
    vcol2.metric("Angle consistency", f"{v.angle_consistency_score:.2f}")
    vcol3.metric("Detail consistency", f"{v.detail_consistency_score:.2f}")
    st.write("Semantic match:", "✅" if v.semantic_match else "❌")

    holistic_score = _finding_confidence(shared_findings, "consistency_submethod_holistic")
    with st.expander(
        f"Angle consistency breakdown — combined by MIN, {v.consistency_disagreement:.2f} disagreement "
        "across methods"
    ):
        bcol1, bcol2, bcol3 = st.columns(3)
        bcol1.metric("Holistic reasoning", f"{holistic_score:.2f}" if holistic_score is not None else "—")
        bcol2.metric("Geometric (deterministic)", f"{v.geometric_consistency_score:.2f}")
        bcol3.metric("Shape match (cropped)", f"{v.shape_match_score:.2f}")
        st.caption(
            "Holistic: full-image reasoning. Geometric: landmark-relative damage position/extent, "
            "computed in Python, not model judgment. Shape match: focused comparison on a "
            "cropped-and-scaled composite of just the damage regions. Combined score is the "
            "minimum of the three so none can dilute a genuine flag from the others."
        )

    other_shared_findings = [
        f for f in shared_findings if not f.type.startswith("consistency_submethod_")
    ]
    if other_shared_findings:
        st.write("**Cross-image findings** (identity/comparison checks, about both photos together):")
        for f in other_shared_findings:
            st.write(f"- `{f.type}` ({f.confidence:.2f}): {f.description}")


def _render_round(record: RoundRecord) -> None:
    st.markdown(
        f"#### Round {record.round_number} — {_decision_badge(record.supervisor_result.decision)} "
        f"(fraud confidence: {record.supervisor_result.fraud_confidence:.2f})",
        unsafe_allow_html=True,
    )
    red_col, blue_col = st.columns(2)

    with red_col:
        st.markdown("**🔴 Red Team**")
        st.caption(f"Tactic: `{record.attack.tactic}` · Technique: `{record.attack.technique}`")
        if record.payload.chat_messages:
            for message in record.payload.chat_messages:
                _render_chat_bubble(message)
        else:
            st.text(record.payload.chat_transcript)

    with blue_col:
        st.markdown("**🔵 Blue Team**")
        if record.sanitizer_result.injection_detected:
            st.error(f"Prompt injection detected: {record.sanitizer_result.flagged_phrases}")
        else:
            st.success(f"Sanitizer: no injection detected (chat risk {record.sanitizer_result.manipulation_risk_score:.2f})")
        if record.supervisor_result.reasons:
            st.write("**Reasons for decision:**", "; ".join(record.supervisor_result.reasons))

    st.divider()
    _render_evidence(record)


with tab_live:
    st.subheader("Live Red Team vs Blue Team Simulation")

    # A widget's session_state key can't be written to after that widget has already
    # been instantiated in the same script run -- so a randomize click can't set
    # tactic_select/technique_select directly and then rerun; it has to queue the pick
    # and apply it here, at the top of the NEXT run, before the selectboxes below exist.
    if "_pending_random_attack" in st.session_state:
        picked = st.session_state.pop("_pending_random_attack")
        st.session_state["tactic_select"] = picked.tactic
        st.session_state["technique_select"] = picked.technique

    attack_col1, attack_col2, attack_col3 = st.columns([2, 2, 1])
    with attack_col1:
        tactic = st.selectbox("Tactic", list(taxonomy.SOCIAL_ENGINEERING_TACTICS), key="tactic_select")
    with attack_col2:
        technique = st.selectbox("Technique", list(taxonomy.IMAGE_FORGERY_TECHNIQUES), key="technique_select")
    with attack_col3:
        st.write("")
        st.write("")
        random_pick = st.button("🎲 Randomize")

    if random_pick:
        rng = random.Random()
        st.session_state["_pending_random_attack"] = taxonomy.sample_attack(rng)
        st.rerun()

    control_col1, control_col2, control_col3 = st.columns(3)
    with control_col1:
        start_clicked = st.button("▶️ Start New Simulation", type="primary", width="stretch")
    with control_col2:
        next_clicked = st.button("⏭️ Run Next Round", width="stretch")
    with control_col3:
        auto_clicked = st.button("⏩ Auto-run to Cap", width="stretch")

    rounds = state.get_rounds()
    live_chat_slot = st.empty()

    if start_clicked:
        attack = AttackSpec(tactic=tactic, technique=technique)
        state.reset_rounds()
        on_message = _make_live_chat_callback(live_chat_slot)
        state.add_round(orchestrator.start_interactive(attack, on_message=on_message))
        live_chat_slot.empty()
        rounds = state.get_rounds()

    if next_clicked and rounds:
        on_message = _make_live_chat_callback(live_chat_slot)
        nxt = orchestrator.continue_interactive(rounds[-1], on_message=on_message)
        live_chat_slot.empty()
        if nxt is not None:
            state.add_round(nxt)
        else:
            st.info("Simulation already resolved (approved/escalated) or round cap reached.")
        rounds = state.get_rounds()

    if auto_clicked and rounds:
        while True:
            on_message = _make_live_chat_callback(live_chat_slot)
            nxt = orchestrator.continue_interactive(rounds[-1], on_message=on_message)
            live_chat_slot.empty()
            if nxt is None:
                break
            state.add_round(nxt)
            rounds = state.get_rounds()

    if not rounds:
        st.info("Pick a tactic/technique (or randomize) and click **Start New Simulation**.")
    else:
        for record in reversed(rounds):
            _render_round(record)
            st.divider()


with tab_manual:
    st.subheader("🧑‍⚖️ Try It Yourself — Live Fraud Triage")
    st.caption(
        "You chat with the real support bot yourself and upload two real photos -- detection "
        "signals update live on the right as the conversation happens. Then you decide: accept, "
        "escalate, ask for more evidence, or deny. Same Blue Team pipeline as everywhere else, "
        "just fed real typed/uploaded input instead of Red Team-generated input."
    )

    if st.button("🔄 Reset case", key="manual_reset"):
        state.reset_manual_session()
        st.rerun()

    manual_history = state.get_manual_history()
    manual_sanitizer_history = state.get_manual_sanitizer_history()
    manual_images = state.get_manual_images()
    manual_vision = state.get_manual_vision_result()
    manual_metadata = state.get_manual_metadata()
    manual_action = state.get_manual_action()
    case_closed = manual_action is not None

    manual_chat_col, manual_signal_col = st.columns([3, 2])

    with manual_chat_col:
        st.markdown("#### 💬 Dispute chat")
        chat_box = st.container(height=360)
        with chat_box:
            if not manual_history:
                st.caption("Type a message below as the customer to begin -- try a normal complaint, "
                           "or something manipulative, or an outright prompt-injection attempt.")
            for message in manual_history:
                _render_chat_bubble(message)

        typed = st.chat_input("Type as the customer...", key="manual_chat_input", disabled=case_closed)
        if typed:
            state.add_manual_message(ChatMessage(role="customer", content=typed))
            transcript_so_far = render_transcript(state.get_manual_history())
            with st.spinner("Screening message..."):
                sanitizer_result = sanitize(get_text_provider(session_api_key), transcript_so_far)
            state.add_manual_sanitizer_result(sanitizer_result)
            with st.spinner("Support bot is replying..."):
                bot_reply = orchestrator.support_bot.respond(state.get_manual_history())
            state.add_manual_message(ChatMessage(role="support_bot", content=bot_reply))
            st.rerun()

        st.markdown("#### 📎 Upload two photos (required)")
        uploaded_files = st.file_uploader(
            "Upload exactly two images of the claimed damage",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="manual_upload",
            disabled=case_closed,
        )
        if uploaded_files:
            if len(uploaded_files) != 2:
                st.warning(f"Exactly 2 images are required -- currently {len(uploaded_files)}.")
            else:
                # Metadata forensics needs the RAW bytes as uploaded -- getvalue() must
                # happen before/independent of PIL's PNG re-encode below, which would
                # otherwise strip EXIF before there's any chance to read it.
                raw_bytes_list = [f.getvalue() for f in uploaded_files]
                new_images = [
                    ImageSubmission(
                        angle=f"upload_{i + 1}",
                        data_b64=_png_b64_from_bytes(raw_bytes),
                        generation_strategy="human_uploaded",
                    )
                    for i, raw_bytes in enumerate(raw_bytes_list)
                ]
                if [img.data_b64 for img in new_images] != [img.data_b64 for img in (manual_images or [])]:
                    new_metadata = combine_metadata_reports([inspect_metadata(b) for b in raw_bytes_list])
                    # Run vision inspection immediately on upload rather than waiting for a
                    # separate button click -- a demo viewer uploads photos expecting the
                    # signals panel to react right away, the same way the chat risk score
                    # reacts the moment a message is sent.
                    with st.spinner("Running forensic vision inspection on the uploaded photos..."):
                        new_vision = inspect(
                            get_vision_provider(session_api_key),
                            [img.data_b64 for img in new_images],
                            "Defective merchandise (reason code 4853)",
                        )
                    state.set_manual_images(new_images)
                    state.set_manual_vision_result(new_vision)
                    state.set_manual_metadata(new_metadata)
                    manual_images = new_images
                    manual_vision = new_vision
                    manual_metadata = new_metadata
                    st.rerun()

        if manual_images:
            img_cols = st.columns(2)
            for col, img in zip(img_cols, manual_images):
                with col:
                    _show_image(img.data_b64, caption=img.angle)

    with manual_signal_col:
        st.markdown("#### 🚦 Live detection signals")

        combined_sanitizer = combine_sanitizer_history(manual_sanitizer_history)
        if not manual_sanitizer_history:
            st.info("Chat risk will appear once the conversation starts.")
        else:
            if combined_sanitizer.injection_detected:
                st.error("🚨 **Chat risk: HIGH** — prompt injection detected")
            elif combined_sanitizer.manipulation_risk_score >= 0.5:
                st.warning(f"⚠️ **Chat risk: ELEVATED** ({combined_sanitizer.manipulation_risk_score:.2f})")
            else:
                st.success(f"✅ **Chat risk: LOW** ({combined_sanitizer.manipulation_risk_score:.2f})")

            # Every turn's own screening result, shown regardless of risk level -- a
            # low-risk turn with a flagged phrase is still worth seeing, not just the
            # turns that happen to cross the ELEVATED/HIGH banner threshold above.
            customer_turns = [m for m in manual_history if m.role == "customer"]
            with st.expander("Turn-by-turn breakdown", expanded=True):
                for turn_number, (message, result) in enumerate(zip(customer_turns, manual_sanitizer_history), 1):
                    if result.injection_detected:
                        icon = "🚨"
                    elif result.manipulation_risk_score >= 0.5:
                        icon = "⚠️"
                    else:
                        icon = "✅"
                    preview = message.content if len(message.content) <= 90 else message.content[:87] + "..."
                    st.markdown(f"{icon} **Turn {turn_number}** (risk {result.manipulation_risk_score:.2f}) — _{preview}_")
                    if result.reason:
                        st.caption(result.reason)
                    if result.flagged_phrases:
                        st.caption("Flagged: " + ", ".join(result.flagged_phrases))
                    elif not result.reason:
                        st.caption("Nothing flagged.")

        st.divider()
        st.markdown("**🔬 Image forensics**")
        if not manual_images:
            st.info("Upload two images -- forensic analysis runs automatically the moment both are in.")
        else:
            if manual_vision:
                fcol1, fcol2, fcol3 = st.columns(3)
                fcol1.metric("Artifact", f"{manual_vision.artifact_score:.2f}")
                fcol2.metric("Angle consistency", f"{manual_vision.angle_consistency_score:.2f}")
                fcol3.metric("Detail consistency", f"{manual_vision.detail_consistency_score:.2f}")
                st.write("Semantic match:", "✅" if manual_vision.semantic_match else "❌")

                holistic_score = _finding_confidence(manual_vision.findings, "consistency_submethod_holistic")
                with st.expander(
                    f"Angle consistency breakdown — combined by MIN, "
                    f"{manual_vision.consistency_disagreement:.2f} disagreement across methods"
                ):
                    bcol1, bcol2, bcol3 = st.columns(3)
                    bcol1.metric("Holistic reasoning", f"{holistic_score:.2f}" if holistic_score is not None else "—")
                    bcol2.metric("Geometric (deterministic)", f"{manual_vision.geometric_consistency_score:.2f}")
                    bcol3.metric("Shape match (cropped)", f"{manual_vision.shape_match_score:.2f}")

                other_findings = [
                    f for f in manual_vision.findings if not f.type.startswith("consistency_submethod_")
                ]
                if other_findings:
                    st.write("**Findings:**")
                    for finding in other_findings:
                        st.write(f"- `{finding.type}` ({finding.confidence:.2f}): {finding.description}")
                if st.button("🔁 Re-run analysis", key="manual_reanalyze", disabled=case_closed):
                    with st.spinner("Re-running vision inspection..."):
                        manual_vision = inspect(
                            get_vision_provider(session_api_key),
                            [img.data_b64 for img in manual_images],
                            "Defective merchandise (reason code 4853)",
                        )
                    state.set_manual_vision_result(manual_vision)
                    st.rerun()
            else:
                st.caption("Analysis is running above -- results will appear here in a moment.")

        st.divider()
        st.markdown("**📄 Image metadata**")
        if not manual_metadata:
            st.caption("Runs automatically on upload -- no API call, pure file inspection.")
        else:
            if manual_metadata.has_c2pa_marker:
                st.error("🚨 C2PA content-credential marker found -- file declares AI generation/editing provenance.")
            elif manual_metadata.has_ai_digital_source_marker:
                st.error("🚨 IPTC DigitalSourceType marker found -- file declares AI-generated/edited content.")
            elif manual_metadata.has_camera_exif:
                camera = " ".join(filter(None, [manual_metadata.camera_make, manual_metadata.camera_model]))
                st.success(f"✅ Camera EXIF present" + (f" ({camera})" if camera else ""))
            else:
                st.warning("⚠️ No camera EXIF found (make/model/timestamp all missing)")
            for note in manual_metadata.notes:
                st.caption(note)

        st.divider()
        st.markdown("**🤖 AI recommendation**")
        if combined_sanitizer.injection_detected:
            recommendation = decide(combined_sanitizer, manual_vision, OrderMetadata(), manual_metadata)
            st.markdown(
                f"{_decision_badge(recommendation.decision)} (fraud confidence: {recommendation.fraud_confidence:.2f})",
                unsafe_allow_html=True,
            )
        elif manual_vision:
            recommendation = decide(combined_sanitizer, manual_vision, OrderMetadata(), manual_metadata)
            st.markdown(
                f"{_decision_badge(recommendation.decision)} (fraud confidence: {recommendation.fraud_confidence:.2f})",
                unsafe_allow_html=True,
            )
            if recommendation.reasons:
                st.caption("; ".join(recommendation.reasons))
        else:
            st.caption("Available once images are analyzed (or immediately if an injection is detected).")

        st.divider()
        st.markdown("**🧑‍⚖️ Your decision**")
        if manual_action:
            st.success(f"Case closed: **{manual_action}**")
        else:
            action_row1 = st.columns(2)
            if action_row1[0].button("✅ Accept", width="stretch", key="manual_accept"):
                state.set_manual_action("ACCEPTED")
                st.rerun()
            if action_row1[1].button("⚠️ Escalate", width="stretch", key="manual_escalate"):
                state.set_manual_action("ESCALATED")
                st.rerun()
            action_row2 = st.columns(2)
            if action_row2[0].button("📧 Ask for more evidence", width="stretch", key="manual_ask_evidence"):
                state.set_manual_action("REQUESTED_MORE_EVIDENCE")
                st.rerun()
            if action_row2[1].button("❌ Deny", width="stretch", key="manual_deny"):
                state.set_manual_action("DENIED")
                st.rerun()


with tab_batch:
    st.subheader("Batch Evaluation — Precision / Recall / F1 / AUC")
    st.caption(
        "Runs every tactic × technique combination (Red Team, no mutation) plus the canned "
        "legit fixtures through the Blue Team pipeline only, to measure detection efficacy "
        "and the false-positive rate on genuine claims."
    )

    if st.button("▶️ Run Batch Evaluation", type="primary", key="run_batch_now"):
        text_provider = get_text_provider(session_api_key)
        vision_provider = get_vision_provider(session_api_key)
        with st.spinner("Running batch evaluation across all attack combinations..."):
            results = run_batch(text_provider, vision_provider, red_team)
            state.set_batch_results(results)

    results = state.get_batch_results()

    if not results:
        st.info("Click **Run Batch Evaluation** to evaluate the Blue Team pipeline.")
    else:
        threshold = st.slider(
            "Decision threshold (fraud confidence ≥ threshold ⇒ flagged)",
            min_value=0.0,
            max_value=1.0,
            value=config.REJECT_ABOVE,
            step=0.01,
        )

        confusion = confusion_at_threshold(results, threshold)
        pr = precision_recall_f1(confusion)
        roc_points = roc_curve(results)
        auc_value = auc(roc_points)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision", f"{pr['precision']:.2f}")
        m2.metric("Recall", f"{pr['recall']:.2f}")
        m3.metric("F1", f"{pr['f1']:.2f}")
        m4.metric("False Positive Rate", f"{pr['false_positive_rate']:.2f}")
        st.metric("AUC (threshold-independent)", f"{auc_value:.3f}")

        st.write("**Confusion matrix**")
        st.table(
            {
                "": ["Predicted Fraud", "Predicted Legit"],
                "Actual Fraud": [confusion["tp"], confusion["fn"]],
                "Actual Legit": [confusion["fp"], confusion["tn"]],
            }
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[p["fpr"] for p in roc_points],
                y=[p["tpr"] for p in roc_points],
                mode="lines",
                name="Blue Team detector",
                line=dict(color="#1B9E77", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random baseline",
                line=dict(color="#9E9E9E", width=1, dash="dash"),
            )
        )
        fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=10, r=10, t=30, b=10),
            height=420,
        )
        st.plotly_chart(fig, width="stretch")

        with st.expander("Per-case results"):
            st.dataframe(
                [
                    {
                        "case_id": r.case_id,
                        "ground_truth": "fraud" if r.is_fraud_ground_truth else "legit",
                        "tactic": r.attack.tactic if r.attack else "-",
                        "technique": r.attack.technique if r.attack else "-",
                        "decision": r.supervisor_result.decision.value,
                        "fraud_confidence": r.supervisor_result.fraud_confidence,
                    }
                    for r in results
                ],
                width="stretch",
            )

        if st.button("💾 Export results for the .docx walkthrough"):
            provider_mode = "live" if is_live_mode(session_api_key) else "mock"
            path = export_batch_results(results, confusion, pr, auc_value, provider_mode=provider_mode)
            st.success(f"Exported to `{path}` — run `python -m docs.generate_walkthrough` to build the .docx.")


with tab_taxonomy:
    st.subheader("Attack Taxonomy")
    st.caption("The catalog the Red Team samples from (Live Simulation) or sweeps entirely (Batch Evaluation).")

    st.write("### Social-engineering tactics")
    st.table(
        {
            "Tactic": list(taxonomy.SOCIAL_ENGINEERING_TACTICS),
            "Description": list(taxonomy.SOCIAL_ENGINEERING_TACTICS.values()),
        }
    )

    st.write("### Image-forgery techniques")
    st.table(
        {
            "Technique": list(taxonomy.IMAGE_FORGERY_TECHNIQUES),
            "Description": list(taxonomy.IMAGE_FORGERY_TECHNIQUES.values()),
        }
    )

    total = len(taxonomy.SOCIAL_ENGINEERING_TACTICS) * len(taxonomy.IMAGE_FORGERY_TECHNIQUES)
    st.info(f"{total} tactic × technique combinations covered by Batch Evaluation.")


with tab_gallery:
    st.subheader("Example Gallery")
    st.caption(
        "Pre-generated, fully offline examples -- real Red Team vs Blue Team rounds captured "
        "earlier -- so there's always something to show regardless of live network/API status. "
        "Regenerate with `python -m scripts.generate_fraud_samples`."
    )

    gallery = load_example_gallery()
    if not gallery:
        st.info("No examples yet -- run `python -m scripts.generate_fraud_samples` to generate them.")
    else:
        for sequence in gallery:
            first = sequence[0]
            st.markdown(f"### {first.attack.tactic} × {first.attack.technique}")
            for record in sequence:
                _render_round(record)
                st.divider()
