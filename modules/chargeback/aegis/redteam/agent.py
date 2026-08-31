from aegis.attacks import taxonomy
from aegis.chat_prompts import (
    build_customer_followup_prompt,
    build_customer_opening_prompt,
    build_customer_resubmission_prompt,
    render_transcript,
)
from aegis.redteam.prompts import (
    CROSS_ANGLE_ESCALATION_KEYWORDS,
    MUTATION_KEYWORD_RULES,
    build_chat_prompt,
    build_image_prompt,
    build_mutation_prompt,
)
from aegis.schemas import AttackSpec, ChatMessage, DisputePayload, ImageSubmission, SupervisorResult

ANGLES = ("front", "45deg_side")


class RedTeamAgent:
    def __init__(self, text_provider, image_provider) -> None:
        self._text = text_provider
        self._image = image_provider

    def generate_initial_payload(self, attack: AttackSpec) -> DisputePayload:
        system, user = build_chat_prompt(attack)
        chat_transcript = self._text.generate_text(system, user)
        strategy = taxonomy.initial_generation_strategy(attack.technique)
        images = self._generate_angle_images(attack, strategy, mutation_suffix="")
        return DisputePayload(chat_transcript=chat_transcript, images=images)

    def mutate(
        self,
        previous_payload: DisputePayload,
        attack: AttackSpec,
        supervisor_result: SupervisorResult,
    ) -> DisputePayload:
        system, user = build_mutation_prompt(previous_payload.chat_transcript, supervisor_result.findings)
        chat_transcript = self._text.generate_text(system, user)

        strategy, mutation_suffix = self.next_strategy_and_suffix(previous_payload, supervisor_result)
        images = self._generate_angle_images(attack, strategy, mutation_suffix)
        return DisputePayload(chat_transcript=chat_transcript, images=images)

    def next_strategy_and_suffix(
        self, previous_payload: DisputePayload, supervisor_result: SupervisorResult
    ) -> tuple[str, str]:
        # Substring-matched against each finding's own type+description text rather
        # than an exact finding.type lookup -- see the comment above
        # MUTATION_KEYWORD_RULES in aegis/redteam/prompts.py for why: the live vision
        # inspector's five independent checks each invent their own freeform type
        # labels, so an exact match against a small fixed vocabulary essentially never
        # fires against real model output.
        findings_text = " ".join(f"{f.type} {f.description}" for f in supervisor_result.findings).lower()
        current_strategy = previous_payload.images[0].generation_strategy
        # A cross-angle/geometric hit is the trigger to escalate sophistication:
        # naive independent generation gets abandoned in favor of img2img conditioning.
        strategy = (
            "img2img_conditioned"
            if any(keyword in findings_text for keyword in CROSS_ANGLE_ESCALATION_KEYWORDS)
            else current_strategy
        )
        phrases: list[str] = []
        for keyword, phrase in MUTATION_KEYWORD_RULES:
            if keyword in findings_text and phrase not in phrases:
                phrases.append(phrase)
        return strategy, " ".join(phrases)

    # ---- interactive chat (Live Simulation tab) ------------------------
    # Each method below generates exactly one next customer message, driven by a
    # fixed turn structure in orchestrator.py, instead of one one-shot transcript --
    # so the UI can render real chat bubbles as each message actually gets generated.

    def opening_message(self, attack: AttackSpec) -> str:
        system, user = build_customer_opening_prompt(attack)
        return self._text.generate_text(system, user)

    def followup_message(self, attack: AttackSpec, history: list[ChatMessage]) -> str:
        system, user = build_customer_followup_prompt(attack, history)
        return self._text.generate_text(system, user)

    def resubmission_message(self, attack: AttackSpec) -> str:
        system, user = build_customer_resubmission_prompt(attack)
        return self._text.generate_text(system, user)

    def build_payload_from_chat(
        self,
        attack: AttackSpec,
        history: list[ChatMessage],
        strategy: str,
        mutation_suffix: str,
    ) -> DisputePayload:
        images = self._generate_angle_images(attack, strategy, mutation_suffix)
        return DisputePayload(
            chat_transcript=render_transcript(history), chat_messages=history, images=images
        )

    def _generate_angle_images(
        self, attack: AttackSpec, strategy: str, mutation_suffix: str
    ) -> list[ImageSubmission]:
        front_prompt = build_image_prompt(attack, ANGLES[0], mutation_suffix)
        front_b64 = self._image.generate_image(front_prompt)

        side_prompt = build_image_prompt(attack, ANGLES[1], mutation_suffix)
        reference = front_b64 if strategy == "img2img_conditioned" else None
        side_b64 = self._image.generate_image(side_prompt, reference_b64=reference)

        return [
            ImageSubmission(angle=ANGLES[0], data_b64=front_b64, generation_strategy=strategy),
            ImageSubmission(angle=ANGLES[1], data_b64=side_b64, generation_strategy=strategy),
        ]
