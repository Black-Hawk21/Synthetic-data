from typing import Callable

from aegis import config
from aegis.attacks import taxonomy
from aegis.blueteam.sanitizer import sanitize
from aegis.blueteam.supervisor import decide
from aegis.blueteam.vision_inspector import inspect
from aegis.redteam.agent import RedTeamAgent
from aegis.schemas import (
    AttackSpec,
    ChatMessage,
    Decision,
    DisputePayload,
    RoundRecord,
    SanitizerResult,
    SupervisorResult,
    VisionInspectionResult,
)
from aegis.support_bot import SupportBotAgent


def evaluate_dispute(
    text_provider, vision_provider, payload: DisputePayload
) -> tuple[SanitizerResult, VisionInspectionResult | None, SupervisorResult]:
    """Runs the Blue Team pipeline (sanitize -> inspect -> decide) on a single payload.
    Shared by the live round loop below and by Batch Evaluation (aegis/metrics.py),
    which calls this directly without any Red Team mutation."""
    sanitizer_result = sanitize(text_provider, payload.chat_transcript)

    vision_result = None
    if not sanitizer_result.injection_detected:
        images_b64 = [img.data_b64 for img in payload.images]
        vision_result = inspect(vision_provider, images_b64, payload.claimed_reason)

    supervisor_result = decide(sanitizer_result, vision_result, payload.order_metadata)
    return sanitizer_result, vision_result, supervisor_result


class Orchestrator:
    def __init__(
        self,
        red_team: RedTeamAgent,
        text_provider,
        vision_provider,
        round_cap: int = config.ROUND_CAP,
        support_bot: SupportBotAgent | None = None,
    ) -> None:
        self.red_team = red_team
        self.text_provider = text_provider
        self.vision_provider = vision_provider
        self.round_cap = round_cap
        self.support_bot = support_bot or SupportBotAgent(text_provider)

    def start(self, attack: AttackSpec) -> RoundRecord:
        payload = self.red_team.generate_initial_payload(attack)
        return self._build_record(1, attack, payload)

    def continue_round(self, previous_record: RoundRecord) -> RoundRecord | None:
        if previous_record.supervisor_result.decision != Decision.REJECT:
            return None
        if previous_record.round_number >= self.round_cap:
            return None
        payload = self.red_team.mutate(
            previous_record.payload, previous_record.attack, previous_record.supervisor_result
        )
        return self._build_record(previous_record.round_number + 1, previous_record.attack, payload)

    def run_to_cap(self, attack: AttackSpec) -> list[RoundRecord]:
        records = [self.start(attack)]
        while (next_record := self.continue_round(records[-1])) is not None:
            records.append(next_record)
        return records

    # ---- interactive chat (Live Simulation tab) ------------------------
    # Same round semantics as start/continue_round above, but the chat_transcript is
    # built turn-by-turn (customer <-> support bot) instead of one one-shot call, and
    # on_message fires after each individual message so the UI can render it live.

    def start_interactive(
        self, attack: AttackSpec, on_message: Callable[[ChatMessage], None] | None = None
    ) -> RoundRecord:
        history: list[ChatMessage] = []
        add = self._make_adder(history, on_message)

        add("customer", self.red_team.opening_message(attack))
        add("support_bot", self.support_bot.ask_for_photos(history))
        add("customer", self.red_team.followup_message(attack, history))
        add("support_bot", self.support_bot.acknowledge(history))

        strategy = taxonomy.initial_generation_strategy(attack.technique)
        payload = self.red_team.build_payload_from_chat(attack, history, strategy, mutation_suffix="")
        return self._build_record(1, attack, payload)

    def continue_interactive(
        self, previous_record: RoundRecord, on_message: Callable[[ChatMessage], None] | None = None
    ) -> RoundRecord | None:
        if previous_record.supervisor_result.decision != Decision.REJECT:
            return None
        if previous_record.round_number >= self.round_cap:
            return None

        attack = previous_record.attack
        history: list[ChatMessage] = []
        add = self._make_adder(history, on_message)

        add("customer", self.red_team.resubmission_message(attack))
        add("support_bot", self.support_bot.acknowledge(history))

        strategy, mutation_suffix = self.red_team.next_strategy_and_suffix(
            previous_record.payload, previous_record.supervisor_result
        )
        payload = self.red_team.build_payload_from_chat(attack, history, strategy, mutation_suffix)
        return self._build_record(previous_record.round_number + 1, attack, payload)

    @staticmethod
    def _make_adder(history: list[ChatMessage], on_message: Callable[[ChatMessage], None] | None):
        callback = on_message or (lambda _m: None)

        def add(role: str, text: str) -> None:
            message = ChatMessage(role=role, content=text)
            history.append(message)
            callback(message)

        return add

    def _build_record(self, round_number: int, attack: AttackSpec, payload: DisputePayload) -> RoundRecord:
        sanitizer_result, vision_result, supervisor_result = evaluate_dispute(
            self.text_provider, self.vision_provider, payload
        )
        return RoundRecord(
            round_number=round_number,
            attack=attack,
            payload=payload,
            sanitizer_result=sanitizer_result,
            vision_result=vision_result,
            supervisor_result=supervisor_result,
        )
