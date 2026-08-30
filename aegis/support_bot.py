"""The customer-facing chat agent -- the surface a real attacker actually targets.

Deliberately NOT part of the trusted Blue Team boundary (aegis/blueteam/). Even if
an attacker fully jailbreaks this bot into saying "yes, your refund is approved!",
that text is irrelevant to the outcome: sanitizer.py and supervisor.py never trust
anything this bot says, only their own classification of the raw transcript plus
the vision inspector's structured findings.
"""

from aegis.chat_prompts import (
    build_support_bot_ack_prompt,
    build_support_bot_ask_prompt,
    build_support_bot_general_prompt,
)
from aegis.schemas import ChatMessage


class SupportBotAgent:
    def __init__(self, text_provider) -> None:
        self._text = text_provider

    def ask_for_photos(self, history: list[ChatMessage]) -> str:
        system, user = build_support_bot_ask_prompt(history)
        return self._text.generate_text(system, user)

    def acknowledge(self, history: list[ChatMessage]) -> str:
        system, user = build_support_bot_ack_prompt(history)
        return self._text.generate_text(system, user)

    def respond(self, history: list[ChatMessage]) -> str:
        """Used by the interactive 'Try It Yourself' tab: a general-purpose reply to
        whatever the human just typed, rather than the simulated chat's fixed script."""
        system, user = build_support_bot_general_prompt(history)
        return self._text.generate_text(system, user)
