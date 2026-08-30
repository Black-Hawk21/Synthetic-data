"""Adds one fresh, fully live, interactive-chat example to data/fraud_samples.json --
complementing the img2img-sophisticated Sony ULT WEAR entry (scripts/build_sony_example.py)
with a naive-generation attack that gets rejected in round 1, then escalates its
generation strategy (naive -> img2img conditioning) in round 2 after the Red Team
agent reacts to the rejection, demonstrating the adaptive-attacker feedback loop live.

Runs against whichever provider is currently configured (.env's ANTHROPIC_API_KEY /
IMAGE_PROVIDER); capped at 2 rounds to bound live API cost regardless of config.ROUND_CAP.

Usage: python -m scripts.build_live_demo_example
"""

import json
from pathlib import Path

from aegis.orchestrator import Orchestrator
from aegis.providers.factory import get_image_provider, get_text_provider, get_vision_provider, is_live_mode
from aegis.redteam.agent import RedTeamAgent
from aegis.schemas import AttackSpec, RoundRecord
from aegis.support_bot import SupportBotAgent

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "fraud_samples.json"
ATTACK = AttackSpec(tactic="urgency_authority_impersonation", technique="naive_independent_two_angle")
MAX_ROUNDS = 2


def main() -> None:
    print(f"Provider mode: {'LIVE' if is_live_mode() else 'MOCK'}")
    text_provider = get_text_provider()
    vision_provider = get_vision_provider()
    image_provider = get_image_provider()
    red_team = RedTeamAgent(text_provider, image_provider)
    support_bot = SupportBotAgent(text_provider)
    orchestrator = Orchestrator(red_team, text_provider, vision_provider, round_cap=MAX_ROUNDS, support_bot=support_bot)

    records = [orchestrator.start_interactive(ATTACK)]
    print(
        f"Round 1: decision={records[0].supervisor_result.decision.value} "
        f"confidence={records[0].supervisor_result.fraud_confidence}"
    )
    while len(records) < MAX_ROUNDS:
        nxt = orchestrator.continue_interactive(records[-1])
        if nxt is None:
            break
        records.append(nxt)
        print(
            f"Round {len(records)}: decision={records[-1].supervisor_result.decision.value} "
            f"confidence={records[-1].supervisor_result.fraud_confidence}"
        )

    sequences = json.loads(OUTPUT_PATH.read_text()) if OUTPUT_PATH.exists() else []
    sequences.append([json.loads(r.model_dump_json()) for r in records])
    OUTPUT_PATH.write_text(json.dumps(sequences, indent=2))
    print(f"Appended {len(records)}-round sequence to {OUTPUT_PATH} ({len(sequences)} examples total)")


if __name__ == "__main__":
    main()
