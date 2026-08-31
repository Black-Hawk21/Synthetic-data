"""One-off generator for data/fraud_samples.json -- a curated, offline gallery of
real Red Team vs Blue Team rounds, one per image-forgery technique, so the Example
Gallery tab always has something to show regardless of live network/API status at
demo time. Runs against whichever provider is currently configured in .env (live or
mock); re-run any time to refresh with new live-generated content.

Usage: python -m scripts.generate_fraud_samples
"""

import json
from pathlib import Path

from aegis.orchestrator import Orchestrator
from aegis.providers.factory import get_image_provider, get_text_provider, get_vision_provider, is_live_mode
from aegis.redteam.agent import RedTeamAgent
from aegis.schemas import AttackSpec, RoundRecord
from aegis.support_bot import SupportBotAgent

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "fraud_samples.json"

# One (tactic, technique, max_rounds) per image-forgery technique -- covers every
# technique and every tactic at least once. naive_independent_two_angle is paired
# with direct_prompt_injection and given 3 rounds: round 1 is a guaranteed REJECT
# (injection short-circuit), so continuing it captures the Red Team abandoning the
# caught tactic and resubmitting normally -- the adversarial-feedback-loop "wow" arc,
# as a static example that doesn't depend on live scoring variance to reproduce.
EXAMPLES: list[tuple[str, str, int]] = [
    ("urgency_authority_impersonation", "diffusion_generated_damage", 1),
    ("fabricated_policy_citation", "lighting_shadow_inconsistency", 1),
    ("empathy_hardship_framing", "sensor_noise_exif_spoofing", 1),
    ("urgency_authority_impersonation", "spliced_edited_photo", 1),
    ("direct_prompt_injection", "naive_independent_two_angle", 3),
    ("fabricated_policy_citation", "img2img_conditioned_two_angle", 1),
]


def _generate_sequence(orchestrator: Orchestrator, attack: AttackSpec, max_rounds: int) -> list[RoundRecord]:
    records = [orchestrator.start_interactive(attack)]
    while len(records) < max_rounds:
        next_record = orchestrator.continue_interactive(records[-1])
        if next_record is None:
            break
        records.append(next_record)
    return records


def main() -> None:
    print(f"Provider mode: {'LIVE' if is_live_mode() else 'MOCK'}")
    text_provider = get_text_provider()
    vision_provider = get_vision_provider()
    image_provider = get_image_provider()
    red_team = RedTeamAgent(text_provider, image_provider)
    support_bot = SupportBotAgent(text_provider)
    orchestrator = Orchestrator(red_team, text_provider, vision_provider, round_cap=3, support_bot=support_bot)

    sequences = []
    for tactic, technique, max_rounds in EXAMPLES:
        attack = AttackSpec(tactic=tactic, technique=technique)
        print(f"Generating: {tactic} x {technique} (up to {max_rounds} round(s))...")
        sequence = _generate_sequence(orchestrator, attack, max_rounds)
        print(f"  -> {len(sequence)} round(s), final decision: {sequence[-1].supervisor_result.decision.value}")
        sequences.append([json.loads(record.model_dump_json()) for record in sequence])

    OUTPUT_PATH.write_text(json.dumps(sequences, indent=2))
    print(f"Wrote {len(sequences)} examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
