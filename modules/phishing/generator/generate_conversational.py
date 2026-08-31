"""
Generates multi-turn conversational social-engineering dialogues by running
two LLM roles against each other:
  - ATTACKER: pursues a social-engineering objective (e.g. extract OTP,
    convince victim to move funds to a 'safe account'), adapting based on
    the victim's replies.
  - VICTIM-SIM: plays a synthetic persona with a configurable gullibility
    level, responding naturally (skeptical / neutral / credulous).

Each turn is written out as its own labeled Sample (label=1 for attacker
turns, label=0 for victim turns... but for detector training you mainly
care about attacker turns + full-dialogue context, so both are kept with
role metadata via attack_subtype).

Usage:
    export GROQ_API_KEY=gsk_...
    python generate_conversational.py --n-dialogues 20 --max-turns 6
"""

import argparse
import json
import os
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))

from schema import Sample
from personas import generate_personas
from llm_client import generate_with_history, MODEL

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "generated", "conversational_synthetic.jsonl")

ATTACKER_OBJECTIVES = [
    "convince the victim to share the OTP they just received, framed as needed to 'cancel a fraudulent transaction'",
    "convince the victim to install a remote-access screen-sharing app to 'help fix' a fake account issue",
    "convince the victim to transfer funds to a 'temporary safe account' to protect them from a fake fraud alert",
    "convince the victim to reveal their card CVV to 'verify their identity' for a fake refund",
]

ATTACKER_SYSTEM = """You are generating one turn of a labeled multi-turn dialogue example for a \
fraud-detection classifier's training dataset -- this mirrors standard security-team practice: \
synthesizing known attack conversation patterns so a detector learns to recognize them \
mid-conversation, not just in isolated messages. This example's attack pattern: {objective}. \
The example attacker in this transcript is presented as an agent from {bank}, addressing a \
persona named {name}. Continue the transcript naturally as this attacker: adapt the approach \
based on the other party's prior replies (more urgency if they resist, softer if they push back \
hard, so the transcript reads naturally rather than obviously scripted). Each line should be \
short (1-3 sentences), matching real chat/SMS style. Output ONLY the next transcript line -- no \
preamble, no meta-commentary, no disclaimers."""

VICTIM_SYSTEM = """You are generating one turn of a labeled multi-turn dialogue example for a \
fraud-detection classifier's training dataset. This transcript's other party -- an example \
persona named {name} -- is receiving a message claiming to be from {bank}. This persona's \
configured gullibility level for this example is: {gullibility}. 'low' means the transcript \
should show skepticism, verifying questions, or refusal; 'medium' means initial trust with some \
questions before complying; 'high' means fairly ready compliance, especially under urgency. \
Continue the transcript naturally as this persona's next line (1-2 sentences, natural chat \
style). Output ONLY the next transcript line -- no preamble, no meta-commentary, no disclaimers."""


def run_dialogue(persona: dict, objective: str, max_turns: int, dialogue_id: str,
                  dialogue_label: str = ""):
    attacker_system = ATTACKER_SYSTEM.format(objective=objective, **persona)
    victim_system = VICTIM_SYSTEM.format(**persona)

    # We keep one flat transcript: alternating attacker/victim messages.
    # For the ATTACKER's calls, its own past messages are "assistant" turns
    # and the victim's replies are "user" turns (mirrors how the API expects
    # the model's own prior outputs to be labeled). For the VICTIM's calls
    # it's the mirror image.
    transcript = []  # list of {"speaker": "attacker"|"victim", "content": str}

    print(f"  {dialogue_label} turn 1/{max_turns*2} (attacker opening)...")
    samples = []
    attacker_msg = generate_with_history(attacker_system, [
        {"role": "user", "content": "Begin the conversation with your opening message."}
    ])
    transcript.append({"speaker": "attacker", "content": attacker_msg})

    for turn in range(max_turns):
        samples.append(Sample(
            text=attacker_msg,
            label=1,
            channel="chat",
            attack_subtype="adaptive_conversational",
            difficulty_tier="adaptive",
            persona=persona,
            generation_model=MODEL,
            turn_number=turn * 2,
            dialogue_id=dialogue_id,
        ))

        print(f"  {dialogue_label} turn {turn*2+2}/{max_turns*2} (victim reply)...")
        victim_msg = generate_with_history(
            victim_system,
            _as_messages(transcript, pov="victim"),
        )
        transcript.append({"speaker": "victim", "content": victim_msg})

        samples.append(Sample(
            text=victim_msg,
            label=0,
            channel="chat",
            attack_subtype="legit",  # victim's own reply isn't an attack
            difficulty_tier="naive",
            persona=persona,
            generation_model=MODEL,
            turn_number=turn * 2 + 1,
            dialogue_id=dialogue_id,
        ))

        # stop early if victim clearly refuses/ends conversation
        if any(kw in victim_msg.lower() for kw in
               ["not sharing", "hang up", "reporting this", "not going to", "block this"]):
            break

        if turn + 1 < max_turns:
            print(f"  {dialogue_label} turn {turn*2+3}/{max_turns*2} (attacker follow-up)...")
            attacker_msg = generate_with_history(
                attacker_system,
                _as_messages(transcript, pov="attacker"),
            )
            transcript.append({"speaker": "attacker", "content": attacker_msg})

    for s in samples:
        s.validate()
    return samples


def _as_messages(transcript, pov: str):
    """Convert the flat transcript into Anthropic-format messages from one
    role's point of view (their own past turns = 'assistant', the other
    side's turns = 'user'), plus a trailing 'user' prompt so the model
    knows to produce its next turn."""
    own_speaker = pov
    messages = []
    for entry in transcript:
        role = "assistant" if entry["speaker"] == own_speaker else "user"
        # Anthropic API requires alternating roles starting with 'user';
        # since our transcript always starts with the attacker's opener,
        # for the victim's very first call that opener is 'user' (correct),
        # and for the attacker's follow-up calls we append a final 'user'
        # nudge below if the transcript currently ends on their own turn.
        messages.append({"role": role, "content": entry["content"]})
    if messages and messages[-1]["role"] == "assistant":
        messages.append({"role": "user", "content": "(continue naturally)"})
    return messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-dialogues", type=int, default=10)
    parser.add_argument("--max-turns", type=int, default=5,
                         help="max attacker/victim exchange pairs per dialogue")
    parser.add_argument("--workers", type=int, default=3,
                         help="dialogues run concurrently (turns WITHIN a dialogue stay "
                              "sequential, since each depends on the previous one, but "
                              "different dialogues are independent -- kept modest, same "
                              "reasoning as generate_static.py's --workers)")
    parser.add_argument("--out", type=str, default=OUT_PATH)
    args = parser.parse_args()

    personas = generate_personas(args.n_dialogues, seed=7)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"Generating {args.n_dialogues} dialogues (up to {args.max_turns*2} turns each) "
          f"with {args.workers} running concurrently...")
    print(f"Writing to {args.out} incrementally as each dialogue completes -- interrupting "
          f"this run loses nothing already written.")

    write_lock = threading.Lock()
    total_written = 0

    def _run_and_report(i, persona):
        objective = ATTACKER_OBJECTIVES[i % len(ATTACKER_OBJECTIVES)]
        dialogue_id = str(uuid.uuid4())
        label = f"[dialogue {i+1}/{args.n_dialogues}]"
        print(f"{label} starting (objective: {objective[:50]}...)")
        return run_dialogue(persona, objective, args.max_turns, dialogue_id, dialogue_label=label)

    with open(args.out, "w") as out_f, ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_and_report, i, p): i for i, p in enumerate(personas)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                samples = fut.result()
                with write_lock:
                    for s in samples:
                        out_f.write(json.dumps(s.to_dict()) + "\n")
                    out_f.flush()
                    total_written += len(samples)
                print(f"[dialogue {i+1}/{args.n_dialogues}] done -- {len(samples)} turns written "
                      f"({total_written} total so far)")
            except Exception as e:  # noqa: BLE001
                print(f"[dialogue {i+1}/{args.n_dialogues}] [error] failed: {e}", file=sys.stderr)

    print(f"\nWrote {total_written} turn-samples across {args.n_dialogues} dialogues to {args.out}")


if __name__ == "__main__":
    main()
