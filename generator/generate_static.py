"""
Batch-generates static (single-shot) phishing/smishing samples and matched
legit samples, writing them to data/generated/phishing_synthetic.jsonl in
the schema defined in schema.py.

Usage:
    export GROQ_API_KEY=gsk_...
    python generate_static.py --n-per-cell 15

--n-per-cell controls how many samples per (subtype, channel, difficulty)
cell for fraud, and per (kind, channel) cell for legit. Total fraud samples
= n_per_cell * len(subtypes) * len(channels) * len(difficulties).
Start small (5-10) to sanity check output before scaling up.
"""

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))

from schema import Sample, VALID_SUBTYPES
from personas import generate_personas
from templates import build_prompts, build_legit_prompts, SUBTYPE_TEMPLATES, LEGIT_TEMPLATES
from llm_client import generate_text, MODEL

CHANNELS = ["email", "sms"]
DIFFICULTIES = ["naive", "moderate", "adaptive"]
FRAUD_SUBTYPES = [s for s in VALID_SUBTYPES if s != "legit" and s in SUBTYPE_TEMPLATES]

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "generated", "phishing_synthetic.jsonl")


def _gen_fraud_sample(persona, subtype, channel, difficulty):
    system, user = build_prompts(subtype, channel, difficulty, persona)
    text = generate_text(system, user)
    sample = Sample(
        text=text,
        label=1,
        channel=channel,
        attack_subtype=subtype,
        difficulty_tier=difficulty,
        persona=persona,
        generation_model=MODEL,
    )
    sample.validate()
    return sample


def _gen_legit_sample(persona, kind, channel):
    system, user = build_legit_prompts(kind, channel, persona)
    text = generate_text(system, user)
    sample = Sample(
        text=text,
        label=0,
        channel=channel,
        attack_subtype="legit",
        difficulty_tier="naive",
        persona=persona,
        generation_model=MODEL,
    )
    sample.validate()
    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-cell", type=int, default=10,
                         help="samples per (subtype, channel, difficulty) cell for fraud")
    parser.add_argument("--workers", type=int, default=3,
                         help="parallel API calls (kept modest -- Groq free tier is ~25-30 req/min, "
                              "enforced client-side in llm_client.py regardless of this number)")
    parser.add_argument("--out", type=str, default=OUT_PATH)
    args = parser.parse_args()

    jobs = []
    persona_pool = generate_personas(max(50, args.n_per_cell * 4))

    # fraud jobs
    for subtype in FRAUD_SUBTYPES:
        for channel in CHANNELS:
            for difficulty in DIFFICULTIES:
                for i in range(args.n_per_cell):
                    persona = persona_pool[(hash((subtype, channel, difficulty, i))) % len(persona_pool)]
                    jobs.append(("fraud", persona, subtype, channel, difficulty))

    # legit jobs (roughly balance class sizes: same total count, spread over kinds/channels)
    n_legit = len(jobs)
    legit_kinds = list(LEGIT_TEMPLATES.keys())
    for i in range(n_legit):
        persona = persona_pool[i % len(persona_pool)]
        kind = legit_kinds[i % len(legit_kinds)]
        channel = CHANNELS[i % len(CHANNELS)]
        jobs.append(("legit", persona, kind, channel))

    print(f"Generating {len(jobs)} samples ({n_legit} fraud + {n_legit} legit) "
          f"with {args.workers} parallel workers...")
    print(f"Writing to {args.out} incrementally as each sample completes -- if you interrupt "
          f"this run (Ctrl+C or closing the terminal), everything written so far is safely on "
          f"disk already, nothing is lost.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    results = []
    errors = 0
    exact_dupes = 0
    from collections import Counter
    failed_subtypes = Counter()

    write_lock = threading.Lock()
    seen_text = set()

    with open(args.out, "w") as out_f, ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for job in jobs:
            if job[0] == "fraud":
                _, persona, subtype, channel, difficulty = job
                fut = executor.submit(_gen_fraud_sample, persona, subtype, channel, difficulty)
            else:
                _, persona, kind, channel = job
                fut = executor.submit(_gen_legit_sample, persona, kind, channel)
            futures[fut] = job

        for i, fut in enumerate(as_completed(futures), 1):
            job = futures[fut]
            try:
                sample = fut.result()
                key = sample.text.strip().lower()
                with write_lock:
                    if key in seen_text:
                        exact_dupes += 1
                    else:
                        seen_text.add(key)
                        results.append(sample)
                        out_f.write(json.dumps(sample.to_dict()) + "\n")
                        out_f.flush()  # ensure it's actually on disk, not just buffered, in
                                        # case the process is killed rather than exiting cleanly
            except Exception as e:  # noqa: BLE001
                errors += 1
                kind_or_subtype = job[2]  # subtype for fraud jobs, kind for legit jobs
                channel = job[3]
                difficulty = job[4] if job[0] == "fraud" else "n/a"
                failed_subtypes[kind_or_subtype] += 1
                print(f"  [error] {job[0]}/{kind_or_subtype}/{channel}/{difficulty} failed: {e}",
                      file=sys.stderr)
            if i % 20 == 0:
                print(f"  ...{i}/{len(jobs)} done ({len(results)} written so far)")

    if failed_subtypes:
        print(f"\nFailures by subtype: {dict(failed_subtypes)}")
        print("If one subtype dominates the failures, its wording in templates.py is likely "
              "triggering refusals more than the others -- worth rewording that specific "
              "template rather than just raising retries.")

    near_dupes = _count_near_duplicates(results)

    print(f"\nWrote {len(results)} samples to {args.out} ({errors} generation errors, "
          f"{exact_dupes} exact duplicates dropped)")
    if near_dupes:
        print(f"  [note] found {near_dupes} near-duplicate pairs (>90% similar text within the "
              f"same subtype/channel/difficulty group) -- these were kept (not auto-removed, since "
              f"some templated phrasing overlap is expected at 'naive' difficulty), but consider "
              f"raising --n-per-cell diversity or reviewing manually if this number is high.")
    print(f"Fraud: {sum(1 for s in results if s.label == 1)} | "
          f"Legit: {sum(1 for s in results if s.label == 0)}")


def _count_near_duplicates(samples, threshold: float = 0.9, max_group_size: int = 60):
    """Within each (subtype, channel, difficulty) group, counts pairs of
    messages that are near-identical (difflib similarity ratio > threshold).
    Skips groups larger than max_group_size to keep this O(n^2) check cheap --
    at hackathon batch sizes this is just a diagnostic, not a hard filter."""
    from collections import defaultdict
    from difflib import SequenceMatcher

    groups = defaultdict(list)
    for s in samples:
        groups[(s.attack_subtype, s.channel, s.difficulty_tier)].append(s.text)

    total_pairs = 0
    for texts in groups.values():
        if len(texts) < 2 or len(texts) > max_group_size:
            continue
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if SequenceMatcher(None, texts[i], texts[j]).ratio() > threshold:
                    total_pairs += 1
    return total_pairs


if __name__ == "__main__":
    main()
