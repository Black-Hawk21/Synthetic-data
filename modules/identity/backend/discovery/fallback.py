"""Deterministic, dependency-free attack-discovery engine (section 6). This
is the engine that ALWAYS works, with or without a local LLM -- it is what
`discovery/attack_discovery.py` falls back to when Ollama is unavailable.
"""
from __future__ import annotations

from backend.red_team.registry import get_attack, list_attacks

WEAK_RECALL_THRESHOLD = 0.85


def find_weak_attack_types(per_attack_type_recall: dict) -> list[str]:
    weak = [
        attack_type for attack_type, stats in per_attack_type_recall.items()
        if stats.get("recall", 1.0) < WEAK_RECALL_THRESHOLD
    ]
    # If nothing is weak yet (early iterations), target the hardest-by-design
    # attack types so the loop still has somewhere to push.
    if not weak:
        weak = [t for t in ["MULTI_SIGNAL_SYNTHETIC_IDENTITY", "FRAUD_RING", "IDENTITY_ATTRIBUTE_INCONSISTENCY"] if t in list_attacks()]
    return weak


def propose_hypotheses(per_attack_type_recall: dict, n: int = 5) -> list[dict]:
    weak_types = find_weak_attack_types(per_attack_type_recall)
    hypotheses = []
    for i, attack_type in enumerate(weak_types[:n]):
        try:
            strategy = get_attack(attack_type)
        except KeyError:
            continue
        recall = per_attack_type_recall.get(attack_type, {}).get("recall")
        reason = (
            f"Detector recall on {attack_type} is "
            f"{'unmeasured yet' if recall is None else f'{recall:.0%}'}, "
            "below the reliability bar. Pushing difficulty higher and blending "
            "in a second weak signal should probe whether the model is relying "
            "on a single dominant feature for this attack family."
        )
        hypotheses.append({
            "attack_type": attack_type,
            "reason": reason,
            "target_weakness": f"Underweights subtle variants of {attack_type.lower().replace('_', ' ')} when individual signals look near-legitimate.",
            "features_to_manipulate": strategy.features_affected,
            "difficulty": round(min(0.95, 0.75 + 0.05 * i), 2),
        })

    # Always include one true cross-family hybrid proposal
    if len(weak_types) >= 2:
        a, b = weak_types[0], weak_types[1]
        try:
            fa, fb = get_attack(a).features_affected, get_attack(b).features_affected
            hypotheses.append({
                "attack_type": f"{a}+{b}_HYBRID",
                "reason": f"Composing {a} and {b} tests whether the detector generalizes across attack families instead of memorizing per-family signatures.",
                "target_weakness": "Single-family overfitting in the current detector.",
                "features_to_manipulate": sorted(set(fa) | set(fb)),
                "difficulty": 0.85,
                "compose_of": [a, b],
            })
        except KeyError:
            pass

    return hypotheses
