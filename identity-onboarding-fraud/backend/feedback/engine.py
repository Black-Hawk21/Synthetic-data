"""The adversarial / hard-negative closed loop (section 14) -- the heart of
this project: generate attacks -> detect -> find what got missed -> analyze
why -> generate harder attacks targeting that weakness -> retrain -> repeat.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from backend.blue_team.evaluate import compute_metrics, per_attack_type_recall
from backend.blue_team.predict import ModelBundle, score_dataframe
from backend.blue_team.train import train_and_evaluate
from backend.discovery.attack_discovery import discover_attack_hypotheses
from backend.graph.features import calculate_graph_features
from backend.red_team.registry import get_attack, list_attacks

logger = logging.getLogger(__name__)


def evaluate_attacks(df: pd.DataFrame, bundle: ModelBundle, threshold: float = 0.5) -> pd.DataFrame:
    """Score every row in `df` with the given model bundle. Returns a copy
    of df with `fraud_probability` and `predicted_fraud` columns added."""
    out = df.copy()
    out["fraud_probability"] = score_dataframe(df, bundle)
    out["predicted_fraud"] = (out["fraud_probability"] >= threshold).astype(int)
    return out


def find_false_negatives(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Fraud rows the detector scored below the decision threshold."""
    return scored_df[(scored_df["is_fraud"] == 1) & (scored_df["predicted_fraud"] == 0)].copy()


def analyze_detector_weaknesses(scored_df: pd.DataFrame, top_n: int = 5) -> dict:
    """Human-readable + structured weakness analysis of the current model,
    grounded in ACTUAL false negatives (never hard-coded, section 32)."""
    fn = find_false_negatives(scored_df)
    if fn.empty:
        return {"weak_attack_types": [], "false_negative_count": 0, "summary": "No false negatives found on this batch.", "recall_by_attack_type": {}}

    per_type = per_attack_type_recall(scored_df, scored_df["fraud_probability"].values)
    weakest = sorted(per_type.items(), key=lambda kv: kv[1]["recall"])[:top_n]

    fn_summary = fn.groupby("attack_type").size().sort_values(ascending=False)
    narrative_bits = []
    for attack_type, count in fn_summary.items():
        sub = fn[fn["attack_type"] == attack_type]
        narrative_bits.append(
            f"{attack_type}: {count} missed (avg suspicious_cluster_score={sub['suspicious_cluster_score'].mean():.2f}, "
            f"avg device_reuse_count={sub['device_reuse_count'].mean():.1f})"
        )

    summary = (
        f"Detector missed {len(fn)} of {int(scored_df['is_fraud'].sum())} fraud cases in this batch. "
        f"Weakest attack types by recall: {', '.join(t for t, _ in weakest)}. "
        + " | ".join(narrative_bits[:top_n])
    )

    return {
        "weak_attack_types": [t for t, _ in weakest],
        "false_negative_count": int(len(fn)),
        "recall_by_attack_type": per_type,
        "summary": summary,
    }


def _compose_attack_batch(attack_type_a: str, attack_type_b: str, n: int, difficulty: float, seed: int | None) -> pd.DataFrame:
    """Apply two attacks' mutations sequentially to the same base
    population -- used for LLM/rule-proposed cross-family hybrid
    hypotheses."""
    strat_a = get_attack(attack_type_a)
    strat_b = get_attack(attack_type_b)
    rng = np.random.default_rng(seed)
    from backend.data.generator import generate_legitimate_applicants
    base = generate_legitimate_applicants(n, seed=int(rng.integers(0, 2_000_000_000)))
    mutated = strat_a.mutate(base, rng, difficulty)
    mutated = strat_b.mutate(mutated, rng, difficulty)
    import uuid
    mutated["is_fraud"] = 1
    mutated["attack_type"] = f"{attack_type_a}+{attack_type_b}_HYBRID"
    mutated["difficulty"] = difficulty
    mutated["attack_id"] = [f"ATK_HYBRID_{uuid.uuid4().hex[:10]}" for _ in range(len(mutated))]
    mutated["source"] = "synthetic_attack"
    return mutated


def generate_harder_attacks(weakness_report: dict, n_per_type: int = 150, use_llm: bool = True, seed: int | None = None) -> pd.DataFrame:
    """Uses the discovery agent to propose targeted hypotheses, then
    materializes them as new synthetic attack batches at elevated
    difficulty (section 14, steps 5-6)."""
    hyp_result = discover_attack_hypotheses(weakness_report.get("recall_by_attack_type", {}), n=5, use_llm=use_llm)
    hypotheses = hyp_result["hypotheses"]
    rng = np.random.default_rng(seed)
    batches = []

    for h in hypotheses:
        difficulty = float(np.clip(h.get("difficulty", 0.8), 0.5, 0.98))
        compose_of = h.get("compose_of")
        try:
            if compose_of and len(compose_of) == 2 and compose_of[0] in list_attacks() and compose_of[1] in list_attacks():
                batch = _compose_attack_batch(compose_of[0], compose_of[1], n_per_type, difficulty, int(rng.integers(0, 2_000_000_000)))
            else:
                attack_type = h.get("attack_type")
                if attack_type not in list_attacks():
                    continue
                batch = get_attack(attack_type).generate(n_per_type, difficulty=difficulty, seed=int(rng.integers(0, 2_000_000_000)))
            batches.append(batch)
        except Exception as e:
            logger.warning("Skipping hypothesis %s: %s", h.get("attack_type"), e)

    if not batches:
        # Guaranteed fallback: harden the globally hardest attack type
        batch = get_attack("MULTI_SIGNAL_SYNTHETIC_IDENTITY").generate(n_per_type, difficulty=0.9, seed=seed)
        batches.append(batch)

    result = pd.concat(batches, ignore_index=True)
    result["hypothesis_source"] = hyp_result["source"]
    return result


def augment_training_data(original_df: pd.DataFrame, new_attacks_df: pd.DataFrame) -> pd.DataFrame:
    """Combine, then RECOMPUTE graph features on the full combined
    population (reuse counts/cluster sizes must reflect the new rows too)."""
    combined = pd.concat([original_df, new_attacks_df.drop(columns=["hypothesis_source"], errors="ignore")], ignore_index=True)
    return calculate_graph_features(combined)


def retrain_model(augmented_df: pd.DataFrame, seed: int = 42) -> dict:
    return train_and_evaluate(augmented_df, seed=seed)


def run_closed_loop_iteration(
    current_df: pd.DataFrame, bundle: ModelBundle | None, n_per_type: int = 150,
    use_llm: bool = True, seed: int | None = None, holdout_frac: float = 0.3,
) -> dict:
    """One full pass of the loop (section 14/31). If `bundle` is None, this
    is iteration 1 and there is nothing to evaluate/harden yet -- just
    train the first model.

    From iteration 2 onward, the before/after comparison is measured on a
    HELD-OUT slice of the newly generated harder-attack batch: the OLD
    model scores it first (this is what makes it "hard" -- the old model
    was weak on it), only the remaining rows are folded into training, and
    the NEW model is then scored on that same untouched holdout. This keeps
    the reported improvement genuine rather than measuring a model on
    examples it was just trained on (section 32: no fake metrics)."""
    rng = np.random.default_rng(seed)

    if bundle is None:
        train_result = retrain_model(current_df, seed=seed or 42)
        return {
            "iteration": 1, "before_metrics": None,
            "after_metrics": train_result["final_model_metrics"],
            "weakness_report": None, "new_attack_count": 0,
            "dataset_size_before": len(current_df), "dataset_size_after": len(current_df),
            "model_version": train_result["version"],
        }

    scored = evaluate_attacks(current_df, bundle)
    weakness_report = analyze_detector_weaknesses(scored)
    harder_attacks = generate_harder_attacks(weakness_report, n_per_type=n_per_type, use_llm=use_llm, seed=int(rng.integers(0, 2_000_000_000)))
    harder_attacks = calculate_graph_features(pd.concat([current_df, harder_attacks.drop(columns=["hypothesis_source"], errors="ignore")], ignore_index=True)).tail(len(harder_attacks)).reset_index(drop=True)

    n_holdout = max(10, int(len(harder_attacks) * holdout_frac))
    shuffled = harder_attacks.sample(frac=1.0, random_state=int(rng.integers(0, 2_000_000_000))).reset_index(drop=True)
    holdout, train_add = shuffled.iloc[:n_holdout], shuffled.iloc[n_holdout:]

    holdout_prob_before = score_dataframe(holdout, bundle)
    before_metrics = compute_metrics(holdout["is_fraud"].values, holdout_prob_before)

    augmented = augment_training_data(current_df, train_add)
    train_result = retrain_model(augmented, seed=int(rng.integers(0, 2_000_000_000)))

    new_bundle = ModelBundle.load(version=train_result["version"])
    holdout_prob_after = score_dataframe(holdout, new_bundle)
    after_metrics = compute_metrics(holdout["is_fraud"].values, holdout_prob_after)

    return {
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "overall_model_metrics": train_result["final_model_metrics"],
        "weakness_report": weakness_report,
        "new_attack_count": int(len(harder_attacks)),
        "holdout_size": int(len(holdout)),
        "dataset_size_before": len(current_df), "dataset_size_after": len(augmented),
        "model_version": train_result["version"],
        "augmented_df": augmented,
    }
