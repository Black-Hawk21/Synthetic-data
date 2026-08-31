import json
from pathlib import Path

from aegis.attacks.taxonomy import all_combinations
from aegis.orchestrator import evaluate_dispute
from aegis.redteam.agent import RedTeamAgent
from aegis.schemas import AttackSpec, BatchCaseResult, DisputePayload

LEGIT_SAMPLES_PATH = Path(__file__).resolve().parent.parent / "data" / "legit_samples.json"
RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def load_legit_cases() -> list[tuple[str, DisputePayload]]:
    raw = json.loads(LEGIT_SAMPLES_PATH.read_text())
    cases = []
    for case in raw:
        payload = DisputePayload(
            chat_transcript=case["chat_transcript"],
            images=case["images"],
            claimed_reason=case["claimed_reason"],
            order_metadata=case["order_metadata"],
        )
        cases.append((case["case_id"], payload))
    return cases


def generate_fraud_cases(
    red_team: RedTeamAgent, attacks: list[AttackSpec] | None = None
) -> list[tuple[str, AttackSpec, DisputePayload]]:
    attacks = attacks if attacks is not None else all_combinations()
    cases = []
    for i, attack in enumerate(attacks):
        payload = red_team.generate_initial_payload(attack)
        cases.append((f"fraud-{i:03d}", attack, payload))
    return cases


def run_batch(
    text_provider,
    vision_provider,
    red_team: RedTeamAgent,
    attacks: list[AttackSpec] | None = None,
) -> list[BatchCaseResult]:
    """Runs the Blue Team pipeline only (no Red Team mutation) over every taxonomy
    combination plus the canned legit fixtures, for precision/recall/F1/AUC."""
    results: list[BatchCaseResult] = []

    for case_id, attack, payload in generate_fraud_cases(red_team, attacks):
        _, _, supervisor_result = evaluate_dispute(text_provider, vision_provider, payload)
        results.append(
            BatchCaseResult(
                case_id=case_id, is_fraud_ground_truth=True, attack=attack, supervisor_result=supervisor_result
            )
        )

    for case_id, payload in load_legit_cases():
        _, _, supervisor_result = evaluate_dispute(text_provider, vision_provider, payload)
        results.append(
            BatchCaseResult(
                case_id=case_id, is_fraud_ground_truth=False, attack=None, supervisor_result=supervisor_result
            )
        )

    return results


def confusion_at_threshold(results: list[BatchCaseResult], threshold: float) -> dict[str, int]:
    tp = fp = tn = fn = 0
    for r in results:
        predicted_fraud = r.supervisor_result.fraud_confidence >= threshold
        if r.is_fraud_ground_truth and predicted_fraud:
            tp += 1
        elif r.is_fraud_ground_truth and not predicted_fraud:
            fn += 1
        elif not r.is_fraud_ground_truth and predicted_fraud:
            fp += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def precision_recall_f1(confusion: dict[str, int]) -> dict[str, float]:
    tp, fp, fn, tn = confusion["tp"], confusion["fp"], confusion["fn"], confusion["tn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "false_positive_rate": false_positive_rate}


def roc_curve(results: list[BatchCaseResult], steps: int = 101) -> list[dict[str, float]]:
    points = []
    for i in range(steps):
        threshold = i / (steps - 1)
        confusion = confusion_at_threshold(results, threshold)
        tp, fp, tn, fn = confusion["tp"], confusion["fp"], confusion["tn"], confusion["fn"]
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        points.append({"threshold": threshold, "tpr": tpr, "fpr": fpr})
    return points


def auc(roc_points: list[dict[str, float]]) -> float:
    pts = sorted(roc_points, key=lambda p: p["fpr"])
    area = 0.0
    for a, b in zip(pts, pts[1:]):
        area += (b["fpr"] - a["fpr"]) * (a["tpr"] + b["tpr"]) / 2
    return area


def export_batch_results(
    results: list[BatchCaseResult],
    confusion: dict[str, int],
    pr: dict[str, float],
    auc_value: float,
    path: Path | None = None,
    provider_mode: str = "mock",
) -> Path:
    """Writes real batch-eval numbers to disk so docs/generate_walkthrough.py can
    populate the required .docx deliverable with actual metrics, not estimates.
    provider_mode records whether these came from the live Claude pipeline or the
    deterministic mock provider, so the walkthrough can represent them honestly."""
    RUNS_DIR.mkdir(exist_ok=True)
    path = path or (RUNS_DIR / "latest_batch.json")
    payload = {
        "provider_mode": provider_mode,
        "confusion": confusion,
        "metrics": {**pr, "auc": auc_value},
        "cases": [
            {
                "case_id": r.case_id,
                "is_fraud_ground_truth": r.is_fraud_ground_truth,
                "attack": r.attack.model_dump() if r.attack else None,
                "decision": r.supervisor_result.decision.value,
                "fraud_confidence": r.supervisor_result.fraud_confidence,
            }
            for r in results
        ],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path
