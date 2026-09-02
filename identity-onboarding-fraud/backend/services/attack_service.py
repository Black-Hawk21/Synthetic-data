from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from backend.graph.features import calculate_graph_features
from backend.red_team.registry import all_attack_meta, get_attack, list_attacks
from backend.services import persistence
from backend.services.state import state


def list_attack_catalog(difficulty: float = 0.5) -> list[dict]:
    return all_attack_meta(difficulty)


def generate_attack(session: Session, attack_type: str, difficulty: float, n: int, seed: int | None = None) -> dict:
    strategy = get_attack(attack_type)
    batch = strategy.generate(n, difficulty=difficulty, seed=seed)

    batch_id = f"BATCH_{uuid.uuid4().hex[:10]}"
    full = state.append_dataset(batch)
    full = calculate_graph_features(full)
    state.set_dataset(full)
    state.last_attack_batches[batch_id] = full[full["attack_id"].isin(batch["attack_id"])].copy()

    persistence.save_applicants(session, batch)
    persistence.save_attack_batch(
        session, batch_id, attack_type, difficulty, len(batch),
        strategy.description(), strategy.severity(difficulty),
    )
    persistence.sync_graph(session, full)

    generated_slice = full[full["attack_id"].isin(batch["attack_id"])]
    return {
        "batch_id": batch_id,
        "attack_type": attack_type,
        "description": strategy.description(),
        "severity": strategy.severity(difficulty),
        "difficulty": difficulty,
        "n_generated": int(len(batch)),
        "suspicious_clusters": int((generated_slice["suspicious_cluster_score"] > 0.35).sum()),
        "infra_reuse_cases": int(((generated_slice["device_reuse_count"] > 0) | (generated_slice["ip_reuse_count"] > 0)).sum()),
        "total_dataset_size": int(len(full)),
    }


def run_red_team(session: Session, attack_configs: list[dict]) -> dict:
    results = [
        generate_attack(session, cfg["attack_type"], cfg.get("difficulty", 0.5), cfg["n"], cfg.get("seed"))
        for cfg in attack_configs
    ]
    return {
        "batches": results,
        "total_generated": sum(r["n_generated"] for r in results),
        "total_dataset_size": results[-1]["total_dataset_size"] if results else 0,
    }
