"""Backs the Onboarding Simulator page (section 16, Page 3): produces one
synthetic applicant -- either a clean legitimate one or a specific attack
type -- and (optionally) folds it into the working dataset so the Identity
Graph page can immediately show its connections."""
from __future__ import annotations

from sqlalchemy.orm import Session

from backend.graph.features import calculate_graph_features
from backend.red_team.registry import get_attack
from backend.data.generator import generate_legitimate_applicants
from backend.services import persistence
from backend.services.state import state


def simulate_applicant(session: Session, attack_type: str | None = None, difficulty: float = 0.5, seed: int | None = None) -> dict:
    if attack_type and attack_type != "NONE":
        strategy = get_attack(attack_type)
        row_df = strategy.generate(1, difficulty=difficulty, seed=seed)
    else:
        row_df = generate_legitimate_applicants(1, seed=seed if seed is not None else 7)

    full = state.append_dataset(row_df)
    full = calculate_graph_features(full)
    state.set_dataset(full)
    persistence.save_applicants(session, row_df)

    applicant = full[full["applicant_id"] == row_df.iloc[0]["applicant_id"]].iloc[0]
    return applicant.to_dict()
