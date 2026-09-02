from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from backend.data.generator import generate_legitimate_applicants
from backend.graph.features import calculate_graph_features
from backend.services import persistence
from backend.services.state import state

logger = logging.getLogger(__name__)


def generate_applicants(session: Session, n: int, seed: int | None = None) -> dict:
    df = generate_legitimate_applicants(n, seed=seed if seed is not None else 42)
    df = calculate_graph_features(df)
    full = state.append_dataset(df)
    full = calculate_graph_features(full)  # recompute reuse counts over the WHOLE population
    state.set_dataset(full)

    persistence.save_applicants(session, df)
    persistence.sync_graph(session, full)

    return {
        "generated": int(len(df)),
        "total_dataset_size": int(len(full)),
        "fraud_rate": float(full["is_fraud"].mean()),
    }


def current_dataset_summary() -> dict:
    df = state.get_dataset()
    if df is None:
        return {"total_applicants": 0, "total_fraud": 0, "fraud_rate": 0.0, "attack_types": {}, "manual_submissions": 0}

    manual_mask = df["source"] == "manual" if "source" in df.columns else (df["applicant_id"] == "__never__")
    labeled = df[~manual_mask]

    return {
        "total_applicants": int(len(df)),
        "total_fraud": int(labeled["is_fraud"].sum()),
        "fraud_rate": float(labeled["is_fraud"].mean()) if len(labeled) else 0.0,
        "attack_types": labeled.loc[labeled.is_fraud == 1, "attack_type"].value_counts().to_dict(),
        "manual_submissions": int(manual_mask.sum()),
    }
