from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from backend.blue_team.predict import ModelBundle
from backend.feedback.engine import run_closed_loop_iteration
from backend.services import persistence
from backend.services.state import state
from backend.services.detection_service import NoDatasetError

logger = logging.getLogger(__name__)


def run_closed_loop(session: Session, iterations: int = 3, n_per_type: int = 150, use_llm: bool = True, seed: int = 42) -> dict:
    df = state.get_dataset()
    if df is None or df["is_fraud"].nunique() < 2:
        raise NoDatasetError("Need a dataset with both legitimate and fraud examples -- generate applicants and at least one attack first.")

    bundle = ModelBundle.load()
    history = []
    prev_version = bundle.version if bundle else None

    for i in range(1, iterations + 1):
        result = run_closed_loop_iteration(df, bundle, n_per_type=n_per_type, use_llm=use_llm, seed=seed + i)
        iteration_record = {
            "iteration": i,
            "before_metrics": result["before_metrics"],
            "after_metrics": result["after_metrics"],
            "new_attack_count": result["new_attack_count"],
            "model_version": result["model_version"],
            "weakness_summary": (result.get("weakness_report") or {}).get("summary"),
            "dataset_size": result["dataset_size_after"],
        }
        history.append(iteration_record)

        persistence.save_model_version(session, result["model_version"], result["after_metrics"])
        persistence.save_feedback_iteration(
            session, i, result["before_metrics"], result["after_metrics"],
            prev_version, result["model_version"], result["new_attack_count"],
            iteration_record["weakness_summary"],
        )

        if "augmented_df" in result:
            df = result["augmented_df"]
        prev_version = result["model_version"]
        bundle = ModelBundle.load(version=result["model_version"])

    state.set_dataset(df)
    state.feedback_history.extend(history)
    persistence.sync_graph(session, df)

    return {"iterations": history, "final_model_version": prev_version, "final_dataset_size": len(df)}


def get_feedback_history() -> list[dict]:
    return state.feedback_history
