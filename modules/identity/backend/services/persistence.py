"""All SQLite writes live here -- routes and other services never touch the
ORM directly (section 17: no ML/DB logic in route handlers)."""
from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy.orm import Session

from backend.graph.builder import build_graph
from backend.models.orm import (
    Applicant, Attack, FeedbackIteration, GraphEntity, GraphRelationship,
    ModelVersion, Prediction, TrainingRun,
)

logger = logging.getLogger(__name__)

MAX_GRAPH_SYNC_ROWS = 20_000


def save_applicants(session: Session, df: pd.DataFrame) -> None:
    existing = {row[0] for row in session.query(Applicant.applicant_id).all()}
    objs = []
    for _, row in df.iterrows():
        if row["applicant_id"] in existing:
            continue
        objs.append(Applicant(
            applicant_id=row["applicant_id"], name=row.get("name", ""),
            is_fraud=int(row.get("is_fraud", 0)), attack_type=row.get("attack_type", "NONE"),
            difficulty=float(row.get("difficulty") or 0.0), device_id=row.get("device_id"),
            ip_id=row.get("ip_id"), phone=row.get("phone"), email=row.get("email"),
            raw_json=_row_to_json(row),
        ))
    if objs:
        session.bulk_save_objects(objs)
        session.commit()


def _row_to_json(row: pd.Series) -> dict:
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items() if pd.notna(v) if not isinstance(v, (list, dict))}


def save_attack_batch(session: Session, batch_id: str, attack_type: str, difficulty: float, n_records: int, description: str, severity: str) -> None:
    session.add(Attack(
        attack_batch_id=batch_id, attack_type=attack_type, difficulty=difficulty,
        n_records=n_records, description=description, severity=severity,
    ))
    session.commit()


def save_prediction(session: Session, applicant_id: str, model_version: int, fraud_probability: float, risk_score: float, decision: str, ground_truth: int | None, risk_factors: list | None) -> None:
    session.add(Prediction(
        applicant_id=applicant_id, model_version=model_version, fraud_probability=fraud_probability,
        risk_score=risk_score, decision=decision, ground_truth=ground_truth, risk_factors=risk_factors,
    ))
    session.commit()


def save_model_version(session: Session, version: int, metrics: dict, metadata: dict | None = None) -> None:
    session.query(ModelVersion).update({ModelVersion.is_current: False})
    session.add(ModelVersion(
        version=version, precision=metrics.get("precision", 0), recall=metrics.get("recall", 0),
        f1=metrics.get("f1", 0), pr_auc=metrics.get("pr_auc", 0), roc_auc=metrics.get("roc_auc"),
        is_current=True, metadata_json=metadata,
    ))
    session.commit()


def save_training_run(session: Session, version: int, n_samples: int, metrics_payload: dict) -> None:
    session.add(TrainingRun(model_version=version, n_samples=n_samples, metrics_json=metrics_payload))
    session.commit()


def save_feedback_iteration(session: Session, iteration: int, before: dict | None, after: dict, version_before: int | None, version_after: int, new_attack_count: int, weakness_summary: str | None) -> None:
    session.add(FeedbackIteration(
        iteration=iteration,
        model_version_before=version_before, model_version_after=version_after,
        recall_before=(before or {}).get("recall"), recall_after=after.get("recall"),
        precision_before=(before or {}).get("precision"), precision_after=after.get("precision"),
        f1_before=(before or {}).get("f1"), f1_after=after.get("f1"),
        new_attack_count=new_attack_count, weakness_summary=weakness_summary,
    ))
    session.commit()


def sync_graph(session: Session, df: pd.DataFrame, dataset_tag: str = "default") -> bool:
    """Rebuild graph_entities/graph_relationships from the current dataset.
    Skipped above MAX_GRAPH_SYNC_ROWS to keep the demo responsive -- the
    live NetworkX graph (backend/graph/*) is still used for all real-time
    queries regardless of whether this snapshot ran."""
    if len(df) > MAX_GRAPH_SYNC_ROWS:
        logger.info("Skipping graph_entities DB sync for %d rows (> %d cap)", len(df), MAX_GRAPH_SYNC_ROWS)
        return False

    session.query(GraphEntity).filter(GraphEntity.dataset_tag == dataset_tag).delete()
    session.query(GraphRelationship).filter(GraphRelationship.dataset_tag == dataset_tag).delete()

    graph = build_graph(df)
    entities = [
        GraphEntity(node_id=node, kind=data.get("kind", "unknown"), value=data.get("value") or data.get("applicant_id"), dataset_tag=dataset_tag)
        for node, data in graph.nodes(data=True)
    ]
    relationships = [
        GraphRelationship(source_node_id=u, target_node_id=v, relation=data.get("relation", ""), dataset_tag=dataset_tag)
        for u, v, data in graph.edges(data=True)
    ]
    session.bulk_save_objects(entities)
    session.bulk_save_objects(relationships)
    session.commit()
    return True
