"""SQLAlchemy ORM tables (section 18): applicants, attacks, predictions,
graph_entities, graph_relationships, training_runs, model_versions,
feedback_iterations."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.models.db import Base


class Applicant(Base):
    __tablename__ = "applicants"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    applicant_id: Mapped[str] = mapped_column(String, unique=True, index=True)
    name: Mapped[str] = mapped_column(String)
    is_fraud: Mapped[int] = mapped_column(Integer, default=0)
    attack_type: Mapped[str] = mapped_column(String, default="NONE")
    difficulty: Mapped[float] = mapped_column(Float, default=0.0)
    device_id: Mapped[str] = mapped_column(String, nullable=True)
    ip_id: Mapped[str] = mapped_column(String, nullable=True)
    phone: Mapped[str] = mapped_column(String, nullable=True)
    email: Mapped[str] = mapped_column(String, nullable=True)
    raw_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Attack(Base):
    __tablename__ = "attacks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    attack_batch_id: Mapped[str] = mapped_column(String, index=True)
    attack_type: Mapped[str] = mapped_column(String, index=True)
    difficulty: Mapped[float] = mapped_column(Float)
    n_records: Mapped[int] = mapped_column(Integer)
    description: Mapped[str] = mapped_column(String)
    severity: Mapped[str] = mapped_column(String)
    detected_count: Mapped[int] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    applicant_id: Mapped[str] = mapped_column(String, index=True)
    model_version: Mapped[int] = mapped_column(Integer)
    fraud_probability: Mapped[float] = mapped_column(Float)
    risk_score: Mapped[float] = mapped_column(Float)
    decision: Mapped[str] = mapped_column(String)
    ground_truth: Mapped[int] = mapped_column(Integer, nullable=True)
    risk_factors: Mapped[dict] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class GraphEntity(Base):
    __tablename__ = "graph_entities"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    node_id: Mapped[str] = mapped_column(String, index=True)
    kind: Mapped[str] = mapped_column(String)  # person, phone, email, address, device, ip, document
    value: Mapped[str] = mapped_column(String, nullable=True)
    dataset_tag: Mapped[str] = mapped_column(String, index=True, default="default")


class GraphRelationship(Base):
    __tablename__ = "graph_relationships"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_node_id: Mapped[str] = mapped_column(String, index=True)
    target_node_id: Mapped[str] = mapped_column(String, index=True)
    relation: Mapped[str] = mapped_column(String)
    dataset_tag: Mapped[str] = mapped_column(String, index=True, default="default")


class TrainingRun(Base):
    __tablename__ = "training_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version: Mapped[int] = mapped_column(Integer)
    n_samples: Mapped[int] = mapped_column(Integer)
    metrics_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[int] = mapped_column(Integer, unique=True)
    precision: Mapped[float] = mapped_column(Float)
    recall: Mapped[float] = mapped_column(Float)
    f1: Mapped[float] = mapped_column(Float)
    pr_auc: Mapped[float] = mapped_column(Float)
    roc_auc: Mapped[float] = mapped_column(Float, nullable=True)
    is_current: Mapped[bool] = mapped_column(Boolean, default=False)
    metadata_json: Mapped[dict] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class FeedbackIteration(Base):
    __tablename__ = "feedback_iterations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    iteration: Mapped[int] = mapped_column(Integer)
    model_version_before: Mapped[int] = mapped_column(Integer, nullable=True)
    model_version_after: Mapped[int] = mapped_column(Integer)
    recall_before: Mapped[float] = mapped_column(Float, nullable=True)
    recall_after: Mapped[float] = mapped_column(Float, nullable=True)
    precision_before: Mapped[float] = mapped_column(Float, nullable=True)
    precision_after: Mapped[float] = mapped_column(Float, nullable=True)
    f1_before: Mapped[float] = mapped_column(Float, nullable=True)
    f1_after: Mapped[float] = mapped_column(Float, nullable=True)
    new_attack_count: Mapped[int] = mapped_column(Integer, default=0)
    weakness_summary: Mapped[str] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
