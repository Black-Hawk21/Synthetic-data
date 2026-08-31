"""Central configuration for the Identity & Onboarding Fraud Defense Lab.

Everything here is free/open-source and works with zero environment
variables. Optional integrations (Ollama, InsightFace, Tesseract) are
detected at runtime and gracefully disabled if unavailable.
"""
from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    app_name: str = "Identity & Onboarding Fraud Defense Lab"
    env: str = os.environ.get("APP_ENV", "local")

    # Storage
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    synthetic_dir: Path = BASE_DIR / "data" / "synthetic"
    processed_dir: Path = BASE_DIR / "data" / "processed"
    models_dir: Path = BASE_DIR / "models"
    db_path: Path = BASE_DIR / "identity_fraud.db"

    # Decision thresholds (configurable, section 7)
    approve_threshold: float = 0.30
    review_threshold: float = 0.70

    # Optional integrations
    ollama_host: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.environ.get("OLLAMA_MODEL", "llama3.2")

    # Reproducibility
    random_seed: int = 42

    # API
    cors_origins: list[str] = ["*"]

    class Config:
        env_prefix = "IFDL_"


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.synthetic_dir.mkdir(parents=True, exist_ok=True)
settings.processed_dir.mkdir(parents=True, exist_ok=True)
settings.models_dir.mkdir(parents=True, exist_ok=True)
