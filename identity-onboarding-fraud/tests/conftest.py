from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from backend.red_team.registry import load_all

load_all()


@pytest.fixture(scope="session")
def small_legit_df():
    from backend.data.generator import generate_legitimate_applicants
    return generate_legitimate_applicants(500, seed=1)


@pytest.fixture
def db_session(tmp_path):
    """A throwaway SQLite-backed session for tests that exercise
    backend/services/persistence.py without touching the real project DB."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from backend.models.db import Base
    from backend.models import orm  # noqa: F401 -- register tables

    engine = create_engine(f"sqlite:///{tmp_path}/test.db", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
