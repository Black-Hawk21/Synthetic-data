"""SQLite (default) database engine/session setup. Postgres is a drop-in
swap via IFDL_DB_URL, but nothing in this project requires it (section 18)."""
from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from backend.config import settings

DB_URL = os.environ.get("IFDL_DB_URL", f"sqlite:///{settings.db_path}")

connect_args = {"check_same_thread": False} if DB_URL.startswith("sqlite") else {}
engine = create_engine(DB_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    from backend.models import orm  # noqa: F401  (ensures models are registered)
    Base.metadata.create_all(bind=engine)


def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
