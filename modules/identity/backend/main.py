"""FastAPI application entrypoint.

Run with:  uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import (
    routes_applicants, routes_attacks, routes_detection, routes_feedback,
    routes_graph,
)
from backend.config import settings
from backend.models.db import init_db
from backend.red_team.registry import load_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all()  # register every red-team attack plugin
    init_db()
    logging.getLogger(__name__).info("Identity Fraud Defense Lab backend ready.")
    yield


app = FastAPI(
    title=settings.app_name,
    description="AI Defense Lab for Payment Security -- Identity & Onboarding Fraud module (Mastercard Innovation Challenge 2026)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["system"])
def health():
    from backend.red_team.registry import list_attacks
    from backend.blue_team.predict import ModelBundle
    bundle = ModelBundle.load()
    return {
        "status": "ok",
        "app": settings.app_name,
        "attacks_registered": len(list_attacks()),
        "model_loaded": bundle is not None,
        "model_version": bundle.version if bundle else None,
    }


app.include_router(routes_applicants.router)
app.include_router(routes_attacks.router)
app.include_router(routes_detection.router)
app.include_router(routes_graph.router)
app.include_router(routes_feedback.router)
