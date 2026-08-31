"""Pydantic request/response models for the FastAPI layer (section 17)."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class GenerateApplicantsRequest(BaseModel):
    n: int = Field(1000, gt=0, le=500_000)
    seed: Optional[int] = None


class GenerateAttackRequest(BaseModel):
    attack_type: str
    difficulty: float = Field(0.5, ge=0.0, le=1.0)
    n: int = Field(200, gt=0, le=200_000)
    seed: Optional[int] = None


class RunRedTeamRequest(BaseModel):
    attacks: list[GenerateAttackRequest]


class ScoreApplicantRequest(BaseModel):
    applicant_id: Optional[str] = None
    applicant: Optional[dict[str, Any]] = None
    top_n_factors: int = 5


class RunDetectionRequest(BaseModel):
    threshold: Optional[float] = None


class RunBlueTeamRequest(BaseModel):
    seed: int = 42


class RunClosedLoopRequest(BaseModel):
    iterations: int = Field(3, ge=1, le=10)
    n_per_type: int = Field(150, ge=10, le=5000)
    use_llm: bool = True
    seed: int = 42


class OnboardingSimulateRequest(BaseModel):
    attack_type: Optional[str] = None
    difficulty: float = Field(0.5, ge=0.0, le=1.0)
    seed: Optional[int] = None
