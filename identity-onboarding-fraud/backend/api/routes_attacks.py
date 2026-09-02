from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.models.db import get_session
from backend.schemas.api_schemas import GenerateAttackRequest, RunRedTeamRequest
from backend.services import attack_service

router = APIRouter(tags=["attacks"])


@router.get("/api/attacks")
def list_attacks():
    return {"attacks": attack_service.list_attack_catalog()}


@router.post("/api/generate-attack")
def generate_attack(req: GenerateAttackRequest, session: Session = Depends(get_session)):
    try:
        return attack_service.generate_attack(session, req.attack_type, req.difficulty, req.n, req.seed)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/run-red-team")
def run_red_team(req: RunRedTeamRequest, session: Session = Depends(get_session)):
    try:
        return attack_service.run_red_team(session, [a.model_dump() for a in req.attacks])
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
