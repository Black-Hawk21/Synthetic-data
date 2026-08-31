from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.models.db import get_session
from backend.schemas.api_schemas import RunClosedLoopRequest
from backend.services import feedback_service
from backend.services.detection_service import NoDatasetError

router = APIRouter(tags=["feedback"])


@router.post("/api/run-closed-loop")
def run_closed_loop(req: RunClosedLoopRequest, session: Session = Depends(get_session)):
    try:
        return feedback_service.run_closed_loop(session, req.iterations, req.n_per_type, req.use_llm, req.seed)
    except NoDatasetError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api/feedback")
def feedback_history():
    return {"history": feedback_service.get_feedback_history()}
