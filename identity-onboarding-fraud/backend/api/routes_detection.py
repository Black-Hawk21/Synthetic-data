from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.blue_team.predict import ModelBundle
from backend.models.db import get_session
from backend.schemas.api_schemas import RunBlueTeamRequest, RunDetectionRequest, ScoreApplicantRequest
from backend.services import detection_service
from backend.services.detection_service import NoDatasetError, NoModelError

router = APIRouter(tags=["detection"])


@router.post("/api/run-blue-team")
def run_blue_team(req: RunBlueTeamRequest, session: Session = Depends(get_session)):
    try:
        return detection_service.run_blue_team(session, req.seed)
    except NoDatasetError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/score-applicant")
def score_applicant(req: ScoreApplicantRequest, session: Session = Depends(get_session)):
    if not req.applicant_id:
        raise HTTPException(status_code=422, detail="applicant_id is required")
    try:
        return detection_service.score_applicant_by_id(session, req.applicant_id, req.top_n_factors)
    except (NoDatasetError, NoModelError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/api/run-detection")
def run_detection(req: RunDetectionRequest, session: Session = Depends(get_session)):
    try:
        return detection_service.run_detection(session, req.threshold)
    except (NoDatasetError, NoModelError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api/attack-results")
def attack_results(session: Session = Depends(get_session)):
    try:
        return detection_service.attack_results_summary(session)
    except NoDatasetError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api/model-info")
def model_info():
    bundle = ModelBundle.load()
    if bundle is None:
        return {"model_loaded": False}
    return {"model_loaded": True, "version": bundle.version, "metadata": bundle.metadata}


@router.get("/api/metrics")
def metrics(session: Session = Depends(get_session)):
    from backend.config import settings
    from backend.services import dataset_service
    bundle = ModelBundle.load()
    summary = dataset_service.current_dataset_summary()
    model_metrics = None
    if bundle is not None:
        import json
        metrics_path = settings.models_dir / f"model_v{bundle.version}" / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                model_metrics = json.load(f)
    return {"dataset": summary, "model_version": bundle.version if bundle else None, "model_metrics": model_metrics}
