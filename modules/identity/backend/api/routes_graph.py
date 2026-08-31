from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.services import graph_service
from backend.services.detection_service import NoDatasetError

router = APIRouter(tags=["graph"])


@router.get("/api/graph/rings")
def graph_rings(min_size: int = Query(3, ge=2, le=50)):
    try:
        return graph_service.get_fraud_rings(min_size)
    except NoDatasetError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api/graph/{applicant_id}")
def graph_for_applicant(applicant_id: str):
    try:
        info = graph_service.get_applicant_graph(applicant_id)
    except NoDatasetError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not info.get("found"):
        raise HTTPException(status_code=404, detail=f"Unknown applicant_id: {applicant_id}")
    return info
