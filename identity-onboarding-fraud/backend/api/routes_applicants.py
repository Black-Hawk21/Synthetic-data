from __future__ import annotations

import io
import json
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.document.generator import (
    apply_perturbations, generate_document_image, random_synthetic_fields,
)
from backend.models.db import get_session
from backend.schemas.api_schemas import GenerateApplicantsRequest, OnboardingSimulateRequest
from backend.services import dataset_service, manual_onboarding_service, onboarding_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["applicants"])

MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8MB


async def _read_upload(file: UploadFile | None) -> bytes | None:
    if file is None or not file.filename:
        return None
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"{file.filename} exceeds the 8MB upload limit")
    return data or None


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"


@router.post("/api/generate-applicants")
def generate_applicants(req: GenerateApplicantsRequest, session: Session = Depends(get_session)):
    return dataset_service.generate_applicants(session, req.n, req.seed)


@router.get("/api/dataset-summary")
def dataset_summary():
    return dataset_service.current_dataset_summary()


@router.post("/api/onboarding/simulate")
def simulate_onboarding(req: OnboardingSimulateRequest, session: Session = Depends(get_session)):
    try:
        return onboarding_service.simulate_applicant(session, req.attack_type, req.difficulty, req.seed)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api/document/sample")
def sample_document(blur: bool = False, noise: bool = False, rotate: float = 0.0, tamper_fields: bool = False, seed: int | None = None):
    """Generates a fresh, clearly-labeled SYNTHETIC test document (Live KYC
    Form 'Download a sample ID to test with' button). Never a real ID."""
    from faker import Faker
    fake = Faker()
    if seed is not None:
        Faker.seed(seed)
    fields = random_synthetic_fields(fake, seed=seed)
    if tamper_fields:
        fields.name = fields.name.split(" ")[0] + " " + fake.last_name()  # swap surname -> field-manipulation demo
    img = generate_document_image(fields)
    if blur or noise or rotate:
        img = apply_perturbations(img, blur=blur, noise=noise, rotate_degrees=rotate)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png", headers={"Content-Disposition": "inline; filename=sample_synthetic_id.png"})


@router.post("/api/onboarding/submit")
async def submit_manual_onboarding(
    request: Request,
    session: Session = Depends(get_session),
    name: str = Form(...),
    date_of_birth: str = Form(...),
    address: str = Form(...),
    phone: str = Form(...),
    email: str = Form(...),
    document_type: str = Form("NATIONAL_ID"),
    document_number: str = Form(""),
    telemetry: str = Form("{}"),
    device_fingerprint: str = Form("{}"),
    document_image: UploadFile | None = File(None),
    selfie_image: UploadFile | None = File(None),
):
    """The 'Live KYC Form' endpoint -- a REAL onboarding submission: typed
    identity fields, an uploaded document photo, a captured selfie, and
    real client-side behavioral/device telemetry. See
    backend/services/manual_onboarding_service.py for the full pipeline."""
    try:
        telemetry_dict = json.loads(telemetry) if telemetry else {}
        device_dict = json.loads(device_fingerprint) if device_fingerprint else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="telemetry/device_fingerprint must be valid JSON strings")

    doc_bytes = await _read_upload(document_image)
    selfie_bytes = await _read_upload(selfie_image)

    try:
        result = manual_onboarding_service.submit_manual_application(
            session,
            name=name, date_of_birth=date_of_birth, address=address, phone=phone, email=email,
            document_type=document_type, document_number=document_number or None,
            document_image_bytes=doc_bytes, selfie_image_bytes=selfie_bytes,
            telemetry=telemetry_dict, device_fingerprint=device_dict, client_ip=_client_ip(request),
        )
    except Exception as e:
        logger.exception("Manual onboarding submission failed")
        raise HTTPException(status_code=500, detail=f"Could not process submission: {e}")

    return result
