from __future__ import annotations

from fastapi import APIRouter

from app.schemas.inference import ChangeDetectionRequest, ChangeDetectionResponse
from app.services.inference_service import inference_service

router = APIRouter(prefix="/v1/change-detection", tags=["inference"])


@router.post(
    "",
    response_model=ChangeDetectionResponse,
    summary="Run satellite change detection on a pair of images.",
)
async def run_change_detection(
    payload: ChangeDetectionRequest,
) -> ChangeDetectionResponse:
    return inference_service.run_change_detection(payload)

