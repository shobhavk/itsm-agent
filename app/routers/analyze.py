import io
import logging

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.models.schemas import AnalysisResponse
from app.security import validate_upload, verify_api_key
from app.services.pipeline import run_pipeline_from_bytes, run_pipeline_from_text

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["analysis"])

# In-memory cache of the last result per API key, so /export can regenerate
# a CSV without re-uploading. Fine for v1 single-instance deployments;
# swap for Redis/S3 if you scale horizontally.
_LAST_RESULT: dict[str, AnalysisResponse] = {}


@router.post("/analyze/file", response_model=AnalysisResponse)
async def analyze_file(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    validate_upload(file)
    try:
        content = await file.read()
        result = await run_pipeline_from_bytes(file.filename, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("Analysis pipeline failed")
        raise HTTPException(status_code=500, detail="Failed to analyze the uploaded file.")

    _LAST_RESULT[api_key] = result
    return result


@router.post("/analyze/text", response_model=AnalysisResponse)
async def analyze_text(raw_text: str, api_key: str = Depends(verify_api_key)):
    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="raw_text is empty.")
    if len(raw_text) > 200_000:
        raise HTTPException(status_code=413, detail="Text input too large (max 200,000 characters).")
    try:
        result = await run_pipeline_from_text(raw_text)
    except Exception:
        logger.exception("Analysis pipeline failed")
        raise HTTPException(status_code=500, detail="Failed to analyze the provided text.")

    _LAST_RESULT[api_key] = result
    return result


@router.get("/export/csv")
async def export_csv(api_key: str = Depends(verify_api_key)):
    result = _LAST_RESULT.get(api_key)
    if not result:
        raise HTTPException(status_code=404, detail="No analysis found for this API key yet. Run /analyze first.")

    df = pd.DataFrame([r.model_dump() for r in result.results])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=itsm_quality_analysis.csv"},
    )
