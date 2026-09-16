import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import get_current_api_key
from app.config import Settings, get_settings
from app.db import get_db, init_db
from app.db_models import APIKey, DetectionLog
from app.inference import annotate_image, build_detection_response
from app.model_loader import get_model, get_weights_path_used, run_inference
from app.routers.auth import router as auth_router
from app.schemas import DetectionLogOut, HealthResponse

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()
    init_db()
    yield


app = FastAPI(title="Vehicle Damage Detection API", version="0.1.0", lifespan=lifespan)
app.include_router(auth_router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error while processing request")
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text())


@app.get("/health", response_model=HealthResponse)
async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=get_weights_path_used() is not None,
        weights_path=get_weights_path_used() or str(settings.model_weights_path),
    )


@app.post("/api/v1/detect")
async def detect(
    file: UploadFile,
    annotate: bool = False,
    settings: Settings = Depends(get_settings),
    db: Session = Depends(get_db),
    api_key: APIKey = Depends(get_current_api_key),
):
    if file.content_type not in settings.allowed_content_types:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported content type: {file.content_type}",
        )

    contents = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"file exceeds max size of {settings.max_upload_size_mb}MB",
        )

    try:
        results = run_inference(contents)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result = results[0]
    # Computed unconditionally (even for the annotate=True response) so a
    # DetectionLog row can always be written without duplicating the
    # detection-extraction logic.
    response = build_detection_response(result, filename=file.filename or "upload")

    class_counts: dict[str, int] = {}
    for detection in response.detections:
        class_counts[detection.class_name] = class_counts.get(detection.class_name, 0) + 1

    db.add(
        DetectionLog(
            user_id=api_key.user_id,
            api_key_id=api_key.id,
            filename=response.filename,
            detection_count=response.count,
            class_counts=class_counts,
        )
    )
    db.commit()

    if annotate:
        return Response(content=annotate_image(result), media_type="image/png")

    return response


@app.get("/api/v1/detections", response_model=list[DetectionLogOut])
async def list_detections(
    limit: int = 20,
    offset: int = 0,
    api_key: APIKey = Depends(get_current_api_key),
    db: Session = Depends(get_db),
):
    limit = max(1, min(limit, 100))
    stmt = (
        select(DetectionLog)
        .where(DetectionLog.user_id == api_key.user_id)
        .order_by(DetectionLog.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(db.scalars(stmt))
