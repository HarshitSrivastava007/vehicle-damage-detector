import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.requests import Request
from fastapi.responses import JSONResponse, Response

from app.config import Settings, get_settings
from app.inference import annotate_image, build_detection_response
from app.model_loader import get_model, get_weights_path_used, run_inference
from app.schemas import DetectionResponse, HealthResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()
    yield


app = FastAPI(title="Vehicle Damage Detection API", version="0.1.0", lifespan=lifespan)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error while processing request")
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


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

    if annotate:
        return Response(content=annotate_image(result), media_type="image/png")

    return build_detection_response(result, filename=file.filename or "upload")
