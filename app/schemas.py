from pydantic import BaseModel


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: BBox
    polygon: list[list[float]] | None = None


class DetectionResponse(BaseModel):
    filename: str
    image_width: int
    image_height: int
    count: int
    detections: list[Detection]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    weights_path: str
