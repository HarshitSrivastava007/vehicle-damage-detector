from datetime import datetime

from pydantic import BaseModel, EmailStr


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


class RegisterRequest(BaseModel):
    email: EmailStr


class RegisterResponse(BaseModel):
    user_id: int
    email: EmailStr
    api_key: str


class UserOut(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    email: EmailStr
    created_at: datetime


class APIKeyCreateResponse(BaseModel):
    id: int
    api_key: str
    created_at: datetime


class APIKeyOut(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    prefix: str
    created_at: datetime
    last_used_at: datetime | None
    revoked_at: datetime | None


class DetectionLogOut(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    filename: str
    detection_count: int
    class_counts: dict[str, int]
    created_at: datetime
