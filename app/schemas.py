from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, EmailStr, Field


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
    # Populated only when ?include_annotated=true — a base64-encoded PNG,
    # so a caller can get detections + the annotated image in one request
    # instead of two (two calls to /detect would log two DetectionLog rows
    # for what a user experiences as a single "run detection" action).
    annotated_image_base64: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    weights_path: str


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str | None = None


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
    cost: Decimal
    created_at: datetime


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class SessionUserOut(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    email: EmailStr
    is_admin: bool
    is_active: bool
    cost_per_call: Decimal
    created_at: datetime


class AdminUserOut(SessionUserOut):
    total_detections: int


class UpdateCostRequest(BaseModel):
    cost_per_call: Decimal = Field(ge=0)


class AdminCreateUserRequest(BaseModel):
    email: EmailStr
    password: str
    is_admin: bool = False
    cost_per_call: Decimal = Field(default=Decimal("0"), ge=0)


class AdminCreateUserResponse(AdminUserOut):
    api_key: str


class UpdateUserStatusRequest(BaseModel):
    is_active: bool


class UsageSummaryOut(BaseModel):
    calls_this_month: int
    cost_this_month: Decimal
    calls_all_time: int
    cost_all_time: Decimal
