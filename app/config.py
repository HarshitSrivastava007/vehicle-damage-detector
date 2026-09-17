from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_weights_path: Path = Path("models/best.pt")
    fallback_weights_path: str = "yolov8s-seg.pt"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "cpu"

    max_upload_size_mb: int = 10
    # Stored as a raw comma-separated string (not list[str]) because
    # pydantic-settings JSON-decodes complex env values before validators
    # run, which breaks on a plain comma-separated ALLOWED_CONTENT_TYPES.
    allowed_content_types_raw: str = Field(
        default="image/jpeg,image/png,image/webp",
        validation_alias="ALLOWED_CONTENT_TYPES",
    )

    class_names: dict[int, str] = {
        0: "dent",
        1: "scratch",
        2: "crack",
        3: "broken",
        4: "tire_flat",
    }

    # Single mask/box/label color used for all classes in annotated output.
    mask_color_rgb: tuple[int, int, int] = (132, 255, 0)  # #84ff00

    database_url: str = "sqlite:///./app.db"

    # Gates the session cookie's `secure` flag: off in dev (plain HTTP),
    # on in production (HTTPS only). Set ENVIRONMENT=production to deploy.
    environment: str = "development"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def allowed_content_types(self) -> list[str]:
        return [item.strip() for item in self.allowed_content_types_raw.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
