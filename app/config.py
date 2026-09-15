from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    kaggle_username: str = ""
    kaggle_key: str = ""

    dataset_slug: str = "gabrielfcarvalho/cardd-with-yolo-annotations-images-labels"
    dataset_raw_dir: Path = Path("data/raw")
    dataset_processed_dir: Path = Path("data/processed")

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
        3: "glass_shatter",
        4: "lamp_broken",
        5: "tire_flat",
    }

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def allowed_content_types(self) -> list[str]:
        return [item.strip() for item in self.allowed_content_types_raw.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
