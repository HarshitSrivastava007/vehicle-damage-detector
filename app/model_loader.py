import logging
import threading

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results

from app.config import get_settings

logger = logging.getLogger(__name__)

_model: YOLO | None = None
_model_lock = threading.Lock()
_weights_path_used: str | None = None


def _resolve_weights_path() -> str:
    settings = get_settings()
    if settings.model_weights_path.exists():
        return str(settings.model_weights_path)
    logger.warning(
        "Fine-tuned weights not found at %s — falling back to pretrained %s "
        "(will not detect damage classes until the model is trained).",
        settings.model_weights_path,
        settings.fallback_weights_path,
    )
    return settings.fallback_weights_path


def get_model() -> YOLO:
    global _model, _weights_path_used
    if _model is None:
        with _model_lock:
            if _model is None:
                settings = get_settings()
                weights_path = _resolve_weights_path()
                model = YOLO(weights_path)
                if settings.device != "cpu":
                    model.to(settings.device)
                _model = model
                _weights_path_used = weights_path
    return _model


def get_weights_path_used() -> str | None:
    return _weights_path_used


def reload_model() -> None:
    global _model, _weights_path_used
    with _model_lock:
        _model = None
        _weights_path_used = None


def _decode_image(image: bytes | np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    if isinstance(image, Image.Image):
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    if isinstance(image, bytes):
        array = np.frombuffer(image, dtype=np.uint8)
        decoded = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("corrupt or unsupported image")
        return decoded
    raise TypeError(f"unsupported image type: {type(image)!r}")


def run_inference(
    image: bytes | np.ndarray | Image.Image,
    conf: float | None = None,
    iou: float | None = None,
) -> list[Results]:
    settings = get_settings()
    decoded = _decode_image(image)
    model = get_model()
    return model(
        decoded,
        conf=conf if conf is not None else settings.confidence_threshold,
        iou=iou if iou is not None else settings.iou_threshold,
        verbose=False,
    )
