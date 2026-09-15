import logging

import cv2
from ultralytics.engine.results import Results

from app.config import get_settings
from app.schemas import BBox, Detection, DetectionResponse

logger = logging.getLogger(__name__)


def extract_detections(result: Results) -> list[Detection]:
    if result.boxes is None or len(result.boxes) == 0:
        return []

    settings = get_settings()
    detections: list[Detection] = []
    for i in range(len(result.boxes)):
        cls_id = int(result.boxes.cls[i])
        class_name = result.names.get(cls_id, str(cls_id))
        if class_name not in settings.class_names.values():
            logger.warning(
                "Model predicted class %r not present in configured class_names "
                "— config and model weights may be out of sync.",
                class_name,
            )

        confidence = float(result.boxes.conf[i])
        x1, y1, x2, y2 = result.boxes.xyxy[i].tolist()
        polygon = result.masks.xy[i].tolist() if result.masks is not None else None

        detections.append(
            Detection(
                class_name=class_name,
                confidence=confidence,
                bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                polygon=polygon,
            )
        )
    return detections


def build_detection_response(result: Results, filename: str) -> DetectionResponse:
    detections = extract_detections(result)
    height, width = result.orig_shape
    return DetectionResponse(
        filename=filename,
        image_width=width,
        image_height=height,
        count=len(detections),
        detections=detections,
    )


def annotate_image(result: Results) -> bytes:
    annotated = result.plot()
    success, buffer = cv2.imencode(".png", annotated)
    if not success:
        raise ValueError("failed to encode annotated image")
    return buffer.tobytes()
