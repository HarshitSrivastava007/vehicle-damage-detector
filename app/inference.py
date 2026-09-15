import logging

import cv2
import numpy as np
from ultralytics.engine.results import Results

from app.config import get_settings
from app.schemas import BBox, Detection, DetectionResponse

logger = logging.getLogger(__name__)

MASK_ALPHA = 0.45
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2


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
    """Draw masks/boxes/labels in a single configured color, label only (no
    confidence score) — a custom renderer instead of Results.plot(), which
    always colors by class and always prints the confidence."""
    settings = get_settings()
    color_rgb = settings.mask_color_rgb
    color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

    image = result.orig_img.copy()

    if result.masks is not None:
        overlay = image.copy()
        for polygon in result.masks.xy:
            points = polygon.astype(np.int32)
            cv2.fillPoly(overlay, [points], color_bgr)
        image = cv2.addWeighted(overlay, MASK_ALPHA, image, 1 - MASK_ALPHA, 0)

    if result.boxes is not None:
        for i in range(len(result.boxes)):
            x1, y1, x2, y2 = (int(v) for v in result.boxes.xyxy[i].tolist())
            cls_id = int(result.boxes.cls[i])
            label = result.names.get(cls_id, str(cls_id))

            cv2.rectangle(image, (x1, y1), (x2, y2), color_bgr, BOX_THICKNESS)

            (text_w, text_h), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
            label_y1 = max(y1 - text_h - baseline - 4, 0)
            cv2.rectangle(image, (x1, label_y1), (x1 + text_w + 4, y1), color_bgr, -1)
            cv2.putText(
                image,
                label,
                (x1 + 2, y1 - baseline - 2),
                FONT,
                FONT_SCALE,
                (0, 0, 0),
                FONT_THICKNESS,
                cv2.LINE_AA,
            )

    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise ValueError("failed to encode annotated image")
    return buffer.tobytes()
