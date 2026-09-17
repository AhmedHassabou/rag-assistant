"""
Vision service (Extended Track).

Runs a YOLO model on an uploaded study-notes page image (scanned page,
diagram, handwritten figure, etc.) and returns detected objects/classes.
These detections are then fused into the RAG prompt as extra context,
e.g. "the page contains: a labeled diagram, a table".
"""
import io
import logging
from typing import List

from PIL import Image

from app.core.config import settings

logger = logging.getLogger("rag_assistant")

_model = None


def load_vision_model():
    """Load the YOLO model once at startup (only if ENABLE_VISION is true)."""
    global _model
    if not settings.ENABLE_VISION:
        logger.info("Vision component disabled (ENABLE_VISION=False).")
        return
    try:
        from ultralytics import YOLO

        logger.info("Loading YOLO model from %s", settings.YOLO_MODEL_PATH)
        _model = YOLO(settings.YOLO_MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load YOLO model (%s). Vision endpoint will be disabled.", exc)
        _model = None


def run_detection(image_bytes: bytes) -> List[str]:
    """Run YOLO inference on raw image bytes, return a list of human-readable
    detection strings, e.g. ['diagram (0.91)', 'table (0.77)']."""
    if _model is None:
        return []

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = _model.predict(image, verbose=False)

    detections = []
    for r in results:
        names = r.names
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            detections.append(f"{names[cls_id]} ({conf:.2f})")
    return detections
