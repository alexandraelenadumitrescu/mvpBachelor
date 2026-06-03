import os
from io import BytesIO

import numpy as np
from PIL import Image
from ultralytics import YOLO
from blur_api.badge_detector import detect_badges

_SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_model = YOLO(os.path.join(_SERVER_DIR, "yolov8n.pt"))

# COCO class IDs that map to sensitive content (no faces — user preference)
SENSITIVE_CLASSES = {
    62: "screen",    # tv
    63: "screen",    # laptop
    67: "screen",    # cell phone
    73: "document",  # book
}

CONFIDENCE_THRESHOLD = 0.4


def detect_sensitive(image_bytes: bytes) -> list[dict]:
    """Detect sensitive regions locally with YOLOv8 nano. Same interface as gemini.py."""
    img = np.array(Image.open(BytesIO(image_bytes)).convert("RGB"))
    results = _model(img, verbose=False)

    regions = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        if cls_id not in SENSITIVE_CLASSES:
            continue
        if float(box.conf[0]) < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        regions.append({
            "label": SENSITIVE_CLASSES[cls_id],
            "x": x1,
            "y": y1,
            "w": x2 - x1,
            "h": y2 - y1,
        })

    regions += detect_badges(image_bytes)
    return regions
