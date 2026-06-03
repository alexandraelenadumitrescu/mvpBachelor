"""
Badge/ID card detector using OpenCV contour analysis.
Detects rectangular card-shaped objects (aspect ratio ~1.4–2.2, min area threshold).

TODO: replace or complement with a fine-tuned YOLOv8 model once trained.
      Training: download dataset from Roboflow Universe (search "id card detection"),
      fine-tune with `yolo train model=yolov8n.pt data=dataset.yaml epochs=100`,
      then set BADGE_MODEL_PATH env var to the resulting best.pt path.
"""

import os
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

# If a fine-tuned model path is provided, use it instead of OpenCV
BADGE_MODEL_PATH = os.environ.get("BADGE_MODEL_PATH")

_yolo_model = None


def _get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO(BADGE_MODEL_PATH)
    return _yolo_model


def detect_badges(image_bytes: bytes) -> list[dict]:
    """Detect badge/ID card regions. Returns list of {label, x, y, w, h}."""
    if BADGE_MODEL_PATH and os.path.exists(BADGE_MODEL_PATH):
        return _detect_yolo(image_bytes)
    return _detect_contours(image_bytes)


def _detect_yolo(image_bytes: bytes) -> list[dict]:
    img = np.array(Image.open(BytesIO(image_bytes)).convert("RGB"))
    results = _get_yolo_model()(img, verbose=False)
    regions = []
    for box in results[0].boxes:
        if float(box.conf[0]) < 0.4:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        regions.append({"label": "badge", "x": x1, "y": y1,
                        "w": x2 - x1, "h": y2 - y1})
    return regions


def _detect_contours(image_bytes: bytes) -> list[dict]:
    img = np.array(Image.open(BytesIO(image_bytes)).convert("RGB"))
    h, w = img.shape[:2]
    min_area = (w * h) * 0.003   # cel puțin 0.3% din imagine
    max_area = (w * h) * 0.25    # cel mult 25% din imagine

    gray    = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 30, 100)
    edges   = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    seen    = []  # dedup suprapuneri

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area < area < max_area):
            continue

        rect   = cv2.minAreaRect(cnt)
        bw, bh = sorted(rect[1])          # bw ≤ bh
        if bh == 0:
            continue
        ratio  = bh / bw                  # ≥ 1 always
        if not (1.3 <= ratio <= 2.3):     # card aspect ratio range
            continue

        box_pts = cv2.boxPoints(rect)
        x, y, rw, rh = cv2.boundingRect(box_pts.astype(np.int32))
        x, y = max(0, x), max(0, y)
        rw   = min(rw, w - x)
        rh   = min(rh, h - y)

        # skip if heavily overlapping with already accepted region
        if _overlaps(x, y, rw, rh, seen):
            continue

        seen.append((x, y, rw, rh))
        regions.append({"label": "badge", "x": x, "y": y, "w": rw, "h": rh})

    return regions


def _overlaps(x, y, w, h, seen: list, threshold=0.5) -> bool:
    for sx, sy, sw, sh in seen:
        ix = max(0, min(x + w, sx + sw) - max(x, sx))
        iy = max(0, min(y + h, sy + sh) - max(y, sy))
        inter = ix * iy
        union = w * h + sw * sh - inter
        if union > 0 and inter / union > threshold:
            return True
    return False
