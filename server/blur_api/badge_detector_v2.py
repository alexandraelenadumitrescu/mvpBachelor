"""
Badge/ID card detector v2 — OpenCV cu filtre suplimentare.

Imbunatatiri fata de badge_detector.py (v1):
  1. CLAHE          — egalizare locala de contrast inainte de edge detection
  2. Canny adaptiv  — thresholds derivate din mediana imaginii (robust la lumini mixte)
  3. RETR_LIST      — prinde si contururi interioare (badge-ul de pe piept, nu doar
                      silueta exterioara a persoanei)
  4. approxPolyDP   — accepta 4-6 colturi: dreptunghi real, nu oval/blob
  5. Solidity > 0.75 — aria_contur / aria_convex_hull; fetele/ovalele ~0.70,
                       cardurile rectangulare ~0.85-0.98
  6. Rectangularitate > 0.60 — aria_contur / aria_minAreaRect; filtre dublura
  7. Luminozitate interioara > 60 — badge-urile conferinta sunt deschise la culoare

De ce nu MORPH_CLOSE:
  Un kernel 5x5 topeşte conturul badge-ului mic in silueta persoanei adiacente,
  generand un singur contur uriaş cu rectangularitate ~0.035.
  Dilate simplu (3x3, 2 iter) pastreaza contururile individuale separate.
"""

import os
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

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
    return _detect_contours_v2(image_bytes)


def _detect_yolo(image_bytes: bytes) -> list[dict]:
    img = np.array(Image.open(BytesIO(image_bytes)).convert("RGB"))
    results = _get_yolo_model()(img, verbose=False)
    regions = []
    for box in results[0].boxes:
        if float(box.conf[0]) < 0.25:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        regions.append({"label": "badge", "x": x1, "y": y1,
                        "w": x2 - x1, "h": y2 - y1})
    return regions


def _detect_contours_v2(image_bytes: bytes) -> list[dict]:
    img = np.array(Image.open(BytesIO(image_bytes)).convert("RGB"))
    h, w = img.shape[:2]
    min_area = (w * h) * 0.001
    max_area = (w * h) * 0.25

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 1. CLAHE — scoate marginile badge-urilor din zone cu lumini mixte
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray_eq, (5, 5), 0)

    # 2. Canny adaptiv — thresholds din mediana, nu valori fixe
    v     = float(np.median(blurred))
    lower = max(0,   int(0.66 * v))
    upper = min(255, int(1.33 * v))
    edges = cv2.Canny(blurred, lower, upper)

    # 3. Dilate simplu (nu MORPH_CLOSE) — pastreaza contururi individuale separate
    kernel  = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # RETR_LIST: prinde toate contururile, inclusiv cele interioare (badge pe piept)
    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    seen    = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area < area < max_area):
            continue

        # 4. approxPolyDP — 4-6 colturi: patrilater, nu oval sau blob
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if not (4 <= len(approx) <= 6):
            continue

        rect   = cv2.minAreaRect(cnt)
        bw, bh = sorted(rect[1])
        if bh == 0:
            continue
        ratio = bh / bw
        if not (1.2 <= ratio <= 2.5):
            continue

        box_pts = cv2.boxPoints(rect)
        x, y, rw, rh = cv2.boundingRect(box_pts.astype(np.int32))
        x, y = max(0, x), max(0, y)
        rw   = min(rw, w - x)
        rh   = min(rh, h - y)
        if rw == 0 or rh == 0:
            continue

        # 5. Solidity — aria_contur / aria_convex_hull
        #    dreptunghi: ~0.85-0.98  |  oval/fata: ~0.70-0.78  |  L/U shapes: <0.70
        hull      = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity  = area / hull_area if hull_area > 0 else 0
        if solidity < 0.75:
            continue

        # 6. Rectangularitate — aria_contur / aria_minAreaRect
        #    un card solid: ~0.75-0.95  |  contur tip rama: <0.40
        rect_area      = bw * bh
        rectangularity = area / rect_area if rect_area > 0 else 0
        if rectangularity < 0.55:
            continue

        # 7. Luminozitate interioara — badge-urile conferinta sunt albe/deschise
        roi_mean = float(gray[y:y + rh, x:x + rw].mean())
        if roi_mean < 60:
            continue

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
