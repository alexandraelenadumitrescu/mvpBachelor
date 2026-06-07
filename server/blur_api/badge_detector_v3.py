"""
Badge/ID card detector v3 — face-guided search cu suport partial ocluzie/unghi.

Strategie: detecteaza fetele cu Haar cascades, apoi cauta badge-ul
in ROI-ul de sub fiecare fata (zona gat/piept).

Avantaje fata de v1/v2:
  - Zero false positives pe fete (cautarea e explicit SUB fata, nu pe ea)
  - Zero false positives pe fundal (cautare locala, nu globala)
  - Constrange spatial cautarea folosind cunostinta de domeniu

Suport ocluzie/unghi (fata de v3 initial):
  - approxPolyDP pe CONVEX HULL, nu pe conturul brut
    → un badge 40% acoperit are contur U/L cu 10+ varfuri, dar
      hull-ul lui are 3-5 varfuri (patrilater aproape complet)
  - 3-6 colturi acceptate pe hull (in loc de 4-6 pe contur)
  - Solidity relaxata: area/hull_area >= 0.40 (in loc de 0.75)
    → permite badge-uri partial vizibile
  - Rectangularitate pe hull vs bbox: hull_area/(bw*bh) >= 0.55
    → hull-ul unui badge ocludat e tot compact; un blob aleator nu
  - minAreaRect pe hull → estimare unghiuri mai stabila
    pentru forme incomplete

Risc asumat: daca Haar cascade rateaza o fata, badge-ul e ratat.
Ocluzie >60%: greu de detectat fara ML.
"""

import os
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

# Cauta badge_model.pt in acelasi director, cu fallback la env var
_DEFAULT_BADGE_MODEL = os.path.join(os.path.dirname(__file__), "badge_model.pt")
BADGE_MODEL_PATH = os.environ.get("BADGE_MODEL_PATH") or (
    _DEFAULT_BADGE_MODEL if os.path.exists(_DEFAULT_BADGE_MODEL) else None
)

_yolo_model    = None
_face_cascade  = None


def _get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO(BADGE_MODEL_PATH)
    return _yolo_model


def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(xml)
    return _face_cascade


def detect_badges(image_bytes: bytes) -> list[dict]:
    """Detect badge/ID card regions. Returns list of {label, x, y, w, h}."""
    if BADGE_MODEL_PATH and os.path.exists(BADGE_MODEL_PATH):
        return _detect_yolo(image_bytes)
    return _detect_face_guided(image_bytes)


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


def _detect_face_guided(image_bytes: bytes) -> list[dict]:
    img = np.array(Image.open(BytesIO(image_bytes)).convert("RGB"))
    ih, iw = img.shape[:2]
    gray   = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # ── 1. Detectie fete ──────────────────────────────────────────────────────
    cascade = _get_face_cascade()

    # scaleFactor 1.05 = mai precis dar mai lent; minNeighbors 4 = sensibil
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(int(iw * 0.03), int(iw * 0.03)),
    )

    if len(faces) == 0:
        return []

    # ── 2. Pentru fiecare fata, defineste ROI badge ───────────────────────────
    regions = []
    seen    = []

    for (fx, fy, fw, fh) in faces:
        # ROI: porneste sub gat (1.3*fh), nu la barbie (0.8*fh)
        # → evita nasul, barbia, gatul
        # Se opreste la 2.8*fh (mijloc piept), nu 3.3*fh
        # → evita peretii, plantele, decorurile din fundal
        roi_x  = max(0,  fx - int(fw * 0.5))
        roi_y  = max(0,  fy + int(fh * 1.3))
        roi_x2 = min(iw, fx + fw + int(fw * 0.5))
        roi_y2 = min(ih, fy + int(fh * 2.8))
        roi_w  = roi_x2 - roi_x
        roi_h  = roi_y2 - roi_y

        if roi_w < 20 or roi_h < 20:
            continue

        roi_gray  = gray[roi_y:roi_y2, roi_x:roi_x2]
        roi_color = img[roi_y:roi_y2, roi_x:roi_x2]   # pentru HSV/texture
        roi_area  = roi_w * roi_h

        # ── 3. Pipeline detectie in ROI ──────────────────────────────────────
        clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        eq      = clahe.apply(roi_gray)
        blurred = cv2.GaussianBlur(eq, (5, 5), 0)

        v     = float(np.median(blurred))
        lower = max(0,   int(0.66 * v))
        upper = min(255, int(1.33 * v))
        edges = cv2.Canny(blurred, lower, upper)

        kernel  = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        # Aria relativa la ROI: badge = 1.5%–35%
        min_area = roi_area * 0.015
        max_area = roi_area * 0.35

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_area < area < max_area):
                continue

            hull      = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue

            hull_peri   = cv2.arcLength(hull, True)
            hull_approx = cv2.approxPolyDP(hull, 0.04 * hull_peri, True)
            if not (3 <= len(hull_approx) <= 6):
                continue

            rect   = cv2.minAreaRect(hull)
            bw, bh = sorted(rect[1])
            if bh == 0:
                continue
            ratio = bh / bw
            if not (1.2 <= ratio <= 2.5):
                continue

            box_pts = cv2.boxPoints(rect)
            rx, ry, rw, rh = cv2.boundingRect(box_pts.astype(np.int32))
            rx = max(0, rx);  ry = max(0, ry)
            rw = min(rw, roi_w - rx)
            rh = min(rh, roi_h - ry)
            if rw == 0 or rh == 0:
                continue

            rect_area = bw * bh
            if rect_area == 0 or (hull_area / rect_area) < 0.58:
                continue

            if (area / hull_area) < 0.50:
                continue

            # ── Filtre pe aparenta (culoare + textura) ────────────────────────

            # 1. Luminozitate pe CLAHE > 80
            #    Badge-urile sunt albe/deschise; gatul/plantele sunt mai intunecate
            if float(eq[ry:ry + rh, rx:rx + rw].mean()) < 80:
                continue

            # 2. Saturatie HSV < 90
            #    Badge alb = saturatie mica; piele/plante = saturatie mare
            roi_hsv = cv2.cvtColor(
                roi_color[ry:ry + rh, rx:rx + rw], cv2.COLOR_RGB2HSV
            )
            if float(roi_hsv[:, :, 1].mean()) > 90:
                continue

            # 3. Textura (varianta Laplacian) > 25
            #    Badge-ul are text imprimat = varfuri de frecventa inalta
            #    Gatul / gusha / plantele sunt netede = varianta mica
            lap_var = cv2.Laplacian(
                roi_gray[ry:ry + rh, rx:rx + rw], cv2.CV_64F
            ).var()
            if lap_var < 25:
                continue

            # Coordonate absolute in imaginea originala
            abs_x = roi_x + rx
            abs_y = roi_y + ry

            if _overlaps(abs_x, abs_y, rw, rh, seen):
                continue

            seen.append((abs_x, abs_y, rw, rh))
            regions.append({
                "label": "badge",
                "x": abs_x, "y": abs_y,
                "w": rw,    "h": rh,
            })

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
