"""
evaluate_badge_prototypes.py — prototipuri noi de detectie badge, comparate cu
v1 (YOLO fine-tuned, activ in productie) / v1 contur / v2 CLAHE+contur / v3
face-guided (vezi evaluate_scraper_models.py si docs/model_assessment_results.md).

Prototipuri noi testate:
  A. person_anchor  — inlocuieste Haar face cascade din v3 cu clasa COCO
                       "person" (YOLOv8n), ROI de piept derivat din bbox
                       persoanei intregi, apoi acelasi pipeline de filtre
                       (CLAHE, Canny adaptiv, solidity, rectangularitate,
                       luminozitate, saturatie, textura) ca in v3.
  B. color_shape     — fara nicio ancora (fara fata/persoana): cauta global
                       blob-uri deschise la culoare, dreptunghiulare, cu
                       contrast fata de fundal (MSER).
  C. yolo_world      — YOLO-World zero-shot, text prompts ("id badge",
                       "lanyard", "name tag", "employee badge card"), fara
                       antrenare, folosind acelasi pachet ultralytics deja
                       instalat.

Set de test: 5 imagini cu badge (recall) + 2 imagini de birou fara badge
(pexels_graph_monitors_office, pexels_workspace_monitor_laptop — cele care au
generat cele mai multe fals-pozitive in assessment-ul anterior) ca sa vedem
rata de fals-pozitive a fiecarei metode noi.

Run:  python evaluate_badge_prototypes.py
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "server"))

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "evaluation_assets", "badge_prototypes")
os.makedirs(OUT_DIR, exist_ok=True)

BADGE_DIR = os.path.join(ROOT, "badge_test_outputs", "raw")
BADGE_IMAGES = [os.path.join(BADGE_DIR, f) for f in os.listdir(BADGE_DIR)]

NEGATIVE_CONTROL_DIR = os.path.join(ROOT, "screen_test_outputs", "before_detection")
NEGATIVE_IMAGES = [
    os.path.join(NEGATIVE_CONTROL_DIR, f)
    for f in ["pexels_graph_monitors_office.jpg", "pexels_workspace_monitor_laptop.jpg"]
]

ALL_IMAGES = BADGE_IMAGES + NEGATIVE_IMAGES

AMBER = (76, 168, 201)


def draw_regions(img_bgr, regions, color=AMBER):
    out = img_bgr.copy()
    for r in regions:
        x, y, w, h = int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])
        cv2.rectangle(out, (x, y), (x + w, y + h), color, max(2, out.shape[1] // 400))
        label = str(r.get("label", "?"))
        cv2.putText(out, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    out.shape[1] / 1800, color, max(1, out.shape[1] // 700))
    return out


def save_annotated(path, img_bgr, regions, suffix):
    name = os.path.splitext(os.path.basename(path))[0]
    out = draw_regions(img_bgr, regions)
    out_path = os.path.join(OUT_DIR, f"{name}__{suffix}.jpg")
    cv2.imwrite(out_path, out)
    return out_path


def _overlaps(x, y, w, h, seen, threshold=0.5):
    for sx, sy, sw, sh in seen:
        ix = max(0, min(x + w, sx + sw) - max(x, sx))
        iy = max(0, min(y + h, sy + sh) - max(y, sy))
        inter = ix * iy
        union = w * h + sw * sh - inter
        if union > 0 and inter / union > threshold:
            return True
    return False


# ── A. person_anchor: YOLO "person" ROI + filtrele geometrice din v3 ────────

_person_model = None


def _get_person_model():
    global _person_model
    if _person_model is None:
        from ultralytics import YOLO
        _person_model = YOLO(os.path.join(ROOT, "server", "yolov8n.pt"))
    return _person_model


def detect_person_anchor(image_bytes: bytes) -> list[dict]:
    img = np.array(Image.open(__import__("io").BytesIO(image_bytes)).convert("RGB"))
    ih, iw = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    results = _get_person_model()(img, classes=[0], verbose=False)  # 0 = person
    boxes = results[0].boxes

    regions = []
    seen = []

    for box in boxes:
        if float(box.conf[0]) < 0.3:
            continue
        px1, py1, px2, py2 = map(int, box.xyxy[0])
        pw, ph = px2 - px1, py2 - py1
        if pw < 20 or ph < 20:
            continue

        # ROI de piept: 15%-50% din inaltimea persoanei, sub zona capului
        roi_x = max(0, px1)
        roi_y = max(0, py1 + int(ph * 0.15))
        roi_x2 = min(iw, px2)
        roi_y2 = min(ih, py1 + int(ph * 0.50))
        roi_w, roi_h = roi_x2 - roi_x, roi_y2 - roi_y
        if roi_w < 20 or roi_h < 20:
            continue

        roi_gray = gray[roi_y:roi_y2, roi_x:roi_x2]
        roi_color = img[roi_y:roi_y2, roi_x:roi_x2]
        roi_area = roi_w * roi_h

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        eq = clahe.apply(roi_gray)
        blurred = cv2.GaussianBlur(eq, (5, 5), 0)
        v = float(np.median(blurred))
        lower, upper = max(0, int(0.66 * v)), min(255, int(1.33 * v))
        edges = cv2.Canny(blurred, lower, upper)
        dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        min_area, max_area = roi_area * 0.015, roi_area * 0.35

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_area < area < max_area):
                continue
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            hull_peri = cv2.arcLength(hull, True)
            hull_approx = cv2.approxPolyDP(hull, 0.04 * hull_peri, True)
            if not (3 <= len(hull_approx) <= 6):
                continue
            rect = cv2.minAreaRect(hull)
            bw, bh = sorted(rect[1])
            if bh == 0:
                continue
            ratio = bh / bw
            if not (1.2 <= ratio <= 2.5):
                continue
            box_pts = cv2.boxPoints(rect)
            rx, ry, rw, rh = cv2.boundingRect(box_pts.astype(np.int32))
            rx, ry = max(0, rx), max(0, ry)
            rw = min(rw, roi_w - rx)
            rh = min(rh, roi_h - ry)
            if rw == 0 or rh == 0:
                continue
            rect_area = bw * bh
            if rect_area == 0 or (hull_area / rect_area) < 0.58:
                continue
            if (area / hull_area) < 0.50:
                continue
            if float(eq[ry:ry + rh, rx:rx + rw].mean()) < 80:
                continue
            roi_hsv = cv2.cvtColor(roi_color[ry:ry + rh, rx:rx + rw], cv2.COLOR_RGB2HSV)
            if float(roi_hsv[:, :, 1].mean()) > 90:
                continue
            lap_var = cv2.Laplacian(roi_gray[ry:ry + rh, rx:rx + rw], cv2.CV_64F).var()
            if lap_var < 25:
                continue

            abs_x, abs_y = roi_x + rx, roi_y + ry
            if _overlaps(abs_x, abs_y, rw, rh, seen):
                continue
            seen.append((abs_x, abs_y, rw, rh))
            regions.append({"label": "badge_person_anchor", "x": abs_x, "y": abs_y, "w": rw, "h": rh})

    return regions


# ── B. color_shape: MSER global, fara ancora ────────────────────────────────

def detect_color_shape(image_bytes: bytes) -> list[dict]:
    img = np.array(Image.open(__import__("io").BytesIO(image_bytes)).convert("RGB"))
    ih, iw = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    mser = cv2.MSER_create()
    mser.setMinArea(int(iw * ih * 0.0015))
    mser.setMaxArea(int(iw * ih * 0.08))
    regions_raw, _ = mser.detectRegions(gray)

    regions = []
    seen = []
    for pts in regions_raw:
        x, y, w, h = cv2.boundingRect(pts.reshape(-1, 1, 2))
        if w == 0 or h == 0:
            continue
        ratio = max(w, h) / min(w, h)
        if not (1.2 <= ratio <= 2.5):
            continue
        roi_gray = gray[y:y + h, x:x + w]
        roi_hsv = hsv[y:y + h, x:x + w]
        # card = deschis la culoare, saturatie mica, contrast cu vecinatatea
        if roi_gray.mean() < 140:
            continue
        if roi_hsv[:, :, 1].mean() > 70:
            continue
        # contrast cu fundalul imediat (margine de 10% in jurul cutiei)
        pad = max(4, int(max(w, h) * 0.15))
        by1, by2 = max(0, y - pad), min(ih, y + h + pad)
        bx1, bx2 = max(0, x - pad), min(iw, x + w + pad)
        surround = gray[by1:by2, bx1:bx2]
        if surround.size == 0:
            continue
        contrast = abs(float(roi_gray.mean()) - float(surround.mean()))
        if contrast < 15:
            continue
        if _overlaps(x, y, w, h, seen):
            continue
        seen.append((x, y, w, h))
        regions.append({"label": "badge_color_shape", "x": int(x), "y": int(y), "w": int(w), "h": int(h)})

    return regions


# ── D. face_mser: ancora de fata (v3, dovedita) + cautare MSER in ROI ───────
# In loc de Canny+contur (v3 original), foloseste MSER (care a localizat
# perfect cardul pe poza de studio in metoda B) DAR restrans la ROI-ul de
# piept sub fata, ca sa evite exploziile de fals-pozitive pe scene aglomerate.

_face_cascade = None


def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(xml)
    return _face_cascade


def detect_face_mser(image_bytes: bytes) -> list[dict]:
    img = np.array(Image.open(__import__("io").BytesIO(image_bytes)).convert("RGB"))
    ih, iw = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    cascade = _get_face_cascade()
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=4, minSize=(int(iw * 0.03), int(iw * 0.03))
    )
    if len(faces) == 0:
        return []

    regions = []
    seen = []

    for (fx, fy, fw, fh) in faces:
        roi_x = max(0, fx - int(fw * 0.5))
        roi_y = max(0, fy + int(fh * 1.3))
        roi_x2 = min(iw, fx + fw + int(fw * 0.5))
        roi_y2 = min(ih, fy + int(fh * 2.8))
        roi_w, roi_h = roi_x2 - roi_x, roi_y2 - roi_y
        if roi_w < 20 or roi_h < 20:
            continue

        roi_gray = gray[roi_y:roi_y2, roi_x:roi_x2]
        roi_hsv = cv2.cvtColor(img[roi_y:roi_y2, roi_x:roi_x2], cv2.COLOR_RGB2HSV)
        roi_area = roi_w * roi_h

        mser = cv2.MSER_create()
        mser.setMinArea(max(20, int(roi_area * 0.015)))
        mser.setMaxArea(int(roi_area * 0.5))
        regions_raw, _ = mser.detectRegions(roi_gray)

        for pts in regions_raw:
            x, y, w, h = cv2.boundingRect(pts.reshape(-1, 1, 2))
            if w == 0 or h == 0:
                continue
            ratio = max(w, h) / min(w, h)
            if not (1.2 <= ratio <= 2.5):
                continue
            blob_gray = roi_gray[y:y + h, x:x + w]
            blob_hsv = roi_hsv[y:y + h, x:x + w]
            if blob_gray.mean() < 130:
                continue
            if blob_hsv[:, :, 1].mean() > 80:
                continue
            pad = max(4, int(max(w, h) * 0.15))
            by1, by2 = max(0, y - pad), min(roi_h, y + h + pad)
            bx1, bx2 = max(0, x - pad), min(roi_w, x + w + pad)
            surround = roi_gray[by1:by2, bx1:bx2]
            if surround.size == 0:
                continue
            if abs(float(blob_gray.mean()) - float(surround.mean())) < 12:
                continue

            abs_x, abs_y = roi_x + x, roi_y + y
            if _overlaps(abs_x, abs_y, w, h, seen):
                continue
            seen.append((abs_x, abs_y, w, h))
            regions.append({"label": "badge_face_mser", "x": abs_x, "y": abs_y, "w": w, "h": h})

    return regions


# ── E. face_person_gated: v3 (Canny/contur, dovedit) + poarta de persoana ───
# Fata gasita de Haar e acceptata doar daca centrul ei cade in interiorul unei
# cutii YOLO "person" — elimina fals-pozitivele Haar pe textura organica
# (planta/terrariu, vazut la metoda D) fara sa schimbe pipeline-ul intern
# (Canny+contur+CLAHE) care a functionat corect pe pexels_office_id.

def detect_face_person_gated(image_bytes: bytes) -> list[dict]:
    img = np.array(Image.open(__import__("io").BytesIO(image_bytes)).convert("RGB"))
    ih, iw = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    cascade = _get_face_cascade()
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=4, minSize=(int(iw * 0.03), int(iw * 0.03))
    )
    if len(faces) == 0:
        return []

    person_results = _get_person_model()(img, classes=[0], verbose=False)
    person_boxes = [tuple(map(int, b.xyxy[0])) for b in person_results[0].boxes if float(b.conf[0]) >= 0.3]

    def _face_in_person(fx, fy, fw, fh):
        cx, cy = fx + fw / 2, fy + fh / 2
        for px1, py1, px2, py2 in person_boxes:
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                return True
        return False

    gated_faces = [f for f in faces if _face_in_person(*f)]
    if not gated_faces:
        return []

    regions = []
    seen = []

    for (fx, fy, fw, fh) in gated_faces:
        roi_x = max(0, fx - int(fw * 0.5))
        roi_y = max(0, fy + int(fh * 1.3))
        roi_x2 = min(iw, fx + fw + int(fw * 0.5))
        roi_y2 = min(ih, fy + int(fh * 2.8))
        roi_w, roi_h = roi_x2 - roi_x, roi_y2 - roi_y
        if roi_w < 20 or roi_h < 20:
            continue

        roi_gray = gray[roi_y:roi_y2, roi_x:roi_x2]
        roi_color = img[roi_y:roi_y2, roi_x:roi_x2]
        roi_area = roi_w * roi_h

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        eq = clahe.apply(roi_gray)
        blurred = cv2.GaussianBlur(eq, (5, 5), 0)
        v = float(np.median(blurred))
        lower, upper = max(0, int(0.66 * v)), min(255, int(1.33 * v))
        edges = cv2.Canny(blurred, lower, upper)
        dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        min_area, max_area = roi_area * 0.015, roi_area * 0.35

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_area < area < max_area):
                continue
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            hull_peri = cv2.arcLength(hull, True)
            hull_approx = cv2.approxPolyDP(hull, 0.04 * hull_peri, True)
            if not (3 <= len(hull_approx) <= 6):
                continue
            rect = cv2.minAreaRect(hull)
            bw, bh = sorted(rect[1])
            if bh == 0:
                continue
            ratio = bh / bw
            if not (1.2 <= ratio <= 2.5):
                continue
            box_pts = cv2.boxPoints(rect)
            rx, ry, rw, rh = cv2.boundingRect(box_pts.astype(np.int32))
            rx, ry = max(0, rx), max(0, ry)
            rw = min(rw, roi_w - rx)
            rh = min(rh, roi_h - ry)
            if rw == 0 or rh == 0:
                continue
            rect_area = bw * bh
            if rect_area == 0 or (hull_area / rect_area) < 0.58:
                continue
            if (area / hull_area) < 0.50:
                continue
            if float(eq[ry:ry + rh, rx:rx + rw].mean()) < 80:
                continue
            roi_hsv = cv2.cvtColor(roi_color[ry:ry + rh, rx:rx + rw], cv2.COLOR_RGB2HSV)
            if float(roi_hsv[:, :, 1].mean()) > 90:
                continue
            lap_var = cv2.Laplacian(roi_gray[ry:ry + rh, rx:rx + rw], cv2.CV_64F).var()
            if lap_var < 25:
                continue

            abs_x, abs_y = roi_x + rx, roi_y + ry
            if _overlaps(abs_x, abs_y, rw, rh, seen):
                continue
            seen.append((abs_x, abs_y, rw, rh))
            regions.append({"label": "badge_face_person_gated", "x": abs_x, "y": abs_y, "w": rw, "h": rh})

    return regions


# ── F. ensemble: v2 (global CLAHE contur) + E (face+persoana, gated) + NMS ──

def _nms_union(region_lists: list[list[dict]], iou_threshold: float = 0.4) -> list[dict]:
    """Uneste regiuni din mai multe metode; elimina duplicate suprapuse (IoU>threshold),
    pastrand cutia din prima lista care a produs-o (ordinea listelor = prioritate)."""
    all_regions = []
    for lst in region_lists:
        all_regions.extend(lst)

    kept = []
    for r in all_regions:
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        dup = False
        for k in kept:
            ix = max(0, min(x + w, k["x"] + k["w"]) - max(x, k["x"]))
            iy = max(0, min(y + h, k["y"] + k["h"]) - max(y, k["y"]))
            inter = ix * iy
            union = w * h + k["w"] * k["h"] - inter
            if union > 0 and inter / union > iou_threshold:
                dup = True
                break
        if not dup:
            kept.append(r)
    return kept


def detect_ensemble(image_bytes: bytes) -> list[dict]:
    sys.path.insert(0, os.path.join(ROOT, "server"))
    from blur_api import badge_detector_v2 as bd_v2
    v2_regions = bd_v2._detect_contours_v2(image_bytes)
    e_regions = detect_face_person_gated(image_bytes)
    return _nms_union([e_regions, v2_regions])


# ── C. yolo_world: zero-shot text-prompted ──────────────────────────────────

_yw_model = None
_YW_PROMPTS = ["id badge", "lanyard with card", "employee badge", "name tag on lanyard"]


def _get_yw_model():
    global _yw_model
    if _yw_model is None:
        from ultralytics import YOLOWorld
        _yw_model = YOLOWorld(os.path.join(ROOT, "evaluation_assets", "prototype_models", "yolov8s-world.pt"))
        _yw_model.set_classes(_YW_PROMPTS)
    return _yw_model


def detect_yolo_world(image_bytes: bytes) -> list[dict]:
    img = np.array(Image.open(__import__("io").BytesIO(image_bytes)).convert("RGB"))
    results = _get_yw_model()(img, verbose=False, conf=0.1)
    regions = []
    for box in results[0].boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        regions.append({
            "label": f"{_YW_PROMPTS[cls_id]}:{conf:.2f}",
            "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1,
        })
    return regions


# ── Runner ────────────────────────────────────────────────────────────────

METHODS = {
    "A_person_anchor": detect_person_anchor,
    "B_color_shape": detect_color_shape,
    "C_yolo_world": detect_yolo_world,
    "D_face_mser": detect_face_mser,
    "E_face_person_gated": detect_face_person_gated,
    "F_ensemble": detect_ensemble,
}


def main():
    results = []
    for path in ALL_IMAGES:
        is_negative = path in NEGATIVE_IMAGES
        with open(path, "rb") as f:
            raw = f.read()
        img_bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)

        row = {"image": os.path.basename(path), "is_negative_control": is_negative}
        print(f"\n{os.path.basename(path)} {'[NEGATIVE CONTROL]' if is_negative else ''}")
        for name, fn in METHODS.items():
            t0 = time.perf_counter()
            try:
                regions = fn(raw)
            except Exception as exc:
                print(f"  [WARN] {name} failed: {exc}")
                regions = []
            elapsed = time.perf_counter() - t0
            row[name] = len(regions)
            row[f"{name}_time_s"] = round(elapsed, 3)
            save_annotated(path, img_bgr, regions, name)
            print(f"  {name:20s} {len(regions)} regiuni  ({elapsed:.2f}s)")

        results.append(row)

    summary_path = os.path.join(OUT_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSalvat: {OUT_DIR}\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
