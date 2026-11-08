"""
evaluate_scraper_models.py — Assessment rapid al modelelor de detectie folosite
in pipeline-ul scraper (Etapa 1 din docs/etape_scraper_blur.md).

Ce testeaza:
  1. Pipeline-ul de PRODUCTIE (local_detector.detect_sensitive): YOLOv8n COCO
     (screen/document) + badge_detector v1 in ramura YOLO fine-tuned
     (BADGE_MODEL_PATH e setat in .env) — pe cele 10 imagini deja adunate in
     badge_test_outputs/raw + screen_test_outputs/before_detection.
  2. Comparatie badge detector v1 (contur, fallback fara model) vs v2 vs v3,
     apeland direct functiile interne _detect_contours / _detect_contours_v2 /
     _detect_face_guided, ca sa ocolim ramura YOLO si sa vedem cum s-ar
     comporta fara badge_model.pt.

Nu calculeaza precision/recall in sensul strict (nu exista ground-truth cu
cutii adnotate) — produce numar de regiuni + confidence per imagine si
imagini adnotate, pentru inspectie vizuala. Vezi docs/model_assessment_results.md
pentru interpretare.

Run:  python evaluate_scraper_models.py
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
OUT_DIR = os.path.join(ROOT, "evaluation_assets", "scraper_assessment")
os.makedirs(OUT_DIR, exist_ok=True)

BADGE_IMAGES = [
    os.path.join(ROOT, "badge_test_outputs", "raw", f)
    for f in os.listdir(os.path.join(ROOT, "badge_test_outputs", "raw"))
]
SCREEN_IMAGES = [
    os.path.join(ROOT, "screen_test_outputs", "before_detection", f)
    for f in os.listdir(os.path.join(ROOT, "screen_test_outputs", "before_detection"))
]

AMBER = (76, 168, 201)   # BGR for cv2 (approx #C9A84C)
RED   = (85, 85, 255)
BLUE  = (239, 175, 97)


def draw_regions(img_bgr, regions, color=AMBER):
    out = img_bgr.copy()
    for r in regions:
        x, y, w, h = int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])
        cv2.rectangle(out, (x, y), (x + w, y + h), color, max(2, out.shape[1] // 400))
        label = f'{r.get("label", "?")} {r.get("conf", "")}'
        cv2.putText(out, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    out.shape[1] / 1800, color, max(1, out.shape[1] // 700))
    return out


def save_annotated(path, img_bgr, regions, suffix):
    name = os.path.splitext(os.path.basename(path))[0]
    out = draw_regions(img_bgr, regions)
    out_path = os.path.join(OUT_DIR, f"{name}__{suffix}.jpg")
    cv2.imwrite(out_path, out)
    return out_path


# ── 1. Pipeline de productie (local_detector) ───────────────────────────────

def run_production_pipeline():
    from blur_api.local_detector import detect_sensitive

    results = []
    for path in BADGE_IMAGES + SCREEN_IMAGES:
        with open(path, "rb") as f:
            raw = f.read()
        t0 = time.perf_counter()
        regions = detect_sensitive(raw)
        elapsed = time.perf_counter() - t0

        img_bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        out_path = save_annotated(path, img_bgr, regions, "production")

        results.append({
            "image": os.path.basename(path),
            "n_regions": len(regions),
            "labels": [r["label"] for r in regions],
            "elapsed_s": round(elapsed, 3),
            "annotated": out_path,
        })
        print(f"  [production] {os.path.basename(path):35s} "
              f"{len(regions)} regiuni  ({elapsed:.2f}s)  {[r['label'] for r in regions]}")
    return results


# ── 2. Comparatie badge detector v1 (contur) / v2 / v3 ──────────────────────

def run_badge_variant_comparison():
    # Import direct din blur_api ca sa apelam ramurile interne fara YOLO
    sys.path.insert(0, os.path.join(ROOT, "server"))
    from blur_api import badge_detector as bd_v1
    from blur_api import badge_detector_v2 as bd_v2
    from blur_api import badge_detector_v3 as bd_v3

    variants = {
        "v1_contour": bd_v1._detect_contours,
        "v2_contour_clahe": bd_v2._detect_contours_v2,
        "v3_face_guided": bd_v3._detect_face_guided,
    }
    # v1 YOLO fine-tuned = ce ruleaza deja in "production" de mai sus (badge images)
    from blur_api.badge_detector import detect_badges as v1_yolo

    results = []
    for path in BADGE_IMAGES:
        with open(path, "rb") as f:
            raw = f.read()
        img_bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)

        row = {"image": os.path.basename(path)}

        t0 = time.perf_counter()
        regions_yolo = v1_yolo(raw)
        row["v1_yolo_finetuned"] = len(regions_yolo)
        row["v1_yolo_finetuned_time_s"] = round(time.perf_counter() - t0, 3)
        save_annotated(path, img_bgr, regions_yolo, "v1_yolo_finetuned")

        for name, fn in variants.items():
            t0 = time.perf_counter()
            try:
                regions = fn(raw)
            except Exception as exc:
                print(f"    [WARN] {name} failed on {path}: {exc}")
                regions = []
            elapsed = time.perf_counter() - t0
            row[name] = len(regions)
            row[f"{name}_time_s"] = round(elapsed, 3)
            save_annotated(path, img_bgr, regions, name)

        print(f"  [badge-variants] {row['image']:30s} "
              f"yolo_ft={row['v1_yolo_finetuned']}  "
              f"v1_contour={row['v1_contour']}  "
              f"v2_clahe={row['v2_contour_clahe']}  "
              f"v3_face={row['v3_face_guided']}")
        results.append(row)
    return results


def main():
    print("=" * 70)
    print("1) Pipeline de productie (YOLOv8n COCO + badge YOLO fine-tuned)")
    print("=" * 70)
    production = run_production_pipeline()

    print()
    print("=" * 70)
    print("2) Comparatie badge detector: v1 YOLO vs v1/v2/v3 contur")
    print("=" * 70)
    badge_variants = run_badge_variant_comparison()

    summary = {"production": production, "badge_variants": badge_variants}
    summary_path = os.path.join(OUT_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nRezultate salvate in: {OUT_DIR}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
