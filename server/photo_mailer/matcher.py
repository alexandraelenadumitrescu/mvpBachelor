import os
import time
import threading
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
from deepface import DeepFace
from photo_mailer import tflite_embedder

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
THRESHOLD = 0.70   # cosine similarity on L2-normalized TFLite embeddings
REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "match_report.txt")


def match_photos(
    photos_dir: str,
    db: dict[str, np.ndarray],
    threshold: float = THRESHOLD,
) -> dict[str, list[str]]:
    """
    Detect faces with RetinaFace, embed with FaceNet TFLite (same model as Android),
    match against employee DB. Processes photos in parallel.
    Returns {email: [photo_paths]}.
    """
    results: dict[str, list[str]] = {}
    lock = threading.Lock()

    photo_files = [
        os.path.join(photos_dir, f)
        for f in os.listdir(photos_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    if not photo_files:
        print(f"[matcher] no images found in {photos_dir}")
        return results

    def process_photo(photo_path: str) -> tuple[dict[str, list[str]], list[str]]:
        local: dict[str, list[str]] = {}
        name = os.path.basename(photo_path)
        t0 = time.perf_counter()
        try:
            faces = DeepFace.extract_faces(
                img_path=photo_path,
                enforce_detection=False,
                detector_backend="opencv",
            )
        except Exception as exc:
            lines = [f"  ✗ {name}  SKIPPED: {exc}"]
            print(lines[0])
            return local, lines

        t_detect = time.perf_counter()
        lines = [f"\n--- {name}  ({len(faces)} face(s) detected, {t_detect-t0:.2f}s) ---"]

        for face_idx, face in enumerate(faces):
            face_img   = Image.fromarray((face["face"] * 255).astype(np.uint8))
            embedding  = tflite_embedder.embed(face_img)
            top        = _top_matches(embedding, db, top_k=5)
            best_email, best_sim = top[0]

            lines.append(f"  Face #{face_idx+1}  top-5 matches:")
            for rank, (email, sim) in enumerate(top, 1):
                marker = "✓ MATCH" if sim >= threshold and rank == 1 else ("      " if rank > 1 else "✗     ")
                lines.append(f"    {rank}. {sim:.4f}  {email}  {marker}")

            if best_sim >= threshold:
                local.setdefault(best_email, [])
                if photo_path not in local[best_email]:
                    local[best_email].append(photo_path)

        print("\n".join(lines))
        return local, lines

    report_lines = [
        f"MATCH REPORT — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Photos: {len(photo_files)}  |  Threshold: {threshold}  |  Employees: {len(db)}",
        "=" * 70,
    ]
    report_lock = threading.Lock()

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(process_photo, p): p for p in photo_files}
        for future in as_completed(futures):
            photo_result, photo_lines = future.result()
            with lock:
                for email, paths in photo_result.items():
                    results.setdefault(email, [])
                    for p in paths:
                        if p not in results[email]:
                            results[email].append(p)
            with report_lock:
                report_lines.extend(photo_lines)

    t_total = time.perf_counter() - t_start

    summary = [
        "=" * 70,
        f"SUMMARY: {len(photo_files)} photos in {t_total:.1f}s → {len(results)} person(s) matched",
    ]
    for email, paths in sorted(results.items(), key=lambda x: -len(x[1])):
        summary.append(f"  {email}: {len(paths)} photo(s)")
    report_lines.extend(summary)

    report_text = "\n".join(report_lines)
    print(report_text)
    try:
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report_text + "\n")
        print(f"  [matcher] report saved → {os.path.abspath(REPORT_PATH)}")
    except Exception as e:
        print(f"  [matcher] could not save report: {e}")

    return results


def _top_matches(
    embedding: np.ndarray,
    db: dict[str, np.ndarray],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    scores = [(email, float(np.dot(embedding, db_emb))) for email, db_emb in db.items()]
    scores.sort(key=lambda x: -x[1])
    return scores[:top_k]
