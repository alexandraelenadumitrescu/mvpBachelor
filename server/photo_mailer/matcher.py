import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from deepface import DeepFace

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
THRESHOLD = 0.10


def match_photos(
    photos_dir: str,
    db: dict[str, list[float]],
    threshold: float = THRESHOLD,
) -> dict[str, list[str]]:
    results: dict[str, list[str]] = {}
    lock = threading.Lock()

    photo_files = [
        os.path.join(photos_dir, f)
        for f in os.listdir(photos_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    if not photo_files:
        print(f"No images found in {photos_dir}")
        return results

    def process_photo(photo_path: str) -> dict[str, list[str]]:
        local: dict[str, list[str]] = {}
        print(f"  Scanning {os.path.basename(photo_path)}...")
        try:
            faces = DeepFace.represent(
                img_path=photo_path,
                model_name="Facenet",
                enforce_detection=False,
            )
        except Exception as exc:
            print(f"    ✗ skipped: {exc}")
            return local
        for face in faces:
            best_email, best_sim = _best_match(face["embedding"], db)
            print(f"    best match: {best_email} (sim={best_sim:.3f}, threshold={threshold})")
            if best_sim >= threshold:
                local.setdefault(best_email, [])
                if photo_path not in local[best_email]:
                    local[best_email].append(photo_path)
                print(f"    ✓ matched {best_email} (sim={best_sim:.3f})")
        return local

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(process_photo, p): p for p in photo_files}
        for future in as_completed(futures):
            with lock:
                for email, paths in future.result().items():
                    results.setdefault(email, [])
                    for p in paths:
                        if p not in results[email]:
                            results[email].append(p)

    return results


def _best_match(
    embedding: list[float],
    db: dict[str, list[float]],
) -> tuple[str, float]:
    best_email = ""
    best_sim   = -1.0
    for email, db_emb in db.items():
        sim = _cosine_sim(embedding, db_emb)
        if sim > best_sim:
            best_sim   = sim
            best_email = email
    return best_email, best_sim


def _cosine_sim(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0
