import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
from deepface import DeepFace
from photo_mailer import tflite_embedder

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
THRESHOLD = 0.60   # cosine similarity on L2-normalized TFLite embeddings


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

    def process_photo(photo_path: str) -> dict[str, list[str]]:
        local: dict[str, list[str]] = {}
        name = os.path.basename(photo_path)
        t0 = time.perf_counter()
        try:
            faces = DeepFace.extract_faces(
                img_path=photo_path,
                enforce_detection=False,
                detector_backend="retinaface",
            )
        except Exception:
            try:
                faces = DeepFace.extract_faces(
                    img_path=photo_path,
                    enforce_detection=False,
                    detector_backend="ssd",
                )
            except Exception as exc:
                print(f"  [matcher] ✗ {name} skipped: {exc}")
                return local

        t_detect = time.perf_counter()
        print(f"  [matcher] {name}: {len(faces)} face(s) in {t_detect - t0:.2f}s")

        for face in faces:
            face_img = Image.fromarray((face["face"] * 255).astype(np.uint8))
            embedding = tflite_embedder.embed(face_img)
            best_email, best_sim = _best_match(embedding, db)
            if best_sim >= threshold:
                local.setdefault(best_email, [])
                if photo_path not in local[best_email]:
                    local[best_email].append(photo_path)
                print(f"    ✓ {best_email} sim={best_sim:.3f}")
            else:
                print(f"    ✗ best={best_email} sim={best_sim:.3f} < {threshold}")
        return local

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(process_photo, p): p for p in photo_files}
        for future in as_completed(futures):
            with lock:
                for email, paths in future.result().items():
                    results.setdefault(email, [])
                    for p in paths:
                        if p not in results[email]:
                            results[email].append(p)

    t_total = time.perf_counter() - t_start
    print(f"  [matcher] done: {len(photo_files)} photos in {t_total:.1f}s "
          f"→ {len(results)} person(s) matched")
    return results


def _best_match(
    embedding: np.ndarray,
    db: dict[str, np.ndarray],
) -> tuple[str, float]:
    best_email = ""
    best_sim   = -1.0
    for email, db_emb in db.items():
        sim = float(np.dot(embedding, db_emb))  # both L2-normalized → cosine sim
        if sim > best_sim:
            best_sim   = sim
            best_email = email
    return best_email, best_sim
