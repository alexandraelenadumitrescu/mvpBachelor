import os

import numpy as np
from deepface import DeepFace

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def match_photos(
    photos_dir: str,
    db: dict[str, list[float]],
    threshold: float = 0.60,
) -> dict[str, list[str]]:
    """
    Detect all faces in each event photo and match against the face DB.
    Returns {email: [photo_paths]} for every employee with at least one match.
    """
    results: dict[str, list[str]] = {}
    photo_files = [
        os.path.join(photos_dir, f)
        for f in os.listdir(photos_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    if not photo_files:
        print(f"No images found in {photos_dir}")
        return results

    for photo_path in photo_files:
        print(f"  Scanning {os.path.basename(photo_path)}...")
        try:
            faces = DeepFace.represent(
                img_path=photo_path,
                model_name="Facenet",
                enforce_detection=False,  # don't crash if no face detected
            )
        except Exception as exc:
            print(f"    ✗ skipped: {exc}")
            continue

        for face in faces:
            embedding = face["embedding"]
            best_email, best_sim = _best_match(embedding, db)
            if best_sim >= threshold:
                results.setdefault(best_email, [])
                if photo_path not in results[best_email]:
                    results[best_email].append(photo_path)
                print(f"    ✓ matched {best_email} (sim={best_sim:.3f})")

    return results


def _best_match(
    embedding: list[float],
    db: dict[str, list[float]],
) -> tuple[str, float]:
    """Return (email, cosine_similarity) for the closest DB entry."""
    best_email = ""
    best_sim   = -1.0
    for email, db_embedding in db.items():
        sim = _cosine_sim(embedding, db_embedding)
        if sim > best_sim:
            best_sim   = sim
            best_email = email
    return best_email, best_sim


def _cosine_sim(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0
