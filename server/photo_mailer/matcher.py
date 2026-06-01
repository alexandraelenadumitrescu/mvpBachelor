import os
import numpy as np
from PIL import Image
from deepface import DeepFace
from photo_mailer import tflite_embedder

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
THRESHOLD = 0.10


def match_photos(
    photos_dir: str,
    db: dict[str, np.ndarray],
    threshold: float = THRESHOLD,
) -> dict[str, list[str]]:
    """
    Detect faces in each event photo, embed with FaceNet TFLite, match against DB.
    Returns {email: [photo_paths]} for every employee with at least one match.
    """
    results: dict[str, list[str]] = {}
    photo_files = [
        os.path.join(photos_dir, f)
        for f in os.listdir(photos_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    for photo_path in photo_files:
        print(f"  Scanning {os.path.basename(photo_path)}...")
        try:
            faces = DeepFace.extract_faces(
                img_path=photo_path,
                enforce_detection=False,
                detector_backend="opencv",
            )
        except Exception as exc:
            print(f"    ✗ skipped: {exc}")
            continue

        for face in faces:
            face_img = Image.fromarray((face["face"] * 255).astype(np.uint8))
            embedding = tflite_embedder.embed(face_img)
            best_email, best_sim = _best_match(embedding, db)
            print(f"    best match: {best_email} (sim={best_sim:.3f}, threshold={threshold})")
            if best_sim >= threshold:
                results.setdefault(best_email, [])
                if photo_path not in results[best_email]:
                    results[best_email].append(photo_path)
                print(f"    ✓ matched {best_email} (sim={best_sim:.3f})")

    return results


def _best_match(embedding: np.ndarray, db: dict[str, np.ndarray]) -> tuple[str, float]:
    best_email, best_sim = "", -1.0
    for email, db_emb in db.items():
        sim = float(np.dot(embedding, db_emb))
        if sim > best_sim:
            best_sim, best_email = sim, email
    return best_email, best_sim
