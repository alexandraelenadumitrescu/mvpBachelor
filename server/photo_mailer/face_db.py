import os
import tempfile
import requests
import numpy as np
from PIL import Image
from deepface import DeepFace
from photo_mailer import tflite_embedder


def build_db_facenet(employees: list[dict]) -> dict[str, list[float]]:
    """Download each employee photo, embed with DeepFace Facenet (used by /delivery/run)."""
    db = {}
    for emp in employees:
        name, email, url = emp["name"], emp["email"], emp["photo_url"]
        print(f"  Processing {name} ({email})...")
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(resp.content)
                tmp_path = tmp.name
            try:
                result = DeepFace.represent(
                    img_path=tmp_path,
                    model_name="Facenet",
                    enforce_detection=False,
                )
                db[email] = result[0]["embedding"]
                print(f"    ✓ embedding computed ({len(db[email])} dims)")
            finally:
                os.unlink(tmp_path)
        except Exception as exc:
            print(f"    ✗ skipped: {exc}")
    return db


def build_db(employees: list[dict]) -> dict[str, np.ndarray]:
    """Download each employee photo, detect face, embed with FaceNet TFLite (used by /delivery/match-embeddings)."""
    db = {}
    for emp in employees:
        name, email, url = emp["name"], emp["email"], emp["photo_url"]
        print(f"  Processing {name} ({email})...")
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(resp.content)
                tmp_path = tmp.name

            try:
                faces = DeepFace.extract_faces(
                    img_path=tmp_path,
                    enforce_detection=False,
                    detector_backend="opencv",
                )
                if not faces:
                    print(f"    ✗ no face detected in photo for {name}")
                    continue
                face_arr = faces[0]["face"]  # float64 array [0,1]
                face_img = Image.fromarray((face_arr * 255).astype(np.uint8))
                embedding = tflite_embedder.embed(face_img)
                db[email] = embedding
                print(f"    ✓ embedding computed ({len(embedding)} dims)")
            finally:
                os.unlink(tmp_path)

        except Exception as exc:
            print(f"    ✗ skipped: {exc}")

    return db
