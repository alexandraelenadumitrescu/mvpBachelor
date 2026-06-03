import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np
from PIL import Image
from deepface import DeepFace
from photo_mailer import tflite_embedder

_tflite_lock = threading.Lock()


def build_db_facenet(employees: list[dict]) -> dict[str, list[float]]:
    """Download + embed employee photos with DeepFace Facenet in parallel (used by /delivery/run)."""
    db   = {}
    lock = threading.Lock()

    def _process(emp):
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
                emb = result[0]["embedding"]
                print(f"    ✓ {name}: {len(emb)} dims")
                return email, emb
            finally:
                os.unlink(tmp_path)
        except Exception as exc:
            print(f"    ✗ {name} skipped: {exc}")
            return None, None

    with ThreadPoolExecutor(max_workers=4) as pool:
        for email, emb in pool.map(_process, employees):
            if email is not None:
                with lock:
                    db[email] = emb
    return db


def build_db(employees: list[dict]) -> dict[str, np.ndarray]:
    """Download + embed employee photos with FaceNet TFLite (used by /delivery/match-embeddings)."""
    db = {}

    def _process(emp):
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
                    print(f"    ✗ no face detected for {name}")
                    return None, None
                face_arr = faces[0]["face"]
                face_img = Image.fromarray((face_arr * 255).astype(np.uint8))
                with _tflite_lock:
                    embedding = tflite_embedder.embed(face_img)
                print(f"    ✓ {name}: {len(embedding)} dims")
                return email, embedding
            finally:
                os.unlink(tmp_path)
        except Exception as exc:
            print(f"    ✗ {name} skipped: {exc}")
            return None, None

    # TFLite inference must be serialized (lock inside _process), but downloads are parallel
    with ThreadPoolExecutor(max_workers=4) as pool:
        for email, emb in pool.map(_process, employees):
            if email is not None:
                db[email] = emb
    return db
