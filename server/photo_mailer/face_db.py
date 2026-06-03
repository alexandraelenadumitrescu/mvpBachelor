import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

import requests
import numpy as np
from PIL import Image
from deepface import DeepFace
from photo_mailer import tflite_embedder


def build_db(employees: list[dict]) -> dict[str, np.ndarray]:
    """
    Download each employee photo, detect face with RetinaFace,
    embed with FaceNet TFLite — same model as Android.
    Downloads run in parallel; TFLite inference is serialized via lock in tflite_embedder.
    """
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
                    detector_backend="retinaface",
                )
                if not faces:
                    print(f"    ✗ no face detected for {name}")
                    return None, None
                face_arr = faces[0]["face"]
                face_img = Image.fromarray((face_arr * 255).astype(np.uint8))
                embedding = tflite_embedder.embed(face_img)
                print(f"    ✓ {name}: {len(embedding)} dims")
                return email, embedding
            finally:
                os.unlink(tmp_path)
        except Exception as exc:
            print(f"    ✗ {name} skipped: {exc}")
            return None, None

    with ThreadPoolExecutor(max_workers=4) as pool:
        for email, emb in pool.map(_process, employees):
            if email is not None:
                db[email] = emb
    return db
