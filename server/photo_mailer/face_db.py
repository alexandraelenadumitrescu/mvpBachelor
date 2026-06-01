import os
import tempfile
import requests
import numpy as np
from PIL import Image
from deepface import DeepFace
from photo_mailer import tflite_embedder


def build_db(employees: list[dict]) -> dict[str, np.ndarray]:
    """Download each employee photo, detect face, embed with FaceNet TFLite."""
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
