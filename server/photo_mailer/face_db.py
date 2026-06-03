import os
import tempfile

import requests
import numpy as np
from PIL import Image
from deepface import DeepFace
from photo_mailer import tflite_embedder
from photo_mailer.face_utils import resize_to_max, crop_with_padding


def build_db(employees: list[dict]) -> dict[str, np.ndarray]:
    """
    Download each employee photo, detect face, embed with FaceNet TFLite.
    Mirrors Android FaceGroupsActivity pipeline:
      - resize to 1024px max (same as decodeBitmap(uri, 1024))
      - crop with 20% padding (same as cropFace(bmp, box, 0.20f))
      - embed with facenet.tflite (same model)
    Sequential — DeepFace is not thread-safe.
    """
    db = {}
    for emp in employees:
        name, email, url = emp["name"], emp["email"], emp["photo_url"]
        print(f"  Processing {name} ({email})...")
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()

            img = resize_to_max(
                Image.open(__import__("io").BytesIO(resp.content)).convert("RGB"),
                max_side=1024,
            )

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                img.save(tmp, format="JPEG", quality=92)
                tmp_path = tmp.name

            try:
                faces = DeepFace.extract_faces(
                    img_path=tmp_path,
                    enforce_detection=False,
                    detector_backend="opencv",
                )
                if not faces:
                    print(f"    ✗ no face detected for {name}")
                    continue

                fa      = faces[0].get("facial_area", {})
                crop    = crop_with_padding(img, fa, padding=0.20)
                emb     = tflite_embedder.embed(crop)
                db[email] = emb
                print(f"    ✓ {name}: {len(emb)} dims")
            finally:
                os.unlink(tmp_path)

        except Exception as exc:
            print(f"    ✗ {name} skipped: {exc}")

    return db
