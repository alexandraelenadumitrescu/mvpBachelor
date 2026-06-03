import os
import io
import base64
import numpy as np
from PIL import Image
from deepface import DeepFace
from sklearn.cluster import DBSCAN

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def cluster_faces(photos_dir: str, eps: float = 0.45) -> list[dict]:
    """
    Detect all faces in photos_dir, cluster by similarity using DBSCAN.
    Returns list of {cluster_id, faces: [{photo_name, face_image}]}
    """
    all_faces = []

    photo_files = [
        os.path.join(photos_dir, f)
        for f in os.listdir(photos_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    for photo_path in photo_files:
        photo_name = os.path.basename(photo_path)
        try:
            results = DeepFace.represent(
                img_path=photo_path,
                model_name="Facenet",
                enforce_detection=False,
            )
        except Exception as e:
            print(f"  [cluster] skipped {photo_name}: {e}")
            continue

        img = Image.open(photo_path).convert("RGB")
        for r in results:
            embedding = np.array(r["embedding"], dtype=np.float32)
            fa = r.get("facial_area", {})
            all_faces.append({
                "embedding":  embedding,
                "face_b64":   _crop_face_b64(img, fa),
                "photo_name": photo_name,
            })

    if not all_faces:
        return []

    embeddings = np.array([f["embedding"] for f in all_faces], dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    labels = DBSCAN(eps=eps, min_samples=1, metric="cosine").fit(embeddings).labels_

    clusters: dict[int, list] = {}
    for i, label in enumerate(labels):
        clusters.setdefault(int(label), []).append({
            "photo_name": all_faces[i]["photo_name"],
            "face_image": all_faces[i]["face_b64"],
        })

    return [
        {"cluster_id": cid, "faces": faces}
        for cid, faces in sorted(clusters.items())
    ]


def _crop_face_b64(img: Image.Image, fa: dict) -> str:
    x = fa.get("x", 0)
    y = fa.get("y", 0)
    w = fa.get("w", img.width)
    h = fa.get("h", img.height)
    pad = int(min(w, h) * 0.2)
    crop = img.crop((
        max(0, x - pad),
        max(0, y - pad),
        min(img.width,  x + w + pad),
        min(img.height, y + h + pad),
    )).resize((100, 100))
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()
