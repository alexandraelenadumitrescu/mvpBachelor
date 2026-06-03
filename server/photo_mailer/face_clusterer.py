"""
Server-side face clustering — mirrors FaceClusterer.java logic exactly.

Pipeline (same as Android FaceGroupsActivity):
  1. Detect all faces in all photos
  2. Embed each face with FaceNet TFLite (L2-normalised, 128-dim)
  3. Greedy nearest-neighbour clustering (threshold 0.75)
  4. Sort clusters by size (largest first)
"""
import os
import time
import tempfile
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from deepface import DeepFace
from photo_mailer import tflite_embedder
from photo_mailer.face_utils import resize_to_max, crop_with_padding

SUPPORTED_EXTENSIONS  = {".jpg", ".jpeg", ".png"}
CLUSTER_THRESHOLD     = 0.75   # same as FaceClusterer.java FACE_SIMILARITY_THRESHOLD


@dataclass
class FaceItem:
    photo_path: str
    face_idx:   int
    embedding:  np.ndarray   # 128-dim L2-normalised


@dataclass
class FaceCluster:
    faces:    list[FaceItem] = field(default_factory=list)
    centroid: np.ndarray     = field(default=None)   # computed after clustering

    @property
    def photo_paths(self) -> set[str]:
        return {f.photo_path for f in self.faces}


def extract_all_faces(photos_dir: str) -> list[FaceItem]:
    """
    Detect + embed every face in every photo in photos_dir.
    Sequential (DeepFace not thread-safe).
    Mirrors Android: resize 1024px → detect → crop 20% padding → TFLite embed.
    """
    all_faces: list[FaceItem] = []
    photo_files = sorted([
        os.path.join(photos_dir, f)
        for f in os.listdir(photos_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ])

    for photo_path in photo_files:
        name = os.path.basename(photo_path)
        t0 = time.perf_counter()
        try:
            img = resize_to_max(Image.open(photo_path).convert("RGB"), max_side=1024)

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                img.save(tmp, format="JPEG", quality=92)
                tmp_path = tmp.name
            try:
                faces = DeepFace.extract_faces(
                    img_path=tmp_path,
                    enforce_detection=False,
                    detector_backend="opencv",
                )
            finally:
                os.unlink(tmp_path)

            for idx, face in enumerate(faces):
                fa   = face.get("facial_area", {})
                crop = crop_with_padding(img, fa, padding=0.20)
                emb  = tflite_embedder.embed(crop)
                all_faces.append(FaceItem(photo_path=photo_path, face_idx=idx, embedding=emb))

            print(f"  [cluster] {name}: {len(faces)} face(s) in {time.perf_counter()-t0:.2f}s")

        except Exception as exc:
            print(f"  [cluster] ✗ {name} skipped: {exc}")

    return all_faces


def cluster_faces(faces: list[FaceItem], threshold: float = CLUSTER_THRESHOLD) -> list[FaceCluster]:
    """
    Greedy nearest-neighbour clustering — identical logic to FaceClusterer.java.
    Each unassigned face starts a new cluster; all faces with dot-product > threshold
    are added to it (same as Java's dotProduct check with FACE_SIMILARITY_THRESHOLD).
    """
    n        = len(faces)
    assigned = [False] * n
    clusters: list[FaceCluster] = []

    for i in range(n):
        if assigned[i]:
            continue
        cluster = FaceCluster()
        cluster.faces.append(faces[i])
        assigned[i] = True

        for j in range(i + 1, n):
            if not assigned[j]:
                sim = float(np.dot(faces[i].embedding, faces[j].embedding))
                if sim > threshold:
                    cluster.faces.append(faces[j])
                    assigned[j] = True

        clusters.append(cluster)

    # Sort largest first (same as FaceClusterer.java)
    clusters.sort(key=lambda c: -len(c.faces))

    # Compute centroid for each cluster
    for cluster in clusters:
        embs             = np.stack([f.embedding for f in cluster.faces])
        centroid         = embs.mean(axis=0)
        norm             = np.linalg.norm(centroid)
        cluster.centroid = centroid / norm if norm > 0 else centroid

    return clusters


def match_clusters_to_employees(
    clusters:  list[FaceCluster],
    db:        dict[str, np.ndarray],
    threshold: float = 0.70,
) -> dict[str, set[str]]:
    """
    For each cluster, compare centroid to employee DB.
    Returns {email: set_of_photo_paths}.
    """
    results: dict[str, set[str]] = {}

    for i, cluster in enumerate(clusters):
        if cluster.centroid is None:
            continue

        best_email = ""
        best_sim   = -1.0
        for email, db_emb in db.items():
            sim = float(np.dot(cluster.centroid, db_emb))
            if sim > best_sim:
                best_sim   = sim
                best_email = email

        tag = "✓ MATCH" if best_sim >= threshold else "✗"
        print(f"  [cluster {i+1}/{len(clusters)}] "
              f"{len(cluster.faces)} face(s) across {len(cluster.photo_paths)} photo(s) "
              f"→ {best_email} sim={best_sim:.3f} {tag}")

        if best_sim >= threshold:
            if best_email not in results:
                results[best_email] = set()
            results[best_email].update(cluster.photo_paths)

    return results
