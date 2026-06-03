"""
PhotoMatch FastAPI Server — hybrid retrieval + CLAHE + 3D LUT
Run: uvicorn server:app --host 0.0.0.0 --port 8000

Expected folder layout (all relative to this file):
  server/
    server.py
    hybrid_vectors.npz
    defect_head.pt
    images/
      raw/      ← FiveK RAW JPEGs
      edited/   ← expert-edited JPEGs

Install: pip install -r requirements.txt
"""

import os
import sys
import uuid
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
import base64
import threading
import time
import statistics
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from io import BytesIO
from pathlib import Path

import cv2
import faiss
import tempfile
import numpy as np
import open_clip
import torch
import torch.nn as nn
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response

from blur_api.blur import apply_blur
from blur_api.gemini import detect_sensitive as _detect_gemini
from blur_api.local_detector import detect_sensitive as _detect_local
from photo_mailer.scraper  import scrape_employees as _scrape
from photo_mailer.face_db  import build_db          as _build_db
from photo_mailer.face_db  import build_db_facenet  as _build_db_facenet
from photo_mailer.matcher  import match_photos       as _match_photos
from photo_mailer.cluster  import cluster_faces      as _cluster_faces
from PIL import Image
from pydantic import BaseModel
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import map_coordinates
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score as sk_silhouette
from torchvision import models, transforms

# ============================================================
# CONFIG
# ============================================================

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
RAW_DIR      = os.path.join(BASE_DIR, "images", "raw")
EDITED_DIR   = os.path.join(BASE_DIR, "images", "edited")
VECTORS_PATH  = os.path.join(BASE_DIR, "hybrid_vectors.npz")
LUT_CACHE_PATH = os.path.join(BASE_DIR, "lut_cache.npz")
MODEL_PATH   = os.path.join(BASE_DIR, "defect_head.pt")

CLIP_WEIGHT   = 1.0
DEFECT_WEIGHT = 5.0
LUT_STRENGTH  = 0.2
LUT_SIZE      = 17
DEFECT_NAMES  = ["blur", "noise", "overexposure", "underexposure", "compression"]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ============================================================
# STARTUP CHECKS
# ============================================================

_search_ready = True

if not os.path.exists(VECTORS_PATH):
    print(f"WARNING: hybrid_vectors.npz not found — /search endpoint disabled", file=sys.stderr)
    _search_ready = False

if not os.path.exists(MODEL_PATH):
    print(f"WARNING: defect_head.pt not found — /search endpoint disabled", file=sys.stderr)
    _search_ready = False

# ============================================================
# LOAD MODELS
# ============================================================

clip_model = clip_preprocess = backbone = defect_transform = None

if _search_ready:
    print("Loading CLIP ViT-B-32...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    clip_model = clip_model.to(device).eval()
    print("✅ CLIP loaded")

    print("Loading MobileNetV3Small defect head...")
    backbone = models.mobilenet_v3_small(weights=None)
    in_features = backbone.classifier[0].in_features
    backbone.classifier = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.Hardswish(),
        nn.Dropout(0.2),
        nn.Linear(128, 5),
        nn.Sigmoid(),
    )
    backbone.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    backbone = backbone.to(device).eval()
    print("✅ MobileNetV3Small defect head loaded")

defect_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ============================================================
# LOAD VECTORS + BUILD FAISS INDEX
# ============================================================

hybrid_matrix = weighted_matrix = weighted_matrix_norm = None
image_names: list = []
name_to_idx: dict = {}
index = None

if _search_ready:
    print("Loading hybrid vectors...")
    data          = np.load(VECTORS_PATH, allow_pickle=True)
    hybrid_matrix = data["vectors"].astype("float32")
    image_names   = data["names"].tolist()
    print(f"Loaded {len(image_names)} vectors of dimension {hybrid_matrix.shape[1]}")

    clip_part   = hybrid_matrix[:, :512]
    defect_part = hybrid_matrix[:, 512:]
    clip_norm   = clip_part   / (np.linalg.norm(clip_part,   axis=1, keepdims=True) + 1e-8)
    defect_norm = defect_part / (np.linalg.norm(defect_part, axis=1, keepdims=True) + 1e-8)
    weighted_matrix = np.concatenate(
        [clip_norm * CLIP_WEIGHT, defect_norm * DEFECT_WEIGHT], axis=1
    ).astype("float32")

    final_norms = np.linalg.norm(weighted_matrix, axis=1, keepdims=True)
    weighted_matrix_norm = weighted_matrix / (final_norms + 1e-8)
    name_to_idx = {name: i for i, name in enumerate(image_names)}

    index = faiss.IndexFlatIP(weighted_matrix_norm.shape[1])
    index.add(weighted_matrix_norm)
    print(f"✅ FAISS index ready with {index.ntotal} vectors (dim={weighted_matrix_norm.shape[1]})")

# ============================================================
# LUT CACHE (pre-computed in background)
# ============================================================

# Set this event to pause background precomputation threads during CPU/IO-heavy operations
_bg_pause = threading.Event()  # set = pause, clear = run

# Per-URL caches — avoids re-scraping + re-embedding on repeated delivery calls
# Two separate caches because TFLite and Facenet embeddings are in different spaces
_employees_cache:  dict[str, tuple] = {}  # url -> (employees_list, name_by_email)
_face_db_tflite:   dict[str, dict]  = {}  # url -> {email: tflite_embedding}
_face_db_facenet:  dict[str, dict]  = {}  # url -> {email: facenet_embedding}

lut_cache:     dict = {}  # basename -> np.ndarray shape (LUT_SIZE, LUT_SIZE, LUT_SIZE, 3)
style_profiles: dict = {}  # session_id -> np.ndarray shape (N, 517) weighted-normalized

# Load persisted LUT cache from disk if available
if os.path.exists(LUT_CACHE_PATH):
    _data = np.load(LUT_CACHE_PATH)
    lut_cache.update({k: _data[k] for k in _data.files})
    del _data
    print(f"✅ LUT cache loaded from disk: {len(lut_cache)} entries")

# Latency telemetry — rolling window of last 100 requests per endpoint
_latency: dict = {
    "faiss_search_ms":   [],
    "rerank_ms":         [],
    "lut_serialize_ms":  [],
    "lut_compute_ms":    [],
    "search_total_ms":   [],
}
_latency_lock = threading.Lock()

def _record(key: str, ms: float):
    with _latency_lock:
        buf = _latency[key]
        buf.append(ms)
        if len(buf) > 100:
            buf.pop(0)


def _load_image(path: str) -> np.ndarray:
    return np.array(Image.open(path)).astype(np.float32) / 255.0


def _load_image_small(path: str, max_px: int = 512) -> np.ndarray:
    """Load image downsampled to max_px on the long edge — sufficient for aesthetic metrics."""
    img = Image.open(path)
    w, h = img.size
    scale = min(max_px / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    return np.array(img.convert("RGB")).astype(np.float32) / 255.0


def _precompute_luts():
    """Daemon thread: pre-compute and cache 3D LUTs for all RAW→edited pairs."""
    print(f"LUT pre-computation starting for {len(image_names)} pairs...")
    computed = 0
    for name in image_names:
        while _bg_pause.is_set():
            time.sleep(0.5)
        if name in lut_cache:  # skip if already cached on-demand
            continue
        raw_path    = os.path.join(RAW_DIR,    name + ".jpg")
        edited_path = os.path.join(EDITED_DIR, name + ".jpg")
        if not os.path.exists(raw_path) or not os.path.exists(edited_path):
            continue
        try:
            raw    = _load_image(raw_path)
            edited = _load_image(edited_path)
            lut_cache[name] = extract_colour_lut(raw, edited)
            computed += 1
            if computed % 100 == 0:
                print(f"  LUT cache: {computed}/{len(image_names)}")
        except Exception as exc:
            print(f"  LUT warning [{name}]: {exc}")
    print(f"✅ LUT pre-computation done: {computed} LUTs cached")
    if computed > 0 or not os.path.exists(LUT_CACHE_PATH):
        np.savez_compressed(LUT_CACHE_PATH, **lut_cache)
        print(f"✅ LUT cache saved to disk ({len(lut_cache)} entries)")


threading.Thread(target=_precompute_luts, daemon=True).start()
print("LUT pre-computation started in background (server ready)")

aesthetic_cache: dict = {}  # basename -> float score 0.0-1.0


def _precompute_aesthetic_scores():
    """Daemon thread: pre-compute and cache aesthetic scores for all edited reference images."""
    print(f"Aesthetic pre-computation starting for {len(image_names)} images...")
    computed = 0
    for name in image_names:
        while _bg_pause.is_set():
            time.sleep(0.5)
        edited_path = os.path.join(EDITED_DIR, name + ".jpg")
        if not os.path.exists(edited_path):
            continue
        try:
            img = _load_image_small(edited_path)
            aesthetic_cache[name] = compute_aesthetic_score(img)
            computed += 1
            if computed % 100 == 0:
                print(f"  Aesthetic cache: {computed}/{len(image_names)}")
        except Exception as exc:
            print(f"  Aesthetic warning [{name}]: {exc}")
    print(f"✅ Aesthetic pre-computation done: {computed} scores cached")


threading.Thread(target=_precompute_aesthetic_scores, daemon=True).start()
print("Aesthetic pre-computation started in background")


def _warmup_deepface():
    try:
        from deepface import DeepFace as _DF
        import numpy as _np
        print("Warming up DeepFace Facenet...")
        dummy = _np.zeros((160, 160, 3), dtype=_np.uint8)
        _DF.represent(dummy, model_name="Facenet", enforce_detection=False)
        print("✅ DeepFace Facenet warmed up")
    except Exception as e:
        print(f"  DeepFace warmup warning: {e}")

threading.Thread(target=_warmup_deepface, daemon=True).start()


def _auto_k(X: np.ndarray, max_k: int = 8) -> int:
    """Elbow method: pick k where marginal inertia drop is <20% of total drop."""
    n = len(X)
    max_k = min(max_k, max(2, n - 1))
    if n <= 2:
        return min(n, 2)
    inertias = []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        km.fit(X)
        inertias.append(km.inertia_)
    total_drop = (inertias[0] - inertias[-1]) + 1e-8
    for i in range(1, len(inertias)):
        if (inertias[i - 1] - inertias[i]) / total_drop < 0.20:
            return i + 1  # index 0 → k=2, so index i → k=i+2, return i+1 = k
    return max_k

# ============================================================
# APP
# ============================================================

app = FastAPI(title="PhotoMatch Server")

from auth.router import router as auth_router
from auth.dependencies import get_current_active_user
from auth import database, models
models.Base.metadata.create_all(bind=database.engine)
app.include_router(auth_router, prefix="/auth", tags=["auth"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# INTERNAL HELPERS
# ============================================================

def get_hybrid_vector(pil_img):
    """Extract CLIP (512-dim) and defect (5-dim) vectors from a PIL image."""
    img_tensor = clip_preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        clip_vec = clip_model.encode_image(img_tensor)
        clip_vec = clip_vec / clip_vec.norm(dim=-1, keepdim=True)
    clip_vec = clip_vec.cpu().numpy()[0]

    img_tensor2 = defect_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        defect_vec = backbone(img_tensor2).cpu().numpy()[0]

    return clip_vec, defect_vec


def build_hybrid_query(clip_vec: np.ndarray, defect_vec: np.ndarray) -> np.ndarray:
    """Build the 517-dim L2-normalized weighted query vector."""
    clip_v   = clip_vec   / (np.linalg.norm(clip_vec)   + 1e-8) * CLIP_WEIGHT
    defect_v = defect_vec / (np.linalg.norm(defect_vec) + 1e-8) * DEFECT_WEIGHT
    q = np.concatenate([clip_v, defect_v]).astype("float32")
    return q / (np.linalg.norm(q) + 1e-8)


def _style_constrained_retrieve(
    query: np.ndarray,
    style_vecs: np.ndarray,
    D: np.ndarray,
    I: np.ndarray,
) -> tuple:
    """
    Style-region FAISS filtering + semantic guard.
    Returns (faiss_row_idx, similarity, style_matched, style_fallback).
    D/I are pre-computed top-51 FAISS results for `query`.
    """
    valid = [i for i in I[0] if i >= 0]

    centroid = style_vecs.mean(axis=0)
    centroid /= (np.linalg.norm(centroid) + 1e-8)
    sims_to_centroid = style_vecs @ centroid
    radius    = float(1.0 - sims_to_centroid.min())
    threshold = 1.0 - radius * 1.2
    filtered  = [i for i in valid if float(weighted_matrix_norm[i] @ centroid) >= threshold]

    style_matched  = True
    style_fallback = False
    if not filtered:
        filtered      = valid
        style_matched = False

    filtered_arr = np.array(filtered)
    scores       = weighted_matrix_norm[filtered_arr] @ query
    best_local   = int(filtered_arr[np.argmax(scores)])
    similarity   = float(np.max(scores))

    # Semantic guard: CLIP-only cosine sim (weighted_matrix[:,:512] == clip_norm)
    query_clip_norm    = query[:512] / (np.linalg.norm(query[:512]) + 1e-8)
    ref_idx  = name_to_idx[image_names[best_local]]
    clip_sim = float(query_clip_norm @ weighted_matrix[ref_idx, :512])
    if clip_sim < 0.55:
        best_local     = valid[0]
        similarity     = float(D[0][0])
        style_matched  = False
        style_fallback = True

    return best_local, similarity, style_matched, style_fallback


def retrieve_similar(clip_vec, defect_vec, top_k=5):
    """Search the FAISS index using the same weighted scheme as the index."""
    query = build_hybrid_query(clip_vec, defect_vec)
    D, I = index.search(query.reshape(1, -1), top_k + 1)
    return [(image_names[i], float(D[0][j])) for j, i in enumerate(I[0]) if i >= 0]


def _aesthetic_rerank(
    candidates: list[tuple[str, float]],
    aesthetic_weight: float,
) -> tuple[str, float, float]:
    """
    Re-rank FAISS candidates by combined score.
    Returns (best_name, similarity, aesthetic_score).
    Falls back to 0.5 for images not yet in aesthetic_cache.
    """
    w_sim = 1.0 - aesthetic_weight
    w_aes = aesthetic_weight
    best_name  = candidates[0][0]
    best_sim   = candidates[0][1]
    best_aes   = aesthetic_cache.get(candidates[0][0], 0.5)
    best_score = w_sim * best_sim + w_aes * best_aes
    for name, sim in candidates[1:]:
        aes   = aesthetic_cache.get(name, 0.5)
        score = w_sim * sim + w_aes * aes
        if score > best_score:
            best_score = score
            best_name  = name
            best_sim   = sim
            best_aes   = aes
    return best_name, best_sim, best_aes


def correct_clahe(img_np):
    """Apply CLAHE on the L channel of LAB (float32 RGB [0,1] → float32 RGB [0,1])."""
    img_uint8 = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    # Bilateral filter on L only — smooths noise before CLAHE amplifies it
    lab[:, :, 0] = cv2.bilateralFilter(lab[:, :, 0], d=5, sigmaColor=20, sigmaSpace=20)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return result.astype(np.float32) / 255.0


def extract_colour_lut(raw, edited, lut_size=LUT_SIZE):
    """Extract a 3D colour LUT from a RAW→edited image pair."""
    h, w = raw.shape[:2]
    edited_resized = np.array(
        Image.fromarray((edited * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS)
    ).astype(np.float32) / 255.0

    n_samples = min(50000, h * w)
    idx = np.random.choice(h * w, n_samples, replace=False)
    raw_flat    = raw.reshape(-1, 3)[idx]
    edited_flat = edited_resized.reshape(-1, 3)[idx]

    interp_r = LinearNDInterpolator(raw_flat, edited_flat[:, 0], fill_value=0)
    interp_g = LinearNDInterpolator(raw_flat, edited_flat[:, 1], fill_value=0)
    interp_b = LinearNDInterpolator(raw_flat, edited_flat[:, 2], fill_value=0)

    steps = np.linspace(0, 1, lut_size)
    r, g, b = np.meshgrid(steps, steps, steps, indexing="ij")
    grid = np.stack([r.ravel(), g.ravel(), b.ravel()], axis=1).astype(np.float32)

    lut_r = interp_r(grid).reshape(lut_size, lut_size, lut_size)
    lut_g = interp_g(grid).reshape(lut_size, lut_size, lut_size)
    lut_b = interp_b(grid).reshape(lut_size, lut_size, lut_size)

    return np.stack([lut_r, lut_g, lut_b], axis=-1).astype(np.float32)


def apply_lut(image, lut):
    """Apply a 3D LUT to a float32 RGB image [0,1]."""
    lut_size = lut.shape[0]
    img_flat = image.reshape(-1, 3)
    coords   = img_flat * (lut_size - 1)

    result = np.zeros_like(img_flat)
    for c in range(3):
        result[:, c] = map_coordinates(
            lut[:, :, :, c],
            [coords[:, 0], coords[:, 1], coords[:, 2]],
            order=1, mode="nearest",
        )
    return np.clip(result.reshape(image.shape), 0, 1)


def apply_lut_moderated(image, lut, strength=LUT_STRENGTH):
    """Blend original and LUT-corrected image at the given strength."""
    return image * (1 - strength) + apply_lut(image, lut) * strength


def _correct_image(img_np: np.ndarray, basename: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Apply CLAHE then 3D LUT. Returns (clahe_result, final_result, note)."""
    corrected = correct_clahe(img_np)
    lut = lut_cache.get(basename)
    if lut is not None:
        return corrected, apply_lut_moderated(corrected, lut), ""
    return corrected, corrected, f"LUT not yet cached for {basename} — CLAHE only"


def compute_aesthetic_score(img_np: np.ndarray) -> float:
    """
    Returns aesthetic quality score in [0, 1].
    img_np: float32 H×W×3, values 0–1.
    Weights: sharpness 0.4, exposure 0.3, contrast 0.2, noise 0.1.
    """
    img_u8 = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    gray   = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)

    # Sharpness — Laplacian variance, normalised to ~0-1
    lap       = cv2.Laplacian(gray, cv2.CV_32F)
    sharpness = min(float(np.var(lap)) / 1000.0, 1.0)

    # Exposure — mean L in LAB; penalise outside [0.3, 0.8]
    lab      = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
    l_mean   = lab[:, :, 0].mean() / 255.0
    exposure = 1.0 - max(0.0, 0.3 - l_mean) * 3.33 - max(0.0, l_mean - 0.8) * 5.0
    exposure = float(np.clip(exposure, 0.0, 1.0))

    # Contrast — std of L channel normalised
    contrast = min(float(lab[:, :, 0].std()) / 80.0, 1.0)

    # Noise — inverse of high-freq energy
    blur   = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    hf_rms = float(np.sqrt(np.mean((gray.astype(np.float32) - blur) ** 2)))
    noise  = max(0.0, 1.0 - hf_rms / 15.0)

    score = 0.4 * sharpness + 0.3 * exposure + 0.2 * contrast + 0.1 * noise
    return float(np.clip(score, 0.0, 1.0))


def np_to_base64(img_np):
    """Convert a float32 RGB numpy array [0,1] to a base64-encoded JPEG string."""
    img_uint8 = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    pil_img   = Image.fromarray(img_uint8)
    buf       = BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ============================================================
# PYDANTIC MODELS
# ============================================================

class SearchRequest(BaseModel):
    vector: list[float]
    top_k: int = 5

class SearchResult(BaseModel):
    basename: str
    distance: float
    raw_url: str
    edited_url: str

class SearchResponse(BaseModel):
    results: list[SearchResult]

class ProcessResponse(BaseModel):
    original_b64:          str
    corrected_b64:         str
    final_b64:             str
    defects:               dict
    retrieved:             str
    similarity:            float
    raw_b64:               str
    edited_b64:            str
    note:                  str   = ""   # empty when LUT applied; message when fallback used
    match_aesthetic_score: float = 0.0

class SearchAndCorrectRequest(BaseModel):
    vector: list[float]  # 517-dim hybrid vector from Android
    top_k:  int = 5

class SearchAndCorrectResponse(BaseModel):
    retrieved:             str
    similarity:            float
    raw_b64:               str   = ""   # empty when include_images=False
    edited_b64:            str   = ""   # empty when include_images=False
    lut_cached:            bool
    match_aesthetic_score: float = 0.0

class ApplyLutRequest(BaseModel):
    image_b64:          str
    retrieved_basename: str

class ApplyLutResponse(BaseModel):
    final_b64:  str
    lut_cached: bool
    note:       str = ""

class StyleSearchRequest(BaseModel):
    vector:     list[float]
    session_id: str

class StyleSearchResponse(BaseModel):
    retrieved:      str
    similarity:     float
    style_matched:  bool
    style_fallback: bool
    lut_cached:     bool

class LutResponse(BaseModel):
    lut_b64:  str   # flat float32 array (LUT_SIZE^3 * 3 values) as base64
    lut_size: int   # LUT grid side length (17)

class StyleVectorsRequest(BaseModel):
    vectors:    list[list[float]]  # pre-computed 517-dim hybrid vectors from Android
    session_id: str | None = None  # update existing session if provided

class StyleVectorsResponse(BaseModel):
    session_id:     str
    vectors_stored: int

class StyleUploadResponse(BaseModel):
    session_id:     str
    vectors_stored: int

class StyleProcessResponse(ProcessResponse):
    style_matched:  bool
    style_fallback: bool

class MailSendResponse(BaseModel):
    sent: bool
    to: str
    photos_attached: int

class DeliveryDetail(BaseModel):
    email:        str
    photos_count: int

class DeliveryRunResponse(BaseModel):
    matched:     int
    emails_sent: int
    failed:      int
    details:     list[DeliveryDetail]

class FaceClusterFace(BaseModel):
    photo_name: str
    face_image: str  # base64 JPEG 100x100

class FaceClusterGroup(BaseModel):
    cluster_id: int
    faces:      list[FaceClusterFace]

class FaceClusterResponse(BaseModel):
    clusters:    list[FaceClusterGroup]
    total_faces: int

class ClusterRequest(BaseModel):
    vectors:    list[list[float]]
    n_clusters: int | None = None

class ClusterInfo(BaseModel):
    cluster_id: int
    indices:    list[int]

class ClusterResponse(BaseModel):
    clusters:         list[ClusterInfo]
    n_clusters:       int
    silhouette_score: float

class BatchResult(BaseModel):
    index:                 int
    retrieved:             str
    similarity:            float
    corrected_b64:         str    # CLAHE + LUT applied to the user's own input image
    match_aesthetic_score: float = 0.0

class BatchResponse(BaseModel):
    results:   list[BatchResult]
    processed: int
    failed:    int

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/latency")
def latency_report():
    """Returns mean ± std (ms) for all instrumented server-side operations."""
    report = {}
    with _latency_lock:
        for key, vals in _latency.items():
            if len(vals) >= 2:
                report[key] = {
                    "n":    len(vals),
                    "mean": round(statistics.mean(vals), 3),
                    "std":  round(statistics.stdev(vals), 3),
                    "min":  round(min(vals), 3),
                    "max":  round(max(vals), 3),
                }
            elif len(vals) == 1:
                report[key] = {"n": 1, "mean": round(vals[0], 3), "std": 0.0}
            else:
                report[key] = {"n": 0, "mean": None, "std": None}
    return report


@app.get("/health")
def health():
    return {
        "status": "ok",
        "vectors_loaded": len(image_names),
        "vector_dim": hybrid_matrix.shape[1],
        "device": device,
        "lut_cache_size": len(lut_cache),
        "aesthetic_cache_size": len(aesthetic_cache),
    }


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """Search by a pre-computed hybrid vector (e.g. from the Android client)."""
    if not _search_ready:
        raise HTTPException(status_code=503, detail="Search models not loaded (missing defect_head.pt or hybrid_vectors.npz)")
    expected_dim = weighted_matrix_norm.shape[1]
    if len(req.vector) != expected_dim:
        raise HTTPException(
            status_code=400,
            detail=f"Vector dimension mismatch: got {len(req.vector)}, expected {expected_dim}",
        )

    query = np.array(req.vector, dtype="float32").reshape(1, -1)
    query = query / (np.linalg.norm(query) + 1e-8)

    D, I = index.search(query, req.top_k + 1)

    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        basename    = image_names[idx]
        raw_path    = os.path.join(RAW_DIR,    basename + ".jpg")
        edited_path = os.path.join(EDITED_DIR, basename + ".jpg")
        if not os.path.exists(raw_path) or not os.path.exists(edited_path):
            continue
        results.append(SearchResult(
            basename=basename,
            distance=float(dist),
            raw_url=f"/image/raw/{basename}",
            edited_url=f"/image/edited/{basename}",
        ))
        if len(results) >= req.top_k:
            break

    return SearchResponse(results=results)


@app.post("/search_and_correct", response_model=SearchAndCorrectResponse)
def search_and_correct(
    req: SearchAndCorrectRequest,
    aesthetic_weight: float = 0.3,
    include_images:   bool  = True,

):
    if not _search_ready:
        raise HTTPException(status_code=503, detail="Search models not loaded")
    """
    Fast path: Android pre-computes CLIP+defect vectors locally and sends the 517-dim
    hybrid vector. Server does only FAISS search + reference image retrieval.
    No CLIP/defect inference on the server side.
    Returns lut_cached=True once the background warmup has covered this match.
    """
    expected_dim = weighted_matrix_norm.shape[1]
    if len(req.vector) != expected_dim:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected_dim} dims, got {len(req.vector)}",
        )

    query = np.array(req.vector, dtype="float32").reshape(1, -1)
    query = query / (np.linalg.norm(query) + 1e-8)

    t0 = time.perf_counter()
    D, I  = index.search(query, max(req.top_k, 10) + 1)
    faiss_ms = (time.perf_counter() - t0) * 1000
    _record("faiss_search_ms", faiss_ms)

    candidates = [(image_names[i], float(D[0][j])) for j, i in enumerate(I[0]) if i >= 0]
    if not candidates:
        raise HTTPException(status_code=500, detail="FAISS returned no results")

    t1 = time.perf_counter()
    retrieved_basename, similarity, match_aes = _aesthetic_rerank(candidates, aesthetic_weight)
    rerank_ms = (time.perf_counter() - t1) * 1000
    _record("rerank_ms", rerank_ms)
    _record("search_total_ms", faiss_ms + rerank_ms)

    raw_b64    = ""
    edited_b64 = ""
    if include_images:
        raw_path    = os.path.join(RAW_DIR,    retrieved_basename + ".jpg")
        edited_path = os.path.join(EDITED_DIR, retrieved_basename + ".jpg")
        raw_retr    = _load_image(raw_path)
        edited_retr = _load_image(edited_path)
        raw_b64     = np_to_base64(raw_retr)
        edited_b64  = np_to_base64(edited_retr)

    return SearchAndCorrectResponse(
        retrieved             = retrieved_basename,
        similarity            = similarity,
        raw_b64               = raw_b64,
        edited_b64            = edited_b64,
        lut_cached            = retrieved_basename in lut_cache,
        match_aesthetic_score = match_aes,
    )


@app.post("/apply_lut", response_model=ApplyLutResponse)
def apply_lut_endpoint(req: ApplyLutRequest):
    """
    Apply CLAHE + cached 3D LUT to a base64-encoded image.
    Separated from FAISS retrieval so the client can reuse vectors
    across pipeline steps without re-running server-side CLIP inference.
    """
    img_bytes = base64.b64decode(req.image_b64)
    pil_img   = Image.open(BytesIO(img_bytes)).convert("RGB")
    img_np    = np.array(pil_img).astype(np.float32) / 255.0

    _, final, note = _correct_image(img_np, req.retrieved_basename)
    lut_cached     = note == ""

    return ApplyLutResponse(
        final_b64  = np_to_base64(final),
        lut_cached = lut_cached,
        note       = note,
    )


@app.post("/process", response_model=ProcessResponse)
async def process(file: UploadFile = File(...), aesthetic_weight: float = 0.3):
    if not _search_ready:
        raise HTTPException(status_code=503, detail="Search models not loaded")
    """
    Full pipeline:
    1. Extract CLIP (512) + defect (5) vectors
    2. Weighted FAISS retrieval
    3. CLAHE correction
    4. 3D LUT from best match RAW→edited pair
    5. Apply LUT at moderated strength (0.3)
    6. Return all images as base64 JPEG
    """
    contents = await file.read()
    pil_img  = Image.open(BytesIO(contents)).convert("RGB")
    img_np   = np.array(pil_img).astype(np.float32) / 255.0

    clip_vec, defect_vec = get_hybrid_vector(pil_img)
    defects_dict = {name: float(val) for name, val in zip(DEFECT_NAMES, defect_vec)}

    candidates = retrieve_similar(clip_vec, defect_vec, top_k=10)
    retrieved_basename, similarity, match_aes = _aesthetic_rerank(candidates, aesthetic_weight)

    raw_path    = os.path.join(RAW_DIR,    retrieved_basename + ".jpg")
    edited_path = os.path.join(EDITED_DIR, retrieved_basename + ".jpg")
    raw_retr    = _load_image(raw_path)
    edited_retr = _load_image(edited_path)

    corrected, final, note = _correct_image(img_np, retrieved_basename)

    return ProcessResponse(
        original_b64          = np_to_base64(img_np),
        corrected_b64         = np_to_base64(corrected),
        final_b64             = np_to_base64(final),
        defects               = defects_dict,
        retrieved             = retrieved_basename,
        similarity            = similarity,
        raw_b64               = np_to_base64(raw_retr),
        edited_b64            = np_to_base64(edited_retr),
        note                  = note,
        match_aesthetic_score = match_aes,
    )


@app.post("/batch/process", response_model=BatchResponse)
async def batch_process(files: list[UploadFile] = File(...)):
    if not _search_ready:
        raise HTTPException(status_code=503, detail="Search models not loaded")
    """
    Full correction pipeline applied to each uploaded image.
    Identical to /process but accepts up to 100 images in one request.
    Returns CLAHE + LUT corrected version of each INPUT image (not a reference image).
    """
    if len(files) > 100:
        raise HTTPException(400, "Maximum 100 images per batch")

    results = []
    failed  = 0

    for idx, f in enumerate(files):
        try:
            contents = await f.read()
            pil_img  = Image.open(BytesIO(contents)).convert("RGB")
            img_np   = np.array(pil_img).astype(np.float32) / 255.0

            clip_vec, defect_vec = get_hybrid_vector(pil_img)

            candidates         = retrieve_similar(clip_vec, defect_vec, top_k=10)
            retrieved_basename, similarity, match_aes = _aesthetic_rerank(candidates, 0.3)

            corrected = correct_clahe(img_np)
            lut       = lut_cache.get(retrieved_basename)
            final     = apply_lut_moderated(corrected, lut) if lut is not None else corrected

            results.append(BatchResult(
                index                 = idx,
                retrieved             = retrieved_basename,
                similarity            = similarity,
                corrected_b64         = np_to_base64(final),
                match_aesthetic_score = match_aes,
            ))
        except Exception as exc:
            print(f"  [/batch/process] item {idx} failed: {exc}")
            failed += 1

    return BatchResponse(results=results, processed=len(results), failed=failed)


@app.post("/cluster", response_model=ClusterResponse)
def cluster_photos(req: ClusterRequest):
    """K-Means clustering of 517-dim hybrid vectors. Max 500 vectors."""
    n = len(req.vectors)
    if n > 500:
        raise HTTPException(400, "Maximum 500 vectors per request")
    if n < 2:
        raise HTTPException(400, "Need at least 2 vectors to cluster")

    X = np.array(req.vectors, dtype="float32")
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    k = req.n_clusters if req.n_clusters is not None else _auto_k(X)
    k = max(2, min(k, n))

    km     = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X)

    n_labels = len(set(labels))
    sil = float(sk_silhouette(X, labels)) if 1 < n_labels < len(labels) else 0.0

    clusters = [
        ClusterInfo(
            cluster_id = cid,
            indices    = [int(i) for i, lbl in enumerate(labels) if lbl == cid],
        )
        for cid in range(k)
    ]
    return ClusterResponse(clusters=clusters, n_clusters=k, silhouette_score=sil)


@app.post("/style/upload", response_model=StyleUploadResponse)
async def style_upload(files: list[UploadFile] = File(...)):
    """
    Encode up to 20 reference photos into 517-dim style vectors and store them
    under a new session ID.
    """
    if len(files) > 20:
        raise HTTPException(400, "Maximum 20 reference images")
    vecs = []
    for f in files:
        data = await f.read()
        pil  = Image.open(BytesIO(data)).convert("RGB")
        clip_vec, defect_vec = get_hybrid_vector(pil)
        vecs.append(build_hybrid_query(clip_vec, defect_vec))
    session_id = str(uuid.uuid4())
    style_profiles[session_id] = np.stack(vecs).astype("float32")
    return StyleUploadResponse(session_id=session_id, vectors_stored=len(vecs))


@app.post("/style/process", response_model=StyleProcessResponse)
async def style_process(
    file:       UploadFile = File(...),
    session_id: str        = Form(...),

):
    if not _search_ready:
        raise HTTPException(status_code=503, detail="Search models not loaded")
    """
    Full pipeline with style-constrained retrieval:
    1. Build query vector from uploaded image
    2. Compute centroid + radius of the style profile vectors
    3. FAISS top-50, filter to style region; fall back to global if empty
    4. Semantic guard: if CLIP-only cosine sim < 0.55, fall back to global top-1
    5. CLAHE + LUT as usual
    """
    if session_id not in style_profiles:
        raise HTTPException(404, "Session not found — re-upload style images")

    contents = await file.read()
    pil_img  = Image.open(BytesIO(contents)).convert("RGB")
    img_np   = np.array(pil_img).astype(np.float32) / 255.0

    clip_vec, defect_vec = get_hybrid_vector(pil_img)
    defects_dict = {n: float(v) for n, v in zip(DEFECT_NAMES, defect_vec)}

    query      = build_hybrid_query(clip_vec, defect_vec)       # (517,)
    style_vecs = style_profiles[session_id]                     # (N, 517)

    D, I = index.search(query.reshape(1, -1), 51)
    best_local, similarity, style_matched, style_fallback = \
        _style_constrained_retrieve(query, style_vecs, D, I)

    retrieved_basename = image_names[best_local]

    raw_path    = os.path.join(RAW_DIR,    retrieved_basename + ".jpg")
    edited_path = os.path.join(EDITED_DIR, retrieved_basename + ".jpg")
    raw_retr    = _load_image(raw_path)
    edited_retr = _load_image(edited_path)

    corrected, final, note = _correct_image(img_np, retrieved_basename)

    return StyleProcessResponse(
        original_b64  = np_to_base64(img_np),
        corrected_b64 = np_to_base64(corrected),
        final_b64     = np_to_base64(final),
        defects       = defects_dict,
        retrieved     = retrieved_basename,
        similarity    = similarity,
        raw_b64       = np_to_base64(raw_retr),
        edited_b64    = np_to_base64(edited_retr),
        note          = note,
        style_matched  = style_matched,
        style_fallback = style_fallback,
    )


@app.post("/style/search", response_model=StyleSearchResponse)
def style_search(req: StyleSearchRequest):
    if not _search_ready:
        raise HTTPException(status_code=503, detail="Search models not loaded")
    """
    Style-constrained FAISS retrieval using a pre-computed 517-dim hybrid vector.
    Returns only retrieval metadata — client calls /apply_lut for the actual correction.
    """
    if req.session_id not in style_profiles:
        raise HTTPException(404, "Session not found — re-upload style images")

    expected_dim = weighted_matrix_norm.shape[1]
    if len(req.vector) != expected_dim:
        raise HTTPException(400, f"Expected {expected_dim} dims, got {len(req.vector)}")

    query      = np.array(req.vector, dtype="float32").reshape(1, -1)
    query      = query / (np.linalg.norm(query) + 1e-8)
    style_vecs = style_profiles[req.session_id]

    D, I = index.search(query, 51)
    best_local, similarity, style_matched, style_fallback = \
        _style_constrained_retrieve(query[0], style_vecs, D, I)

    retrieved_basename = image_names[best_local]

    return StyleSearchResponse(
        retrieved      = retrieved_basename,
        similarity     = similarity,
        style_matched  = style_matched,
        style_fallback = style_fallback,
        lut_cached     = retrieved_basename in lut_cache,
    )


@app.get("/image/raw/{basename}")
def get_raw_image(basename: str):
    basename = Path(basename).name
    path = os.path.join(RAW_DIR, basename + ".jpg")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="RAW image not found")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/image/edited/{basename}")
def get_edited_image(basename: str):
    basename = Path(basename).name
    path = os.path.join(EDITED_DIR, basename + ".jpg")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Edited image not found")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/lut/{basename}", response_model=LutResponse)
def get_lut(basename: str):
    """
    Return the 3D LUT for a given reference image basename as a flat base64-encoded
    float32 byte array.  Shape: (LUT_SIZE, LUT_SIZE, LUT_SIZE, 3) → flattened.
    Android client downloads once, caches on-device, and applies trilinear
    interpolation locally — no image ever reaches the server.
    If the LUT is not yet pre-computed, it is built on-demand and added to lut_cache.
    """
    basename = Path(basename).name
    lut = lut_cache.get(basename)
    if lut is None:
        raw_path    = os.path.join(RAW_DIR,    basename + ".jpg")
        edited_path = os.path.join(EDITED_DIR, basename + ".jpg")
        if not os.path.exists(raw_path) or not os.path.exists(edited_path):
            raise HTTPException(status_code=404, detail=f"Images not found for '{basename}'")
        t_compute = time.perf_counter()
        raw    = _load_image(raw_path)
        edited = _load_image(edited_path)
        lut    = extract_colour_lut(raw, edited)
        lut_cache[basename] = lut
        _record("lut_compute_ms", (time.perf_counter() - t_compute) * 1000)

    t_ser = time.perf_counter()
    lut_bytes = lut.flatten().astype(np.float32).tobytes()
    lut_b64   = base64.b64encode(lut_bytes).decode("utf-8")
    _record("lut_serialize_ms", (time.perf_counter() - t_ser) * 1000)

    return LutResponse(lut_b64=lut_b64, lut_size=int(lut.shape[0]))


@app.post("/style/vectors", response_model=StyleVectorsResponse)
def style_vectors(req: StyleVectorsRequest):
    """
    Store pre-computed 517-dim hybrid vectors as a style profile.
    Android computes CLIP + defect vectors locally and sends only the numbers —
    no image bytes ever reach the server.
    Mirrors /style/upload but without server-side CLIP inference.
    """
    expected_dim = weighted_matrix_norm.shape[1]
    for i, vec in enumerate(req.vectors):
        if len(vec) != expected_dim:
            raise HTTPException(
                status_code=400,
                detail=f"Vector {i}: expected {expected_dim} dims, got {len(vec)}",
            )
    vecs       = np.array(req.vectors, dtype="float32")
    session_id = req.session_id or str(uuid.uuid4())
    style_profiles[session_id] = vecs
    return StyleVectorsResponse(session_id=session_id, vectors_stored=len(vecs))


# ============================================================
# BLUR SENSITIVE
# ============================================================

class BlurRegionItem(BaseModel):
    label: str
    x: int
    y: int
    w: int
    h: int

class BlurDetectResponse(BaseModel):
    regions:     list[BlurRegionItem]
    image_width:  int
    image_height: int


@app.post("/blur-detect", response_model=BlurDetectResponse)
async def blur_detect(
    file:     UploadFile = File(...),
    detector: str        = "gemini",
    current_user = Depends(get_current_active_user),
):
    """Detect sensitive regions and return bounding boxes — does NOT apply blur."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    if detector not in ("gemini", "local"):
        raise HTTPException(400, "detector must be 'gemini' or 'local'")
    image_bytes = await file.read()
    regions = _detect_local(image_bytes) if detector == "local" else _detect_gemini(image_bytes)
    from PIL import Image as _PILImage
    from io import BytesIO as _BytesIO
    img = _PILImage.open(_BytesIO(image_bytes))
    return BlurDetectResponse(
        regions=[BlurRegionItem(**r) for r in regions],
        image_width=img.width,
        image_height=img.height,
    )


@app.post("/blur-apply")
async def blur_apply(
    file:    UploadFile = File(...),
    regions: str        = Form(...),
    current_user = Depends(get_current_active_user),
):
    """Apply Gaussian blur to the given regions (JSON list of {x,y,w,h}). No re-detection."""
    import json as _json
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    image_bytes  = await file.read()
    region_dicts = _json.loads(regions)
    result = apply_blur(image_bytes, region_dicts)
    return Response(content=result, media_type="image/jpeg")


@app.post("/blur-sensitive")
async def blur_sensitive(
    file:     UploadFile = File(...),
    detector: str        = "gemini",
    current_user = Depends(get_current_active_user),
):
    """
    Detect sensitive regions and blur them.
    ?detector=gemini  — Gemini 1.5 Flash vision API (default)
    ?detector=local   — YOLOv8 nano on-server, no external API call
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    if detector not in ("gemini", "local"):
        raise HTTPException(status_code=400, detail="detector must be 'gemini' or 'local'")
    image_bytes = await file.read()
    regions = _detect_local(image_bytes) if detector == "local" else _detect_gemini(image_bytes)
    result = apply_blur(image_bytes, regions)
    return Response(content=result, media_type="image/jpeg")


# ============================================================
# MAIL
# ============================================================

@app.post("/mail/send", response_model=MailSendResponse)
async def mail_send(
    to_email: str = Form(...),
    subject:  str = Form(default="Fotografiile tale"),
    message:  str = Form(default=""),
    files:    list[UploadFile] = File(default=[]),
    current_user = Depends(get_current_active_user),
):
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASSWORD")
    if not smtp_user or not smtp_pass:
        raise HTTPException(500, "SMTP_USER and SMTP_PASSWORD environment variables must be set")

    msg = MIMEMultipart()
    msg["From"]    = smtp_user
    msg["To"]      = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(message or f"Fotografii trimise de {current_user.email}", "plain"))

    for f in files:
        data = await f.read()
        img_part = MIMEImage(data, name=f.filename or "photo.jpg")
        img_part.add_header("Content-Disposition", "attachment", filename=f.filename or "photo.jpg")
        msg.attach(img_part)

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.ehlo()
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, to_email, msg.as_string())

    return MailSendResponse(sent=True, to=to_email, photos_attached=len(files))


# ============================================================
# MOCK EMPLOYEES PAGE
# ============================================================

_MOCK_EMPLOYEES_HTML = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Company Team</title></head>
<body>
  <h1>Our Team</h1>

  <div class="employee-card">
    <img src="https://randomuser.me/api/portraits/women/1.jpg">
    <h3>Alice</h3>
    <p>alexandradumitrescu04@gmail.com</p>
  </div>

  <div class="employee-card">
    <img src="https://randomuser.me/api/portraits/men/2.jpg">
    <h3>Bob</h3>
    <p>alexandradumitrescu04@gmail.com</p>
  </div>

  <div class="employee-card">
    <img src="https://randomuser.me/api/portraits/women/3.jpg">
    <h3>Carol</h3>
    <p>alexandradumitrescu04@gmail.com</p>
  </div>
</body>
</html>"""


@app.get("/mock-employees", response_class=HTMLResponse)
def mock_employees():
    """Public endpoint — serves the mock company team page for the delivery demo."""
    return _MOCK_EMPLOYEES_HTML


# ============================================================
# FACE CLUSTERING (diagnostic)
# ============================================================

@app.post("/cluster/faces", response_model=FaceClusterResponse)
async def face_cluster(
    photos: list[UploadFile] = File(...),
    current_user = Depends(get_current_active_user),
):
    """Detect and cluster all faces from uploaded photos. Returns cropped face thumbnails grouped by identity."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        for photo in photos:
            data  = await photo.read()
            fname = photo.filename or f"{uuid.uuid4()}.jpg"
            with open(os.path.join(tmp_dir, fname), "wb") as fh:
                fh.write(data)

        clusters = _cluster_faces(tmp_dir)

    total = sum(len(c["faces"]) for c in clusters)
    print(f"[cluster/faces] {total} faces in {len(clusters)} clusters")
    return FaceClusterResponse(
        clusters=[FaceClusterGroup(
            cluster_id=c["cluster_id"],
            faces=[FaceClusterFace(**f) for f in c["faces"]],
        ) for c in clusters],
        total_faces=total,
    )


# ============================================================
# DELIVERY v2 — embedding-based (memory-efficient)
# ============================================================

class PhotoEmbeddingsItem(BaseModel):
    photo_index: int
    embeddings:  list[list[float]]

class MatchEmbeddingsRequest(BaseModel):
    employees_url: str
    photos:        list[PhotoEmbeddingsItem]

class PhotoMatchItem(BaseModel):
    photo_index: int
    email:       str
    name:        str = ""

class MatchEmbeddingsResponse(BaseModel):
    matches:         list[PhotoMatchItem]
    employees_count: int

class SendMatchedDetail(BaseModel):
    email:        str
    photos_count: int

class SendMatchedResponse(BaseModel):
    emails_sent: int
    failed:      int
    details:     list[SendMatchedDetail]


_MATCH_THRESHOLD = 0.55


@app.post("/delivery/match-embeddings", response_model=MatchEmbeddingsResponse)
async def delivery_match_embeddings(
    request: MatchEmbeddingsRequest,
    current_user = Depends(get_current_active_user),
):
    """Phase 1: receive on-device embeddings, scrape employee DB, return matches."""
    url = request.employees_url
    if url not in _employees_cache:
        try:
            employees = _scrape(url)
        except Exception as e:
            raise HTTPException(500, f"Scrape failed: {e}")
        if not employees:
            raise HTTPException(400, f"No employees found at {url}")
        _employees_cache[url] = (employees, {e["email"]: e.get("name", "") for e in employees})
    employees, name_by_email = _employees_cache[url]

    if url not in _face_db_tflite:
        db = _build_db(employees)
        if not db:
            raise HTTPException(500, "Face DB build failed")
        _face_db_tflite[url] = db
    db = _face_db_tflite[url]
    print(f"[match-emb] db={len(db)} employees, photos={len(request.photos)}")
    print(f"[match-emb] db={len(db)} employees, photos={len(request.photos)}")

    matches: list[PhotoMatchItem] = []
    for photo_data in request.photos:
        matched_in_photo: set[str] = set()
        for raw_emb in photo_data.embeddings:
            emb = np.array(raw_emb, dtype=np.float32)
            norm = float(np.linalg.norm(emb))
            if norm > 0:
                emb = emb / norm
            best_email, best_sim = "", -1.0
            for email, db_emb in db.items():
                sim = float(np.dot(emb, db_emb))
                if sim > best_sim:
                    best_sim, best_email = sim, email
            print(f"  photo={photo_data.photo_index} best={best_email} sim={best_sim:.3f}")
            if best_sim >= _MATCH_THRESHOLD and best_email not in matched_in_photo:
                matches.append(PhotoMatchItem(
                    photo_index=photo_data.photo_index,
                    email=best_email,
                    name=name_by_email.get(best_email, ""),
                ))
                matched_in_photo.add(best_email)

    print(f"[match-emb] {len(matches)} matches found")
    return MatchEmbeddingsResponse(matches=matches, employees_count=len(db))


@app.post("/delivery/send-matched", response_model=SendMatchedResponse)
async def delivery_send_matched(
    to_email: str              = Form(...),
    to_name:  str              = Form(default=""),
    photos:   list[UploadFile] = File(...),
    current_user = Depends(get_current_active_user),
):
    """Phase 2 (per-person): receive photos for one person, send one email."""
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASSWORD")
    if not smtp_user or not smtp_pass:
        raise HTTPException(500, "SMTP_USER and SMTP_PASSWORD environment variables must be set")

    demo_recipient = "alexandradumitrescu04@gmail.com"
    subject = f"{to_name} — {to_email}" if to_name else to_email

    try:
        msg = MIMEMultipart()
        msg["From"]    = smtp_user
        msg["To"]      = demo_recipient
        msg["Subject"] = subject
        msg.attach(MIMEText(
            f"Hi,\n\nPhotos identified for {to_name or to_email}.\n\nBest regards", "plain"))
        for photo in photos:
            data = await photo.read()
            img_part = MIMEImage(data)
            img_part.add_header("Content-Disposition", "attachment",
                                filename=photo.filename or "photo.jpg")
            msg.attach(img_part)

        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
            server.ehlo(); server.starttls(); server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, demo_recipient, msg.as_string())

        print(f"[send-matched] sent for {to_email} ({len(photos)} photos)")
        return SendMatchedResponse(
            emails_sent=1, failed=0,
            details=[SendMatchedDetail(email=to_email, photos_count=len(photos))],
        )
    except Exception as e:
        print(f"[send-matched] failed for {to_email}: {e}")
        return SendMatchedResponse(emails_sent=0, failed=1, details=[])


# ============================================================
# DELIVERY (legacy — full server-side pipeline)
# ============================================================

@app.post("/delivery/run", response_model=DeliveryRunResponse)
async def delivery_run(
    employees_url: str              = Form(...),
    photos:        list[UploadFile] = File(...),
    current_user = Depends(get_current_active_user),
):
    """
    Full delivery pipeline:
    1. Scrape employee photos + emails from employees_url
    2. Build FaceNet face DB in-memory
    3. Match uploaded event photos against face DB (cosine ≥ 0.60)
    4. Send matched photos to each recognised person via email
    Returns matched count, emails sent, per-person details.
    Requires: SMTP_USER, SMTP_PASSWORD env vars.
    """
    import traceback as _tb
    _bg_pause.set()
    print("[delivery/run] background precomputation paused")
    try:
        return await _delivery_run_impl(employees_url, photos, current_user)
    finally:
        _bg_pause.clear()
        print("[delivery/run] background precomputation resumed")


async def _delivery_run_impl(employees_url, photos, current_user):
    import traceback as _tb
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASSWORD")
    print(f"[delivery/run] SMTP_USER={smtp_user!r}")
    if not smtp_user or not smtp_pass:
        raise HTTPException(500, "SMTP_USER and SMTP_PASSWORD environment variables must be set")

    if employees_url not in _employees_cache:
        try:
            employees = _scrape(employees_url)
        except Exception as e:
            print(f"[delivery/run] _scrape failed: {e}\n{_tb.format_exc()}")
            raise HTTPException(500, f"Scrape failed: {e}")
        print(f"[delivery/run] scraped {len(employees)} employees")
        if not employees:
            raise HTTPException(400, f"No employees found at {employees_url}")
        _employees_cache[employees_url] = (employees, {e["email"]: e.get("name", "") for e in employees})
    employees, _ = _employees_cache[employees_url]

    if employees_url not in _face_db_facenet:
        try:
            db = _build_db_facenet(employees)
        except Exception as e:
            print(f"[delivery/run] _build_db_facenet failed: {e}\n{_tb.format_exc()}")
            raise HTTPException(500, f"Build DB failed: {e}")
        _face_db_facenet[employees_url] = db
        print(f"[delivery/run] face DB built and cached ({len(db)} entries)")
    else:
        print(f"[delivery/run] face DB from cache ({len(_face_db_facenet[employees_url])} entries)")
    db = _face_db_facenet[employees_url]
    print(f"[delivery/run] db built, {len(db)} entries")
    if not db:
        raise HTTPException(500, "Face DB build failed — no embeddings extracted")

    matched     = 0
    emails_sent = 0
    failed      = 0
    details     = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for photo in photos:
            data  = await photo.read()
            fname = f"{uuid.uuid4()}.jpg"
            pil   = Image.open(BytesIO(data)).convert("RGB")
            w, h  = pil.size
            scale = min(1920 / max(w, h), 1.0)
            if scale < 1.0:
                pil = pil.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
            pil.save(os.path.join(tmp_dir, fname), "JPEG", quality=92)

        matches = _match_photos(tmp_dir, db)
        matched = sum(len(paths) for paths in matches.values())

        _, name_by_email = _employees_cache[employees_url]
        demo_recipient  = "alexandradumitrescu04@gmail.com"

        def _send_one(item):
            to_email, photo_paths = item
            person_name = name_by_email.get(to_email, "")
            subject     = f"{person_name} — {to_email}" if person_name else to_email
            msg = MIMEMultipart()
            msg["From"]    = smtp_user
            msg["To"]      = demo_recipient
            msg["Subject"] = subject
            msg.attach(MIMEText(
                f"Hi,\n\nWe found {len(photo_paths)} photo(s) of you from the event. "
                "See the attachments!\n\nBest regards",
                "plain",
            ))
            for path in photo_paths:
                with open(path, "rb") as fh:
                    img_part = MIMEImage(fh.read())
                img_part.add_header("Content-Disposition", "attachment",
                                    filename=os.path.basename(path))
                msg.attach(img_part)
            with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as conn:
                conn.ehlo(); conn.starttls(); conn.login(smtp_user, smtp_pass)
                conn.sendmail(smtp_user, demo_recipient, msg.as_string())
            return to_email, len(photo_paths)

        from concurrent.futures import ThreadPoolExecutor as _TPE
        with _TPE(max_workers=5) as pool:
            futures = {pool.submit(_send_one, item): item for item in matches.items()}
            for future in futures:
                try:
                    to_email, count = future.result()
                    emails_sent += 1
                    details.append(DeliveryDetail(email=to_email, photos_count=count))
                except Exception as exc:
                    print(f"  [/delivery/run] send failed: {exc}")
                    failed += 1

    return DeliveryRunResponse(
        matched=matched, emails_sent=emails_sent, failed=failed, details=details
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
