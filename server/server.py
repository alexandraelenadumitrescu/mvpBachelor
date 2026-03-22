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
import base64
import threading
from io import BytesIO
from pathlib import Path

import cv2
import faiss
import numpy as np
import open_clip
import torch
import torch.nn as nn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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
VECTORS_PATH = os.path.join(BASE_DIR, "hybrid_vectors.npz")
MODEL_PATH   = os.path.join(BASE_DIR, "defect_head.pt")

CLIP_WEIGHT   = 1.0
DEFECT_WEIGHT = 5.0
LUT_STRENGTH  = 0.2
LUT_SIZE      = 17

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ============================================================
# STARTUP CHECKS
# ============================================================

if not os.path.exists(VECTORS_PATH):
    print(f"ERROR: hybrid_vectors.npz not found at {VECTORS_PATH}", file=sys.stderr)
    sys.exit(1)

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: defect_head.pt not found at {MODEL_PATH}", file=sys.stderr)
    print("Place defect_head.pt in the server/ folder and restart.", file=sys.stderr)
    sys.exit(1)

# ============================================================
# LOAD MODELS
# ============================================================

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

print("Loading hybrid vectors...")
data          = np.load(VECTORS_PATH, allow_pickle=True)
hybrid_matrix = data["vectors"].astype("float32")
image_names   = data["names"].tolist()
print(f"Loaded {len(image_names)} vectors of dimension {hybrid_matrix.shape[1]}")

# Apply same weighting used at index-build time
clip_part   = hybrid_matrix[:, :512]
defect_part = hybrid_matrix[:, 512:]

clip_norm   = clip_part   / (np.linalg.norm(clip_part,   axis=1, keepdims=True) + 1e-8)
defect_norm = defect_part / (np.linalg.norm(defect_part, axis=1, keepdims=True) + 1e-8)

weighted_matrix = np.concatenate(
    [clip_norm * CLIP_WEIGHT, defect_norm * DEFECT_WEIGHT], axis=1
).astype("float32")

final_norms = np.linalg.norm(weighted_matrix, axis=1, keepdims=True)
weighted_matrix_norm = weighted_matrix / (final_norms + 1e-8)

# Retain weighted_matrix (pre-normalized) for CLIP-only semantic guard in style retrieval
# weighted_matrix[:, :512] == L2-normalized clip vectors (CLIP_WEIGHT=1.0)
name_to_idx: dict = {name: i for i, name in enumerate(image_names)}

index = faiss.IndexFlatIP(weighted_matrix_norm.shape[1])
index.add(weighted_matrix_norm)
print(f"✅ FAISS index ready with {index.ntotal} vectors (dim={weighted_matrix_norm.shape[1]})")

# ============================================================
# LUT CACHE (pre-computed in background)
# ============================================================

lut_cache:     dict = {}  # basename -> np.ndarray shape (LUT_SIZE, LUT_SIZE, LUT_SIZE, 3)
style_profiles: dict = {}  # session_id -> np.ndarray shape (N, 517) weighted-normalized


def _precompute_luts():
    """Daemon thread: pre-compute and cache 3D LUTs for all RAW→edited pairs."""
    print(f"LUT pre-computation starting for {len(image_names)} pairs...")
    computed = 0
    for name in image_names:
        if name in lut_cache:  # skip if already cached on-demand
            continue
        raw_path    = os.path.join(RAW_DIR,    name + ".jpg")
        edited_path = os.path.join(EDITED_DIR, name + ".jpg")
        if not os.path.exists(raw_path) or not os.path.exists(edited_path):
            continue
        try:
            raw    = np.array(Image.open(raw_path)).astype(np.float32)    / 255.0
            edited = np.array(Image.open(edited_path)).astype(np.float32) / 255.0
            lut_cache[name] = extract_colour_lut(raw, edited)
            computed += 1
            if computed % 100 == 0:
                print(f"  LUT cache: {computed}/{len(image_names)}")
        except Exception as exc:
            print(f"  LUT warning [{name}]: {exc}")
    print(f"✅ LUT pre-computation done: {computed} LUTs cached")


threading.Thread(target=_precompute_luts, daemon=True).start()
print("LUT pre-computation started in background (server ready)")


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
    original_b64:  str
    corrected_b64: str
    final_b64:     str
    defects:       dict
    retrieved:     str
    similarity:    float
    raw_b64:       str
    edited_b64:    str
    note:          str = ""  # empty when LUT applied; message when fallback used

class SearchAndCorrectRequest(BaseModel):
    vector: list[float]  # 517-dim hybrid vector from Android
    top_k:  int = 5

class SearchAndCorrectResponse(BaseModel):
    retrieved:  str
    similarity: float
    raw_b64:    str   # retrieved RAW reference image
    edited_b64: str   # retrieved edited reference image
    lut_cached: bool  # whether a LUT is available for this match

class StyleUploadResponse(BaseModel):
    session_id:     str
    vectors_stored: int

class StyleProcessResponse(ProcessResponse):
    style_matched:  bool
    style_fallback: bool

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
    index:         int
    retrieved:     str
    similarity:    float
    corrected_b64: str   # CLAHE + LUT applied to the user's own input image

class BatchResponse(BaseModel):
    results:   list[BatchResult]
    processed: int
    failed:    int

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "vectors_loaded": len(image_names),
        "vector_dim": hybrid_matrix.shape[1],
        "device": device,
        "lut_cache_size": len(lut_cache),
    }


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """Search by a pre-computed hybrid vector (e.g. from the Android client)."""
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
def search_and_correct(req: SearchAndCorrectRequest):
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
    D, I  = index.search(query, req.top_k + 1)

    results = [(image_names[i], float(D[0][j])) for j, i in enumerate(I[0]) if i >= 0]
    if not results:
        raise HTTPException(status_code=500, detail="FAISS returned no results")

    retrieved_basename, similarity = results[0]

    raw_path    = os.path.join(RAW_DIR,    retrieved_basename + ".jpg")
    edited_path = os.path.join(EDITED_DIR, retrieved_basename + ".jpg")
    raw_retr    = np.array(Image.open(raw_path)).astype(np.float32)    / 255.0
    edited_retr = np.array(Image.open(edited_path)).astype(np.float32) / 255.0

    return SearchAndCorrectResponse(
        retrieved  = retrieved_basename,
        similarity = similarity,
        raw_b64    = np_to_base64(raw_retr),
        edited_b64 = np_to_base64(edited_retr),
        lut_cached = retrieved_basename in lut_cache,
    )


@app.post("/process", response_model=ProcessResponse)
async def process(file: UploadFile = File(...)):
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

    # Vectors
    clip_vec, defect_vec = get_hybrid_vector(pil_img)
    defect_names = ["blur", "noise", "overexposure", "underexposure", "compression"]
    defects_dict = {name: float(val) for name, val in zip(defect_names, defect_vec)}

    # Retrieval
    results            = retrieve_similar(clip_vec, defect_vec, top_k=5)
    retrieved_basename = results[0][0]
    similarity         = results[0][1]

    # CLAHE
    corrected = correct_clahe(img_np)

    # Load reference images
    raw_path    = os.path.join(RAW_DIR,    retrieved_basename + ".jpg")
    edited_path = os.path.join(EDITED_DIR, retrieved_basename + ".jpg")
    raw_retr    = np.array(Image.open(raw_path)).astype(np.float32)    / 255.0
    edited_retr = np.array(Image.open(edited_path)).astype(np.float32) / 255.0

    # LUT — use cache; fall back to CLAHE-only if not yet pre-computed
    lut  = lut_cache.get(retrieved_basename)
    note = ""
    if lut is not None:
        final = apply_lut_moderated(corrected, lut)
    else:
        final = corrected  # CLAHE applied; LUT not yet available
        note  = f"LUT not yet cached for {retrieved_basename} — CLAHE correction applied only"
        print(f"  [/process] LUT cache miss: {retrieved_basename}")

    return ProcessResponse(
        original_b64  = np_to_base64(img_np),
        corrected_b64 = np_to_base64(corrected),
        final_b64     = np_to_base64(final),
        defects       = defects_dict,
        retrieved     = retrieved_basename,
        similarity    = similarity,
        raw_b64       = np_to_base64(raw_retr),
        edited_b64    = np_to_base64(edited_retr),
        note          = note,
    )


@app.post("/batch/process", response_model=BatchResponse)
async def batch_process(files: list[UploadFile] = File(...)):
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

            results_faiss      = retrieve_similar(clip_vec, defect_vec, top_k=1)
            retrieved_basename = results_faiss[0][0]
            similarity         = results_faiss[0][1]

            corrected = correct_clahe(img_np)
            lut       = lut_cache.get(retrieved_basename)
            final     = apply_lut_moderated(corrected, lut) if lut is not None else corrected

            results.append(BatchResult(
                index         = idx,
                retrieved     = retrieved_basename,
                similarity    = similarity,
                corrected_b64 = np_to_base64(final),
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

    sil = float(sk_silhouette(X, labels)) if len(set(labels)) > 1 else 0.0

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
    defect_names = ["blur", "noise", "overexposure", "underexposure", "compression"]
    defects_dict = {n: float(v) for n, v in zip(defect_names, defect_vec)}

    query      = build_hybrid_query(clip_vec, defect_vec)       # (517,)
    style_vecs = style_profiles[session_id]                     # (N, 517)

    D, I = index.search(query.reshape(1, -1), 51)
    best_local, similarity, style_matched, style_fallback = \
        _style_constrained_retrieve(query, style_vecs, D, I)

    retrieved_basename = image_names[best_local]

    # CLAHE + LUT (identical to /process)
    corrected   = correct_clahe(img_np)
    raw_path    = os.path.join(RAW_DIR,    retrieved_basename + ".jpg")
    edited_path = os.path.join(EDITED_DIR, retrieved_basename + ".jpg")
    raw_retr    = np.array(Image.open(raw_path)).astype(np.float32)    / 255.0
    edited_retr = np.array(Image.open(edited_path)).astype(np.float32) / 255.0

    lut  = lut_cache.get(retrieved_basename)
    note = ""
    if lut is not None:
        final = apply_lut_moderated(corrected, lut)
    else:
        final = corrected
        note  = f"LUT not yet cached for {retrieved_basename} — CLAHE only"

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


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
