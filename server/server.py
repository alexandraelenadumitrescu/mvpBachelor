"""
Server FastAPI complet — retrieval hibrid + CLAHE + LUT
Rulează local: uvicorn server_complete:app --host 0.0.0.0 --port 8000

Structura așteptată:
server_data/
  server_complete.py
  hybrid_vectors.npz
  defect_head.pt
  images/
    raw/
    edited/

Instalare:
pip install fastapi uvicorn numpy faiss-cpu torch torchvision
    open-clip-torch scipy opencv-python pillow python-multipart
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
import numpy as np
import faiss
import os
import cv2
import torch
import torch.nn as nn
import open_clip
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import map_coordinates
from io import BytesIO
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
LUT_STRENGTH  = 0.3
LUT_SIZE      = 17

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ============================================================
# ÎNCĂRCARE MODELE LA STARTUP
# ============================================================

print("Încarc CLIP...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
clip_model = clip_model.to(device)
clip_model.eval()
print("✅ CLIP încărcat")

print("Încarc MobileNet head...")
backbone = models.mobilenet_v3_small(weights=None)
in_features = backbone.classifier[0].in_features
backbone.classifier = nn.Sequential(
    nn.Linear(in_features, 128),
    nn.Hardswish(),
    nn.Dropout(0.2),
    nn.Linear(128, 5),
    nn.Sigmoid()
)
backbone.load_state_dict(torch.load(MODEL_PATH, map_location=device))
backbone = backbone.to(device)
backbone.eval()
print("✅ MobileNet head încărcat")

defect_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

print("Încarc vectorii hibrizi...")
data          = np.load(VECTORS_PATH, allow_pickle=True)
hybrid_matrix = data["vectors"].astype("float32")
image_names   = data["names"].tolist()
print(f"Încarcat {len(image_names)} vectori de dimensiune {hybrid_matrix.shape[1]}")

# Index cu ponderi
clip_part   = hybrid_matrix[:, :512]
defect_part = hybrid_matrix[:, 512:]

clip_norm   = clip_part   / (np.linalg.norm(clip_part,   axis=1, keepdims=True) + 1e-8)
defect_norm = defect_part / (np.linalg.norm(defect_part, axis=1, keepdims=True) + 1e-8)

weighted_matrix = np.concatenate([
    clip_norm   * CLIP_WEIGHT,
    defect_norm * DEFECT_WEIGHT
], axis=1).astype("float32")

final_norms = np.linalg.norm(weighted_matrix, axis=1, keepdims=True)
weighted_matrix_norm = weighted_matrix / (final_norms + 1e-8)

index = faiss.IndexFlatIP(weighted_matrix.shape[1])
index.add(weighted_matrix_norm)
print(f"✅ Index FAISS gata cu {index.ntotal} vectori")

# ============================================================
# FUNCȚII INTERNE
# ============================================================

def get_hybrid_vector(pil_img):
    """Extrage vectorul hibrid CLIP + defecte dintr-o imagine PIL"""
    # CLIP
    img_tensor = clip_preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        clip_vec = clip_model.encode_image(img_tensor)
        clip_vec = clip_vec / clip_vec.norm(dim=-1, keepdim=True)
    clip_vec = clip_vec.cpu().numpy()[0]

    # Defecte
    img_tensor2 = defect_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        defect_vec = backbone(img_tensor2).cpu().numpy()[0]

    return clip_vec, defect_vec


def retrieve_similar(clip_vec, defect_vec, top_k=5):
    """Caută imagini similare în index cu ponderi"""
    clip_v   = clip_vec   / (np.linalg.norm(clip_vec)   + 1e-8) * CLIP_WEIGHT
    defect_v = defect_vec / (np.linalg.norm(defect_vec) + 1e-8) * DEFECT_WEIGHT

    query = np.concatenate([clip_v, defect_v]).astype("float32")
    query = query / (np.linalg.norm(query) + 1e-8)

    D, I = index.search(query.reshape(1, -1), top_k + 1)
    return [(image_names[i], float(D[0][j])) for j, i in enumerate(I[0]) if i >= 0]


def correct_clahe(img_np):
    """Corecție CLAHE pentru underexposure"""
    img_uint8 = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return result.astype(np.float32) / 255.0


def extract_colour_lut(raw, edited, lut_size=LUT_SIZE):
    """Extrage un 3D LUT din perechea RAW → edited"""
    h, w = raw.shape[:2]

    # Redimensionăm edited la același size ca raw
    edited_resized = np.array(
        Image.fromarray((edited * 255).astype(np.uint8)).resize(
            (w, h), Image.LANCZOS
        )
    ).astype(np.float32) / 255.0

    n_samples = min(50000, h * w)
    idx = np.random.choice(h * w, n_samples, replace=False)
    raw_flat    = raw.reshape(-1, 3)[idx]
    edited_flat = edited_resized.reshape(-1, 3)[idx]

    interp_r = LinearNDInterpolator(raw_flat, edited_flat[:, 0], fill_value=0)
    interp_g = LinearNDInterpolator(raw_flat, edited_flat[:, 1], fill_value=0)
    interp_b = LinearNDInterpolator(raw_flat, edited_flat[:, 2], fill_value=0)

    steps = np.linspace(0, 1, lut_size)
    r, g, b = np.meshgrid(steps, steps, steps, indexing='ij')
    grid = np.stack([r.ravel(), g.ravel(), b.ravel()], axis=1).astype(np.float32)

    lut_r = interp_r(grid).reshape(lut_size, lut_size, lut_size)
    lut_g = interp_g(grid).reshape(lut_size, lut_size, lut_size)
    lut_b = interp_b(grid).reshape(lut_size, lut_size, lut_size)

    return np.stack([lut_r, lut_g, lut_b], axis=-1).astype(np.float32)


def apply_lut(image, lut):
    """Aplică un 3D LUT pe o imagine"""
    lut_size = lut.shape[0]
    img_flat = image.reshape(-1, 3)
    coords   = img_flat * (lut_size - 1)

    result = np.zeros_like(img_flat)
    for c in range(3):
        result[:, c] = map_coordinates(
            lut[:, :, :, c],
            [coords[:, 0], coords[:, 1], coords[:, 2]],
            order=1, mode='nearest'
        )
    return np.clip(result.reshape(image.shape), 0, 1)


def apply_lut_moderated(image, lut, strength=LUT_STRENGTH):
    """Blend între original și LUT aplicat"""
    return image * (1 - strength) + apply_lut(image, lut) * strength


def np_to_base64(img_np):
    """Convertește numpy array la base64 JPEG"""
    img_uint8 = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    pil_img   = Image.fromarray(img_uint8)
    buffer    = BytesIO()
    pil_img.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ============================================================
# MODELE PYDANTIC
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

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "vectors_loaded": len(image_names),
        "vector_dim": hybrid_matrix.shape[1],
        "device": device
    }


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """Caută imagini similare pe baza unui vector hibrid pre-calculat"""
    if len(req.vector) != hybrid_matrix.shape[1]:
        raise HTTPException(
            status_code=400,
            detail=f"Vector dimension mismatch: primit {len(req.vector)}, așteptat {hybrid_matrix.shape[1]}"
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


@app.post("/process", response_model=ProcessResponse)
async def process(file: UploadFile = File(...)):
    """
    Pipeline complet:
    1. Încarcă imaginea
    2. Extrage CLIP + defect vector
    3. Retrieval hibrid cu ponderi
    4. Corecție CLAHE
    5. Extrage LUT din perechea retrieved
    6. Aplică LUT moderat
    7. Returnează toate imaginile ca base64
    """
    # Citim imaginea
    contents = await file.read()
    pil_img  = Image.open(BytesIO(contents)).convert("RGB")
    img_np   = np.array(pil_img).astype(np.float32) / 255.0

    # 1. Extrage vectori
    clip_vec, defect_vec = get_hybrid_vector(pil_img)

    defect_names = ["blur", "noise", "overexposure", "underexposure", "compression"]
    defects_dict = {name: float(val) for name, val in zip(defect_names, defect_vec)}

    # 2. Retrieval
    results = retrieve_similar(clip_vec, defect_vec, top_k=5)
    retrieved_basename = results[0][0]
    similarity         = results[0][1]

    # 3. CLAHE
    corrected = correct_clahe(img_np)

    # 4. LUT
    raw_retr    = np.array(Image.open(os.path.join(RAW_DIR,    retrieved_basename + ".jpg"))).astype(np.float32) / 255.0
    edited_retr = np.array(Image.open(os.path.join(EDITED_DIR, retrieved_basename + ".jpg"))).astype(np.float32) / 255.0
    lut         = extract_colour_lut(raw_retr, edited_retr)

    # 5. Aplică LUT
    final = apply_lut_moderated(corrected, lut)

    return ProcessResponse(
        original_b64  = np_to_base64(img_np),
        corrected_b64 = np_to_base64(corrected),
        final_b64     = np_to_base64(final),
        defects       = defects_dict,
        retrieved     = retrieved_basename,
        similarity    = similarity,
        raw_b64       = np_to_base64(raw_retr),
        edited_b64    = np_to_base64(edited_retr),
    )


@app.get("/image/raw/{basename}")
def get_raw_image(basename: str):
    basename = Path(basename).name
    path = os.path.join(RAW_DIR, basename + ".jpg")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Imagine RAW negăsită")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/image/edited/{basename}")
def get_edited_image(basename: str):
    basename = Path(basename).name
    path = os.path.join(EDITED_DIR, basename + ".jpg")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Imagine editată negăsită")
    return FileResponse(path, media_type="image/jpeg")


# ============================================================
# RULARE
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)