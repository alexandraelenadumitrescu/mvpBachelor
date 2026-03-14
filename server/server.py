"""
Server FastAPI pentru retrieval hibrid CLIP + defecte
Rulează local: uvicorn server:app --host 0.0.0.0 --port 8000

Structura așteptată:
server_data/
  server.py              <- acest fișier
  hybrid_vectors.npz     <- vectorii din Colab
  images/
    raw/                 <- imaginile RAW convertite
    edited/              <- imaginile editate de expert
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import faiss
import os
import json
from pathlib import Path

app = FastAPI()

# CORS pentru Android
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# CONFIG — modifică doar astea
# ============================================================

# Folderul cu imaginile RAW din FiveK (copiate local)
RAW_DIR     = "./images/raw"

# Folderul cu imaginile editate de expert A
EDITED_DIR  = "./images/edited"

# Fișierul cu vectorii hibrizi pre-calculați
VECTORS_PATH = "./hybrid_vectors.npz"

# ============================================================
# ÎNCĂRCARE DATE LA STARTUP
# ============================================================

print("Încarc vectorii hibrizi...")
data         = np.load(VECTORS_PATH, allow_pickle=True)
hybrid_matrix = data["vectors"].astype("float32")
image_names   = data["names"].tolist()

print(f"Încarcat {len(image_names)} vectori de dimensiune {hybrid_matrix.shape[1]}")

print("Construiesc index FAISS...")
dimension = hybrid_matrix.shape[1]
index     = faiss.IndexFlatIP(dimension)  # Inner Product = cosine pe vectori normalizați

# Normalizăm vectorii pentru cosine similarity
norms = np.linalg.norm(hybrid_matrix, axis=1, keepdims=True)
hybrid_matrix_norm = hybrid_matrix / (norms + 1e-8)
index.add(hybrid_matrix_norm)

print(f"Index FAISS gata cu {index.ntotal} vectori")

# ============================================================
# MODELE PYDANTIC
# ============================================================

class SearchRequest(BaseModel):
    vector: list[float]   # vectorul hibrid de la telefon (517 dim)
    top_k: int = 5

class SearchResult(BaseModel):
    basename: str
    distance: float
    raw_url: str
    edited_url: str

class SearchResponse(BaseModel):
    results: list[SearchResult]

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok", "vectors_loaded": len(image_names)}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):

    if len(req.vector) != hybrid_matrix.shape[1]:
        raise HTTPException(
            status_code=400,
            detail=f"Vector dimension mismatch: primit {len(req.vector)}, așteptat {hybrid_matrix.shape[1]}"
        )

    # Normalizăm query pentru cosine similarity
    query = np.array(req.vector, dtype="float32").reshape(1, -1)
    norm  = np.linalg.norm(query)
    query = query / (norm + 1e-8)

    # Căutăm top_k + 1 ca să excludem query-ul însuși dacă e în bază
    D, I = index.search(query, req.top_k + 1)

    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0:
            continue

        basename = image_names[idx]

        # Verificăm că imaginile există
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


@app.get("/image/raw/{basename}")
def get_raw_image(basename: str):
    """Returnează imaginea RAW ca JPEG"""
    # Sanitizare basename
    basename = Path(basename).name
    path = os.path.join(RAW_DIR, basename + ".jpg")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Imagine RAW negăsită")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/image/edited/{basename}")
def get_edited_image(basename: str):
    """Returnează imaginea editată de expert ca JPEG"""
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