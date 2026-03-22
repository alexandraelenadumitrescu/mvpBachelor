# PhotoMatch Server

FastAPI server for hybrid image retrieval and colour correction.
Combines CLIP semantic embeddings with defect quality scores for retrieval,
then applies CLAHE + 3D LUT correction matched from the retrieved image pair.

## Prerequisites

- Python 3.10 or newer
- CUDA (optional — falls back to CPU automatically)

## Setup

### 1. Install dependencies

```bash
cd server
pip install -r requirements.txt
```

### 2. Place required files

| File | Where | Notes |
|---|---|---|
| `defect_head.pt` | `server/defect_head.pt` | MobileNetV3Small trained checkpoint |
| `hybrid_vectors.npz` | `server/hybrid_vectors.npz` | Pre-computed — already present |
| `images/raw/*.jpg` | `server/images/raw/` | FiveK RAW JPEGs — already present |
| `images/edited/*.jpg` | `server/images/edited/` | Expert-edited JPEGs — already present |

### 3. Start the server

**Windows (double-click):**
```
start_server.bat
```

**Command line:**
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Interactive API docs are available at `http://localhost:8000/docs` once the server is running.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Server status and vector count |
| `POST` | `/process` | Full pipeline: embed → retrieve → CLAHE → LUT |
| `POST` | `/search` | Vector similarity search (pre-computed vector) |
| `GET` | `/image/raw/{basename}` | Serve a RAW image by basename |
| `GET` | `/image/edited/{basename}` | Serve an edited image by basename |

### `/health` response

```json
{
  "status": "ok",
  "vectors_loaded": 3499,
  "vector_dim": 517,
  "device": "cpu"
}
```

### `/process` request

Multipart form upload with field name `file`.

```bash
curl -X POST http://localhost:8000/process \
     -F "file=@photo.jpg"
```

### `/process` response

```json
{
  "original_b64":  "<base64 JPEG>",
  "corrected_b64": "<base64 JPEG — CLAHE applied>",
  "final_b64":     "<base64 JPEG — LUT applied>",
  "defects": {
    "blur": 0.12,
    "noise": 0.08,
    "overexposure": 0.05,
    "underexposure": 0.31,
    "compression": 0.19
  },
  "retrieved": "a1234-Expert-A-001",
  "similarity": 0.94,
  "raw_b64":    "<base64 JPEG — retrieved RAW reference>",
  "edited_b64": "<base64 JPEG — retrieved edited reference>"
}
```

## Android connection

In `app-photomatch/src/main/java/com/photomatch/api/ApiClient.java`, set:

```java
private static final String BASE_URL = "http://<your-PC-IP>:8000/";
```

Find your PC's local IP with `ipconfig` (Windows) and look for the IPv4 address
on your Wi-Fi adapter. Make sure the phone and PC are on the same Wi-Fi network.

The app requires `usesCleartextTraffic="true"` in the manifest (already set) because
the server runs over plain HTTP.

## Pipeline details

```
Input image
  → CLIP ViT-B-32 encode         → 512-dim embedding (L2 normalised)
  → MobileNetV3Small defect head → 5-dim scores [blur, noise, over, under, compression]
  → Weighted concat (1.0×CLIP + 5.0×defect, re-normalised) → 517-dim query
  → FAISS IndexFlatIP search     → top-5 similar images
  → CLAHE on L channel (LAB)     → exposure-corrected image
  → 3D LUT from best match RAW→edited pair (17³ grid, LinearNDInterpolator)
  → Moderated LUT blend (strength=0.3) → final image
```
