# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PhotoMatch is a hybrid Android + FastAPI image processing platform. The Android client handles local ML inference (CLIP ViT-B-32, defect detection via MobileNetV3Small) and sends pre-computed 517-dim vectors to the server. The server runs FAISS similarity search against 3,499 MIT FiveK reference images and applies CLAHE + 3D LUT color correction.

## Build & Run

### Server
```bash
cd server
pip install -r requirements.txt
# Create server/.env (see Environment section below)
uvicorn server:app --host 0.0.0.0 --port 8000
# Windows shortcut: start_server.bat
```
API docs at `http://localhost:8000/docs`.

### Android
```bash
cd android
./gradlew assembleDebug     # build APK
./gradlew installDebug      # build + install on connected device
```
Built with Gradle 8.3.2, Java 11, compileSdk/targetSdk 34, minSdk 26. Phone and server must be on the same Wi-Fi network. Set server IP in-app on first launch (default `10.33.163.180`; find via `ipconfig`).

## Architecture

### Android — Java only, no Kotlin

**Activity hierarchy:**
```
AppCompatActivity
  └─ BaseActivity          — auth guard (onStart redirect), api() helper, showError(), showServerIpDialog()
       └─ BaseLocalActivity — ExecutorService executor, setProcessing(boolean) controls primaryButton + progressBar
            └─ BaseServerActivity — used for all network-calling activities (no extra logic beyond BaseLocalActivity)
```

Every new activity that calls the server extends `BaseServerActivity`. Assign `primaryButton` and `progressBar` fields in `onCreate` so `setProcessing()` works automatically.

**Async pattern — no ViewModel or LiveData:**
```java
executor.execute(() -> {
    // background work, or synchronous .execute() Retrofit calls
    runOnUiThread(() -> { /* UI updates */ });
});
// fire-and-forget alternative:
api().method(...).enqueue(new Callback<T>() { ... });
```

**API client:** `ApiClient` is a singleton initialised in `PhotoMatchApp.onCreate()`. Server IP stored in SharedPreferences (`photomatch_prefs` / `server_ip`). An OkHttp interceptor attaches `Authorization: Bearer {token}` to every request except `/auth/login` and `/auth/register`. Base URL: `http://{SERVER_IP}:8000/`.

**Image transport to server:**
- **JSON POST** — vector-based endpoints (`/search_and_correct`, `/cluster`, `/scrape/images`): send pre-computed `float[]` as JSON body; Android runs CLIP + defect inference locally via TFLite.
- **Multipart** — file endpoints (`/blur-detect`, `/blur-sensitive`, `/delivery/run`): `MultipartBody.Part` with part name `"file"`, JPEG bytes.

**Request/response POJOs:** flat classes with public fields in `android/src/main/java/com/photomatch/api/`. Gson deserialises by field name.

**Room database** (`db/`): single `FavoritePhoto` entity, used only in `ResultsActivity` and `FavoritesActivity`.

### Server — Python / FastAPI

**Route organisation:**
- `auth/router.py` → mounted at `/auth` (register, login, me)
- `scrape_api/router.py` → mounted at `/scrape` (images)
- Everything else inline in `server.py`: blur (`/blur-*`), delivery (`/delivery/*`), image serving (`/image/*`), search, process, cluster, style

**Auth pattern:** endpoints that require auth use `Depends(get_current_active_user)` from `auth/dependencies.py`. JWT HS256, 24 h expiry. Unprotected: `/health`, `/search*`, `/process`, `/batch`, `/cluster`, `/style/*`, `/image/*`. Auth-required: all `/blur-*`, `/delivery/*`, `/scrape/*`, `/mail/*`.

**New router convention:** create `feature_api/router.py` with its own `APIRouter()`, add `Depends(get_current_active_user)` on every protected endpoint, mount in `server.py` with `app.include_router(feature_api.router, prefix="/feature")`.

**Blur pipeline (three endpoints, all auth-required):**
- `/blur-detect` → returns `{regions, image_width, image_height}`, no blur applied
- `/blur-apply` → caller supplies region list as JSON form field, returns JPEG bytes
- `/blur-sensitive` → one-shot detect + apply, returns JPEG bytes
- Backends: `"local"` (YOLOv8n conf=0.25 + OpenCV badge contours) or `"gemini"` (Gemini 2.0 Flash)

**Risk scoring (`scrape_api/image_scraper.py::compute_risk`):**
```
per-region weight: badge/id/license/plate=1.5, document=1.2, person=0.8, screen/monitor/laptop=0.6, other=0.5
risk_score = min(sum_of_weights / 5.0, 1.0)
label:  ≤0 → NONE  |  ≤0.3 → LOW  |  ≤0.7 → MEDIUM  |  >0.7 → HIGH
```

**Hybrid vector (517-dim):** `[CLIP_512 × 1.0, defect_5 × 5.0]` L2-normalised. FAISS `IndexFlatIP` over 3,499 FiveK images.

## Key Constants

| Location | Constant | Value |
|---|---|---|
| `ApiClient.java` | `DEFAULT_IP` | `10.33.163.180` |
| `ApiClient.java` | `PREFS_NAME` | `"photomatch_prefs"` |
| `server.py` | `DEFECT_WEIGHT` | 5.0 |
| `server.py` | `LUT_STRENGTH` | 0.2 |
| `server.py` | `LUT_SIZE` | 17 |
| `blur_api/local_detector.py` | YOLO conf | 0.25 |
| `photo_mailer/matcher.py` | face match threshold | 0.70 |
| `photo_mailer/face_clusterer.py` | cluster threshold | 0.75 |

## Layout Conventions

- Activity layouts: `res/layout/activity_{name}.xml`
- RecyclerView item layouts: `res/layout/item_{name}.xml`
- All text: `android:fontFamily="monospace"`
- Accent colour: `@color/color_amber` (`#C9A84C`)
- Risk colours: HIGH=`#FF5555`, MEDIUM=`#FFB300`, LOW=`#81C784`
- Primary button: amber background (`android:backgroundTint="@color/color_amber"`), black text, 46–52 dp height
- Secondary/text button: `android:background="@android:color/transparent"`, amber text
- `primaryButton` and `progressBar` fields must be assigned in `onCreate` for `BaseLocalActivity.setProcessing()` to work

## Environment

`server/.env` (not committed):
```
GOOGLE_API_KEY=...          # Gemini vision API (for blur-sensitive detector=gemini)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=...
SMTP_PASSWORD=...           # Gmail app password
JWT_SECRET_KEY=...          # Defaults to insecure dev value if absent
BADGE_MODEL_PATH=...        # Optional fine-tuned YOLOv8 for badge detection
```
