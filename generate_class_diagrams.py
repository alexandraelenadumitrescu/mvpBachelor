"""
generate_class_diagrams.py
Generează 3 diagrame de clase UML pentru lucrarea de licență PhotoMatch.

Diagrama 1: Android — ierarhie activități + strat ML
Diagrama 2: Server — module FastAPI + modele Pydantic
Diagrama 3: Vedere de ansamblu client-server

Output: evaluation_assets/class_diagrams/
"""

import os, sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "evaluation_assets", "class_diagrams")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Culori ──────────────────────────────────────────────────────────────────
C_ABSTRACT   = "#D6E4F7"   # albastru deschis — clase abstracte / baze
C_ACTIVITY   = "#E8F5E9"   # verde deschis   — activități concrete
C_ML         = "#FFF3E0"   # portocaliu deschis — componente ML
C_API        = "#F3E5F5"   # mov deschis     — API / DTO
C_DB         = "#FCE4EC"   # roz deschis     — baza de date
C_SERVER     = "#E3F2FD"   # albastru pal    — server
C_PYDANTIC   = "#E8F5E9"   # verde           — modele Pydantic
C_ROUTER     = "#FFF9C4"   # galben          — routere FastAPI
C_BORDER     = "#424242"
C_HEADER     = "#1A237E"
C_TEXT       = "#212121"
C_GRAY       = "#757575"
C_INHERIT    = "#1565C0"
C_COMPOSE    = "#2E7D32"
C_DEPEND     = "#6A1B9A"
C_INTERFACE  = "#BF360C"

FONT = "DejaVu Sans"

# ─────────────────────────────────────────────────────────────────────────────
# Helper: desenează o clasă UML (box cu 3 secțiuni)
# ─────────────────────────────────────────────────────────────────────────────
def draw_class(ax, x, y, w, h,
               name, stereotype=None,
               attrs=None, methods=None,
               color=C_ABSTRACT, is_interface=False,
               is_abstract=False, fontsize=7.5):
    attrs   = attrs   or []
    methods = methods or []

    n_attrs   = len(attrs)
    n_methods = len(methods)
    total_rows = 1 + n_attrs + n_methods + (1 if n_attrs > 0 else 0) + (1 if n_methods > 0 else 0)
    row_h = h / max(total_rows, 3)

    # box principal
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01",
        facecolor=color, edgecolor=C_BORDER, linewidth=1.2, zorder=3)
    ax.add_patch(rect)

    # header (nume clasă)
    header_h = row_h * 1.4
    header_rect = mpatches.FancyBboxPatch(
        (x, y + h - header_h), w, header_h,
        boxstyle="round,pad=0.01",
        facecolor=color, edgecolor=C_BORDER, linewidth=1.2, zorder=4)
    ax.add_patch(header_rect)

    cy = y + h - header_h / 2
    if stereotype:
        ax.text(x + w/2, cy + header_h*0.22,
                f"«{stereotype}»", ha="center", va="center",
                fontsize=fontsize - 1, color=C_INTERFACE if is_interface else C_HEADER,
                fontfamily=FONT, zorder=5, style="italic")
        ax.text(x + w/2, cy - header_h*0.18,
                name, ha="center", va="center",
                fontsize=fontsize, color=C_HEADER, fontfamily=FONT, zorder=5,
                fontweight="bold",
                fontstyle="italic" if is_abstract else "normal")
    else:
        ax.text(x + w/2, cy,
                name, ha="center", va="center",
                fontsize=fontsize, color=C_HEADER, fontfamily=FONT, zorder=5,
                fontweight="bold",
                fontstyle="italic" if is_abstract else "normal")

    # linie separator attrs
    sep1_y = y + h - header_h
    if n_attrs > 0 or n_methods > 0:
        ax.plot([x, x + w], [sep1_y, sep1_y],
                color=C_BORDER, linewidth=0.7, zorder=5)

    # attrs
    row_y = sep1_y - row_h * 0.7
    for attr in attrs:
        ax.text(x + 0.06, row_y,
                attr, ha="left", va="center",
                fontsize=fontsize - 1, color=C_TEXT, fontfamily="monospace", zorder=5)
        row_y -= row_h

    # linie separator methods
    if n_attrs > 0 and n_methods > 0:
        sep2_y = row_y + row_h * 0.3
        ax.plot([x, x + w], [sep2_y, sep2_y],
                color=C_BORDER, linewidth=0.7, linestyle="--", zorder=5)
        row_y = sep2_y - row_h * 0.7

    # methods
    for method in methods:
        ax.text(x + 0.06, row_y,
                method, ha="left", va="center",
                fontsize=fontsize - 1, color=C_GRAY, fontfamily="monospace", zorder=5)
        row_y -= row_h

    return {"x": x, "y": y, "w": w, "h": h,
            "cx": x + w/2, "cy": y + h/2,
            "top": y + h, "bot": y,
            "left": x, "right": x + w}


def arrow(ax, x1, y1, x2, y2, style="inherit", label=""):
    """Desenează o săgeată UML între două puncte."""
    if style == "inherit":
        # linie continuă, triunghi gol la capăt (moștenire)
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=C_INHERIT, lw=1.5,
                        mutation_scale=12,
                        connectionstyle="arc3,rad=0.0"),
                    zorder=6)
    elif style == "compose":
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=C_COMPOSE, lw=1.2,
                        linestyle="solid",
                        mutation_scale=10,
                        connectionstyle="arc3,rad=0.0"),
                    zorder=6)
    elif style == "depend":
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=C_DEPEND, lw=1.0,
                        linestyle=(0, (4, 2)),
                        mutation_scale=9,
                        connectionstyle="arc3,rad=0.1"),
                    zorder=6)
    elif style == "realize":
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=C_INTERFACE, lw=1.2,
                        linestyle=(0, (3, 2)),
                        mutation_scale=10,
                        connectionstyle="arc3,rad=0.0"),
                    zorder=6)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.05, my + 0.05, label,
                fontsize=6, color=C_GRAY, fontfamily=FONT, zorder=7)


def legend(ax, x, y, items):
    """Desenează legenda."""
    for i, (color, label, ls) in enumerate(items):
        yy = y - i * 0.28
        if ls == "fill":
            p = mpatches.Rectangle((x, yy - 0.08), 0.22, 0.16,
                                    facecolor=color, edgecolor=C_BORDER, linewidth=0.8)
            ax.add_patch(p)
        else:
            ax.annotate("", xy=(x + 0.22, yy), xytext=(x, yy),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=1.3, mutation_scale=8,
                                        linestyle=ls))
        ax.text(x + 0.28, yy, label, va="center", fontsize=7,
                color=C_TEXT, fontfamily=FONT)


# ═════════════════════════════════════════════════════════════════════════════
# DIAGRAMA 1: Android — Ierarhie Activități + Strat ML
# ═════════════════════════════════════════════════════════════════════════════
print("Generez Diagrama 1: Android...")

fig, ax = plt.subplots(figsize=(22, 16), facecolor="white")
ax.set_xlim(0, 22); ax.set_ylim(0, 16)
ax.axis("off")
ax.set_facecolor("white")

fig.text(0.5, 0.985,
         "Diagrama de Clase — Android Client  |  PhotoMatch",
         ha="center", va="top", fontsize=14, fontweight="bold",
         color=C_HEADER, fontfamily=FONT)

# ── Ierarhie Activități ───────────────────────────────────────────────────────
# AppCompatActivity (extern, gri)
c_compat = draw_class(ax, 8.5, 13.5, 5, 1.2,
    "AppCompatActivity",
    stereotype="Android SDK",
    color="#EEEEEE", fontsize=8)

# BaseActivity
c_base = draw_class(ax, 7.0, 11.2, 5.0, 1.9,
    "BaseActivity",
    stereotype="abstract",
    is_abstract=True,
    attrs=["# requiresAuth(): boolean"],
    methods=["# api(): ApiService",
             "# showError(msg: String)",
             "# showServerIpDialog()"],
    color=C_ABSTRACT, fontsize=7.5)

# BaseLocalActivity
c_local = draw_class(ax, 6.5, 8.7, 6.0, 2.2,
    "BaseLocalActivity",
    stereotype="abstract",
    is_abstract=True,
    attrs=["# executor: ExecutorService",
           "# primaryButton: Button",
           "# progressBar: ProgressBar"],
    methods=["# setProcessing(b: boolean)"],
    color=C_ABSTRACT, fontsize=7.5)

# BaseServerActivity
c_server_act = draw_class(ax, 6.8, 6.8, 5.4, 1.6,
    "BaseServerActivity",
    stereotype="abstract",
    is_abstract=True,
    attrs=[],
    methods=["(inherits executor + setProcessing)"],
    color=C_ABSTRACT, fontsize=7.5)

# Activități concrete — stânga
c_login = draw_class(ax, 0.2, 9.0, 2.8, 1.4,
    "LoginActivity",
    methods=["login()", "onLoginSuccess()"],
    color=C_ACTIVITY, fontsize=7)
c_register = draw_class(ax, 0.2, 7.2, 2.8, 1.4,
    "RegisterActivity",
    methods=["register()"],
    color=C_ACTIVITY, fontsize=7)
c_favorites = draw_class(ax, 0.2, 5.4, 2.8, 1.4,
    "FavoritesActivity",
    methods=["loadFavorites()", "deleteFavorite()"],
    color=C_ACTIVITY, fontsize=7)

# Activități concrete — mijloc
c_proc = draw_class(ax, 3.5, 4.2, 3.0, 2.2,
    "ProcessingActivity",
    attrs=["- clipEncoder: CLIPEncoder",
           "- defectDetector: DefectDetector"],
    methods=["runInference(uri: Uri)",
             "sendToServer(vector: float[])"],
    color=C_ACTIVITY, fontsize=7)

c_results = draw_class(ax, 3.5, 1.7, 3.0, 2.1,
    "ResultsActivity",
    attrs=["- response: ProcessResponse"],
    methods=["displayResults()",
             "saveFavorite()"],
    color=C_ACTIVITY, fontsize=7)

# Activități concrete — dreapta
c_blur = draw_class(ax, 12.8, 4.2, 2.8, 2.0,
    "BlurActivity",
    attrs=["- regions: List<BlurRegion>"],
    methods=["detectBlur()", "applyBlur()",
             "rescaleRegions()"],
    color=C_ACTIVITY, fontsize=7)

c_scraper = draw_class(ax, 12.8, 1.9, 2.8, 1.9,
    "WebScraperActivity",
    attrs=["- riskFilter: String"],
    methods=["scrapeUrl()", "showResults()"],
    color=C_ACTIVITY, fontsize=7)

c_face = draw_class(ax, 16.0, 4.2, 2.8, 2.0,
    "FaceGroupsActivity",
    attrs=["- clusterer: FaceClusterer"],
    methods=["detectFaces()", "clusterFaces()",
             "showGroups()"],
    color=C_ACTIVITY, fontsize=7)

c_delivery = draw_class(ax, 16.0, 1.9, 2.8, 1.9,
    "DeliveryActivity",
    attrs=[],
    methods=["runDelivery()", "showMatches()"],
    color=C_ACTIVITY, fontsize=7)

c_pipeline = draw_class(ax, 19.2, 4.2, 2.6, 2.0,
    "PipelineActivity",
    attrs=["- steps: List<PipelineStep>"],
    methods=["runPipeline()", "onStepDone()"],
    color=C_ACTIVITY, fontsize=7)

c_cluster = draw_class(ax, 19.2, 1.9, 2.6, 1.9,
    "ClusterActivity",
    attrs=[],
    methods=["cluster(vectors)", "showClusters()"],
    color=C_ACTIVITY, fontsize=7)

# ── Strat ML ──────────────────────────────────────────────────────────────────
ml_y = 0.1
c_clip = draw_class(ax, 0.2, ml_y, 2.8, 1.4,
    "CLIPEncoder",
    attrs=["- dim: 512"],
    methods=["encode(bmp): float[512]"],
    color=C_ML, fontsize=7)

c_defect = draw_class(ax, 3.4, ml_y, 2.8, 1.4,
    "DefectDetector",
    attrs=["- dim: 5"],
    methods=["detect(bmp): float[5]"],
    color=C_ML, fontsize=7)

c_hybrid = draw_class(ax, 6.6, ml_y, 2.8, 1.4,
    "HybridVectorBuilder",
    attrs=["+ DEFECT_WEIGHT=5.0"],
    methods=["build(clip, defect): float[517]"],
    color=C_ML, fontsize=7)

c_corrector = draw_class(ax, 9.8, ml_y, 2.8, 1.4,
    "ImageCorrector",
    attrs=["+ LUT_STRENGTH=0.2"],
    methods=["correct(bmp, lut): Bitmap"],
    color=C_ML, fontsize=7)

c_face_emb = draw_class(ax, 13.0, ml_y, 2.8, 1.4,
    "FaceEmbedder",
    attrs=["- dim: 128"],
    methods=["embed(crop): float[128]"],
    color=C_ML, fontsize=7)

c_face_cl = draw_class(ax, 16.2, ml_y, 2.8, 1.4,
    "FaceClusterer",
    attrs=["+ THRESHOLD=0.75"],
    methods=["cluster(faces): List<FaceCluster>"],
    color=C_ML, fontsize=7)

# ── API Layer ─────────────────────────────────────────────────────────────────
c_api_svc = draw_class(ax, 8.5, 11.3, 5.0, 1.8,
    "ApiService",
    stereotype="interface",
    is_interface=True,
    methods=["searchAndCorrect(req): Response",
             "blurDetect(file): BlurDetectResponse",
             "scrapeImages(req): ScrapeResponse",
             "deliveryRun(…): DeliveryResponse"],
    color=C_API, fontsize=7)

c_api_client = draw_class(ax, 8.5, 8.9, 5.0, 2.1,
    "ApiClient",
    stereotype="singleton",
    attrs=["- instance: ApiClient",
           "+ DEFAULT_IP: String"],
    methods=["service(): ApiService",
             "saveToken(t: String)",
             "isLoggedIn(): boolean"],
    color=C_API, fontsize=7)

# ── DB Layer ──────────────────────────────────────────────────────────────────
c_db = draw_class(ax, 19.0, 11.2, 2.8, 1.7,
    "AppDatabase",
    stereotype="RoomDatabase",
    methods=["favoriteDao(): FavoriteDao"],
    color=C_DB, fontsize=7)

c_dao = draw_class(ax, 19.0, 9.0, 2.8, 1.9,
    "FavoriteDao",
    stereotype="interface",
    is_interface=True,
    methods=["insert(f: FavoritePhoto)",
             "delete(f: FavoritePhoto)",
             "getAll(): List<FavoritePhoto>"],
    color=C_DB, fontsize=7)

c_entity = draw_class(ax, 19.0, 6.8, 2.8, 1.9,
    "FavoritePhoto",
    stereotype="Entity",
    attrs=["+ id: int (PK)",
           "+ correctedBase64: String",
           "+ retrieved: String",
           "+ timestamp: long"],
    color=C_DB, fontsize=7)

# ── PhotoMatchApp ──────────────────────────────────────────────────────────────
c_app = draw_class(ax, 19.0, 14.0, 2.8, 1.5,
    "PhotoMatchApp",
    stereotype="Application",
    methods=["onCreate()",
             "ApiClient.init(this)"],
    color=C_ABSTRACT, fontsize=7)

# ── Relații moștenire (ierarhie activități) ───────────────────────────────────
arrow(ax, c_compat["cx"],  c_compat["bot"],
          c_base["cx"],    c_base["top"],   style="inherit")
arrow(ax, c_base["cx"],    c_base["bot"],
          c_local["cx"],   c_local["top"],  style="inherit")
arrow(ax, c_local["cx"],   c_local["bot"],
          c_server_act["cx"], c_server_act["top"], style="inherit")

# BaseActivity <- LoginActivity, RegisterActivity, FavoritesActivity
for c in [c_login, c_register, c_favorites]:
    arrow(ax, c["right"], c["cy"],
              c_base["left"], c_base["cy"], style="inherit")

# BaseServerActivity <- activitati concrete
for c in [c_proc, c_results]:
    arrow(ax, c["right"], c["cy"],
              c_server_act["left"], c_server_act["cy"], style="inherit")
for c in [c_blur, c_scraper]:
    arrow(ax, c["left"], c["cy"],
              c_server_act["right"], c_server_act["cy"], style="inherit")
for c in [c_face, c_delivery, c_pipeline, c_cluster]:
    arrow(ax, c["left"], c["cy"],
              c_server_act["right"], c_server_act["cy"] - 0.2, style="inherit")

# BaseActivity -> ApiClient
arrow(ax, c_base["right"], c_base["cy"],
          c_api_client["left"], c_api_client["cy"], style="depend", label="uses")

# ApiClient -> ApiService
arrow(ax, c_api_client["cx"], c_api_client["top"],
          c_api_svc["cx"],    c_api_svc["bot"], style="realize")

# ProcessingActivity -> ML
for c_ml in [c_clip, c_defect, c_hybrid, c_corrector]:
    arrow(ax, c_proc["cx"] - 0.2, c_proc["bot"],
              c_ml["cx"], c_ml["top"], style="compose", label="")

# FaceGroupsActivity -> FaceEmbedder, FaceClusterer
arrow(ax, c_face["cx"], c_face["bot"],
          c_face_emb["cx"], c_face_emb["top"], style="compose")
arrow(ax, c_face["cx"] + 0.3, c_face["bot"],
          c_face_cl["cx"], c_face_cl["top"], style="compose")

# DB relații
arrow(ax, c_db["cx"], c_db["bot"],
          c_dao["cx"], c_dao["top"], style="compose")
arrow(ax, c_dao["cx"], c_dao["bot"],
          c_entity["cx"], c_entity["top"], style="depend")

# Results -> AppDatabase
arrow(ax, c_results["right"], c_results["cy"],
          c_db["left"], c_db["cy"] + 0.3, style="depend", label="saves to")

# ── Legendă ───────────────────────────────────────────────────────────────────
legend(ax, 0.2, 15.6, [
    (C_ABSTRACT,  "Clasă abstractă (base)",     "fill"),
    (C_ACTIVITY,  "Activitate concretă (UI)",   "fill"),
    (C_ML,        "Componentă ML / procesare",  "fill"),
    (C_API,       "API / Client Retrofit",       "fill"),
    (C_DB,        "Bază de date (Room)",         "fill"),
    (C_INHERIT,   "Moștenire (extends)",         "solid"),
    (C_COMPOSE,   "Compoziție (uses/has)",       "solid"),
    (C_DEPEND,    "Dependență (calls)",          "solid"),
    (C_INTERFACE, "Realizare (implements)",      "solid"),
])

plt.savefig(os.path.join(OUT_DIR, "diagram1_android.png"),
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print("  Salvat: diagram1_android.png")


# ═════════════════════════════════════════════════════════════════════════════
# DIAGRAMA 2: Server — FastAPI + Module Python
# ═════════════════════════════════════════════════════════════════════════════
print("Generez Diagrama 2: Server...")

fig, ax = plt.subplots(figsize=(20, 14), facecolor="white")
ax.set_xlim(0, 20); ax.set_ylim(0, 14)
ax.axis("off")
ax.set_facecolor("white")

fig.text(0.5, 0.988,
         "Diagrama de Clase — Server FastAPI  |  PhotoMatch",
         ha="center", va="top", fontsize=14, fontweight="bold",
         color=C_HEADER, fontfamily=FONT)

# ── FastAPI App core ──────────────────────────────────────────────────────────
c_app_srv = draw_class(ax, 7.5, 11.5, 5.0, 2.2,
    "server.py  (FastAPI App)",
    stereotype="FastAPI",
    attrs=["+ DEFECT_WEIGHT = 5.0",
           "+ LUT_STRENGTH  = 0.2",
           "+ LUT_SIZE      = 17"],
    methods=["startup(): precompute_luts()",
             "POST /search_and_correct()",
             "POST /process()",
             "POST /blur-detect()",
             "POST /cluster()"],
    color=C_SERVER, fontsize=7.5)

# ── Auth module ───────────────────────────────────────────────────────────────
c_user = draw_class(ax, 0.2, 11.2, 3.2, 2.2,
    "User",
    stereotype="SQLAlchemy Model",
    attrs=["+ id: Integer (PK)",
           "+ email: String (unique)",
           "+ hashed_password: String",
           "+ is_active: Boolean"],
    color=C_DB, fontsize=7)

c_auth_router = draw_class(ax, 0.2, 8.6, 3.2, 2.2,
    "auth/router.py",
    stereotype="APIRouter",
    methods=["POST /register(UserCreate)",
             "POST /login(OAuth2Form)",
             "GET  /me() [auth]"],
    color=C_ROUTER, fontsize=7)

c_auth_dep = draw_class(ax, 0.2, 6.8, 3.2, 1.6,
    "auth/dependencies.py",
    methods=["get_current_active_user()",
             "create_access_token()",
             "verify_password()"],
    color=C_ROUTER, fontsize=7)

# ── Blur module ───────────────────────────────────────────────────────────────
c_blur_detect = draw_class(ax, 0.2, 4.8, 3.2, 1.8,
    "blur_api/local_detector.py",
    attrs=["CONF_THRESHOLD = 0.25",
           "SENSITIVE_CLASSES: dict"],
    methods=["detect_sensitive(img): list",
             "detect_badges(img): list"],
    color=C_PYDANTIC, fontsize=7)

c_badge_det = draw_class(ax, 0.2, 2.9, 3.2, 1.6,
    "blur_api/badge_detector_v3.py",
    attrs=["MODEL: YOLOv8 fine-tuned",
           "CLASSES: Cards, Lanyard"],
    methods=["detect(img, conf): list[dict]"],
    color=C_PYDANTIC, fontsize=7)

c_gemini_det = draw_class(ax, 0.2, 1.2, 3.2, 1.4,
    "blur_api/gemini_detector.py",
    attrs=["API: Gemini 2.0 Flash"],
    methods=["detect(img): list[dict]"],
    color=C_PYDANTIC, fontsize=7)

# ── Scrape module ─────────────────────────────────────────────────────────────
c_scrape_router = draw_class(ax, 4.0, 9.5, 3.4, 2.2,
    "scrape_api/router.py",
    stereotype="APIRouter",
    attrs=["prefix: /scrape"],
    methods=["POST /images(ScrapeRequest) [auth]"],
    color=C_ROUTER, fontsize=7)

c_scraper_logic = draw_class(ax, 4.0, 7.2, 3.4, 2.0,
    "scrape_api/image_scraper.py",
    attrs=["MAX_IMAGES = 30",
           "MAX_SIZE = 3 MB"],
    methods=["scrape_and_analyze(url): dict",
             "compute_risk(regions): float"],
    color=C_PYDANTIC, fontsize=7)

# ── FAISS / Search ────────────────────────────────────────────────────────────
c_faiss = draw_class(ax, 4.0, 4.8, 3.4, 2.2,
    "FAISS Index",
    stereotype="IndexFlatIP",
    attrs=["dim = 517",
           "n_vectors = 3499",
           "metric = cosine (IP)"],
    methods=["search(query, k) -> (D, I)",
             "add(vectors)"],
    color="#E8EAF6", fontsize=7)

c_lut_engine = draw_class(ax, 4.0, 2.5, 3.4, 2.0,
    "LUT Engine (server.py)",
    attrs=["lut_cache: dict",
           "LUT_SIZE = 17"],
    methods=["extract_colour_lut(raw, edited)",
             "apply_lut_moderated(img, lut)",
             "correct_clahe(img)"],
    color="#E8EAF6", fontsize=7)

# ── ML Models (server) ────────────────────────────────────────────────────────
c_clip_srv = draw_class(ax, 12.8, 11.0, 3.2, 2.0,
    "CLIP ViT-B-32",
    stereotype="open_clip",
    attrs=["dim_out = 512",
           "pretrained = openai"],
    methods=["encode_image(img): tensor[512]"],
    color=C_ML, fontsize=7)

c_defect_srv = draw_class(ax, 12.8, 8.6, 3.2, 2.0,
    "DefectHead",
    stereotype="MobileNetV3Small",
    attrs=["dim_out = 5",
           "weights: defect_head.pt"],
    methods=["forward(img): float[5]"],
    color=C_ML, fontsize=7)

c_hybrid_srv = draw_class(ax, 12.8, 6.4, 3.2, 1.8,
    "HybridVector Builder",
    attrs=["DEFECT_WEIGHT = 5.0",
           "dim = 517"],
    methods=["build(clip, defect): float[517]",
             "l2_normalize(v): float[517]"],
    color=C_ML, fontsize=7)

# ── Photo Mailer ──────────────────────────────────────────────────────────────
c_matcher = draw_class(ax, 16.4, 11.0, 3.4, 2.0,
    "photo_mailer/matcher.py",
    attrs=["THRESHOLD = 0.70"],
    methods=["match_photos(dir, db): dict",
             "cosine_sim(a, b): float"],
    color=C_PYDANTIC, fontsize=7)

c_face_cl_srv = draw_class(ax, 16.4, 8.6, 3.4, 2.0,
    "photo_mailer/face_clusterer.py",
    attrs=["cluster_threshold = 0.75"],
    methods=["cluster_faces(faces): list",
             "extract_all_faces(dir): list"],
    color=C_PYDANTIC, fontsize=7)

c_tflite_emb = draw_class(ax, 16.4, 6.4, 3.4, 1.8,
    "photo_mailer/tflite_embedder.py",
    attrs=["model: facenet.tflite",
           "dim_out = 128"],
    methods=["embed(face_img): float[128]"],
    color=C_ML, fontsize=7)

c_face_db = draw_class(ax, 16.4, 4.4, 3.4, 1.8,
    "photo_mailer/face_db.py",
    methods=["build_db(employees): dict",
             "download_photo(url): Image"],
    color=C_PYDANTIC, fontsize=7)

# ── Pydantic DTOs principale ──────────────────────────────────────────────────
c_s_req = draw_class(ax, 7.8, 8.2, 2.4, 1.8,
    "SearchAndCorrect\nRequest",
    stereotype="BaseModel",
    attrs=["vector: list[float]",
           "top_k: int = 5"],
    color=C_API, fontsize=7)

c_s_resp = draw_class(ax, 10.5, 8.2, 2.4, 1.8,
    "SearchAndCorrect\nResponse",
    stereotype="BaseModel",
    attrs=["retrieved: str",
           "similarity: float",
           "lut_cached: bool"],
    color=C_API, fontsize=7)

c_blur_resp = draw_class(ax, 7.8, 5.9, 2.4, 1.8,
    "BlurDetect\nResponse",
    stereotype="BaseModel",
    attrs=["regions: list[BlurRegion]",
           "image_width: int",
           "image_height: int"],
    color=C_API, fontsize=7)

c_cluster_req = draw_class(ax, 10.5, 5.9, 2.4, 1.8,
    "ClusterResponse",
    stereotype="BaseModel",
    attrs=["clusters: list[ClusterInfo]",
           "n_clusters: int",
           "silhouette_score: float"],
    color=C_API, fontsize=7)

# SQLAlchemy DB
c_sql_db = draw_class(ax, 7.8, 3.6, 2.4, 1.6,
    "Database (SQLite)",
    stereotype="SQLAlchemy",
    attrs=["users table: User"],
    color=C_DB, fontsize=7)

# ── Relații ───────────────────────────────────────────────────────────────────
# App include routers
arrow(ax, c_app_srv["left"],  c_app_srv["cy"],
          c_auth_router["right"], c_auth_router["cy"], style="depend", label="includes")
arrow(ax, c_app_srv["left"],  c_app_srv["cy"] - 0.3,
          c_scrape_router["right"], c_scrape_router["cy"], style="depend", label="includes")

# App uses ML
arrow(ax, c_app_srv["right"], c_app_srv["cy"] + 0.3,
          c_clip_srv["left"], c_clip_srv["cy"], style="compose", label="uses")
arrow(ax, c_app_srv["right"], c_app_srv["cy"],
          c_defect_srv["left"], c_defect_srv["cy"], style="compose")
arrow(ax, c_app_srv["right"], c_app_srv["cy"] - 0.3,
          c_hybrid_srv["left"], c_hybrid_srv["cy"], style="compose")

# App uses FAISS + LUT
arrow(ax, c_app_srv["cx"], c_app_srv["bot"],
          c_faiss["cx"],   c_faiss["top"],  style="depend", label="searches")
arrow(ax, c_app_srv["cx"] + 0.3, c_app_srv["bot"],
          c_lut_engine["cx"], c_lut_engine["top"], style="depend", label="corrects")

# Auth
arrow(ax, c_auth_router["cx"], c_auth_router["bot"],
          c_auth_dep["cx"],   c_auth_dep["top"], style="depend")
arrow(ax, c_auth_dep["cx"], c_auth_dep["bot"],
          c_sql_db["left"],  c_sql_db["cy"], style="depend", label="CRUD")

# Blur
arrow(ax, c_blur_detect["cx"], c_blur_detect["bot"],
          c_badge_det["cx"],   c_badge_det["top"], style="depend", label="or")
arrow(ax, c_badge_det["cx"], c_badge_det["bot"],
          c_gemini_det["cx"], c_gemini_det["top"], style="depend", label="or")

# Scrape -> blur_detect
arrow(ax, c_scraper_logic["left"], c_scraper_logic["cy"],
          c_blur_detect["right"], c_blur_detect["cy"], style="depend")

# Matcher -> tflite
arrow(ax, c_matcher["cx"],   c_matcher["bot"],
          c_face_cl_srv["cx"], c_face_cl_srv["top"], style="depend")
arrow(ax, c_face_cl_srv["cx"], c_face_cl_srv["bot"],
          c_tflite_emb["cx"], c_tflite_emb["top"], style="depend")
arrow(ax, c_matcher["cx"] + 0.2, c_matcher["bot"],
          c_face_db["cx"],   c_face_db["top"], style="depend")

# DTOs
arrow(ax, c_app_srv["cx"] - 0.5, c_app_srv["bot"],
          c_s_req["cx"], c_s_req["top"], style="depend")
arrow(ax, c_app_srv["cx"] + 0.5, c_app_srv["bot"],
          c_blur_resp["cx"], c_blur_resp["top"], style="depend")

# Legendă
legend(ax, 0.2, 0.9, [
    (C_SERVER,    "Modul principal FastAPI",       "fill"),
    (C_ROUTER,    "Router / Dependință FastAPI",   "fill"),
    (C_ML,        "Model ML",                      "fill"),
    (C_API,       "Schema Pydantic (DTO)",         "fill"),
    (C_DB,        "Persistență (SQLAlchemy/SQLite)","fill"),
    (C_DEPEND,    "Dependență (uses/calls)",        "solid"),
    (C_COMPOSE,   "Compoziție (has/creates)",       "solid"),
])

plt.savefig(os.path.join(OUT_DIR, "diagram2_server.png"),
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print("  Salvat: diagram2_server.png")


# ═════════════════════════════════════════════════════════════════════════════
# DIAGRAMA 3: Vedere de ansamblu Client-Server
# ═════════════════════════════════════════════════════════════════════════════
print("Generez Diagrama 3: Overview Client-Server...")

fig, ax = plt.subplots(figsize=(18, 11), facecolor="white")
ax.set_xlim(0, 18); ax.set_ylim(0, 11)
ax.axis("off")
fig.text(0.5, 0.99,
         "Vedere de Ansamblu — Arhitectura Sistemului PhotoMatch",
         ha="center", va="top", fontsize=14, fontweight="bold",
         color=C_HEADER, fontfamily=FONT)

# ── Background zones ──────────────────────────────────────────────────────────
android_bg = mpatches.FancyBboxPatch((0.1, 0.3), 7.8, 10.2,
    boxstyle="round,pad=0.1", facecolor="#F1F8E9",
    edgecolor="#4CAF50", linewidth=2, linestyle="--", zorder=1)
ax.add_patch(android_bg)
ax.text(0.5, 10.35, "Android Client  (Java)", fontsize=11,
        color="#2E7D32", fontweight="bold", fontfamily=FONT, zorder=2)

server_bg = mpatches.FancyBboxPatch((10.0, 0.3), 7.8, 10.2,
    boxstyle="round,pad=0.1", facecolor="#E3F2FD",
    edgecolor="#1565C0", linewidth=2, linestyle="--", zorder=1)
ax.add_patch(server_bg)
ax.text(10.4, 10.35, "Server  (Python / FastAPI)", fontsize=11,
        color="#0D47A1", fontweight="bold", fontfamily=FONT, zorder=2)

# ── Android blocks ────────────────────────────────────────────────────────────
c_ui = draw_class(ax, 0.4, 7.5, 3.5, 2.4,
    "UI Layer",
    attrs=["20+ Activities",
           "(ProcessingActivity,",
           " BlurActivity,",
           " WebScraperActivity, ...)"],
    methods=["onUserAction()",
             "displayResult()"],
    color=C_ACTIVITY, fontsize=8)

c_ml_and = draw_class(ax, 0.4, 4.5, 3.5, 2.6,
    "ML Layer (local)",
    attrs=["CLIPEncoder (512-dim)",
           "DefectDetector (5-dim)",
           "HybridVectorBuilder (517-dim)",
           "FaceEmbedder (128-dim)",
           "ImageCorrector (CLAHE+LUT)"],
    color=C_ML, fontsize=8)

c_api_and = draw_class(ax, 0.4, 2.5, 3.5, 1.7,
    "API Layer",
    attrs=["ApiClient (Singleton)",
           "ApiService (Retrofit)"],
    methods=["POST / GET endpoints"],
    color=C_API, fontsize=8)

c_db_and = draw_class(ax, 0.4, 0.6, 3.5, 1.6,
    "Persistence Layer",
    attrs=["AppDatabase (Room)",
           "FavoritePhoto (Entity)"],
    color=C_DB, fontsize=8)

# ── Server blocks ─────────────────────────────────────────────────────────────
c_api_srv = draw_class(ax, 10.4, 7.5, 3.4, 2.4,
    "API Layer",
    attrs=["FastAPI + Pydantic schemas",
           "JWT Auth (24h HS256)",
           "CORS middleware"],
    methods=["20+ REST endpoints"],
    color=C_ROUTER, fontsize=8)

c_ml_srv = draw_class(ax, 10.4, 4.5, 3.4, 2.6,
    "ML Layer (server)",
    attrs=["CLIP ViT-B-32 (open_clip)",
           "MobileNetV3Small defect head",
           "FAISS IndexFlatIP (3499 vec.)",
           "FaceNet TFLite (128-dim)",
           "LUT Engine (17³ grid)"],
    color=C_ML, fontsize=8)

c_data_srv = draw_class(ax, 10.4, 2.5, 3.4, 1.7,
    "Data Layer",
    attrs=["MIT FiveK: 3499 raw/edited pairs",
           "hybrid_vectors.npz (517-dim)",
           "lut_cache.npz"],
    color=C_DB, fontsize=8)

c_persist_srv = draw_class(ax, 10.4, 0.6, 3.4, 1.6,
    "Persistence",
    attrs=["SQLite (User accounts)",
           "LUT cache (NPZ on disk)"],
    color=C_DB, fontsize=8)

# ── Layere interne Android ─────────────────────────────────────────────────────
for c_top, c_bot in [(c_ui, c_ml_and), (c_ml_and, c_api_and), (c_api_and, c_db_and)]:
    arrow(ax, c_top["cx"], c_top["bot"],
              c_bot["cx"], c_bot["top"], style="compose")

# ── Layere interne Server ──────────────────────────────────────────────────────
for c_top, c_bot in [(c_api_srv, c_ml_srv), (c_ml_srv, c_data_srv), (c_data_srv, c_persist_srv)]:
    arrow(ax, c_top["cx"], c_top["bot"],
              c_bot["cx"], c_bot["top"], style="compose")

# ── Săgeți HTTP ───────────────────────────────────────────────────────────────
http_calls = [
    (7.7, "POST /search_and_correct\n(vector 517-dim JSON)", 10.0, 8.5),
    (7.7, "POST /blur-detect\n(multipart JPEG)", 10.0, 8.1),
    (7.7, "POST /scrape/images\n(url JSON)", 10.0, 7.7),
    (7.7, "POST /delivery/run\n(multipart photos)", 10.0, 7.3),
]
for i, (x1, label, x2, y_offset) in enumerate(http_calls):
    yy = 8.7 - i * 0.55
    ax.annotate("", xy=(x2, yy), xytext=(x1, yy),
                arrowprops=dict(arrowstyle="-|>", color="#E65100",
                                lw=1.5, mutation_scale=10))
    ax.text(8.85, yy + 0.08, label, ha="center", va="bottom",
            fontsize=6.5, color="#E65100", fontfamily=FONT)

# Răspuns (săgeată înapoi)
ax.annotate("", xy=(7.7, 6.0), xytext=(10.0, 6.0),
            arrowprops=dict(arrowstyle="-|>", color="#1B5E20",
                            lw=1.5, mutation_scale=10, linestyle=(0,(4,2))))
ax.text(8.85, 6.12, "Response JSON\n(base64 images,\n similarity, risk_label)",
        ha="center", va="bottom", fontsize=6.5, color="#1B5E20", fontfamily=FONT)

# WiFi label
ax.text(8.85, 5.2, "HTTP\n(same Wi-Fi)",
        ha="center", va="center", fontsize=8.5, color="#555555",
        fontfamily=FONT, style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF9C4",
                  edgecolor="#F9A825", linewidth=1))

# Legendă
legend(ax, 14.2, 3.8, [
    (C_ACTIVITY,  "UI / Activitate",         "fill"),
    (C_ML,        "ML / Procesare imagine",  "fill"),
    (C_API,       "API / DTO",               "fill"),
    (C_DB,        "Persistență",             "fill"),
    (C_COMPOSE,   "Relație internă",         "solid"),
    ("#E65100",   "HTTP Request (Android→Server)", "solid"),
    ("#1B5E20",   "HTTP Response (Server→Android)", "solid"),
])

plt.savefig(os.path.join(OUT_DIR, "diagram3_overview.png"),
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print("  Salvat: diagram3_overview.png")

# ── PlantUML source (pentru render extern) ────────────────────────────────────
puml = """@startuml PhotoMatch_Android
skinparam classAttributeIconSize 0
skinparam classFontSize 11
skinparam classFontName Arial
skinparam classHeaderBackgroundColor #D6E4F7
skinparam classBackgroundColor #FAFAFA
skinparam classBorderColor #424242
skinparam arrowColor #1565C0
skinparam linetype ortho

abstract class BaseActivity {
  # requiresAuth() : boolean
  # api() : ApiService
  # showError(msg)
  # showServerIpDialog()
}

abstract class BaseLocalActivity {
  # executor : ExecutorService
  # primaryButton : Button
  # progressBar : ProgressBar
  # setProcessing(b)
}

abstract class BaseServerActivity {
}

class ProcessingActivity {
  - clipEncoder : CLIPEncoder
  - defectDetector : DefectDetector
  + runInference(uri)
  + sendToServer(vector)
}

class BlurActivity {
  - regions : List<BlurRegion>
  + detectBlur()
  + rescaleRegions()
}

class WebScraperActivity {
  - riskFilter : String
  + scrapeUrl()
}

class FaceGroupsActivity {
  - clusterer : FaceClusterer
  + detectFaces()
  + clusterFaces()
}

class CLIPEncoder {
  - dim : 512
  + encode(bmp) : float[512]
}

class DefectDetector {
  - dim : 5
  + detect(bmp) : float[5]
}

class HybridVectorBuilder {
  + DEFECT_WEIGHT : 5.0
  + build(clip, defect) : float[517]
}

class ImageCorrector {
  + LUT_STRENGTH : 0.2
  + correct(bmp, lut) : Bitmap
}

class FaceEmbedder {
  - dim : 128
  + embed(crop) : float[128]
}

class FaceClusterer {
  + THRESHOLD : 0.75
  + cluster(faces) : List
}

interface ApiService {
  + searchAndCorrect(req)
  + blurDetect(file)
  + scrapeImages(req)
  + deliveryRun(...)
}

class ApiClient << singleton >> {
  - instance : ApiClient
  + DEFAULT_IP : String
  + service() : ApiService
  + isLoggedIn() : boolean
}

entity FavoritePhoto {
  + id : int
  + correctedBase64 : String
  + retrieved : String
  + timestamp : long
}

AppCompatActivity <|-- BaseActivity
BaseActivity <|-- BaseLocalActivity
BaseLocalActivity <|-- BaseServerActivity
BaseServerActivity <|-- ProcessingActivity
BaseServerActivity <|-- BlurActivity
BaseServerActivity <|-- WebScraperActivity
BaseServerActivity <|-- FaceGroupsActivity

BaseActivity --> ApiClient : uses
ApiClient ..|> ApiService : creates

ProcessingActivity *-- CLIPEncoder
ProcessingActivity *-- DefectDetector
ProcessingActivity *-- HybridVectorBuilder
ProcessingActivity *-- ImageCorrector
FaceGroupsActivity *-- FaceEmbedder
FaceGroupsActivity *-- FaceClusterer

@enduml"""

with open(os.path.join(OUT_DIR, "photomatch_android.puml"), "w", encoding="utf-8") as f:
    f.write(puml)
print("  Salvat: photomatch_android.puml")

print(f"\nGata. 3 PNG + 1 PlantUML in: {OUT_DIR}")
print("\nPentru a randa .puml: paste la https://plantuml.com/plantuml/uml/")
