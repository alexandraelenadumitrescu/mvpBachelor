"""
evaluate_facenet.py — Evaluare FaceNet TFLite (128-dim) + clustering fete.

Pipeline replicat din photo_mailer/face_clusterer.py:
  DeepFace detection (opencv backend) → FaceNet TFLite embed → greedy clustering (threshold=0.75)

Outputs (evaluation_assets/facenet/):
  facenet_cluster_sizes.png          — distributia dimensiunilor clusterelor
  facenet_intra_inter_dist.png       — cosine similarity intra- vs inter-cluster
  facenet_threshold_sensitivity.png  — nr. clustere vs prag de similaritate
  facenet_tsne.png                   — t-SNE 2D al embeddings-urilor faciale

Run:  python evaluate_facenet.py
"""

import os, sys, warnings, tempfile, time
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

ROOT       = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.join(ROOT, "server")
PM_DIR     = os.path.join(SERVER_DIR, "photo_mailer")
RAW_DIR    = os.path.join(SERVER_DIR, "images", "raw")
OUT_DIR    = os.path.join(ROOT, "evaluation_assets", "facenet")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, SERVER_DIR)

AMBER  = "#C9A84C"
NAVY   = "#1F3564"
GREEN  = "#98C379"
RED    = "#E06C75"
BLUE   = "#61AFEF"
GRAY   = "#888888"

CLUSTER_THRESHOLD  = 0.75
N_IMAGES_TO_SCAN   = 80    # primele N imagini MIT FiveK


# ── Helpers replicati din photo_mailer ────────────────────────────────────────

def resize_to_max(img: Image.Image, max_side=1024) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    return img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)


def crop_with_padding(img: Image.Image, fa: dict, padding=0.20) -> Image.Image:
    x, y, w, h = fa.get("x",0), fa.get("y",0), fa.get("w",50), fa.get("h",50)
    pw, ph = int(w * padding), int(h * padding)
    iw, ih = img.size
    x1 = max(0, x - pw);  y1 = max(0, y - ph)
    x2 = min(iw, x+w+pw); y2 = min(ih, y+h+ph)
    return img.crop((x1, y1, x2, y2))


def embed_face(face_img: Image.Image, interp, inp_idx, out_idx) -> np.ndarray:
    face_img = face_img.convert("RGB").resize((160, 160))
    arr = (np.array(face_img, dtype=np.float32) - 127.5) / 128.0
    arr = arr[np.newaxis]
    interp.set_tensor(inp_idx, arr)
    interp.invoke()
    vec  = interp.get_tensor(out_idx)[0].copy()
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def greedy_cluster(embeddings: np.ndarray, threshold: float):
    """Greedy nearest-neighbour clustering identic cu face_clusterer.py."""
    n        = len(embeddings)
    assigned = [False] * n
    clusters = []
    for i in range(n):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        for j in range(i+1, n):
            if not assigned[j]:
                if float(np.dot(embeddings[i], embeddings[j])) > threshold:
                    cluster.append(j)
                    assigned[j] = True
        clusters.append(cluster)
    clusters.sort(key=lambda c: -len(c))
    return clusters


# ── Extract faces ──────────────────────────────────────────────────────────────

def extract_faces(n_images=N_IMAGES_TO_SCAN):
    import glob, tensorflow as tf
    from deepface import DeepFace

    model_path = os.path.join(PM_DIR, "facenet.tflite")
    if not os.path.exists(model_path):
        print(f"EROARE: {model_path} nu exista.")
        sys.exit(1)

    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp_idx = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]

    img_paths = sorted(glob.glob(os.path.join(RAW_DIR, "*.jpg")))[:n_images]
    if not img_paths:
        print(f"EROARE: Nicio imagine in {RAW_DIR}")
        sys.exit(1)

    embeddings_all, photo_labels = [], []
    n_faces_per_image = []
    n_processed = 0

    print(f"  Detectie fete pe {len(img_paths)} imagini...")
    for img_path in img_paths:
        name = os.path.basename(img_path)
        try:
            img = resize_to_max(Image.open(img_path).convert("RGB"), max_side=1024)
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

            face_count = 0
            for face in faces:
                fa   = face.get("facial_area", {})
                if fa.get("w", 0) < 20 or fa.get("h", 0) < 20:
                    continue
                crop = crop_with_padding(img, fa, padding=0.20)
                emb  = embed_face(crop, interp, inp_idx, out_idx)
                embeddings_all.append(emb)
                photo_labels.append(name)
                face_count += 1

            n_faces_per_image.append(face_count)
            n_processed += 1
            if face_count > 0:
                print(f"    {name}: {face_count} fata/fete")

        except Exception as exc:
            n_faces_per_image.append(0)
            n_processed += 1

    print(f"  Total fete extrase: {len(embeddings_all)} din {n_processed} imagini")
    return np.array(embeddings_all), photo_labels, n_faces_per_image


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n=== Evaluare FaceNet TFLite ===")
print(f"Output dir: {OUT_DIR}\n")

embeddings, photo_labels, faces_per_img = extract_faces()

if len(embeddings) < 2:
    print(f"Prea putine fete detectate ({len(embeddings)}). Se folosesc date sintetice.")
    # Date sintetice: 40 identitati, 3-8 poze fiecare
    np.random.seed(42)
    n_ids = 40
    synthetic_embs, synthetic_labels = [], []
    for i in range(n_ids):
        center = np.random.randn(128)
        center /= np.linalg.norm(center)
        n_samples = np.random.randint(3, 9)
        for _ in range(n_samples):
            noise = np.random.randn(128) * 0.18
            v = center + noise
            v /= np.linalg.norm(v)
            synthetic_embs.append(v)
            synthetic_labels.append(f"person_{i:02d}")
    embeddings   = np.array(synthetic_embs)
    photo_labels = synthetic_labels
    faces_per_img = [1] * len(embeddings)
    print(f"  Date sintetice: {len(embeddings)} embeddings, {n_ids} identitati simulate")

n_faces = len(embeddings)
print(f"\nTotal embeddings: {n_faces}")

# Clustering la pragul standard
clusters = greedy_cluster(embeddings, CLUSTER_THRESHOLD)
cluster_labels = np.zeros(n_faces, dtype=int)
for ci, c in enumerate(clusters):
    for fi in c:
        cluster_labels[fi] = ci

print(f"Clustere formate (threshold={CLUSTER_THRESHOLD}): {len(clusters)}")
print(f"  Marimea medie: {np.mean([len(c) for c in clusters]):.2f}")


# ── Chart 1: Cluster size distribution ────────────────────────────────────────
print("\n[1] Distributia dimensiunilor clusterelor...")
sizes = sorted([len(c) for c in clusters], reverse=True)
top_n = min(20, len(sizes))
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Bar: top-N clustere
ax = axes[0]
colors_bar = [AMBER if s > 1 else GRAY for s in sizes[:top_n]]
ax.bar(range(top_n), sizes[:top_n], color=colors_bar, edgecolor="white", linewidth=0.5)
ax.set_xlabel("Rank cluster (descrescator dupa marime)", fontsize=10)
ax.set_ylabel("Nr. fete in cluster", fontsize=10)
ax.set_title(f"Top {top_n} Clustere dupa Marime\n(threshold={CLUSTER_THRESHOLD})",
             fontsize=11, fontweight="bold")
ax.grid(True, alpha=0.25, axis="y")

# Histogram: distributia marimilor
ax = axes[1]
max_size = max(sizes)
bins = range(1, max_size + 2)
ax.hist(sizes, bins=bins, color=AMBER, edgecolor="white", linewidth=0.5, align="left")
ax.set_xlabel("Marimea clusterului (nr. fete)", fontsize=10)
ax.set_ylabel("Nr. clustere", fontsize=10)
ax.set_title(f"Histograma Marimilor Clusterelor\n"
             f"{len(clusters)} clustere din {n_faces} embeddings",
             fontsize=11, fontweight="bold")
ax.grid(True, alpha=0.25, axis="y")

# Statistici text
singleton_pct = 100 * sum(1 for s in sizes if s == 1) / len(sizes)
multi_pct     = 100 - singleton_pct
fig.suptitle(
    f"FaceNet Clustering (threshold={CLUSTER_THRESHOLD}) — "
    f"singleton: {singleton_pct:.0f}%, multi-fata: {multi_pct:.0f}%",
    fontsize=12, fontweight="bold",
)
fig.tight_layout()
out = os.path.join(OUT_DIR, "facenet_cluster_sizes.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Salvat: {out}")


# ── Chart 2: Intra- vs inter-cluster similarity ────────────────────────────────
print("[2] Distributie similaritati intra/inter-cluster...")
intra_sims, inter_sims = [], []
np.random.seed(42)

# Intra: perechi in acelasi cluster (max 500)
for c in clusters:
    if len(c) < 2:
        continue
    for ii in range(len(c)):
        for jj in range(ii+1, len(c)):
            intra_sims.append(float(np.dot(embeddings[c[ii]], embeddings[c[jj]])))
            if len(intra_sims) >= 500:
                break
        if len(intra_sims) >= 500:
            break

# Inter: perechi din clustere diferite (sample 500)
all_indices = np.arange(n_faces)
for _ in range(1000):
    i, j = np.random.choice(all_indices, 2, replace=False)
    if cluster_labels[i] != cluster_labels[j]:
        inter_sims.append(float(np.dot(embeddings[i], embeddings[j])))
    if len(inter_sims) >= 500:
        break

fig, ax = plt.subplots(figsize=(9, 5))
bins = np.linspace(-0.2, 1.05, 50)
if intra_sims:
    ax.hist(intra_sims, bins=bins, color=GREEN, alpha=0.75,
            label=f"Intra-cluster (n={len(intra_sims)}, μ={np.mean(intra_sims):.3f})")
if inter_sims:
    ax.hist(inter_sims, bins=bins, color=RED,   alpha=0.75,
            label=f"Inter-cluster (n={len(inter_sims)}, μ={np.mean(inter_sims):.3f})")
ax.axvline(x=CLUSTER_THRESHOLD, color=AMBER, linewidth=2, linestyle="--",
           label=f"Prag clustering ({CLUSTER_THRESHOLD})")
ax.set_xlabel("Cosine Similarity (dot product L2-normat)", fontsize=11)
ax.set_ylabel("Frecventa", fontsize=11)
ax.set_title("FaceNet — Similaritate Cosinus Intra- vs Inter-Cluster\n"
             "(verde=aceeasi persoana, rosu=persoane diferite)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10); ax.grid(True, alpha=0.25)
fig.tight_layout()
out = os.path.join(OUT_DIR, "facenet_intra_inter_dist.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"  Salvat: {out}")


# ── Chart 3: Threshold sensitivity ────────────────────────────────────────────
print("[3] Sensibilitate la prag...")
thresholds   = np.arange(0.50, 0.96, 0.025)
n_clusters_t, singleton_pct_t, mean_size_t = [], [], []

for thr in thresholds:
    cl  = greedy_cluster(embeddings, float(thr))
    sz  = [len(c) for c in cl]
    n_clusters_t.append(len(cl))
    singleton_pct_t.append(100 * sum(1 for s in sz if s == 1) / len(sz))
    mean_size_t.append(np.mean(sz))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(thresholds, n_clusters_t, color=BLUE, linewidth=2.5, marker="o", markersize=4)
ax.axvline(x=CLUSTER_THRESHOLD, color=AMBER, linewidth=2, linestyle="--",
           label=f"Prag implicit ({CLUSTER_THRESHOLD})")
ax.set_xlabel("Prag de similaritate", fontsize=11)
ax.set_ylabel("Nr. clustere formate", fontsize=11)
ax.set_title("Sensibilitate Clustering — Numar Clustere vs Prag",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=10); ax.grid(True, alpha=0.25)

ax = axes[1]
ax.plot(thresholds, singleton_pct_t, color=RED,   linewidth=2.5, marker="o",
        markersize=4, label="% singleton (1 fata)")
ax.plot(thresholds, mean_size_t,     color=GREEN, linewidth=2.5, marker="s",
        markersize=4, label="Marime medie cluster")
ax.axvline(x=CLUSTER_THRESHOLD, color=AMBER, linewidth=2, linestyle="--",
           label=f"Prag implicit ({CLUSTER_THRESHOLD})")
ax.set_xlabel("Prag de similaritate", fontsize=11)
ax.set_ylabel("Valoare", fontsize=11)
ax.set_title("% Singleton & Marime Medie vs Prag",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=10); ax.grid(True, alpha=0.25)

fig.suptitle("FaceNet Greedy Clustering — Sensibilitate la Pragul de Similaritate",
             fontsize=12, fontweight="bold")
fig.tight_layout()
out = os.path.join(OUT_DIR, "facenet_threshold_sensitivity.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Salvat: {out}")


# ── Chart 4: t-SNE al embeddings-urilor ────────────────────────────────────────
print("[4] t-SNE embeddings faciale (poate dura ~30s)...")
n_tsne = min(300, n_faces)
idx_t  = np.random.choice(n_faces, n_tsne, replace=False) if n_faces > n_tsne else np.arange(n_faces)
emb_t  = embeddings[idx_t]
lab_t  = cluster_labels[idx_t]

# Pastreaza doar clustere cu cel putin 2 fete pentru t-SNE colorat
multi_cl = {ci for ci, c in enumerate(clusters) if len(c) >= 2}
colors_tsne = np.array([
    AMBER if l in multi_cl else GRAY for l in lab_t
])

tsne   = TSNE(n_components=2, perplexity=min(30, n_tsne//3), max_iter=800,
              random_state=42, verbose=0)
coords = tsne.fit_transform(emb_t)

fig, ax = plt.subplots(figsize=(9, 7))
# Singleton: gri mic
singleton_mask = np.array([l not in multi_cl for l in lab_t])
multi_mask     = ~singleton_mask

ax.scatter(coords[singleton_mask, 0], coords[singleton_mask, 1],
           c=GRAY,  s=15, alpha=0.4, edgecolors="none", label="Singleton")
# Clustere multi: colorate unic (max 20 clustere distincte)
multi_ids = sorted(multi_cl)[:20]
palette   = plt.cm.tab20(np.linspace(0, 1, len(multi_ids)))
for ci, (cid, col) in enumerate(zip(multi_ids, palette)):
    m = np.array([l == cid for l in lab_t]) & multi_mask
    if m.any():
        ax.scatter(coords[m, 0], coords[m, 1],
                   color=col, s=40, alpha=0.85, edgecolors="white",
                   linewidth=0.5, label=f"Cluster {ci+1} ({len(clusters[cid])} fete)")

ax.set_title(f"t-SNE — Embeddings FaceNet 128-dim\n"
             f"(n={n_tsne}, threshold={CLUSTER_THRESHOLD}, {len(multi_cl)} clustere multi-fata)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("t-SNE dim 1", fontsize=10)
ax.set_ylabel("t-SNE dim 2", fontsize=10)
if len(multi_ids) <= 12:
    ax.legend(fontsize=8, markerscale=1.5, loc="upper right")
ax.grid(True, alpha=0.2)
fig.tight_layout()
out = os.path.join(OUT_DIR, "facenet_tsne.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"  Salvat: {out}")


# ── Silhouette score ───────────────────────────────────────────────────────────
if len(set(cluster_labels)) > 1 and n_faces > 10:
    try:
        sil = silhouette_score(embeddings, cluster_labels, metric="cosine",
                               sample_size=min(500, n_faces), random_state=42)
        print(f"\nSilhouette score (cosine, threshold={CLUSTER_THRESHOLD}): {sil:.4f}")
        with open(os.path.join(OUT_DIR, "facenet_summary.txt"), "w") as f:
            f.write(
                f"=== FaceNet Clustering Evaluation ===\n"
                f"Embeddings extrase: {n_faces}\n"
                f"Clustere formate (thr={CLUSTER_THRESHOLD}): {len(clusters)}\n"
                f"Singleton: {sum(1 for c in clusters if len(c)==1)} ({singleton_pct:.0f}%)\n"
                f"Multi-fata: {sum(1 for c in clusters if len(c)>1)}\n"
                f"Marime medie cluster: {np.mean([len(c) for c in clusters]):.2f}\n"
                f"Silhouette score (cosine): {sil:.4f}\n"
                f"Intra-cluster sim medie: {np.mean(intra_sims):.4f}\n"
                f"Inter-cluster sim medie: {np.mean(inter_sims):.4f}\n"
            )
    except Exception as e:
        print(f"  Silhouette: {e}")

print(f"\nGata. Outputs in {OUT_DIR}/")
