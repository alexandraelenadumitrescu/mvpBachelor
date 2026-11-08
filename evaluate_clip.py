"""
evaluate_clip.py — Evaluare retrieval CLIP-only vs Hybrid (512+5 dim, FAISS).

Outputs (evaluation_assets/clip/):
  clip_retrieval_bar.png        — Top-1 / Top-5 cosine similarity: CLIP vs Hybrid
  clip_improvement_hist.png     — Distributia imbunatatirilor per query
  clip_scatter.png              — Scatter CLIP top-1 vs Hybrid top-1 per query
  clip_defect_tsne.png          — t-SNE al scorurilor de defect (5-dim) din hybrid_vectors.npz
  clip_summary.txt              — rezumat numeric

Run:  python evaluate_clip.py
"""

import os, sys, random, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

ROOT        = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR  = os.path.join(ROOT, "server")
VECTORS_PATH = os.path.join(SERVER_DIR, "hybrid_vectors.npz")
OUT_DIR     = os.path.join(ROOT, "evaluation_assets", "clip")
os.makedirs(OUT_DIR, exist_ok=True)

CLIP_WEIGHT   = 1.0
DEFECT_WEIGHT = 5.0
N_QUERIES     = 200
TOP_K         = 5
SEED          = 42

AMBER  = "#C9A84C"
NAVY   = "#1F3564"
GRAY   = "#888888"
GREEN  = "#98C379"
RED    = "#E06C75"
BLUE   = "#61AFEF"
DEFECT_NAMES = ["blur", "noise", "overexposure", "underexposure", "compression"]


# ── Load vectors ───────────────────────────────────────────────────────────────
print("Incarcare vectori hibrizi...")
if not os.path.exists(VECTORS_PATH):
    print(f"EROARE: {VECTORS_PATH} nu exista.")
    sys.exit(1)

data          = np.load(VECTORS_PATH, allow_pickle=True)
hybrid_matrix = data["vectors"].astype("float32")   # (3499, 517)
image_names   = data["names"].tolist()
N             = len(image_names)
print(f"  {N} vectori, dim={hybrid_matrix.shape[1]}")

clip_part   = hybrid_matrix[:, :512]
defect_part = hybrid_matrix[:, 512:]

clip_norm   = clip_part   / (np.linalg.norm(clip_part,   axis=1, keepdims=True) + 1e-8)
defect_norm = defect_part / (np.linalg.norm(defect_part, axis=1, keepdims=True) + 1e-8)

weighted      = np.concatenate(
    [clip_norm * CLIP_WEIGHT, defect_norm * DEFECT_WEIGHT], axis=1
).astype("float32")
weighted_norm = weighted / (np.linalg.norm(weighted, axis=1, keepdims=True) + 1e-8)

clip_index   = faiss.IndexFlatIP(512)
clip_index.add(clip_norm)
hybrid_index = faiss.IndexFlatIP(517)
hybrid_index.add(weighted_norm)

print(f"  FAISS CLIP index: {clip_index.ntotal} | Hybrid index: {hybrid_index.ntotal}")


# ── Retrieval evaluation ───────────────────────────────────────────────────────
def top_k_excl_self(index, vec, q_idx, k):
    D, I = index.search(vec.reshape(1, -1), k + 1)
    return [(float(d), int(i)) for d, i in zip(D[0], I[0]) if i >= 0 and i != q_idx][:k]

random.seed(SEED)
query_indices = random.sample(range(N), min(N_QUERIES, N))
print(f"\nEvaluare retrieval pe {len(query_indices)} query-uri (TOP_K={TOP_K})...")

clip_top1, clip_top5, hybrid_top1, hybrid_top5, improvements = [], [], [], [], []

for pos, q_idx in enumerate(query_indices):
    if (pos + 1) % 50 == 0:
        print(f"  {pos+1}/{len(query_indices)}")

    c_hits = top_k_excl_self(clip_index,   clip_norm[q_idx],    q_idx, TOP_K)
    h_hits = top_k_excl_self(hybrid_index, weighted_norm[q_idx], q_idx, TOP_K)

    ct1 = c_hits[0][0] if c_hits else 0.0
    ct5 = float(np.mean([h[0] for h in c_hits])) if c_hits else 0.0
    ht1 = h_hits[0][0] if h_hits else 0.0
    ht5 = float(np.mean([h[0] for h in h_hits])) if h_hits else 0.0

    clip_top1.append(ct1);  clip_top5.append(ct5)
    hybrid_top1.append(ht1); hybrid_top5.append(ht5)
    improvements.append(ht1 - ct1)

clip_top1    = np.array(clip_top1);    clip_top5    = np.array(clip_top5)
hybrid_top1  = np.array(hybrid_top1);  hybrid_top5  = np.array(hybrid_top5)
improvements = np.array(improvements)

pct_wins = 100.0 * (improvements > 0).sum() / len(query_indices)
print(f"\n  CLIP    Top-1={clip_top1.mean():.4f}  Top-5={clip_top5.mean():.4f}")
print(f"  Hybrid  Top-1={hybrid_top1.mean():.4f}  Top-5={hybrid_top5.mean():.4f}")
print(f"  Hybrid bate CLIP pe Top-1 in {pct_wins:.1f}% din query-uri")


# ── Chart 1: Bar chart ─────────────────────────────────────────────────────────
print("\n[1] Bar chart retrieval...")
fig, ax = plt.subplots(figsize=(8, 5))
x      = np.array([0.0, 1.0])
width  = 0.32
cv = [clip_top1.mean(),   clip_top5.mean()]
hv = [hybrid_top1.mean(), hybrid_top5.mean()]

bc = ax.bar(x - width/2, cv, width, label="CLIP-only (512 dim)", color=GRAY,  alpha=0.9)
bh = ax.bar(x + width/2, hv, width, label=f"Hybrid (517 dim, w={DEFECT_WEIGHT}×defect)", color=AMBER, alpha=0.9)

for bar, val in [(b, v) for bars, vals in [(bc, cv), (bh, hv)] for b, v in zip(bars, vals)]:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(["Top-1 similarity", "Top-5 mean similarity"], fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Cosine similarity", fontsize=11)
ax.set_title("Calitate Retrieval: CLIP-only vs Hybrid Vector\n"
             f"({len(query_indices)} query-uri, {N} imagini MIT FiveK)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10); ax.grid(True, alpha=0.25, axis="y")
fig.tight_layout()
out = os.path.join(OUT_DIR, "clip_retrieval_bar.png")
fig.savefig(out, dpi=150); plt.close(fig)
print(f"  Salvat: {out}")


# ── Chart 2: Improvement histogram ────────────────────────────────────────────
print("[2] Histograma imbunatatiri...")
fig, ax = plt.subplots(figsize=(8, 5))
pos_mask = improvements > 0
neg_mask = improvements < 0
eq_mask  = improvements == 0

bins = np.linspace(improvements.min() - 0.01, improvements.max() + 0.01, 40)
ax.hist(improvements[pos_mask], bins=bins, color=GREEN, alpha=0.8, label=f"Hybrid mai bun ({pos_mask.sum()})")
ax.hist(improvements[neg_mask], bins=bins, color=RED,   alpha=0.8, label=f"CLIP mai bun ({neg_mask.sum()})")
ax.hist(improvements[eq_mask],  bins=bins, color=GRAY,  alpha=0.6, label=f"Egal ({eq_mask.sum()})")
ax.axvline(x=0,                   color="black", linewidth=1.5, linestyle="--")
ax.axvline(x=improvements.mean(), color=AMBER,  linewidth=2,   linestyle="-",
           label=f"Media: {improvements.mean():+.4f}")
ax.set_xlabel("Δ Top-1 (Hybrid − CLIP)", fontsize=11)
ax.set_ylabel("Nr. query-uri", fontsize=11)
ax.set_title("Distributia Imbunatatirii de Retrieval per Query\n(Hybrid Top-1 − CLIP Top-1)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10); ax.grid(True, alpha=0.25)
fig.tight_layout()
out = os.path.join(OUT_DIR, "clip_improvement_hist.png")
fig.savefig(out, dpi=150); plt.close(fig)
print(f"  Salvat: {out}")


# ── Chart 3: Scatter CLIP vs Hybrid per query ─────────────────────────────────
print("[3] Scatter CLIP vs Hybrid...")
fig, ax = plt.subplots(figsize=(7, 7))
colors = np.where(improvements > 0.005, GREEN, np.where(improvements < -0.005, RED, GRAY))
ax.scatter(clip_top1, hybrid_top1, c=colors, alpha=0.5, s=20, edgecolors="none")
lo, hi = min(clip_top1.min(), hybrid_top1.min()) - 0.02, max(clip_top1.max(), hybrid_top1.max()) + 0.02
ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.5, linestyle="--", label="y = x (paritate)")
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color=GREEN, label=f"Hybrid mai bun ({pos_mask.sum()})"),
    Patch(color=RED,   label=f"CLIP mai bun ({neg_mask.sum()})"),
    Patch(color=GRAY,  label=f"Egal (±0.005) ({eq_mask.sum()})"),
], fontsize=10)
ax.set_xlabel("CLIP-only Top-1 cosine similarity", fontsize=11)
ax.set_ylabel("Hybrid Top-1 cosine similarity",    fontsize=11)
ax.set_title("Scatter per Query: CLIP vs Hybrid\n(punctele deasupra diagonalei = Hybrid mai bun)",
             fontsize=12, fontweight="bold")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.grid(True, alpha=0.25)
fig.tight_layout()
out = os.path.join(OUT_DIR, "clip_scatter.png")
fig.savefig(out, dpi=150); plt.close(fig)
print(f"  Salvat: {out}")


# ── Chart 4: t-SNE al scorurilor de defect ────────────────────────────────────
print("[4] t-SNE defect scores (poate dura ~1 min)...")
np.random.seed(SEED)
n_tsne  = min(1000, N)
idx_tsne = np.random.choice(N, n_tsne, replace=False)
defect_subset = defect_part[idx_tsne]        # (1000, 5) — valori brute, nu normate

tsne   = TSNE(n_components=2, perplexity=40, max_iter=800, random_state=SEED, verbose=0)
coords = tsne.fit_transform(defect_subset)   # (1000, 2)

# Coloreaza dupa defectul dominant (argmax al celor 5 scoruri)
dom_defect = np.argmax(defect_subset, axis=1)
DEFECT_COLORS = [RED, BLUE, AMBER, GREEN, NAVY]

fig, ax = plt.subplots(figsize=(9, 7))
for i, (name, color) in enumerate(zip(DEFECT_NAMES, DEFECT_COLORS)):
    mask = dom_defect == i
    ax.scatter(coords[mask, 0], coords[mask, 1],
               c=color, s=14, alpha=0.6, edgecolors="none", label=f"{name} ({mask.sum()})")

ax.set_title("t-SNE — Scoruri de Defect (5-dim) din Hybrid Vectors\n"
             f"(colorat dupa defectul dominant, n={n_tsne})",
             fontsize=12, fontweight="bold")
ax.set_xlabel("t-SNE dim 1", fontsize=10)
ax.set_ylabel("t-SNE dim 2", fontsize=10)
ax.legend(fontsize=10, markerscale=2)
ax.grid(True, alpha=0.2)
fig.tight_layout()
out = os.path.join(OUT_DIR, "clip_defect_tsne.png")
fig.savefig(out, dpi=150); plt.close(fig)
print(f"  Salvat: {out}")


# ── Summary text ───────────────────────────────────────────────────────────────
summary = (
    f"=== CLIP-only vs Hybrid Retrieval (N={len(query_indices)} query-uri) ===\n"
    f"CLIP-only   Top-1: {clip_top1.mean():.4f}   Top-5 mean: {clip_top5.mean():.4f}\n"
    f"Hybrid      Top-1: {hybrid_top1.mean():.4f}   Top-5 mean: {hybrid_top5.mean():.4f}\n"
    f"Delta Top-1: {improvements.mean():+.4f}   Delta Top-5: {(hybrid_top5-clip_top5).mean():+.4f}\n"
    f"Hybrid castiga pe Top-1 in {pct_wins:.1f}% din query-uri\n"
    f"Imbunatatire maxima: {improvements.max():+.4f}   Degradare maxima: {improvements.min():+.4f}\n"
    f"Dataset: {N} imagini MIT FiveK, vectori {hybrid_matrix.shape[1]}-dim\n"
    f"Defect weight: {DEFECT_WEIGHT}x\n"
)
summary_path = os.path.join(OUT_DIR, "clip_summary.txt")
with open(summary_path, "w") as f:
    f.write(summary)
print(f"\n{summary}")
print(f"Gata. Outputs in {OUT_DIR}/")
