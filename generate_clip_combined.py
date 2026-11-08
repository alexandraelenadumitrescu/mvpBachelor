"""
generate_clip_combined.py — Bar chart + histograma distributie intr-o singura figura.
Output: evaluation_assets/clip/clip_combined.png
"""

import os, sys, random, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT         = os.path.dirname(os.path.abspath(__file__))
VECTORS_PATH = os.path.join(ROOT, "server", "hybrid_vectors.npz")
OUT_PATH     = os.path.join(ROOT, "evaluation_assets", "clip", "clip_combined.png")

DEFECT_WEIGHT = 5.0
N_QUERIES     = 200
TOP_K         = 5
SEED          = 42

AMBER = "#C9A84C"
GRAY  = "#666666"
GREEN = "#2E7D32"
RED   = "#C62828"
DARK  = "#FFFFFF"
PANEL = "#F5F5F5"

# ── Load & index ──────────────────────────────────────────────────────────────
data          = np.load(VECTORS_PATH, allow_pickle=True)
hybrid_matrix = data["vectors"].astype("float32")
N             = len(data["names"])

clip_part   = hybrid_matrix[:, :512]
defect_part = hybrid_matrix[:, 512:]

clip_norm   = clip_part   / (np.linalg.norm(clip_part,   axis=1, keepdims=True) + 1e-8)
defect_norm = defect_part / (np.linalg.norm(defect_part, axis=1, keepdims=True) + 1e-8)

weighted      = np.concatenate([clip_norm, defect_norm * DEFECT_WEIGHT], axis=1).astype("float32")
weighted_norm = weighted / (np.linalg.norm(weighted, axis=1, keepdims=True) + 1e-8)

clip_idx   = faiss.IndexFlatIP(512);  clip_idx.add(clip_norm)
hybrid_idx = faiss.IndexFlatIP(517);  hybrid_idx.add(weighted_norm)

# ── Evaluate ──────────────────────────────────────────────────────────────────
random.seed(SEED)
query_ids = random.sample(range(N), min(N_QUERIES, N))

clip_t1, clip_t5, hyb_t1, hyb_t5, delta = [], [], [], [], []

for q in query_ids:
    def hits(idx, vec, k=TOP_K):
        D, I = idx.search(vec.reshape(1, -1), k + 1)
        return [(float(d), int(i)) for d, i in zip(D[0], I[0]) if i >= 0 and i != q][:k]

    ch = hits(clip_idx,   clip_norm[q])
    hh = hits(hybrid_idx, weighted_norm[q])

    ct1 = ch[0][0] if ch else 0.0
    ct5 = float(np.mean([x[0] for x in ch])) if ch else 0.0
    ht1 = hh[0][0] if hh else 0.0
    ht5 = float(np.mean([x[0] for x in hh])) if hh else 0.0

    clip_t1.append(ct1);  clip_t5.append(ct5)
    hyb_t1.append(ht1);   hyb_t5.append(ht5)
    delta.append(ht1 - ct1)

clip_t1  = np.array(clip_t1);  clip_t5  = np.array(clip_t5)
hyb_t1   = np.array(hyb_t1);   hyb_t5   = np.array(hyb_t5)
delta    = np.array(delta)

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 5.5), facecolor=DARK)
gs  = gridspec.GridSpec(1, 2, figure=fig,
                        left=0.07, right=0.97, top=0.82, bottom=0.14,
                        wspace=0.30)

ax_bar  = fig.add_subplot(gs[0, 0])
ax_hist = fig.add_subplot(gs[0, 1])

for ax in (ax_bar, ax_hist):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors="#AAAAAA", labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor("#CCCCCC")
        sp.set_linewidth(0.8)

# ── LEFT: bar chart ───────────────────────────────────────────────────────────
labels  = ["Top-1", "Top-5 (medie)"]
cv      = [clip_t1.mean(), clip_t5.mean()]
hv      = [hyb_t1.mean(),  hyb_t5.mean()]
x       = np.array([0.0, 1.0])
w       = 0.30

bc = ax_bar.bar(x - w/2, cv, w, label="CLIP-only (512D)", color=GRAY,  alpha=0.75, zorder=3)
bh = ax_bar.bar(x + w/2, hv, w, label=f"Hibrid (517D, ×{DEFECT_WEIGHT:.0f} defect)", color=AMBER, alpha=0.90, zorder=3)

for bar, val, is_hyb in [(b, v, False) for b, v in zip(bc, cv)] + \
                         [(b, v, True)  for b, v in zip(bh, hv)]:
    col = "#8B6914" if is_hyb else "#333333"
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.009,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=col)

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(labels, fontsize=10, color="#222222")
ax_bar.set_ylim(0, 1.10)
ax_bar.set_ylabel("Cosine similarity", fontsize=10, color="#333333")
ax_bar.set_title("Calitate retrieval: CLIP vs. Hibrid", fontsize=11,
                 fontweight="bold", color="#111111", pad=8)
ax_bar.legend(fontsize=8.5, facecolor="white", edgecolor="#BBBBBB", labelcolor="#333333",
              framealpha=0.9)
ax_bar.yaxis.grid(True, color="#DDDDDD", linewidth=0.8, zorder=0)
ax_bar.set_axisbelow(True)

# delta annotations between bars
for xi, (dv) in enumerate([hyb_t1.mean()-clip_t1.mean(), hyb_t5.mean()-clip_t5.mean()]):
    ax_bar.annotate(f"+{dv:.3f}", xy=(xi, max(cv[xi], hv[xi]) + 0.055),
                    ha="center", fontsize=8, color=GREEN, fontweight="bold")

# ── RIGHT: improvement histogram ─────────────────────────────────────────────
pos_mask = delta > 0
neg_mask = delta < 0
eq_mask  = delta == 0

bins = np.linspace(delta.min() - 0.005, delta.max() + 0.005, 38)
ax_hist.hist(delta[pos_mask], bins=bins, color=GREEN, alpha=0.75,
             label=f"Hibrid mai bun ({pos_mask.sum()})", zorder=3)
ax_hist.hist(delta[neg_mask], bins=bins, color=RED,   alpha=0.75,
             label=f"CLIP mai bun ({neg_mask.sum()})", zorder=3)
ax_hist.hist(delta[eq_mask],  bins=bins, color=GRAY,  alpha=0.6,
             label=f"Egal ({eq_mask.sum()})", zorder=3)

ax_hist.axvline(x=0, color="#333333", linewidth=1.2, linestyle="--", alpha=0.6, zorder=4)
ax_hist.axvline(x=delta.mean(), color=AMBER, linewidth=2, linestyle="-", zorder=5,
                label=f"Medie: +{delta.mean():.4f}")

ax_hist.set_xlabel("Δ Top-1  (Hibrid − CLIP)", fontsize=10, color="#333333")
ax_hist.set_ylabel("Nr. query-uri", fontsize=10, color="#333333")
ax_hist.set_title("Distribuția îmbunătățirii per query", fontsize=11,
                  fontweight="bold", color="#111111", pad=8)
ax_hist.legend(fontsize=8.5, facecolor="white", edgecolor="#BBBBBB", labelcolor="#333333",
               framealpha=0.9)
ax_hist.yaxis.grid(True, color="#DDDDDD", linewidth=0.8, zorder=0)
ax_hist.set_axisbelow(True)

# ── Main title ────────────────────────────────────────────────────────────────
fig.suptitle(
    f"Evaluare Retrieval: CLIP-only (512D) vs. Vector Hibrid (517D)  |  "
    f"N = {N_QUERIES} query-uri, {N} imagini MIT FiveK",
    fontsize=12, fontweight="bold", color="#111111", y=0.97)

plt.savefig(OUT_PATH, dpi=180, facecolor=DARK, bbox_inches="tight")
plt.close()
print(f"Salvat: {OUT_PATH}")
