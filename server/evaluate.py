"""
evaluate.py — CLIP-only vs Hybrid retrieval quality comparison
Outputs: evaluation_results.csv, evaluation_chart.png, evaluation_summary.txt
"""

import os
import random
import csv

import numpy as np
import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Constants (mirrored from server.py) ──────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
VECTORS_PATH  = os.path.join(BASE_DIR, "hybrid_vectors.npz")
CLIP_WEIGHT   = 1.0
DEFECT_WEIGHT = 5.0

N_QUERIES = 200
TOP_K     = 5
SEED      = 42

# ── Output paths ─────────────────────────────────────────────────────────────
CSV_OUT     = os.path.join(BASE_DIR, "evaluation_results.csv")
CHART_OUT   = os.path.join(BASE_DIR, "evaluation_chart.png")
SUMMARY_OUT = os.path.join(BASE_DIR, "evaluation_summary.txt")


# ── Load vectors ─────────────────────────────────────────────────────────────
print("Loading hybrid vectors...")
data          = np.load(VECTORS_PATH, allow_pickle=True)
hybrid_matrix = data["vectors"].astype("float32")   # (N, 517)
image_names   = data["names"].tolist()
N             = len(image_names)
print(f"  {N} vectors, dim={hybrid_matrix.shape[1]}")


# ── Build CLIP-only index (512 dims) ─────────────────────────────────────────
print("Building CLIP-only index...")
clip_part  = hybrid_matrix[:, :512]
clip_norm  = clip_part / (np.linalg.norm(clip_part, axis=1, keepdims=True) + 1e-8)
clip_index = faiss.IndexFlatIP(512)
clip_index.add(clip_norm)


# ── Build Hybrid index (517 dims, same pipeline as server.py) ─────────────────
print("Building Hybrid index...")
defect_part   = hybrid_matrix[:, 512:]
defect_norm   = defect_part / (np.linalg.norm(defect_part, axis=1, keepdims=True) + 1e-8)
weighted      = np.concatenate(
    [clip_norm * CLIP_WEIGHT, defect_norm * DEFECT_WEIGHT], axis=1
).astype("float32")
weighted_norm = weighted / (np.linalg.norm(weighted, axis=1, keepdims=True) + 1e-8)
hybrid_index  = faiss.IndexFlatIP(517)
hybrid_index.add(weighted_norm)


# ── Sample query indices ──────────────────────────────────────────────────────
random.seed(SEED)
query_indices = random.sample(range(N), min(N_QUERIES, N))
print(f"Evaluating {len(query_indices)} queries (TOP_K={TOP_K})...")


# ── Helper: top-k hits excluding self ────────────────────────────────────────
def top_k_excluding_self(index, query_vec, q_idx, k):
    D, I = index.search(query_vec.reshape(1, -1), k + 1)
    hits = [(float(d), int(i)) for d, i in zip(D[0], I[0]) if i >= 0 and i != q_idx]
    return hits[:k]


# ── Evaluate ──────────────────────────────────────────────────────────────────
rows              = []
clip_top1_scores  = []
clip_top5_scores  = []
hybrid_top1_scores = []
hybrid_top5_scores = []
hybrid_wins_list  = []

for pos, q_idx in enumerate(query_indices):
    if (pos + 1) % 50 == 0:
        print(f"  {pos + 1}/{len(query_indices)}")

    # CLIP-only
    clip_hits = top_k_excluding_self(clip_index, clip_norm[q_idx], q_idx, TOP_K)
    c_top1      = clip_hits[0][0]  if clip_hits else 0.0
    c_top5_mean = float(np.mean([h[0] for h in clip_hits])) if clip_hits else 0.0

    # Hybrid
    hybrid_hits  = top_k_excluding_self(hybrid_index, weighted_norm[q_idx], q_idx, TOP_K)
    h_top1       = hybrid_hits[0][0]  if hybrid_hits else 0.0
    h_top5_mean  = float(np.mean([h[0] for h in hybrid_hits])) if hybrid_hits else 0.0

    wins = 1 if h_top1 > c_top1 else 0

    clip_top1_scores.append(c_top1)
    clip_top5_scores.append(c_top5_mean)
    hybrid_top1_scores.append(h_top1)
    hybrid_top5_scores.append(h_top5_mean)
    hybrid_wins_list.append(wins)

    rows.append({
        "query_name":     image_names[q_idx],
        "clip_top1":      round(c_top1, 6),
        "clip_top5_mean": round(c_top5_mean, 6),
        "hybrid_top1":    round(h_top1, 6),
        "hybrid_top5_mean": round(h_top5_mean, 6),
        "hybrid_wins":    wins,
    })


# ── Aggregate ─────────────────────────────────────────────────────────────────
mean_clip_top1   = float(np.mean(clip_top1_scores))
mean_clip_top5   = float(np.mean(clip_top5_scores))
mean_hybrid_top1 = float(np.mean(hybrid_top1_scores))
mean_hybrid_top5 = float(np.mean(hybrid_top5_scores))
pct_wins         = 100.0 * sum(hybrid_wins_list) / len(query_indices)
delta_top1       = mean_hybrid_top1 - mean_clip_top1
delta_top5       = mean_hybrid_top5 - mean_clip_top5


# ── Export CSV ────────────────────────────────────────────────────────────────
print(f"Writing {CSV_OUT}")
fieldnames = ["query_name", "clip_top1", "clip_top5_mean",
              "hybrid_top1", "hybrid_top5_mean", "hybrid_wins"]
with open(CSV_OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)


# ── Export chart ──────────────────────────────────────────────────────────────
print(f"Writing {CHART_OUT}")
BG       = "#0A0A0A"
GRAY     = "#888888"
AMBER    = "#C9A84C"
WHITE    = "#FFFFFF"

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

x        = np.array([0.0, 1.0])   # two metric groups
width    = 0.35
clip_vals   = [mean_clip_top1,   mean_clip_top5]
hybrid_vals = [mean_hybrid_top1, mean_hybrid_top5]

bars_clip   = ax.bar(x - width / 2, clip_vals,   width, label="CLIP-only", color=GRAY)
bars_hybrid = ax.bar(x + width / 2, hybrid_vals, width, label="Hybrid",    color=AMBER)

# Value labels on top of each bar
for bar in bars_clip:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.2f}",
            ha="center", va="bottom", color=WHITE, fontsize=9)

for bar in bars_hybrid:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.2f}",
            ha="center", va="bottom", color=WHITE, fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(["Top-1 similarity", "Top-5 mean similarity"], color=WHITE, fontsize=11)
ax.set_ylim(0.0, 1.0)
ax.set_ylabel("Cosine similarity", color=WHITE, fontsize=11)
ax.set_title("Retrieval quality: CLIP-only vs Hybrid vector", color=WHITE, fontsize=13, pad=12)

ax.tick_params(colors=WHITE)
for spine in ax.spines.values():
    spine.set_edgecolor("#333333")

ax.yaxis.grid(True, color="#444444", alpha=0.3, linestyle="--")
ax.set_axisbelow(True)

legend = ax.legend(framealpha=0, labelcolor=WHITE, fontsize=10)

plt.tight_layout()
plt.savefig(CHART_OUT, dpi=300, facecolor=BG)
plt.close()


# ── Export summary ────────────────────────────────────────────────────────────
print(f"Writing {SUMMARY_OUT}")
summary = (
    f"CLIP-only  - Top-1: {mean_clip_top1:.3f}  |  Top-5 mean: {mean_clip_top5:.3f}\n"
    f"Hybrid     - Top-1: {mean_hybrid_top1:.3f}  |  Top-5 mean: {mean_hybrid_top5:.3f}\n"
    f"Hybrid outperforms CLIP-only on top-1 in {pct_wins:.1f}% of queries (N={len(query_indices)})\n"
    f"Delta top-1: {delta_top1:+.3f}  |  Delta top-5: {delta_top5:+.3f}\n"
)
with open(SUMMARY_OUT, "w") as f:
    f.write(summary)

print("\n-- Summary --------------------------------------------------")
print(summary, end="")
print("-------------------------------------------------------------")
print("Done.")
