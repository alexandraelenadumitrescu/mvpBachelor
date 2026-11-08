"""
evaluate_correction.py
Evaluare pipeline corecție culoare: CLAHE + 3D LUT vs ground truth (edited/).
Metrici: Delta E 2000, PSNR, SSIM pe 100 query-uri aleatorii.
Outputs: evaluation_assets/correction/
"""

import os, sys, random, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import cv2
import faiss
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.color import deltaE_ciede2000, rgb2lab
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates

ROOT       = os.path.dirname(os.path.abspath(__file__))
SERVER     = os.path.join(ROOT, "server")
RAW_DIR    = os.path.join(SERVER, "images", "raw")
EDITED_DIR = os.path.join(SERVER, "images", "edited")
VECTORS    = os.path.join(SERVER, "hybrid_vectors.npz")
OUT_DIR    = os.path.join(ROOT, "evaluation_assets", "correction")
os.makedirs(OUT_DIR, exist_ok=True)

N_QUERIES     = 100
SEED          = 42
LUT_SIZE      = 17
LUT_STRENGTH  = 0.2
DEFECT_WEIGHT = 5.0

AMBER = "#C9A84C"
BLUE  = "#4A90D9"
GREEN = "#2E7D32"
GRAY  = "#888888"

# ── Load FAISS index ──────────────────────────────────────────────────────────
print("Incarcare vectori...")
data   = np.load(VECTORS, allow_pickle=True)
matrix = data["vectors"].astype("float32")   # (3499, 517)
names  = data["names"].tolist()
N      = len(names)

clip_n   = matrix[:, :512]
defect_n = matrix[:, 512:]
clip_n   /= (np.linalg.norm(clip_n,   axis=1, keepdims=True) + 1e-8)
defect_n /= (np.linalg.norm(defect_n, axis=1, keepdims=True) + 1e-8)
weighted  = np.concatenate([clip_n, defect_n * DEFECT_WEIGHT], axis=1).astype("float32")
weighted /= (np.linalg.norm(weighted, axis=1, keepdims=True) + 1e-8)

index = faiss.IndexFlatIP(517)
index.add(weighted)
print(f"  {N} vectori indexati.")

# ── Correction functions (identice cu server.py) ──────────────────────────────
def correct_clahe(img_f):
    lab = cv2.cvtColor((img_f * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

def extract_lut(raw_f, edited_f, lut_size=LUT_SIZE):
    r = raw_f.reshape(-1, 3)
    e = edited_f.reshape(-1, 3)
    steps = np.linspace(0, 1, lut_size)
    grid  = np.array(np.meshgrid(steps, steps, steps, indexing='ij')).reshape(3, -1).T
    lut_channels = []
    for c in range(3):
        interp = RegularGridInterpolator(
            (steps, steps, steps),
            e[:, c].reshape(lut_size, lut_size, lut_size) if False else
            np.zeros((lut_size, lut_size, lut_size)),   # placeholder
            method='linear', bounds_error=False, fill_value=None)
        lut_channels.append(interp(grid).reshape(lut_size, lut_size, lut_size))
    # simple approach: sample raw pixels into LUT bins
    lut = np.zeros((lut_size, lut_size, lut_size, 3), dtype=np.float32)
    idx = np.clip((r * (lut_size - 1)).astype(int), 0, lut_size - 1)
    np.add.at(lut, (idx[:,0], idx[:,1], idx[:,2]), e)
    count = np.zeros((lut_size, lut_size, lut_size, 1), dtype=np.float32)
    np.add.at(count, (idx[:,0], idx[:,1], idx[:,2]), 1)
    mask = count[..., 0] > 0
    lut[mask] /= count[mask]
    # fill empty bins with identity
    empty = ~mask
    for c in range(3):
        g = np.linspace(0, 1, lut_size)
        gx, gy, gz = np.meshgrid(g, g, g, indexing='ij')
        identity = [gx, gy, gz][c]
        lut[empty, c] = identity[empty]
    return lut

def apply_lut(img_f, lut):
    h, w, _ = img_f.shape
    flat = img_f.reshape(-1, 3)
    coords = flat * (LUT_SIZE - 1)
    result = np.stack([
        map_coordinates(lut[:,:,:,c], coords.T, order=1, mode='nearest')
        for c in range(3)], axis=1)
    return np.clip(result.reshape(h, w, 3), 0, 1).astype(np.float32)

def apply_lut_moderated(img_f, lut):
    return img_f * (1 - LUT_STRENGTH) + apply_lut(img_f, lut) * LUT_STRENGTH

def load_resize(path, size=512):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    return np.array(img).astype(np.float32) / 255.0

def compute_deltae(img1_f, img2_f):
    lab1 = rgb2lab(img1_f)
    lab2 = rgb2lab(img2_f)
    return float(deltaE_ciede2000(lab1, lab2).mean())

# ── Evaluate ──────────────────────────────────────────────────────────────────
random.seed(SEED)
query_idx = random.sample(range(N), N_QUERIES)

results = []   # (name, de_before, de_after, psnr_before, psnr_after, ssim_before, ssim_after)

print(f"\nEvaluare pe {N_QUERIES} query-uri...")
for pos, qi in enumerate(query_idx):
    if (pos + 1) % 20 == 0:
        print(f"  {pos+1}/{N_QUERIES}")

    name = names[qi]
    fname = name if name.endswith(".jpg") else name + ".jpg"
    raw_path    = os.path.join(RAW_DIR,    fname)
    edited_path = os.path.join(EDITED_DIR, fname)
    if not os.path.exists(raw_path) or not os.path.exists(edited_path):
        continue

    raw_f    = load_resize(raw_path)
    edited_f = load_resize(edited_path)

    # FAISS: nearest neighbor excluding self
    D, I = index.search(weighted[qi].reshape(1, -1), 2)
    neighbor_idx = int(I[0][1]) if int(I[0][0]) == qi else int(I[0][0])
    neighbor_name = names[neighbor_idx]
    neighbor_fname = neighbor_name if neighbor_name.endswith(".jpg") else neighbor_name + ".jpg"

    ref_raw_path    = os.path.join(RAW_DIR,    neighbor_fname)
    ref_edited_path = os.path.join(EDITED_DIR, neighbor_fname)
    if not os.path.exists(ref_raw_path) or not os.path.exists(ref_edited_path):
        lut = None
    else:
        ref_raw_f    = load_resize(ref_raw_path)
        ref_edited_f = load_resize(ref_edited_path)
        lut = extract_lut(ref_raw_f, ref_edited_f)

    # apply correction
    clahe_f = correct_clahe(raw_f)
    corrected_f = apply_lut_moderated(clahe_f, lut) if lut is not None else clahe_f

    # metrics
    raw_u8  = (raw_f * 255).astype(np.uint8)
    cor_u8  = (corrected_f * 255).astype(np.uint8)
    edi_u8  = (edited_f * 255).astype(np.uint8)

    de_before  = compute_deltae(raw_f, edited_f)
    de_after   = compute_deltae(corrected_f, edited_f)
    ps_before  = psnr(edi_u8, raw_u8, data_range=255)
    ps_after   = psnr(edi_u8, cor_u8, data_range=255)
    ss_before  = ssim(edi_u8, raw_u8, channel_axis=2, data_range=255)
    ss_after   = ssim(edi_u8, cor_u8, channel_axis=2, data_range=255)

    results.append((name, de_before, de_after, ps_before, ps_after, ss_before, ss_after))

de_b  = np.array([r[1] for r in results])
de_a  = np.array([r[2] for r in results])
ps_b  = np.array([r[3] for r in results])
ps_a  = np.array([r[4] for r in results])
ss_b  = np.array([r[5] for r in results])
ss_a  = np.array([r[6] for r in results])

print(f"\n{'Metrică':<22} {'Inainte':>10} {'Dupa':>10} {'Delta':>10}")
print("-" * 52)
print(f"{'Delta E 2000 (↓)':<22} {de_b.mean():>10.4f} {de_a.mean():>10.4f} {de_a.mean()-de_b.mean():>+10.4f}")
print(f"{'PSNR dB (↑)':<22} {ps_b.mean():>10.4f} {ps_a.mean():>10.4f} {ps_a.mean()-ps_b.mean():>+10.4f}")
print(f"{'SSIM (↑)':<22} {ss_b.mean():>10.4f} {ss_a.mean():>10.4f} {ss_a.mean()-ss_b.mean():>+10.4f}")
pct_de = 100 * (de_a < de_b).sum() / len(results)
pct_ps = 100 * (ps_a > ps_b).sum() / len(results)
pct_ss = 100 * (ss_a > ss_b).sum() / len(results)
print(f"\nImbunatatire Delta E pe {pct_de:.1f}% din query-uri")
print(f"Imbunatatire PSNR pe    {pct_ps:.1f}% din query-uri")
print(f"Imbunatatire SSIM pe    {pct_ss:.1f}% din query-uri")

# ── Save summary ──────────────────────────────────────────────────────────────
with open(os.path.join(OUT_DIR, "correction_summary.txt"), "w", encoding="utf-8") as f:
    f.write(f"Pipeline corecție: CLAHE (clipLimit=1.5, tile=4×4) + 3D LUT (size={LUT_SIZE}, strength={LUT_STRENGTH})\n")
    f.write(f"N query-uri: {len(results)} (din {N} imagini MIT FiveK)\n\n")
    f.write(f"Delta E 2000:  {de_b.mean():.4f} -> {de_a.mean():.4f}  ({de_a.mean()-de_b.mean():+.4f})\n")
    f.write(f"PSNR (dB):     {ps_b.mean():.4f} -> {ps_a.mean():.4f}  ({ps_a.mean()-ps_b.mean():+.4f})\n")
    f.write(f"SSIM:          {ss_b.mean():.4f} -> {ss_a.mean():.4f}  ({ss_a.mean()-ss_b.mean():+.4f})\n")
    f.write(f"Imbunatatire Delta E: {pct_de:.1f}% | PSNR: {pct_ps:.1f}% | SSIM: {pct_ss:.1f}%\n")

# ── Figure 1: Bar chart metrici ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 5), facecolor="white")
fig.subplots_adjust(left=0.07, right=0.97, top=0.85, bottom=0.12, wspace=0.35)

metrics = [
    ("Delta E 2000 ↓", de_b.mean(), de_a.mean(), True),
    ("PSNR (dB) ↑",    ps_b.mean(), ps_a.mean(), False),
    ("SSIM ↑",         ss_b.mean(), ss_a.mean(), False),
]
for ax, (label, vb, va, lower_better) in zip(axes, metrics):
    improved = (va < vb) if lower_better else (va > vb)
    color_after = GREEN if improved else "#E53935"
    bars = ax.bar(["Înainte\n(raw)", "După\n(corecție)"], [vb, va],
                  color=[GRAY, color_after], width=0.5, alpha=0.85)
    for bar, val in zip(bars, [vb, va]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(vb,va)*0.02),
                f"{val:.4f}", ha="center", fontsize=10, fontweight="bold", color="#222222")
    ax.set_title(label, fontsize=11, fontweight="bold", color="#111111")
    ax.set_facecolor("#F9F9F9")
    ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ax.spines.values(): sp.set_edgecolor("#CCCCCC")

fig.suptitle(f"Evaluare Pipeline Corecție Culoare — CLAHE + 3D LUT  (N={len(results)} imagini MIT FiveK)",
             fontsize=12, fontweight="bold", color="#111111", y=0.97)
plt.savefig(os.path.join(OUT_DIR, "correction_metrics_bar.png"), dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print("\nSalvat: correction_metrics_bar.png")

# ── Figure 2: Distributie Delta E per query ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
ax.set_facecolor("#F9F9F9")
delta_de = de_a - de_b
pos_mask = delta_de < 0   # imbunatatire = delta E scade
neg_mask = delta_de >= 0
bins = np.linspace(delta_de.min() - 0.5, delta_de.max() + 0.5, 40)
ax.hist(delta_de[pos_mask], bins=bins, color=GREEN, alpha=0.8, label=f"Îmbunătățit ({pos_mask.sum()})")
ax.hist(delta_de[neg_mask], bins=bins, color="#E53935", alpha=0.8, label=f"Degradat ({neg_mask.sum()})")
ax.axvline(x=0, color="#333333", linewidth=1.5, linestyle="--", alpha=0.6)
ax.axvline(x=delta_de.mean(), color=AMBER, linewidth=2,
           label=f"Medie: {delta_de.mean():+.4f}")
ax.set_xlabel("Δ Delta E 2000  (corecție − original, negativ = mai bine)", fontsize=11, color="#333333")
ax.set_ylabel("Nr. imagini", fontsize=11, color="#333333")
ax.set_title("Distribuția variației Delta E per query", fontsize=12, fontweight="bold", color="#111111")
ax.legend(fontsize=10)
ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.8)
ax.set_axisbelow(True)
for sp in ax.spines.values(): sp.set_edgecolor("#CCCCCC")
plt.savefig(os.path.join(OUT_DIR, "correction_deltae_dist.png"), dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print("Salvat: correction_deltae_dist.png")

# ── Figure 3: Before/After vizual (top 6 imbunatatiri Delta E) ────────────────
improved_idx = np.argsort(delta_de)[:6]   # cel mai mult imbunatatite
fig, axes = plt.subplots(6, 3, figsize=(13, 22), facecolor="white")
fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.01, hspace=0.12, wspace=0.04)

for row, ri in enumerate(improved_idx):
    name, de_b_val, de_a_val, ps_b_val, ps_a_val, ss_b_val, ss_a_val = results[ri]
    fn = name if name.endswith(".jpg") else name + ".jpg"
    raw_f    = load_resize(os.path.join(RAW_DIR, fn))
    edited_f = load_resize(os.path.join(EDITED_DIR, fn))

    # recompute corrected for display
    clahe_f = correct_clahe(raw_f)
    neighbor_qi = query_idx[results.index(results[ri])]
    D2, I2 = index.search(weighted[neighbor_qi].reshape(1,-1), 2)
    nb_idx = int(I2[0][1]) if int(I2[0][0]) == neighbor_qi else int(I2[0][0])
    nb_name = names[nb_idx]
    nb_fname = nb_name if nb_name.endswith(".jpg") else nb_name + ".jpg"
    ref_r = load_resize(os.path.join(RAW_DIR, nb_fname))
    ref_e = load_resize(os.path.join(EDITED_DIR, nb_fname))
    lut2 = extract_lut(ref_r, ref_e)
    cor_f = apply_lut_moderated(clahe_f, lut2)

    panels = [
        (raw_f,    "Original (raw)",   "#888888"),
        (cor_f,    "Corecție CLAHE+LUT", AMBER),
        (edited_f, "Ground truth (edited)", BLUE),
    ]
    for col, (img, title, tc) in enumerate(panels):
        axes[row, col].imshow(np.clip(img, 0, 1))
        axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
        for sp in axes[row, col].spines.values(): sp.set_visible(False)
        if row == 0:
            axes[row, col].set_title(title, fontsize=10, fontweight="bold",
                                     color=tc, pad=5)
    axes[row, 0].set_ylabel(
        f"ΔE: {de_b_val:.2f}→{de_a_val:.2f}\nPSNR: {ps_b_val:.1f}→{ps_a_val:.1f}",
        fontsize=8, color="#444444", rotation=0, labelpad=65, va="center")

fig.suptitle("Top 6 îmbunătățiri — Pipeline Corecție Culoare (raw | corecție | ground truth)",
             fontsize=12, fontweight="bold", color="#111111", y=0.97)
plt.savefig(os.path.join(OUT_DIR, "correction_before_after.png"), dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("Salvat: correction_before_after.png")
print(f"\nGata. Toate outputurile in: {OUT_DIR}")
