"""
generate_underexposure_xai.py — Vizualizare explicabilitate cap detectie defecte:
Original vs Subexpus + Grad-CAM (underexposure, idx=3)

Output: evaluation_assets/xai_underexposure_explanation.png
"""

import os, sys, warnings, glob
warnings.filterwarnings("ignore")
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import torch
import torch.nn as nn
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

ROOT       = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.join(ROOT, "server")
MODEL_PATH = os.path.join(SERVER_DIR, "defect_head.pt")
RAW_DIR    = os.path.join(SERVER_DIR, "images", "raw")
OUT_PATH   = os.path.join(ROOT, "evaluation_assets", "xai_underexposure_explanation.png")

DEFECT_NAMES   = ["Blur", "Noise", "Overexposure", "Underexposure", "Compression"]
DEFECT_COLORS  = ["#61AFEF", "#98C379", "#E06C75", "#C678DD", "#E5C07B"]
AMBER          = "#C9A84C"
NAVY           = "#1F3564"
DARK_BG        = "#12141A"
PANEL_BG       = "#1E2028"
UNDEREXP_IDX   = 3
UNDEREXP_FACTOR = 0.07   # multiply → very dark, max underexposure signal pe imaginea sursa

device = "cpu"

# ── Load model ──────────────────────────────────────────────────────────────
backbone = models.mobilenet_v3_small(weights=None)
in_f = backbone.classifier[0].in_features
backbone.classifier = nn.Sequential(
    nn.Linear(in_f, 128), nn.Hardswish(), nn.Dropout(0.2),
    nn.Linear(128, 5), nn.Sigmoid(),
)
backbone.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
backbone = backbone.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class DefectTarget:
    def __init__(self, idx): self.idx = idx
    def __call__(self, out):
        return out[:, self.idx] if out.dim() == 2 else out[self.idx]

def get_scores(pil_img):
    t = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        return backbone(t)[0].numpy()

def get_heatmap(pil_img, idx):
    t = transform(pil_img).unsqueeze(0)
    with GradCAM(model=backbone, target_layers=[backbone.features[-1]]) as cam:
        gray = cam(input_tensor=t, targets=[DefectTarget(idx)])[0]
    return gray  # (224,224) float32 [0,1]

# ── Load source image ──────────────────────────────────────────────────────
candidates = sorted(glob.glob(os.path.join(RAW_DIR, "*.jpg")))
# Prefer a portrait/person photo for more visual interest
preferred = [p for p in candidates if "ke_-2590" in os.path.basename(p).lower()]
src_path = preferred[0] if preferred else candidates[0]
print(f"Sursa: {os.path.basename(src_path)}")

pil_orig = Image.open(src_path).convert("RGB").resize((224, 224), Image.LANCZOS)
orig_np  = np.array(pil_orig)          # uint8 RGB (224,224,3)
orig_f   = orig_np.astype(np.float32) / 255.0

# ── Create underexposed version ────────────────────────────────────────────
under_np  = np.clip(orig_f * UNDEREXP_FACTOR * 255, 0, 255).astype(np.uint8)
pil_under = Image.fromarray(under_np)
under_f   = under_np.astype(np.float32) / 255.0

# ── Get scores ─────────────────────────────────────────────────────────────
scores_orig  = get_scores(pil_orig)
scores_under = get_scores(pil_under)
print(f"Scores original:   {dict(zip(DEFECT_NAMES, [f'{s:.3f}' for s in scores_orig]))}")
print(f"Scores subexpus:   {dict(zip(DEFECT_NAMES, [f'{s:.3f}' for s in scores_under]))}")

# ── Grad-CAM on underexposed ───────────────────────────────────────────────
print("Calculez Grad-CAM...")
heatmap_under = get_heatmap(pil_under, UNDEREXP_IDX)

# thermal overlay
thermal_bgr = show_cam_on_image(under_f, heatmap_under, use_rgb=True)

# spotlight vignette (underexposed base + heatmap)
darkened   = under_f * 0.2
mask       = heatmap_under[:, :, np.newaxis]
vignette_f = darkened + (under_f - darkened) * mask
vignette_np = np.clip(vignette_f * 255, 0, 255).astype(np.uint8)

# ── Figure ─────────────────────────────────────────────────────────────────
# Layout: 2 rows x 3 cols
# Row 0: [Original | Subexpus | Grad-CAM heatmap]
# Row 1: [Score bars orig | Score bars subexpus | Spotlight vignette]

fig = plt.figure(figsize=(16, 10), facecolor=DARK_BG)
gs  = gridspec.GridSpec(
    2, 3,
    figure=fig,
    hspace=0.06, wspace=0.05,
    top=0.88, bottom=0.04,
    left=0.03, right=0.97,
    height_ratios=[2.2, 1.0],
)

ax_orig    = fig.add_subplot(gs[0, 0])
ax_under   = fig.add_subplot(gs[0, 1])
ax_thermal = fig.add_subplot(gs[0, 2])
ax_bar_o   = fig.add_subplot(gs[1, 0])
ax_bar_u   = fig.add_subplot(gs[1, 1])
ax_vign    = fig.add_subplot(gs[1, 2])

def style_img_ax(ax, img_rgb, title, title_color="#FFFFFF", border_color=None):
    ax.imshow(img_rgb)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11, fontweight="bold",
                 color=title_color, pad=6, loc="center")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(border_color or "#333333")
        spine.set_linewidth(1.5)
    ax.set_facecolor(PANEL_BG)

style_img_ax(ax_orig,    orig_np,     "Original",
             title_color="#AAAAAA", border_color="#444444")
style_img_ax(ax_under,   under_np,    "Subexpus (factor ×0.18)",
             title_color=DEFECT_COLORS[3], border_color=DEFECT_COLORS[3])
style_img_ax(ax_thermal, thermal_bgr, "Grad-CAM — Underexposure (idx=3)",
             title_color=AMBER, border_color=AMBER)

# ── Score bar charts ────────────────────────────────────────────────────────
def draw_score_bars(ax, scores, title, highlight_idx=UNDEREXP_IDX):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333"); spine.set_linewidth(1)

    y_pos = np.arange(len(DEFECT_NAMES))
    colors_bar = [DEFECT_COLORS[i] if i != highlight_idx else DEFECT_COLORS[highlight_idx]
                  for i in range(len(DEFECT_NAMES))]
    alphas = [1.0 if i == highlight_idx else 0.45 for i in range(len(DEFECT_NAMES))]

    bars = ax.barh(y_pos, scores, color=colors_bar, alpha=1.0,
                   height=0.6, edgecolor="none")
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)

    # highlight underexposure bar with bright border
    bars[highlight_idx].set_edgecolor(DEFECT_COLORS[highlight_idx])
    bars[highlight_idx].set_linewidth(2)
    bars[highlight_idx].set_alpha(1.0)

    for i, (bar, score) in enumerate(zip(bars, scores)):
        color = "#FFFFFF" if i == highlight_idx else "#888888"
        weight = "bold" if i == highlight_idx else "normal"
        ax.text(min(score + 0.03, 0.95), bar.get_y() + bar.get_height()/2,
                f"{score:.2f}", va="center", ha="left",
                fontsize=9, color=color, fontweight=weight)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(DEFECT_NAMES, fontsize=9)
    for i, lbl in enumerate(ax.get_yticklabels()):
        lbl.set_color(DEFECT_COLORS[i] if i == highlight_idx else "#888888")
        lbl.set_fontweight("bold" if i == highlight_idx else "normal")
    ax.set_xlim(0, 1.15)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(colors="#666666", labelsize=8)
    ax.axvline(x=scores[highlight_idx], color=DEFECT_COLORS[highlight_idx],
               linewidth=1.2, linestyle="--", alpha=0.5)
    ax.set_title(title, fontsize=9.5, color="#CCCCCC", pad=4)
    ax.xaxis.grid(True, color="#2A2A2A", linewidth=0.8)
    ax.set_axisbelow(True)

draw_score_bars(ax_bar_o, scores_orig,  "Scoruri model — Original")
draw_score_bars(ax_bar_u, scores_under, "Scoruri model — Subexpus")

# ── Spotlight vignette ──────────────────────────────────────────────────────
style_img_ax(ax_vign, vignette_np, "Spotlight Vignette — Grad-CAM",
             title_color="#888888", border_color=AMBER)

# ── Main title ──────────────────────────────────────────────────────────────
fig.suptitle(
    "Explicabilitate Cap Detectie Defecte — MobileNetV3Small  |  Defect: Subexpunere (Underexposure)",
    fontsize=13, fontweight="bold", color="#FFFFFF",
    y=0.95,
)

# Subtitle with scores delta
delta = scores_under[UNDEREXP_IDX] - scores_orig[UNDEREXP_IDX]
fig.text(0.5, 0.905,
         f"Scor Underexposure:  Original = {scores_orig[UNDEREXP_IDX]:.3f}  →  "
         f"Subexpus = {scores_under[UNDEREXP_IDX]:.3f}  "
         f"(+{delta:.3f})",
         ha="center", fontsize=11, color=DEFECT_COLORS[UNDEREXP_IDX], style="italic")

# Arrow annotation between score bars
fig.text(0.50, 0.34, "→", ha="center", va="center",
         fontsize=22, color=DEFECT_COLORS[UNDEREXP_IDX], fontweight="bold")

plt.savefig(OUT_PATH, dpi=180, facecolor=DARK_BG, bbox_inches="tight")
plt.close()
print(f"\nSalvat: {OUT_PATH}")
