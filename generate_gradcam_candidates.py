"""
generate_gradcam_candidates.py
Grad-CAM pe imaginile originale din defect_photos/ (fara distorsii).
6 poze x 5 defecte = 30 candidati, 2 panouri: original | overlay.
Output: evaluation_assets/xai_candidates/
"""

import os, sys, glob, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "server", "defect_head.pt")
SRC_DIR    = os.path.join(ROOT, "evaluation_assets", "defect_photos")
OUT_DIR    = os.path.join(ROOT, "evaluation_assets", "xai_candidates")
os.makedirs(OUT_DIR, exist_ok=True)

DEFECTS = [
    ("blur",          0, "#4A90D9"),
    ("noise",         1, "#7CB342"),
    ("overexposure",  2, "#E53935"),
    ("underexposure", 3, "#FB8C00"),
    ("compression",   4, "#8E24AA"),
]

# ── Model ─────────────────────────────────────────────────────────────────────
backbone = models.mobilenet_v3_small(weights=None)
in_f = backbone.classifier[0].in_features
backbone.classifier = nn.Sequential(
    nn.Linear(in_f, 128), nn.Hardswish(), nn.Dropout(0.2),
    nn.Linear(128, 5), nn.Sigmoid(),
)
backbone.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
backbone.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class DefectTarget:
    def __init__(self, idx): self.idx = idx
    def __call__(self, out): return out[:, self.idx] if out.dim() == 2 else out[self.idx]

def infer(t):
    with torch.no_grad():
        return backbone(t)[0].numpy()

def gradcam_heatmap(t, idx):
    with GradCAM(model=backbone, target_layers=[backbone.features[-1]]) as cam:
        return cam(input_tensor=t, targets=[DefectTarget(idx)])[0]

# ── Generate ──────────────────────────────────────────────────────────────────
paths = sorted(glob.glob(os.path.join(SRC_DIR, "*.jpg")) +
               glob.glob(os.path.join(SRC_DIR, "*.png")))

if not paths:
    print("EROARE: nicio imagine in", SRC_DIR); sys.exit(1)

for img_idx, path in enumerate(paths, start=1):
    fname    = os.path.splitext(os.path.basename(path))[0][:30]
    pil_full = Image.open(path).convert("RGB")          # rezolutie originala
    pil_224  = pil_full.resize((224, 224), Image.LANCZOS)  # doar pentru model

    arr_full = np.array(pil_full)
    arr_f224 = np.array(pil_224).astype(np.float32) / 255.0
    t        = transform(pil_224).unsqueeze(0)
    scores   = infer(t)

    W, H = pil_full.size

    for def_name, def_idx, color in DEFECTS:
        heatmap_224 = gradcam_heatmap(t, def_idx)           # (224,224)
        # upsample heatmap la rezolutia originala
        import cv2
        heatmap_full = cv2.resize(heatmap_224, (W, H), interpolation=cv2.INTER_LINEAR)
        arr_full_f   = arr_full.astype(np.float32) / 255.0
        overlay      = show_cam_on_image(arr_full_f, heatmap_full, use_rgb=True)
        score   = scores[def_idx]

        fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor="white")
        fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01, wspace=0.03)

        axes[0].imshow(arr_full)
        axes[0].set_title("Original", fontsize=10, color="#333333", pad=4)

        axes[1].imshow(overlay)
        axes[1].set_title("Grad-CAM", fontsize=10, color=color, fontweight="bold", pad=4)

        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)

        fig.suptitle(f"{def_name}  |  scor: {score:.3f}",
                     fontsize=11, fontweight="bold", color=color, y=0.98)

        out_path = os.path.join(OUT_DIR, f"{img_idx:02d}_{def_name}.png")
        plt.savefig(out_path, dpi=200, facecolor="white", bbox_inches="tight")
        plt.close()
        print(f"  {os.path.basename(out_path)}  score={score:.3f}")

print(f"\nGata. {len(paths) * len(DEFECTS)} imagini in: {OUT_DIR}")
