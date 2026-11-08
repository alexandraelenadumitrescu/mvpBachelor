"""
generate_badge_eigencam.py
EigenCAM pe badge detector (YOLOv8 fine-tuned: Cards, Lanyard).
Sursa: evaluation_assets/defect_photos/  (6 imagini)
Output: evaluation_assets/badge_eigencam/  (6 imagini)
Panouri: Original + detectii | EigenCAM overlay
"""

import os, sys, glob, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import cv2
from PIL import Image
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from ultralytics import YOLO
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

ROOT       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "server", "blur_api", "badge_model.pt")
SRC_DIR    = os.path.join(ROOT, "server", "images", "raw")
OUT_DIR    = os.path.join(ROOT, "evaluation_assets", "badge_eigencam")
os.makedirs(OUT_DIR, exist_ok=True)

CONF       = 0.20
CLASS_COLORS = {"Cards": "#4A90D9", "Lanyard": "#FB8C00"}
IMGSZ      = 640

# ── Load model ────────────────────────────────────────────────────────────────
model = YOLO(MODEL_PATH)
names = model.names  # {0: 'Cards', 1: 'Lanyard'}

# Wrapper: YOLO returneaza tuple, EigenCAM asteapta tensor
class YOLOWrapper(torch.nn.Module):
    def __init__(self, inner): super().__init__(); self.inner = inner
    def forward(self, x):
        out = self.inner(x)
        return out[0] if isinstance(out, (tuple, list)) else out

wrapped      = YOLOWrapper(model.model)
target_layer = model.model.model[-2]   # C2f, ultimul layer inainte de Detect
cam = EigenCAM(wrapped, target_layers=[target_layer])

# ── Transform helper (YOLOv8 expects normalized float tensor) ─────────────────
def to_tensor(img_rgb_np, size=IMGSZ):
    resized = cv2.resize(img_rgb_np, (size, size))
    t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0), resized  # (1,3,H,W), resized_np

# ── Process each image ────────────────────────────────────────────────────────
HITS = {
    "a0006-IMG_2787.jpg", "a0009-kme_372.jpg", "a0011-DSC_0082.jpg",
    "a0017-050710_031618__MG_3496.jpg", "a0019-jmac_MG_0653.jpg",
    "a0020-jmac_MG_6225.jpg", "a0023-07-06-02-at-15h06m48-s_MG_1489.jpg",
    "a0024-_DSC8932.jpg", "a0031-WP_CRW_0736.jpg", "a0033-KE_-2590.jpg",
    "a0038-MB_070908_135.jpg", "a0041-IMG_4972.jpg",
}
paths = [p for p in sorted(glob.glob(os.path.join(SRC_DIR, "*.jpg")))
         if os.path.basename(p) in HITS]

for img_idx, path in enumerate(paths, start=1):
    fname    = os.path.basename(path)
    pil_full = Image.open(path).convert("RGB")
    arr_full = np.array(pil_full)          # original resolution, RGB
    H_orig, W_orig = arr_full.shape[:2]

    # tensor for EigenCAM (640x640)
    tensor, arr_640 = to_tensor(arr_full)
    arr_640_f = arr_640.astype(np.float32) / 255.0

    # EigenCAM heatmap at 640x640, upsample to original
    heatmap_640  = cam(input_tensor=tensor)[0]   # (640,640)
    heatmap_full = cv2.resize(heatmap_640, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)

    overlay = show_cam_on_image(arr_full.astype(np.float32) / 255.0,
                                heatmap_full, use_rgb=True)

    # YOLO detections on original
    results  = model.predict(source=path, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
    boxes    = results.boxes

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor="white")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.01, wspace=0.03)

    # Panel 1: original + bounding boxes
    axes[0].imshow(arr_full)
    axes[0].set_title("Original + detecții", fontsize=11, color="#222222", pad=5)

    det_summary = []
    if boxes is not None and len(boxes):
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls   = int(box.cls[0])
            conf  = float(box.conf[0])
            label = names[cls]
            color = CLASS_COLORS.get(label, "#999999")
            rect  = Rectangle((x1, y1), x2 - x1, y2 - y1,
                               linewidth=2.5, edgecolor=color, facecolor="none")
            axes[0].add_patch(rect)
            axes[0].text(x1, y1 - 6, f"{label} {conf:.2f}",
                         fontsize=9, color="white", fontweight="bold",
                         bbox=dict(facecolor=color, alpha=0.85, pad=2, edgecolor="none"))
            det_summary.append(f"{label} {conf:.2f}")
    else:
        axes[0].text(W_orig * 0.5, H_orig * 0.5, "no detections",
                     ha="center", va="center", fontsize=12,
                     color="#AAAAAA", style="italic")

    # Panel 2: EigenCAM overlay
    axes[1].imshow(overlay)
    axes[1].set_title("EigenCAM — activare backbone", fontsize=11,
                      color="#C9A84C", fontweight="bold", pad=5)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    det_str = "  |  ".join(det_summary) if det_summary else "nicio detecție"
    fig.suptitle(f"{fname}  —  {det_str}",
                 fontsize=10, color="#444444", y=0.97)

    out_path = os.path.join(OUT_DIR, f"{img_idx:02d}_{os.path.splitext(fname)[0][:35]}.png")
    plt.savefig(out_path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"  {os.path.basename(out_path)}  [{det_str}]")

print(f"\nGata. {len(paths)} imagini in: {OUT_DIR}")
