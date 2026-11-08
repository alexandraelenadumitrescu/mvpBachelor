"""
generate_xai_visuals.py — Photography-inspired Grad-CAM visualizations for PhotoMatch.

Outputs (written to evaluation_assets/):
  xai_zebra_overexposure.png   — diagonal B&W stripes on overexposed regions
  xai_focus_peaking_blur.png   — gold contours on blurry regions (grayscale base)
  xai_spotlight_vignette.png   — 70% vignette darkening outside high-activation area

Usage:
  python generate_xai_visuals.py [image_path]

If image_path is omitted, the first image in server/images/raw/ is used.

Requirements:
  pip install -r server/requirements.txt
  server/defect_head.pt must exist (MobileNetV3Small weights)
"""

import os
import sys
import glob
import argparse
import warnings

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR  = os.path.join(ROOT, "server")
MODEL_PATH  = os.path.join(SERVER_DIR, "defect_head.pt")
RAW_DIR     = os.path.join(SERVER_DIR, "images", "raw")
OUT_DIR     = os.path.join(ROOT, "evaluation_assets")

DEFECT_NAMES = ["blur", "noise", "overexposure", "underexposure", "compression"]
device       = "cuda" if torch.cuda.is_available() else "cpu"

# Gold colour for focus peaking in OpenCV BGR
GOLD_BGR        = (55, 175, 212)
GOLD_BRIGHT_BGR = (100, 210, 240)


# ── Model helpers ──────────────────────────────────────────────────────────────

def load_defect_model():
    """Load defect_head.pt. Exits with a clear message if the file is missing."""
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model weights not found at:\n  {MODEL_PATH}")
        print("Place defect_head.pt (MobileNetV3Small checkpoint) at that path and re-run.")
        sys.exit(1)
    try:
        backbone = models.mobilenet_v3_small(weights=None)
        in_features = backbone.classifier[0].in_features
        backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(128, 5),
            nn.Sigmoid(),
        )
        backbone.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        backbone = backbone.to(device).eval()
        print(f"Loaded defect model ({device})")
        return backbone
    except Exception as exc:
        print(f"ERROR: Failed to load {MODEL_PATH}: {exc}")
        sys.exit(1)


def get_defect_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ── Image loading ──────────────────────────────────────────────────────────────

def load_image(image_path, transform):
    """
    Load image and return (orig_224_np, input_tensor).
    orig_224_np: uint8 RGB (224, 224, 3)
    input_tensor: float32 (1, 3, 224, 224) on device
    """
    pil_img     = Image.open(image_path).convert("RGB")
    pil_resized = pil_img.resize((224, 224), Image.LANCZOS)
    orig_np     = np.array(pil_resized)                        # uint8 RGB
    input_tensor = transform(pil_resized).unsqueeze(0).to(device)
    return orig_np, input_tensor


# ── Grad-CAM core ─────────────────────────────────────────────────────────────

class DefectOutputTarget:
    """Grad-CAM target for a single output of our 5-output regression model.

    pytorch_grad_cam may pass either a batched (N, 5) or unbatched (5,) tensor
    depending on the internal reduction step, so handle both shapes.
    """
    def __init__(self, idx):
        self.idx = idx

    def __call__(self, model_output):
        if model_output.dim() == 2:
            return model_output[:, self.idx]
        return model_output[self.idx]


def get_gradcam_heatmap(model, input_tensor, defect_idx):
    """
    Compute Grad-CAM heatmap for one defect output.
    Returns: float32 (224, 224) array in [0, 1].
    Target layer: backbone.features[-1] — last Conv2dNormActivation before avgpool.
    """
    from pytorch_grad_cam import GradCAM

    target_layer = model.features[-1]
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[DefectOutputTarget(defect_idx)],
        )
    return grayscale_cam[0]  # (224, 224)


# ── Visualization 1: Zebra Pattern (overexposure) ─────────────────────────────

def visualize_zebra_overexposure(orig_np, heatmap):
    """
    Overlay diagonal B&W zebra stripes on high-activation (overexposed) regions.
    Stripes: 10-pixel pitch at 45°.
    """
    h, w = orig_np.shape[:2]

    # Binary mask of high-activation region
    mask = (heatmap >= 0.5).astype(np.uint8)

    # Generate stripe pattern: white where (r+c)//10 is even, black otherwise
    rows, cols = np.indices((h, w))
    stripe = np.where(((rows + cols) // 10) % 2 == 0, 255, 0).astype(np.uint8)
    stripe_rgb = np.stack([stripe, stripe, stripe], axis=-1)

    result = orig_np.copy()
    result[mask == 1] = stripe_rgb[mask == 1]

    out_path = os.path.join(OUT_DIR, "xai_zebra_overexposure.png")
    cv2.imwrite(out_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"Saved: {out_path}")


# ── Visualization 2: Focus Peaking (blur) ─────────────────────────────────────

def visualize_focus_peaking_blur(orig_np, heatmap):
    """
    Desaturate image to grayscale; draw gold contours on high-activation (blurry) areas.
    Gold BGR: (55, 175, 212).  Glow effect via two-pass contour draw.
    """
    gray   = cv2.cvtColor(orig_np, cv2.COLOR_RGB2GRAY)
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Top 30% activation threshold
    threshold_val = float(np.percentile(heatmap, 70))
    high_mask = (heatmap >= threshold_val).astype(np.uint8) * 255

    contours, _ = cv2.findContours(high_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Outer glow (thicker, slightly dimmer)
    cv2.drawContours(canvas, contours, -1, GOLD_BGR, thickness=3)
    # Inner highlight (thinner, brighter)
    cv2.drawContours(canvas, contours, -1, GOLD_BRIGHT_BGR, thickness=1)

    out_path = os.path.join(OUT_DIR, "xai_focus_peaking_blur.png")
    cv2.imwrite(out_path, canvas)
    print(f"Saved: {out_path}")


# ── Visualization 3: Spotlight Vignette (all defects) ─────────────────────────

def visualize_spotlight_vignette(orig_np, heatmap):
    """
    Darken image by 70% everywhere except where Grad-CAM activation is high.
    High-activation areas remain at original exposure.
    """
    orig_f   = orig_np.astype(np.float32) / 255.0
    darkened = orig_f * 0.3

    mask     = heatmap[:, :, np.newaxis]          # (224, 224, 1) for broadcast
    result_f = darkened + (orig_f - darkened) * mask
    result   = np.clip(result_f * 255, 0, 255).astype(np.uint8)

    out_path = os.path.join(OUT_DIR, "xai_spotlight_vignette.png")
    cv2.imwrite(out_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def resolve_image_path(arg_path):
    if arg_path:
        if not os.path.exists(arg_path):
            print(f"ERROR: Image not found: {arg_path}")
            sys.exit(1)
        return arg_path

    candidates = sorted(glob.glob(os.path.join(RAW_DIR, "*.jpg")))
    if not candidates:
        print(f"ERROR: No image path given and no .jpg files found in:\n  {RAW_DIR}")
        print("Either pass an image path as argument or place MIT FiveK images in that directory.")
        sys.exit(1)

    chosen = candidates[0]
    print(f"No image specified — using: {chosen}")
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_path", nargs="?", default=None,
                        help="Path to input image (JPEG/PNG). Defaults to first image in server/images/raw/.")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    image_path = resolve_image_path(args.image_path)
    print(f"Input image : {image_path}")
    print(f"Device      : {device}")
    print(f"Output dir  : {OUT_DIR}")

    model     = load_defect_model()
    transform = get_defect_transform()
    orig_np, input_tensor = load_image(image_path, transform)

    print("\nComputing Grad-CAM heatmaps…")
    heatmap_overexp  = get_gradcam_heatmap(model, input_tensor, defect_idx=2)  # overexposure
    heatmap_blur     = get_gradcam_heatmap(model, input_tensor, defect_idx=0)  # blur
    heatmaps_all     = [get_gradcam_heatmap(model, input_tensor, i) for i in range(5)]
    heatmap_vignette = np.mean(heatmaps_all, axis=0)

    print("\nRendering visualizations…")
    visualize_zebra_overexposure(orig_np, heatmap_overexp)
    visualize_focus_peaking_blur(orig_np, heatmap_blur)
    visualize_spotlight_vignette(orig_np, heatmap_vignette)

    print("\nDone.")


if __name__ == "__main__":
    main()
