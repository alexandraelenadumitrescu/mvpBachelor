"""
Shared face processing helpers — mirrors FaceGroupsActivity.java logic.
"""
from PIL import Image


def resize_to_max(img: Image.Image, max_side: int = 1024) -> Image.Image:
    """Resize so the long edge = max_side (same as Android decodeBitmap(uri, 1024))."""
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        return img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    return img


def crop_with_padding(img: Image.Image, facial_area: dict, padding: float = 0.20) -> Image.Image:
    """
    Crop face with proportional padding — mirrors Android cropFace(bmp, box, 0.20f).
    facial_area: dict with keys x, y, w, h (from DeepFace.extract_faces).
    """
    x = facial_area.get("x", 0)
    y = facial_area.get("y", 0)
    w = facial_area.get("w", img.width)
    h = facial_area.get("h", img.height)
    pad_x  = int(w * padding)
    pad_y  = int(h * padding)
    left   = max(0, x - pad_x)
    top    = max(0, y - pad_y)
    right  = min(img.width,  x + w + pad_x)
    bottom = min(img.height, y + h + pad_y)
    return img.crop((left, top, right, bottom))
