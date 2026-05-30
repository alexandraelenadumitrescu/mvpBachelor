import cv2
import numpy as np


def apply_blur(image_bytes: bytes, regions: list[dict]) -> bytes:
    """Apply GaussianBlur to each sensitive region. Returns JPEG bytes."""
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    for r in regions:
        x1 = max(0, int(r["x"]))
        y1 = max(0, int(r["y"]))
        x2 = min(w, x1 + int(r["w"]))
        y2 = min(h, y1 + int(r["h"]))

        if x2 <= x1 or y2 <= y1:
            continue

        img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (51, 51), 0)

    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buffer.tobytes()
