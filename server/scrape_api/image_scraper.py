import base64
import sys
import os
from io import BytesIO
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from PIL import Image

# Ensure blur_api is importable (uvicorn runs from server/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blur_api.local_detector import detect_sensitive

RISK_NORMALIZATION_FACTOR = 5.0

_LABEL_WEIGHTS = {
    "badge": 1.5, "id": 1.5, "license": 1.5, "plate": 1.5,
    "document": 1.2,
    "person": 0.8,
    "screen": 0.6, "monitor": 0.6, "laptop": 0.6,
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch_page_image_urls(page_url: str) -> list[str]:
    resp = requests.get(page_url, headers=_HEADERS, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    seen = set()
    urls = []

    for tag in soup.find_all(["img", "source"]):
        candidates = []
        for attr in ("src", "data-src", "data-lazy-src"):
            val = tag.get(attr, "").strip()
            if val:
                candidates.append(val)
        srcset = tag.get("srcset", "").strip()
        if srcset:
            first = srcset.split(",")[0].split()[0]
            if first:
                candidates.append(first)

        for raw in candidates:
            absolute = urljoin(page_url, raw)
            if absolute not in seen and absolute.startswith("http"):
                seen.add(absolute)
                urls.append(absolute)

    return urls


def fetch_image_bytes(image_url: str, max_bytes: int = 3_145_728) -> bytes | None:
    try:
        with requests.get(image_url, headers=_HEADERS, stream=True, timeout=8) as r:
            r.raise_for_status()
            content_length = r.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                return None
            chunks = []
            accumulated = 0
            for chunk in r.iter_content(chunk_size=65536):
                accumulated += len(chunk)
                if accumulated > max_bytes:
                    return None
                chunks.append(chunk)
            return b"".join(chunks)
    except Exception:
        return None


def resize_to_thumbnail(
    image_bytes: bytes, max_side: int = 600
) -> tuple[bytes, int, int, int, int]:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = img.size
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    thumb_w, thumb_h = img.size
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return buf.getvalue(), orig_w, orig_h, thumb_w, thumb_h


def _label_weight(label: str) -> float:
    label_lower = label.lower()
    for key, weight in _LABEL_WEIGHTS.items():
        if key in label_lower:
            return weight
    return 0.5


def compute_risk(regions: list[dict]) -> tuple[float, str]:
    if not regions:
        return 0.0, "NONE"
    raw = sum(_label_weight(r.get("label", "")) for r in regions)
    score = min(raw / RISK_NORMALIZATION_FACTOR, 1.0)
    if score <= 0:
        label = "NONE"
    elif score <= 0.3:
        label = "LOW"
    elif score <= 0.7:
        label = "MEDIUM"
    else:
        label = "HIGH"
    return round(score, 3), label


def scrape_and_analyze(page_url: str, detector: str = "yolo", cap: int = 30) -> dict:
    image_urls = fetch_page_image_urls(page_url)
    total_found = len(image_urls)
    capped = image_urls[:cap]

    results = []
    skipped = 0
    risky = 0

    for url in capped:
        raw_bytes = fetch_image_bytes(url)
        if raw_bytes is None:
            skipped += 1
            continue

        try:
            thumb_bytes, orig_w, orig_h, thumb_w, thumb_h = resize_to_thumbnail(raw_bytes)
        except Exception:
            skipped += 1
            continue

        try:
            regions = detect_sensitive(raw_bytes)
        except Exception:
            regions = []

        risk_score, risk_label = compute_risk(regions)
        if risk_score > 0:
            risky += 1

        results.append({
            "source_url":     url,
            "thumbnail_b64":  base64.b64encode(thumb_bytes).decode("utf-8"),
            "original_width":  orig_w,
            "original_height": orig_h,
            "thumbnail_width":  thumb_w,
            "thumbnail_height": thumb_h,
            "regions":    regions,
            "risk_score": risk_score,
            "risk_label": risk_label,
        })

    return {
        "page_url":          page_url,
        "total_images_found": total_found,
        "images_processed":  len(results),
        "images_with_risk":  risky,
        "skipped_count":     skipped,
        "detector":          detector,
        "results":           results,
    }
