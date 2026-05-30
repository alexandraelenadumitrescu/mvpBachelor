import json
import os
import re
from io import BytesIO

import google.generativeai as genai
from PIL import Image

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
_model = genai.GenerativeModel("gemini-1.5-flash")

_PROMPT = """Identify all sensitive regions in this image: visible screens with data,
documents, badges, and ID cards.

Return ONLY a JSON array with absolute pixel coordinates, no markdown, no explanation:
[{"label": "screen", "x": 120, "y": 45, "w": 300, "h": 200}]

If no sensitive regions are found, return exactly: []"""


def detect_sensitive(image_bytes: bytes) -> list[dict]:
    """Send image to Gemini and return list of sensitive bounding boxes."""
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    response = _model.generate_content([_PROMPT, pil_image])
    return _parse_regions(response.text)


def _parse_regions(text: str) -> list[dict]:
    # Strip markdown code fences if Gemini wraps output anyway
    text = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
    try:
        regions = json.loads(text)
        if isinstance(regions, list):
            return regions
    except json.JSONDecodeError:
        pass
    return []
