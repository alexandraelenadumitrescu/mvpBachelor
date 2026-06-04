import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth.dependencies import get_current_active_user
from scrape_api.image_scraper import scrape_and_analyze

router = APIRouter()


class ScrapeRequest(BaseModel):
    url:      str
    detector: str = "yolo"


class ScrapedRegion(BaseModel):
    label: str
    x:     int
    y:     int
    w:     int
    h:     int


class ScrapedImageResult(BaseModel):
    source_url:      str
    thumbnail_b64:   str
    original_width:  int
    original_height: int
    thumbnail_width:  int
    thumbnail_height: int
    regions:    list[ScrapedRegion]
    risk_score: float
    risk_label: str


class ScrapeResponse(BaseModel):
    page_url:           str
    total_images_found: int
    images_processed:   int
    images_with_risk:   int
    skipped_count:      int
    detector:           str
    results:            list[ScrapedImageResult]


@router.post("/images", response_model=ScrapeResponse)
def scrape_images(
    req: ScrapeRequest,
    current_user=Depends(get_current_active_user),
):
    """
    Scrape all images from a webpage, run sensitive-region detection on each,
    and return results with risk scores. Reuses the existing YOLOv8 + badge
    detection pipeline. Cap: 30 images, max 3 MB per image.
    """
    if not req.url.startswith(("http://", "https://")):
        raise HTTPException(400, "URL must start with http:// or https://")
    try:
        data = scrape_and_analyze(req.url, req.detector)
    except Exception as e:
        raise HTTPException(500, f"Scrape failed: {e}")
    return ScrapeResponse(**data)
