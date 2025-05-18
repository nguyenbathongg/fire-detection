from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, HttpUrl
from sqlalchemy.orm import Session
from typing import List, Dict
from app.db.base import get_db
from app.services import video_analysis

router = APIRouter()

class VideoURLRequest(BaseModel):
    url: HttpUrl

class FireDetectionResult(BaseModel):
    frame: int
    video_time: str
    fire_detected: bool
    total_area: float

@router.post("/detect-by-url")
def detect_fire_by_url(
    req: VideoURLRequest,
    db: Session = Depends(get_db)
):
    try:
        processed_url, results = video_analysis.detect_fire_from_url(req.url, db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "processed_url": processed_url,
        "detections": results
    }
