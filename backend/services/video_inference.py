import cv2
import yt_dlp
import numpy as np
import asyncio
import uuid
import time
from pathlib import Path
from typing import AsyncGenerator
import json

FRAME_INTERVAL = 0.5  # process one frame every 0.5 seconds
MAX_FRAMES = 300       # safety cap
TEMP_DIR = Path("data/video_temp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

async def extract_youtube_stream_url(youtube_url: str) -> str:
    """Extract direct stream URL from YouTube link using yt-dlp."""
    ydl_opts = {
        "format": "best[height<=720][ext=mp4]/best[height<=720]/best",
        "quiet": True,
        "no_warnings": True,
    }
    loop = asyncio.get_event_loop()
    def _extract():
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            return info.get("url") or info["formats"][-1]["url"]
    return await loop.run_in_executor(None, _extract)

async def process_video_frames(
    source: str,  # file path, youtube URL, or RTSP URL
    country: str = "USA",
    conf_threshold: float = 0.25,
    max_frames: int = MAX_FRAMES,
) -> AsyncGenerator[dict, None]:
    """
    Generator that yields detection results frame by frame.
    Each yield is a dict:
    {
      frame_number: int,
      timestamp_sec: float,
      detections: [...],
      annotated_frame_base64: str,
      total_aircraft: int,
      source_type: str
    }
    """
    from services.detection_service import get_detection_service
    service = get_detection_service()
    
    # Resolve source
    source_type = "file"
    actual_source = source
    
    if source.startswith(("http://", "https://")) and "youtube" in source or "youtu.be" in source:
        source_type = "youtube"
        actual_source = await extract_youtube_stream_url(source)
    elif source.startswith("rtsp://") or (
         source.startswith("http") and source.endswith((".m3u8", ".mjpg", ".mjpeg"))):
        source_type = "stream"
    
    cap = cv2.VideoCapture(actual_source)
    if not cap.isOpened():
        yield {"error": f"Cannot open source: {source}", "source_type": source_type}
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_skip = max(1, int(fps * FRAME_INTERVAL))
    frame_count = 0
    processed_count = 0
    
    try:
        while processed_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue
            
            timestamp = frame_count / fps
            
            # Run detection
            try:
                result = service.detect_and_classify(
                    frame, conf_threshold=conf_threshold, country=country
                )
                yield {
                    "frame_number": frame_count,
                    "timestamp_sec": round(timestamp, 2),
                    "detections": result["detections"],
                    "annotated_frame_base64": result["annotated_image_base64"],
                    "total_aircraft": result["total_aircraft"],
                    "source_type": source_type,
                }
            except Exception as e:
                yield {
                    "frame_number": frame_count,
                    "timestamp_sec": round(timestamp, 2),
                    "error": str(e),
                    "source_type": source_type,
                }
            
            processed_count += 1
            await asyncio.sleep(0)  # yield control
    finally:
        cap.release()
