import cv2
import uuid
import os
import tempfile
from typing import List, Dict, Tuple
from ultralytics import YOLO
from app.utils.cloudinary_service import upload_video_to_cloudinary
from sqlalchemy.orm import Session
from app.models import Video

MODEL_PATH = "./model/bestyolov11-25k.pt"
model = YOLO(MODEL_PATH)

def detect_fire_from_url(video_url: str, db: Session) -> Tuple[str, List[Dict]]:
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        raise ValueError(f"Không thể mở video từ URL: {video_url}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    resize_factor = 0.5
    results: List[Dict] = []
    frame_idx = 0
    processed_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        seconds = frame_idx / fps
        video_time = f"{int(seconds//3600):02}:{int((seconds%3600)//60):02}:{int(seconds%60):02}"
        resized_frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)

        frame_results = model.predict(
            resized_frame, save=False, save_txt=False, conf=0.5, verbose=False,
            imgsz=max(320, int(min(width, height) * resize_factor))
        )[0]

        detections = frame_results.boxes
        scale_factor = 1.0 / resize_factor
        total_fire_area = 0.0
        fire_detected = False

        for detection in detections:
            if int(detection.cls[0].item()) == 0:
                bbox = detection.xyxy[0].cpu().numpy() * scale_factor
                x1, y1, x2, y2 = bbox.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width-1, x2), min(height-1, y2)
                fire_area = ((x2 - x1) * (y2 - y1)) / (width * height) * 100
                total_fire_area += fire_area
                fire_detected = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                cv2.putText(frame, f"{fire_area:.2f}%", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        processed_frames.append(frame)
        results.append({
            "frame": frame_idx,
            "video_time": video_time,
            "fire_detected": fire_detected,
            "total_area": float(round(total_fire_area, 4))
        })
        frame_idx += 1
    cap.release()

    # Export processed video
    temp_video_path = os.path.join(tempfile.gettempdir(), f"processed_{uuid.uuid4()}.mp4")
    out = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for f in processed_frames:
        out.write(f)
    out.release()

    # Upload lên cloud
    success, msg, upload_result = upload_video_to_cloudinary(temp_video_path)
    if not success:
        raise RuntimeError(f"Lỗi upload video: {msg}")
    
    processed_url = upload_result.get("secure_url")
    processed_public_id = upload_result.get("public_id")

    # Gắn vào DB nếu tồn tại
    video_obj = db.query(Video).filter(Video.original_video_url == str(video_url)).first()
    if video_obj:
        video_obj.processed_video_url = processed_url
        video_obj.cloudinary_processed_id = processed_public_id
        db.commit()

    # Xóa file tạm
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)

    return processed_url, results
