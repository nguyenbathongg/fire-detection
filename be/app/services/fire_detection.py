import os
import uuid
import cv2
import numpy as np
import time
import json
import io
import torch  # Thêm import torch ở đầu file
from typing import Dict, List, Tuple, Optional, Generator, Any, Union, BinaryIO
import logging
from datetime import datetime
from pathlib import Path
import asyncio
import threading
import queue
from fastapi import WebSocket
import hashlib

from app.core.config import settings
from app.utils.cloudinary_service import upload_bytes_to_cloudinary, download_from_cloudinary, delete_from_cloudinary

logger = logging.getLogger(__name__)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "video_codec;h264_cuvid"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"  # Tắt debug messages
cv2.setLogLevel(0)
# Cache toàn cục cho các URL Cloudinary để tái sử dụng giữa các lần gọi
# Sử dụng dict để lưu trữ URL và thông tin liên quan
global_cloudinary_cache = {}
# Biến lưu trữ URL của lần tải lên gần nhất
last_cloudinary_url = None
last_cloudinary_result = None


def predict_and_display(model, video_path, output_path=None, initial_skip_frames=2):
    """
    Xử lý video để phát hiện đám cháy và trả về từng frame đã xử lý.
    Hoạt động như một generator để hỗ trợ streaming realtime.
    
    Args:
        model: Mô hình YOLO cho phát hiện đám cháy
        video_path: Đường dẫn hoặc URL của video
        output_path: Đường dẫn lưu video kết quả (nếu None, không lưu)
        initial_skip_frames: Số frame ban đầu bỏ qua
        
    Yields:
        Tuple[np.ndarray, Dict]: Frame đã xử lý và thông tin kèm theo
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Không thể mở video: {video_path}")
        return

    # Warm-up model với frame đầu tiên
    ret, dummy_frame = cap.read()
    if ret:
        model.predict(dummy_frame, save=False, conf=0.5, verbose=False)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Khởi tạo VideoWriter nếu có output_path
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        out = cv2.VideoWriter(output_path, fourcc, fps_video, (width, height))

    frame_queue = queue.Queue(maxsize=0)
    result_queue = queue.Queue(maxsize=0)
    stop_event = threading.Event()
    frame_idx = 0

    def capture_thread():
        nonlocal frame_idx
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                stop_event.set()
                break
            frame_queue.put((frame_idx, frame))
            frame_idx += 1

    def inference_thread():
        processing_times = []
        max_samples = 10
        skip_frames = initial_skip_frames
        frame_duration = 1.0 / fps_video

        while not stop_event.is_set() or not frame_queue.empty():
            try:
                idx, frame = frame_queue.get(timeout=0.1)
                frames_left = total_frames - idx
                if frames_left <= fps_video:
                    skip = False
                else:
                    skip = idx % skip_frames != 0

                if skip:
                    result_queue.put((idx, frame, None, None, True, 0.0, True))
                    frame_queue.task_done()
                    continue

                start_time = time.time()
                result = model.predict(frame, save=False, conf=0.5, verbose=False)[0]
                detections = result.boxes
                segments = getattr(result, 'masks', None)
                processing_time = time.time() - start_time

                processing_times.append(processing_time)
                if len(processing_times) > max_samples:
                    processing_times.pop(0)

                if idx >= 10:
                    avg_processing_time = sum(processing_times) / len(processing_times)
                    skip_frames = min(int(np.ceil(avg_processing_time / frame_duration)), 3) if avg_processing_time > frame_duration else 0

                result_queue.put((idx, frame, detections, segments, False,
                                  sum(processing_times) / len(processing_times) if processing_times else 0, False))
                frame_queue.task_done()
            except queue.Empty:
                continue

    def draw_and_yield():
        prev_time = time.time()
        avg_fps = 0.0
        
        # Bộ nhớ tạm để giữ bounding box
        bbox_cache = {}  
        # Số frame giữ lại bounding box sau khi mất
        max_hold_frames = 3
        # Bộ nhớ tạm cho các mask
        mask_cache = {}
        mask_hold_frames = 2

        while not stop_event.is_set() or not result_queue.empty() or not frame_queue.empty():
            try:
                idx, frame, detections, segments, skip_frames, avg_processing_time, is_skipped = result_queue.get(timeout=0.1)
                video_time = idx / fps_video
                video_time_str = time.strftime("%H:%M:%S", time.gmtime(video_time))

                fire_detected = False
                total_fire_area = 0.0
                current_boxes = []
                current_masks = []
                current_confidences = []  # Thêm list để lưu confidence của frame hiện tại
                
                # Xử lý và lưu trữ mask
                if segments is not None:
                    masks = segments.data.cpu().numpy()
                    for mask in masks:
                        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                        mask_hash = hash(mask_resized.tobytes())
                        current_masks.append((mask_resized, mask_hash))
                        
                        blue_mask = np.zeros_like(frame, dtype=np.uint8)
                        blue_mask[mask_resized > 0.5] = (255, 0, 0)
                        alpha = 0.6
                        overlay = frame.copy()
                        overlay[mask_resized > 0.5] = blue_mask[mask_resized > 0.5]
                        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                
                        fire_pixels = np.sum(mask_resized > 0.5)
                        fire_area = fire_pixels / (width * height) * 100
                        total_fire_area += fire_area
                        
                # Cập nhật cache mask
                new_mask_cache = {}
                for mask_resized, mask_hash in current_masks:
                    new_mask_cache[mask_hash] = (mask_resized, mask_hold_frames)
                
                for mask_hash, (stored_mask, remain) in mask_cache.items():
                    if mask_hash not in new_mask_cache and remain > 0:
                        new_mask_cache[mask_hash] = (stored_mask, remain - 1)
                        blue_mask = np.zeros_like(frame, dtype=np.uint8)
                        blue_mask[stored_mask > 0.5] = (255, 0, 0)
                        alpha = 0.6
                        overlay = frame.copy()
                        overlay[stored_mask > 0.5] = blue_mask[stored_mask > 0.5]
                        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                
                mask_cache = new_mask_cache

                # Xử lý bounding box và confidence
                if detections is not None:
                    for det in detections:
                        if int(det.cls[0].item()) == 0:  # Nếu là class fire
                            x1, y1, x2, y2 = det.xyxy[0].cpu().numpy().astype(int)
                            conf = float(det.conf[0].item())
                            current_boxes.append(((x1, y1, x2, y2), conf))
                            current_confidences.append(conf)  # Thêm confidence vào list

                # Cập nhật cache bounding box
                new_cache = {}
                for (x1, y1, x2, y2), conf in current_boxes:
                    key = (x1, y1, x2, y2)
                    new_cache[key] = (max_hold_frames, conf)

                for key, (remain, old_conf) in bbox_cache.items():
                    if key not in new_cache and remain > 0:
                        new_cache[key] = (remain - 1, old_conf)
                        current_confidences.append(old_conf)  # Thêm confidence của box cũ vào list

                bbox_cache = new_cache

                # Vẽ tất cả bounding box từ cache và tính diện tích
                for (x1, y1, x2, y2), (remain, conf) in bbox_cache.items():
                    if remain > 0:
                        box_alpha = 1.0 if remain == max_hold_frames else 0.6 + (0.4 * remain / max_hold_frames)
                        color_intensity = min(255, int(200 + (conf * 55)))
                        box_color = (color_intensity, 0, 0)
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 4)
                        label = f"{conf:.2f}"
                        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 1)
                        cv2.rectangle(frame, (x1, y1 - text_height - 8), (x1 + text_width + 6, y1), box_color, -1)
                        cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                        
                        # area = ((x2 - x1) * (y2 - y1)) / (width * height) * 100
                        # total_fire_area += area
                        fire_detected = True

                # Tính FPS
                if not is_skipped:
                    current_time = time.time()
                    avg_fps = 1.0 / (current_time - prev_time)
                    prev_time = current_time

                # Vẽ FPS
                text = f"FPS: {avg_fps:.0f}"
                position = (5, 35)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.2
                thickness = 2

                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                x, y = position
                box_coords = ((x - 5, y - text_height - 10), (x + text_width + 5, y + 5))

                overlay = frame.copy()
                cv2.rectangle(overlay, box_coords[0], box_coords[1], (255, 0, 0), -1)
                alpha = 0.25
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                cv2.putText(frame, text, position, font, font_scale, (255, 255, 255), thickness)

                # Tính confidence trung bình cho frame hiện tại
                avg_confidence = sum(current_confidences) / len(current_confidences) if current_confidences else 0.0

                # Cập nhật frame_info với confidence
                frame_info = {
                    "frame": idx,
                    "video_time": video_time_str,
                    "fire_detected": fire_detected,
                    "total_area": round(float(total_fire_area), 4),
                    "confidence": round(float(avg_confidence), 4)  # Thêm confidence vào frame_info
                }

                if out is not None:
                    out.write(frame)
                if not is_skipped:
                    yield frame, frame_info

                result_queue.task_done()

            except queue.Empty:
                continue

    capture_t = threading.Thread(target=capture_thread, daemon=True)
    inference_t = threading.Thread(target=inference_thread, daemon=True)
    capture_t.start()
    inference_t.start()

    try:
        yield from draw_and_yield()
    finally:
        stop_event.set()
        capture_t.join(timeout=1)
        inference_t.join(timeout=1)
        cap.release()
        if out is not None:
            out.release()

class FireDetectionService:      
    def __init__(self):
        self.model_path = settings.MODEL_PATH
        self.model = None
        self.device = None
        
        # Kiểm tra GPU
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.device = torch.device('cuda')
            logger.info(f"GPU được sử dụng cho phát hiện đám cháy: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            logger.info("Sử dụng CPU cho phát hiện đám cháy")
        
        # Tải model YOLO khi khởi tạo service
        self.load_model()
        
    def load_model(self) -> bool:
        """
        Tải model YOLO và trả về True nếu thành công, False nếu thất bại
        
        Returns:
            bool: Trạng thái tải model
        """
        try:
            # Xử lý đường dẫn tương đối nếu cần
            if self.model_path.startswith('./') or self.model_path.startswith('../'):
                # Nếu là đường dẫn tương đối, chuyển nó thành đường dẫn tuyệt đối
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                absolute_model_path = os.path.normpath(os.path.join(base_dir, self.model_path))
                logger.info(f"Chuyển đổi đường dẫn model từ {self.model_path} thành {absolute_model_path}")
                self.model_path = absolute_model_path
              # Kiểm tra xem file model có tồn tại hay không
            if not os.path.exists(self.model_path):
                logger.error(f"Không tìm thấy file model tại đường dẫn: {self.model_path}")
                return False
                
            # Ghi nhật ký các thông tin về version ultralytics đang sử dụng
            try:
                import pkg_resources
                ultralytics_version = pkg_resources.get_distribution("ultralytics").version
                logger.info(f"Đang sử dụng ultralytics version: {ultralytics_version}")
                
                # Kiểm tra phiên bản và đưa ra cảnh báo nếu cần
                if ultralytics_version == "8.0.176":
                    logger.warning("Phiên bản ultralytics 8.0.176 có thể không tương thích với model YOLOv11")
                    logger.warning("Khuyến nghị: pip install ultralytics>=8.1.0")
            except:
                logger.warning("Không thể xác định phiên bản ultralytics")
            
            # Đặt biến môi trường để tắt kiểm tra bảo mật của PyTorch 2.6
            os.environ["TORCH_WEIGHTS_ONLY"] = "0"
            
            try:
                # Import thư viện sau khi đặt biến môi trường
                from ultralytics import YOLO
                
                # Configure torch serialization to allow loading model
                torch.backends.cudnn.benchmark = True  # Cải thiện hiệu suất
                
                # Patch the torch.load function to use weights_only=False
                original_load = torch.load
                def patched_load(*args, **kwargs):
                    kwargs['weights_only'] = False
                    return original_load(*args, **kwargs)
                
                # Apply the patch
                torch.load = patched_load
                
                # Tải model
                start_time = time.time()
                self.model = YOLO(self.model_path)
                
                # Đưa model lên GPU nếu có
                if self.use_gpu:
                    self.model.to(self.device)
                
                # Restore original torch.load
                torch.load = original_load
                
                logger.info(f"Thời gian tải model: {time.time() - start_time:.2f} giây")
                logger.info(f"Đã tải model YOLO thành công từ {self.model_path}")
                return True
            except AttributeError as e:                
                if "C3k2" in str(e):
                    logger.error(f"Lỗi với YOLOv11: {str(e)}")
                    logger.error("Model YOLOv11 yêu cầu phiên bản ultralytics mới hơn. Vui lòng nâng cấp: pip install ultralytics --upgrade")
                    logger.warning("Hệ thống sẽ tiếp tục chạy trong chế độ suy giảm (không có model)")
                    return False
                else:
                    raise  # Re-raise nếu lỗi AttributeError không liên quan đến C3k2
        except Exception as e:
            logger.error(f"Lỗi khi tải model YOLO: {str(e)}")
            self.model = None
            return False
            


        