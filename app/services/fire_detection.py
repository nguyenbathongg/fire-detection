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
                
                # Xử lý và lưu trữ mask
                if segments is not None:
                    masks = segments.data.cpu().numpy()
                    for mask in masks:
                        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                        # Lưu mask vào cache hiện tại
                        mask_hash = hash(mask_resized.tobytes())
                        current_masks.append((mask_resized, mask_hash))
                        
                        # Vẽ mask
                        blue_mask = np.zeros_like(frame, dtype=np.uint8)
                        blue_mask[mask_resized > 0.5] = (255, 0, 0)
                        alpha = 0.6
                        overlay = frame.copy()
                        overlay[mask_resized > 0.5] = blue_mask[mask_resized > 0.5]
                        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                
                # Cập nhật cache mask
                new_mask_cache = {}
                for mask_resized, mask_hash in current_masks:
                    new_mask_cache[mask_hash] = (mask_resized, mask_hold_frames)
                
                # Giữ lại các mask trong cache
                for mask_hash, (stored_mask, remain) in mask_cache.items():
                    if mask_hash not in new_mask_cache and remain > 0:
                        new_mask_cache[mask_hash] = (stored_mask, remain - 1)
                        
                        # Vẽ mask từ cache
                        blue_mask = np.zeros_like(frame, dtype=np.uint8)
                        blue_mask[stored_mask > 0.5] = (255, 0, 0)
                        alpha = 0.6  # Giảm độ đậm cho mask cũ
                        overlay = frame.copy()
                        overlay[stored_mask > 0.5] = blue_mask[stored_mask > 0.5]
                        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                
                mask_cache = new_mask_cache

                # Xử lý bounding box
                if detections is not None:
                    for det in detections:
                        if int(det.cls[0].item()) == 0:
                            x1, y1, x2, y2 = det.xyxy[0].cpu().numpy().astype(int)
                            conf = float(det.conf[0].item())
                            current_boxes.append(((x1, y1, x2, y2), conf))

                # Cập nhật cache bounding box
                new_cache = {}

                # Thêm các box mới vào cache
                for (x1, y1, x2, y2), conf in current_boxes:
                    key = (x1, y1, x2, y2)
                    new_cache[key] = (max_hold_frames, conf)  # (số frame giữ lại, confidence)

                # Giữ lại các box cũ trong cache
                for key, (remain, old_conf) in bbox_cache.items():
                    if key not in new_cache and remain > 0:
                        new_cache[key] = (remain - 1, old_conf)

                bbox_cache = new_cache

                # Vẽ tất cả bounding box từ cache
                for (x1, y1, x2, y2), (remain, conf) in bbox_cache.items():
                    if remain > 0:
                        # Alpha giảm dần theo số frame còn lại để tạo hiệu ứng mờ dần
                        box_alpha = 1.0 if remain == max_hold_frames else 0.6 + (0.4 * remain / max_hold_frames)
                        
                        # Màu đậm dần theo confidence
                        color_intensity = min(255, int(200 + (conf * 55)))
                        box_color = (color_intensity, 0, 0)
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 4)
                        label = f"{conf:.2f}"
                        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 1)
                        cv2.rectangle(frame, (x1, y1 - text_height - 8), (x1 + text_width + 6, y1), box_color, -1)
                        cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                        
                        area = ((x2 - x1) * (y2 - y1)) / (width * height) * 100
                        total_fire_area += area
                        fire_detected = True

                # Tính FPS đơn giản nếu không skip frame
                if not is_skipped:
                    current_time = time.time()
                    avg_fps = 1.0 / (current_time - prev_time)
                    prev_time = current_time

                # Vẽ FPS
                # Nội dung và vị trí
                text = f"FPS: {avg_fps:.0f}"
                position = (5, 35)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.2
                thickness = 2

                # Tính kích thước ô chữ
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                x, y = position
                box_coords = ((x - 5, y - text_height - 10), (x + text_width + 5, y + 5))

                # Tạo lớp overlay để vẽ nền trong suốt
                overlay = frame.copy()
                cv2.rectangle(overlay, box_coords[0], box_coords[1], (255, 0, 0), -1)  # màu nền xanh dương

                # Áp dụng overlay trong suốt
                alpha = 0.25  # Độ trong suốt
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                # Vẽ chữ màu trắng lên frame
                cv2.putText(frame, text, position, font, font_scale, (255, 255, 255), thickness)

                frame_info = {
                    "frame": idx,
                    "video_time": video_time_str,
                    "fire_detected": fire_detected,
                    "total_area": round(float(total_fire_area), 4)
                }

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
            

        
    def detect_fire_from_memory(self, video_data: bytes) -> Tuple[bool, List[Dict], Optional[Tuple[bytes, str]]]:
        """
        Phát hiện đám cháy trong video từ bộ nhớ
        
        Args:
            video_data: Dữ liệu nhị phân của video
            
        Returns:
            Tuple[bool, List[Dict], Optional[Tuple[bytes, str]]]: 
                - Boolean cho biết có phát hiện đám cháy hay không
                - Danh sách các đoạn thời gian phát hiện đám cháy
                - Tuple gồm dữ liệu nhị phân của frame đám cháy rõ nhất và định dạng
        """
        global last_cloudinary_url, last_cloudinary_result, global_cloudinary_cache
        
        try:
            # Đo thời gian xử lý
            total_start_time = time.time()
            
            # Kiểm tra xem có URL cache mới nhất không
            cloudinary_url = None
            if last_cloudinary_url is not None:
                logger.info(f"Sử dụng URL Cloudinary gần nhất: {last_cloudinary_url}")
                cloudinary_url = last_cloudinary_url
            else:
                # Kiểm tra cache dựa trên hash của video
                video_hash = hashlib.md5(video_data[:1024*1024]).hexdigest()
                
                if video_hash in global_cloudinary_cache:
                    logger.info(f"Sử dụng URL Cloudinary từ cache cho hash {video_hash[:8]}")
                    cloudinary_url = global_cloudinary_cache[video_hash]['url']
                    last_cloudinary_url = cloudinary_url
                    last_cloudinary_result = global_cloudinary_cache[video_hash]['result']
                else:
                    # Tải video lên Cloudinary nếu chưa có trong cache
                    logger.info(f"Đang tải video lên Cloudinary...")
                    upload_start_time = time.time()
                    process_uuid = str(uuid.uuid4())
                    success, message, result = upload_bytes_to_cloudinary(video_data, filename=f"fire_detection_{process_uuid}.mp4")
                    
                    if not success or not result:
                        logger.error(f"Không thể tải video lên Cloudinary: {message}")
                        return False, [], None
                    
                    cloudinary_url = result.get('secure_url')
                    logger.info(f"Đã tải video lên Cloudinary thành công: {cloudinary_url}")
                    logger.info(f"Thời gian tải lên Cloudinary: {time.time() - upload_start_time:.2f} giây")
                    
                    # Lưu vào cache toàn cục
                    global_cloudinary_cache[video_hash] = {
                        'url': cloudinary_url,
                        'result': result,
                        'timestamp': time.time()
                    }
                    
                    # Cập nhật lần tải lên gần nhất
                    last_cloudinary_url = cloudinary_url
                    last_cloudinary_result = result
            
            # Mở video từ URL Cloudinary
            open_start = time.time()
            cap = cv2.VideoCapture(cloudinary_url)
            if not cap.isOpened():
                logger.error(f"Không thể mở video từ URL Cloudinary: {cloudinary_url}")
                return False, [], None
            logger.info(f"Thời gian mở video từ Cloudinary: {time.time() - open_start:.2f} giây")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(f"Video FPS: {fps}, Frame count: {frame_count}")
            
            # Tối ưu cho hiệu năng
            # Tăng số frame bỏ qua để tăng tốc độ xử lý, tỷ lệ thuận với độ dài video
            skip_frames_ratio = max(1, min(10, int(frame_count / 1000)))
            skip_frames = max(6, skip_frames_ratio)
            logger.info(f"Processing with skip_frames={skip_frames}")
            
            detections = []
            fire_detected = False
            is_in_fire_segment = False
            start_time = None
            current_fire_segment = {}
            max_confidence = 0
            max_confidence_frame = None
            max_confidence_frame_number = None
            
            # Lưu thời gian bắt đầu để đo hiệu suất
            processing_start_time = time.time()
            
            # Đọc từng frame và xử lý
            frame_number = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Thời gian của frame hiện tại
                current_time = frame_number / fps
                
                # Xử lý một số frame để tăng tốc độ
                if frame_number % skip_frames != 0:
                    frame_number += 1
                    continue
                
                # Resize frame để tăng tốc độ (giảm xuống còn 640x360 hoặc giữ tỷ lệ)
                h, w = frame.shape[:2]
                detection_size = 640  # Kích thước tham chiếu cho phát hiện
                
                # Tính tỷ lệ resize, giữ tỷ lệ khung hình
                if w > h:
                    new_w = detection_size
                    new_h = int(h * (detection_size / w))
                else:
                    new_h = detection_size
                    new_w = int(w * (detection_size / h))
                    
                # Resize frame cho phát hiện
                detection_frame = cv2.resize(frame, (new_w, new_h))
                
                # Phát hiện đám cháy sử dụng model YOLO
                if self.model:
                    result = self.model.predict(
                        detection_frame, 
                        save=False, 
                        conf=0.5, 
                        verbose=False,
                        device=self.device if self.use_gpu else 'cpu',
                        half=self.use_gpu  # Sử dụng FP16 nếu có GPU
                    )[0]
                    detections_yolo = result.boxes
                    
                    # Kiểm tra có lửa được phát hiện không (lửa là class 0)
                    has_fire_in_frame = False
                    max_conf_in_frame = 0
                    
                    for det in detections_yolo:
                        if int(det.cls[0].item()) == 0:  # Class 0 là lửa
                            has_fire_in_frame = True
                            conf = float(det.conf[0].item())
                            if conf > max_conf_in_frame:
                                max_conf_in_frame = conf
                        is_fire_in_frame = has_fire_in_frame
                    confidence = max_conf_in_frame if has_fire_in_frame else 0.0
                else:
                    # Không có model và không dùng phương án dự phòng
                    is_fire_in_frame = False
                    confidence = 0.0
                
                if is_fire_in_frame:
                    fire_detected = True
                    
                    # Kiểm tra confidence
                    if confidence > max_confidence:
                        max_confidence = confidence
                        max_confidence_frame = frame.copy()
                        max_confidence_frame_number = frame_number
                    
                    # Nếu không đang trong đoạn phát hiện lửa, bắt đầu đoạn mới
                    if not is_in_fire_segment:
                        is_in_fire_segment = True
                        start_time = current_time
                        current_fire_segment = {
                            "start_time": start_time,
                            "confidence": confidence
                        }
                        
                else:
                    # Nếu đang trong đoạn phát hiện lửa, kết thúc đoạn
                    if is_in_fire_segment:
                        is_in_fire_segment = False
                        current_fire_segment["end_time"] = current_time
                        detections.append(current_fire_segment)
                        
                frame_number += 1
            
            # Xử lý đoạn lửa cuối cùng nếu video kết thúc mà vẫn đang phát hiện lửa
            if is_in_fire_segment:
                current_fire_segment["end_time"] = frame_number / fps
                detections.append(current_fire_segment)
            
            # Lưu frame có đám cháy rõ nhất vào bộ nhớ
            max_fire_frame_data = None
            if max_confidence_frame is not None:
                # Chuyển đổi frame thành bytes
                _, buffer = cv2.imencode('.jpg', max_confidence_frame)
                max_fire_frame_data = (buffer.tobytes(), '.jpg')
            
            cap.release()
            
            # Xóa file tạm
            if 'is_temp_file_created' in locals() and is_temp_file_created:
                if os.path.exists(temp_input_path):
                    os.remove(temp_input_path)
                    logger.info(f"Đã xóa file tạm: {temp_input_path}")
            
            # Chuẩn bị kết quả
            result_detections = []
            for detection in detections:
                result_detections.append({
                    "fire_start_time": detection["start_time"],
                    "fire_end_time": detection["end_time"],
                    "confidence": detection["confidence"],
                    "max_fire_frame": max_confidence_frame_number
                })
            
            logger.info(f"Thời gian xử lý: {time.time() - total_start_time:.2f} giây")
            logger.info(f"Thời gian xử lý thực tế: {time.time() - processing_start_time:.2f} giây")
            
            return fire_detected, result_detections, max_fire_frame_data
            
        except Exception as e:
            logger.error(f"Error in fire detection: {str(e)}")
            logger.exception(e)
            
            # Xóa file tạm nếu có
            if 'is_temp_file_created' in locals() and 'temp_input_path' in locals() and is_temp_file_created:
                if os.path.exists(temp_input_path):
                    try:
                        os.remove(temp_input_path)
                    except:
                        pass
                        
            return False, [], None

    def process_video_from_memory(self, video_data: bytes) -> Tuple[bool, bytes, Dict]:
        """
        Xử lý video để đánh dấu các vùng phát hiện đám cháy,
        sử dụng Cloudinary để xử lý video và xử lý đa luồng
        
        Args:
            video_data: Dữ liệu nhị phân của video
            
        Returns:
            Tuple[bool, bytes, Dict]: 
                - Boolean cho biết có xử lý thành công không
                - Dữ liệu nhị phân của video đã xử lý
                - Thông tin phát hiện
        """
        global last_cloudinary_url, last_cloudinary_result, global_cloudinary_cache
        
        # Đo thời gian xử lý
        total_start_time = time.time()
        temp_output_path = None
        
        try:
            # Kiểm tra xem có URL cache mới nhất không
            cloudinary_url = None
            if last_cloudinary_url is not None:
                logger.info(f"Sử dụng URL Cloudinary gần nhất: {last_cloudinary_url}")
                cloudinary_url = last_cloudinary_url
            else:
                # Kiểm tra cache dựa trên hash của video
                video_hash = hashlib.md5(video_data[:1024*1024]).hexdigest()
                
                if video_hash in global_cloudinary_cache:
                    logger.info(f"Sử dụng URL Cloudinary từ cache cho hash {video_hash[:8]}")
                    cloudinary_url = global_cloudinary_cache[video_hash]['url']
                    last_cloudinary_url = cloudinary_url
                    last_cloudinary_result = global_cloudinary_cache[video_hash]['result']
                else:
                    # Tải video lên Cloudinary nếu chưa có trong cache
                    logger.info(f"Đang tải video lên Cloudinary...")
                    upload_start_time = time.time()
                    process_uuid = str(uuid.uuid4())
                    success, message, result = upload_bytes_to_cloudinary(video_data, filename=f"fire_detection_{process_uuid}.mp4")
                    
                    if not success or not result:
                        logger.error(f"Không thể tải video lên Cloudinary: {message}")
                        return False, None, {}
                    
                    cloudinary_url = result.get('secure_url')
                    logger.info(f"Đã tải video lên Cloudinary thành công: {cloudinary_url}")
                    logger.info(f"Thời gian tải lên Cloudinary: {time.time() - upload_start_time:.2f} giây")
                    
                    # Lưu vào cache toàn cục
                    global_cloudinary_cache[video_hash] = {
                        'url': cloudinary_url,
                        'result': result,
                        'timestamp': time.time()
                    }
                    
                    # Cập nhật lần tải lên gần nhất
                    last_cloudinary_url = cloudinary_url
                    last_cloudinary_result = result
            
            # Tạo file tạm để lưu video đã xử lý
            temp_output_path = f"temp_output_{uuid.uuid4()}.mp4"
            
            # Lưu thời gian bắt đầu xử lý
            processing_start_time = time.time()
            
            # Sử dụng hàm predict_and_display để xử lý video với đa luồng
            logger.info(f"Bắt đầu xử lý video đa luồng")
            
            # Thông tin phát hiện đám cháy
            fire_detected = False
            detections = []
            frame_count = 0
            max_confidence = 0
            max_confidence_time = None
            
            # Xử lý video với hàm predict_and_display
            for frame, frame_info in predict_and_display(self.model, cloudinary_url, temp_output_path, initial_skip_frames=3):
                frame_count += 1
                
                # Ghi nhận thông tin phát hiện đám cháy
                if frame_info["fire_detected"]:
                    fire_detected = True
                    
                    # Tính confidence dựa trên diện tích
                    area_confidence = min(1.0, frame_info["total_area"] / 10)
                    
                    # Ghi nhận thời điểm có confidence cao nhất
                    if area_confidence > max_confidence:
                        max_confidence = area_confidence
                        max_confidence_time = frame_info["video_time"]
                    
                    # Thêm thông tin phát hiện
                    detections.append({
                        "frame": frame_info["frame"],
                        "time": frame_info["video_time"],
                        "confidence": area_confidence,
                        "area": frame_info["total_area"]
                    })
            
            logger.info(f"Xử lý xong {frame_count} frames")
            logger.info(f"Thời gian xử lý: {time.time() - processing_start_time:.2f} giây")
            
            # Đọc video đã xử lý vào bộ nhớ
            if os.path.exists(temp_output_path):
                with open(temp_output_path, 'rb') as f:
                    processed_video_data = f.read()
                
                # Xóa file tạm thời
                os.remove(temp_output_path)
                logger.info(f"Đã xóa file tạm: {temp_output_path}")
            else:
                logger.error(f"Không tìm thấy file video đã xử lý: {temp_output_path}")
                return False, None, {}
            
            logger.info(f"Đã xử lý xong video, kích thước: {len(processed_video_data)} bytes")
            
            # Tạo thông tin phát hiện
            detection_info = {
                "fire_detected": fire_detected,
                "detections": detections,
                "frames_processed": frame_count,
                "max_confidence": max_confidence,
                "max_confidence_time": max_confidence_time
            }
            
            return True, processed_video_data, detection_info
            
        except Exception as e:
            logger.error(f"Error processing video from memory: {str(e)}")
            logger.exception(e)
            
            # Xóa file tạm nếu có
            if temp_output_path and os.path.exists(temp_output_path):
                try:
                    os.remove(temp_output_path)
                except:
                    pass
                    
            return False, None, {}
            
    async def process_video_streaming_from_memory(self, video_data: bytes, websocket: WebSocket = None) -> Tuple[bool, bytes, Dict]:
        """
        Xử lý video để phát hiện đám cháy và gửi frame đã xử lý qua WebSocket,
        sử dụng Cloudinary để xử lý video
        
        Args:
            video_data: Dữ liệu nhị phân của video
            websocket: WebSocket để gửi frame đã xử lý
            
        Returns:
            Tuple[bool, bytes, Dict]: 
                - Boolean cho biết có xử lý thành công không
                - Dữ liệu nhị phân của video đã xử lý
                - Thông tin phát hiện
        """
        try:
            logger.info("Bắt đầu phát hiện đám cháy streaming")
            
            # Thử tải lại model nếu chưa tải
            if not self.model:
                logger.info("Đang thử tải lại model YOLO...")
                model_loaded = self.load_model()
                if model_loaded:
                    logger.info("Đã tải lại model YOLO thành công")
                else:
                    logger.warning("Không thể tải model YOLO")
            
            # Kiểm tra xem có URL cache mới nhất không
            global last_cloudinary_url, last_cloudinary_result, global_cloudinary_cache
            
            if websocket:
                await websocket.send_json({"status": "info", "message": "Đang chuẩn bị video..."})
            
            cloudinary_url = None
            if last_cloudinary_url is not None:
                logger.info(f"Sử dụng URL Cloudinary gần nhất: {last_cloudinary_url}")
                cloudinary_url = last_cloudinary_url
                if websocket:
                    await websocket.send_json({"status": "info", "message": "Sử dụng video đã tải lên trước đó"})
            else:
                # Kiểm tra cache dựa trên hash của video
                video_hash = hashlib.md5(video_data[:1024*1024]).hexdigest()
                
                if video_hash in global_cloudinary_cache:
                    logger.info(f"Sử dụng URL Cloudinary từ cache cho hash {video_hash[:8]}")
                    cloudinary_url = global_cloudinary_cache[video_hash]['url']
                    last_cloudinary_url = cloudinary_url
                    last_cloudinary_result = global_cloudinary_cache[video_hash]['result']
                    if websocket:
                        await websocket.send_json({"status": "info", "message": "Sử dụng video đã tải lên trước đó"})
                else:
                    # Tải video lên Cloudinary nếu chưa có trong cache
                    upload_start_time = time.time()
                    if websocket:
                        await websocket.send_json({"status": "info", "message": "Đang tải video lên Cloudinary..."})
                    
                    logger.info(f"Đang tải video lên Cloudinary...")
                    process_uuid = str(uuid.uuid4())
                    success, message, result = upload_bytes_to_cloudinary(video_data, filename=f"fire_detection_{process_uuid}.mp4")
                    
                    if not success or not result:
                        logger.error(f"Không thể tải video lên Cloudinary: {message}")
                        if websocket:
                            await websocket.send_json({"status": "error", "message": f"Không thể tải video lên Cloudinary: {message}"})
                        return False, None, {}
                    
                    cloudinary_url = result.get('secure_url')
                    logger.info(f"Đã tải video lên Cloudinary thành công: {cloudinary_url}")
                    logger.info(f"Thời gian tải lên Cloudinary: {time.time() - upload_start_time:.2f} giây")
                    
                    # Lưu vào cache toàn cục
                    global_cloudinary_cache[video_hash] = {
                        'url': cloudinary_url,
                        'result': result,
                        'timestamp': time.time()
                    }
                    
                    # Cập nhật lần tải lên gần nhất
                    last_cloudinary_url = cloudinary_url
                    last_cloudinary_result = result
                    
                    if websocket:
                        await websocket.send_json({"status": "info", "message": "Video đã sẵn sàng để xử lý"})
            
            # Mở video từ URL Cloudinary
            cap = cv2.VideoCapture(cloudinary_url)
            if not cap.isOpened():
                logger.error(f"Không thể mở video từ URL Cloudinary: {cloudinary_url}")
                if websocket:
                    await websocket.send_json({"status": "error", "message": "Không thể mở video từ Cloudinary"})
                return False, None, {}
                
            # Xử lý video với phát hiện đám cháy
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if websocket:
                await websocket.send_json({
                    "status": "info", 
                    "message": f"Bắt đầu xử lý video: {width}x{height}, {fps} FPS, {frame_count} frames"
                })
            
            frame_idx = 0
            detections = []
            fire_detected = False
            skip_frames = 6  # Bỏ qua một số frame để tăng tốc độ xử lý
            processed_frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Hiển thị tất cả các khung hình, không bỏ qua
                video_time = frame_idx / fps
                video_time_str = time.strftime("%H:%M:%S", time.gmtime(video_time))
                
                # Xử lý phát hiện đám cháy trên frame hiện tại
                process_this_frame = (frame_idx % skip_frames == 0)
                current_fire_detected = False
                total_fire_area = 0.0
                
                if process_this_frame:
                    # Thử sử dụng model YOLO nếu có
                    if self.model:
                        try:
                            # Xử lý phát hiện đám cháy với YOLO
                            result = self.model.predict(frame, save=False, conf=0.5, verbose=False)[0]
                            boxes = result.boxes
                            
                            # Xử lý bounding boxes cho lửa
                            for det in boxes:
                                if int(det.cls[0].item()) == 0:  # Class 0 là lửa
                                    x1, y1, x2, y2 = det.xyxy[0].cpu().numpy().astype(int)
                                    conf = float(det.conf[0].item())
                                    
                                    # Cập nhật confidence
                                    if conf > fire_conf:
                                        fire_conf = conf
                                    
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                    cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                                    
                                    area = ((x2 - x1) * (y2 - y1)) / (width * height) * 100
                                    total_fire_area += area
                                    current_fire_detected = True
                        except Exception as e:
                            logger.error(f"Lỗi khi xử lý frame với YOLO: {str(e)}")
                            # Nếu lỗi với YOLO, sử dụng phương pháp dự phòng
                            current_fire_detected, confidence = self._detect_fire_in_frame(frame)
                        if current_fire_detected:
                            # Phân tích vùng màu lửa
                            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                            lower_red1 = np.array([0, 120, 70])
                            upper_red1 = np.array([10, 255, 255])
                            lower_red2 = np.array([170, 120, 70])
                            upper_red2 = np.array([180, 255, 255])
                            
                            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                            mask = cv2.bitwise_or(mask1, mask2)
                            
                            # Tìm contours của vùng lửa
                            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            # Vẽ contours lên frame
                            for contour in contours:
                                # Lọc những contour quá nhỏ
                                if cv2.contourArea(contour) > 50:  # Ngưỡng diện tích
                                    # Vẽ bounding box màu đỏ quanh vùng lửa
                                    x, y, w, h = cv2.boundingRect(contour)
                                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                                    
                                    area = (w * h) / (width * height) * 100
                                    total_fire_area += area
                    
                # Cập nhật trạng thái phát hiện lửa
                if current_fire_detected:
                    fire_detected = True
                    detections.append({
                        "frame": frame_idx,
                        "time": video_time,
                        "confidence": round(float(0.8), 4),  # Giá trị mặc định
                        "area": round(float(total_fire_area), 4)
                    })
                
                # Thêm timestamp và thông tin phát hiện
                cv2.putText(frame, video_time_str, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                           
                if current_fire_detected:
                    cv2.putText(frame, f"FIRE DETECTED - Area: {total_fire_area:.2f}%", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Gửi frame qua WebSocket
                if websocket and process_this_frame:
                    try:
                        # Chuyển đổi frame thành dạng JPEG
                        _, buffer = cv2.imencode('.jpg', frame)
                        frame_bytes = buffer.tobytes()
                        
                        # Gửi frame qua WebSocket
                        await websocket.send_bytes(frame_bytes)
                        
                        # Gửi thông tin trạng thái
                        status_info = {
                            "status": "processing",
                            "frame": frame_idx,
                            "total_frames": frame_count,
                            "progress": round(frame_idx / frame_count * 100, 2),
                            "fire_detected": current_fire_detected,
                            "fire_area": round(total_fire_area, 2) if current_fire_detected else 0
                        }
                        await websocket.send_json(status_info)
                        processed_frame_count += 1
                    except Exception as e:
                        logger.error(f"Lỗi khi gửi frame qua WebSocket: {str(e)}")
                        continue
                
                frame_idx += 1
            
            # Giải phóng resources
            cap.release()
            
            # Xóa file tạm
            if 'is_temp_file_created' in locals() and is_temp_file_created:
                if os.path.exists(temp_input_path):
                    os.remove(temp_input_path)
                    logger.info(f"Đã xóa file tạm: {temp_input_path}")
            
            # Gửi thông báo hoàn thành
            if websocket:
                await websocket.send_json({
                    "status": "completed",
                    "fire_detected": fire_detected,
                    "total_frames_processed": processed_frame_count,
                    "detections": detections
                })
            
            return True, None, {"fire_detected": fire_detected, "detections": detections}
            
        except Exception as e:
            logger.error(f"Lỗi trong quá trình xử lý streaming: {str(e)}")
            logger.exception(e)
            
            # Xóa file tạm nếu có
            if 'is_temp_file_created' in locals() and 'temp_input_path' in locals() and is_temp_file_created:
                if os.path.exists(temp_input_path):
                    try:
                        os.remove(temp_input_path)
                    except:
                        pass
            
            # Gửi thông báo lỗi
            if websocket:
                await websocket.send_json({
                    "status": "error",
                    "message": f"Lỗi: {str(e)}"
                })
                
            return False, None, {}
            
    async def process_video_streaming_websocket(self, video_url: str, websocket: WebSocket) -> Dict:
        """
        Xử lý video trực tiếp từ Cloudinary URL và stream kết quả qua WebSocket
        
        Args:
            video_url: URL video trên Cloudinary
            websocket: WebSocket để gửi kết quả realtime
            
        Returns:
            Dict: Thông tin về quá trình xử lý và kết quả
        """
        try:
            # Mở video từ URL Cloudinary
            video_url_str = str(video_url)
            cap = cv2.VideoCapture(video_url_str)
            if not cap.isOpened():
                logger.error(f"Không thể mở video từ URL: {video_url}")
                await websocket.send_json({"status": "error", "message": "Không thể mở video từ URL"})
                return {"success": False}
            
            # Lấy thông tin video
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Bắt đầu xử lý streaming video: {fps}fps, {width}x{height}, {frame_count} frames")
            await websocket.send_json({
                "status": "started", 
                "info": {
                    "fps": fps,
                    "width": width,
                    "height": height,
                    "frame_count": frame_count
                }
            })
            
            # Chuẩn bị lưu frame đã xử lý
            processed_frames = []
            detections = []
            fire_detected = False
            frame_idx = 0
            
            # Thiết lập tốc độ xử lý
            skip_frames = 2  # Có thể điều chỉnh để tăng hiệu suất
            process_delay = 1.0 / (fps / 2)  # Thời gian đợi giữa các frame để không quá tải
            
            # Xử lý từng frame
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Tính thời gian hiện tại trong video
                current_time = frame_idx / fps
                video_time_str = time.strftime("%H:%M:%S", time.gmtime(current_time))
                
                # Xử lý frame hiện tại (có thể bỏ qua một số frame để tăng tốc độ)
                current_fire_detected = False
                total_fire_area = 0.0
                fire_confidence = 0.0
                
                # Xử lý phát hiện đám cháy trên frame
                if frame_idx % skip_frames == 0:
                    if self.model:
                        # Sử dụng YOLO để phát hiện đám cháy
                        result = self.model.predict(frame, save=False, conf=0.5, verbose=False)[0]
                        boxes = result.boxes
                        segments = getattr(result, 'masks', None)
                        
                        # Xử lý mask (nếu có)
                        if segments is not None:
                            masks = segments.data.cpu().numpy()
                            for mask in masks:
                                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                                colored_mask = np.zeros_like(frame, dtype=np.uint8)
                                colored_mask[mask_resized > 0.5] = (255, 0, 0)
                                frame = cv2.addWeighted(frame, 1.0, colored_mask, 1, 0)
                        
                        # Xử lý các phát hiện
                        for det in boxes:
                            if int(det.cls[0].item()) == 0:  # Class 0 là lửa
                                x1, y1, x2, y2 = det.xyxy[0].cpu().numpy().astype(int)
                                conf = float(det.conf[0].item())
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                                area = ((x2 - x1) * (y2 - y1)) / (width * height) * 100
                                total_fire_area += area
                                fire_confidence = max(fire_confidence, conf)
                                current_fire_detected = True
                    else:
                        # Phương pháp dự phòng nếu không có model
                        current_fire_detected, fire_confidence = self._detect_fire_in_frame(frame)
                        if current_fire_detected:
                            # Phân tích vùng lửa
                            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                            lower_red1 = np.array([0, 120, 70])
                            upper_red1 = np.array([10, 255, 255])
                            lower_red2 = np.array([170, 120, 70])
                            upper_red2 = np.array([180, 255, 255])
                            
                            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                            mask = cv2.bitwise_or(mask1, mask2)
                            
                            # Tìm contours
                            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            for contour in contours:
                                if cv2.contourArea(contour) > 50:
                                    x, y, w, h = cv2.boundingRect(contour)
                                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                                    area = (w * h) / (width * height) * 100
                                    total_fire_area += area
                
                # Thêm timestamp lên frame
                cv2.putText(frame, video_time_str, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if current_fire_detected:
                    cv2.putText(frame, f"FIRE DETECTED - Area: {total_fire_area:.2f}%", (10, 70),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Lưu thông tin phát hiện
                    fire_detected = True
                    detections.append({
                        "frame": frame_idx,
                        "time": current_time,
                        "time_str": video_time_str,
                        "confidence": round(float(fire_confidence), 4),
                        "area": round(float(total_fire_area), 4)
                    })
                
                # Lưu frame đã xử lý
                processed_frames.append(frame.copy())
                
                # Gửi kết quả qua WebSocket
                # Chuyển đổi frame sang định dạng JPEG để giảm kích thước dữ liệu
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                
                # Gửi frame và thông tin
                await websocket.send_bytes(frame_bytes)
                await websocket.send_json({
                    "status": "processing",
                    "frame_info": {
                        "index": frame_idx,
                        "time": video_time_str,
                        "fire_detected": current_fire_detected,
                        "fire_area": round(float(total_fire_area), 4) if current_fire_detected else 0,
                        "confidence": round(float(fire_confidence), 4) if current_fire_detected else 0,
                        "progress": round((frame_idx / frame_count) * 100, 2) if frame_count > 0 else 0
                    }
                })
                
                # Đợi một chút để không quá tải WebSocket
                await asyncio.sleep(process_delay)
                
                frame_idx += 1
                
            # Đóng VideoCapture
            cap.release()
            
            # Thông báo hoàn thành xử lý frames
            await websocket.send_json({
                "status": "frames_completed",
                "message": "Hoàn thành xử lý từng frame",
                "fire_detected": fire_detected,
                "total_frames": frame_idx,
                "detections_count": len(detections)
            })
            
            # Tạo video từ các frame đã xử lý
            try:
                # Import các thư viện cần thiết
                import imageio
                
                # Thông báo đang tạo video
                await websocket.send_json({
                    "status": "creating_video",
                    "message": "Đang tạo video từ các frame đã xử lý"
                })
                
                # Tạo video trong bộ nhớ
                output_buffer = io.BytesIO()
                
                try:
                    # Tạo writer với các tham số phù hợp
                    writer = imageio.get_writer(
                        output_buffer, 
                        format='mp4', 
                        fps=fps, 
                        codec='h264',
                        pixelformat='yuv420p', 
                        macro_block_size=1,  # Giúp xử lý kích thước lẻ
                        ffmpeg_log_level='error',
                        quality=8
                    )
                    
                    # Ghi từng frame vào video
                    for i, frame in enumerate(processed_frames):
                        # Cập nhật tiến độ tạo video
                        if i % 20 == 0 and len(processed_frames) > 0:
                            progress = round((i / len(processed_frames)) * 100, 2)
                            await websocket.send_json({
                                "status": "creating_video",
                                "progress": progress
                            })
                        
                        # Đảm bảo kích thước frame là số chẵn (yêu cầu của một số codec)
                        h, w = frame.shape[:2]
                        if w % 2 == 1:
                            frame = frame[:, :-1]
                        if h % 2 == 1:
                            frame = frame[:-1, :]
                        
                        # Chuyển từ BGR (OpenCV) sang RGB (imageio)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        writer.append_data(frame_rgb)
                    
                    # Đóng writer
                    writer.close()
                    
                except Exception as e:
                    logger.error(f"Lỗi khi sử dụng imageio: {str(e)}")
                    raise e
                
                # Lấy dữ liệu video đã xử lý
                output_buffer.seek(0)
                processed_video_data = output_buffer.getvalue()
                
                # Thông báo đã tạo video thành công
                await websocket.send_json({
                    "status": "video_created",
                    "message": "Đã tạo video thành công, đang tải lên Cloudinary",
                    "video_size": len(processed_video_data)
                })
                
            except Exception as e:
                logger.error(f"Lỗi khi tạo video: {str(e)}")
                await websocket.send_json({
                    "status": "error",
                    "message": f"Lỗi khi tạo video: {str(e)}"
                })
                
                # Sử dụng phương thức dự phòng với OpenCV và file tạm
                try:
                    await websocket.send_json({
                        "status": "creating_video",
                        "message": "Đang sử dụng phương thức dự phòng để tạo video"
                    })
                    
                    # Tạo file tạm thời với tên duy nhất
                    temp_video_path = f"temp_video_{uuid.uuid4()}.mp4"
                    
                    # Tạo VideoWriter
                    fourcc = cv2.VideoWriter_fourcc(*'X264')
                    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
                    
                    # Ghi từng frame vào video
                    for i, frame in enumerate(processed_frames):
                        if i % 20 == 0 and len(processed_frames) > 0:
                            progress = round((i / len(processed_frames)) * 100, 2)
                            await websocket.send_json({
                                "status": "creating_video",
                                "progress": progress
                            })
                        out.write(frame)
                    
                    # Đóng VideoWriter
                    out.release()
                    
                    # Đọc file video vào bộ nhớ
                    with open(temp_video_path, 'rb') as f:
                        processed_video_data = f.read()
                    
                    # Xóa file tạm thời
                    if os.path.exists(temp_video_path):
                        os.remove(temp_video_path)
                    
                    # Thông báo đã tạo video thành công
                    await websocket.send_json({
                        "status": "video_created",
                        "message": "Đã tạo video thành công bằng phương thức dự phòng, đang tải lên Cloudinary",
                        "video_size": len(processed_video_data)
                    })
                except Exception as backup_error:
                    logger.error(f"Lỗi khi sử dụng phương thức dự phòng: {str(backup_error)}")
                    return {
                        "success": False,
                        "message": f"Không thể tạo video. Lỗi chính: {str(e)}. Lỗi dự phòng: {str(backup_error)}",
                        "fire_detected": fire_detected,
                        "detections": detections
                    }
            
            # Trả về thông tin để tải lên Cloudinary và cập nhật database
            return {
                "success": True,
                "video_data": processed_video_data,
                "fire_detected": fire_detected,
                "detections": detections,
                "info": {
                    "fps": fps,
                    "width": width,
                    "height": height,
                    "total_frames": frame_idx
                }
            }
            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý video streaming: {str(e)}")
            logger.exception(e)
            
            # Thông báo lỗi qua WebSocket
            try:
                await websocket.send_json({
                    "status": "error",
                    "message": f"Lỗi khi xử lý video: {str(e)}"
                })
            except:
                pass
                
            return {"success": False, "message": str(e)} 