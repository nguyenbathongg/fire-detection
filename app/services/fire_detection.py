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
from fastapi import WebSocket

from app.core.config import settings
from app.utils.cloudinary_service import upload_bytes_to_cloudinary, download_from_cloudinary

logger = logging.getLogger(__name__)


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
        try:
            # Đo thời gian xử lý
            total_start_time = time.time()
            
            # Tạo file tạm thời để đọc video
            temp_input_path = f"temp_input_{uuid.uuid4()}.mp4"
            try:
                # Đo thời gian ghi file
                file_write_start = time.time()
                with open(temp_input_path, 'wb') as f:
                    f.write(video_data)
                logger.info(f"Thời gian ghi file tạm: {time.time() - file_write_start:.2f} giây")
                
                # Đo thời gian mở video
                open_start = time.time()
                cap = cv2.VideoCapture(temp_input_path)
                if not cap.isOpened():
                    logger.error(f"Không thể mở video từ file tạm: {temp_input_path}")
                    return False, [], None
                logger.info(f"Thời gian mở video: {time.time() - open_start:.2f} giây")
                
                # Ghi nhớ để xóa file tạm sau khi xử lý xong
                is_temp_file_created = True
            except Exception as e:
                logger.error(f"Không thể tạo file tạm thời để đọc video: {str(e)}")
                return False, [], None
            
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
                    # Fallback sử dụng phân tích màu sắc nếu không có model
                    is_fire_in_frame, confidence = self.detect_fire_with_color_analysis(detection_frame)
                
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

    def _detect_fire_in_frame(self, frame) -> Tuple[bool, float]:
        """
        Phát hiện đám cháy trong một frame bằng cách phân tích màu sắc
        (Phương pháp dự phòng khi không có model YOLO - Legacy method)
        
        Deprecated: Use detect_fire_with_color_analysis instead
        
        Args:
            frame: Frame hình ảnh cần phân tích
            
        Returns:
            Tuple[bool, float]: Có phát hiện đám cháy không và mức độ tin cậy
        """
        # Chuyển đổi frame sang không gian màu HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Định nghĩa các ngưỡng màu đặc trưng cho lửa
        # Đám cháy thường có màu đỏ-cam-vàng trong không gian HSV
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        
        # Tạo mask cho vùng màu lửa
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Tính toán tỷ lệ vùng lửa so với toàn bộ hình ảnh
        fire_pixel_count = cv2.countNonZero(mask)
        total_pixel_count = frame.shape[0] * frame.shape[1]
        fire_ratio = fire_pixel_count / total_pixel_count
        
        # Áp dụng ngưỡng để xác định có đám cháy hay không
        # Ngưỡng 0.005 (0.5%) - có thể điều chỉnh tùy theo độ nhạy mong muốn
        threshold = 0.005
        is_fire = fire_ratio > threshold
        
        # Chuẩn hóa confidence về khoảng [0, 1]
        confidence = min(1.0, fire_ratio * 10)
        
        return is_fire, confidence

    def process_video_from_memory(self, video_data: bytes) -> Tuple[bool, bytes, Dict]:
        """
        Xử lý video để đánh dấu các vùng phát hiện đám cháy,
        không lưu file tạm, xử lý trong bộ nhớ
        
        Args:
            video_data: Dữ liệu nhị phân của video
            
        Returns:
            Tuple[bool, bytes, Dict]: 
                - Boolean cho biết có xử lý thành công không
                - Dữ liệu nhị phân của video đã xử lý
                - Thông tin phát hiện
        """
        # Đo thời gian xử lý
        total_start_time = time.time()
        
        try:
            # Phương án dự phòng: tạo file tạm thời để đọc video
            temp_input_path = f"temp_input_{uuid.uuid4()}.mp4"
            try:
                # Đo thời gian ghi file
                file_write_start = time.time()
                with open(temp_input_path, 'wb') as f:
                    f.write(video_data)
                logger.info(f"Thời gian ghi file tạm: {time.time() - file_write_start:.2f} giây")
                
                # Đo thời gian mở video
                open_start = time.time()
                cap = cv2.VideoCapture(temp_input_path)
                if not cap.isOpened():
                    logger.error(f"Không thể mở video từ file tạm: {temp_input_path}")
                    return False, None, {}
                logger.info(f"Thời gian mở video: {time.time() - open_start:.2f} giây")
                
                # Ghi nhớ để xóa file tạm sau khi xử lý xong
                is_temp_file_created = True
            except Exception as e:
                logger.error(f"Không thể tạo file tạm thời để đọc video: {str(e)}")
                return False, None, {}
            
            # Xử lý video với phát hiện đám cháy
            skip_frames = 3  # Giảm số frame bỏ qua để tránh hiệu ứng nhấp nháy
            
            # Lưu thời gian bắt đầu xử lý
            processing_start_time = time.time()
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_idx = 0
            
            # Lưu các frame đã xử lý
            processed_frames = []
            
            detections = []
            fire_detected = False
            
            logger.info(f"Bắt đầu xử lý video từ bộ nhớ")
            logger.info(f"Video params: fps={fps}, size={width}x{height}")
            
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
                    # Resize frame để tăng tốc độ
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
                    
                    if self.model:
                        # Xử lý các khung hình cần thiết với YOLO
                        result = self.model.predict(
                            detection_frame, 
                            save=False, 
                            conf=0.7,  # Tăng ngưỡng confidence lên để giảm nhận diện sai
                            verbose=False,
                            device=self.device if self.use_gpu else 'cpu',
                            half=self.use_gpu  # Sử dụng FP16 nếu có GPU
                        )[0]
                        boxes = result.boxes
                        segments = getattr(result, 'masks', None)
                        
                        # Xử lý mask (nếu có)
                        if segments is not None:
                            masks = segments.data.cpu().numpy()
                            for i, mask in enumerate(masks):
                                # Chỉ xử lý mask của class lửa (class 0)
                                cls_id = int(boxes[i].cls[0].item()) if i < len(boxes) else -1
                                conf = float(boxes[i].conf[0].item()) if i < len(boxes) else 0
                                
                                if cls_id == 0 and conf >= 0.6:  # Lửa (class 0)
                                    # Resize mask về kích thước gốc và tạo mask nhị phân
                                    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                                    mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                                    
                                    # Tạo colored mask và vẽ lên frame
                                    colored_mask = np.zeros_like(frame, dtype=np.uint8)
                                    colored_mask[mask_resized > 0.5] = (0, 0, 255)  # Màu đỏ cho vùng lửa
                                    frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.5, 0)
                                    
                                    # Tìm các contour trong mask
                                    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    
                                    # Vẽ bounding box xung quanh các contour
                                    for contour in contours:
                                        if cv2.contourArea(contour) > 100:  # Lọc vùng quá nhỏ
                                            x, y, w, h = cv2.boundingRect(contour)
                                            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                                            cv2.putText(frame, f"{conf:.2f}", (x, y-10), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                                            
                                            # Tính diện tích lửa
                                            fire_area = (w * h) / (width * height) * 100
                                            total_fire_area += fire_area
                                            current_fire_detected = True
                        
                        # Xử lý bounding boxes cho lửa
                        # Lọc các class và confidence
                        for det in boxes:
                            cls_id = int(det.cls[0].item())
                            conf = float(det.conf[0].item())
                            
                            # Xác định nếu đây là lửa (class 0) hoặc khói (class 1 nếu có)
                            # Chỉ lấy các phát hiện có độ tin cậy cao
                            if cls_id == 0 and conf >= 0.7:  # Class 0 là lửa - tăng ngưỡng lên 0.7
                                x1, y1, x2, y2 = det.xyxy[0].cpu().numpy().astype(int)
                                
                                # Xác minh thêm bằng cách kiểm tra màu sắc trong vùng bbox
                                roi = frame[max(0, y1):min(y2, height-1), max(0, x1):min(x2, width-1)]
                                if roi.size > 0:  # Đảm bảo ROI có kích thước hợp lệ
                                    # Chuyển đổi sang HSV và kiểm tra vùng màu của lửa
                                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                                    # Định nghĩa range màu lửa trong HSV
                                    lower_fire1 = np.array([0, 120, 100])
                                    upper_fire1 = np.array([10, 255, 255])
                                    lower_fire2 = np.array([170, 120, 100])
                                    upper_fire2 = np.array([180, 255, 255])
                                    
                                    mask1 = cv2.inRange(hsv, lower_fire1, upper_fire1)
                                    mask2 = cv2.inRange(hsv, lower_fire2, upper_fire2)
                                    fire_mask = cv2.bitwise_or(mask1, mask2)
                                    
                                    # Tính % pixel lửa trong bbox
                                    fire_pixel_pct = np.count_nonzero(fire_mask) / fire_mask.size * 100
                                    
                                    # Chỉ vẽ nếu có đủ pixel lửa trong bbox
                                    if fire_pixel_pct > 5:  # Yêu cầu ít nhất 5% pixel trong bbox là lửa
                                        # Vẽ bbox
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                        cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                                        area = ((x2 - x1) * (y2 - y1)) / (width * height) * 100
                                        total_fire_area += area
                                        current_fire_detected = True
                    else:
                        # Phương pháp dự phòng sử dụng phân tích màu sắc
                        # Phân tích màu sắc để phát hiện lửa - sử dụng ngưỡng cao hơn
                        current_fire_detected, confidence = self.detect_fire_with_color_analysis(detection_frame)
                        
                        if current_fire_detected and confidence > 5.0:  # Tăng ngưỡng để giảm false positive
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
                            "confidence": round(float(0.8), 4),  # Giá trị mặc định cho phương pháp dự phòng
                            "area": round(float(total_fire_area), 4)
                        })



                
                # Thêm timestamp và thông tin phát hiện
                cv2.putText(frame, video_time_str, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                           
                if current_fire_detected:
                    cv2.putText(frame, f"FIRE DETECTED - Area: {total_fire_area:.2f}%", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Lưu frame đã xử lý vào mảng
                processed_frames.append(frame)
                frame_idx += 1
            
            # Giải phóng resources
            cap.release()
            
            # Xóa file tạm thời nếu có
            if 'is_temp_file_created' in locals() and is_temp_file_created:
                if os.path.exists(temp_input_path):
                    os.remove(temp_input_path)
                    logger.info(f"Đã xóa file tạm thời: {temp_input_path}")
            
            # Ghi video đã xử lý vào bộ nhớ sử dụng imageio
            processed_video_data = None
            try:
                import imageio
                # Mở buffer cho video đầu ra
                output_buffer = io.BytesIO()
                
                # Đảm bảo kích thước frame là số chẵn (yêu cầu của một số codec)
                for i in range(len(processed_frames)):
                    h, w = processed_frames[i].shape[:2]
                    if w % 2 == 1:
                        processed_frames[i] = processed_frames[i][:, :-1]
                    if h % 2 == 1:
                        processed_frames[i] = processed_frames[i][:-1, :]
                
                # Lấy kích thước mới (có thể đã thay đổi)
                if processed_frames:
                    height, width = processed_frames[0].shape[:2]
                
                # Tạo writer với các tham số phù hợp
                with imageio.get_writer(output_buffer, format='mp4', fps=fps, 
                                       codec='h264', quality=8, macro_block_size=1,
                                       ffmpeg_log_level='error') as writer:
                    # Lưu từng frame vào video
                    for frame in processed_frames:
                        # Chuyển từ BGR (OpenCV) sang RGB (imageio)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        writer.append_data(frame_rgb)
                
                # Lấy dữ liệu video
                output_buffer.seek(0)
                processed_video_data = output_buffer.getvalue()
                
            except Exception as e:
                logger.error(f"Lỗi khi sử dụng imageio: {str(e)}")
                
                # Sử dụng OpenCV để tạo video tạm và đọc lại vào bộ nhớ
                try:
                    logger.info("Sử dụng phương án dự phòng với OpenCV")
                    # Tạo file tạm thời với tên duy nhất
                    temp_video_path = f"temp_video_{uuid.uuid4()}.mp4"
                    
                    # Tạo VideoWriter
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
                    
                    # Ghi từng frame vào video
                    for frame in processed_frames:
                        out.write(frame)
                    
                    # Đóng VideoWriter
                    out.release()
                    
                    # Đọc file video vào bộ nhớ
                    with open(temp_video_path, 'rb') as f:
                        processed_video_data = f.read()
                    
                    # Xóa file tạm thời
                    if os.path.exists(temp_video_path):
                        os.remove(temp_video_path)
                    
                except Exception as backup_error:
                    logger.error(f"Không thể tạo video với phương án dự phòng: {str(backup_error)}")
                    return False, None, {}
            
            if not processed_video_data:
                logger.error("Không thể tạo video, dữ liệu rỗng")
                return False, None, {}
            
            logger.info(f"Đã xử lý xong video trong bộ nhớ, kích thước: {len(processed_video_data)} bytes")
            
            detection_info = {
                "fire_detected": fire_detected,
                "detections": detections,
                "frames_processed": frame_idx,
                "fps": fps,
                "resolution": f"{width}x{height}"
            }
            
            return True, processed_video_data, detection_info
            
        except Exception as e:
            logger.error(f"Error processing video from memory: {str(e)}")
            logger.exception(e)
            return False, None, {}
            
    async def process_video_streaming_from_memory(self, video_data: bytes, websocket: WebSocket = None) -> Tuple[bool, bytes, Dict]:
        """
        Xử lý video để phát hiện đám cháy và gửi frame đã xử lý qua WebSocket
        
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
                    logger.warning("Không thể tải model YOLO, sẽ sử dụng phân tích màu sắc")
            
            # Boặc phần còn lại của code
            
            # Tạo file tạm thời để đọc video
            temp_input_path = f"temp_input_{uuid.uuid4()}.mp4"
            try:
                with open(temp_input_path, 'wb') as f:
                    f.write(video_data)
                
                cap = cv2.VideoCapture(temp_input_path)
                if not cap.isOpened():
                    logger.error(f"Không thể mở video từ file tạm: {temp_input_path}")
                    if websocket:
                        await websocket.send_json({"status": "error", "message": "Không thể mở video"})
                    return False, None, {}
                
                # Ghi nhớ để xóa file tạm sau khi xử lý xong
                is_temp_file_created = True
            except Exception as e:
                logger.error(f"Không thể tạo file tạm: {str(e)}")
                if websocket:
                    await websocket.send_json({"status": "error", "message": f"Lỗi: {str(e)}"})
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
                    else:
                        # Phương pháp dự phòng sử dụng phân tích màu sắc
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
                        break
                
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
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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