import cv2
import numpy as np
import base64
import json
import asyncio
import logging
import os
import time
import traceback
from datetime import datetime
from typing import Dict, Optional, Any, Tuple, List

from fastapi import WebSocket, WebSocketDisconnect

from app.core.config import settings
from app.services.fire_detection import FireDetectionService

logger = logging.getLogger(__name__)

class CameraController:
    """
    Controller xử lý logic liên quan đến WebSocket camera cho phát hiện đám cháy trực tuyến
    """
    
    def __init__(self):
        """
        Khởi tạo controller camera với FireDetectionService
        """
        self.fire_detection_service = FireDetectionService()
        self.connected_clients = set()
        logger.info("Khởi tạo CameraController thành công")
    
    async def handle_camera_connection(self, websocket: WebSocket):
        """
        Xử lý kết nối WebSocket cho camera
        
        Args:
            websocket: Đối tượng WebSocket của client
        """
        await websocket.accept()
        self.connected_clients.add(websocket)
        logger.info(f"Client kết nối camera thành công, đang khởi động camera... (clients: {len(self.connected_clients)})")
        
        try:
            await self._process_camera_feed(websocket)
        except WebSocketDisconnect:
            logger.info("Client đã ngắt kết nối camera")
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Lỗi không mong muốn trong xử lý camera: {str(e)}")
            logger.error(f"Chi tiết lỗi: {error_details}")
        finally:
            self.connected_clients.remove(websocket)
            await websocket.close()
            logger.info(f"Đã đóng kết nối WebSocket camera (clients còn lại: {len(self.connected_clients)})")
    
    async def _process_camera_feed(self, websocket: WebSocket):
        """
        Xử lý luồng video từ camera
        
        Args:
            websocket: Đối tượng WebSocket của client
        """
        cap = None
        try:
            # Khởi tạo camera với backend mặc định
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            await asyncio.sleep(1)  # Thời gian khởi động camera
            
            if not cap.isOpened():
                logger.error("Không thể mở camera")
                await websocket.send_json({"status": "error", "message": "Không thể khởi động camera"})
                return

            logger.info("Camera đã sẵn sàng")
            await websocket.send_json({"status": "ready", "message": "Camera đã sẵn sàng"})
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_idx = 0
            
            # Khởi tạo bảng ghi nhớ về khung hình có cháy để giảm flicker
            fire_memory = []
            max_memory_frames = 5

            while True:
                # Kiểm tra lệnh dừng từ client
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                    command = json.loads(data) if isinstance(data, str) else data
                    
                    if isinstance(command, str) and command == "stop":
                        logger.info("Nhận lệnh dừng từ client")
                        break
                    elif isinstance(command, dict):
                        if command.get("action") == "stop":
                            logger.info("Nhận lệnh dừng từ client")
                            break
                except asyncio.TimeoutError:
                    pass  # Không có message, tiếp tục xử lý
                
                # Đọc frame từ camera
                ret, frame = cap.read()
                if not ret:
                    logger.error("Không thể đọc frame từ camera")
                    await websocket.send_json({"status": "error", "message": "Lỗi đọc dữ liệu từ camera"})
                    break

                # Xử lý frame
                processed_frame, frame_info = await self._process_frame(
                    frame, frame_idx, width, height, fire_memory
                )
                
                # Cập nhật bộ nhớ cháy
                if frame_info["fire_detected"]:
                    fire_memory.append(frame_info)
                    if len(fire_memory) > max_memory_frames:
                        fire_memory.pop(0)
                elif len(fire_memory) > 0:
                    fire_memory.pop(0) if len(fire_memory) > 0 else None
                
                # Gửi frame và thông tin phát hiện
                try:
                    await websocket.send_text(json.dumps(frame_info))
                except Exception as e:
                    logger.error(f"Lỗi khi gửi dữ liệu: {str(e)}")
                    break

                frame_idx += 1
                
        except Exception as e:
            logger.error(f"Lỗi trong quá trình xử lý camera: {str(e)}")
            await websocket.send_json({"status": "error", "message": f"Lỗi xử lý: {str(e)}"})
        finally:
            if cap is not None and cap.isOpened():
                cap.release()
                logger.info("Đã giải phóng camera")
    
    async def _process_frame(self, frame: np.ndarray, frame_idx: int, width: int, height: int, 
                            fire_memory: List[Dict[str, Any]]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Xử lý một frame từ camera để phát hiện đám cháy
        
        Args:
            frame: Khung hình cần xử lý
            frame_idx: Chỉ số của frame
            width: Chiều rộng frame
            height: Chiều cao frame
            fire_memory: Bộ nhớ các frame có cháy
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Frame đã xử lý và thông tin kèm theo
        """
        try:
            # Đảo ngược hình ảnh camera để phù hợp hiển thị
            frame = cv2.flip(frame, 1)
            current_time = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
            
            # Phát hiện đám cháy bằng mô hình
            results = self.fire_detection_service.model.predict(
                frame, save=False, save_txt=False, conf=0.5, verbose=False
            )[0]
            
            detections = results.boxes
            segments = getattr(results, 'masks', None)
            
            # Xử lý phân đoạn nếu có
            if segments is not None:
                masks = segments.data.cpu().numpy()
                for mask in masks:
                    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    colored_mask = np.zeros_like(frame, dtype=np.uint8)
                    colored_mask[mask_resized > 0.5] = (255, 0, 0)
                    frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.5, 0)
            
            # Phân tích các phát hiện
            total_fire_area = 0.0
            fire_detected = False
            confidences = []
            
            # Vẽ bounding box và tính toán diện tích
            for det in detections:
                if int(det.cls[0].item()) == 0:  # Class 0 là đám cháy
                    x1, y1, x2, y2 = det.xyxy[0].cpu().numpy().astype(int)
                    conf = float(det.conf[0].item())
                    confidences.append(conf)
                    
                    # Vẽ bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 4)
                    label = f"{conf:.2f}"
                    cv2.putText(frame, label, (x1 + 3, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Tính diện tích
                    area = ((x2 - x1) * (y2 - y1)) / (width * height) * 100
                    total_fire_area += area
                    fire_detected = True
            
            # Hiển thị thời gian
            cv2.putText(frame, current_time, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Hiển thị cảnh báo cháy nếu phát hiện
            if fire_detected:
                warning_text = "CẢNH BÁO CHÁY!"
                cv2.putText(frame, warning_text, (width // 2 - 120, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # Mã hóa frame thành base64 để gửi qua WebSocket
            _, buffer = cv2.imencode('.jpg', frame)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Tính trung bình confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Tạo thông tin frame để gửi
            frame_info = {
                "frame_idx": frame_idx,
                "time": current_time,
                "fire_detected": fire_detected,
                "total_area": round(total_fire_area, 4),
                "confidence": round(avg_confidence, 4),
                "frame": frame_b64
            }
            
            return frame, frame_info
            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý frame: {str(e)}")
            # Trả về frame gốc và thông tin cơ bản nếu có lỗi
            _, buffer = cv2.imencode('.jpg', frame)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            return frame, {
                "frame_idx": frame_idx,
                "time": datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
                "fire_detected": False,
                "total_area": 0.0,
                "confidence": 0.0,
                "frame": frame_b64,
                "error": str(e)
            } 