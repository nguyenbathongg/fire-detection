import cv2
import numpy as np
import base64
import json
from datetime import datetime
from ultralytics import YOLO
import torch
import logging
import os
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FireDetection")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
model_path = os.path.join(BASE_DIR, 'model', 'bestyolov11-27k.pt')
logger.info(f"Đường dẫn model: {model_path}")

model = YOLO(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
logger.info(f"Đang sử dụng thiết bị: {device}")

ws_router = APIRouter()

@ws_router.websocket("/fire")
async def websocket_fire_detection(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client kết nối thành công, đang khởi động camera...")
    
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        await asyncio.sleep(1)
        if not cap.isOpened():
            logger.error("Không thể mở camera")
            await websocket.send_json({"status": "error", "message": "Không thể khởi động camera"})
            await websocket.close()
            return

        logger.info("Camera đã sẵn sàng")
        await websocket.send_json({"status": "ready", "message": "Camera đã sẵn sàng"})
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_idx = 0

        while True:
            # Kiểm tra lệnh dừng từ client
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                if data == "stop":
                    logger.info("Nhận lệnh dừng từ client")
                    break
            except asyncio.TimeoutError:
                pass  # Không có message, tiếp tục xử lý
            
            ret, frame = cap.read()
            if not ret:
                logger.error("Không thể đọc frame từ camera")
                break

            frame = cv2.flip(frame, 1)
            current_time = datetime.now().strftime("%H:%M:%S %d/%m/%Y")

            results = model.predict(frame, save=False, save_txt=False, conf=0.5, verbose=False)[0]
            detections = results.boxes
            segments = getattr(results, 'masks', None)

            if segments is not None:
                masks = segments.data.cpu().numpy()
                for mask in masks:
                    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    colored_mask = np.zeros_like(frame, dtype=np.uint8)
                    colored_mask[mask_resized > 0.5] = (255, 0, 0)
                    frame = cv2.addWeighted(frame, 1.0, colored_mask, 2, 0)

            total_fire_area = 0.0
            fire_detected = False

            for det in detections:
                if int(det.cls[0].item()) == 0:
                    x1, y1, x2, y2 = det.xyxy[0].cpu().numpy().astype(int)
                    conf = float(det.conf[0].item())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 4)
                    label = f"{conf:.2f}"
                    cv2.putText(frame, label, (x1 + 3, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    area = ((x2 - x1) * (y2 - y1)) / (width * height) * 100
                    total_fire_area += area
                    fire_detected = True

            _, buffer = cv2.imencode('.jpg', frame)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')

            data = {
                "frame_idx": frame_idx,
                "time": current_time,
                "fire_detected": fire_detected,
                "total_area": round(total_fire_area, 4),
                "frame": frame_b64
            }

            try:
                await websocket.send_text(json.dumps(data))
            except Exception as e:
                logger.error(f"Lỗi khi gửi dữ liệu: {str(e)}")
                break

            frame_idx += 1

    except WebSocketDisconnect:
        logger.info("Client đã ngắt kết nối")
    except Exception as e:
        logger.error(f"Lỗi không mong muốn: {str(e)}")
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
            logger.info("Đã giải phóng camera")
        await websocket.close()
        logger.info("Đã đóng kết nối WebSocket")