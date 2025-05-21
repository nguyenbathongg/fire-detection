import json
import asyncio
import uuid
import logging
from typing import Dict, List, Any, Optional
import traceback

from fastapi import WebSocket, WebSocketDisconnect, status
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.models import Video, FireDetection, UserHistory
from app.models.enums import StatusEnum
from app.services.fire_detection import FireDetectionService
from app.utils.cloudinary_service import upload_bytes_to_cloudinary, download_from_cloudinary
from app.utils.video import download_youtube_video
from app.controllers.user_history_controller import UserHistoryController


logger = logging.getLogger(__name__)


class WebSocketController:
    """
    Controller xử lý logic liên quan đến WebSocket cho xử lý video
    """
    
    fire_service = FireDetectionService()
    
    @staticmethod
    async def process_video_by_id(websocket: WebSocket, video_id: uuid.UUID, db: Session):
        """
        Xử lý video từ ID thông qua WebSocket
        """
        try:
            await websocket.accept()
            
            # Lấy thông tin video
            video = db.query(Video).filter(Video.video_id == video_id).first()
            if not video:
                await websocket.send_json({
                    "status": "error",
                    "message": "Không tìm thấy video",
                })
                await websocket.close()
                return
            
            # Gửi thông báo bắt đầu
            await websocket.send_json({
                "status": "processing",
                "message": "Bắt đầu xử lý video...",
                "progress": 0
            })
            
            # Cập nhật trạng thái video
            video.status = StatusEnum.PROCESSING
            db.commit()
            
            try:
                # Tải xuống video từ Cloudinary URL vào bộ nhớ
                await websocket.send_json({
                    "status": "processing",
                    "message": "Đang tải xuống video...",
                    "progress": 10
                })
                
                success, message, video_data = download_from_cloudinary(video.video_url)
                
                if not success or not video_data:
                    logger.error(f"Không thể tải xuống video: {message}")
                    raise Exception(f"Không thể tải xuống video: {message}")
                
                # Phát hiện đám cháy
                await websocket.send_json({
                    "status": "processing",
                    "message": "Đang phân tích video tìm đám cháy...",
                    "progress": 30
                })
                
                # Tải model nếu cần
                if not WebSocketController.fire_service.model:
                    await websocket.send_json({
                        "status": "processing",
                        "message": "Đang tải model AI...",
                        "progress": 35
                    })
                    if not WebSocketController.fire_service.load_model():
                        raise Exception("Không thể tải model YOLO")
                
                # Phát hiện đám cháy
                fire_detected, detections, max_fire_frame = WebSocketController.fire_service.detect_fire_from_memory(video_data)
                
                # Gửi kết quả phát hiện
                await websocket.send_json({
                    "status": "processing",
                    "message": "Đã phát hiện đám cháy, đang xử lý video...",
                    "progress": 60,
                    "fire_detected": fire_detected,
                    "detections_count": len(detections)
                })
                
                # Xử lý video với vùng phát hiện
                success, processed_video_data, detection_info = WebSocketController.fire_service.process_video_from_memory(video_data)
                
                if not success or not processed_video_data:
                    raise Exception("Không thể xử lý video")
                
                # Tải video đã xử lý lên Cloudinary
                await websocket.send_json({
                    "status": "processing",
                    "message": "Đang tải video đã xử lý lên cloud...",
                    "progress": 80
                })
                
                processed_filename = f"processed_{uuid.uuid4()}.mp4"
                upload_success, upload_message, result = upload_bytes_to_cloudinary(
                    processed_video_data, 
                    filename=processed_filename
                )
                
                if upload_success:
                    processed_video_url = result.get("secure_url")
                    cloudinary_processed_id = result.get("public_id")
                    logger.info(f"Đã tải video đã xử lý lên Cloudinary: {processed_video_url}")
                else:
                    processed_video_url = None
                    cloudinary_processed_id = None
                    logger.error(f"Lỗi khi tải video đã xử lý lên Cloudinary: {upload_message}")
                    raise Exception(f"Lỗi khi tải video đã xử lý lên Cloudinary: {upload_message}")
                
                # Cập nhật database với detections và kết quả xử lý
                has_fire = bool(detections and len(detections) > 0)
                video.fire_detected = has_fire
                video.status = StatusEnum.COMPLETED
                video.processed_video_url = processed_video_url
                video.cloudinary_processed_id = cloudinary_processed_id
                db.commit()
                
                # Lưu các phát hiện vào cơ sở dữ liệu
                for detection in detections:
                    # Nếu có max fire frame, tải lên Cloudinary
                    frame_cloudinary_url = None
                    if max_fire_frame:
                        frame_data, frame_ext = max_fire_frame
                        frame_filename = f"fire_frame_{uuid.uuid4()}{frame_ext}"
                        
                        success, _, img_result = upload_bytes_to_cloudinary(
                            frame_data, 
                            filename=frame_filename,
                            resource_type="image"
                        )
                        
                        if success:
                            frame_cloudinary_url = img_result.get("secure_url")
                    
                    fire_detection = FireDetection(
                        detection_id=uuid.uuid4(),
                        video_id=video_id,
                        fire_start_time=detection.get("fire_start_time"),
                        fire_end_time=detection.get("fire_end_time"),
                        confidence=detection.get("confidence"),
                        max_fire_frame=detection.get("max_fire_frame"),
                        max_fire_frame_image_path=frame_cloudinary_url
                    )
                    db.add(fire_detection)
                db.commit()
                
                # Gửi kết quả cuối cùng
                await websocket.send_json({
                    "status": "completed",
                    "message": "Xử lý video hoàn tất",
                    "progress": 100,
                    "fire_detected": has_fire,
                    "detections_count": len(detections),
                    "video_url": video.video_url,
                    "processed_video_url": processed_video_url
                })
                
                # Thêm lịch sử
                if video.user_id:
                    UserHistoryController.add_history(
                        db=db,
                        user_id=video.user_id,
                        action_type="process_video",
                        description=f"Xử lý video thành công: {video.title or video.video_id}"
                    )
            
            except Exception as e:
                logger.error(f"Lỗi khi xử lý video: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Cập nhật trạng thái video thành lỗi
                video.status = StatusEnum.ERROR
                db.commit()
                
                # Gửi thông báo lỗi
                await websocket.send_json({
                    "status": "error",
                    "message": f"Lỗi khi xử lý video: {str(e)}",
                })
                
                # Thêm lịch sử
                if video.user_id:
                    UserHistoryController.add_history(
                        db=db,
                        user_id=video.user_id,
                        action_type="process_video_error",
                        description=f"Lỗi khi xử lý video: {str(e)}"
                    )
        
        except WebSocketDisconnect:
            logger.warning(f"WebSocket bị đóng kết nối trong quá trình xử lý video {video_id}")
        except Exception as e:
            logger.error(f"Lỗi không xác định khi xử lý WebSocket: {str(e)}")
            logger.error(traceback.format_exc())
            try:
                await websocket.send_json({
                    "status": "error",
                    "message": f"Lỗi không xác định: {str(e)}",
                })
            except:
                pass 