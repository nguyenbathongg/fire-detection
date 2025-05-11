import os
import uuid
from typing import Any, List, Optional
import asyncio
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks, status, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from sqlalchemy import desc
import requests
import io
import logging
import traceback
from datetime import datetime

from app.db.base import get_db
from app.api.deps import get_current_active_user, get_current_active_admin
from app.models import User, Video, FireDetection, UserHistory, Notification
from app.schemas import Video as VideoSchema, VideoWithDetections, VideoCreate, VideoUpload
from app.models.enums import VideoTypeEnum, StatusEnum
from app.utils.video import save_upload_file, download_youtube_video
from app.services.fire_detection import FireDetectionService
from app.utils.cloudinary_service import upload_bytes_to_cloudinary, download_from_cloudinary, delete_from_cloudinary
from app.utils.email_service import send_fire_detection_notification
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()
fire_service = FireDetectionService()


async def process_video_task(db: Session, video_id: uuid.UUID, video_url: str):
    """
    Task nền để xử lý video
    
    Args:
        db: Session database
        video_id: ID của video
        video_url: URL của video trên Cloudinary
    """
    try:
        # Cập nhật trạng thái video
        video = db.query(Video).filter(Video.video_id == video_id).first()
        if not video:
            return
        
        video.status = StatusEnum.PROCESSING
        db.commit()
        
        try:
            # Tải xuống video từ Cloudinary URL vào bộ nhớ
            logger.info(f"Tải xuống video từ Cloudinary: {video_url}")
            success, message, video_data = download_from_cloudinary(video_url)
            
            if not success or not video_data:
                logger.error(f"Không thể tải xuống video: {message}")
                raise Exception(f"Không thể tải xuống video: {message}")
            
            # Phát hiện đám cháy trực tiếp từ dữ liệu nhị phân
            logger.info(f"Bắt đầu phát hiện đám cháy")
            
            # Thử phát hiện đám cháy với số lần thử tối đa
            max_retries = 3
            current_retry = 0
            
            while current_retry < max_retries:
                if not fire_service.model:
                    logger.warning(f"Model YOLO chưa được tải. Đang thử tải lại model (lần {current_retry + 1}/{max_retries}).")
                    if fire_service.load_model():
                        logger.info("Đã tải lại model YOLO thành công")
                    else:
                        logger.error("Không thể tải lại model YOLO")
                        current_retry += 1
                        if current_retry >= max_retries:
                            raise Exception("Không thể tải model YOLO sau nhiều lần thử, không thể tiếp tục xử lý")
                        continue
                
                try:
                    # Thử phát hiện đám cháy
                    fire_detected, detections, max_fire_frame = fire_service.detect_fire_from_memory(video_data)
                    break  # Nếu không có lỗi, thoát khỏi vòng lặp
                except Exception as e:
                    logger.error(f"Lỗi khi phát hiện đám cháy: {str(e)}")
                    current_retry += 1
                    if current_retry >= max_retries:
                        raise Exception(f"Không thể phát hiện đám cháy sau nhiều lần thử: {str(e)}")
            
            # Xử lý video (đánh dấu vùng phát hiện đám cháy)
            logger.info(f"Xử lý video với vùng phát hiện đám cháy trong bộ nhớ")
            success, processed_video_data, detection_info = fire_service.process_video_from_memory(video_data)
            
            if not success or not processed_video_data:
                logger.error("Không thể xử lý video")
                raise Exception("Không thể xử lý video")
            
            # Tải video đã xử lý lên Cloudinary
            processed_filename = f"processed_{uuid.uuid4()}.mp4"
            logger.info(f"Tải video đã xử lý lên Cloudinary")
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
            
            # Cập nhật database với detections và kết quả xử lý
            # fire_detected = False (nếu không có detection nào), True (nếu có ít nhất 1 detection)
            has_fire = bool(detections and len(detections) > 0)
            video.fire_detected = has_fire
            video.status = StatusEnum.COMPLETED
            video.processed_video_url = processed_video_url
            video.cloudinary_processed_id = cloudinary_processed_id
            db.commit()
            
            # Nếu phát hiện đám cháy, chỉ gửi email nếu user bật nhận cảnh báo qua email
            if has_fire and video.user_id:
                user = db.query(User).filter(User.user_id == video.user_id).first()
                if user:
                    # Lấy notification_settings gần nhất
                    notification_settings = db.query(Notification).filter(
                        Notification.user_id == video.user_id
                    ).order_by(desc(Notification.created_at)).first()
                    logger.info(f"[DEBUG] has_fire={has_fire}, user.email={user.email}, detections={detections}")
                    if notification_settings:
                        logger.info(f"[DEBUG] notification_settings.enable_email_notification={notification_settings.enable_email_notification}")
                    else:
                        logger.info(f"[DEBUG] notification_settings is None")
                    if notification_settings and notification_settings.enable_email_notification:
                        try:
                            # Lấy tiêu đề video nếu có, nếu không thì "Video ID: ..."
                            video_title = video.title if hasattr(video, 'title') and video.title else f"Video ID: {video_id}"
                            # Lấy thời gian phát hiện đầu tiên trong các detection
                            if detections and len(detections) > 0 and 'time' in detections[0]:
                                detection_time = detections[0]['time']
                            else:
                                detection_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                            # Lấy thông tin độ tin cậy cao nhất
                            max_confidence = max([d.get("confidence", 0) for d in detections]) if detections else 0.8
                            send_fire_detection_notification(
                                user_email=user.email,
                                video_title=video_title,
                                detection_time=detection_time,
                                video_url=video_url,
                                processed_video_url=processed_video_url,
                                confidence=max_confidence
                            )
                            logger.info(f"Đã gửi email thông báo đám cháy đến {user.email} cho video {video_id}")
                            # Tạo notification trong hệ thống
                            notification = Notification(
                                notification_id=uuid.uuid4(),
                                user_id=video.user_id,
                                video_id=video_id,
                                title="Phát hiện đám cháy trong video của bạn",
                                message=f"Hệ thống đã phát hiện đám cháy trong video của bạn với độ tin cậy {max_confidence:.2%}.",
                                enable_email_notification=True,
                                enable_website_notification=True
                            )
                            db.add(notification)
                            db.commit()
                        except Exception as e:
                            logger.error(f"Lỗi khi gửi email thông báo: {str(e)}")
                    else:
                        logger.info(f"[DEBUG] Không gửi email vì notification_settings không tồn tại hoặc không bật enable_email_notification")
            
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
                        logger.info(f"Đã tải frame đám cháy lên Cloudinary: {frame_cloudinary_url}")
                
                fire_detection = FireDetection(
                    detection_id=uuid.uuid4(),
                    video_id=video_id,
                    fire_start_time=detection["fire_start_time"],
                    fire_end_time=detection["fire_end_time"],
                    confidence=detection["confidence"],
                    max_fire_frame=detection["max_fire_frame"],
                    max_fire_frame_image_path=frame_cloudinary_url
                )
                db.add(fire_detection)
            
            # Thêm lịch sử
            user_history = UserHistory(
                history_id=uuid.uuid4(),
                user_id=video.user_id,
                action_type="process_video",
                video_id=video_id,
                description=f"Xử lý video thành công. Phát hiện đám cháy: {fire_detected}"
            )
            db.add(user_history)
            db.commit()
        
        except Exception as e:
            # Cập nhật trạng thái lỗi
            video = db.query(Video).filter(Video.video_id == video_id).first()
            if video:
                video.status = StatusEnum.FAILED
                db.commit()
            
            # Thêm lịch sử
            user_history = UserHistory(
                history_id=uuid.uuid4(),
                user_id=video.user_id,
                action_type="process_video_error",
                video_id=video_id,
                description=f"Lỗi khi xử lý video: {str(e)}"
            )
            db.add(user_history)
            db.commit()
            
            # Ghi log lỗi
            logger.error(f"Lỗi khi xử lý video {video_id}: {str(e)}", exc_info=True)
            
    except Exception as e:
        logger.error(f"Lỗi chung khi xử lý video {video_id}: {str(e)}", exc_info=True)


@router.post("", response_model=VideoSchema)
async def create_video(
    file: UploadFile = File(None, description="Video file để tải lên từ máy tính (.mp4, .avi). Click vào nút Choose File để chọn file video."),
    youtube_url: str = Form(None, description="URL YouTube (thay thế cho việc tải video)"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Tạo video mới - tự động xác định loại video:
    - Nếu upload file: video_type = UPLOAD - Click vào nút "Choose File" để chọn file từ máy tính
    - Nếu cung cấp youtube_url: video_type = YOUTUBE
    
    QUAN TRỌNG: Khi tải file từ máy cục bộ, phải sử dụng form multipart.
    """
    video_url = None
    cloudinary_public_id = None
    
    # Tự động xác định loại video dựa trên dữ liệu đầu vào
    if file and file.filename:
        video_type_enum = VideoTypeEnum.UPLOAD
        # Tải lên Cloudinary trực tiếp từ bộ nhớ
        video_url, cloudinary_public_id = await save_upload_file(file)
    elif youtube_url:
        video_type_enum = VideoTypeEnum.YOUTUBE
        # Tải video từ YouTube và upload lên Cloudinary
        success, message, video_url, cloudinary_public_id = download_youtube_video(youtube_url)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message,
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Phải cung cấp file video hoặc URL YouTube",
        )
    
    # Tạo bản ghi video
    video_id = uuid.uuid4()
    video = Video(
        video_id=video_id,
        user_id=current_user.user_id,
        original_video_url=video_url,
        video_type=video_type_enum,
        youtube_url=youtube_url if video_type_enum == VideoTypeEnum.YOUTUBE else None,
        status=StatusEnum.PENDING,
        fire_detected=False,
        cloudinary_public_id=cloudinary_public_id
    )
    db.add(video)
    
    # Thêm lịch sử
    user_history = UserHistory(
        history_id=uuid.uuid4(),
        user_id=current_user.user_id,
        action_type="upload_video",
        video_id=video_id,
        description=f"Tải lên video mới. Loại: {video_type_enum}"
    )
    db.add(user_history)
    db.commit()
    db.refresh(video)
    
    # Khởi chạy task xử lý video trong nền
    background_tasks.add_task(process_video_task, db, video_id, video_url)
    
    return video


@router.get("", response_model=List[VideoSchema])
def read_videos(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Đọc danh sách video của người dùng hiện tại
    """
    videos = db.query(Video).filter(Video.user_id == current_user.user_id) \
        .order_by(desc(Video.created_at)) \
        .offset(skip).limit(limit).all()
    return videos


@router.get("/all", response_model=List[VideoSchema])
def read_all_videos(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_admin),
) -> Any:
    """
    Đọc tất cả video (chỉ admin)
    """
    videos = db.query(Video).order_by(desc(Video.created_at)).offset(skip).limit(limit).all()
    return videos


@router.get("/{video_id}", response_model=VideoWithDetections)
def read_video(
    *,
    db: Session = Depends(get_db),
    video_id: uuid.UUID,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Đọc thông tin chi tiết của video
    """
    video = db.query(Video).filter(Video.video_id == video_id).first()
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video không tồn tại",
        )
    
    # Kiểm tra quyền truy cập
    if video.user_id != current_user.user_id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Không có quyền truy cập video này",
        )
    
    # Thêm lịch sử
    user_history = UserHistory(
        history_id=uuid.uuid4(),
        user_id=current_user.user_id,
        action_type="view_video",
        video_id=video_id,
        description=f"Xem chi tiết video"
    )
    db.add(user_history)
    db.commit()
    
    return video


@router.get("/test-cloudinary")
def test_cloudinary_connection():
    """
    Kiểm tra kết nối với Cloudinary
    """
    try:
        # Tạo dữ liệu nhị phân để kiểm tra
        test_content = b"Test Cloudinary connection"
        test_filename = f"test_{uuid.uuid4()}.txt"
        
        # Tải lên Cloudinary trực tiếp từ bộ nhớ
        success, message, result = upload_bytes_to_cloudinary(
            test_content, 
            filename=test_filename, 
            resource_type="raw"
        )
        
        if success:
            # Xóa file trên Cloudinary
            public_id = result.get("public_id")
            delete_from_cloudinary(public_id, resource_type="raw")
            
            return {
                "status": "success",
                "message": "Kết nối Cloudinary thành công",
                "details": {
                    "cloud_name": settings.CLOUDINARY_CLOUD_NAME,
                    "folder": settings.CLOUDINARY_FOLDER,
                    "test_result": result
                }
            }
        else:
            return {
                "status": "error",
                "message": f"Lỗi kết nối Cloudinary: {message}"
            }
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra Cloudinary: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "message": f"Lỗi khi kiểm tra Cloudinary: {str(e)}"
        }


@router.delete("/{video_id}")
def delete_video(
    *,
    db: Session = Depends(get_db),
    video_id: uuid.UUID,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Xóa video và tất cả tài nguyên liên quan
    """
    video = db.query(Video).filter(Video.video_id == video_id).first()
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video không tồn tại",
        )
    
    # Kiểm tra quyền truy cập
    if video.user_id != current_user.user_id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Không có quyền xóa video này",
        )
    
    try:
        # Xóa video gốc khỏi Cloudinary
        if video.cloudinary_public_id:
            success, message = delete_from_cloudinary(video.cloudinary_public_id)
            if success:
                logger.info(f"Đã xóa video gốc từ Cloudinary: {video.cloudinary_public_id}")
            else:
                logger.warning(f"Không thể xóa video gốc từ Cloudinary: {message}")
        
        # Xóa video đã xử lý khỏi Cloudinary
        if video.cloudinary_processed_id:
            success, message = delete_from_cloudinary(video.cloudinary_processed_id)
            if success:
                logger.info(f"Đã xóa video đã xử lý từ Cloudinary: {video.cloudinary_processed_id}")
            else:
                logger.warning(f"Không thể xóa video đã xử lý từ Cloudinary: {message}")
        
        # Xóa các phát hiện đám cháy
        fire_detections = db.query(FireDetection).filter(FireDetection.video_id == video_id).all()
        for detection in fire_detections:
            # Nếu hình ảnh frame được lưu trên Cloudinary, xóa nó
            if detection.max_fire_frame_image_path and "cloudinary.com" in detection.max_fire_frame_image_path:
                # Lấy public_id từ URL
                try:
                    # Thông thường URL có dạng .../v1/folder/public_id
                    # Lấy phần sau "v1/folder"
                    parts = detection.max_fire_frame_image_path.split("v1/")
                    if len(parts) > 1:
                        path_parts = parts[1].split(".")
                        if len(path_parts) > 1:
                            public_id = path_parts[0]  # Bỏ phần mở rộng file
                            delete_from_cloudinary(public_id, resource_type="image")
                            logger.info(f"Đã xóa frame đám cháy từ Cloudinary: {public_id}")
                except Exception as e:
                    logger.warning(f"Không thể xóa frame đám cháy từ Cloudinary: {str(e)}")
        
        # Xóa video từ cơ sở dữ liệu (cascade sẽ xóa tất cả fire_detections)
        db.delete(video)
        db.commit()
        
        # Thêm lịch sử
        user_history = UserHistory(
            history_id=uuid.uuid4(),
            user_id=current_user.user_id,
            action_type="delete_video",
            description=f"Đã xóa video (ID: {video_id})"
        )
        db.add(user_history)
        db.commit()
        
        return {"message": "Đã xóa video thành công"}
        
    except Exception as e:
        logger.error(f"Lỗi khi xóa video {video_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi xóa video: {str(e)}",
        )


@router.websocket("/ws/process/{video_id}")
async def process_video_streaming(
    websocket: WebSocket,
    video_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """
    WebSocket endpoint để stream video đang xử lý theo thời gian thực
    """
    await websocket.accept()
    
    try:
        # Lấy thông tin video
        video = db.query(Video).filter(Video.video_id == video_id).first()
        if not video:
            await websocket.send_json({"error": "Video không tồn tại"})
            await websocket.close()
            return
            
        # Cập nhật trạng thái video
        video.status = StatusEnum.PROCESSING
        db.commit()
        
        try:
            # Tải xuống video từ Cloudinary URL
            await websocket.send_json({"status": "downloading", "message": "Đang tải xuống video gốc..."})
            success, message, video_data = download_from_cloudinary(video.original_video_url)
            
            if not success or not video_data:
                raise Exception(f"Không thể tải xuống video: {message}")
            
            # Xử lý video và stream kết quả
            await websocket.send_json({"status": "processing", "message": "Đang xử lý video..."})
            
            # Xử lý video với streaming kết quả
            success, processed_data, detection_info = await fire_service.process_video_streaming_from_memory(video_data, websocket)
            
            if success and processed_data:
                # Tải video đã xử lý lên Cloudinary
                processed_filename = f"processed_{uuid.uuid4()}.mp4"
                upload_success, upload_message, result = upload_bytes_to_cloudinary(
                    processed_data, 
                    filename=processed_filename
                )
                
                if upload_success:
                    processed_video_url = result.get("secure_url")
                    cloudinary_processed_id = result.get("public_id")
                    
                    # Cập nhật database với kết quả
                    video.status = StatusEnum.COMPLETED
                    video.processed_video_url = processed_video_url
                    video.cloudinary_processed_id = cloudinary_processed_id
                    video.fire_detected = detection_info.get("fire_detected", False)
                    db.commit()
                    
                    # Gửi email nếu phát hiện cháy và user đã bật thông báo
                    if video.fire_detected and video.user_id:
                        user = db.query(User).filter(User.user_id == video.user_id).first()
                        if user:
                            notification_settings = db.query(Notification).filter(
                                Notification.user_id == video.user_id
                            ).order_by(desc(Notification.created_at)).first()
                            if notification_settings and notification_settings.enable_email_notification:
                                try:
                                    detection_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                                    # Nếu detection_info có confidence thì lấy, không thì mặc định 0.8
                                    max_confidence = 0.8
                                    if detection_info and isinstance(detection_info, dict):
                                        max_confidence = detection_info.get("confidence", 0.8)
                                    send_fire_detection_notification(
                                        user_email=user.email,
                                        video_title=f"Video ID: {video_id}",
                                        detection_time=detection_time,
                                        video_url=video.original_video_url,
                                        processed_video_url=processed_video_url,
                                        confidence=max_confidence
                                    )
                                    logger.info(f"Đã gửi email thông báo đám cháy (WebSocket) đến {user.email} cho video {video_id}")
                                    notification = Notification(
                                        notification_id=uuid.uuid4(),
                                        user_id=video.user_id,
                                        video_id=video_id,
                                        title="Phát hiện đám cháy trong video của bạn (WebSocket)",
                                        message=f"Hệ thống đã phát hiện đám cháy trong video của bạn với độ tin cậy {max_confidence:.2%}.",
                                        enable_email_notification=True,
                                        enable_website_notification=True
                                    )
                                    db.add(notification)
                                    db.commit()
                                except Exception as e:
                                    logger.error(f"Lỗi khi gửi email thông báo (WebSocket): {str(e)}")
                    # Thông báo hoàn thành
                    await websocket.send_json({
                        "status": "completed", 
                        "message": "Xử lý video hoàn tất",
                        "processed_url": processed_video_url
                    })
                else:
                    # Xử lý lỗi
                    video.status = StatusEnum.FAILED
                    db.commit()
                    await websocket.send_json({"status": "error", "message": f"Lỗi khi tải lên: {upload_message}"})
            else:
                # Xử lý lỗi
                video.status = StatusEnum.FAILED
                db.commit()
                await websocket.send_json({"status": "error", "message": "Lỗi khi xử lý video"})
        
        except Exception as e:
            logger.error(f"Lỗi trong WebSocket: {str(e)}", exc_info=True)
            try:
                await websocket.send_json({"status": "error", "message": f"Lỗi: {str(e)}"})
            except:
                pass
            
            # Cập nhật trạng thái lỗi
            video.status = StatusEnum.FAILED
            db.commit()
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for video {video_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket: {str(e)}", exc_info=True)
        try:
            await websocket.send_json({"status": "error", "message": f"Lỗi: {str(e)}"})
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass


@router.websocket("/ws/process_url/{video_id}")
async def process_video_from_url(
    websocket: WebSocket,
    video_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """
    WebSocket endpoint để xử lý video trực tiếp từ Cloudinary URL
    Hiển thị kết quả realtime và lưu video được xử lý lên Cloudinary
    """
    await websocket.accept()
    
    try:
        # Lấy thông tin video
        video = db.query(Video).filter(Video.video_id == video_id).first()
        if not video:
            await websocket.send_json({"status": "error", "message": "Video không tồn tại"})
            await websocket.close()
            return
            
        # Cập nhật trạng thái video
        video.status = StatusEnum.PROCESSING
        db.commit()
        
        try:
            # Lấy URL video gốc từ Cloudinary
            video_url = video.original_video_url
            if not video_url:
                raise Exception("URL video không tồn tại")
            
            # Thông báo bắt đầu xử lý
            await websocket.send_json({
                "status": "starting", 
                "message": "Bắt đầu xử lý video từ Cloudinary",
                "video_url": video_url
            })
            
            # Xử lý video và stream kết quả
            result = await fire_service.process_video_streaming_websocket(video_url, websocket)
            
            if not result.get("success", False):
                raise Exception(result.get("message", "Không thể xử lý video"))
            
            # Tải video đã xử lý lên Cloudinary
            processed_video_data = result.get("video_data")
            processed_filename = f"processed_{uuid.uuid4()}.mp4"
            
            # Thông báo đang tải lên Cloudinary
            await websocket.send_json({
                "status": "uploading",
                "message": "Đang tải video đã xử lý lên Cloudinary"
            })
            
            # Tải lên Cloudinary
            upload_success, upload_message, upload_result = upload_bytes_to_cloudinary(
                processed_video_data, 
                filename=processed_filename
            )
            
            if upload_success:
                processed_video_url = upload_result.get("secure_url")
                cloudinary_processed_id = upload_result.get("public_id")
                
                # Cập nhật database với kết quả
                video.status = StatusEnum.COMPLETED
                video.processed_video_url = processed_video_url
                video.cloudinary_processed_id = cloudinary_processed_id
                video.fire_detected = result.get("fire_detected", False)
                db.commit()
                
                # Lưu các phát hiện đám cháy vào database
                detections = result.get("detections", [])
                for det_info in detections:
                    # Chỉ lưu các phát hiện có độ tin cậy cao
                    if det_info.get("confidence", 0) >= 0.5:
                        fire_detection = FireDetection(
                            detection_id=uuid.uuid4(),
                            video_id=video_id,
                            fire_start_time=det_info.get("time", 0),
                            fire_end_time=det_info.get("time", 0) + 1, # +1 giây để tạo khoảng thời gian
                            confidence=det_info.get("confidence", 0),
                            max_fire_frame=det_info.get("frame", 0)
                        )
                        db.add(fire_detection)
                
                # Thêm lịch sử
                user_history = UserHistory(
                    history_id=uuid.uuid4(),
                    user_id=video.user_id,
                    action_type="process_video_streaming",
                    video_id=video_id,
                    description=f"Xử lý video streaming thành công. Phát hiện đám cháy: {result.get('fire_detected', False)}"
                )
                db.add(user_history)
                db.commit()
                
                # Thông báo hoàn thành
                await websocket.send_json({
                    "status": "completed", 
                    "message": "Xử lý video hoàn tất",
                    "processed_url": processed_video_url,
                    "fire_detected": result.get("fire_detected", False),
                    "detections_count": len(detections)
                })
            else:
                # Xử lý lỗi khi tải lên Cloudinary
                video.status = StatusEnum.FAILED
                db.commit()
                await websocket.send_json({
                    "status": "error", 
                    "message": f"Lỗi khi tải video lên Cloudinary: {upload_message}"
                })
        
        except Exception as e:
            logger.error(f"Lỗi trong WebSocket: {str(e)}", exc_info=True)
            try:
                await websocket.send_json({
                    "status": "error", 
                    "message": f"Lỗi: {str(e)}"
                })
            except:
                pass
            
            # Cập nhật trạng thái lỗi
            video.status = StatusEnum.FAILED
            db.commit()
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for video {video_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket: {str(e)}", exc_info=True)
        try:
            await websocket.send_json({
                "status": "error", 
                "message": f"Lỗi: {str(e)}"
            })
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass 