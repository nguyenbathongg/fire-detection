import os
import uuid
import logging
from typing import List, Optional, Tuple, Dict, Any, BinaryIO
from datetime import datetime
from fastapi import UploadFile, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.models import User, Video, FireDetection, UserHistory, Notification
from app.schemas import VideoCreate, VideoUpdate
from app.models.enums import VideoTypeEnum, StatusEnum
from app.utils.video import save_upload_file, download_youtube_video
from app.utils.cloudinary_service import upload_bytes_to_cloudinary, download_from_cloudinary, delete_from_cloudinary
from app.utils.email_service import send_fire_detection_notification
from app.services.fire_detection import FireDetectionService

logger = logging.getLogger(__name__)


class VideoController:
    """
    Controller xử lý logic nghiệp vụ liên quan đến video
    """
    
    fire_service = FireDetectionService()
    
    @staticmethod
    def get_video_by_id(db: Session, video_id: uuid.UUID) -> Optional[Video]:
        """Lấy video theo ID"""
        return db.query(Video).filter(Video.video_id == video_id).first()
    
    @staticmethod
    def get_videos(db: Session, user_id: uuid.UUID = None, skip: int = 0, limit: int = 100) -> List[Video]:
        """Lấy danh sách video của người dùng hoặc tất cả video"""
        query = db.query(Video)
        if user_id:
            query = query.filter(Video.user_id == user_id)
        return query.order_by(desc(Video.created_at)).offset(skip).limit(limit).all()
    
    @staticmethod
    async def create_video_from_upload(
        db: Session, 
        file: UploadFile, 
        user_id: uuid.UUID
    ) -> Video:
        """Tạo video mới từ file upload"""
        if not file:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Vui lòng cung cấp file video"
            )
            
        # Lưu video tạm thời
        content = await file.read()
        
        # Kiểm tra kích thước file
        file_size = len(content)
        if file_size > 50 * 1024 * 1024:  # 50MB
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Kích thước file vượt quá giới hạn (50MB)"
            )
        
        # Upload lên Cloudinary
        filename = f"original_{uuid.uuid4()}.mp4"
        success, message, result = upload_bytes_to_cloudinary(content, filename=filename)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Không thể tải lên Cloudinary: {message}"
            )
            
        video_url = result.get("secure_url")
        cloudinary_id = result.get("public_id")
        
        # Tạo bản ghi video mới trong database
        video_id = uuid.uuid4()
        video = Video(
            video_id=video_id,
            user_id=user_id,
            title=file.filename,
            video_url=video_url,
            cloudinary_id=cloudinary_id,
            video_type=VideoTypeEnum.UPLOAD,
            status=StatusEnum.PENDING,
            fire_detected=False
        )
        
        db.add(video)
        
        # Thêm lịch sử
        history = UserHistory(
            history_id=uuid.uuid4(),
            user_id=user_id,
            action_type="video_upload",
            description=f"Tải lên video: {file.filename}"
        )
        db.add(history)
        
        db.commit()
        db.refresh(video)
        return video
    
    @staticmethod
    async def create_video_from_youtube(
        db: Session, 
        youtube_url: str, 
        user_id: uuid.UUID
    ) -> Video:
        """Tạo video mới từ URL YouTube"""
        if not youtube_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Vui lòng cung cấp URL YouTube"
            )
            
        try:
            # Tải video từ YouTube
            video_title, video_data = download_youtube_video(youtube_url)
            
            # Kiểm tra kích thước file
            file_size = len(video_data)
            if file_size > 50 * 1024 * 1024:  # 50MB
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Kích thước video YouTube vượt quá giới hạn (50MB)"
                )
                
            # Upload lên Cloudinary
            filename = f"youtube_{uuid.uuid4()}.mp4"
            success, message, result = upload_bytes_to_cloudinary(video_data, filename=filename)
            
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Không thể tải lên Cloudinary: {message}"
                )
                
            video_url = result.get("secure_url")
            cloudinary_id = result.get("public_id")
            
            # Tạo bản ghi video mới trong database
            video_id = uuid.uuid4()
            video = Video(
                video_id=video_id,
                user_id=user_id,
                title=video_title,
                video_url=video_url,
                cloudinary_id=cloudinary_id,
                video_type=VideoTypeEnum.YOUTUBE,
                status=StatusEnum.PENDING,
                fire_detected=False,
                youtube_url=youtube_url
            )
            
            db.add(video)
            
            # Thêm lịch sử
            history = UserHistory(
                history_id=uuid.uuid4(),
                user_id=user_id,
                action_type="video_youtube",
                description=f"Tải video từ YouTube: {video_title}"
            )
            db.add(history)
            
            db.commit()
            db.refresh(video)
            return video
            
        except Exception as e:
            logger.error(f"Lỗi khi tải video từ YouTube: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Không thể tải video từ YouTube: {str(e)}"
            )
    
    @staticmethod
    async def process_video(db: Session, video_id: uuid.UUID):
        """
        Xử lý video (phát hiện đám cháy)
        """
        # Lấy thông tin video
        video = VideoController.get_video_by_id(db, video_id)
        if not video:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Không tìm thấy video"
            )
            
        # Cập nhật trạng thái video
        video.status = StatusEnum.PROCESSING
        db.commit()
        
        try:
            # Tải xuống video từ Cloudinary URL vào bộ nhớ
            logger.info(f"Tải xuống video từ Cloudinary: {video.video_url}")
            success, message, video_data = download_from_cloudinary(video.video_url)
            
            if not success or not video_data:
                logger.error(f"Không thể tải xuống video: {message}")
                raise Exception(f"Không thể tải xuống video: {message}")
            
            # Phát hiện đám cháy trực tiếp từ dữ liệu nhị phân
            logger.info(f"Bắt đầu phát hiện đám cháy")
            
            # Thử phát hiện đám cháy với số lần thử tối đa
            max_retries = 3
            current_retry = 0
            
            while current_retry < max_retries:
                if not VideoController.fire_service.model:
                    logger.warning(f"Model YOLO chưa được tải. Đang thử tải lại model (lần {current_retry + 1}/{max_retries}).")
                    if VideoController.fire_service.load_model():
                        logger.info("Đã tải lại model YOLO thành công")
                    else:
                        logger.error("Không thể tải lại model YOLO")
                        current_retry += 1
                        if current_retry >= max_retries:
                            raise Exception("Không thể tải model YOLO sau nhiều lần thử, không thể tiếp tục xử lý")
                        continue
                
                try:
                    # Thử phát hiện đám cháy
                    fire_detected, detections, max_fire_frame = VideoController.fire_service.detect_fire_from_memory(video_data)
                    break  # Nếu không có lỗi, thoát khỏi vòng lặp
                except Exception as e:
                    logger.error(f"Lỗi khi phát hiện đám cháy: {str(e)}")
                    current_retry += 1
                    if current_retry >= max_retries:
                        raise Exception(f"Không thể phát hiện đám cháy sau nhiều lần thử: {str(e)}")
            
            # Xử lý video (đánh dấu vùng phát hiện đám cháy)
            logger.info(f"Xử lý video với vùng phát hiện đám cháy trong bộ nhớ")
            success, processed_video_data, detection_info = VideoController.fire_service.process_video_from_memory(video_data)
            
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
                VideoController._send_fire_notification(db, video, detections)
            
            # Lưu các phát hiện vào cơ sở dữ liệu
            for detection in detections:
                VideoController._save_fire_detection(db, video_id, detection, max_fire_frame)
            
            # Thêm lịch sử
            history = UserHistory(
                history_id=uuid.uuid4(),
                user_id=video.user_id,
                action_type="video_processed",
                description=f"Đã xử lý video: {video.title}"
            )
            db.add(history)
            db.commit()
            
            return video
        
        except Exception as e:
            logger.error(f"Lỗi khi xử lý video: {str(e)}")
            # Cập nhật trạng thái video
            video.status = StatusEnum.ERROR
            db.commit()
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Lỗi khi xử lý video: {str(e)}"
            )
    
    @staticmethod
    def delete_video(db: Session, video_id: uuid.UUID, user_id: uuid.UUID, is_admin: bool = False) -> None:
        """
        Xóa video và các tài nguyên liên quan
        """
        # Lấy thông tin video
        video = db.query(Video).filter(Video.video_id == video_id).first()
        if not video:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video không tồn tại"
            )
        
        # Kiểm tra quyền xóa - cho phép admin xóa bất kỳ video nào
        if not is_admin and video.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Không có quyền xóa video này"
            )
        
        # Xóa video gốc từ Cloudinary nếu có
        if video.cloudinary_public_id:
            try:
                delete_from_cloudinary(video.cloudinary_public_id)
            except Exception as e:
                logger.error(f"Lỗi khi xóa video gốc từ Cloudinary: {str(e)}")
        
        # Xóa video đã xử lý từ Cloudinary nếu có
        if video.cloudinary_processed_id:
            try:
                delete_from_cloudinary(video.cloudinary_processed_id)
            except Exception as e:
                logger.error(f"Lỗi khi xóa video đã xử lý từ Cloudinary: {str(e)}")
        
        # Xóa các bản ghi liên quan trong database
        db.query(FireDetection).filter(FireDetection.video_id == video_id).delete()
        db.query(UserHistory).filter(UserHistory.video_id == video_id).delete()
        db.query(Notification).filter(Notification.video_id == video_id).delete()
        
        # Xóa video
        db.delete(video)
        db.commit()
    
    @staticmethod
    def _send_fire_notification(db: Session, video: Video, detections: List[Dict]):
        """Gửi thông báo khi phát hiện đám cháy"""
        user = db.query(User).filter(User.user_id == video.user_id).first()
        if not user:
            return
            
        # Lấy notification_settings gần nhất
        notification_settings = db.query(Notification).filter(
            Notification.user_id == video.user_id
        ).order_by(desc(Notification.created_at)).first()
        
        # Kiểm tra xem người dùng có bật thông báo qua email không
        if notification_settings and notification_settings.enable_email_notification:
            try:
                # Lấy tiêu đề video nếu có, nếu không thì "Video ID: ..."
                video_title = video.title if hasattr(video, 'title') and video.title else f"Video ID: {video.video_id}"
                # Lấy thời gian phát hiện đầu tiên trong các detection
                if detections and len(detections) > 0 and 'time' in detections[0]:
                    detection_time = detections[0]['time']
                else:
                    detection_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                # Lấy thông tin độ tin cậy cao nhất
                max_confidence = max([d.get("confidence", 0) for d in detections]) if detections else 0.8
                
                # Gửi email thông báo
                email_sent = send_fire_detection_notification(
                    user_email=user.email,
                    video_title=video_title,
                    detection_time=detection_time,
                    video_url=video.video_url,
                    processed_video_url=video.processed_video_url,
                    confidence=max_confidence
                )
                
                if email_sent:
                    logger.info(f"Đã gửi email thông báo đám cháy đến {user.email} cho video {video.video_id}")
                else:
                    logger.error(f"Không thể gửi email thông báo đến {user.email}")
                
                # Chỉ tạo bản ghi thông báo nếu người dùng đã bật thông báo email hoặc thông báo trên web
                if notification_settings.enable_email_notification or notification_settings.enable_website_notification:
                    # Tạo notification trong hệ thống
                    notification = Notification(
                        notification_id=uuid.uuid4(),
                        user_id=video.user_id,
                        video_id=video.video_id,
                        title="Phát hiện đám cháy trong video của bạn",
                        message=f"Hệ thống đã phát hiện đám cháy trong video của bạn với độ tin cậy {max_confidence:.2%}.",
                        enable_email_notification=notification_settings.enable_email_notification,
                        enable_website_notification=notification_settings.enable_website_notification
                    )
                    db.add(notification)
                    db.commit()
                    logger.info(f"Đã tạo thông báo trong cơ sở dữ liệu cho user_id={user.user_id}, video_id={video.video_id}")
            except Exception as e:
                logger.error(f"Lỗi khi gửi email thông báo: {str(e)}")
    
    @staticmethod
    def _save_fire_detection(db: Session, video_id: uuid.UUID, detection: Dict, max_fire_frame: Tuple = None):
        """Lưu thông tin phát hiện đám cháy"""
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
            fire_start_time=detection.get("fire_start_time"),
            fire_end_time=detection.get("fire_end_time"),
            confidence=detection.get("confidence"),
            max_fire_frame=detection.get("max_fire_frame"),
            max_fire_frame_image_path=frame_cloudinary_url
        )
        db.add(fire_detection) 