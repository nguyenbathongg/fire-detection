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
from app.controllers.video_controller import VideoController
from app.controllers.user_history_controller import UserHistoryController

logger = logging.getLogger(__name__)

router = APIRouter()
fire_service = FireDetectionService()






@router.get("", response_model=List[VideoSchema])
async def read_videos(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Lấy danh sách video của người dùng hiện tại
    """
    videos = VideoController.get_videos(
        db=db, 
        user_id=current_user.user_id, 
        skip=skip, 
        limit=limit
    )
    
    # Thêm lịch sử
    UserHistoryController.add_history(
        db=db,
        user_id=current_user.user_id,
        action_type="view_videos",
        description="Xem danh sách video của mình"
    )
    
    return videos


@router.get("/all", response_model=List[VideoSchema])
async def read_all_videos(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_admin),
) -> Any:
    """
    Lấy danh sách tất cả video (chỉ admin)
    """
    videos = VideoController.get_videos(db=db, skip=skip, limit=limit)
    
    # Thêm lịch sử
    UserHistoryController.add_history(
        db=db,
        user_id=current_user.user_id,
        action_type="view_all_videos",
        description="Xem danh sách tất cả video"
    )
    
    return videos


@router.get("/{video_id}", response_model=VideoWithDetections)
async def read_video(
    *,
    db: Session = Depends(get_db),
    video_id: uuid.UUID,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Lấy thông tin chi tiết video
    """
    video = VideoController.get_video_by_id(db=db, video_id=video_id)
    
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video không tồn tại"
        )
    
    # Kiểm tra quyền truy cập
    if video.user_id != current_user.user_id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Không có quyền truy cập video này"
        )
    
    # Thêm lịch sử
    UserHistoryController.add_history(
        db=db,
        user_id=current_user.user_id,
        action_type="view_video_detail",
        description=f"Xem chi tiết video: {video.file_name or video.video_id}"
    )
    
    return video


@router.delete("/{video_id}")
async def delete_video(
    *,
    db: Session = Depends(get_db),
    video_id: uuid.UUID,
    current_user: User = Depends(get_current_active_admin),
) -> Any:
    """
    Xóa video (chỉ admin)
    """
    VideoController.delete_video(
        db=db, 
        video_id=video_id, 
        user_id=current_user.user_id,
        is_admin=True  # Admin luôn có quyền xóa video
    )
    
    # Thêm lịch sử
    UserHistoryController.add_history(
        db=db,
        user_id=current_user.user_id,
        action_type="delete_video",
        description=f"Admin xóa video: {video_id}"
    )
    
    return {"message": "Xóa video thành công"}


