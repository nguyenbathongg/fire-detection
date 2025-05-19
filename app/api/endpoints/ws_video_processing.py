import os
import uuid
import cv2
import io
import numpy as np
import time
import logging
import tempfile
import asyncio
from typing import Dict, List, Optional, Union
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException, Depends
from starlette.websockets import WebSocketState
from pydantic import BaseModel
from sqlalchemy.orm import Session
from jose import jwt, JWTError
from datetime import datetime

from app.services.fire_detection import FireDetectionService, predict_and_display
from app.utils.cloudinary_service import upload_bytes_to_cloudinary
from app.utils.video import download_youtube_video_file
from app.utils.email_service import send_fire_detection_notification
from app.models.notification import Notification
from app.models.user import User
from app.db.base import get_db
from app.core.config import settings
from app.schemas.user import TokenPayload

logger = logging.getLogger(__name__)
router = APIRouter()
fire_service = FireDetectionService()

# Xử lý xác thực được tích hợp trực tiếp vào endpoint

@router.websocket("/direct-process")
async def process_direct_video_websocket(websocket: WebSocket):
    """
    WebSocket endpoint để xử lý video trực tiếp từ người dùng, có thể là tải lên hoặc từ YouTube URL.
    Hiển thị kết quả realtime và lưu video được xử lý lên Cloudinary.
    
    Hỗ trợ xác thực qua token và gửi email khi phát hiện đám cháy (nếu người dùng đã bật thông báo).
    """
    await websocket.accept()
    
    # Khởi tạo các biến cần thiết
    user = None
    db = next(get_db())
    
    try:
        # Xử lý token nếu có (tùy chọn)
        try:
            auth_data = await websocket.receive_json()
            token = auth_data.get("token")
            
            # Nếu có token, cố gắng xác thực người dùng
            if token:
                try:
                    # Cố gắng xác thực token, nhưng không bắt buộc
                    payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
                    token_data = TokenPayload(**payload)
                    user = db.query(User).filter(User.user_id == token_data.sub).first()
                    
                    if user:
                        await websocket.send_json({"status": "auth", "message": f"Xác thực thành công, xin chào {user.username}"})
                        logger.info(f"Người dùng {user.username} đã xác thực và kết nối WebSocket")
                    else:
                        user = None  # Đặt lại user về None nếu không tìm thấy
                        await websocket.send_json({"status": "auth", "message": "Token hợp lệ nhưng không tìm thấy người dùng, tiếp tục dưới dạng khách"})
                except (JWTError, ValueError):
                    user = None  # Không xác thực được
                    await websocket.send_json({"status": "auth", "message": "Token không hợp lệ, tiếp tục dưới dạng khách"})
                    logger.warning("Token không hợp lệ, người dùng kết nối dưới dạng khách")
            else:
                user = None  # Không có token
                await websocket.send_json({"status": "auth", "message": "Kết nối dưới dạng khách"})
                logger.info("Khách kết nối WebSocket")
                
        except WebSocketDisconnect:
            logger.info("Client ngắt kết nối khi xác thực")
            return
        except Exception as e:
            # Bỏ qua lỗi xác thực, cho phép kết nối không xác thực
            user = None
            logger.error(f"Lỗi khi xác thực: {str(e)}")
            try:
                await websocket.send_json({"status": "auth", "message": "Bỏ qua xác thực, tiếp tục dưới dạng khách"})
            except:
                pass
            
        # Nhận thông tin loại video
        try:
            data = await websocket.receive_json()
            video_type = data.get("type")  # "upload", "youtube", hoặc "chunk_info"
        except WebSocketDisconnect:
            logger.info("Client ngắt kết nối khi gửi loại video")
            return
        except Exception as e:
            logger.error(f"Lỗi khi nhận loại video: {str(e)}")
            try:
                await websocket.send_json({"status": "error", "message": f"Lỗi: {str(e)}"})
                await websocket.close()
            except:
                pass
            return
        
        video_data = None
        
        # Xử lý theo kích thước phần
        if video_type == "chunk_info":
            # Nhận thông tin về chuỗi phần
            total_chunks = data.get("totalChunks", 0)
            file_size = data.get("fileSize", 0)
            chunk_size = data.get("chunkSize", 0)
            
            # Thông báo sẵn sàng nhận chuỗi phần
            try:
                logger.info(f"Chuẩn bị nhận {total_chunks} phần, tổng cộng {file_size/1024/1024:.2f} MB")
                await websocket.send_json({"status": "ready", "message": f"Sẵn sàng nhận {total_chunks} phần"})
            except:
                logger.info("Client ngắt kết nối khi chuẩn bị nhận chuỗi phần")
                return
                
            # Chuẩn bị buộc để lưu trữ dữ liệu
            all_chunks = bytearray()
            received_chunks = 0
            
            # Nhận từng phần
            while received_chunks < total_chunks:
                try:
                    # Nhận phần tiếp theo
                    chunk = await websocket.receive_bytes()
                    all_chunks.extend(chunk)
                    received_chunks += 1
                    
                    # Cập nhật tiến trình
                    if received_chunks % 5 == 0 or received_chunks == total_chunks:
                        try:
                            percent = min(100, int((received_chunks / total_chunks) * 100))
                            await websocket.send_json({
                                "status": "receiving", 
                                "message": f"Nhận {received_chunks}/{total_chunks} phần ({percent}%)"
                            })
                        except:
                            pass
                except WebSocketDisconnect:
                    logger.info(f"Client ngắt kết nối sau khi gửi {received_chunks}/{total_chunks} phần")
                    return
                except Exception as e:
                    logger.error(f"Lỗi khi nhận phần {received_chunks+1}: {str(e)}")
                    try:
                        await websocket.send_json({"status": "error", "message": f"Lỗi khi nhận phần {received_chunks+1}: {str(e)}"})
                    except:
                        pass
                    return
            
            # Đã nhận đủ các phần
            video_data = bytes(all_chunks)
            try:
                await websocket.send_json({"status": "info", "message": f"Nhận đủ {total_chunks} phần, tổng cộng {len(video_data)/1024/1024:.2f} MB"})
            except:
                logger.info("Client ngắt kết nối sau khi nhận đủ dữ liệu")
                return
        
        # Xử lý upload thông thường
        elif video_type == "upload":
            # Thông báo cho client sẵn sàng nhận dữ liệu
            try:
                await websocket.send_json({"status": "ready", "message": "Sẵn sàng nhận dữ liệu video..."})
            except:
                logger.info("Client ngắt kết nối khi chuẩn bị nhận video")
                return
            
            # Nhận video trực tiếp từ websocket
            try:
                try:
                    logger.info("Bắt đầu nhận dữ liệu video binary")
                    binary_data = await websocket.receive_bytes()
                    logger.info(f"Nhận được {len(binary_data)} bytes dữ liệu")
                    
                    # Kiểm tra xem dữ liệu nhận có hợp lệ không
                    if not isinstance(binary_data, bytes):
                        logger.error(f"Kiểu dữ liệu không hợp lệ: {type(binary_data).__name__}")
                        raise ValueError(f"Dữ liệu không phải kiểu bytes: {type(binary_data).__name__}")
                    
                    video_data = binary_data
                except Exception as inner_e:
                    logger.error(f"Lỗi khi xử lý binary_data: {type(inner_e).__name__}, {str(inner_e)}")
                    raise inner_e
                
                try:
                    await websocket.send_json({"status": "info", "message": f"Nhận được {len(video_data)/1024/1024:.2f} MB dữ liệu"})
                except:
                    logger.info("Client ngắt kết nối sau khi nhận xong dữ liệu")
                    return
            except WebSocketDisconnect:
                logger.info("Client ngắt kết nối khi đang gửi dữ liệu")
                return
            except Exception as e:
                logger.error(f"Lỗi khi nhận dữ liệu video: {type(e).__name__}, {str(e)}")
                try:
                    await websocket.send_json({"status": "error", "message": f"Lỗi khi nhận dữ liệu: {str(e)}"})
                    await websocket.close()
                except:
                    pass
                return
            
        elif video_type == "youtube":
            youtube_url = data.get("youtube_url")
            if not youtube_url:
                try:
                    await websocket.send_json({"status": "error", "message": "URL YouTube không hợp lệ"})
                    await websocket.close()
                except:
                    pass
                return
            
            try:
                await websocket.send_json({"status": "info", "message": "Đang tải video từ YouTube..."})
            except:
                logger.info("Client ngắt kết nối khi chuẩn bị tải YouTube")
                return
            
            # Tải video từ YouTube
            try:
                video_data = await download_youtube_video(youtube_url, websocket)
            except WebSocketDisconnect:
                logger.info("Client ngắt kết nối khi đang tải YouTube")
                return
            except Exception as e:
                logger.error(f"Lỗi khi tải video từ YouTube: {str(e)}")
                try:
                    await websocket.send_json({"status": "error", "message": f"Lỗi khi tải YouTube: {str(e)}"})
                    await websocket.close()
                except:
                    pass
                return
        else:
            try:
                await websocket.send_json({"status": "error", "message": "Loại video không hợp lệ"})
                await websocket.close()
            except:
                pass
            return
            
        # Tải video lên Cloudinary (một lần duy nhất)
        try:
            await websocket.send_json({"status": "uploading", "message": "Đang tải video lên Cloudinary..."})
        except:
            logger.info("Client ngắt kết nối trước khi tải lên Cloudinary")
            return
        
        upload_filename = f"fire_detection_{uuid.uuid4()}.mp4"
        try:
            success, message, result = upload_bytes_to_cloudinary(video_data, filename=upload_filename)
        except Exception as e:
            logger.error(f"Lỗi khi tải lên Cloudinary: {str(e)}")
            try:
                await websocket.send_json({"status": "error", "message": f"Lỗi khi tải lên Cloudinary: {str(e)}"})
                await websocket.close()
            except:
                pass
            return
            
        if not success:
            logger.error(f"Không thể tải video lên Cloudinary: {message}")
            try:
                await websocket.send_json({"status": "error", "message": f"Lỗi khi tải lên Cloudinary: {message}"})
                await websocket.close()
            except:
                pass
            return
            
        # Lấy URL video từ Cloudinary
        cloudinary_url = result.get('secure_url')
        cloudinary_id = result.get('public_id')
        try:
            await websocket.send_json({
                "status": "info", 
                "message": "Đã tải video lên Cloudinary thành công",
                "original_url": cloudinary_url
            })
        except:
            logger.info("Client ngắt kết nối sau khi tải lên Cloudinary thành công")
            return
        
        # Tạo file output tạm thời
        try:
            temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            temp_output_path = temp_output.name
            temp_output.close()
        except Exception as e:
            logger.error(f"Lỗi khi tạo file tạm: {str(e)}")
            try:
                await websocket.send_json({"status": "error", "message": f"Lỗi khi tạo file tạm: {str(e)}"})
                await websocket.close()
            except:
                pass
            return
        
        # Xử lý video và stream kết quả qua WebSocket
        try:
            await websocket.send_json({"status": "processing", "message": "Đang xử lý video..."})
        except:
            logger.info("Client ngắt kết nối trước khi bắt đầu xử lý video")
            try:
                # Xóa file tạm nếu client đã ngắt kết nối
                if os.path.exists(temp_output_path):
                    os.remove(temp_output_path)
            except:
                pass
            return
        
        # Sử dụng predict_and_display với generator
        try:
            # Mở video từ URL
            video_writer = None
            fire_detected = False
            frame_count = 0
            
            # Sử dụng giải pháp đa luồng đã có sẵn trong predict_and_display
            # Biến để theo dõi email đã được gửi chưa 
            email_sent = False
            detection_frame_info = None
            
            for frame, frame_info in predict_and_display(fire_service.model, cloudinary_url, temp_output_path):
                frame_count += 1
                
                # Nếu phát hiện lửa, cập nhật trạng thái
                if frame_info.get("fire_detected", False) and not fire_detected:
                    fire_detected = True
                    detection_frame_info = frame_info.copy()  # Lưu thông tin frame phát hiện lửa
                    
                    # Gửi thông báo phát hiện lửa qua WebSocket
                    try:
                        await websocket.send_json({
                            "status": "alert", 
                            "message": "PHÁT HIỆN LỬA!",
                            "frame_info": frame_info
                        })
                    except:
                        pass
                    
                    # Chỉ ghi nhận phát hiện lửa, không gửi email ngay (sẽ gửi sau khi hoàn tất xử lý)
                    if user:
                        # Ghi log debug
                        logger.info(f"Người dùng đã đăng nhập: {user.username}, user_id: {user.user_id}")
                        
                        # Kiểm tra cài đặt thông báo của người dùng
                        notification_settings = db.query(Notification).filter(Notification.user_id == user.user_id).first()
                        
                        # Ghi log thông tin notification settings
                        if notification_settings:
                            logger.info(f"Tìm thấy cài đặt thông báo: enable_email_notification={notification_settings.enable_email_notification}")
                            # Thông báo cho client về trạng thái email
                            if notification_settings.enable_email_notification:
                                await websocket.send_json({"status": "info", "message": "Sẽ gửi email thông báo sau khi xử lý hoàn tất..."})
                        else:
                            logger.info(f"Không tìm thấy cài đặt thông báo cho user_id={user.user_id}")
                
                # Chuyển frame thành JPEG và gửi qua WebSocket
                try:
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frame_bytes = buffer.tobytes()
                    
                    # Gửi frame
                    await websocket.send_bytes(frame_bytes)
                    
                    # Gửi thông tin trạng thái
                    await websocket.send_json({
                        "status": "frame", 
                        "frame_info": frame_info
                    })
                    
                    # Nếu đã xử lý nhiều hơn 100 frame mà chưa có thông báo, thông báo tiến độ
                    if frame_count % 100 == 0:
                        await websocket.send_json({
                            "status": "progress", 
                            "frames_processed": frame_count
                        })
                except WebSocketDisconnect:
                    logger.info(f"Client ngắt kết nối sau khi đã xử lý {frame_count} frames")
                    # Hủy xử lý và xóa file
                    if os.path.exists(temp_output_path):
                        try:
                            os.remove(temp_output_path)
                        except:
                            pass
                    return
                except Exception as e:
                    logger.error(f"Lỗi khi gửi frame: {str(e)}")
                    # Tiếp tục xử lý các frame tiếp theo dù gặp lỗi gửi
                    continue
                
            # Đóng tài nguyên
            cv2.destroyAllWindows()
            
            # Tải video đã xử lý lên Cloudinary
            try:
                await websocket.send_json({"status": "info", "message": "Đang tải video đã xử lý lên Cloudinary..."})
            except:
                logger.info("Client ngắt kết nối sau khi xử lý xong video")
                # Xóa file tạm
                if os.path.exists(temp_output_path):
                    try:
                        os.remove(temp_output_path)
                    except:
                        pass
                return
            
            try:
                with open(temp_output_path, "rb") as f:
                    processed_data = f.read()
            except Exception as e:
                logger.error(f"Lỗi khi đọc file đã xử lý: {str(e)}")
                try:
                    await websocket.send_json({"status": "error", "message": f"Lỗi khi đọc file đã xử lý: {str(e)}"})
                    # Xóa file tạm
                    if os.path.exists(temp_output_path):
                        os.remove(temp_output_path)
                except:
                    pass
                return
                
            processed_filename = f"processed_fire_detection_{uuid.uuid4()}.mp4"
            try:
                upload_success, upload_message, processed_result = upload_bytes_to_cloudinary(
                    processed_data, 
                    filename=processed_filename
                )
            except Exception as e:
                logger.error(f"Lỗi khi tải video đã xử lý lên Cloudinary: {str(e)}")
                try:
                    await websocket.send_json({"status": "error", "message": f"Lỗi khi tải video đã xử lý: {str(e)}"})
                except:
                    pass
                finally:
                    # Xóa file tạm
                    if os.path.exists(temp_output_path):
                        try:
                            os.remove(temp_output_path)
                        except:
                            pass
                return
            
            # Xóa file tạm sau khi đã tải lên
            if os.path.exists(temp_output_path):
                try:
                    os.remove(temp_output_path)
                except Exception as e:
                    logger.warning(f"Không thể xóa file tạm: {str(e)}")
                
            if upload_success:
                processed_url = processed_result.get("secure_url")
                
                # Thông báo hoàn thành
                try:
                    await websocket.send_json({
                        "status": "completed",
                        "message": "Đã xử lý video xong",
                        "original_url": cloudinary_url,
                        "processed_url": processed_url,
                        "fire_detected": fire_detected,
                        "frames_processed": frame_count
                    })
                except:
                    logger.info("Client ngắt kết nối trước khi nhận kết quả cuối cùng")
                
                # Gửi email thông báo nếu phát hiện lửa và chưa gửi trước đó 
                # Bây giờ chúng ta có cả URL của video gốc và video đã xử lý
                if fire_detected and user and not email_sent:
                    logger.info(f"Bắt đầu gửi email thông báo phát hiện lửa sau khi hoàn tất xử lý")
                    try:
                        # Thông báo cho client
                        try:
                            await websocket.send_json({"status": "info", "message": "Chuẩn bị gửi email thông báo phát hiện lửa..."})
                        except:
                            pass  # Không dừng quá trình nếu gửi thông báo thất bại
                            
                        # Kiểm tra cài đặt thông báo của người dùng lần nữa
                        notification_settings = db.query(Notification).filter(Notification.user_id == user.user_id).first()
                        logger.info(f"Kiểm tra lại cài đặt thông báo cho user_id={user.user_id}")
                        
                        if notification_settings and notification_settings.enable_email_notification:
                            # Gửi email thông báo phát hiện lửa
                            detection_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                            video_title = f"Phát hiện đám cháy trong video"
                            confidence = detection_frame_info.get("confidence", 0.8) if detection_frame_info else 0.8
                            
                            logger.info(f"Gửi email thông báo cho {user.email} với video gốc: {cloudinary_url} và video đã xử lý: {processed_url}")
                            
                            # Gọi hàm gửi email
                            email_result = send_fire_detection_notification(
                                user_email=user.email,
                                video_title=video_title,
                                detection_time=detection_time,
                                video_url=cloudinary_url,
                                processed_video_url=processed_url,
                                confidence=confidence
                            )
                            
                            if email_result:
                                logger.info(f"Đã gửi email thông báo phát hiện lửa đến {user.email} thành công")
                                email_sent = True
                                try:
                                    await websocket.send_json({"status": "email", "message": f"Đã gửi email thông báo phát hiện lửa đến {user.email}"})
                                except:
                                    pass
                            else:
                                logger.error(f"Không thể gửi email thông báo đến {user.email}")
                    except Exception as email_error:
                        logger.error(f"Lỗi khi gửi email thông báo: {str(email_error)}")
                        # Lỗi này không ảnh hưởng đến kết quả cuối cùng, chỉ ghi log
            else:
                # Thông báo lỗi
                try:
                    await websocket.send_json({
                        "status": "error",
                        "message": f"Lỗi khi tải video đã xử lý: {upload_message}",
                        "original_url": cloudinary_url
                    })
                except:
                    logger.info("Client ngắt kết nối trước khi nhận thông báo lỗi")
                
        except WebSocketDisconnect:
            logger.info("WebSocket đã ngắt kết nối")
            # Xóa file tạm nếu có
            if 'temp_output_path' in locals() and os.path.exists(temp_output_path):
                try:
                    os.remove(temp_output_path)
                except:
                    pass
        except Exception as e:
            logger.error(f"Lỗi khi xử lý video: {str(e)}", exc_info=True)
            try:
                await websocket.send_json({"status": "error", "message": f"Lỗi khi xử lý video: {str(e)}"})
            except:
                pass
            
            # Xóa file tạm nếu có lỗi
            if 'temp_output_path' in locals() and os.path.exists(temp_output_path):
                try:
                    os.remove(temp_output_path)
                except:
                    pass
                    
    except WebSocketDisconnect:
        logger.info("WebSocket đã ngắt kết nối")
    except Exception as e:
        logger.error(f"Lỗi chung: {str(e)}", exc_info=True)
        try:
            await websocket.send_json({"status": "error", "message": f"Lỗi: {str(e)}"})
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass


async def download_youtube_video(youtube_url: str, websocket: Optional[WebSocket] = None) -> bytes:
    """
    Tải video từ YouTube URL
    
    Args:
        youtube_url: URL YouTube
        websocket: WebSocket để thông báo tiến độ (nếu có)
        
    Returns:
        bytes: Dữ liệu video
    """
    # Tạo file tạm thời
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_path = temp_file.name
    
    try:
        if websocket:
            await websocket.send_json({"status": "info", "message": "Đang trích xuất thông tin YouTube..."})
        
        # Sử dụng hàm có sẵn trong utils.video
        # Do hàm download_youtube_video_file là hàm đồng bộ, chúng ta chạy nó trong một thread riêng
        
        # Thực hiện tải xuống
        if websocket:
            await websocket.send_json({"status": "info", "message": "Đang tải video từ YouTube..."})
            
        # Chạy trong event loop riêng để không block
        await asyncio.to_thread(download_youtube_video_file, youtube_url, temp_path)
        
        # Báo cáo kích thước file
        file_size = os.path.getsize(temp_path) / (1024 * 1024)  # MB
        if websocket:
            await websocket.send_json({
                "status": "info", 
                "message": f"Đã tải xong video YouTube ({file_size:.2f} MB)"
            })
        
        # Đọc dữ liệu nhị phân
        with open(temp_path, "rb") as f:
            video_data = f.read()
            
        return video_data
    
    except Exception as e:
        logger.error(f"Lỗi khi tải video từ YouTube: {str(e)}")
        raise
    finally:
        # Dọn dẹp file tạm
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
