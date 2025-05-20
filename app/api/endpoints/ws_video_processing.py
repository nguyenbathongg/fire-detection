import os
import uuid
import cv2
import io
import numpy as np
import time
import logging
import tempfile
import asyncio
from typing import Dict, List, Optional, Union, Tuple
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
from app.models.video import Video
from app.models.user_history import UserHistory
from app.models.enums import VideoTypeEnum, StatusEnum

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
    # Khởi tạo biến lưu tên file và loại video
    original_file_name = None 
    video_type_enum = None
    
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
            video_type = data.get("type")  # "upload", "youtube", "chunk_info", hoặc "chunk_meta"
            logger.info(f"Nhận yêu cầu xử lý video loại: {video_type}")
        except WebSocketDisconnect:
            logger.info("Client ngắt kết nối khi gửi loại video")
            return
        except Exception as e:
            logger.error(f"Lỗi khi nhận loại video: {str(e)}")
            try:
                await websocket.send_json({"status": "error", "message": f"Lỗi khi nhận loại video: {str(e)}"})
                await websocket.close()
            except:
                pass
            return
        
        video_data = None
        
        # Biến để lưu trữ các phần (chunks) của file lớn
        chunked_data = {
            "total_chunks": 0,
            "chunks": {},
            "received_chunks": 0,
            "file_size": 0,
            "file_name": "",
            "mime_type": ""
        }
        
        # Xử lý theo loại video (upload thường, YouTube, hoặc chunking)
        if video_type == "chunk_info":
            # Nhận thông tin về chuỗi phần
            total_chunks = data.get("totalChunks", 0)
            file_size = data.get("fileSize", 0)
            file_name = data.get("fileName", "uploaded_video.mp4")
            mime_type = data.get("mimeType", "video/mp4")
            
            # Lưu thông tin file name vào log để debug
            logger.info(f"Chế độ chunk: Nhận tên file: {file_name}")
            
            # Kiểm tra thông tin hợp lệ
            if total_chunks <= 0 or file_size <= 0:
                try:
                    logger.error(f"Thông tin file không hợp lệ: total_chunks={total_chunks}, file_size={file_size}")
                    await websocket.send_json({"status": "error", "message": "Thông tin file không hợp lệ"})
                except:
                    pass
                return
            
            # Lưu trữ thông tin về các phần
            chunked_data["total_chunks"] = total_chunks
            chunked_data["file_size"] = file_size
            chunked_data["file_name"] = file_name
            chunked_data["mime_type"] = mime_type
            chunked_data["chunks"] = {}
            chunked_data["received_chunks"] = 0  # Khởi tạo số phần đã nhận
            
            # Thông báo sẵn sàng nhận chuỗi phần
            try:
                logger.info(f"Sẵn sàng nhận file lớn {file_name} ({file_size/1024/1024:.2f} MB) trong {total_chunks} phần")
                await websocket.send_json({"status": "ready", "message": f"Sẵn sàng nhận {total_chunks} phần"})  
            except Exception as e:
                logger.error(f"Lỗi khi gửi thông báo sẵn sàng nhận chunk: {str(e)}")
                return
        
        # Xử lý metadata cho từng phần
        elif video_type == "chunk_meta":
            # Kiểm tra xem đã khởi tạo chunked_data chưa
            if "total_chunks" not in chunked_data or "chunks" not in chunked_data:
                logger.error("Nhận chunk_meta trước khi gửi chunk_info")
                try:
                    await websocket.send_json({"status": "error", "message": "Cần gửi chunk_info trước khi gửi chunk_meta"})
                except:
                    pass
                return
                
            # Nhận thông tin về phần chuỗi hiện tại
            chunk_index = data.get("chunkIndex", 0)
            total_chunks = data.get("totalChunks", chunked_data.get("total_chunks", 0))
            chunk_size = data.get("chunkSize", 0)
            
            # Kiểm tra thông tin hợp lệ
            if chunk_index < 0 or chunk_index >= total_chunks:
                logger.error(f"Chỉ số chunk không hợp lệ: {chunk_index}/{total_chunks}")
                try:
                    await websocket.send_json({"status": "error", "message": f"Chỉ số chunk không hợp lệ: {chunk_index}/{total_chunks}"})
                except:
                    pass
                return
            
            logger.info(f"Chuẩn bị nhận phần {chunk_index + 1}/{chunked_data['total_chunks']} kích thước {chunk_size/1024:.2f} KB")
            
            try:
                # Báo sẵn sàng nhận chunk
                await websocket.send_json({"status": "chunk_ready", "message": f"Sẵn sàng nhận phần {chunk_index + 1}"})
                
                # Nhận dữ liệu binary cho phần này
                chunk_data = await websocket.receive_bytes()
                
                if not isinstance(chunk_data, bytes):
                    raise ValueError(f"Dữ liệu chunk không phải bytes: {type(chunk_data).__name__}")
                
                # Kiểm tra kích thước
                actual_size = len(chunk_data)
                logger.info(f"Nhận được chunk {chunk_index + 1} kích thước {actual_size/1024:.2f} KB")
                
                # Lưu trữ phần dữ liệu này
                chunked_data["chunks"][chunk_index] = chunk_data
                chunked_data["received_chunks"] += 1
                
                # Thông báo tiến trình
                percent = min(100, int((chunked_data["received_chunks"] / chunked_data["total_chunks"]) * 100))
                try:
                    await websocket.send_json({
                        "status": "receiving", 
                        "message": f"Nhận phần {chunked_data['received_chunks']}/{chunked_data['total_chunks']} ({percent}%)",
                        "percent": percent,
                        "currentChunk": chunked_data["received_chunks"],
                        "totalChunks": chunked_data["total_chunks"]
                    })
                except Exception as e:
                    logger.warning(f"Không gửi được thông báo tiến trình: {str(e)}")
                
                # Kiểm tra xem đã nhận đủ các phần chưa
                if chunked_data["received_chunks"] == chunked_data["total_chunks"]:
                    # Ghép tất cả phần lại với nhau
                    logger.info(f"Đã nhận đủ {chunked_data['total_chunks']} phần, đang ghép lại")
                    all_chunks = bytearray()
                    
                    # Ghép theo thứ tự các phần
                    for i in range(chunked_data["total_chunks"]):
                        if i in chunked_data["chunks"]:
                            chunk = chunked_data["chunks"][i]
                            all_chunks.extend(chunk)
                            logger.info(f"Ghép phần {i+1}/{chunked_data['total_chunks']}, kích thước {len(chunk)/1024:.2f} KB")
                    
                    # Chuyển thành dạng bytes
                    video_data = bytes(all_chunks)
                    logger.info(f"Hoàn tất ghép file, tổng kích thước {len(video_data)/1024/1024:.2f} MB")
                    
                    # Thông báo hoàn tất nhận
                    try:
                        await websocket.send_json({
                            "status": "received", 
                            "message": f"Đã nhận xong tất cả {chunked_data['total_chunks']} phần, tổng cộng {len(video_data)/1024/1024:.2f} MB"
                        })
                    except Exception as e:
                        logger.error(f"Lỗi khi gửi thông báo hoàn tất nhận: {str(e)}")
                    
                    # Xóa bỏ các chunk riêng lẻ để tiết kiệm bộ nhớ
                    chunked_data["chunks"].clear()
                    
                    # Tiếp tục xử lý video
                    # (code tiếp tục ở phần sau, video_data đã sẵn sàng để xử lý)
                    
            except WebSocketDisconnect:
                logger.info(f"Client ngắt kết nối khi đang gửi phần {chunk_index + 1}")
                return
            except Exception as e:
                logger.error(f"Lỗi khi nhận phần {chunk_index + 1}: {str(e)}")
                try:
                    await websocket.send_json({"status": "error", "message": f"Lỗi khi nhận phần {chunk_index + 1}: {str(e)}"})
                except:
                    pass
                return
        
        # Xử lý upload thông thường
        elif video_type == "upload":
            # Lấy thông tin tên file nếu có
            original_file_name = data.get("fileName", None)
            video_type_enum = VideoTypeEnum.UPLOAD
            
            # Ghi log chi tiết về tên file
            if original_file_name:
                logger.info(f"Chế độ upload: Nhận tên file: '{original_file_name}'")
            else:
                logger.warning("Chế độ upload: Không nhận được tên file từ client")
                logger.debug(f"Dữ liệu từ client: {data}")
            
            # Thông báo cho client sẵn sàng nhận dữ liệu
            try:
                logger.info("Gửi thông báo sẵn sàng nhận dữ liệu video")
                await websocket.send_json({
                    "status": "ready", 
                    "message": "Sẵn sàng nhận dữ liệu video...",
                    "fileNameConfirmed": original_file_name  # Xác nhận lại tên file với client
                })
            except Exception as e:
                logger.error(f"Lỗi khi gửi thông báo sẵn sàng: {str(e)}")
                return
            
            # Nhận video trực tiếp từ websocket với thời gian chờ lâu hơn
            try:
                logger.info("Bắt đầu nhận dữ liệu video binary")
                
                # Tận dụng try-except để ghi log chi tiết hơn
                try:
                    # Đặt thời gian chờ (60 giây) đủ dài để file lưu trên mạng
                    binary_data = await asyncio.wait_for(websocket.receive_bytes(), timeout=60.0) 
                    logger.info(f"Nhận được dữ liệu có kích thước: {len(binary_data)} bytes")
                except asyncio.TimeoutError:
                    logger.error("Hết thời gian chờ nhận dữ liệu (60s)")
                    await websocket.send_json({"status": "error", "message": "Hết thời gian chờ nhận dữ liệu"})
                    return
                except Exception as inner_e:
                    logger.error(f"Lỗi khi nhận bytes: {type(inner_e).__name__}: {str(inner_e)}")
                    raise inner_e
                
                # Kiểm tra xem dữ liệu nhận có hợp lệ không
                if not isinstance(binary_data, bytes):
                    logger.error(f"Kiểu dữ liệu không hợp lệ: {type(binary_data).__name__}")
                    raise ValueError(f"Dữ liệu không phải kiểu bytes: {type(binary_data).__name__}")
                
                if len(binary_data) == 0:
                    logger.error("Nhận được dữ liệu rỗng")
                    raise ValueError("Dữ liệu video trống, không thể xử lý")
                
                video_data = binary_data
                # Trong trường hợp upload, youtube_title luôn là None
                youtube_title = None
                logger.info(f"Nhận thành công {len(video_data)/1024/1024:.2f} MB dữ liệu")
                
                # Gửi thông báo nhận thành công
                try:
                    await websocket.send_json({
                        "status": "info", 
                        "message": f"Nhận được {len(video_data)/1024/1024:.2f} MB dữ liệu",
                        "fileSize": len(video_data)
                    })
                    logger.info("Gửi thông báo nhận thành công")
                except Exception as notify_e:
                    logger.warning(f"Không thể gửi thông báo nhận thành công: {str(notify_e)}")
            except WebSocketDisconnect as wd:
                logger.info("Client ngắt kết nối khi đang nhận dữ liệu upload")
                return
            except Exception as e:
                logger.error(f"Lỗi khi nhận dữ liệu upload: {str(e)}")
                try:
                    await websocket.send_json({"status": "error", "message": f"Lỗi khi nhận dữ liệu upload: {str(e)}"})
                    await websocket.close()
                except:
                    pass
                return
                
        # Xử lý video từ YouTube URL
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
                # Sử dụng hàm đa luồng để tải video từ YouTube và lấy tiêu đề
                video_data, youtube_title = await download_youtube_video(youtube_url, websocket)
                logger.info(f"Tải video YouTube thành công, kích thước: {len(video_data)/1024/1024:.2f} MB")
                if youtube_title:
                    logger.info(f"Tiêu đề video YouTube: {youtube_title}")
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
            # Thông báo lỗi nếu loại video không hợp lệ
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
            logger.info("Client ngắt kết nối trước khi tải lên")
            return
        
        # Khai báo các biến lưu thông tin Cloudinary
        cloudinary_url = None
        cloudinary_public_id = None
        
        # Chuẩn bị tên file để tải lên
        try:
            filename = f"fire_detection_{uuid.uuid4()}.mp4"
            upload_success, upload_message, upload_result = upload_bytes_to_cloudinary(video_data, filename)
            
            # Lưu lại URL và public_id
            if upload_success:
                cloudinary_url = upload_result.get("secure_url")
                cloudinary_public_id = upload_result.get("public_id")
                logger.info(f"Tải lên Cloudinary thành công, URL: {cloudinary_url}, public_id: {cloudinary_public_id}")
        except Exception as e:
            logger.error(f"Lỗi khi tải video lên Cloudinary: {str(e)}")
            try:
                await websocket.send_json({"status": "error", "message": f"Lỗi khi tải video lên Cloudinary: {str(e)}"})
                await websocket.close()
            except:
                pass
            return
            
        # Sử dụng URL và public_id đã lưu ở trên
        if cloudinary_url is None and "upload_result" in locals():
            cloudinary_url = upload_result.get('secure_url')
            cloudinary_public_id = upload_result.get('public_id')
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
                
                # Lấy public_id của video đã xử lý
                cloudinary_processed_id = processed_result.get("public_id")
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
                cloudinary_processed_id = processed_result.get("public_id")
                
                # Lưu thông tin video vào cơ sở dữ liệu
                if user and user.user_id and db:
                    logger.info(f"Người dùng đã đăng nhập với user_id={user.user_id}, chuẩn bị lưu video vào CSDL")
                    try:
                        # Tạo ID cho video mới
                        video_id = uuid.uuid4()
                        
                        # Xác định loại video
                        if video_type == "youtube":
                            video_type_enum = VideoTypeEnum.YOUTUBE
                            youtube_url_value = youtube_url
                        else:
                            video_type_enum = VideoTypeEnum.UPLOAD
                            youtube_url_value = None
                        
                        # Xác định file_name dựa vào loại video
                        if video_type_enum == VideoTypeEnum.UPLOAD:
                            # Tạo biến tạm để lưu giá trị tên file
                            saved_file_name = None
                            
                            # FIXME: Bắt buộc sử dụng giá trị original_file_name để sửa lỗi
                            if original_file_name:
                                saved_file_name = original_file_name
                                logger.info(f"Sử dụng tên file gốc được cung cấp bởi client: '{saved_file_name}'")
                            else:
                                # Chỉ sử dụng tên mặc định khi hoàn toàn không có tên file từ client
                                saved_file_name = f"uploaded_video_{video_id}.mp4"
                                logger.warning(f"Không có tên file gốc, sử dụng tên mặc định: '{saved_file_name}'")
                            
                            # Ánh xạ rõ ràng
                            file_name = saved_file_name
                            
                            # Log thêm để khẳng định giá trị cuối cùng
                            logger.info(f"Tên file cuối cùng sẽ được lưu vào cơ sở dữ liệu: '{file_name}'")
                        
                        
                        elif video_type_enum == VideoTypeEnum.YOUTUBE and youtube_url_value:
                            # Sử dụng tiêu đề YouTube nếu có
                            if 'youtube_title' in locals() and youtube_title:
                                file_name = youtube_title
                                logger.info(f"Sử dụng tiêu đề YouTube: {file_name}")
                            else:
                                # Nếu không có tiêu đề, sử dụng ID video
                                video_id_part = youtube_url_value.split('v=')[-1] if 'v=' in youtube_url_value else youtube_url_value.split('/')[-1]
                                file_name = f"youtube_{video_id_part}"
                                logger.info(f"Sử dụng YouTube ID: {file_name}")
                        else:
                            file_name = f"video_{video_id}.mp4"
                            logger.info(f"Sử dụng tên mặc định: {file_name}")
                        
                        # Đảm bảo tên file không rỗng trước khi lưu vào database
                        if not file_name and original_file_name:
                            file_name = original_file_name
                            logger.warning(f"Khôi phục tên file từ original_file_name='{original_file_name}'")
                        
                        # Kiểm tra cuối cùng, nếu vẫn không có tên file thì dùng mặc định
                        if not file_name:                        
                            file_name = f"uploaded_video_{video_id}.mp4"
                            logger.warning(f"Vẫn không có tên file, dùng mặc định cuối cùng: '{file_name}'")
                        
                        # Tạo video trong cơ sở dữ liệu
                        logger.info(f"Tạo video với file_name='{file_name}', user_id={user.user_id}, video_type={video_type_enum}")
                        
                        new_video = Video(
                            video_id=video_id,
                            user_id=user.user_id if user else None,
                            video_type=video_type_enum,
                            youtube_url=youtube_url_value,
                            original_video_url=cloudinary_url,
                            processed_video_url=processed_url,
                            status=StatusEnum.COMPLETED,
                            fire_detected=fire_detected,
                            file_name=file_name,
                            cloudinary_public_id=cloudinary_public_id,
                            cloudinary_processed_id=cloudinary_processed_id  # ID video đã xử lý
                        )
                        db.add(new_video)
                        
                        # Thêm lịch sử hoạt động của người dùng
                        user_history = UserHistory(
                            history_id=uuid.uuid4(),
                            user_id=user.user_id,
                            action_type="upload_video_websocket",
                            video_id=video_id,
                            description=f"Tải lên video qua WebSocket. Loại: {video_type_enum}, Phát hiện đám cháy: {'Có' if fire_detected else 'Không'}"
                        )
                        db.add(user_history)
                        
                        # Lưu vào cơ sở dữ liệu
                        db.commit()
                        
                        logger.info(f"Lưu thông tin video thành công, video_id={video_id}")
                    except Exception as db_error:
                        logger.error(f"Lỗi khi lưu thông tin video vào cơ sở dữ liệu: {str(db_error)}")
                        db.rollback()  # Hoàn tác nếu có lỗi
                
                # Kiểm tra xem video có được lưu vào CSDL không
                video_saved = False
                save_message = "Đã xử lý video xong"
                
                if not user or not user.user_id:
                    logger.warning("Không lưu video vào CSDL vì người dùng chưa đăng nhập")
                    save_message = "Đã xử lý video xong. Hãy đăng nhập để lưu video vào tài khoản của bạn"
                else:
                    video_saved = True
                    save_message = "Đã xử lý và lưu video vào tài khoản của bạn"
                
                # Thông báo hoàn thành
                try:
                    await websocket.send_json({
                        "status": "completed",
                        "message": save_message,
                        "original_url": cloudinary_url,
                        "processed_url": processed_url,
                        "fire_detected": fire_detected,
                        "frames_processed": frame_count,
                        "video_saved": video_saved,
                        "requires_login": not video_saved
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


async def download_youtube_video(youtube_url: str, websocket: Optional[WebSocket] = None) -> Tuple[bytes, Optional[str]]:
    """
    Tải video từ YouTube URL và lấy tiêu đề video
    
    Args:
        youtube_url: URL YouTube
        websocket: WebSocket để thông báo tiến độ (nếu có)
        
    Returns:
        Tuple[bytes, Optional[str]]: Dữ liệu video và tiêu đề video (nếu có)
    """
    # Tạo file tạm thời
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_path = temp_file.name
    
    try:
        if websocket:
            await websocket.send_json({"status": "info", "message": "Đang trích xuất thông tin YouTube..."})
        
        # Lấy tiêu đề video YouTube
        video_title = None
        try:
            import yt_dlp
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'noplaylist': True,
                'skip_download': True,  # Chỉ lấy thông tin, không tải video
            }
            # Chạy trong thread riêng để không block
            def extract_info():
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(youtube_url, download=False)
                    return info.get('title') if info else None
                    
            video_title = await asyncio.to_thread(extract_info)
            
            if video_title:
                logger.info(f"Lấy được tiêu đề video YouTube: {video_title}")
                if websocket:
                    await websocket.send_json({"status": "info", "message": f"Lấy được thông tin video: {video_title}"})
        except Exception as e:
            logger.warning(f"Không thể lấy thông tin video YouTube: {str(e)}")
            # Tiếp tục tải video mà không có tiêu đề
        
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
            
        return video_data, video_title
    
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