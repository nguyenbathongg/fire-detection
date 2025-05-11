import cloudinary
import cloudinary.uploader
import cloudinary.api
import os
import logging
import traceback
from typing import Dict, Optional, Tuple, BinaryIO, Union
import io
import uuid

from app.core.config import settings

logger = logging.getLogger(__name__)

# Khởi tạo Cloudinary (sẽ lấy thông tin từ biến môi trường)
def init_cloudinary():
    try:
        cloudinary.config(
            cloud_name=settings.CLOUDINARY_CLOUD_NAME,
            api_key=settings.CLOUDINARY_API_KEY,
            api_secret=settings.CLOUDINARY_API_SECRET,
            secure=True
        )
        logger.info("Cloudinary đã được khởi tạo thành công")
        logger.info(f"Cloudinary config: cloud_name={settings.CLOUDINARY_CLOUD_NAME}, api_key={settings.CLOUDINARY_API_KEY[:5]}***")
    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo Cloudinary: {str(e)}")
        logger.error(traceback.format_exc())

# Tải video lên Cloudinary
def upload_video_to_cloudinary(file_path: str, resource_type: str = "video") -> Tuple[bool, str, Optional[Dict]]:
    """
    Tải video lên Cloudinary
    
    Args:
        file_path: Đường dẫn tới file cục bộ
        resource_type: Loại tài nguyên (mặc định là "video")
        
    Returns:
        Tuple[bool, str, Optional[Dict]]: 
            - Trạng thái thành công
            - Thông báo
            - Thông tin về video đã tải lên
    """
    try:
        # Kiểm tra file có tồn tại không
        if not os.path.exists(file_path):
            error_msg = f"File không tồn tại: {file_path}"
            logger.error(error_msg)
            return False, error_msg, None
        
        file_size = os.path.getsize(file_path)
        logger.info(f"Tải lên file {file_path} với kích thước {file_size} bytes, loại {resource_type}")
        
        # Thông tin cấu hình Cloudinary
        cloud_config = cloudinary.config()
        logger.info(f"Cloudinary config check: cloud_name={cloud_config.cloud_name}, api_key={cloud_config.api_key[:5]}***")
        
        if resource_type == "video" and file_size < 100:  # Quá nhỏ cho video
            error_msg = f"File quá nhỏ để là video hợp lệ: {file_size} bytes"
            logger.warning(error_msg)
            # Tiếp tục tải lên vì có thể là video demo
        
        # Tải lên Cloudinary
        logger.info(f"Bắt đầu tải lên Cloudinary với folder={settings.CLOUDINARY_FOLDER}")
        result = cloudinary.uploader.upload(
            file_path,
            resource_type=resource_type,
            folder=settings.CLOUDINARY_FOLDER,
            use_filename=True,
            unique_filename=True,
            overwrite=False
        )
        
        logger.info(f"Tải lên Cloudinary thành công: {result.get('public_id')}")
        logger.info(f"URL Cloudinary: {result.get('secure_url')}")
        
        # Nếu tải lên thành công, có thể xóa file cục bộ để tiết kiệm dung lượng
        if os.path.exists(file_path) and settings.DELETE_LOCAL_FILES_AFTER_UPLOAD:
            os.remove(file_path)
            logger.info(f"Đã xóa file cục bộ: {file_path}")
        
        return True, "Tải lên Cloudinary thành công", result
    
    except Exception as e:
        error_details = traceback.format_exc()
        error_msg = f"Lỗi khi tải lên Cloudinary: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Chi tiết lỗi: {error_details}")
        return False, error_msg, None

# Tải video lên Cloudinary trực tiếp từ dữ liệu nhị phân
def upload_bytes_to_cloudinary(file_data: Union[bytes, BinaryIO], filename: str = None, resource_type: str = "video") -> Tuple[bool, str, Optional[Dict]]:
    """
    Tải dữ liệu nhị phân lên Cloudinary mà không cần lưu thành file
    
    Args:
        file_data: Dữ liệu nhị phân của file hoặc FileIO object
        filename: Tên file để Cloudinary sử dụng
        resource_type: Loại tài nguyên (mặc định là "video")
        
    Returns:
        Tuple[bool, str, Optional[Dict]]: 
            - Trạng thái thành công
            - Thông báo
            - Thông tin về video đã tải lên
    """
    try:
        # Tạo tên file nếu không được cung cấp
        if not filename:
            filename = f"{uuid.uuid4()}.mp4"
        
        # Thông tin cấu hình Cloudinary
        cloud_config = cloudinary.config()
        logger.info(f"Cloudinary config check: cloud_name={cloud_config.cloud_name}, api_key={cloud_config.api_key[:5]}***")
        
        # Tải lên Cloudinary
        logger.info(f"Bắt đầu tải lên Cloudinary từ dữ liệu nhị phân, folder={settings.CLOUDINARY_FOLDER}")
        result = cloudinary.uploader.upload(
            file_data,
            resource_type=resource_type,
            folder=settings.CLOUDINARY_FOLDER,
            filename=filename,
            use_filename=True,
            unique_filename=True,
            overwrite=False
        )
        
        logger.info(f"Tải lên Cloudinary thành công: {result.get('public_id')}")
        logger.info(f"URL Cloudinary: {result.get('secure_url')}")
        
        return True, "Tải lên Cloudinary thành công", result
    
    except Exception as e:
        error_details = traceback.format_exc()
        error_msg = f"Lỗi khi tải lên Cloudinary từ dữ liệu nhị phân: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Chi tiết lỗi: {error_details}")
        return False, error_msg, None

# Lấy URL của video từ Cloudinary
def get_cloudinary_url(public_id: str, resource_type: str = "video") -> str:
    """
    Lấy URL của video từ Cloudinary
    
    Args:
        public_id: ID công khai của tài nguyên trên Cloudinary
        resource_type: Loại tài nguyên (mặc định là "video")
        
    Returns:
        str: URL của video
    """
    return cloudinary.CloudinaryVideo(public_id).url

# Tải xuống dữ liệu từ Cloudinary
def download_from_cloudinary(url: str) -> Tuple[bool, str, Optional[bytes]]:
    """
    Tải xuống dữ liệu từ URL Cloudinary thành bytes
    
    Args:
        url: URL của tài nguyên trên Cloudinary
        
    Returns:
        Tuple[bool, str, Optional[bytes]]: 
            - Trạng thái thành công
            - Thông báo
            - Dữ liệu nhị phân
    """
    try:
        import requests
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            return True, "Tải xuống thành công", response.content
        else:
            return False, f"Lỗi khi tải xuống: HTTP {response.status_code}", None
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Lỗi khi tải xuống từ Cloudinary: {str(e)}")
        logger.error(f"Chi tiết lỗi: {error_details}")
        return False, f"Lỗi khi tải xuống từ Cloudinary: {str(e)}", None

# Xóa video khỏi Cloudinary
def delete_from_cloudinary(public_id: str, resource_type: str = "video") -> Tuple[bool, str]:
    """
    Xóa tài nguyên khỏi Cloudinary
    
    Args:
        public_id: ID công khai của tài nguyên trên Cloudinary
        resource_type: Loại tài nguyên (mặc định là "video")
        
    Returns:
        Tuple[bool, str]: Trạng thái thành công và thông báo
    """
    try:
        result = cloudinary.uploader.destroy(public_id, resource_type=resource_type)
        if result.get("result") == "ok":
            return True, "Xóa thành công"
        return False, f"Không thể xóa: {str(result)}"
    
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Lỗi khi xóa khỏi Cloudinary: {str(e)}")
        logger.error(f"Chi tiết lỗi: {error_details}")
        return False, f"Lỗi khi xóa khỏi Cloudinary: {str(e)}" 