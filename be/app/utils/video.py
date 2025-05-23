import os
import uuid
import logging
import traceback
from typing import Optional, Tuple, BinaryIO, Union
import requests
import io
import yt_dlp

from fastapi import UploadFile, HTTPException, status

from app.core.config import settings
from app.models.enums import VideoTypeEnum
from app.utils.cloudinary_service import upload_bytes_to_cloudinary, download_from_cloudinary

logger = logging.getLogger(__name__)


def download_youtube_video_file(url: str, output_path: str = 'video.mp4') -> str:
    """
    Tải video từ YouTube về máy tính.
    Ưu tiên định dạng mp4 progressive 18 (360p), sau đó thử best[ext=mp4], cuối cùng là best.
    :param url: Đường dẫn URL của video YouTube cần tải.
    :param output_path: Đường dẫn và tên file video tải về. Mặc định là 'video.mp4'.
    :return: Đường dẫn tới file video tải về.
    """
    import yt_dlp
    
    # Xóa file cũ nếu tồn tại để tránh thông báo "file đã tải về"
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
            logger.debug(f"Xóa file cũ {output_path} trước khi tải")
        except Exception as e:
            logger.warning(f"Không thể xóa file cũ: {e}")
    
    # Chỉ dùng định dạng 18 (360p) và các định dạng dự phòng
    formats_to_try = [
        '18',              # mp4 360p progressive (YouTube Shorts luôn có định dạng này)
        'best[ext=mp4]',   # mp4 tốt nhất (có thể là DASH)
        'best'             # bất kỳ định dạng tốt nhất
    ]
    last_error = None
    
    # Kiểm tra nếu là YouTube Shorts
    is_shorts = 'shorts' in url.lower()
    logger.debug(f"Đang tải video {'YouTube Shorts' if is_shorts else 'YouTube thường'} từ URL: {url}")
    
    # Nếu là YouTube Shorts, chỉ thử định dạng 18 trước
    if is_shorts:
        logger.info("Đây là YouTube Shorts, chỉ thử định dạng 18 (360p)")
        
        # Thử tải với định dạng '18' (Chỉ dùng định dạng này cho Shorts)
        fmt = '18'
        
        ydl_opts = {
            'format': fmt,
            'outtmpl': output_path,
            'quiet': False,
            'noplaylist': True,
            'overwrites': True  # Ghi đè file nếu tồn tại
        }
        
        try:
            logger.debug(f"Tải YouTube Shorts với định dạng {fmt}")
            
            # Tải video
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=True)
                actual_format = info_dict.get('format_id', fmt) if info_dict else fmt
                logger.debug(f"Thông tin từ yt-dlp: format_id = {actual_format}")
            
            # Kiểm tra file đã tải
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                logger.info(f"### THÀNH CÔNG: Video Shorts tải về với định dạng {actual_format}, kích thước: {file_size_mb:.2f} MB ###")
                return output_path
        except Exception as e:
            last_error = str(e)
            logger.error(f"Không thể tải YouTube Shorts với định dạng {fmt}: {e}")
            # Nếu là Shorts nhưng không tải được định dạng 18, thử các định dạng khác
    
    # Nếu không phải Shorts hoặc Shorts thất bại, thử lần lượt các định dạng
    logger.debug(f"Thử tải video YouTube với các định dạng theo thứ tự: {formats_to_try}")
    
    for i, fmt in enumerate(formats_to_try):
        # Đảm bảo file cũ đã bị xóa để tránh thông báo "đã tải rồi"
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                logger.debug(f"Xóa file để chuẩn bị cho lần thử mới")
            except Exception:
                pass
        
        ydl_opts = {
            'format': fmt,
            'outtmpl': output_path,
            'quiet': False,
            'noplaylist': True,
            'overwrites': True  # Ghi đè file nếu tồn tại
        }
        
        try:
            logger.debug(f"Lần thử {i+1}: Đang thử tải với định dạng {fmt}")
            
            # Tải video
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=True)
                actual_format = info_dict.get('format_id', fmt) if info_dict else fmt
                logger.debug(f"Thông tin từ yt-dlp: format_id = {actual_format}")
            
            # Kiểm tra file có tồn tại và không rỗng
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                logger.info(f"### THÀNH CÔNG: Video tải về với định dạng YEU CẦU={fmt}, FORMAT THỰC TẾ={actual_format}, kích thước: {file_size_mb:.2f} MB ###")
                return output_path
        except Exception as e:
            last_error = str(e)
            logger.warning(f"Lần thử {i+1}: Không tải được video với định dạng {fmt}: {e}")
    
    logger.error(f"Tất cả các định dạng đều thất bại")
    raise RuntimeError(f"Không thể tải video từ YouTube (không tìm được định dạng mp4 phù hợp hoặc file rỗng). Lỗi cuối: {last_error}")


async def save_upload_file(file: UploadFile, filename: Optional[str] = None) -> Tuple[str, str]:
    """
    Lưu file upload trực tiếp lên Cloudinary không sử dụng file tạm
    
    Args:
        file: File từ UploadFile của FastAPI
        filename: Tên file tùy chọn, nếu không cung cấp sẽ tạo UUID
        
    Returns:
        Tuple[str, str]: URL Cloudinary và public_id
    """
    try:
        # Tạo tên file nếu không được cung cấp
        if not filename:
            ext = os.path.splitext(file.filename)[1] if file.filename else ".mp4"
            filename = f"{uuid.uuid4()}{ext}"
        
        # Đọc nội dung file vào bộ nhớ
        file_content = await file.read()
        
        try:
            # Tải lên Cloudinary trực tiếp từ bộ nhớ
            logger.info(f"Bắt đầu tải lên Cloudinary: {filename}")
            success, message, result = upload_bytes_to_cloudinary(file_content, filename=filename)
            logger.info(f"Kết quả tải lên Cloudinary: {success}, {message}")
            
            if not success:
                raise Exception(message)
            
            # Trả về URL Cloudinary
            cloud_url = result.get("secure_url")
            logger.info(f"Video đã được tải lên Cloudinary: {cloud_url}")
            
            # Lưu file id để tham chiếu sau này
            cloudinary_public_id = result.get("public_id")
            
            return cloud_url, cloudinary_public_id
        
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Lỗi khi tải lên Cloudinary: {str(e)}")
            logger.error(f"Chi tiết lỗi: {error_details}")
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Lỗi khi tải lên video: {str(e)}",
            )
    
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Lỗi khi tải lên Cloudinary: {str(e)}")
        logger.error(f"Chi tiết lỗi: {error_details}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi tải lên video: {str(e)}",
        )


def download_youtube_video(youtube_url: str) -> Tuple[bool, str, Optional[str], Optional[str], Optional[str]]:
    """
    Tải video từ YouTube và upload trực tiếp lên Cloudinary
    
    Args:
        youtube_url: URL YouTube
        
    Returns:
        Tuple[bool, str, Optional[str], Optional[str], Optional[str]]: 
            - Trạng thái thành công
            - Thông báo
            - URL Cloudinary nếu tải thành công
            - Cloudinary public_id
            - Tiêu đề video YouTube
    """
    import tempfile
    try:
        # Lấy thông tin video, bao gồm tiêu đề
        video_title = None
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'noplaylist': True,
                'skip_download': True,  # Chỉ lấy thông tin, không tải video
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                if info and 'title' in info:
                    video_title = info['title']
                    logger.info(f"Thông tin video YouTube: {video_title}")
        except Exception as e:
            logger.warning(f"Không thể lấy thông tin video YouTube: {str(e)}")
            # Tiếp tục tải video mà không có tiêu đề
            
        # Tải video thật từ YouTube về file tạm
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        try:
            download_youtube_video_file(youtube_url, tmp_path)
            # Kiểm tra file đã tải về có tồn tại và dung lượng hợp lệ không
            if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
                logger.error(f"File tải về không hợp lệ hoặc rỗng: {tmp_path}")
                return False, "Không thể tải video từ YouTube (file rỗng)", None, None, None
            # Đọc nội dung file vừa tải
            with open(tmp_path, 'rb') as f:
                video_bytes = f.read()
            # Upload lên Cloudinary
            filename = f"youtube_{uuid.uuid4()}.mp4"
            success, message, result = upload_bytes_to_cloudinary(video_bytes, filename=filename)
        finally:
            # Xóa file tạm
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        if not success:
            return False, message, None, None, None
        cloud_url = result.get("secure_url")
        public_id = result.get("public_id")
        logger.info(f"Video YouTube đã được tải lên Cloudinary: {cloud_url}")
        return True, "Tải video YouTube thành công", cloud_url, public_id, video_title
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Lỗi khi tải video YouTube: {str(e)}")
        logger.error(f"Chi tiết lỗi: {error_details}")
        return False, f"Lỗi khi tải video: {str(e)}", None, None, None


def upload_video_to_cloudinary_and_cleanup(local_path: str, folder: str = None):
    """
    Upload file video đã tải lên Cloudinary, sau đó xoá file tạm local.
    :param local_path: Đường dẫn file video local
    :param folder: Thư mục Cloudinary (nếu có)
    :return: (cloud_url, public_id)
    :raises: Exception nếu upload thất bại
    """
    from app.utils.cloudinary_service import upload_file_to_cloudinary
    import os
    try:
        cloud_url, public_id = upload_file_to_cloudinary(local_path, folder)
    finally:
        # Luôn xoá file local sau khi upload (kể cả khi upload lỗi)
        if os.path.exists(local_path):
            try:
                os.remove(local_path)
            except Exception as cleanup_err:
                import logging
                logging.getLogger(__name__).warning(f"Không thể xoá file tạm: {local_path}: {cleanup_err}")
    return cloud_url, public_id

def get_absolute_file_path(relative_path: str) -> str:
    """
    Chuyển đổi đường dẫn tương đối thành đường dẫn tuyệt đối
    
    Args:
        relative_path: Đường dẫn tương đối
        
    Returns:
        str: Đường dẫn tuyệt đối
    """
    # Xóa ký tự '/' ở đầu nếu có
    if relative_path.startswith('/'):
        relative_path = relative_path[1:]
    
    return os.path.abspath(os.path.join(os.getcwd(), relative_path)) 