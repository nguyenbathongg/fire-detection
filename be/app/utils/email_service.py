import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

def send_email(
    to_email: str,
    subject: str,
    html_content: str,
    cc_emails: Optional[List[str]] = None
) -> bool:
    """
    Gửi email thông báo
    
    Args:
        to_email: Địa chỉ email người nhận
        subject: Tiêu đề email
        html_content: Nội dung HTML của email
        cc_emails: Danh sách email CC (nếu có)
        
    Returns:
        bool: Kết quả gửi email thành công hay không
    """
    try:
        logger.info(f"EMAIL_USERNAME={settings.EMAIL_USERNAME}, EMAIL_PASSWORD={'***' if settings.EMAIL_PASSWORD else ''}, EMAIL_SENDER={settings.EMAIL_SENDER}, EMAIL_HOST={settings.EMAIL_HOST}, EMAIL_PORT={settings.EMAIL_PORT}, EMAIL_USE_TLS={settings.EMAIL_USE_TLS}")
        # Tạo message
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = settings.EMAIL_SENDER
        message["To"] = to_email
        
        # Thêm CC nếu có
        if cc_emails:
            message["Cc"] = ", ".join(cc_emails)
            
        # Đính kèm nội dung HTML
        html_part = MIMEText(html_content, "html")
        message.attach(html_part)
        
        # Kết nối đến SMTP server
        with smtplib.SMTP(settings.EMAIL_HOST, settings.EMAIL_PORT) as server:
            if settings.EMAIL_USE_TLS:
                server.starttls()
            
            # Đăng nhập nếu cần
            if settings.EMAIL_USERNAME and settings.EMAIL_PASSWORD:
                server.login(settings.EMAIL_USERNAME, settings.EMAIL_PASSWORD)
            
            # Tạo danh sách email nhận
            recipients = [to_email]
            if cc_emails:
                recipients.extend(cc_emails)
                
            # Gửi email
            server.sendmail(
                settings.EMAIL_SENDER,
                recipients,
                message.as_string()
            )
            
        logger.info(f"Đã gửi email thông báo thành công đến {to_email}")
        return True
        
    except Exception as e:
        logger.error(f"Lỗi khi gửi email: {str(e)}")
        return False

def send_fire_detection_notification(
    user_email: str,
    video_title: str,
    detection_time: str,
    video_url: str,
    processed_video_url: str,
    confidence: float
) -> bool:
    """
    Gửi thông báo qua email khi phát hiện đám cháy
    
    Args:
        user_email: Email người dùng
        video_title: Tên/tiêu đề video
        detection_time: Thời điểm phát hiện đám cháy
        video_url: URL video gốc
        processed_video_url: URL video đã xử lý
        confidence: Độ tin cậy của phát hiện
        
    Returns:
        bool: Kết quả gửi email thành công hay không
    """
    subject = f"Cảnh Báo Cháy: Kiểm Tra Ngay!"

    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #222; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background-color: #f5f5f5; color: #333; padding: 10px 20px; text-align: left; }}
            .content {{ padding: 20px; background-color: #fafbfc; }}
            .video-container {{ margin: 20px 0; }}
            .button {{ display: inline-block; padding: 10px 20px; background-color: #1976d2; color: white; 
                      text-decoration: none; border-radius: 5px; margin-top: 10px; }}
            .footer {{ font-size: 12px; text-align: center; margin-top: 30px; color: #888; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h2>Xin chào {user_email},</h2>
            </div>
            <div class="content">
                <p>Chúng tôi gửi thông tin về video <b>{video_title}</b> mà bạn đã tải lên hệ thống vào lúc <b>{detection_time}</b>.</p>
                <p>Hệ thống phát hiện có dấu hiệu bất thường với độ tin cậy <b>{confidence:.2%}</b>.</p>
                <ul>
                    <li>Tên video: {video_title}</li>
                    <li>Thời điểm kiểm tra: {detection_time}</li>
                </ul>
                <div class="video-container">
                    <p>Bạn có thể xem lại video:</p>
                    <a href="{video_url}" class="button">Xem video gốc</a>
                    <a href="{processed_video_url}" class="button">Xem video đã xử lý</a>
                </div>
                <p>Nếu bạn không phải là người nhận dự kiến hoặc có bất kỳ thắc mắc nào, vui lòng liên hệ lại với chúng tôi qua email: <a href="mailto:nguyennbaathongg@gmail.com">nguyennbaathongg@gmail.com</a>.</p>
            </div>
            <div class="footer">
                <p>Email này được gửi tự động từ hệ thống phát hiện video. Nếu bạn không mong muốn nhận được thông báo này, xin vui lòng bỏ qua.</p>
                <p>Địa chỉ gửi: nguyennbaathongg@gmail.com</p>
            </div>
        </div>
    </body>
    </html>
    """

    return send_email(to_email=user_email, subject=subject, html_content=html_content)
