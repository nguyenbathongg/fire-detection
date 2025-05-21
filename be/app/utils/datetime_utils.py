from datetime import datetime, timedelta

def utcnow_vn():
    """
    Trả về thời gian hiện tại ở múi giờ Việt Nam (UTC+7)
    Hàm này được sử dụng để thay thế datetime.utcnow() trong các model
    """
    return datetime.utcnow() + timedelta(hours=7) 