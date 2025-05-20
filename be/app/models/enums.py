import enum

class RoleEnum(str, enum.Enum):
    USER = "user"
    ADMIN = "admin"

class VideoTypeEnum(str, enum.Enum):
    UPLOAD = "upload"
    YOUTUBE = "youtube"

class StatusEnum(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed" 