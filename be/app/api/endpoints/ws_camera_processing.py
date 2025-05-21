import logging
from fastapi import APIRouter, WebSocket

from app.controllers.camera_controller import CameraController

logger = logging.getLogger(__name__)

# Khởi tạo router
ws_router = APIRouter()

# Khởi tạo controller
camera_controller = CameraController()

@ws_router.websocket("/fire")
async def websocket_fire_detection(websocket: WebSocket):
    """
    Endpoint WebSocket cho việc phát hiện đám cháy qua camera
    """
    logger.info("Nhận yêu cầu kết nối camera WebSocket")
    await camera_controller.handle_camera_connection(websocket)