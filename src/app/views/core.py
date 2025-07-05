from fastapi import APIRouter
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import uuid


from app.middlewares.track_clients import track_clients_middleware
from app.services.consumer import frame_consumer
from app.services.producer import frame_producer
from app.services.manager import client_manager,app_state
from src.app.services.utils import draw_boxes_from_roi, draw_boxes, draw_boxes_with_recovery
from src.app.services.common import DISPLAY_BUFFER,lock,result_consumer,generate_frames
router = APIRouter()




@router.get("/video_feed")
async def video_feed(request: Request):
    client_id = request.headers.get('X-Client-ID', str(uuid.uuid4()))
    return StreamingResponse(
        generate_frames(client_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={'X-Client-ID': client_id}
    )


@router.get("/buffer_status")
async def buffer_status():
    return {
        "buffer_size": len(DISPLAY_BUFFER),
        "delay_seconds": len(DISPLAY_BUFFER) / app_state.shared_dict.get('fps',25),
        "clients_count": len(client_manager.active_clients),
    }


@router.get("/tortilla_stats")
async def tortilla_stats():
    return {
        "producer_alive": app_state.producer.is_alive(),
        "consumer_alive": app_state.consumer.is_alive(),
        "queue_input": app_state.queues[0].qsize() if app_state.queues else 0,
        "queue_result": app_state.queues[1].qsize() if app_state.queues else 0,}