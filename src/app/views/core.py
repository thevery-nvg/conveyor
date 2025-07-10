import os
import uuid
from pathlib import Path

from fastapi import APIRouter
from fastapi import Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from services.consumer import DISPLAY_BUFFER, generate_frames
from services.manager import client_manager, app_state
from services.utils import get_gpu_utilization

router = APIRouter()
current_dir = Path(__file__).resolve().parent
templates_dir = os.path.join(current_dir, "templates")
templates = Jinja2Templates(directory=templates_dir)

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
        "delay_seconds": len(DISPLAY_BUFFER) / app_state.shared_dict.get('fps', 25),
        "clients_count": len(client_manager.active_clients),
    }

import psutil

@router.get("/tortilla_stats")
async def tortilla_stats():
    gpu,vram=get_gpu_utilization()
    return {
        "producer_alive": app_state.producer.is_alive(),
        "consumer_alive": app_state.consumer.is_alive(),
        "queue_input": app_state.queues[0].qsize() if app_state.queues else 0,
        "queue_result": app_state.queues[1].qsize() if app_state.queues else 0,
        "gpu":gpu,
        "vRAM":vram,
        "CPU":f"{psutil.cpu_percent()} %",
        "RAM":f"{psutil.virtual_memory().percent} %"
    }
