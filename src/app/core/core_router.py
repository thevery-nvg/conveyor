from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import os
import uuid
from pathlib import Path
from fastapi import APIRouter
from fastapi import Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from app.services.consumer import DISPLAY_BUFFER, generate_frames
from app.services.manager import client_manager, app_state
from app.services.utils import get_gpu_utilization
import psutil


core_router = APIRouter(
    tags=["Core"],
)
current_dir = Path(__file__).resolve().parent.parent
templates_dir = os.path.join(current_dir, "templates")
static_dir = os.path.join(current_dir, "static")

templates = Jinja2Templates(directory=templates_dir)
core_router.mount("/static", StaticFiles(directory=static_dir), name="static")


@core_router.get("/favicon.ico")
async def favicon():
    return RedirectResponse("https://img.icons8.com/3d-fluency/94/globe-africa.png")


@core_router.get("/video_feed")
async def video_feed(request: Request):
    client_id = request.headers.get('X-Client-ID', str(uuid.uuid4()))
    return StreamingResponse(
        generate_frames(client_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={'X-Client-ID': client_id}
    )


@core_router.get("/buffer_status")
async def buffer_status():
    return {
        "buffer_size": len(DISPLAY_BUFFER),
        "delay_seconds": len(DISPLAY_BUFFER) / app_state.shared_dict.get('fps', 25),
        "clients_count": len(client_manager.active_clients),
    }


@core_router.get("/tortilla_stats")
async def tortilla_stats():
    gpu, vram = get_gpu_utilization()
    return {
        "producer_alive": app_state.producer.is_alive(),
        "consumer_alive": app_state.consumer.is_alive(),
        "queue_input": app_state.queues[0].qsize() if app_state.queues else 0,
        "queue_result": app_state.queues[1].qsize() if app_state.queues else 0,
        "gpu": gpu,
        "vRAM": vram,
        "CPU": f"{psutil.cpu_percent()} %",
        "RAM": f"{psutil.virtual_memory().percent} %"
    }


@core_router.get("/", response_class=HTMLResponse)
async def video_page(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": "system > status > video_feed"}
    )

@core_router.get("/login", response_class=HTMLResponse)
async def stats_page(request: Request):
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "title": "system > login"}
    )
