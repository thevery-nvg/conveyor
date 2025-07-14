from datetime import date

from sqlalchemy import select, desc

from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import os
import uuid
from pathlib import Path
from fastapi import APIRouter, Depends
from fastapi import Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import desc

from app.core.models.db_helper import db_helper
from app.core.models.base import TortillaStats
from app.core.schemas import TortillaStatsResponse

from app.services.consumer import DISPLAY_BUFFER, generate_frames
from app.services.manager import client_manager, app_state
from app.services.utils import get_gpu_utilization
import psutil
from app.auth.auth_routers import current_user
from app.auth.models import User

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
    }


@core_router.get("/tortilla_stats")
async def tortilla_stats(user: User = Depends(current_user),session=Depends(db_helper.session_getter)):
    gpu, vram = get_gpu_utilization()
    today = date.today()
    result = await session.execute(
        select(TortillaStats).where(TortillaStats.date == today)
    )
    stats = result.scalar_one_or_none()
    return {
        "gpu": gpu,
        "vRAM": vram,
        "CPU": f"{psutil.cpu_percent()} %",
        "RAM": f"{psutil.virtual_memory().percent} %",
        "today": stats.date,
        "perfect tortilla": stats.valid,
        "invalid oval": stats.invalid_oval,
        "invalid size": stats.invalid_size,
        "total": stats.valid+stats.invalid_oval+stats.invalid_size
    }


@core_router.get("/", response_class=HTMLResponse)
async def root(request: Request, user: User = Depends(current_user)):
    # if user:
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": "system > status > video_feed","user": user}
    )
    # else:
    #     return RedirectResponse(url="/login")

@core_router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, user: User = Depends(current_user)):
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "title": "system > login","user": user}
    )

@core_router.get("/stats", response_class=HTMLResponse)
async def stats_page(request: Request, user: User = Depends(current_user)):
    return templates.TemplateResponse(
        "stats.html",
        {"request": request, "title": "system > status > stats","user": user}
    )
@core_router.get("/profile", response_class=HTMLResponse)
async def stats_page(request: Request, user: User = Depends(current_user)):
    return templates.TemplateResponse(
        "profile.html",
        {"request": request, "title": "profiles > me","user": user}
    )

@core_router.get("/api/stats",  response_model=list[TortillaStatsResponse])
async def stats_page(user: User = Depends(current_user),session=Depends(db_helper.session_getter)):
    result = await session.execute(
        select(TortillaStats).order_by(desc(TortillaStats.date)).limit(10)
    )
    return result.scalars().all()

