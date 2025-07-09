from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from multiprocessing import Queue, Process, Manager
import threading
import os
from pathlib import Path

from app.middlewares.track_clients import track_clients_middleware
from app.services.consumer import frame_consumer
from app.services.producer import frame_producer
from app.services.manager import app_state
from src.app.services.consumer import result_consumer
from src.app.views.core import router

current_dir = Path(__file__).resolve().parent
templates_dir = os.path.join(current_dir, "templates")
templates = Jinja2Templates(directory=templates_dir)


@asynccontextmanager
async def lifespan(application: FastAPI):
    # startup
    input_queue = Queue(maxsize=50)
    result_queue = Queue(maxsize=50)
    manager = Manager()
    shared_dict = manager.dict()
    shared_dict['avg_time'] = 0
    shared_dict['count_tortillas'] = 0
    producer = Process(
        target=frame_producer,
        args=("D:\\Python\\tortillas_static\\data\\video\\vid1.avi", input_queue, shared_dict)
    )

    consumer = Process(
        target=frame_consumer,
        args=(input_queue, result_queue, shared_dict)
    )

    threading.Thread(
        target=result_consumer,
        args=(result_queue, shared_dict),
        daemon=True
    ).start()

    producer.start()
    consumer.start()

    app_state.producer = producer
    app_state.consumer = consumer
    app_state.queues = (input_queue, result_queue)
    app_state.shared_dict = shared_dict

    yield

    # shutdown
    input_queue, result_queue = app_state.queues
    input_queue.put(None)
    result_queue.put(None)

    app_state.producer.join()
    app_state.consumer.join()

    if app_state.producer.is_alive():
        app_state.producer.terminate()
    if app_state.consumer.is_alive():
        app_state.consumer.terminate()


app = FastAPI(lifespan=lifespan)
app.middleware("http")(track_clients_middleware)

static_dir = os.path.join(current_dir, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")
app.include_router(router)


@app.get("/", response_class=HTMLResponse)
async def video_page(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": "Stream"}
    )


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
