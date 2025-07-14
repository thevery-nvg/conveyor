from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from multiprocessing import Queue, Process, Manager
import threading
import os
from pathlib import Path

from starlette.middleware.cors import CORSMiddleware

from app.middlewares.track_clients import track_clients_middleware
from app.services.consumer import frame_consumer
from app.services.producer import frame_producer
from app.services.manager import app_state
from app.services.consumer import result_consumer
from app.core.core_router import core_router


from app.auth.auth_routers import auth_router,users_router


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

    app_state.producer.join(timeout=5)
    app_state.consumer.join(timeout=5)

    app_state.producer.terminate()
    app_state.consumer.terminate()


app = FastAPI(lifespan=lifespan,
              docs_url=None,
              openapi_url=None,
              redoc_url=None,
              )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://0.0.0.0:8000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.middleware("http")(track_clients_middleware)

current_dir = Path(__file__).resolve().parent
static_dir = os.path.join(current_dir, "static")
templates_dir = os.path.join(current_dir, "templates")

templates = Jinja2Templates(directory=templates_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

app.include_router(core_router)
app.include_router(auth_router)
app.include_router(users_router)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
