import queue
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from multiprocessing import Queue, Process,Manager
import cv2
from collections import deque
import threading
import uuid
import time


from app.middlewares.track_clients import track_clients_middleware
from app.services.consumer import frame_consumer
from app.services.producer import frame_producer
from app.services.manager import client_manager,app_state
from src.app.services.utils import draw_boxes_from_roi

templates = Jinja2Templates(directory="templates")

# Глобальные буферы
DISPLAY_BUFFER = deque(maxlen=50)
lock = threading.Lock()

@asynccontextmanager
async def lifespan(application: FastAPI):
    # startup
    input_queue = Queue(maxsize=100)
    result_queue = Queue(maxsize=100)
    manager = Manager()
    shared_dict=manager.dict()
    producer = Process(
        target=frame_producer,
        args=("D:\\Python\\tortillas_static\\data\\video\\vid1.avi", input_queue,shared_dict)
    )

    consumer = Process(
        target=frame_consumer,
        args=(input_queue, result_queue)
    )

    threading.Thread(
        target=result_consumer,
        args=(result_queue,),
        daemon=True
    ).start()

    producer.start()
    consumer.start()

    app_state.producer = producer
    app_state.consumer = consumer
    app_state.queues = (input_queue, result_queue)
    app_state.shared_dict=shared_dict

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
app.mount("/static", StaticFiles(directory="static"), name="static")

def result_consumer(result_queue: Queue):
    roi_x1, roi_x2 = 550, 1800
    roi_y1, roi_y2 = 230, 850
    try:
        while True:
            try:
                result = result_queue.get(timeout=1.0)
                if result is None:
                    break

                with lock:
                    if DISPLAY_BUFFER:
                        DISPLAY_BUFFER.pop()
                    frame = result['frame']
                    results = result['results']
                    draw_boxes_from_roi(results, frame, roi_x1, roi_y1)
                    result['frame'] = frame
                    DISPLAY_BUFFER.append(result)

            except queue.Empty:
                continue
    finally:
        pass


def generate_frames(client_id: str):
    last_frame_num = -1
    while True:
        with lock:
            if not DISPLAY_BUFFER:
                continue
            current_frame = DISPLAY_BUFFER[-1]
            if current_frame['frame_num'] == last_frame_num:
                time.sleep(0.01)
                continue
            last_frame_num = current_frame['frame_num']
            frame = current_frame['frame']

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.get("/", response_class=HTMLResponse)
async def video_page(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": "Stream"}
    )


@app.get("/video_feed")
async def video_feed(request: Request):
    client_id = request.headers.get('X-Client-ID', str(uuid.uuid4()))
    return StreamingResponse(
        generate_frames(client_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={'X-Client-ID': client_id}
    )


@app.get("/buffer_status")
async def buffer_status():
    return {
        "buffer_size": len(DISPLAY_BUFFER),
        "delay_seconds": len(DISPLAY_BUFFER) / app_state.shared_dict.get('fps',25),
        "clients_count": len(client_manager.active_clients)
    }


@app.get("/status")
async def status():
    return {
        "producer_alive": app_state.producer.is_alive(),
        "consumer_alive": app_state.consumer.is_alive(),
        "queues_size": {
            "input": app_state.queues[0].qsize() if app_state.queues else 0,
            "result": app_state.queues[1].qsize() if app_state.queues else 0
        }
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)