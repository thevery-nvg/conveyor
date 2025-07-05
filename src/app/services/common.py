import threading
from collections import deque
from multiprocessing import Queue
import cv2
import queue
import time

from app.services.utils import draw_boxes

DISPLAY_BUFFER = deque(maxlen=50)
lock = threading.Lock()

def result_consumer(result_queue: Queue, shared_dict):
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
                    shared_dict['frame_num'] = result['frame_num']
                    draw_boxes(frame,results)
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