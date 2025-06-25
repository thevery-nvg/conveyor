import cv2
import time
from multiprocessing import Queue

from src.app.services.utils import get_video_params


def frame_producer(video_path: str, output_queue: Queue,shared_dict):
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    shared_dict.update(get_video_params(cap))
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Ограничиваем размер очереди
            if output_queue.qsize() > 50:
                time.sleep(0.1)
                continue

            frame_data = {
                'frame': frame,
                'timestamp': time.time(),
                'frame_num': frame_num
            }

            output_queue.put(frame_data)
            frame_num += 1

            # Имитация реального времени
            time.sleep(0.04)  # ~25 fps

    finally:
        cap.release()
        output_queue.put(None)