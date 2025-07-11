import time
from datetime import datetime
from multiprocessing import Queue

import cv2
from app.services.utils import get_video_params


def frame_producer(video_path: str, output_queue: Queue,shared_dict):
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    shared_dict.update(get_video_params(cap))
    skip_frames = 1
    roi_x1, roi_x2 = 550, 1800
    roi_y1, roi_y2 = 230, 850
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            roi_frame = frame.copy()#[roi_y1:roi_y2, roi_x1:roi_x2].copy()
            del frame
            # Ограничиваем размер очереди
            if output_queue.qsize() > 50:
                time.sleep(0.1)
                continue
            if frame_num % (skip_frames + 1) == 1:
                cv2.putText(roi_frame, str(frame_num), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                frame_data = {
                    'frame': roi_frame,
                    'timestamp': time.time(),
                    'frame_num': frame_num,
                    "time": datetime.now()
                }
                output_queue.put(frame_data)
            frame_num += 1

            # Имитация реального времени
            time.sleep(0.04)  # ~25 fps

    finally:
        cap.release()
        output_queue.put(None)