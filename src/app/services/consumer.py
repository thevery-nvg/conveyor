import queue
from ultralytics import YOLO
from multiprocessing import Queue
import time

def frame_consumer(input_queue: Queue, output_queue: Queue):
    model = YOLO("D:\\Python\\tortillas_static\\src\\yolo_net\\models\\best_for_tracking.pt")
    roi_x1, roi_x2 = 550, 1800
    roi_y1, roi_y2 = 230, 850
    try:
        while True:
            try:
                frame_data = input_queue.get(timeout=1.0)
                if frame_data is None:
                    break

                # Ограничиваем размер очереди результатов
                if output_queue.qsize() > 20:
                    continue
                roi = frame_data['frame'][roi_y1:roi_y2, roi_x1:roi_x2]
                results = model.track(roi, persist=True, conf=0.8)

                result_data = {
                    'frame': frame_data['frame'],
                    'results': results,
                    'timestamp': frame_data['timestamp'],
                    'frame_num': frame_data['frame_num'],
                    'processing_time': time.time() - frame_data['timestamp']
                }

                output_queue.put(result_data)

            except queue.Empty:
                continue

    finally:
        del model
        output_queue.put(None)