import queue
from ultralytics import YOLO
from multiprocessing import Queue
import time
import numpy as np
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cuda.enable_flash_sdp(True)  # Для RTX 30XX+
torch.set_flush_denormal(True)

def frame_consumer(input_queue: Queue, output_queue: Queue):
    #pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
    model = YOLO(model="D:\\Python\\conveyor\\src\\models\\best_for_tracking_512.pt").to("cuda")
    tracker_config = "bytetrack.yaml"
    frame_buffer = []

    try:
        while True:
            try:

                frame_data = input_queue.get(timeout=1.0)
                if frame_data is None:
                    break

                # Ограничиваем размер очереди результатов
                if output_queue.qsize() > 100:
                    continue
                frame = frame_data['frame']
                frame_buffer.append(frame)
                # roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                with torch.no_grad():
                    results = model.track(
                        source=frame,
                        tracker=tracker_config,
                        persist=True,
                        conf=0.5,
                        device='cuda',
                        half=True,  # FP16 ускорение
                        imgsz=512,
                        classes=[0]
                    )
                if results[0].boxes.id is None and len(frame_buffer) > 1:
                    for prev_frame in reversed(frame_buffer[:-1]):
                        prev_results = model.track(
                            source=prev_frame,
                            tracker=tracker_config,
                            persist=True,
                            device='cuda'
                        )
                        if prev_results[0].boxes.id is not None:
                            results = prev_results
                            break
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
        torch.cuda.empty_cache()
        output_queue.put(None)