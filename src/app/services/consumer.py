import asyncio
import os
import queue
import threading
import time
from collections import deque, defaultdict
from multiprocessing import Queue
from pathlib import Path
from loguru import logger
import cv2
import torch
from ultralytics import YOLO

from app.core.models.db_helper import db_helper
from app.services.utils import (draw_boxes_from_roi,
                                zones,
                                is_box_fully_in_zone,
                                global_sizes,
                                get_sizes_from_contours, write_stats)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.enable_flash_sdp(True)  # Для RTX 30XX+
torch.set_flush_denormal(True)

roi_x1, roi_x2 = 550, 1800
roi_y1, roi_y2 = 230, 850
DISPLAY_BUFFER = deque(maxlen=50)
lock = threading.Lock()

current_dir = Path(__file__).resolve().parent.parent.parent
models_dir = os.path.join(current_dir, "models")


def frame_consumer(input_queue: Queue, output_queue: Queue, shared_dict):
    model = YOLO(model=os.path.join(models_dir, "best_for_tracking_512.pt")).to("cuda")
    # pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
    tracker_config = "bytetrack.yaml"
    frame_buffer = []

    try:
        while True:
            try:
                frame_data = input_queue.get(timeout=1.0)
                if frame_data is None:
                    break

                # Ограничиваем размер очереди результатов
                if output_queue.qsize() > 50:
                    continue

                frame = frame_data['frame'][roi_y1:roi_y2, roi_x1:roi_x2]

                frame_buffer = [frame, *frame_buffer][:5]
                frame_buffer.append(frame)

                torch.cuda.empty_cache()

                with torch.no_grad(), torch.cuda.amp.autocast(True,):
                    results = model.track(
                        source=frame,
                        tracker=tracker_config,
                        persist=True,
                        conf=0.8,
                        device='cuda',
                        half=True,
                        imgsz=512,
                        verbose=False,
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
                }

                output_queue.put(result_data)

            except queue.Empty:
                continue

    finally:
        del model
        torch.cuda.empty_cache()
        output_queue.put(None)

stats_counter=defaultdict(int)

def result_consumer(result_queue: Queue, shared_dict,loop):
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
                    zone_ids = {}
                    if results[0].boxes.id is not None:
                        boxes = results[0].boxes.xyxy.cpu()
                        track_ids = results[0].boxes.id.int().cpu().tolist()
                        for box, track_id in zip(boxes, track_ids):
                            for zone_name, zone_coords in zones.items():
                                if is_box_fully_in_zone(box, zone_coords):
                                    if track_id not in global_sizes:
                                        zone_ids[zone_name] = track_id
                                    break
                    sizes = get_sizes_from_contours(frame, zone_ids)
                    for d in sizes:
                        valid_oval=sizes[d].get("valid_oval")
                        valid_size=sizes[d].get("valid_size")
                        if valid_oval is not None and valid_size is not None:
                            if not valid_oval:
                                stats_counter['invalid_oval']+=1
                            elif not valid_size:
                                stats_counter['invalid_size']+=1
                            else:
                                stats_counter['valid']+=1
                        if sum(stats_counter.values())>50:
                            s=stats_counter.copy()
                            asyncio.run_coroutine_threadsafe(write_stats(db_helper.session_factory, s),loop)
                            stats_counter.clear()
                            logger.info("Stats written to db")

                    global_sizes.update(sizes)
                    draw_boxes_from_roi(results, frame, roi_x1, roi_y1, global_sizes)
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
