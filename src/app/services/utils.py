from collections import deque
import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results
import numpy as np


def draw_boxes(frame,results):
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        classes = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().tolist()
        for box, track_id, cls, conf in zip(boxes, track_ids, classes, confidences):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"ID: {track_id}"
            cv2.putText(frame, label, (int(x1) + 10, int(y1) + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)



def draw_boxes_from_roi(results: list[Results],frame: np.ndarray,roi_x1:int,roi_y1:int)->None:
    if results[0].boxes.id is not None:
        # Получаем данные о треках
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        classes = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().tolist()
        for box, track_id, cls, conf in zip(boxes, track_ids, classes, confidences):
            x1, y1, x2, y2 = box
            x1=roi_x1+int(x1)
            y1=roi_y1+int(y1)
            x2=roi_x1+int(x2)
            y2=roi_y1+int(y2)


            # Рисуем рамку
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Добавляем информацию об объекте
            label = f"ID: {track_id}"
            cv2.putText(frame, label, (int(x1) + 10, int(y1) + 30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

TRACK_HISTORY = deque(maxlen=30)
def draw_boxes_with_recovery(frame, results):
    global TRACK_HISTORY

    if results[0].boxes.id is not None:
        TRACK_HISTORY.append(results)
    elif len(TRACK_HISTORY) > 0:
        results = TRACK_HISTORY[-1]
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        classes = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().tolist()
        for box, track_id, cls, conf in zip(boxes, track_ids, classes, confidences):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"ID: {track_id}"
            cv2.putText(frame, label, (int(x1) + 10, int(y1) + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

def get_video_params(cap: cv2.VideoCapture)-> dict:
    # Получение параметров видео
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Ширина кадра
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Высота кадра
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Общее количество кадров
    fps = cap.get(cv2.CAP_PROP_FPS)  # Кадров в секунду
    duration = total_frames / fps if fps > 0 else 0  # Длительность видео
    return {"width": frame_width, "height": frame_height, "total_frames": total_frames, "fps": fps, "duration": duration}

def pixels_to_centimeters(pixels, max_pixels=116, max_length_cm=21):
    """
    Переводит пиксели в сантиметры.

    :param pixels: Значение в пикселях, которое нужно перевести.
    :param max_pixels: Максимальное количество пикселей, соответствующее max_length_cm.
    :param max_length_cm: Максимальная длина в сантиметрах, соответствующая max_pixels.
    :return: Длина в сантиметрах.
    """
    # Вычисляем коэффициент перевода
    conversion_factor = max_length_cm / max_pixels
    # Переводим пиксели в сантиметры
    centimeters = pixels * conversion_factor
    return float(centimeters)


def pixels_to_centimeters_ultimate(pixels):
    return (pixels*1.8451)/10




model = YOLO(model="D:\\Python\\conveyor\\src\\models\\best_for_tracking_512.pt").to("cuda")
data = np.load("D:\\Python\\tortillas_static\\src\\yolo_net\\perspective\\camera_calibration_3x5.npz")
def curve_roi(frame):
    roi_x1, roi_x2 = 550, 1200
    roi_y1, roi_y2 = 230, 850
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]
    w, h = roi.shape[:2]
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted = cv2.undistort(roi, camera_matrix, dist_coeffs, None, newcameramtx)
    results = model.predict(undistorted, conf=0.8, classes=[0])
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu()
        for box in boxes:
            x1, y1, x2, y2 = box
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            if 388<x1<405 and 570<x2<583:
                w=pixels_to_centimeters_ultimate(w)
                h=pixels_to_centimeters_ultimate(h)




