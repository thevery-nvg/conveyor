from collections import deque

import cv2
from ultralytics.engine.results import Results
import numpy as np
from ultralytics import YOLO

def draw_boxes(results,frame):
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

def watch_video():
    cap = cv2.VideoCapture("D:\\Python\\tortillas_static\\data\\video\\vid1.avi")
    pause_mode = False
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видеофайл!")
        exit()
    green = (0, 255, 0)
    red = (0, 0, 255)
    frame_count = 0
    skip_frames = 2
    pause_mode = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("asda", frame)
        while pause_mode:
            key = cv2.waitKey(0)
            if key == ord(' '):  # Пробел - продолжить
                break
            elif key == ord('q'):  # Q - выход
                cap.release()
                cv2.destroyAllWindows()
                exit()
            elif key == ord('p'):  # P - переключить режим паузы
                pause_mode = not pause_mode
                print(f"Режим паузы: {'вкл' if pause_mode else 'выкл'}")
                break
        if not pause_mode:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def measure_tortillas(frame):
    model = YOLO("D:\\Python\\tortillas_static\\src\\yolo_net\\models\\best_for_tracking.pt")
    chess_upper=cv2.imread("D:\\Python\\tortillas_static\\src\\yolo_net\\calib_images_cut\\chess_upper.jpg")
    chess_lower=cv2.imread("D:\\Python\\tortillas_static\\src\\yolo_net\\calib_images_cut\\chess_lower.jpg")
    chess_middle=cv2.imread("D:\\Python\\tortillas_static\\src\\yolo_net\\calib_images_cut\\chess_middle.jpg")
    data = np.load("D:\\Python\\tortillas_static\\src\\yolo_net\\perspective\\camera_calibration_3x5.npz")
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]
    roi_x1, roi_x2 = 550, 1200
    roi_y1, roi_y2 = 230, 850
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    h, w = roi.shape[:2]
    newcameramtx,roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted_pancakes = cv2.undistort(roi, camera_matrix, dist_coeffs, None, newcameramtx)
    cv2.imwrite("undistorted_pancakes.jpg",undistorted_pancakes)
    undistorted_chess=[cv2.undistort(chess_upper, camera_matrix, dist_coeffs, None, newcameramtx),
                       cv2.undistort(chess_middle, camera_matrix, dist_coeffs, None, newcameramtx),
                       cv2.undistort(chess_lower, camera_matrix, dist_coeffs, None, newcameramtx)
                       ]

    results = model.predict(undistorted_pancakes, conf=0.8)
    boxes=[]
    if results[0].boxes.xyxy.cpu() is not None:
        boxes=[box for box in results[0].boxes.xyxy.cpu() if 960 <= int(box[0]) <= 1200]
        boxes=sorted(boxes,key=lambda x:x[1])
    for i,box in enumerate(boxes):
        x1, y1, x2, y2 = box
        x1=int(x1)
        y1=int(y1)
        x2=int(x2)
        y2=int(y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        H=build_homography(undistorted_chess[0])
        side1,side2=measure_box_dimensions(x1,y1,w,h,H)
        print(f"{side1:.2f},{side2:.2f}")

model = YOLO("D:\\Python\\tortillas_static\\src\\yolo_net\\models\\best_for_tracking.pt")
data = np.load("D:\\Python\\tortillas_static\\src\\yolo_net\\perspective\\camera_calibration_3x5.npz")

sizes = {}
img=cv2.imread("D:\\Python\\tortillas_static\\src\\yolo_net\\calib_images_cut\\chess2.jpg")
camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]
def curve_roi(frame):
    # Область интереса
    roi_x1, roi_x2 = 550, 1200
    roi_y1, roi_y2 = 230, 850
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    w, h = roi.shape[:2]
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted = cv2.undistort(roi, camera_matrix, dist_coeffs, None, newcameramtx)
    results = model.track(undistorted, persist=True, tracker="bytetrack.yaml", conf=0.8)
    if results[0].boxes.id is not None:
        # Получаем данные о треках
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        classes = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().tolist()

        for box, track_id, cls, conf in zip(boxes, track_ids, classes, confidences):
            x1, y1, x2, y2 = box
            # x1 = int(x1)
            # y1 = int(y1)
            # x2 = int(x2)
            # y2 = int(y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            size_cm_w = pixels_to_centimeters(w)
            size_cm_h = pixels_to_centimeters(h)
            if 388<x1<400:
                if track_id not in sizes:
                    sizes[track_id] ={"h":size_cm_h,"w":size_cm_w}
            # Рисуем рамку
            cv2.rectangle(undistorted, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"ID: {track_id}"
            cv2.putText(undistorted, label, (int(x1) + 10, int(y1) + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if track_id in sizes:
                cv2.putText(undistorted, f'height {sizes[track_id]["h"]:.2f}cm', (int(x1) + 10, int(y1) + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(undistorted, f"width{sizes[track_id]['w']:.2f}cm", (int(x1) + 10, int(y1) + 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return undistorted


