from collections import deque, defaultdict
import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results
import numpy as np
import subprocess

zones = {
    "upper": (380, 0, 580, 195),
    "middle": (380, 200, 580, 405),
    "lower": (380, 410, 580, 610)
}

global_sizes = {}
old=set()
data = np.load("D:\\Python\\conveyor\\src\\models\\camera_calibration_3x5.npz")
camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]

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


red=(0,0,255)
green=(0,255,0)
blue=(255,0,0)
def draw_boxes_from_roi(results: list[Results],frame: np.ndarray,roi_x1:int,roi_y1:int,sizes)->None:
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
            cv2.putText(frame, label, (int(x1) + 10, int(y1) + 30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, green, 2)
            if track_id in sizes:
                w = sizes[track_id].get("w")
                h = sizes[track_id].get("h")
                cv2.putText(frame, f"w:{w:.2f}", (int(x1) + 10, int(y1) + 60),cv2.FONT_HERSHEY_SIMPLEX, 0.6, green, 2)
                cv2.putText(frame, f"h:{h:.2f}", (int(x1) + 10, int(y1) + 90),cv2.FONT_HERSHEY_SIMPLEX, 0.6, green, 2)
                if abs(w-h)<1.91:
                    cv2.putText(frame, f"o:{w-h:.2f}", (int(x1) + 10, int(y1) + 120),cv2.FONT_HERSHEY_SIMPLEX, 0.6, green, 2)
                else:
                    cv2.putText(frame, f"o:{w-h:.2f}", (int(x1) + 10, int(y1) + 120),cv2.FONT_HERSHEY_SIMPLEX, 0.6, red, 2)

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





def calculate_coordinate_ranges(coordinates):
    # Преобразуем входной массив в numpy-массив для удобства
    coords = np.array(coordinates)

    # Извлекаем все x и y
    x = coords[:, 0, 0]  # Все первые элементы каждой пары
    y = coords[:, 0, 1]  # Все вторые элементы каждой пары

    # Находим min и max для x и y
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)

    # Вычисляем разницу
    x_range = max_x - min_x
    y_range = max_y - min_y

    return {
        "width": x_range,
        "height": y_range,
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y
    }


def calculate_contour_dimensions(contour, tolerance=5):
    points = contour.squeeze(1)  # преобразуем в массив (N, 2)

    # Группируем точки по близким Y (для поиска ширины)
    y_groups = defaultdict(list)
    for x, y in points:
        # Находим ближайший существующий Y (с учетом tolerance)
        found = False
        for existing_y in y_groups.keys():
            if abs(y - existing_y) <= tolerance:
                y_groups[existing_y].append(x)
                found = True
                break
        if not found:
            y_groups[y].append(x)

    # Группируем точки по близким X (для поиска высоты)
    x_groups = defaultdict(list)
    for x, y in points:
        # Находим ближайший существующий X (с учетом tolerance)
        found = False
        for existing_x in x_groups.keys():
            if abs(x - existing_x) <= tolerance:
                x_groups[existing_x].append(y)
                found = True
                break
        if not found:
            x_groups[x].append(y)

    # Вычисляем "локальную ширину" для каждого Y
    local_widths = [max(xs) - min(xs) for xs in y_groups.values() if xs]
    max_width = max(local_widths) if local_widths else 0

    # Вычисляем "локальную высоту" для каждого X
    local_heights = [max(ys) - min(ys) for ys in x_groups.values() if ys]
    max_height = max(local_heights) if local_heights else 0

    return {
        'width': max_width,
        'height': max_height,
        # 'min_x': np.min(points[:, 0]),
        # 'max_x': np.max(points[:, 0]),
        # 'min_y': np.min(points[:, 1]),
        # 'max_y': np.max(points[:, 1]),
    }





def is_box_fully_in_zone(box, zone):
    #полностью в зоне
    box_x1, box_y1, box_x2, box_y2 = box
    zone_x1, zone_y1, zone_x2, zone_y2 = zone

    #  все углы бокса внутри зоны
    return (box_x1 >= zone_x1 and #box_x2 <= zone_x2 and
            box_y1 >= zone_y1 and box_y2 <= zone_y2)




def get_sizes_from_contours(frame, zone_ids):
    roi_x1, roi_x2 = 550, 1200
    roi_y1, roi_y2 = 230, 850
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    sizes={}
    # Оптимизация: вычисляем newcameramtx один раз (если разрешение не меняется)
    if not hasattr(get_sizes_from_contours, '_newcameramtx_cache'):
        w, h = roi.shape[:2]
        get_sizes_from_contours._newcameramtx_cache = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )[0]
    undistorted = cv2.undistort(roi, camera_matrix, dist_coeffs, None, get_sizes_from_contours._newcameramtx_cache)
    zone_rois = {}

    # вырезаем зону из распрямленного изображения
    for zone_name, (x1, y1, x2, y2) in zones.items():
        zone_ids = {k: v for k, v in zone_ids.items() if v not in global_sizes}
        if zone_name in zone_ids:
            zone_roi = undistorted[y1:y2, x1:x2]
            zone_rois[zone_name] = zone_roi
    for zone_name, zone_roi in zone_rois.items():
        gray=cv2.cvtColor(zone_roi,cv2.COLOR_BGR2GRAY)
        _, blackened = cv2.threshold(gray, thresh=235, maxval=255, type=cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(blackened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        sizes = calculate_contour_dimensions(cnts[0])
        w = pixels_to_centimeters_ultimate(sizes["width"])
        h = pixels_to_centimeters_ultimate(sizes["height"])
        sizes[zone_ids[zone_name]] = {"w": w, "h": h}

    return sizes







def get_gpu_utilization():
    try:
        result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
        '--format=csv,nounits,noheader'],
        encoding='utf-8'
        )
        lines = result.strip().split('\n')
        for idx, line in enumerate(lines):
            gpu_util, mem_used, mem_total = map(str.strip, line.split(','))
        return f" {idx} {gpu_util}%",f"{mem_used} MiB / {mem_total} MiB"
    except FileNotFoundError:
        return "❌ nvidia-smi not found. Make sure NVIDIA drivers are installed."
    except Exception as e:
        return f"⚠️ Error: {e}"





