import cv2

def process_stream(url=None):
    cap = cv2.VideoCapture("D:\\Python\\tortillas_static\\data\\video\\vid2.mp4")
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видеофайл!")
        exit()
    pause_mode = True
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("pancakes", frame)
    # Управление воспроизведением
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
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    process_stream(0)
