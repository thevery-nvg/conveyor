from multiprocessing import Process
from multiprocessing import Queue

from src.app.services.consumer import *
from src.app.services.producer import *

if __name__ == "__main__":

    frame_queue = Queue(maxsize=25)
    result_queue = Queue()

    producer = Process(
        target=frame_producer,
        args=(0, frame_queue, 65)  # 0 - камера, 65 - пропуск кадров
    )

    consumer = Process(
        target=frame_consumer,
        args=(frame_queue, result_queue)
    )

    producer.start()
    consumer.start()

    # Мониторинг результатов (можно вынести в отдельный процесс)
    try:
        while True:
            result = result_queue.get()
            if result is None:
                break
            print(f"Обработан кадр {result['frame_num']}")
            # Здесь можно сохранять результаты на диск и т.д.
    except KeyboardInterrupt:
        pass
    finally:
        producer.terminate()
        consumer.terminate()
        producer.join()
        consumer.join()