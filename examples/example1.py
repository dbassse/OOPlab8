from queue import Empty, Queue
from threading import Condition, Thread
from time import sleep

cv = Condition()
q = Queue()


def order_processor(name):
    """Функция обработчика заказов"""
    while True:
        with cv:
            # Ждем, пока очередь не станет непустой
            while q.empty():
                cv.wait()

            try:
                # Получаем заказ из очереди
                order = q.get_nowait()
                print(f"{name}: {order}")

                # Если получили команду "stop", завершаем поток
                if order == "stop":
                    break

            except Empty:
                # Если очередь пуста (хотя мы этого ожидали)
                pass

        # Небольшая пауза для имитации обработки
        sleep(0.1)


if __name__ == "__main__":
    # Создаем и запускаем потоки обработчиков
    threads = []
    for i in range(1, 4):
        thread = Thread(target=order_processor, args=(f"Thread {i}",))
        thread.start()
        threads.append(thread)

    # Добавляем заказы в очередь
    for i in range(10):
        q.put(f"order {i}")
        sleep(0.05)  # Небольшая задержка между добавлением заказов

    # Добавляем стоп-сигналы для каждого потока
    for _ in range(3):
        q.put("stop")

    # Уведомляем все потоки о новых данных
    with cv:
        cv.notify_all()

    # Ожидаем завершения всех потоков
    for thread in threads:
        thread.join()

    print("Все заказы обработаны")
