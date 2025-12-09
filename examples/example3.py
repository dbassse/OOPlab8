from threading import Event, Thread
from time import sleep, time

event = Event()


def worker(name: str):
    print(f"Worker {name}: waiting for event...")
    event.wait()  # Ждем, пока событие не будет установлено
    print(f"Worker {name}: started at {time():.2f}")


if __name__ == "__main__":
    # Очищаем событие (устанавливаем в состояние "не произошло")
    event.clear()

    # Создаем и запускаем рабочих
    workers = [Thread(target=worker, args=(f"wrk {i}",)) for i in range(5)]
    for w in workers:
        w.start()

    print("Main thread: workers are waiting, starting in 2 seconds...")
    sleep(2)  # Имитация работы основного потока

    # Устанавливаем событие - все потоки могут продолжить работу
    print("Main thread: setting event")
    event.set()

    # Ждем завершения всех рабочих потоков
    for w in workers:
        w.join()

    print("All workers finished")
