from threading import BoundedSemaphore, Thread
from time import sleep, time

ticket_office = BoundedSemaphore(value=3)


def ticket_buyer(number):
    start_service = time()
    with ticket_office:
        sleep(1)  # Имитация времени обслуживания
        service_time = time() - start_service
        print(f"client {number}, service time: {service_time:.2f} seconds")


if __name__ == "__main__":
    # Создаем список потоков для 5 клиентов
    buyers = [Thread(target=ticket_buyer, args=(i,)) for i in range(5)]

    # Запускаем все потоки
    for buyer in buyers:
        buyer.start()

    # Ждем завершения всех потоков
    for buyer in buyers:
        buyer.join()

    print("Все клиенты обслужены")
