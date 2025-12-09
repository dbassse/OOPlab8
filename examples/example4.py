from threading import Timer
from time import time

# Создаем таймер, который выполнит функцию через 3 секунды
timer = Timer(interval=3, function=lambda: print("Message from Timer!"))
print(f"Timer created at {time():.2f}, will execute in 3 seconds")

timer.start()  # Запускаем таймер

# Основной поток продолжает работу
print("Main thread continues to work...")

# Ждем завершения таймера
timer.join()
print(f"Timer finished at {time():.2f}")
