from threading import Barrier, Thread
from time import sleep

br = Barrier(3)  # Барьер для 3 потоков
store = []


def f1(x):
    print(f"f1: Calculating part1 for x={x}")
    result = x**2
    sleep(0.5)  # Имитация вычислений
    store.append(result)
    print(f"f1: Added {result} to store")
    br.wait()  # Ждем остальные потоки
    print("f1: Barrier passed, continuing...")


def f2(x):
    print(f"f2: Calculating part2 for x={x}")
    result = x * 2
    sleep(1)  # Имитация вычислений
    store.append(result)
    print(f"f2: Added {result} to store")
    br.wait()  # Ждем остальные потоки
    print("f2: Barrier passed, continuing...")


if __name__ == "__main__":
    print("Main: Starting threads...")

    # Сохраняем ссылки на потоки для последующего join
    t1 = Thread(target=f1, args=(3,))
    t2 = Thread(target=f2, args=(7,))

    t1.start()
    t2.start()

    print("Main: Waiting at barrier...")
    br.wait()  # Главный поток тоже ждет у барьера

    print("Main: Barrier passed, all calculations done")
    print(f"Main: Store contents: {store}")
    print(f"Main: Result (sum): {sum(store)}")

    # Ждем завершения потоков
    t1.join()
    t2.join()

    print("Main: All threads finished")
