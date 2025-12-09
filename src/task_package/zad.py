import math
import queue
import threading


class SeriesCalculator:
    """Класс для вычисления суммы ряда с заданной точностью"""

    def __init__(self, epsilon: float = 1e-7):
        self.epsilon = epsilon

    def calculate_s(self, x: float) -> float:
        """Вычисляет сумму ряда S = Σ cos(nx)/n с точностью epsilon"""
        result = 0.0
        n = 1
        term = math.cos(n * x) / n

        while abs(term) >= self.epsilon:
            result += term
            n += 1
            term = math.cos(n * x) / n
            # Защита от бесконечного цикла для кратных 2π случаев
            if n > 1000000:  # Практический предел
                break

        return result


class PipelineThreads:
    """Организация конвейера вычислений"""

    def __init__(self, epsilon: float = 1e-7):
        self.epsilon = epsilon
        self.series_calc = SeriesCalculator(epsilon)
        self.queue = queue.Queue[float]()  # Типизированная очередь
        self.lock = threading.Lock()

    def calculate_first_function(self, x: float) -> None:
        """Первая функция конвейера: вычисление суммы ряда"""
        try:
            result = self.series_calc.calculate_s(x)
            self.queue.put(result)
        except Exception as e:
            print(f"Ошибка в первой функции: {e}")
            self.queue.put(float("nan"))

    def calculate_second_function(self, x: float) -> None:
        """Вторая функция конвейера: вычисление y"""
        try:
            # Ожидаем результат от первой функции
            s_result = self.queue.get(timeout=10.0)  # Таймаут 10 секунд

            if math.isnan(s_result):
                print("Первая функция вернула некорректный результат")
                return

            # Вычисляем вторую функцию
            # Избегаем отрицательного аргумента логарифма
            sin_arg = 2 * math.sin(x / 2)
            if sin_arg <= 0:
                y_result = float("nan")
                print("Внимание: аргумент логарифма неположительный")
            else:
                y_result = -math.log(sin_arg)

            # Выводим результаты
            with self.lock:
                print(f"Результат первой функции S(x) = {s_result:.10f}")
                print(f"Результат второй функции y(x) = {y_result:.10f}")

                if not (math.isnan(s_result) or math.isnan(y_result)):
                    difference = abs(s_result - y_result)
                    print(f"Разность |S - y| = {difference:.10f}")

                    if difference < self.epsilon:
                        print("Результаты совпадают с заданной точностью!")
                    else:
                        print("Результаты не совпадают с заданной точностью")

        except queue.Empty:
            print("Ошибка: первая функция не вернула результат за отведенное время")
        except Exception as e:
            print(f"Ошибка во второй функции: {e}")


def run_pipeline(x: float, epsilon: float = 1e-7) -> None:
    """Запуск конвейера вычислений"""

    print(f"\nВычисление для x = {x:.6f} (π = {math.pi:.6f})")
    print(f"Точность ε = {epsilon}")
    print("-" * 50)

    pipeline = PipelineThreads(epsilon)

    # Создаем потоки
    thread1 = threading.Thread(target=pipeline.calculate_first_function, args=(x,))

    thread2 = threading.Thread(target=pipeline.calculate_second_function, args=(x,))

    # Запускаем потоки одновременно
    thread1.start()
    thread2.start()

    # Ждем завершения потоков
    thread1.join(timeout=30.0)
    thread2.join(timeout=30.0)

    if thread1.is_alive():
        print("Предупреждение: первая функция не завершилась за 30 секунд")
    if thread2.is_alive():
        print("Предупреждение: вторая функция не завершилась за 30 секунд")

    print("-" * 50)


def main() -> None:
    """Основная функция"""

    epsilon = 1e-7

    # Вычисление для x = π
    x_pi = math.pi
    run_pipeline(x_pi, epsilon)

    # Дополнительные тесты для других значений x
    print("\n" + "=" * 50)
    print("Дополнительные тесты:")
    print("=" * 50)

    test_values = [math.pi / 2, math.pi / 3, math.pi / 4, 2 * math.pi / 3]

    for i, x in enumerate(test_values, 1):
        print(f"\nТест {i}: x = {x:.6f} ({x/math.pi:.3f}π)")
        run_pipeline(x, epsilon)


if __name__ == "__main__":
    main()
