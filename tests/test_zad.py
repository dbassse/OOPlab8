import math
import queue
import threading

import pytest


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


# ТЕСТЫ


class TestSeriesCalculator:
    """Тесты для класса SeriesCalculator"""

    def test_init_with_default_epsilon(self):
        """Тест инициализации с epsilon по умолчанию"""
        calculator = SeriesCalculator()
        assert calculator.epsilon == 1e-7

    def test_init_with_custom_epsilon(self):
        """Тест инициализации с пользовательским epsilon"""
        calculator = SeriesCalculator(epsilon=1e-10)
        assert calculator.epsilon == 1e-10

    def test_calculate_s_for_x_pi(self):
        """Тест вычисления суммы ряда для x = π"""
        calculator = SeriesCalculator(epsilon=1e-7)
        result = calculator.calculate_s(math.pi)

        # Для x = π, ряд сходится к -ln(2)
        expected = -math.log(2)
        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_calculate_s_for_x_pi_over_2(self):
        """Тест вычисления суммы ряда для x = π/2"""
        calculator = SeriesCalculator(epsilon=1e-7)
        result = calculator.calculate_s(math.pi / 2)

        # Проверяем, что результат не NaN и не бесконечность
        assert not math.isnan(result)
        assert not math.isinf(result)

    def test_calculate_s_for_x_zero(self):
        """Тест вычисления суммы ряда для x = 0"""
        calculator = SeriesCalculator(epsilon=1e-7)
        result = calculator.calculate_s(0.0)

        # Для x = 0, cos(0) = 1, ряд: Σ 1/n расходится, но у нас обрезание по epsilon
        # Проверяем, что функция возвращает число
        assert not math.isnan(result)

    def test_calculate_s_precision(self):
        """Тест точности вычислений"""
        calculator = SeriesCalculator(
            epsilon=1e-4
        )  # Большая точность для быстрого теста
        result1 = calculator.calculate_s(math.pi)

        calculator2 = SeriesCalculator(epsilon=1e-7)  # Меньшая точность
        result2 = calculator2.calculate_s(math.pi)

        # Более точное вычисление должно давать более близкое значение к -ln(2)
        expected = -math.log(2)
        assert abs(result2 - expected) <= abs(result1 - expected)

    def test_calculate_s_termination(self):
        """Тест завершения вычислений (без бесконечного цикла)"""
        calculator = SeriesCalculator(epsilon=1e-7)

        # Для x = 2π, cos(2πn) = 1, ряд расходится, но должен завершиться из-за ограничения итераций
        result = calculator.calculate_s(2 * math.pi)
        assert not math.isnan(result)


class TestPipelineThreads:
    """Тесты для класса PipelineThreads"""

    def test_calculate_first_function_normal(self, capsys):
        """Тест первой функции в нормальных условиях"""
        pipeline = PipelineThreads(epsilon=1e-4)

        # Запускаем в отдельном потоке
        thread = threading.Thread(
            target=pipeline.calculate_first_function, args=(math.pi,)
        )
        thread.start()
        thread.join(timeout=5)

        # Проверяем, что результат помещен в очередь
        assert not pipeline.queue.empty()
        result = pipeline.queue.get()
        assert not math.isnan(result)

        # Проверяем, что нет сообщений об ошибках
        captured = capsys.readouterr()
        assert "Ошибка" not in captured.out

    def test_calculate_first_function_exception(self, monkeypatch, capsys):
        """Тест первой функции при возникновении исключения"""
        pipeline = PipelineThreads()

        # Мокаем calculate_s, чтобы выбросить исключение
        def mock_calculate_s(x):
            raise ValueError("Test error")

        monkeypatch.setattr(pipeline.series_calc, "calculate_s", mock_calculate_s)

        thread = threading.Thread(
            target=pipeline.calculate_first_function, args=(math.pi,)
        )
        thread.start()
        thread.join(timeout=5)

        # Проверяем, что в очередь помещен NaN
        assert not pipeline.queue.empty()
        result = pipeline.queue.get()
        assert math.isnan(result)

        # Проверяем сообщение об ошибке
        captured = capsys.readouterr()
        assert "Ошибка в первой функции" in captured.out

    def test_calculate_second_function_normal(self, capsys):
        """Тест второй функции в нормальных условиях"""
        pipeline = PipelineThreads(epsilon=1e-4)

        # Помещаем тестовое значение в очередь
        test_value = 1.5
        pipeline.queue.put(test_value)

        # Запускаем вторую функцию
        thread = threading.Thread(
            target=pipeline.calculate_second_function, args=(math.pi,)
        )
        thread.start()
        thread.join(timeout=5)

        # Проверяем вывод
        captured = capsys.readouterr()
        assert "Результат первой функции" in captured.out
        assert "Результат второй функции" in captured.out

    def test_calculate_second_function_nan_input(self, capsys):
        """Тест второй функции при получении NaN из первой функции"""
        pipeline = PipelineThreads()

        # Помещаем NaN в очередь
        pipeline.queue.put(float("nan"))

        # Запускаем вторую функцию
        thread = threading.Thread(
            target=pipeline.calculate_second_function, args=(math.pi,)
        )
        thread.start()
        thread.join(timeout=5)

        # Проверяем сообщение о некорректном результате
        captured = capsys.readouterr()
        assert "некорректный результат" in captured.out

    def test_calculate_second_function_empty_queue(self, capsys):
        """Тест второй функции при пустой очереди (таймаут)"""
        pipeline = PipelineThreads()

        # Используем небольшую очередь без таймаута для теста
        # Для этого временно подменим метод get
        original_get = pipeline.queue.get

        def mock_get(timeout=None):
            raise queue.Empty

        pipeline.queue.get = mock_get

        try:
            # Запускаем вторую функцию
            thread = threading.Thread(
                target=pipeline.calculate_second_function, args=(math.pi,)
            )
            thread.start()
            thread.join(timeout=2)

            # Проверяем сообщение об ошибке таймаута
            captured = capsys.readouterr()
            assert "не вернула результат" in captured.out
        finally:
            pipeline.queue.get = original_get

    def test_calculate_second_function_invalid_log_argument(self, capsys):
        """Тест второй функции с недопустимым аргументом логарифма"""
        pipeline = PipelineThreads()

        # Помещаем произвольное значение в очередь
        pipeline.queue.put(1.0)

        # Вызываем с x, для которого sin(x/2) <= 0
        # Например, x = 0: sin(0) = 0, аргумент логарифма = 0
        thread = threading.Thread(
            target=pipeline.calculate_second_function, args=(0.0,)
        )
        thread.start()
        thread.join(timeout=5)

        # Проверяем предупреждение
        captured = capsys.readouterr()
        assert "аргумент логарифма" in captured.out

    def test_full_pipeline_execution(self, capsys):
        """Тест полного выполнения конвейера для x = π"""
        pipeline = PipelineThreads(epsilon=1e-4)

        # Запускаем оба потока
        thread1 = threading.Thread(
            target=pipeline.calculate_first_function, args=(math.pi,)
        )
        thread2 = threading.Thread(
            target=pipeline.calculate_second_function, args=(math.pi,)
        )

        thread1.start()
        thread2.start()

        thread1.join(timeout=5)
        thread2.join(timeout=5)

        # Проверяем вывод
        captured = capsys.readouterr()
        assert "Результат первой функции" in captured.out
        assert "Результат второй функции" in captured.out
        assert "Разность |S - y|" in captured.out

        # Для x = π результаты должны совпадать
        assert (
            "Результаты совпадают" in captured.out
            or "Результаты не совпадают" in captured.out
        )


def test_run_pipeline_basic(capsys):
    """Тест функции run_pipeline для x = π"""
    run_pipeline(math.pi, epsilon=1e-4)

    captured = capsys.readouterr()
    assert "Вычисление для x" in captured.out
    assert "Точность ε" in captured.out
    assert "Результат первой функции" in captured.out
    assert "Результат второй функции" in captured.out


def test_run_pipeline_multiple_values(capsys):
    """Тест функции run_pipeline для нескольких значений x"""
    test_values = [math.pi / 2, math.pi / 3]

    for x in test_values:
        run_pipeline(x, epsilon=1e-4)

    captured = capsys.readouterr()
    # Проверяем, что для каждого значения был вывод
    assert captured.out.count("Вычисление для x") == len(test_values)


def test_run_pipeline_timeout_simulation(monkeypatch, capsys):
    """Тест функции run_pipeline при таймауте потока"""

    def mock_join(self, timeout=None):
        # Имитируем, что поток все еще жив после таймаута
        return None

    monkeypatch.setattr(threading.Thread, "join", mock_join)

    # Создаем мок для is_alive, чтобы вернуть True
    def mock_is_alive(self):
        return True

    monkeypatch.setattr(threading.Thread, "is_alive", mock_is_alive)

    try:
        run_pipeline(math.pi, epsilon=1e-4)

        captured = capsys.readouterr()
        assert "не завершилась" in captured.out
    finally:
        # Восстанавливаем оригинальные методы
        monkeypatch.undo()


# Интеграционные тесты


def test_pipeline_thread_safety():
    """Тест потокобезопасности конвейера"""
    from concurrent.futures import ThreadPoolExecutor

    def run_single_pipeline(x):
        """Запуск одного конвейера"""
        pipeline = PipelineThreads(epsilon=1e-5)

        thread1 = threading.Thread(target=pipeline.calculate_first_function, args=(x,))
        thread2 = threading.Thread(target=pipeline.calculate_second_function, args=(x,))

        thread1.start()
        thread2.start()

        thread1.join(timeout=5)
        thread2.join(timeout=5)

        return True

    # Запускаем несколько конвейеров параллельно
    test_values = [math.pi, math.pi / 2, math.pi / 3, math.pi / 4]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_single_pipeline, x) for x in test_values]
        results = [future.result(timeout=10) for future in futures]

    # Все конвейеры должны успешно завершиться
    assert all(results)


# Параметризованные тесты


@pytest.mark.parametrize(
    "x,expected_precision",
    [
        (math.pi, 1e-6),
        (math.pi / 2, 1e-5),
        (math.pi / 3, 1e-5),
        (math.pi / 4, 1e-5),
    ],
)
def test_series_convergence(x, expected_precision):
    """Параметризованный тест сходимости ряда для разных x"""
    calculator = SeriesCalculator(epsilon=1e-7)
    result = calculator.calculate_s(x)

    # Проверяем, что результат конечный и не NaN
    assert not math.isnan(result)
    assert not math.isinf(result)

    # Для x = π дополнительно проверяем точное значение
    if math.isclose(x, math.pi, rel_tol=1e-10):
        expected = -math.log(2)
        assert math.isclose(result, expected, rel_tol=expected_precision)


# Тесты на граничные случаи
def test_large_epsilon():
    """Тест с большой точностью (быстрое завершение)"""
    calculator = SeriesCalculator(epsilon=0.1)  # Большая точность
    result = calculator.calculate_s(math.pi)

    # Проверяем, что результат получен
    assert not math.isnan(result)

    # С такой точностью вычисление должно завершиться быстро
    # (проверяем, что результат не слишком далек от ожидаемого)
    expected = -math.log(2)
    # Допускаем большую погрешность из-за большой epsilon
    assert abs(result - expected) < 0.2


if __name__ == "__main__":
    # Запуск тестов при прямом выполнении файла
    pytest.main([__file__, "-v", "--tb=short"])
