import math
from task_package.zad import run_pipeline

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
