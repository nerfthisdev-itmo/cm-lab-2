import math
import numpy as np
import matplotlib.pyplot as plt

# Определение нелинейных уравнений
equations = [
    {
        'id': 1,
        'name': '-2.4x³ + 1.27x² + 8.36x + 2.31',
        'f': lambda x: -2.4 * x**3 + 1.27 * x**2 + 8.36 * x + 2.31,
        'df': lambda x: -7.2 * x**2 + 2.54 * x + 8.36,
        'd2f': lambda x: -14.4 * x + 2.54,
    },
    {
        'id': 2,
        'name': '5.74x³ - 2.95x² - 10.28x - 3.23',
        'f': lambda x: 5.74 * x**3 - 2.95 * x**2 - 10.28 * x - 3.23,
        'df': lambda x: 17.22 * x**2 - 5.9 * x - 10.28,
        'd2f': lambda x: 34.44 * x - 5.9,
    },
    {
        'id': 3,
        'name': 'x³ + 2.64x² - 5.41x - 11.76',
        'f': lambda x: x**3 + 2.64 * x**2 - 5.41 * x - 11.76,
        'df': lambda x: 3 * x**2 + 5.28 * x - 5.41,
        'd2f': lambda x: 6 * x + 5.28,
    },
    {
        'id': 4,
        'name': 'sin(x) - e^(-x)',
        'f': lambda x: math.sin(x) - math.exp(-x),
        'df': lambda x: math.cos(x) + math.exp(-x),
        'd2f': lambda x: -math.sin(x) - math.exp(-x),
    },
    {
        'id': 5,
        'name': '2.74 * x^3 - 1.93 * x^2 - 15.28 * x - 3.72',
        'f': lambda x: 2.74 * x ** 3 - 1.93 * x ** 2 - 15.28 * x - 3.72,
        'df': lambda x:  8.22 * x**2 - 3.86 * x - 15.28,
        'd2f': lambda x:   16.44 * x - 3.86,
    },

]
 

# Определение систем уравнений
systems = [
    {
        'id': 1,
        'name': 'sin(x+1) - y = 1.2; 2x + cos(y) = 2',
        'F': lambda x, y: np.array([
            math.sin(x + 1) - y - 1.2,
            2 * x + math.cos(y) - 2
        ]),
        'J': lambda x, y: np.array([
            [math.cos(x + 1), -1],
            [2, -math.sin(y)]
        ])
    },
    {
        'id': 2,
        'name': 'sin(x) + 2y = 2; x + cos(y-1) = 0.7',
        'F': lambda x, y: np.array([
            math.sin(x) + 2 * y - 2,
            x + math.cos(y - 1) - 0.7
        ]),
        'J': lambda x, y: np.array([
            [math.cos(x), 2],
            [1, -math.sin(y - 1)]
        ])
    },
    {
        'id': 3,
        'name': 'sin(x+y) = 1.5x -0.1; x² + 2y² =1',
        'F': lambda x, y: np.array([
            math.sin(x + y) - 1.5 * x + 0.1,
            x**2 + 2 * y**2 - 1
        ]),
        'J': lambda x, y: np.array([
            [math.cos(x + y) - 1.5, math.cos(x + y)],
            [2 * x, 4 * y]
        ])
    }
]


def find_tau(df, a, b):
    n = 1000
    xs = [a + (b - a) * i / n for i in range(n + 1)]
    dfs = [abs(df(x)) for x in xs]
    m = min(dfs)
    M = max(dfs)
    if m == 0:
        raise ValueError("На интервале производная равна 0 — метод не сойдётся")
    return 2 / (m + M)



def chord_method(f, a, b, eps, max_iter=1000):
    if f(a) * f(b) >= 0:
        return None, 0, "Интервал не содержит корня или содержит несколько корней"
    iter_count = 0
    x_prev = a
    for _ in range(max_iter):
        x_new = x_prev - f(x_prev) * (b - x_prev) / (f(b) - f(x_prev))
        if abs(x_new - x_prev) < eps:
            return x_new, iter_count + 1, None
        iter_count += 1
        x_prev = x_new
    return x_prev, iter_count, "Достигнуто максимальное количество итераций"

def newton_method(f, df, x0, eps, max_iter=1000):
    iter_count = 0
    x_prev = x0
    for _ in range(max_iter):
        fx = f(x_prev)
        if abs(fx) < eps:
            return x_prev, iter_count, None
        dfx = df(x_prev)
        if dfx == 0:
            return None, iter_count, "Производная равна нулю"
        x_new = x_prev - fx / dfx
        if abs(x_new - x_prev) < eps:
            return x_new, iter_count + 1, None
        iter_count += 1
        x_prev = x_new
    return x_prev, iter_count, "Достигнуто максимальное количество итераций"

def simple_iteration_method(phi, dphi, a, b, eps, max_iter=1000):
    # Проверка условия сходимости |phi'(x)| < 1
    x_samples = [a + (b - a) * i / 1000 for i in range(1001)]
    if max(abs(dphi(x)) for x in x_samples) >= 1:
        return None, 0, "Условие сходимости не выполнено"
    
    # Выбор начального приближения
    x_prev = (a + b) / 2  # Можно оптимизировать выбор
    iter_count = 0
    for _ in range(max_iter):
        x_new = phi(x_prev)
        if abs(x_new - x_prev) < eps:
            return x_new, iter_count + 1, None
        iter_count += 1
        x_prev = x_new
    return x_prev, iter_count, "Достигнуто максимальное количество итераций"

def newton_system(F, J, x0, y0, eps, max_iter=100):
    x_prev = np.array([x0, y0], dtype=float)
    errors = []
    iter_count = 0
    for _ in range(max_iter):
        Fx = F(x_prev[0], x_prev[1])
        if np.linalg.norm(Fx) < eps:
            break
        Jx = J(x_prev[0], x_prev[1])
        try:
            delta = np.linalg.solve(Jx, -Fx)
        except np.linalg.LinAlgError:
            return x_prev, iter_count, errors, "Матрица Якоби вырождена"
        x_new = x_prev + delta
        errors.append(np.linalg.norm(x_new - x_prev))
        if errors[-1] < eps:
            break
        iter_count += 1
        x_prev = x_new
    else:
        return x_prev, iter_count, errors, "Достигнуто максимальное количество итераций"
    return x_prev, iter_count + 1, errors, None

def plot_equation(f, a, b, root=None):
    x = np.linspace(a, b, 400)
    y = [f(xi) for xi in x]
    plt.plot(x, y, label='f(x)')
    plt.axhline(0, color='black', linewidth=0.5)
    if root is not None:
        plt.scatter(root, f(root), color='red', label='Корень')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid()
    plt.show()

def plot_system(sys, solution=None):
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(x, y)
    F1 = np.vectorize(lambda x, y: sys['F'](x, y)[0])(X, Y)
    F2 = np.vectorize(lambda x, y: sys['F'](x, y)[1])(X, Y)
    plt.contour(X, Y, F1, levels=[0], colors='r')
    plt.contour(X, Y, F2, levels=[0], colors='b')
    if solution is not None:
        plt.scatter(solution[0], solution[1], color='green', label='Решение')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()

def main():
    print("Выберите задачу:")
    print("1. Решить нелинейное уравнение")
    print("2. Решить систему уравнений")
    choice = input("Ваш выбор: ")

    if choice == '1':
        print("\nВыберите уравнение:")
        for eq in equations:
            print(f"{eq['id']}. {eq['name']}")
        eq_id = int(input("Номер уравнения: ")) - 1
        eq = equations[eq_id]

        print("\nВыберите метод:")
        print("1. Метод хорд")
        print("2. Метод Ньютона")
        print("3. Метод простой итерации")
        method = input("Номер метода: ")

        a = float(input("Введите a: "))
        b = float(input("Введите b: "))
        eps = float(input("Введите точность epsilon: "))

        if method == '1':
            root, iters, error = chord_method(eq['f'], a, b, eps)
        elif method == '2':
            x0 = a if abs(eq['f'](a)) < abs(eq['f'](b)) else b
            root, iters, error = newton_method(eq['f'], eq['df'], x0, eps)
        elif method == '3':
            # Преобразование к виду x = phi(x)
            tau = find_tau(eq['df'], a, b)
            phi = lambda x: x - tau * eq['f'](x)
            dphi = lambda x: 1 - tau * eq['df'](x)
            root, iters, error = simple_iteration_method(phi, dphi, a, b, eps)
        else:
            print("Неверный метод")
            return

        if error:
            print(f"Ошибка: {error}")
        else:
            print(f"\nКорень: {root:.5f}")
            print(f"Значение функции: {eq['f'](root):.5f}")
            print(f"Итераций: {iters}")
            plot_equation(eq['f'], a, b, root)

    elif choice == '2':
        print("\nВыберите систему:")
        for sys in systems:
            print(f"{sys['id']}. {sys['name']}")
        sys_id = int(input("Номер системы: ")) - 1
        sys = systems[sys_id]

        x0 = float(input("Введите x0: "))
        y0 = float(input("Введите y0: "))
        eps = float(input("Введите точность epsilon: "))

        sol, iters, errors, error = newton_system(sys['F'], sys['J'], x0, y0, eps)
        if error:
            print(f"Ошибка: {error}")
        else:
            print(f"\nРешение: x = {sol[0]:.5f}, y = {sol[1]:.5f}")
            print(f"Итераций: {iters}")
            print("Вектор погрешностей (норма разности):")
            for i, err in enumerate(errors):
                print(f"Итерация {i+1}: {err:.5f}")
            plot_system(sys, sol)

    else:
        print("Неверный выбор")

if __name__ == "__main__":
    main()
