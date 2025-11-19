import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches
from matplotlib.axes import Axes

# Глобальные настройки стиля
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'


# Задача 1: Оптимизация производства электроники

def solve_task1():
    """Решение задачи оптимизации производства"""

    print("\n" + "=" * 80)
    print("Задача 1: Оптимизация производства электроники")
    print("=" * 80)

    # Исходные данные
    print("\nИсходные данные:")
    print("  Смартфоны:  прибыль 8000 руб, требует 2ч CPU, 4ГБ RAM, 1 аккумулятор")
    print("  Планшеты:   прибыль 12000 руб, требует 3ч CPU, 6ГБ RAM, 2 аккумулятора")
    print("  Ресурсы:    240ч CPU, 480ГБ RAM, 150 аккумуляторов")

    print("\nМатематическая модель:")
    print("  Переменные: x1 - смартфоны, x2 - планшеты")
    print("  Целевая функция: P = 8000*x1 + 12000*x2 -> max")
    print("  Ограничения:")
    print("    2*x1 + 3*x2 <= 240  (процессорное время)")
    print("    x1 + 2*x2 <= 150    (аккумуляторы)")
    print("    x1, x2 >= 0")

    # Целевая функция (для максимизации меняем знак)
    c = [-8000, -12000]

    # Ограничения A_ub @ x <= b_ub
    # Примечание: ограничение по памяти (4x1 + 6x2 <= 480) избыточное,
    # так как эквивалентно ограничению по CPU после деления на 2
    A_ub = [
        [2, 3],  # Процессорное время
        [1, 2]  # Аккумуляторы
    ]
    b_ub = [240, 150]

    # Границы переменных
    bounds = [(0, None), (0, None)]

    # Решение
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if result.success:
        x1, x2 = result.x
        profit = -result.fun

        print(f"\nРешение найдено:")
        print(f"  x1 (смартфоны)  = {x1:.2f} шт")
        print(f"  x2 (планшеты)   = {x2:.2f} шт")
        print(f"  Максимальная прибыль: {profit:,.2f} руб")

        # Проверка использования ресурсов
        cpu_used = 2 * x1 + 3 * x2
        battery_used = x1 + 2 * x2

        print("\nИспользование ресурсов:")
        print(f"  Процессорное время: {cpu_used:.2f} / 240 ч  ", end="")
        print("(полностью)" if abs(cpu_used - 240) < 1e-6 else f"(остаток {240 - cpu_used:.2f})")
        print(f"  Аккумуляторы:       {battery_used:.2f} / 150 шт ", end="")
        print("(полностью)" if abs(battery_used - 150) < 1e-6 else f"(остаток {150 - battery_used:.2f})")

        print("\nАнализ:")
        print(f"  Производить {x1:.0f} смартфонов и {x2:.0f} планшетов")
        print(f"  Дефицитные ресурсы: ", end="")
        deficient = []
        if abs(cpu_used - 240) < 1e-6:
            deficient.append("процессорное время")
        if abs(battery_used - 150) < 1e-6:
            deficient.append("аккумуляторы")
        print(", ".join(deficient) if deficient else "нет")

        # Визуализация
        visualize_task1(result.x)
        return result
    else:
        print(f"Ошибка: {result.message}")
        return None


def visualize_task1(optimal_point):

    """Построение графика для задачи 1"""

    fig, ax = plt.subplots(figsize=(10, 8))

    x1 = np.linspace(0, 160, 400)

    # Линии ограничений
    x2_cpu = np.maximum((240 - 2 * x1) / 3, 0)
    x2_battery = np.maximum((150 - x1) / 2, 0)

    # Построение прямых ограничений
    ax.plot(x1, x2_cpu, color='#1F77B4', linestyle='-', linewidth=2.5, label='Процессорное время: 2x1 + 3x2 <= 240')
    ax.plot(x1, x2_battery, color='#7F7F7F', linestyle='-', linewidth=2.5, label='Аккумуляторы: x1 + 2x2 <= 150')

    # Вершины многогранника допустимой области
    vertices = [[0, 0], [0, 75], [30, 60], [120, 0]]

    # Закрашивание допустимой области
    polygon = Polygon(vertices, alpha=0.35, facecolor='#AEC7E8', edgecolor='#1F77B4', linewidth=1.0,
                      label='Допустимая область')
    ax.add_patch(polygon)

    # Расчет максимальной прибыли (целевая функция) для оптимальной точки
    profit = 8000 * optimal_point[0] + 12000 * optimal_point[1]

    # Линии уровня целевой функции (Изопрофиты)
    for p_level in np.linspace(profit * 0.2, profit, 5):
        x2_profit = (p_level - 8000 * x1) / 12000
        ax.plot(x1, x2_profit, color='#999999', linestyle='--', alpha=0.5, linewidth=1)

    # Оптимальная точка
    ax.plot(optimal_point[0], optimal_point[1], 'o', color='red', markersize=12,
            label=f'Оптимум ({optimal_point[0]:.0f}, {optimal_point[1]:.0f})', zorder=5)

    ax.annotate(f'Максимальная прибыль:\n{profit:,.0f} руб',
                xy=(optimal_point[0], optimal_point[1]),
                xytext=(optimal_point[0] + 5, optimal_point[1] + 18),
                color='red', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='square,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))

    # Установка меток и заголовка
    ax.set_xlabel('x₁ (Смартфоны, шт)', fontsize=14, fontweight='bold')
    ax.set_ylabel('x₂ (Планшеты, шт)', fontsize=14, fontweight='bold')
    ax.set_title('Задача 1: Геометрическая визуализация (Оптимум: x₁=30, x₂=60)',
                 fontsize=16, fontweight='bold', pad=20)

    # Настройка легенды, сетки и ограничений осей
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.5, linestyle=':')
    ax.set_xlim(-5, 130)
    ax.set_ylim(-5, 90)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig('task1.png', dpi=300, bbox_inches='tight')
    plt.show()



# Задача 2: Транспортная задача

def solve_task2():
    """Решение транспортной задачи"""

    print("\n" + "=" * 80)
    print("Задача 2: Оптимизация снабжения военных баз")
    print("=" * 80)

    print("\nИсходные данные:")
    print("\nСклады:")
    print("  Склад 1: 150 тонн")
    print("  Склад 2: 250 тонн")
    print("  Всего:   400 тонн")

    print("\nБазы:")
    print("  База Альфа: 120 тонн")
    print("  База Бета:  180 тонн")
    print("  База Гамма: 100 тонн")
    print("  Всего:      400 тонн")

    print("\nСтоимость перевозки (усл.ед/тонна):")
    print("              Альфа  Бета  Гамма")
    print("  Склад 1:      8     6     10")
    print("  Склад 2:      9     7      5")

    # Проверка баланса
    supply = 150 + 250
    demand = 120 + 180 + 100
    print(f"\nБаланс: {supply} = {demand} -> задача сбалансирована")

    print("\nМатематическая модель:")
    print("  Переменные: x_ij - тонн со склада i на базу j")
    print("  Целевая функция: Z = 8*x11 + 6*x12 + 10*x13 + 9*x21 + 7*x22 + 5*x23 -> min")
    print("  Ограничения по складам:")
    print("    x11 + x12 + x13 = 150")
    print("    x21 + x22 + x23 = 250")
    print("  Ограничения по базам:")
    print("    x11 + x21 = 120")
    print("    x12 + x22 = 180")
    print("    x13 + x23 = 100")

    # Целевая функция
    c = [8, 6, 10, 9, 7, 5]

    # Ограничения-равенства A_eq @ x = b_eq
    A_eq = [
        [1, 1, 1, 0, 0, 0],  # Склад 1
        [0, 0, 0, 1, 1, 1],  # Склад 2
        [1, 0, 0, 1, 0, 0],  # База Альфа
        [0, 1, 0, 0, 1, 0],  # База Бета
        [0, 0, 1, 0, 0, 1]  # База Гамма
    ]
    b_eq = [150, 250, 120, 180, 100]

    # Границы переменных
    bounds = [(0, None)] * 6

    # Решение
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if result.success:
        x11, x12, x13, x21, x22, x23 = result.x

        print(f"\nРешение найдено:")
        print("\nОптимальный план перевозок:")
        print("  Со Склада 1:")
        print(f"    -> Альфа: {x11:6.2f} т  (стоимость: {8 * x11:8.2f})")
        print(f"    -> Бета:  {x12:6.2f} т  (стоимость: {6 * x12:8.2f})")
        print(f"    -> Гамма: {x13:6.2f} т  (стоимость: {10 * x13:8.2f})")

        print("  Со Склада 2:")
        print(f"    -> Альфа: {x21:6.2f} т  (стоимость: {9 * x21:8.2f})")
        print(f"    -> Бета:  {x22:6.2f} т  (стоимость: {7 * x22:8.2f})")
        print(f"    -> Гамма: {x23:6.2f} т  (стоимость: {5 * x23:8.2f})")

        print(f"\n  Минимальная стоимость: {result.fun:.2f} усл.ед")

        # Проверка
        print("\nПроверка ограничений:")
        print(f"  Склад 1: {x11 + x12 + x13:6.2f} = 150")
        print(f"  Склад 2: {x21 + x22 + x23:6.2f} = 250")
        print(f"  Альфа:   {x11 + x21:6.2f} = 120")
        print(f"  Бета:    {x12 + x22:6.2f} = 180")
        print(f"  Гамма:   {x13 + x23:6.2f} = 100")

        # Анализ маршрутов
        routes = [
            ("Склад 1 -> Альфа", x11, 8),
            ("Склад 1 -> Бета", x12, 6),
            ("Склад 1 -> Гамма", x13, 10),
            ("Склад 2 -> Альфа", x21, 9),
            ("Склад 2 -> Бета", x22, 7),
            ("Склад 2 -> Гамма", x23, 5)
        ]

        print("\nИспользуемые маршруты:")
        for name, qty, cost in routes:
            if qty > 1e-6:
                print(f"  {name:20s}: {qty:6.2f} т x {cost:2d} = {qty * cost:8.2f} усл.ед")

        print("\nНеиспользуемые маршруты:")
        for name, qty, cost in routes:
            if qty <= 1e-6:
                print(f"  {name:20s}: {qty:6.2f} т (стоимость {cost} - невыгодно)")

        # Визуализация
        visualize_task2(result.x, result.fun)
        return result
    else:
        print(f"Ошибка: {result.message}")
        return None


def visualize_task2(solution, total_cost):
    """Построение чистой сетевой диаграммы для задачи 2"""

    fig, ax = plt.subplots(figsize=(14, 10))

    # Координаты узлов
    warehouses = {'Склад 1': (2, 8), 'Склад 2': (2, 3)}
    bases = {'Альфа': (12, 10), 'Бета': (12, 5.5), 'Гамма': (12, 1)}

    def draw_node(ax: Axes, x, y, width, height, label, info, color):
        box = FancyBboxPatch(
            (x - width / 2, y - height / 2), width, height,
            boxstyle="round,pad=0.15",
            edgecolor='#333333', facecolor=color, linewidth=1.5
        )
        ax.add_patch(box)
        ax.text(x, y + 0.35, label, ha='center', va='center',
                fontsize=14, fontweight='bold', color='#333333')
        ax.text(x, y - 0.35, info, ha='center', va='center',
                fontsize=12, style='italic', color='#333333')

    # Рисуем узлы
    node_color = '#E0F2F7'
    supplies = {'Склад 1': 150, 'Склад 2': 250}
    for name, (x, y) in warehouses.items():
        draw_node(ax, x, y, 2.8, 1.3, name, f'Запас: {supplies[name]} т', node_color)

    demands = {'Альфа': 120, 'Бета': 180, 'Гамма': 100}
    for name, (x, y) in bases.items():
        draw_node(ax, x, y, 2.8, 1.3, f'База {name}', f'Нужно: {demands[name]} т', node_color)

    # Потоки
    x11, x12, x13, x21, x22, x23 = solution
    flows = [
        ('Склад 1', 'Альфа', x11, 8),
        ('Склад 1', 'Бета', x12, 6),
        ('Склад 1', 'Гамма', x13, 10),
        ('Склад 2', 'Альфа', x21, 9),
        ('Склад 2', 'Бета', x22, 7),
        ('Склад 2', 'Гамма', x23, 5)
    ]

    # Рисуем потоки
    for warehouse, base, quantity, cost in flows:
        if quantity > 1e-6:
            x_start, y_start = warehouses[warehouse]
            x_end, y_end = bases[base]

            # Увеличенный offset для предотвращения наложения
            offset = 0.30 if '2' in warehouse else -0.30
            width = max(2.0, quantity / 25)

            # Цветовая схема
            color = '#007ACC' if cost <= 7 else '#E4572E'

            arrow = FancyArrowPatch(
                (x_start + 1.4, y_start + offset),
                (x_end - 1.4, y_end + offset),
                arrowstyle='->,head_width=0.6,head_length=0.6',
                linewidth=width,
                color=color,
                alpha=0.9,
                zorder=1
            )
            ax.add_patch(arrow)

            mid_x = (x_start + x_end) / 2
            mid_y = (y_start + y_end) / 2 + offset

            # Коррекция положения текстов потоков
            if warehouse == 'Склад 1' and base == 'Бета':
                mid_x += 0.8
                mid_y -= 0.1

            # 2. Склад 2 -> Альфа (Если бы был ненулевой поток)
            elif warehouse == 'Склад 2' and base == 'Альфа':
                mid_y += 0.5
                mid_x += 0.5

            label = f'{quantity:.0f} т\n({cost}x{quantity:.0f}={cost * quantity:.0f})'
            ax.text(mid_x, mid_y, label,
                    ha='center', va='center',
                    fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.4',
                              facecolor='white', edgecolor='#333333', linewidth=1.0))

    # Результат
    ax.text(7, 10.8, f'Минимальная стоимость: {total_cost:.0f} усл.ед',
            ha='center',
            fontsize=10,
            fontweight='bold',
            color='#333333',
            bbox=dict(
                boxstyle='square,pad=0.3',
                facecolor='none',
                edgecolor='none',
                linewidth=0
            )
            )

    legend_elements = [
        mpatches.Patch(facecolor='#E0F2F7', edgecolor='#333333', label='Склады / Базы'),
        mpatches.Patch(facecolor='#007ACC', alpha=0.9, label='Выгодный маршрут (≤7 усл. ед.)'),
        mpatches.Patch(facecolor='#E4572E', alpha=0.9, label='Дорогой маршрут (≥8 усл. ед.)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9, edgecolor='#333333')

    ax.set_xlim(-1, 15)
    ax.set_ylim(-0.5, 13)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Задача 2: Сетевая диаграмма снабжения (Транспортная задача)',
                 fontsize=17, fontweight='bold', pad=20, color='#333333')

    plt.tight_layout()
    plt.savefig('task2.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


# Главная программа

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Лабораторная работа: Линейное программирование")
    print("=" * 80)

    # Решение задачи 1
    result1 = solve_task1()

    print("\n" + "-" * 80 + "\n")

    # Решение задачи 2
    result2 = solve_task2()

    print("\n" + "=" * 80)
    print("Все задачи решены")
    print("=" * 80)