import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

#На граници нашей сетки устанавливаем нули, чтобы жифотное не вышло за них
def transition_matrix(grid_size):
    transition_matrix = np.random.rand(grid_size, grid_size, 4)  # Вверх, вниз, влево, вправо

    # Добавление дополнительных вероятностей нуля
    num_obstacles = int((grid_size * grid_size) * 0.2)  # 20% узлов с нулевой вероятностью для одного направления
    for _ in range(num_obstacles):
        i = np.random.randint(1, grid_size-1)
        j = np.random.randint(1, grid_size-1)
        direction = np.random.choice(4)
        transition_matrix[i, j, direction] = 0

    for i in range(grid_size):
        for j in range(grid_size):
            if i == 0:  # Верхняя граница
                transition_matrix[i, j, 0] = 0
            if i == grid_size - 1:  # Нижняя граница
                transition_matrix[i, j, 1] = 0
            if j == 0:  # Левая граница
                transition_matrix[i, j, 2] = 0
            if j == grid_size - 1:  # Правая граница
                transition_matrix[i, j, 3] = 0
            normalization_factor = np.sum(transition_matrix[i, j])
            if normalization_factor > 0:
                transition_matrix[i, j] /= normalization_factor

    return transition_matrix


def simulate_random_walk(grid_size, sensor_position, transition_matrix, num_simulations=1):
    steps_list = []

    for _ in range(num_simulations):
        animal_position = (np.random.randint(grid_size), np.random.randint(grid_size))
        while animal_position == sensor_position:
            animal_position = (np.random.randint(grid_size), np.random.randint(grid_size))

        steps = 0
        path = [animal_position]

        while animal_position != sensor_position:
            i, j = animal_position
            move = np.random.choice(4, p=transition_matrix[i, j])
            if move == 0:  # Вверх
                i -= 1
            elif move == 1:  # Вниз
                i += 1
            elif move == 2:  # Влево
                j -= 1
            elif move == 3:  # Вправо
                j += 1

            animal_position = (i, j)
            path.append(animal_position)
            steps += 1

        steps_list.append((steps, path))

    return steps_list


def plot_paths(grid_size, paths):
    G = nx.grid_2d_graph(grid_size, grid_size)
    pos = {(x, y): (y, -x) for x, y in G.nodes()}  # Позиционирование на плоскости
    nx.draw(G, pos, node_color='lightgray', with_labels=False, node_size=300)

    # Цвета для разных путей
    colors = plt.cm.jet(np.linspace(0, 1, len(paths)))

    for idx, path in enumerate(paths):
        path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color=[colors[idx]], width=2, alpha=0.5)

        # Отметим начальную и конечную точки каждого пути
        nx.draw_networkx_nodes(G, pos, nodelist=[path[0]], node_color='green', node_size=500, label='Start' if idx == 0 else "")
        nx.draw_networkx_nodes(G, pos, nodelist=[path[-1]], node_color='red', node_size=500, label='Sensor' if idx == 0 else "")

    plt.title("Визуализация пути")
    plt.show()

def plot_histogram(steps_list):
    steps = [s[0] for s in steps_list]
    plt.hist(steps, bins=20, alpha=0.75)
    plt.xlabel('Количество шагов')
    plt.ylabel('Частота')
    plt.title('Гистограмма о количестве шагод для достижения цели')
    plt.show()

# Параметры
grid_size = int(input("Введите размер карты: "))
sensor_position = (np.random.randint(grid_size), np.random.randint(grid_size))
num_simulations = int(input("Введите количество животных: "))

# Генерация матрицы переходов
transition_matrix = transition_matrix(grid_size)

# Симуляция блужданий
steps_list = simulate_random_walk(grid_size, sensor_position, transition_matrix, num_simulations)

plot_histogram(steps_list)

# Визуализация пути симуляций
paths = list(map(lambda x: x[1], steps_list))
plot_paths(grid_size, paths)