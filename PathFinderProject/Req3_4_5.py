import numpy as np
import matplotlib.pyplot as plt
import random
from heapq import heappop, heappush

# 定义全局变量
california_bounds = [32.5, 42, -124.5, -114]
num_sub_areas_x = 30
num_sub_areas_y = 60
r = np.linspace(0, 35, 100)
radius_index = 50


def load_data(data):
    data = np.load(f'{data}.npy')
    return data


def ripley_k(data, r):
    n = len(data)
    k = np.zeros(len(r))
    for i in range(n):
        for j, radius in enumerate(r):
            in_circle = np.linalg.norm(data[i] - data, axis=1) <= radius
            k[j] += np.sum(in_circle)
    return k / n


def calculate_sub_area_k(data, lat_min, lat_max, lon_min, lon_max, r):
    sub_area_data = data[(data[:, 0] >= lat_min) & (data[:, 0] <= lat_max) &
                         (data[:, 1] >= lon_min) & (data[:, 1] <= lon_max)]
    if len(sub_area_data) == 0:
        return np.zeros(len(r))
    return ripley_k(sub_area_data, r)


def generate_k_matrix(data_combined):
    lat_step = (california_bounds[1] - california_bounds[0]) / num_sub_areas_y
    lon_step = (california_bounds[3] - california_bounds[2]) / num_sub_areas_x
    k_matrix = np.zeros((num_sub_areas_y, num_sub_areas_x, len(r)))
    for i in range(num_sub_areas_y):
        for j in range(num_sub_areas_x):
            lat_min = california_bounds[0] + i * lat_step
            lat_max = lat_min + lat_step
            lon_min = california_bounds[2] + j * lon_step
            lon_max = lon_min + lon_step

            k_matrix[i, j, :] = calculate_sub_area_k(data_combined, lat_min, lat_max, lon_min, lon_max, r)

            # If K value is 0,make it 1 or zero, otherwise , it will never find a right path.
            k_matrix[i, j, k_matrix[i, j, :] == 0] = 0.0  #

    return k_matrix


def get_sub_area_index(lat, lon, bounds, lat_step, lon_step):
    lat_min, lat_max, lon_min, lon_max = bounds
    x_idx = int((lon - lon_min) / lon_step)
    y_idx = int((lat - lat_min) / lat_step)
    return y_idx, x_idx


def generate_graph():
    graph = {}
    for i in range(num_sub_areas_y):
        for j in range(num_sub_areas_x):
            neighbors = []
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < num_sub_areas_y and 0 <= nj < num_sub_areas_x:
                    neighbors.append((ni, nj))
            graph[(i, j)] = neighbors
    return graph


def dynamic_threshold_a_star(graph, start, goals, k_matrix, radius_index):
    initial_threshold = 0.0  # 设置初始阈值为零
    threshold = initial_threshold
    step = 0.01  # 细微调整步长

    while True:
        frontier = []
        heappush(frontier, (0, start))
        came_from = {}
        cost_so_far = {}
        came_from[start] = None
        cost_so_far[start] = 0

        while frontier:
            _, current = heappop(frontier)

            if current in goals:
                return came_from, cost_so_far, threshold  # 找到路径时立即返回

            for next in graph[current]:
                new_cost = cost_so_far[current] + 1
                # 确保 K 值为零的区域也可以被搜索到
                if (next not in cost_so_far or new_cost < cost_so_far[next]) and k_matrix[
                    next[0], next[1], radius_index] >= threshold:
                    cost_so_far[next] = new_cost
                    priority = new_cost + min(abs(goal[0] - next[0]) + abs(goal[1] - next[1]) for goal in goals)
                    heappush(frontier, (priority, next))
                    came_from[next] = current

        # 如果没有找到路径，略微降低阈值
        threshold -= step
        if threshold < 0:
            break

    return None, None, None  # 没有找到路径时返回 None


def reconstruct_path(came_from, start, goals):
    if came_from is None:
        return []  # 没有路径时返回空路径

    current = next(goal for goal in goals if goal in came_from)
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path


def find_optimal_path(graph, target_areas, k_matrix, radius_index):
    start = target_areas[0]
    targets = set(target_areas)
    optimal_path = []
    final_threshold = None
    while targets:
        came_from, cost_so_far, final_threshold = dynamic_threshold_a_star(graph, start, targets, k_matrix,
                                                                           radius_index)
        if came_from is None:
            print(f"No path found from {start} to any target even after adjusting threshold.")
            break  # 如果没有找到有的路径就直接退出

        path = reconstruct_path(came_from, start, targets)
        optimal_path.extend(path)
        targets.difference_update(path)
        start = path[-1]

    return optimal_path, final_threshold


def visualize_path(optimal_path, target_areas):
    plt.figure(figsize=(10, 6))

    # 将子区域索引转换为实际纬度和经度
    lat_step = (california_bounds[1] - california_bounds[0]) / num_sub_areas_y
    lon_step = (california_bounds[3] - california_bounds[2]) / num_sub_areas_x

    for i, (y, x) in enumerate(optimal_path):
        lat = california_bounds[0] + y * lat_step
        lon = california_bounds[2] + x * lon_step
        # plt.scatter(lon, lat, color='red')
        if i > 0:
            prev_y, prev_x = optimal_path[i - 1]
            prev_lat = california_bounds[0] + prev_y * lat_step
            prev_lon = california_bounds[2] + prev_x * lon_step
            plt.plot([prev_lon, lon], [prev_lat, lat], color='red', linestyle='--')

    plt.title("Optimal Path")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()


def visualize_k_matrix_and_buildings(k_matrix_file, all_buildings_file):
    # 加载 k_matrix.npy 和所有建筑物总和 .npy 文件
    k_matrix = np.load(k_matrix_file)
    all_buildings = np.load(all_buildings_file)

    # 创建图形
    plt.figure(figsize=(10, 6))

    # 绘制 K 值矩阵
    plt.imshow(k_matrix[:, :, radius_index],
               extent=[california_bounds[2], california_bounds[3], california_bounds[0], california_bounds[1]],
               origin='lower', cmap='viridis', interpolation='nearest')
    plt.colorbar(label="K Value")

    plt.title(f"K Value Distribution with Buildings (Radius {r[radius_index]})")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()


def find_optimal_plan(data):
    data = load_data(data)
    k_matrix = generate_k_matrix(data)
    graph = generate_graph()

    lat_step = (california_bounds[1] - california_bounds[0]) / num_sub_areas_y
    lon_step = (california_bounds[3] - california_bounds[2]) / num_sub_areas_x

    all_places = data
    target_areas = [get_sub_area_index(lat, lon, california_bounds, lat_step, lon_step) for lat, lon in all_places]

    np.save('k_matrix.npy', k_matrix)
    np.save('all_buildings.npy', all_places)

    visualize_k_matrix_and_buildings('k_matrix.npy', 'all_buildings.npy')

    optimal_path, final_threshold = find_optimal_path(graph, target_areas, k_matrix, radius_index)

    if final_threshold is not None:
        visualize_path(optimal_path, target_areas)
        total_distance = len(optimal_path) - 1
        print("Total Distance:", total_distance)
        return optimal_path, total_distance, k_matrix
    else:
        print("No valid path found.")
        return None, None, k_matrix


if __name__ == '__main__':
    data = 'data_hospital'
    find_optimal_plan(data)
