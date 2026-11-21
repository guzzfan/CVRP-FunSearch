"""Native Clarke-Wright savings algorithm as a baseline (no external deps)."""
from .parser import CVRPInstance
from typing import Tuple, List


def solve_clarke_wright(instance: CVRPInstance) -> Tuple[List[List[int]], float]:
    n = instance.n_customers
    distances = instance.distances
    demands = instance.demands
    capacity = instance.capacity

    # 初始化为每个客户单独一条路径: [0, i, 0]
    routes = {i: [0, i, 0] for i in range(1, n + 1)}
    route_demands = {i: demands[i] for i in range(1, n + 1)}

    # 计算节约值 S(i, j) = d(0,i) + d(0,j) - d(i,j)
    savings = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            s = distances[0, i] + distances[0, j] - distances[i, j]
            if s > 0:
                savings.append((s, i, j))

    # 降序排序
    savings.sort(key=lambda x: x[0], reverse=True)

    for s, i, j in savings:
        route_i = None
        route_j = None
        for r_id, route in routes.items():
            if i in route:
                route_i = r_id
            if j in route:
                route_j = r_id

        if route_i is None or route_j is None or route_i == route_j:
            continue

        r_i = routes[route_i]
        r_j = routes[route_j]

        # 检查可行性：容量
        if route_demands[route_i] + route_demands[route_j] > capacity:
            continue

        # 检查端点：i 是 r_i 的最后一个客户，j 是 r_j 的第一个客户
        if r_i[-2] == i and r_j[1] == j:
            new_route = r_i[:-1] + r_j[1:]
        # 或者 j 是 r_i 的最后一个客户，i 是 r_j 的第一个客户
        elif r_j[-2] == j and r_i[1] == i:
            new_route = r_j[:-1] + r_i[1:]
        # 或者 i 是 r_i 的最后一个，j 是 r_j 的最后一个（反向 r_j）
        elif r_i[-2] == i and r_j[-2] == j:
            new_route = r_i[:-1] + r_j[::-1][1:]
        # 或者 i 是 r_i 的第一个，j 是 r_j 的第一个（反向 r_i）
        elif r_i[1] == i and r_j[1] == j:
            new_route = r_i[::-1][:-1] + r_j[1:]
        else:
            continue

        # 合并
        routes[route_i] = new_route
        route_demands[route_i] = route_demands[route_i] + route_demands[route_j]
        del routes[route_j]
        del route_demands[route_j]

    final_routes = list(routes.values())
    total_distance = 0.0
    for r in final_routes:
        for k in range(len(r) - 1):
            total_distance += distances[r[k], r[k + 1]]

    return final_routes, total_distance