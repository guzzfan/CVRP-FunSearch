"""Constructive solver shell: builds routes by calling the LLM-evolved priority function."""
import numpy as np
from typing import Callable, List, Tuple
from .parser import CVRPInstance

# 注意：增加了 current_idx (当前车辆所在节点)
PriorityFn = Callable[[int, int, np.ndarray, np.ndarray, float, np.ndarray], float]


def construct_routes(
        instance: CVRPInstance,
        priority_fn: PriorityFn,
        max_routes: int = 100,
) -> Tuple[List[List[int]], float]:
    n = instance.n_customers
    unvisited = set(range(1, n + 1))
    routes: List[List[int]] = []
    visited_mask = np.zeros(n + 1, dtype=bool)
    visited_mask[0] = True

    while unvisited and len(routes) < max_routes:
        route = [0]
        load = 0.0
        current = 0

        while True:
            feasible = [
                c for c in unvisited
                if load + instance.demands[c] <= instance.capacity
            ]
            if not feasible:
                break

            scores = []
            for c in feasible:
                try:
                    # 传入 current 和 c
                    score = priority_fn(
                        current,
                        c,
                        instance.coords,
                        instance.demands,
                        instance.capacity - load,
                        visited_mask.copy(),
                    )
                    if not isinstance(score, (int, float)) or np.isnan(score):
                        score = -1e9
                except Exception:
                    score = -1e9
                scores.append((score, c))

            scores.sort(reverse=True)
            next_c = scores[0][1]

            route.append(next_c)
            load += instance.demands[next_c]
            unvisited.remove(next_c)
            visited_mask[next_c] = True
            current = next_c

        route.append(0)
        routes.append(route)

    total_distance = 0.0
    for r in routes:
        for i in range(len(r) - 1):
            total_distance += instance.distances[r[i], r[i + 1]]

    return routes, total_distance


def verify_solution(instance: CVRPInstance, routes: List[List[int]], tolerance: float = 1e-6) -> Tuple[bool, str]:
    visited = set()
    for route in routes:
        load = 0.0
        for c in route[1:-1]:
            if c in visited:
                return False, f"Customer {c} visited more than once"
            visited.add(c)
            load += instance.demands[c]
        if load > instance.capacity + tolerance:
            return False, f"Route {route} exceeds capacity ({load} > {instance.capacity})"

    if len(visited) != instance.n_customers:
        return False, f"Visited {len(visited)} customers, expected {instance.n_customers}"

    return True, "OK"