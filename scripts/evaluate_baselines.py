import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parser import load_instance
from src.solver import construct_routes, verify_solution
from src.baseline import solve_clarke_wright

def nearest_neighbor_priority(current_idx, customer_idx, coords, demands, remaining_capacity, visited_mask):
    if remaining_capacity < demands[customer_idx]:
        return -1e9
    # 距离当前节点最近的客户（真正的最短视距）
    return -np.linalg.norm(coords[customer_idx] - coords[current_idx])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=str, required=True)
    args = parser.parse_args()

    inst = load_instance(args.instance)
    print(f"Instance: {inst.name}, n={inst.n_customers}, Q={inst.capacity}")

    # 1. Clarke-Wright
    try:
        cw_routes, cw_dist = solve_clarke_wright(inst)
        print(f"\n[Clarke-Wright] distance = {cw_dist:.2f}")
    except Exception as e:
        print(f"\n[Clarke-Wright] failed: {e}")

    # 2. Nearest-Neighbor
    nn_routes, nn_dist = construct_routes(inst, nearest_neighbor_priority)
    ok, msg = verify_solution(inst, nn_routes)
    print(f"\n[Nearest-Neighbor] distance = {nn_dist:.2f}, feasible = {ok}")

if __name__ == "__main__":
    main()