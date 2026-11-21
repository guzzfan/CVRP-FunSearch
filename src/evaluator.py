"""Multi-objective evaluator: distance, AST complexity, and runtime."""
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from .parser import CVRPInstance
from .solver import construct_routes, verify_solution, PriorityFn
from .sandbox import execute_priority_fn

# Weights for weighted scalarization
W_DISTANCE = 0.6
W_SIMPLICITY = 0.25
W_RUNTIME = 0.15

# Normalization constants (empirically tuned)
MAX_DISTANCE = 2000.0
MAX_AST = 500
MAX_RUNTIME = 2.0  # seconds


def evaluate_code_on_instance(
    code: str,
    instance: CVRPInstance,
) -> Dict:
    """
    Run one candidate on one instance.
    Returns a dict with status, distance, ast_nodes, runtime, and objective score.
    """
    # Step 1: sandbox validation
    ok, err, ast_nodes = execute_priority_fn(code, timeout=30)
    if not ok:
        return {
            "valid": False,
            "error": err,
            "distance": None,
            "ast_nodes": ast_nodes,
            "runtime": None,
            "objective": -1e9,
        }

    # Step 2: compile the priority function in-process
    import math
    namespace = {}
    try:
        # 传入 np, numpy 和 math 给生成的代码使用
        exec(code, {"np": np, "numpy": np, "math": math}, namespace)
        priority_fn: PriorityFn = namespace["priority"]
    except Exception as e:
        return {
            "valid": False,
            "error": f"Compile error: {e}",
            "distance": None,
            "ast_nodes": ast_nodes,
            "runtime": None,
            "objective": -1e9,
        }

    # Step 3: run constructive solver
    t0 = time.perf_counter()
    try:
        routes, total_distance = construct_routes(instance, priority_fn)
        runtime = time.perf_counter() - t0
    except Exception as e:
        return {
            "valid": False,
            "error": f"Solver error: {e}",
            "distance": None,
            "ast_nodes": ast_nodes,
            "runtime": None,
            "objective": -1e9,
        }

    # Step 4: feasibility check
    feasible, msg = verify_solution(instance, routes)
    if not feasible:
        return {
            "valid": False,
            "error": f"Infeasible solution: {msg}",
            "distance": total_distance,
            "ast_nodes": ast_nodes,
            "runtime": runtime,
            "objective": -1e8,
        }

    # Step 5: multi-objective scalarization
    dist_norm = min(total_distance / MAX_DISTANCE, 2.0)
    ast_norm = min(ast_nodes / MAX_AST, 2.0)
    time_norm = min(runtime / MAX_RUNTIME, 2.0)

    objective = (
        - W_DISTANCE * dist_norm
        - W_SIMPLICITY * ast_norm
        - W_RUNTIME * time_norm
    )

    return {
        "valid": True,
        "error": None,
        "distance": total_distance,
        "ast_nodes": ast_nodes,
        "runtime": runtime,
        "objective": objective,
    }


def evaluate_on_suite(
    code: str,
    instances: List[CVRPInstance],
) -> Dict:
    """
    Evaluate a candidate on a suite of instances.
    Returns aggregate statistics (mean distance, mean objective, etc.).
    """
    results = [evaluate_code_on_instance(code, inst) for inst in instances]

    valid_results = [r for r in results if r["valid"]]
    if not valid_results:
        return {
            "valid_count": 0,
            "total_count": len(results),
            "mean_distance": float("inf"),
            "mean_objective": -1e9,
            "mean_ast": float("inf"),
            "mean_runtime": float("inf"),
        }

    return {
        "valid_count": len(valid_results),
        "total_count": len(results),
        "mean_distance": float(np.mean([r["distance"] for r in valid_results])),
        "mean_objective": float(np.mean([r["objective"] for r in valid_results])),
        "mean_ast": float(np.mean([r["ast_nodes"] for r in valid_results])),
        "mean_runtime": float(np.mean([r["runtime"] for r in valid_results])),
    }