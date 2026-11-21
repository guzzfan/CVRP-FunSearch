"""CVRPLIB instance parser using the vrplib library."""
import vrplib
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class CVRPInstance:
    name: str
    n_customers: int
    capacity: float
    depot: np.ndarray          # shape (2,)
    coords: np.ndarray         # shape (n+1, 2), index 0 is depot
    demands: np.ndarray        # shape (n+1,), index 0 is depot (demand 0)
    distances: np.ndarray      # shape (n+1, n+1)

def load_instance(path: str) -> CVRPInstance:
    """Load a .vrp file from CVRPLIB."""
    inst = vrplib.read_instance(path)
    coords = inst["node_coord"]
    demands = inst["demand"]
    capacity = float(inst["capacity"])
    distances = inst["edge_weight"].astype(float)

    return CVRPInstance(
        name=inst.get("name", "unknown"),
        n_customers=len(demands) - 1,
        capacity=capacity,
        depot=coords[0],
        coords=coords,
        demands=demands,
        distances=distances,
    )