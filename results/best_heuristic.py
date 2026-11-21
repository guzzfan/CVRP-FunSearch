import numpy as np

def priority(current_idx, customer_idx, coords, demands, remaining_capacity, visited_mask) -> float:
    # Distance from current location to candidate
    dx = coords[current_idx, 0] - coords[customer_idx, 0]
    dy = coords[current_idx, 1] - coords[customer_idx, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    
    # Demand relative to remaining capacity: prefer customers that use capacity well
    # but avoid ones that would leave awkward leftover capacity
    dem = demands[customer_idx]
    capacity_fit = 1.0 - abs(remaining_capacity - dem) / (remaining_capacity + 1e-9)
    
    # Clustering: prefer customers close to already visited customers in current route
    visited_customers = np.where(visited_mask)[0]
    visited_customers = visited_customers[visited_customers != 0]
    
    if len(visited_customers) > 0:
        # Distance to nearest visited customer
        diffs = coords[visited_customers] - coords[customer_idx]
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        min_dist_to_visited = np.min(dists)
    else:
        # No visited customers yet, just use depot distance
        min_dist_to_visited = 0.0
    
    # Score: higher is better
    # Primary: minimize distance from current (nearest-neighbor)
    # Secondary: clustering bonus (close to visited customers)
    # Tertiary: capacity fit bonus
    score = -dist + 0.3 * capacity_fit * (remaining_capacity + 1e-9) - 0.2 * min_dist_to_visited
    
    return score