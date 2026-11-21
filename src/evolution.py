"""Island-model evolutionary search over generated priority functions."""
import random
import numpy as np
from typing import List, Dict, Optional, Tuple
from .parser import CVRPInstance
from .evaluator import evaluate_on_suite
from .llm_api import generate_candidate, extract_code
from .sandbox import count_ast_nodes

# ---- Program database entries ----

class ProgramEntry:
    __slots__ = ("code", "score", "ast_nodes", "metrics")

    def __init__(self, code: str, score: float, ast_nodes: int, metrics: Dict):
        self.code = code
        self.score = score
        self.ast_nodes = ast_nodes
        self.metrics = metrics


# ---- Island model ----

class Island:
    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.programs: List[ProgramEntry] = []

    def add(self, entry: ProgramEntry):
        self.programs.append(entry)
        self.programs.sort(key=lambda p: p.score, reverse=True)
        self.programs = self.programs[: self.capacity]


class ProgramDatabase:
    def __init__(self, n_islands: int = 5, island_capacity: int = 10):
        self.islands = [Island(island_capacity) for _ in range(n_islands)]
        self._seen_hashes = set()

    @staticmethod
    def _hash_code(code: str) -> str:
        """Simple AST-normalized hash for deduplication."""
        import ast
        try:
            tree = ast.parse(code)
            # Use structural dump without attributes/names
            return str(ast.dump(tree, annotate_fields=False))
        except SyntaxError:
            return code[:200]  # fallback

    def try_add(self, code: str, score: float, ast_nodes: int, metrics: Dict) -> bool:
        """Add a program if it is not a duplicate."""
        h = self._hash_code(code)
        if h in self._seen_hashes:
            return False
        self._seen_hashes.add(h)
        entry = ProgramEntry(code, score, ast_nodes, metrics)
        island = random.choice(self.islands)
        island.add(entry)
        return True

    def best_program(self) -> Optional[ProgramEntry]:
        best = None
        for island in self.islands:
            if island.programs:
                top = island.programs[0]
                if best is None or top.score > best.score:
                    best = top
        return best

    def sample_context(self) -> Tuple[str, str]:
        """
        Sample a context for the LLM prompt.
        Returns (best_code, best_score_str).
        """
        best = self.best_program()
        if best is None:
            return "", "N/A"
        return best.code, f"{best.metrics.get('mean_distance', 'N/A')} (objective {best.score:.4f})"


# ---- Prompt builder ----

def build_prompt(context_code: str, best_score: str, instance_name: str) -> str:
    parts = [
        f"We are solving CVRP instance: {instance_name}.",
        "Here is the current best priority function found so far:",
        "```python",
        context_code if context_code else "# No previous program yet. Write a new one.",
        "```",
        f"Its performance: {best_score}.",
        "",
        "Please propose an improved version of the `priority` function.",
        "Focus on reducing total travel distance while keeping the code simple and fast.",
        "You may try a different strategy (e.g., nearest-neighbor, capacity-aware, clustering).",
        "Output ONLY the function code.",
    ]
    return "\n".join(parts)


# ---- Main evolution loop ----

def run_evolution(
    instances: List[CVRPInstance],
    n_islands: int = 5,
    island_capacity: int = 10,
    max_calls: int = 100,
    seed: int = 42,
):
    random.seed(seed)
    np.random.seed(seed)

    db = ProgramDatabase(n_islands, island_capacity)
    best_overall: Optional[ProgramEntry] = None

    # Bootstrap with a simple nearest-neighbor heuristic
    bootstrap_code = """
    def priority(current_idx, customer_idx, coords, demands, remaining_capacity, visited_mask):
        if remaining_capacity < demands[customer_idx]:
            return -1e9
        # Simple nearest-neighbor from current node
        dist = np.linalg.norm(coords[customer_idx] - coords[current_idx])
        return -dist
    """
    ok = True
    metrics = evaluate_on_suite(bootstrap_code, instances)
    if metrics["valid_count"] > 0:
        db.try_add(bootstrap_code, metrics["mean_objective"], metrics["mean_ast"], metrics)
        print(f"[Bootstrap] valid={metrics['valid_count']}, distance={metrics['mean_distance']:.2f}")

    call_count = 0
    while call_count < max_calls:
        context_code, best_score = db.sample_context()
        instance_name = instances[0].name if instances else "CVRP"
        prompt = build_prompt(context_code, best_score, instance_name)

        print(f"[Call {call_count + 1}/{max_calls}] Querying LLM...")
        raw = generate_candidate(prompt)
        code = extract_code(raw)
        if code is None:
            print("  -> No valid code extracted, skipping.")
            call_count += 1
            continue

        metrics = evaluate_on_suite(code, instances)
        if metrics["valid_count"] == 0:
            print("  -> Candidate invalid on all instances, skipping.")
        else:
            added = db.try_add(
                code,
                metrics["mean_objective"],
                metrics["mean_ast"],
                metrics,
            )
            if added:
                print(
                    f"  -> Accepted. distance={metrics['mean_distance']:.2f}, "
                    f"ast={metrics['mean_ast']:.0f}, runtime={metrics['mean_runtime']:.3f}s"
                )
            else:
                print("  -> Duplicate (AST hash), skipping.")

        call_count += 1

    best = db.best_program()
    if best is not None:
        print("\n=== Best program found ===")
        print(f"Objective: {best.score:.4f}")
        print(f"Metrics: {best.metrics}")
        print("Code:\n")
        print(best.code)
        return best
    else:
        print("No valid program found.")
        return None