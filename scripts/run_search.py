"""Entry point: run the FunSearch-style evolutionary loop."""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parser import load_instance
from src.evolution import run_evolution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=str, required=True, help="Path to .vrp file")
    parser.add_argument("--calls", type=int, default=50, help="Number of LLM calls")
    parser.add_argument("--islands", type=int, default=5, help="Number of islands")
    args = parser.parse_args()

    inst = load_instance(args.instance)
    print(f"Loaded instance: {inst.name}, n={inst.n_customers}, Q={inst.capacity}")

    best = run_evolution(
        instances=[inst],
        n_islands=args.islands,
        max_calls=args.calls,
    )

    if best:
        os.makedirs("results", exist_ok=True)
        with open("results/best_heuristic.py", "w") as f:
            f.write(best.code)
        print("\nSaved best heuristic to results/best_heuristic.py")


if __name__ == "__main__":
    main()