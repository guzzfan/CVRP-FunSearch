"""Execute LLM-generated code in an isolated subprocess with timeout.
Also provides AST-based node counting for code simplicity."""
import ast
import subprocess
import sys
import tempfile
import textwrap
from typing import Optional, Tuple
import os

BLOCKED_MODULES = {"os", "subprocess", "sys", "socket", "requests", "shutil", "pathlib"}
BLOCKED_NAMES = {"eval", "exec", "__import__", "compile", "open"}


def validate_code(code: str) -> Tuple[bool, str]:
    """Static AST validation to block dangerous imports and calls."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name.split(".")[0]
                if mod in BLOCKED_MODULES:
                    return False, f"Import of '{mod}' is not allowed"
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod in BLOCKED_MODULES:
                return False, f"Import from '{mod}' is not allowed"
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in BLOCKED_NAMES:
                return False, f"Use of '{node.id}' is not allowed"

    return True, "OK"


def count_ast_nodes(code: str) -> int:
    """Count AST nodes as a proxy for code complexity."""
    try:
        tree = ast.parse(code)
        return sum(1 for _ in ast.walk(tree))
    except SyntaxError:
        return -1


def execute_priority_fn(
    code: str,
    timeout: int = 60,
) -> Tuple[bool, Optional[str], Optional[int]]:
    """
    Run user code in a subprocess. The code must define a `priority` function.
    """
    is_safe, msg = validate_code(code)
    if not is_safe:
        return False, msg, None

    ast_nodes = count_ast_nodes(code)

    # 使用 replace 而不是 f-string，避免字典大括号转义报错
    driver_template = """
import json
import math
import numpy as np

{generated_code}

# Tiny synthetic test
coords = np.array([[0,0],[1,0],[2,0]], dtype=float)
demands = np.array([0, 1, 1], dtype=float)
visited = np.array([True, False, False])

try:
    # 调用我们约定的 6 参数版本
    s0 = priority(0, 1, coords, demands, 10.0, visited)
    s1 = priority(0, 2, coords, demands, 10.0, visited)
    assert isinstance(s0, (int, float)) and isinstance(s1, (int, float))
    print(json.dumps({"ok": True}))
except Exception as e:
    print(json.dumps({"ok": False, "error": str(e)}))
"""
    driver = driver_template.replace("{generated_code}", code)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(driver)
        path = f.name

    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return False, result.stderr or "Execution failed", ast_nodes

        import json
        out = json.loads(result.stdout.strip())
        if out.get("ok"):
            return True, None, ast_nodes
        else:
            return False, out.get("error", "Unknown error"), ast_nodes
    except subprocess.TimeoutExpired:
        return False, "Execution timed out", ast_nodes
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass