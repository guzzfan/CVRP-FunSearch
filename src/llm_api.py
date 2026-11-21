"""DeepSeek API wrapper for generating candidate priority functions."""
import os
from openai import OpenAI
from typing import Optional

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise EnvironmentError("DEEPSEEK_API_KEY environment variable is not set.")
        _client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
    return _client


SYSTEM_PROMPT = """You are an expert in combinatorial optimization and Python programming.
You will write a Python function named `priority` for a Capacitated Vehicle Routing Problem (CVRP).

The solver works as follows:
- It builds routes one by one. In each route, it starts at the depot (index 0).
- At each step, among capacity-feasible unvisited customers, it picks the customer with the HIGHEST score returned by `priority`.
- The route ends when no unvisited customer can fit in the remaining capacity, then the vehicle returns to the depot.

Your function signature MUST be:
def priority(current_idx, customer_idx, coords, demands, remaining_capacity, visited_mask) -> float:
    ...

Parameters:
- current_idx: int, the current location of the vehicle (0 is depot, 1..n are customers)
- customer_idx: int, the candidate customer index to evaluate
- coords: numpy array of shape (n+1, 2), coordinates
- demands: numpy array of shape (n+1,), demands (demands[0] = 0)
- remaining_capacity: float, remaining vehicle capacity
- visited_mask: numpy boolean array of shape (n+1,), True if already visited

Return: a float score. Higher means more likely to be selected next.

Strategy guidelines:
- Consider distance from the current_idx to the candidate.
- Consider demand relative to remaining capacity.
- Consider clustering: prefer customers near already-visited ones in the current route.
- You may use any numpy or math functions.

Only output the function code. No explanation, no markdown fences."""


def generate_candidate(
    prompt_context: str,
    temperature: float = 0.8,
    max_tokens: int = 1024,
) -> Optional[str]:
    """
    Ask the LLM to generate a candidate priority function.
    Returns the raw code string, or None on failure.
    """
    client = get_client()
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_context},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        return content
    except Exception as e:
        print(f"[LLM] API call failed: {e}")
        return None


def extract_code(raw: str) -> Optional[str]:
    """Extract Python code from LLM output (handles markdown fences)."""
    if raw is None:
        return None
    text = raw.strip()
    if text.startswith("```python"):
        text = text[len("```python"):]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    if "def priority" not in text:
        return None
    return text