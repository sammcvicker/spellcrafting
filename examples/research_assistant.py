"""Research assistant with tools and observability.

Demonstrates:
- Tools for extending spell capabilities (LLM can call Python functions)
- .with_metadata() for token/cost tracking
- Visible tool execution
"""

from spellcrafting import spell, Config


# --- Tools the LLM can call ---


def search_wikipedia(query: str) -> str:
    """Search Wikipedia for information."""
    print(f"    [tool] search_wikipedia('{query}')")
    # Fake response for demo - real impl would call Wikipedia API
    responses = {
        "france population": "France has a population of approximately 68 million people (2024).",
        "python programming": "Python is a high-level programming language created by Guido van Rossum.",
    }
    for key, response in responses.items():
        if key in query.lower():
            return response
    return f"Wikipedia article about '{query}' - contains detailed information."


def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    print(f"    [tool] calculate('{expression}')")
    try:
        # Safe eval with no builtins
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# --- Spell with tools ---


@spell(model="reasoning", tools=[search_wikipedia, calculate])
def research(question: str) -> str:
    """Answer the question using available tools when helpful.
    Cite sources when using search results.
    """
    ...


# --- Demo ---

if __name__ == "__main__":
    config = Config(
        models={
            "reasoning": {"model": "anthropic:claude-sonnet-4-5"},
        }
    )

    with config:
        print("=" * 50)
        print("Research Assistant (watch for tool calls)")
        print("=" * 50)

        question = "What is the population of France divided by 1000?"
        print(f"\nQuestion: {question}\n")
        print("Tool calls:")

        result = research.with_metadata(question)

        print(f"\nAnswer: {result.output}")

        print("\n--- Execution Stats ---")
        print(f"Model: {result.model_used}")
        print(f"Tokens: {result.input_tokens} in / {result.output_tokens} out")
        print(f"Duration: {result.duration_ms:.0f}ms")

        if result.cost_estimate:
            print(f"Estimated cost: ${result.cost_estimate:.6f}")
