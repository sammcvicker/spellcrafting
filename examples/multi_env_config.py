"""Multi-environment configuration management.

Demonstrates:
- Config context manager for scoped model selection
- Model aliases (fast/reasoning) vs literal model strings
- Runtime config switching without code changes
- .with_metadata() to see which model actually ran
"""

from spellcrafting import spell, Config

# --- Spells use aliases, not literal models ---


@spell(model="fast")
def quick_classify(text: str) -> str:
    """Classify text into: question, statement, or command."""
    ...


@spell(model="reasoning")
def deep_analysis(text: str) -> str:
    """Provide detailed analysis of the text's tone, intent, and key themes."""
    ...


# --- Environment configs ---

DEV_CONFIG = Config(
    models={
        "fast": {"model": "anthropic:claude-haiku-4-5", "temperature": 0.3},
        "reasoning": {"model": "anthropic:claude-haiku-4-5"},  # cheap for dev!
    }
)

PROD_CONFIG = Config(
    models={
        "fast": {"model": "anthropic:claude-haiku-4-5", "temperature": 0.1},
        "reasoning": {"model": "anthropic:claude-sonnet-4-5", "temperature": 0.7},
    }
)


# --- Demo ---

if __name__ == "__main__":
    text = "Can you help me understand how this feature works?"

    # Run same spell with different configs
    print("=" * 50)
    print("Same spell, different configs")
    print("=" * 50)

    print("\n[DEV MODE] - both aliases use haiku (cheap!)")
    with DEV_CONFIG:
        result = deep_analysis.with_metadata(text)
        print(f"  'reasoning' resolved to: {result.model_used}")
        print(f"  Result: {result.output[:80]}...")

    print("\n[PROD MODE] - reasoning uses sonnet")
    with PROD_CONFIG:
        result = deep_analysis.with_metadata(text)
        print(f"  'reasoning' resolved to: {result.model_used}")
        print(f"  Result: {result.output[:80]}...")

    # Show token/cost difference
    print("\n" + "=" * 50)
    print("Cost comparison")
    print("=" * 50)

    with DEV_CONFIG:
        dev_result = quick_classify.with_metadata(text)
        print(f"DEV:  {dev_result.total_tokens} tokens, {dev_result.duration_ms:.0f}ms")

    with PROD_CONFIG:
        prod_result = quick_classify.with_metadata(text)
        print(f"PROD: {prod_result.total_tokens} tokens, {prod_result.duration_ms:.0f}ms")
