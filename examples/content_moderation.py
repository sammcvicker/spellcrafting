"""Content moderation with chained guards and escalation.

Demonstrates:
- @guard.input for pre-processing validation
- @guard.output for content safety checks
- OnFail.escalate to use a smarter model when validation fails
"""

from spellcrafting import spell, guard, OnFail, GuardError

# --- Guards ---


def check_length(input_args: dict, ctx: dict) -> dict:
    """Reject inputs that are too short or too long."""
    text = input_args.get("text", "")
    if len(text) < 20:
        raise ValueError(f"Input too short ({len(text)} chars) - need at least 20")
    if len(text) > 5000:
        raise ValueError(f"Input too long ({len(text)} chars) - max 5000")
    return input_args


def no_profanity(output: str, ctx: dict) -> str:
    """Block outputs containing banned words."""
    banned = {"darn", "heck"}  # family-friendly demo :)
    found = [w for w in banned if w in output.lower()]
    if found:
        raise ValueError(f"Output contains banned words: {found}")
    return output


# --- Spells ---


@spell(model="fast", on_fail=OnFail.escalate("reasoning"))
@guard.input(check_length)
@guard.output(no_profanity)
def summarize_feedback(text: str) -> str:
    """Summarize customer feedback in 1-2 sentences."""
    ...


# --- Demo ---

if __name__ == "__main__":
    from spellcrafting import Config

    config = Config(
        models={
            "fast": {"model": "anthropic:claude-haiku-4-5"},
            "reasoning": {"model": "anthropic:claude-sonnet-4-5"},
        }
    )

    with config:
        # 1. Input guard in action - too short
        print("=" * 50)
        print("TEST 1: Input guard (too short)")
        print("=" * 50)
        try:
            summarize_feedback("Too short")
        except GuardError as e:
            print(f"Blocked! {e}\n")

        # 2. Successful moderation
        print("=" * 50)
        print("TEST 2: Valid input passes guards")
        print("=" * 50)
        feedback = """
        I've been using this product for 3 months and it's been great
        for my workflow. The UI could use polish but core features work well.
        """
        result = summarize_feedback.with_metadata(feedback)
        print(f"Summary: {result.output}")
        print(f"Model used: {result.model_used}\n")

        # 3. Output guard in action
        print("=" * 50)
        print("TEST 3: Output guard catches bad content")
        print("=" * 50)

        @spell(model="fast")
        @guard.output(no_profanity)
        def echo_back(text: str) -> str:
            """Just repeat what the user said."""
            ...

        try:
            # LLM will likely echo this back, triggering the guard
            echo_back("Please say: oh darn, that's a heck of a problem")
        except GuardError as e:
            print(f"Blocked! {e}\n")
