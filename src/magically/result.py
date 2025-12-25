"""SpellResult type for accessing execution metadata alongside spell output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class SpellResult(Generic[T]):
    """Result wrapper that includes execution metadata alongside the output.

    This type is returned when calling a spell with `.with_metadata()`:

        @spell(model="fast")
        def classify(text: str) -> Category:
            '''Classify the text.'''
            ...

        # Normal call - just returns Category
        result = classify("some text")

        # With metadata - returns SpellResult[Category]
        result = classify.with_metadata("some text")
        print(result.output)  # Category
        print(result.input_tokens)  # 50
        print(result.model_used)  # "openai:gpt-4o-mini"

    Attributes:
        output: The actual spell output (the value you'd get from a normal call)
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens used
        model_used: The actual model used (may differ from alias if escalation occurred)
        attempt_count: Number of execution attempts (1 = no retries)
        duration_ms: Execution duration in milliseconds
    """

    output: T

    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0

    # Execution info
    model_used: str = ""
    attempt_count: int = 1
    duration_ms: float = 0.0
