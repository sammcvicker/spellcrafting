"""SpellResult type for accessing execution metadata alongside spell output.

Design Note: Pydantic vs Dataclass Usage
----------------------------------------
SpellResult uses a dataclass (not Pydantic) because:
- It's an internal type created by spell execution, not user-provided input
- No parsing or validation from external sources is needed
- Dataclasses have lower overhead and simpler semantics
- Generic[T] support is cleaner with dataclasses

See config.py for contrast - it uses Pydantic for parsing user configuration
where validation and helpful error messages are important.
"""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Generic, ParamSpec, Protocol, TypeVar

P = ParamSpec("P")
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


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
        print(result.total_tokens)  # 150
        print(result.cost_estimate)  # 0.00015 (USD)
        print(result.trace_id)  # "abc123..." (for log correlation)

    Attributes:
        output: The actual spell output (the value you'd get from a normal call)
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens used
        model_used: The actual model used (may differ from alias if escalation occurred)
        attempt_count: Number of execution attempts (1 = no retries)
        duration_ms: Execution duration in milliseconds
        cost_estimate: Estimated cost in USD (None if pricing unavailable)
        trace_id: Trace ID for correlation with logs (None if no trace context)
    """

    output: T

    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0

    # Execution info
    model_used: str = ""
    attempt_count: int = 1
    duration_ms: float = 0.0

    # Extended metadata
    cost_estimate: float | None = None
    trace_id: str | None = None

    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)."""
        return self.input_tokens + self.output_tokens


class SyncSpell(Protocol[T_co]):
    """Protocol for synchronous spell-decorated functions.

    This allows proper type hints for functions that accept spell functions:

        def run_spell(spell_fn: SyncSpell[Result]) -> SpellResult[Result]:
            return spell_fn.with_metadata("input")

    Note: Due to Protocol limitations with ParamSpec, this Protocol captures
    the essential spell interface (callable + with_metadata) but not the
    exact parameter signature. Use for type hints when you need to express
    "any sync spell returning T".
    """

    def __call__(self, *args: object, **kwargs: object) -> T_co:
        """Call the spell with arguments."""
        ...

    def with_metadata(self, *args: object, **kwargs: object) -> SpellResult[T_co]:
        """Call the spell and return output with execution metadata."""
        ...


class AsyncSpell(Protocol[T_co]):
    """Protocol for asynchronous spell-decorated functions.

    This allows proper type hints for functions that accept async spell functions:

        async def run_spell(spell_fn: AsyncSpell[Result]) -> SpellResult[Result]:
            return await spell_fn.with_metadata("input")

    Note: Due to Protocol limitations with ParamSpec, this Protocol captures
    the essential spell interface (callable + with_metadata) but not the
    exact parameter signature. Use for type hints when you need to express
    "any async spell returning T".
    """

    def __call__(self, *args: object, **kwargs: object) -> Awaitable[T_co]:
        """Call the spell with arguments."""
        ...

    def with_metadata(self, *args: object, **kwargs: object) -> Awaitable[SpellResult[T_co]]:
        """Call the spell and return output with execution metadata."""
        ...
