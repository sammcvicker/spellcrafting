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
from typing import TYPE_CHECKING, Generic, ParamSpec, Protocol, TypeVar

if TYPE_CHECKING:
    from spellcrafting.logging import SpellExecutionLog

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

    def __repr__(self) -> str:
        """Return a concise representation, truncating long output."""
        output_repr = repr(self.output)
        if len(output_repr) > 100:
            output_repr = output_repr[:97] + "..."
        return (
            f"SpellResult(output={output_repr}, "
            f"tokens={self.total_tokens}, "
            f"model={self.model_used!r}, "
            f"duration_ms={self.duration_ms:.1f})"
        )

    def content_eq(self, other: SpellResult[T]) -> bool:
        """Compare content without timing/tokens.

        This is useful for testing when you want to verify the output
        and model match but don't care about variable fields like
        duration_ms, token counts, or attempt_count.

        Args:
            other: Another SpellResult to compare against

        Returns:
            True if output and model_used match, False otherwise
        """
        return self.output == other.output and self.model_used == other.model_used

    @classmethod
    def from_execution_log(cls, output: T, log: SpellExecutionLog) -> SpellResult[T]:
        """Create a SpellResult from execution log data.

        This allows with_metadata to reuse the core execution path and
        construct the result from captured log metadata.

        Args:
            output: The spell output value
            log: The execution log with metadata

        Returns:
            SpellResult populated from the log
        """
        # Only include cost estimate if there are actual tokens used
        cost = None
        if log.cost_estimate and log.token_usage.total_tokens > 0:
            cost = log.cost_estimate.total_cost

        return cls(
            output=output,
            input_tokens=log.token_usage.input_tokens,
            output_tokens=log.token_usage.output_tokens,
            model_used=log.model,
            attempt_count=(log.validation.attempt_count if log.validation else 1),
            duration_ms=log.duration_ms or 0.0,
            cost_estimate=cost,
            trace_id=log.trace_id,
        )


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
