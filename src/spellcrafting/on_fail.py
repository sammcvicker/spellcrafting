"""OnFail strategies for handling Pydantic validation failures in spells.

OnFail strategies define what happens when the LLM output fails Pydantic validation
after exhausting all retries. This extends PydanticAI's retry mechanism with more
sophisticated failure handling patterns.

Example:
    from spellcrafting import spell, OnFail

    # Escalate to a more capable model on validation failure
    @spell(model="fast", on_fail=OnFail.escalate("reasoning"))
    def complex_task(query: str) -> Analysis: ...

    # Return a default value instead of raising
    @spell(on_fail=OnFail.fallback(default=DefaultResponse()))
    def optional_enrichment(data: str) -> Enriched: ...

    # Custom handler for domain-specific fixes
    @spell(on_fail=OnFail.custom(my_fix_handler))
    def parse_dates(text: str) -> Dates: ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class RetryStrategy:
    """Default: retry with validation error in context (PydanticAI default)."""


@dataclass(frozen=True)
class EscalateStrategy:
    """Retry with a more capable model after validation failures exhaust retries."""

    model: str
    retries: int = 1  # Retries for the escalated model


@dataclass(frozen=True)
class FallbackStrategy(Generic[T]):
    """Return a default value instead of raising on validation failure."""

    default: T


@dataclass(frozen=True)
class CustomStrategy(Generic[T]):
    """Custom handler for domain-specific fixes or escalation logic.

    The handler receives:
        - error: The exception (UnexpectedModelBehavior or ValidationError)
        - attempt: The current attempt number (1-based)
        - context: Dict with spell_name, model, input_args

    It should return:
        - A valid output value to use instead of raising
        - Or raise an exception to propagate the error

    Example:
        def my_handler(error: Exception, attempt: int, context: dict) -> MyOutput:
            if "date format" in str(error):
                return fix_date_format(context["input_args"])
            raise error  # Re-raise if we can't fix it
    """

    handler: Callable[[Exception, int, dict[str, Any]], T]


@dataclass(frozen=True)
class RaiseStrategy:
    """Raise an error on failure (used for guards and validators)."""


@dataclass(frozen=True)
class FixStrategy:
    """Attempt to fix the value to satisfy validation (used for llm_validator)."""


# Type alias for all strategy types
OnFailStrategy = (
    RetryStrategy | EscalateStrategy | FallbackStrategy | CustomStrategy | RaiseStrategy | FixStrategy
)

# Type alias for validator-specific strategies
ValidatorOnFailStrategy = RaiseStrategy | FixStrategy


class OnFail:
    """Factory for on_fail strategies.

    For guards:
        @guard.input(my_guard, on_fail=OnFail.RAISE)  # default

    For spells:
        @spell(on_fail=OnFail.escalate("reasoning"))
        def my_spell(...): ...

        @spell(on_fail=OnFail.fallback(default=DefaultValue()))
        def my_spell(...): ...

    For validators:
        llm_validator("rule", on_fail=OnFail.RAISE)  # default
        llm_validator("rule", on_fail=OnFail.FIX)    # attempt to fix
    """

    # Strategy constants
    RAISE = RaiseStrategy()
    FIX = FixStrategy()

    @staticmethod
    def retry() -> RetryStrategy:
        """Default: retry with validation error in context.

        This is the default behavior - PydanticAI automatically retries
        with the validation error included in the context.

        Returns:
            RetryStrategy instance
        """
        return RetryStrategy()

    @staticmethod
    def escalate(model: str, *, retries: int = 1) -> EscalateStrategy:
        """Retry with a more capable model after retries are exhausted.

        When the original model fails validation after all retries, create
        a new agent with the specified model and try again.

        Args:
            model: Model alias or literal (e.g., "reasoning", "openai:gpt-4o")
            retries: Number of retries for the escalated model (default: 1)

        Returns:
            EscalateStrategy instance

        Raises:
            ValueError: If model is empty or retries is negative.

        Example:
            @spell(model="fast", on_fail=OnFail.escalate("reasoning"))
            def complex_task(query: str) -> Analysis:
                '''Analyze the query and return structured insights.'''
                ...
        """
        # Input validation (issue #176)
        if not model or not model.strip():
            raise ValueError("model cannot be empty")
        if retries < 0:
            raise ValueError(f"retries must be non-negative, got {retries}")
        return EscalateStrategy(model=model.strip(), retries=retries)

    @staticmethod
    def fallback(default: T) -> FallbackStrategy[T]:
        """Return a default value instead of raising on validation failure.

        Use this for optional enrichment or when a sensible default exists.

        Args:
            default: The default value to return on failure

        Returns:
            FallbackStrategy instance

        Example:
            @spell(on_fail=OnFail.fallback(default=EmptyAnalysis()))
            def optional_enrichment(data: str) -> Analysis:
                '''Optionally enrich the data with analysis.'''
                ...
        """
        return FallbackStrategy(default=default)

    @staticmethod
    def custom(
        handler: Callable[[Exception, int, dict[str, Any]], T]
    ) -> CustomStrategy[T]:
        """Use a custom handler for domain-specific fixes.

        The handler receives the error, attempt number, and context.
        It can return a fixed value or re-raise to propagate the error.

        Args:
            handler: Callable(error, attempt, context) -> output or raise

        Returns:
            CustomStrategy instance

        Example:
            def fix_dates(error: Exception, attempt: int, ctx: dict) -> Dates:
                if "date format" in str(error):
                    # Extract and fix dates manually
                    return parse_dates_manually(ctx["input_args"]["text"])
                raise error

            @spell(on_fail=OnFail.custom(fix_dates))
            def extract_dates(text: str) -> Dates:
                '''Extract dates from text.'''
                ...
        """
        return CustomStrategy(handler=handler)


__all__ = [
    "OnFail",
    "OnFailStrategy",
    "ValidatorOnFailStrategy",
    "RaiseStrategy",
    "FixStrategy",
    "RetryStrategy",
    "EscalateStrategy",
    "FallbackStrategy",
    "CustomStrategy",
]
