"""Adapter layer for pydantic_ai internals.

This module isolates all imports from pydantic_ai's internal modules (exceptions,
settings, agent internals) to a single location. If pydantic_ai changes these
internals, only this file needs to be updated.

The stable public API from pydantic_ai (Agent) is imported directly here as well
to provide a single source for all pydantic_ai dependencies.
"""

from __future__ import annotations

# Import Agent - this is stable public API
from pydantic_ai import Agent


class ValidationError(Exception):
    """Raised when LLM output fails Pydantic validation after retries.

    This is the user-facing exception that wraps pydantic_ai's internal
    UnexpectedModelBehavior exception. Users should catch this instead of
    importing from pydantic_ai directly.

    Example:
        from magically import spell, ValidationError

        @spell
        def my_spell(text: str) -> MyModel:
            '''Process text.'''
            ...

        try:
            result = my_spell("input")
        except ValidationError as e:
            print(f"LLM output failed validation: {e}")

    Attributes:
        message: Human-readable error message
        original_error: The underlying pydantic_ai exception (if available)
    """

    def __init__(self, message: str, original_error: Exception | None = None):
        super().__init__(message)
        self.message = message
        self.original_error = original_error

    def __str__(self) -> str:
        return self.message

# Import internal modules with fallback handling
try:
    from pydantic_ai.exceptions import UnexpectedModelBehavior
except ImportError:
    # Fallback if pydantic_ai renames/moves this exception
    try:
        from pydantic_ai.errors import UnexpectedModelBehavior  # type: ignore[no-redef]
    except ImportError:
        # Ultimate fallback - create our own exception class
        class UnexpectedModelBehavior(Exception):  # type: ignore[no-redef]
            """Fallback exception when pydantic_ai's exception is unavailable."""
            pass

try:
    from pydantic_ai.settings import ModelSettings
except ImportError:
    # Fallback if pydantic_ai renames/moves ModelSettings
    try:
        from pydantic_ai.models import ModelSettings  # type: ignore[no-redef]
    except ImportError:
        # Ultimate fallback - use dict-like TypedDict
        from typing import TypedDict

        class ModelSettings(TypedDict, total=False):  # type: ignore[no-redef]
            """Fallback ModelSettings when pydantic_ai's is unavailable."""
            temperature: float
            max_tokens: int
            top_p: float
            timeout: float

try:
    from pydantic_ai.agent import EndStrategy
except ImportError:
    # Fallback for EndStrategy
    from typing import Literal
    EndStrategy = Literal["early", "exhaustive"]  # type: ignore[misc]


__all__ = [
    "Agent",
    "UnexpectedModelBehavior",
    "ModelSettings",
    "EndStrategy",
    "ValidationError",
]
