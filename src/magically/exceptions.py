"""Exception hierarchy for magically.

All magically exceptions inherit from MagicallyError, allowing users to catch
all library errors with a single except clause:

    from magically import MagicallyError

    try:
        result = my_spell(text)
    except MagicallyError as e:
        # Handle any magically error
        ...

For more specific handling, catch the individual exception types:

    from magically import GuardError, ValidationError, MagicallyConfigError

    try:
        result = my_spell(text)
    except GuardError:
        # Guard validation failed
        ...
    except ValidationError:
        # LLM output failed Pydantic validation
        ...
    except MagicallyConfigError:
        # Configuration error (missing alias, invalid config, etc.)
        ...
"""

from __future__ import annotations


class MagicallyError(Exception):
    """Base exception for all magically errors.

    Catch this to handle any error from the magically library.
    """

    pass


class MagicallyConfigError(MagicallyError):
    """Raised when configuration is invalid or missing.

    Examples:
        - Unknown model alias
        - Invalid pyproject.toml schema
        - Missing required config fields
    """

    pass


class GuardError(MagicallyError):
    """Raised when a guard validation fails.

    Input guards raise this when they reject input before the LLM call.
    Output guards raise this when they reject output after the LLM call.
    """

    pass


class ValidationError(MagicallyError):
    """Raised when LLM output fails Pydantic validation after retries.

    This wraps pydantic_ai's internal UnexpectedModelBehavior exception.
    Users should catch this instead of importing from pydantic_ai directly.

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


__all__ = [
    "MagicallyError",
    "MagicallyConfigError",
    "GuardError",
    "ValidationError",
]
