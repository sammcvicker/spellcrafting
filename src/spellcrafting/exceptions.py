"""Exception hierarchy for spellcrafting.

All spellcrafting exceptions inherit from SpellcraftingError, allowing users to catch
all library errors with a single except clause:

    from spellcrafting import SpellcraftingError

    try:
        result = my_spell(text)
    except SpellcraftingError as e:
        # Handle any spellcrafting error
        ...

For more specific handling, catch the individual exception types:

    from spellcrafting import GuardError, ValidationError, SpellcraftingConfigError

    try:
        result = my_spell(text)
    except GuardError:
        # Guard validation failed
        ...
    except ValidationError:
        # LLM output failed Pydantic validation
        ...
    except SpellcraftingConfigError:
        # Configuration error (missing alias, invalid config, etc.)
        ...

Naming Convention (issue #171, #177):
------------------------------------
Exception classes use the *Error suffix consistently:
- SpellcraftingError: Base class (prefix indicates library origin)
- SpellcraftingConfigError: Prefixed for clarity about library-specific config errors
- GuardError: No prefix needed (guard is a spellcrafting-specific concept)
- ValidationError: No prefix (matches common Python convention)

All exceptions inherit from SpellcraftingError for catch-all handling.
"""

from __future__ import annotations


class SpellcraftingError(Exception):
    """Base exception for all spellcrafting errors.

    Catch this to handle any error from the spellcrafting library.
    """

    pass


class SpellcraftingConfigError(SpellcraftingError):
    """Raised when configuration is invalid or missing.

    Examples:
        - Unknown model alias
        - Invalid pyproject.toml schema
        - Missing required config fields
    """

    pass


class GuardError(SpellcraftingError):
    """Raised when a guard validation fails.

    Input guards raise this when they reject input before the LLM call.
    Output guards raise this when they reject output after the LLM call.
    """

    pass


class ValidationError(SpellcraftingError):
    """Raised when LLM output fails Pydantic validation after retries.

    This wraps pydantic_ai's internal UnexpectedModelBehavior exception.
    Users should catch this instead of importing from pydantic_ai directly.

    Example:
        from spellcrafting import spell, ValidationError

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
    "SpellcraftingError",
    "SpellcraftingConfigError",
    "GuardError",
    "ValidationError",
]
