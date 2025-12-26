"""LLM-powered validators for Pydantic field validation.

The llm_validator function creates Pydantic validators powered by LLM calls,
allowing semantic validation rules expressed in natural language.

Design Note: Pydantic vs Dataclass Usage
----------------------------------------
ValidationResult uses Pydantic BaseModel because:
- It is the return type from an LLM call, parsed from structured output
- Pydantic handles the JSON parsing and validation from LLM responses
- This is consistent with how @spell uses Pydantic for return type parsing

Example:
    from spellcrafting import spell, llm_validator, OnFail
    from pydantic import BaseModel, BeforeValidator
    from typing import Annotated

    # Create a validator from a natural language rule
    family_friendly = llm_validator(
        "Content must be appropriate for all ages with no profanity",
        model="fast"  # Use cheap model for validation
    )

    class Response(BaseModel):
        content: Annotated[str, BeforeValidator(family_friendly)]

    @spell(model="reasoning")
    def generate_story(topic: str) -> Response:
        '''Write a short story about the topic.'''
        ...

    # Use FIX strategy to auto-correct values
    professional = llm_validator(
        "Must be professional business communication",
        model="fast",
        on_fail=OnFail.FIX  # Attempt to fix invalid values
    )
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from pydantic import BaseModel

from spellcrafting.on_fail import OnFail, RaiseStrategy, FixStrategy, ValidatorOnFailStrategy
from spellcrafting.spell import spell

T = TypeVar("T")


class ValidationResult(BaseModel):
    """Result from LLM validation check."""

    valid: bool
    reason: str | None = None
    fixed_value: str | None = None


def llm_validator(
    rule: str,
    *,
    model: str = "fast",
    on_fail: ValidatorOnFailStrategy = OnFail.RAISE,
) -> Callable[[Any], Any]:
    """Create a Pydantic validator from a natural language rule.

    This function creates a validator that uses an LLM to check if values
    satisfy a semantic rule. It's designed to be used with Pydantic's
    BeforeValidator for field validation.

    The validator accepts any input type - non-string values are converted
    to their string representation for LLM validation. On success, the
    original value is returned unchanged. With on_fail=OnFail.FIX, the fixed
    value is returned as a string.

    Args:
        rule: Natural language description of the validation rule.
              Example: "Content must be appropriate for all ages"
        model: Model alias to use for validation (default: "fast").
               Use cheap/fast models to minimize latency and cost.
        on_fail: Strategy when validation fails:
                 - OnFail.RAISE: Raise ValueError with the reason (default)
                 - OnFail.FIX: Attempt to fix the value to satisfy the rule

    Returns:
        A validator function that can be used with Pydantic's BeforeValidator.
        The validator accepts any type and returns the original value on
        success (or the fixed string value with on_fail=OnFail.FIX).

    Raises:
        ValueError: If rule is empty or model is empty.

    Example:
        from pydantic import BaseModel, BeforeValidator
        from typing import Annotated
        from spellcrafting import llm_validator, OnFail

        professional = llm_validator(
            "Must be professional and appropriate for business communication",
            model="fast"
        )

        class Email(BaseModel):
            body: Annotated[str, BeforeValidator(professional)]

        # Use FIX strategy to auto-correct values
        family_friendly = llm_validator(
            "Content must be appropriate for all ages",
            on_fail=OnFail.FIX
        )

        class Content(BaseModel):
            text: Annotated[str, BeforeValidator(family_friendly)]

    Note:
        LLM validators add latency and cost to validation. Use them for
        semantic rules that are hard to express in code, and prefer
        fast/cheap models for validation.
    """
    # Input validation (issue #176)
    if not rule or not rule.strip():
        raise ValueError("rule cannot be empty")
    if not model or not model.strip():
        raise ValueError("model cannot be empty")

    # Determine if we should use fix mode
    fix_mode = isinstance(on_fail, FixStrategy)

    # Build the system prompt based on on_fail strategy
    if fix_mode:
        validation_system_prompt = f"""Check if the given value satisfies this rule: {rule}

If the value is valid, return valid=True.
If the value is invalid:
- Return valid=False with a brief reason
- Provide a fixed_value that satisfies the rule while preserving the original intent"""
    else:
        validation_system_prompt = f"""Check if the given value satisfies this rule: {rule}

If the value is valid, return valid=True.
If the value is invalid, return valid=False with a brief reason explaining why."""

    # Use the system_prompt parameter instead of mutating internals
    @spell(model=model, system_prompt=validation_system_prompt)
    def validate(value: str) -> ValidationResult:
        """Validate value against the rule."""
        ...

    def validator(value: Any) -> Any:
        """Pydantic validator function.

        Accepts any type, converts to string for LLM validation,
        and returns the original value on success.
        """
        # Convert to string representation for LLM validation
        str_value = value if isinstance(value, str) else str(value)
        result = validate(str_value)

        if result.valid:
            return value

        if fix_mode and result.fixed_value is not None:
            return result.fixed_value

        raise ValueError(f"Validation failed: {result.reason}")

    return validator


__all__ = ["llm_validator", "ValidationResult"]
