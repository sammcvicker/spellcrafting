"""LLM-powered validators for Pydantic field validation.

The llm_validator function creates Pydantic validators powered by LLM calls,
allowing semantic validation rules expressed in natural language.

Example:
    from magically import spell, llm_validator
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
"""

from __future__ import annotations

from typing import Callable, Literal

from pydantic import BaseModel

from magically.spell import spell


class ValidationResult(BaseModel):
    """Result from LLM validation check."""

    valid: bool
    reason: str | None = None
    fixed_value: str | None = None


def llm_validator(
    rule: str,
    *,
    model: str = "fast",
    on_fail: Literal["raise", "fix"] = "raise",
) -> Callable[[str], str]:
    """Create a Pydantic validator from a natural language rule.

    This function creates a validator that uses an LLM to check if values
    satisfy a semantic rule. It's designed to be used with Pydantic's
    BeforeValidator for field validation.

    Args:
        rule: Natural language description of the validation rule.
              Example: "Content must be appropriate for all ages"
        model: Model alias to use for validation (default: "fast").
               Use cheap/fast models to minimize latency and cost.
        on_fail: Action when validation fails:
                 - "raise": Raise ValueError with the reason (default)
                 - "fix": Attempt to fix the value to satisfy the rule

    Returns:
        A validator function that can be used with Pydantic's BeforeValidator.

    Example:
        from pydantic import BaseModel, BeforeValidator
        from typing import Annotated

        professional = llm_validator(
            "Must be professional and appropriate for business communication",
            model="fast"
        )

        class Email(BaseModel):
            body: Annotated[str, BeforeValidator(professional)]

    Note:
        LLM validators add latency and cost to validation. Use them for
        semantic rules that are hard to express in code, and prefer
        fast/cheap models for validation.
    """
    # Build the system prompt based on on_fail strategy
    if on_fail == "fix":
        system_prompt = f"""Check if the given value satisfies this rule: {rule}

If the value is valid, return valid=True.
If the value is invalid:
- Return valid=False with a brief reason
- Provide a fixed_value that satisfies the rule while preserving the original intent"""
    else:
        system_prompt = f"""Check if the given value satisfies this rule: {rule}

If the value is valid, return valid=True.
If the value is invalid, return valid=False with a brief reason explaining why."""

    @spell(model=model)
    def validate(value: str) -> ValidationResult:
        # Docstring becomes system prompt - we set it dynamically
        ...

    # Override the system prompt
    validate._system_prompt = system_prompt

    # We need to recreate the agent with the new system prompt
    # Clear cached agent to force recreation with new prompt
    from magically.spell import _agent_cache

    spell_id = validate._spell_id
    # Remove any cached agents for this spell
    _agent_cache.remove_by_spell_id(spell_id)

    def validator(value: str) -> str:
        """Pydantic validator function."""
        result = validate(value)

        if result.valid:
            return value

        if on_fail == "fix" and result.fixed_value is not None:
            return result.fixed_value

        raise ValueError(f"Validation failed: {result.reason}")

    return validator


__all__ = ["llm_validator", "ValidationResult"]
