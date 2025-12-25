"""Tests for llm_validator."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, BeforeValidator, ValidationError
from typing import Annotated

from magically.on_fail import OnFail
from magically.validator import llm_validator, ValidationResult


class TestValidationResult:
    """Tests for ValidationResult model."""

    def test_valid_result(self):
        result = ValidationResult(valid=True)
        assert result.valid is True
        assert result.reason is None
        assert result.fixed_value is None

    def test_invalid_result_with_reason(self):
        result = ValidationResult(valid=False, reason="Contains profanity")
        assert result.valid is False
        assert result.reason == "Contains profanity"

    def test_invalid_result_with_fix(self):
        result = ValidationResult(
            valid=False,
            reason="Contains profanity",
            fixed_value="Clean content here",
        )
        assert result.valid is False
        assert result.fixed_value == "Clean content here"


class TestLlmValidatorBasic:
    """Tests for llm_validator basic functionality."""

    def test_creates_callable(self):
        validator = llm_validator("Must be polite")
        assert callable(validator)

    def test_default_model_is_fast(self):
        with patch("magically.validator.spell") as mock_spell:
            def mock_decorator(model, system_prompt):
                def decorator(fn):
                    fn._spell_id = 12345
                    fn._system_prompt = system_prompt
                    return fn
                return decorator

            mock_spell.side_effect = mock_decorator
            llm_validator("Must be polite")
            mock_spell.assert_called_once()
            assert mock_spell.call_args.kwargs["model"] == "fast"
            assert "system_prompt" in mock_spell.call_args.kwargs

    def test_custom_model(self):
        with patch("magically.validator.spell") as mock_spell:
            def mock_decorator(model, system_prompt):
                def decorator(fn):
                    fn._spell_id = 12345
                    fn._system_prompt = system_prompt
                    return fn
                return decorator

            mock_spell.side_effect = mock_decorator
            llm_validator("Must be polite", model="reasoning")
            mock_spell.assert_called_once()
            assert mock_spell.call_args.kwargs["model"] == "reasoning"


class TestLlmValidatorOnFailRaise:
    """Tests for llm_validator with on_fail=OnFail.RAISE (default)."""

    def test_valid_value_passes_through(self):
        validator = llm_validator("Must be polite")

        # Mock the internal validate spell
        mock_result = ValidationResult(valid=True)
        with patch.object(
            validator, "__wrapped__", create=True
        ):
            # Patch at module level since validator creates internal spell
            with patch("magically.validator.spell") as mock_spell:
                # Create a mock that returns our validation result
                mock_validate = MagicMock(return_value=mock_result)
                mock_spell.return_value = lambda fn: mock_validate

                validator_fn = llm_validator("Must be polite")
                result = validator_fn("Hello, how are you?")

                assert result == "Hello, how are you?"

    def test_invalid_value_raises(self):
        mock_result = ValidationResult(valid=False, reason="Contains profanity")

        with patch("magically.validator.spell") as mock_spell:
            mock_validate = MagicMock(return_value=mock_result)
            mock_spell.return_value = lambda fn: mock_validate

            validator_fn = llm_validator("Must be polite")

            with pytest.raises(ValueError, match="Validation failed: Contains profanity"):
                validator_fn("Bad content here")


class TestLlmValidatorOnFailFix:
    """Tests for llm_validator with on_fail=OnFail.FIX."""

    def test_valid_value_passes_through(self):
        mock_result = ValidationResult(valid=True)

        with patch("magically.validator.spell") as mock_spell:
            mock_validate = MagicMock(return_value=mock_result)
            mock_spell.return_value = lambda fn: mock_validate

            validator_fn = llm_validator("Must be polite", on_fail=OnFail.FIX)
            result = validator_fn("Hello!")

            assert result == "Hello!"

    def test_invalid_value_with_fix_returns_fixed(self):
        mock_result = ValidationResult(
            valid=False,
            reason="Contains informal language",
            fixed_value="Good day, how may I assist you?",
        )

        with patch("magically.validator.spell") as mock_spell:
            mock_validate = MagicMock(return_value=mock_result)
            mock_spell.return_value = lambda fn: mock_validate

            validator_fn = llm_validator("Must be formal", on_fail=OnFail.FIX)
            result = validator_fn("Hey, what's up?")

            assert result == "Good day, how may I assist you?"

    def test_invalid_value_without_fix_raises(self):
        # LLM couldn't fix the value
        mock_result = ValidationResult(
            valid=False,
            reason="Cannot be fixed",
            fixed_value=None,
        )

        with patch("magically.validator.spell") as mock_spell:
            mock_validate = MagicMock(return_value=mock_result)
            mock_spell.return_value = lambda fn: mock_validate

            validator_fn = llm_validator("Must be polite", on_fail=OnFail.FIX)

            with pytest.raises(ValueError, match="Validation failed"):
                validator_fn("Unfixable content")


class TestLlmValidatorWithPydantic:
    """Tests for llm_validator integration with Pydantic models."""

    def test_as_before_validator_valid(self):
        mock_result = ValidationResult(valid=True)

        with patch("magically.validator.spell") as mock_spell:
            mock_validate = MagicMock(return_value=mock_result)
            mock_spell.return_value = lambda fn: mock_validate

            family_friendly = llm_validator("Must be family friendly")

            class Content(BaseModel):
                text: Annotated[str, BeforeValidator(family_friendly)]

            content = Content(text="Hello, world!")
            assert content.text == "Hello, world!"

    def test_as_before_validator_invalid(self):
        mock_result = ValidationResult(valid=False, reason="Contains adult content")

        with patch("magically.validator.spell") as mock_spell:
            mock_validate = MagicMock(return_value=mock_result)
            mock_spell.return_value = lambda fn: mock_validate

            family_friendly = llm_validator("Must be family friendly")

            class Content(BaseModel):
                text: Annotated[str, BeforeValidator(family_friendly)]

            with pytest.raises(ValidationError) as exc_info:
                Content(text="Inappropriate content")

            assert "Validation failed" in str(exc_info.value)

    def test_as_before_validator_with_fix(self):
        mock_result = ValidationResult(
            valid=False,
            reason="Contains informal greeting",
            fixed_value="Good morning",
        )

        with patch("magically.validator.spell") as mock_spell:
            mock_validate = MagicMock(return_value=mock_result)
            mock_spell.return_value = lambda fn: mock_validate

            formal = llm_validator("Must be formal", on_fail=OnFail.FIX)

            class Greeting(BaseModel):
                message: Annotated[str, BeforeValidator(formal)]

            greeting = Greeting(message="Hey!")
            assert greeting.message == "Good morning"


class TestLlmValidatorSystemPrompt:
    """Tests for system prompt generation."""

    def test_raise_mode_prompt_does_not_mention_fix(self):
        with patch("magically.validator.spell") as mock_spell:
            captured_prompt = None

            def capture_spell(model, system_prompt):
                nonlocal captured_prompt
                captured_prompt = system_prompt
                def decorator(fn):
                    fn._system_prompt = system_prompt
                    fn._spell_id = 123
                    return fn

                return decorator

            mock_spell.side_effect = capture_spell
            llm_validator("Must be polite", on_fail=OnFail.RAISE)

            # RAISE mode should not mention fixed_value
            assert "fixed_value" not in captured_prompt

    def test_fix_mode_prompt_mentions_fixed_value(self):
        with patch("magically.validator.spell") as mock_spell:
            captured_prompt = None

            def capture_spell(model, system_prompt):
                nonlocal captured_prompt
                captured_prompt = system_prompt
                def decorator(fn):
                    fn._system_prompt = system_prompt
                    fn._spell_id = 123
                    return fn

                return decorator

            mock_spell.side_effect = capture_spell
            llm_validator("Must be polite", on_fail=OnFail.FIX)

            # FIX mode should mention fixed_value
            assert "fixed_value" in captured_prompt


class TestLlmValidatorNonStringTypes:
    """Tests for llm_validator with non-string input types (#115)."""

    def test_dict_value_passes_through(self):
        """Validator should accept dict input and return dict on success."""
        mock_result = ValidationResult(valid=True)

        with patch("magically.validator.spell") as mock_spell:
            mock_validate = MagicMock(return_value=mock_result)
            mock_spell.return_value = lambda fn: mock_validate

            validator_fn = llm_validator("Must have 'name' key")
            input_dict = {"name": "test", "value": 123}
            result = validator_fn(input_dict)

            assert result == input_dict
            assert isinstance(result, dict)

    def test_list_value_passes_through(self):
        """Validator should accept list input and return list on success."""
        mock_result = ValidationResult(valid=True)

        with patch("magically.validator.spell") as mock_spell:
            mock_validate = MagicMock(return_value=mock_result)
            mock_spell.return_value = lambda fn: mock_validate

            validator_fn = llm_validator("Must have at least 2 items")
            input_list = [1, 2, 3]
            result = validator_fn(input_list)

            assert result == input_list
            assert isinstance(result, list)

    def test_int_value_passes_through(self):
        """Validator should accept int input and return int on success."""
        mock_result = ValidationResult(valid=True)

        with patch("magically.validator.spell") as mock_spell:
            mock_validate = MagicMock(return_value=mock_result)
            mock_spell.return_value = lambda fn: mock_validate

            validator_fn = llm_validator("Must be positive")
            result = validator_fn(42)

            assert result == 42
            assert isinstance(result, int)

    def test_dict_converts_to_string_for_validation(self):
        """Non-string values should be converted to string for LLM validation."""
        mock_result = ValidationResult(valid=True)
        captured_calls = []

        with patch("magically.validator.spell") as mock_spell:
            def make_mock_validate(value):
                captured_calls.append(value)
                return mock_result

            mock_validate = MagicMock(side_effect=make_mock_validate)
            mock_spell.return_value = lambda fn: mock_validate

            validator_fn = llm_validator("Must have 'name' key")
            input_dict = {"name": "test"}
            validator_fn(input_dict)

        # The LLM should receive the string representation
        assert len(captured_calls) == 1
        assert captured_calls[0] == str(input_dict)

    def test_invalid_dict_raises(self):
        """Invalid non-string value should raise ValueError."""
        mock_result = ValidationResult(valid=False, reason="Missing 'name' key")

        with patch("magically.validator.spell") as mock_spell:
            mock_validate = MagicMock(return_value=mock_result)
            mock_spell.return_value = lambda fn: mock_validate

            validator_fn = llm_validator("Must have 'name' key")

            with pytest.raises(ValueError, match="Validation failed: Missing 'name' key"):
                validator_fn({"value": 123})

    def test_invalid_dict_with_fix_returns_string(self):
        """With on_fail=OnFail.FIX, fixed value is returned as string."""
        mock_result = ValidationResult(
            valid=False,
            reason="Missing 'name' key",
            fixed_value="{'name': 'unknown', 'value': 123}",
        )

        with patch("magically.validator.spell") as mock_spell:
            mock_validate = MagicMock(return_value=mock_result)
            mock_spell.return_value = lambda fn: mock_validate

            validator_fn = llm_validator("Must have 'name' key", on_fail=OnFail.FIX)
            result = validator_fn({"value": 123})

            # Fixed value is returned as string from LLM
            assert result == "{'name': 'unknown', 'value': 123}"
            assert isinstance(result, str)

    def test_pydantic_model_value(self):
        """Validator should work with Pydantic model instances."""
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        mock_result = ValidationResult(valid=True)

        with patch("magically.validator.spell") as mock_spell:
            mock_validate = MagicMock(return_value=mock_result)
            mock_spell.return_value = lambda fn: mock_validate

            validator_fn = llm_validator("Must have valid person data")
            person = Person(name="Alice", age=30)
            result = validator_fn(person)

            assert result == person
            assert isinstance(result, Person)

    def test_none_value(self):
        """Validator should handle None values."""
        mock_result = ValidationResult(valid=True)

        with patch("magically.validator.spell") as mock_spell:
            mock_validate = MagicMock(return_value=mock_result)
            mock_spell.return_value = lambda fn: mock_validate

            validator_fn = llm_validator("Optional field")
            result = validator_fn(None)

            assert result is None


class TestLlmValidatorSystemPromptParameter:
    """Tests for system_prompt parameter usage (no longer mutates internals)."""

    def test_system_prompt_passed_to_spell(self):
        """llm_validator should use system_prompt parameter instead of mutating internals."""
        with patch("magically.validator.spell") as mock_spell:
            captured_prompt = None

            def capture_spell(model, system_prompt):
                nonlocal captured_prompt
                captured_prompt = system_prompt
                def decorator(fn):
                    fn._spell_id = 12345
                    fn._system_prompt = system_prompt
                    return fn
                return decorator

            mock_spell.side_effect = capture_spell

            llm_validator("Must be polite")

            # system_prompt should be passed directly to @spell
            assert captured_prompt is not None
            assert "Must be polite" in captured_prompt

    def test_no_cache_mutation_needed(self):
        """llm_validator should not need to clear cache since system_prompt is passed directly."""
        from magically.spell import _agent_cache

        # Pre-populate cache
        fake_spell_id = 99999
        original_agent = MagicMock()
        _agent_cache.set((fake_spell_id, 0), original_agent)

        with patch("magically.validator.spell") as mock_spell:
            def make_spell(model, system_prompt):
                def decorator(fn):
                    fn._spell_id = 88888  # Different ID - won't affect cache
                    fn._system_prompt = system_prompt
                    return fn
                return decorator

            mock_spell.side_effect = make_spell

            llm_validator("Test rule")

            # Original cache entry should remain unchanged
            assert _agent_cache.get((fake_spell_id, 0)) is original_agent
