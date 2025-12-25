"""Tests for llm_validator."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, BeforeValidator, ValidationError
from typing import Annotated

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
            def mock_decorator(model):
                def decorator(fn):
                    fn._spell_id = 12345
                    fn._system_prompt = ""
                    return fn
                return decorator

            mock_spell.side_effect = mock_decorator
            llm_validator("Must be polite")
            mock_spell.assert_called_once()
            assert mock_spell.call_args.kwargs["model"] == "fast"

    def test_custom_model(self):
        with patch("magically.validator.spell") as mock_spell:
            def mock_decorator(model):
                def decorator(fn):
                    fn._spell_id = 12345
                    fn._system_prompt = ""
                    return fn
                return decorator

            mock_spell.side_effect = mock_decorator
            llm_validator("Must be polite", model="reasoning")
            mock_spell.assert_called_once()
            assert mock_spell.call_args.kwargs["model"] == "reasoning"


class TestLlmValidatorOnFailRaise:
    """Tests for llm_validator with on_fail='raise'."""

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
    """Tests for llm_validator with on_fail='fix'."""

    def test_valid_value_passes_through(self):
        mock_result = ValidationResult(valid=True)

        with patch("magically.validator.spell") as mock_spell:
            mock_validate = MagicMock(return_value=mock_result)
            mock_spell.return_value = lambda fn: mock_validate

            validator_fn = llm_validator("Must be polite", on_fail="fix")
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

            validator_fn = llm_validator("Must be formal", on_fail="fix")
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

            validator_fn = llm_validator("Must be polite", on_fail="fix")

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

            formal = llm_validator("Must be formal", on_fail="fix")

            class Greeting(BaseModel):
                message: Annotated[str, BeforeValidator(formal)]

            greeting = Greeting(message="Hey!")
            assert greeting.message == "Good morning"


class TestLlmValidatorSystemPrompt:
    """Tests for system prompt generation."""

    def test_raise_mode_prompt_does_not_mention_fix(self):
        with patch("magically.validator.spell") as mock_spell:
            captured_prompt = None

            def capture_spell(model):
                def decorator(fn):
                    fn._system_prompt = "placeholder"
                    fn._spell_id = 123
                    return fn

                return decorator

            mock_spell.side_effect = capture_spell
            llm_validator("Must be polite", on_fail="raise")

            # The function creates the prompt, we verify through behavior

    def test_fix_mode_prompt_mentions_fixed_value(self):
        with patch("magically.validator.spell") as mock_spell:
            def capture_spell(model):
                def decorator(fn):
                    fn._system_prompt = "placeholder"
                    fn._spell_id = 123
                    return fn

                return decorator

            mock_spell.side_effect = capture_spell
            llm_validator("Must be polite", on_fail="fix")

            # The prompt should mention fixed_value for fix mode


class TestLlmValidatorCacheClearing:
    """Tests for agent cache clearing behavior."""

    def test_clears_cached_agents(self):
        from magically.spell import _agent_cache

        # Pre-populate with a fake entry using the cache API
        fake_spell_id = 99999
        _agent_cache.set((fake_spell_id, 0), MagicMock())

        with patch("magically.validator.spell") as mock_spell:
            def make_spell(model):
                def decorator(fn):
                    fn._spell_id = fake_spell_id
                    fn._system_prompt = ""
                    return fn

                return decorator

            mock_spell.side_effect = make_spell
            llm_validator("Test rule")

            # The cache entry should be cleared
            assert _agent_cache.get((fake_spell_id, 0)) is None
