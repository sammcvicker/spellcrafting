"""Tests for configuration system."""

import pytest
from pydantic import ValidationError

from magically.config import MagicallyConfigError, ModelConfig


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_minimal_config(self):
        config = ModelConfig(model="anthropic:claude-sonnet")
        assert config.model == "anthropic:claude-sonnet"
        assert config.temperature is None
        assert config.max_tokens is None

    def test_full_config(self):
        config = ModelConfig(
            model="anthropic:claude-opus-4",
            temperature=0.7,
            max_tokens=8192,
            top_p=0.95,
            timeout=30.0,
            retries=3,
            extra={"custom": "value"},
        )
        assert config.model == "anthropic:claude-opus-4"
        assert config.temperature == 0.7
        assert config.max_tokens == 8192
        assert config.top_p == 0.95
        assert config.timeout == 30.0
        assert config.retries == 3
        assert config.extra == {"custom": "value"}

    def test_from_dict(self):
        config = ModelConfig.model_validate({
            "model": "openai:gpt-4o",
            "temperature": 0.5,
        })
        assert config.model == "openai:gpt-4o"
        assert config.temperature == 0.5

    def test_model_required(self):
        with pytest.raises(ValidationError):
            ModelConfig()

    def test_ignores_unknown_fields(self):
        config = ModelConfig.model_validate({
            "model": "anthropic:claude-sonnet",
            "unknown_field": "ignored",
        })
        assert config.model == "anthropic:claude-sonnet"
        assert not hasattr(config, "unknown_field")

    def test_hashable(self):
        config1 = ModelConfig(model="anthropic:claude-sonnet", temperature=0.7)
        config2 = ModelConfig(model="anthropic:claude-sonnet", temperature=0.7)
        config3 = ModelConfig(model="anthropic:claude-sonnet", temperature=0.5)

        assert hash(config1) == hash(config2)
        assert hash(config1) != hash(config3)

    def test_hashable_with_extra(self):
        config1 = ModelConfig(model="test", extra={"a": 1})
        config2 = ModelConfig(model="test", extra={"a": 1})
        config3 = ModelConfig(model="test", extra={"a": 2})

        assert hash(config1) == hash(config2)
        assert hash(config1) != hash(config3)


class TestMagicallyConfigError:
    """Tests for MagicallyConfigError."""

    def test_is_exception(self):
        assert issubclass(MagicallyConfigError, Exception)

    def test_can_raise_with_message(self):
        with pytest.raises(MagicallyConfigError, match="test message"):
            raise MagicallyConfigError("test message")

    def test_can_include_available_aliases(self):
        available = ["fast", "reasoning", "default"]
        error = MagicallyConfigError(
            f"Unknown model alias 'typo'. Available: {available}"
        )
        assert "typo" in str(error)
        assert "fast" in str(error)
