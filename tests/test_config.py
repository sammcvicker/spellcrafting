"""Tests for configuration system."""

import pytest
from pydantic import ValidationError

import magically.config as config_module
from magically.config import Config, MagicallyConfigError, ModelConfig, current_config


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


@pytest.fixture(autouse=True)
def reset_process_default():
    """Reset process default between tests."""
    yield
    config_module._process_default = None


class TestConfig:
    """Tests for Config container."""

    def test_empty_config(self):
        config = Config()
        assert config.models == {}

    def test_config_with_model_configs(self):
        config = Config(models={
            "fast": ModelConfig(model="anthropic:claude-haiku"),
        })
        assert "fast" in config.models
        assert config.models["fast"].model == "anthropic:claude-haiku"

    def test_config_with_dicts(self):
        config = Config(models={
            "fast": {"model": "anthropic:claude-haiku", "temperature": 0.2},
        })
        assert config.models["fast"].model == "anthropic:claude-haiku"
        assert config.models["fast"].temperature == 0.2

    def test_models_returns_copy(self):
        config = Config(models={
            "fast": ModelConfig(model="test"),
        })
        models = config.models
        models["new"] = ModelConfig(model="hacked")
        assert "new" not in config.models


class TestConfigResolve:
    """Tests for Config.resolve()."""

    def test_resolve_existing_alias(self):
        config = Config(models={
            "fast": ModelConfig(model="anthropic:claude-haiku"),
        })
        resolved = config.resolve("fast")
        assert resolved.model == "anthropic:claude-haiku"

    def test_resolve_missing_alias_raises(self):
        config = Config(models={
            "fast": ModelConfig(model="test"),
        })
        with pytest.raises(MagicallyConfigError, match="Unknown model alias 'missing'"):
            config.resolve("missing")

    def test_resolve_error_shows_available(self):
        config = Config(models={
            "fast": ModelConfig(model="test"),
            "reasoning": ModelConfig(model="test2"),
        })
        with pytest.raises(MagicallyConfigError, match="fast"):
            config.resolve("typo")


class TestConfigContextManager:
    """Tests for Config as context manager."""

    def test_context_manager_sets_current(self):
        config = Config(models={
            "fast": ModelConfig(model="test"),
        })

        assert "fast" not in Config.current().models

        with config:
            assert Config.current() is config
            assert "fast" in Config.current().models

        assert "fast" not in Config.current().models

    def test_nested_contexts(self):
        outer = Config(models={"outer": ModelConfig(model="outer-model")})
        inner = Config(models={"inner": ModelConfig(model="inner-model")})

        with outer:
            assert "outer" in Config.current().models

            with inner:
                assert "inner" in Config.current().models
                assert "outer" not in Config.current().models

            assert "outer" in Config.current().models

    def test_context_returns_self(self):
        config = Config()
        with config as ctx:
            assert ctx is config


class TestConfigProcessDefault:
    """Tests for Config.set_as_default()."""

    def test_set_as_default(self):
        config = Config(models={
            "default-alias": ModelConfig(model="test"),
        })
        config.set_as_default()

        assert "default-alias" in Config.current().models

    def test_context_overrides_default(self):
        default = Config(models={"default": ModelConfig(model="default-model")})
        default.set_as_default()

        override = Config(models={"override": ModelConfig(model="override-model")})

        assert "default" in Config.current().models

        with override:
            assert "override" in Config.current().models
            assert "default" not in Config.current().models

        assert "default" in Config.current().models


class TestCurrentConfig:
    """Tests for current_config() convenience function."""

    def test_current_config_returns_current(self):
        config = Config(models={"test": ModelConfig(model="test")})

        with config:
            assert current_config() is Config.current()
            assert "test" in current_config().models
