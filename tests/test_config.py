"""Tests for configuration system."""

import pytest
from pydantic import ValidationError

import magically.config as config_module
from magically.config import Config, MagicallyConfigError, ModelConfig, current_config


@pytest.fixture(autouse=True)
def reset_file_config_cache():
    """Reset file config cache between tests."""
    config_module._file_config_cache = None
    yield
    config_module._file_config_cache = None


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


class TestConfigFromFile:
    """Tests for Config.from_file()."""

    def test_load_from_pyproject(self, tmp_path):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("""
[tool.magically.models.fast]
model = "anthropic:claude-haiku"

[tool.magically.models.reasoning]
model = "anthropic:claude-opus-4"
temperature = 0.7
""")
        config = Config.from_file(pyproject)
        assert "fast" in config.models
        assert "reasoning" in config.models
        assert config.models["fast"].model == "anthropic:claude-haiku"
        assert config.models["reasoning"].temperature == 0.7

    def test_load_all_model_settings(self, tmp_path):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("""
[tool.magically.models.full]
model = "openai:gpt-4o"
temperature = 0.5
max_tokens = 4096
top_p = 0.9
timeout = 30.0
retries = 3
""")
        config = Config.from_file(pyproject)
        model = config.models["full"]
        assert model.model == "openai:gpt-4o"
        assert model.temperature == 0.5
        assert model.max_tokens == 4096
        assert model.top_p == 0.9
        assert model.timeout == 30.0
        assert model.retries == 3

    def test_missing_file_returns_empty(self, tmp_path):
        missing = tmp_path / "nonexistent.toml"
        config = Config.from_file(missing)
        assert config.models == {}

    def test_malformed_toml_returns_empty(self, tmp_path):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("this is not valid [ toml")
        config = Config.from_file(pyproject)
        assert config.models == {}

    def test_no_magically_section_returns_empty(self, tmp_path):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("""
[project]
name = "myproject"
""")
        config = Config.from_file(pyproject)
        assert config.models == {}

    def test_no_models_section_returns_empty(self, tmp_path):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("""
[tool.magically]
some_other_setting = true
""")
        config = Config.from_file(pyproject)
        assert config.models == {}

    def test_warns_on_unknown_fields(self, tmp_path):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("""
[tool.magically.models.test]
model = "test:model"
unknown_field = "value"
another_unknown = 123
""")
        with pytest.warns(UserWarning, match="Unknown fields.*test.*unknown_field"):
            Config.from_file(pyproject)

    def test_raises_on_non_dict_entries(self, tmp_path):
        """Non-dict model entries should raise a helpful error."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("""
[tool.magically.models]
valid = { model = "test:model" }
invalid = "not a dict"
""")
        with pytest.raises(
            MagicallyConfigError,
            match=r"Invalid config for \[tool\.magically\.models\.invalid\].*expected a table/dict"
        ):
            Config.from_file(pyproject)

    def test_raises_on_invalid_types(self, tmp_path):
        """Invalid types for fields should raise a helpful error."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("""
[tool.magically.models.bad]
model = "test:model"
temperature = "hot"
""")
        with pytest.raises(
            MagicallyConfigError,
            match=r"Invalid config for \[tool\.magically\.models\.bad\]"
        ):
            Config.from_file(pyproject)

    def test_raises_on_missing_required_field(self, tmp_path):
        """Missing required 'model' field should raise a helpful error."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("""
[tool.magically.models.incomplete]
temperature = 0.7
""")
        with pytest.raises(
            MagicallyConfigError,
            match=r"Invalid config for \[tool\.magically\.models\.incomplete\]"
        ):
            Config.from_file(pyproject)


class TestConfigCurrentWithFile:
    """Tests for Config.current() with file config fallback."""

    def test_current_falls_back_to_file(self, tmp_path, monkeypatch):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("""
[tool.magically.models.file_alias]
model = "anthropic:claude-sonnet"
""")
        monkeypatch.chdir(tmp_path)

        config = Config.current()
        assert "file_alias" in config.models

    def test_file_config_is_cached(self, tmp_path, monkeypatch):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("""
[tool.magically.models.cached]
model = "test:model"
""")
        monkeypatch.chdir(tmp_path)

        config1 = Config.current()
        config2 = Config.current()
        assert config1 is config2

    def test_context_overrides_file(self, tmp_path, monkeypatch):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("""
[tool.magically.models.file_alias]
model = "file:model"
""")
        monkeypatch.chdir(tmp_path)

        override = Config(models={"override": ModelConfig(model="override:model")})

        assert "file_alias" in Config.current().models

        with override:
            assert "override" in Config.current().models
            assert "file_alias" not in Config.current().models

    def test_process_default_overrides_file(self, tmp_path, monkeypatch):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("""
[tool.magically.models.file_alias]
model = "file:model"
""")
        monkeypatch.chdir(tmp_path)

        default = Config(models={"default": ModelConfig(model="default:model")})
        default.set_as_default()

        assert "default" in Config.current().models
        assert "file_alias" not in Config.current().models
