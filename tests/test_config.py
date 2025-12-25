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
            # Config.current() returns merged config, not the exact context object
            assert "fast" in Config.current().models
            assert Config.current().resolve("fast").model == "test"

        assert "fast" not in Config.current().models

    def test_nested_contexts(self):
        """Nested contexts replace each other (inner takes precedence)."""
        outer = Config(models={"outer": ModelConfig(model="outer-model")})
        inner = Config(models={"inner": ModelConfig(model="inner-model")})

        with outer:
            assert "outer" in Config.current().models

            with inner:
                # Inner context replaces outer (only one context active at a time)
                assert "inner" in Config.current().models
                assert "outer" not in Config.current().models

            assert "outer" in Config.current().models

    def test_context_returns_self(self):
        config = Config()
        with config as ctx:
            assert ctx is config

    def test_context_manager_propagates_exception(self):
        """Exceptions inside context should propagate correctly (#173)."""
        config = Config(models={"test": ModelConfig(model="test")})

        with pytest.raises(ValueError, match="test error"):
            with config:
                raise ValueError("test error")

        # Config should be properly reset even after exception
        assert "test" not in Config.current().models

    def test_context_manager_resets_on_exception(self):
        """Config should be reset even when exception occurs in nested context (#173)."""
        outer = Config(models={"outer": ModelConfig(model="outer")})
        inner = Config(models={"inner": ModelConfig(model="inner")})

        with outer:
            with pytest.raises(RuntimeError):
                with inner:
                    # Inner context replaces outer
                    assert "inner" in Config.current().models
                    assert "outer" not in Config.current().models
                    raise RuntimeError("boom")

            # After inner context exits with exception, outer should be active
            assert "outer" in Config.current().models
            assert "inner" not in Config.current().models

    def test_context_manager_exception_does_not_swallow_error(self):
        """Context manager __exit__ should not suppress exceptions (#173)."""
        config = Config(models={"test": ModelConfig(model="test")})
        error_raised = False

        try:
            with config:
                raise KeyError("specific error")
        except KeyError as e:
            error_raised = True
            assert str(e) == "'specific error'"

        assert error_raised

    def test_context_manager_resets_token_on_exception(self):
        """The internal _token should be reset even on exception (#173)."""
        config = Config(models={"test": ModelConfig(model="test")})

        # Before entering context
        assert config._token is None

        try:
            with config:
                # Inside context, token should be set
                assert config._token is not None
                raise ValueError("test")
        except ValueError:
            pass

        # After exception, token should be reset
        assert config._token is None

    def test_deeply_nested_contexts_with_exceptions(self):
        """Multiple nested contexts should all reset properly on exception (#173)."""
        c1 = Config(models={"c1": ModelConfig(model="m1")})
        c2 = Config(models={"c2": ModelConfig(model="m2")})
        c3 = Config(models={"c3": ModelConfig(model="m3")})

        with c1:
            assert "c1" in Config.current().models
            with c2:
                # Inner context replaces outer
                assert "c2" in Config.current().models
                assert "c1" not in Config.current().models
                with pytest.raises(Exception):
                    with c3:
                        # Innermost context is active
                        assert "c3" in Config.current().models
                        assert "c2" not in Config.current().models
                        raise Exception("deep error")
                # After c3 exits with error, c2 should be active
                assert "c2" in Config.current().models
                assert "c3" not in Config.current().models
            # After c2 exits, c1 should be active
            assert "c1" in Config.current().models
            assert "c2" not in Config.current().models


class TestConfigProcessDefault:
    """Tests for Config.set_as_default()."""

    def test_set_as_default(self):
        config = Config(models={
            "default-alias": ModelConfig(model="test"),
        })
        config.set_as_default()

        assert "default-alias" in Config.current().models

    def test_context_merges_with_default(self):
        """Context config merges with process default, not replaces."""
        default = Config(models={"default": ModelConfig(model="default-model")})
        default.set_as_default()

        override = Config(models={"override": ModelConfig(model="override-model")})

        assert "default" in Config.current().models

        with override:
            # Both should be available due to merging
            assert "override" in Config.current().models
            assert "default" in Config.current().models

        assert "default" in Config.current().models


class TestCurrentConfig:
    """Tests for current_config() convenience function."""

    def test_current_config_returns_current(self):
        config = Config(models={"test": ModelConfig(model="test")})

        with config:
            # current_config() and Config.current() both return merged configs
            # with the same models (though different object instances)
            assert current_config().models == Config.current().models
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

    def test_context_merges_with_file(self, tmp_path, monkeypatch):
        """Context config merges with file config, not replaces."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("""
[tool.magically.models.file_alias]
model = "file:model"
""")
        monkeypatch.chdir(tmp_path)

        override = Config(models={"override": ModelConfig(model="override:model")})

        assert "file_alias" in Config.current().models

        with override:
            # Both should be available due to merging
            assert "override" in Config.current().models
            assert "file_alias" in Config.current().models

    def test_process_default_merges_with_file(self, tmp_path, monkeypatch):
        """Process default merges with file config, not replaces."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("""
[tool.magically.models.file_alias]
model = "file:model"
""")
        monkeypatch.chdir(tmp_path)

        default = Config(models={"default": ModelConfig(model="default:model")})
        default.set_as_default()

        # Both should be available due to merging
        assert "default" in Config.current().models
        assert "file_alias" in Config.current().models


class TestConfigMerge:
    """Tests for Config.merge() method and config merging behavior."""

    def test_merge_combines_aliases(self):
        """Merge should combine aliases from both configs."""
        base = Config(models={"a": ModelConfig(model="model-a")})
        other = Config(models={"b": ModelConfig(model="model-b")})

        merged = base.merge(other)

        assert "a" in merged.models
        assert "b" in merged.models
        assert merged.resolve("a").model == "model-a"
        assert merged.resolve("b").model == "model-b"

    def test_merge_other_takes_precedence(self):
        """Aliases in 'other' should override those in 'self'."""
        base = Config(models={"fast": ModelConfig(model="base-fast")})
        other = Config(models={"fast": ModelConfig(model="other-fast")})

        merged = base.merge(other)

        assert merged.resolve("fast").model == "other-fast"

    def test_merge_preserves_originals(self):
        """Merge should not modify the original configs."""
        base = Config(models={"a": ModelConfig(model="model-a")})
        other = Config(models={"b": ModelConfig(model="model-b")})

        base.merge(other)

        # Originals should be unchanged
        assert "b" not in base.models
        assert "a" not in other.models

    def test_merge_returns_new_config(self):
        """Merge should return a new Config instance."""
        base = Config(models={"a": ModelConfig(model="model-a")})
        other = Config(models={"b": ModelConfig(model="model-b")})

        merged = base.merge(other)

        assert merged is not base
        assert merged is not other

    def test_context_merges_with_file_config(self, tmp_path, monkeypatch):
        """Context config should merge with file config (issue #96)."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("""
[tool.magically.models.fast]
model = "file:fast-model"

[tool.magically.models.reasoning]
model = "file:reasoning-model"
""")
        monkeypatch.chdir(tmp_path)

        # Context config with different alias and override for 'fast'
        context = Config(models={
            "fast": ModelConfig(model="ctx:fast-model"),  # override
            "creative": ModelConfig(model="ctx:creative-model"),  # new
        })

        with context:
            # All three should be available
            assert "fast" in Config.current().models
            assert "reasoning" in Config.current().models
            assert "creative" in Config.current().models

            # Context takes precedence for 'fast'
            assert Config.current().resolve("fast").model == "ctx:fast-model"
            # File config provides 'reasoning'
            assert Config.current().resolve("reasoning").model == "file:reasoning-model"
            # Context provides 'creative'
            assert Config.current().resolve("creative").model == "ctx:creative-model"

    def test_process_default_merges_with_file_and_context(self, tmp_path, monkeypatch):
        """All three config sources should merge correctly."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("""
[tool.magically.models.file_only]
model = "file:model"
""")
        monkeypatch.chdir(tmp_path)

        # Reset file cache to pick up the new file
        config_module._file_config_cache = None

        # Process default with its own alias
        process_default = Config(models={
            "default_only": ModelConfig(model="default:model"),
        })
        process_default.set_as_default()

        # Context config with its own alias
        context = Config(models={
            "context_only": ModelConfig(model="ctx:model"),
        })

        with context:
            # All three should be available
            assert "file_only" in Config.current().models
            assert "default_only" in Config.current().models
            assert "context_only" in Config.current().models

    def test_context_overrides_process_default(self, tmp_path, monkeypatch):
        """Context should take precedence over process default for same alias."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("")
        monkeypatch.chdir(tmp_path)

        # Reset file cache
        config_module._file_config_cache = None

        process_default = Config(models={
            "fast": ModelConfig(model="default:fast"),
        })
        process_default.set_as_default()

        context = Config(models={
            "fast": ModelConfig(model="ctx:fast"),
        })

        with context:
            assert Config.current().resolve("fast").model == "ctx:fast"

    def test_process_default_overrides_file(self, tmp_path, monkeypatch):
        """Process default should take precedence over file for same alias."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("""
[tool.magically.models.fast]
model = "file:fast"
""")
        monkeypatch.chdir(tmp_path)

        # Reset file cache
        config_module._file_config_cache = None

        process_default = Config(models={
            "fast": ModelConfig(model="default:fast"),
        })
        process_default.set_as_default()

        assert Config.current().resolve("fast").model == "default:fast"
