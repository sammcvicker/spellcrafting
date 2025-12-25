"""Tests for the @spell decorator."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

import sys

import magically.config as config_module
from magically import spell
from magically.config import Config, MagicallyConfigError, ModelConfig
from magically.spell import _build_user_prompt, _is_literal_model

# Access the actual spell module (not the function)
spell_module = sys.modules["magically.spell"]


@pytest.fixture(autouse=True)
def reset_config():
    """Reset config state between tests."""
    config_module._file_config_cache = None
    config_module._process_default = None
    yield
    config_module._file_config_cache = None
    config_module._process_default = None


@pytest.fixture(autouse=True)
def reset_agent_cache():
    """Reset agent cache between tests."""
    spell_module._agent_cache.clear()
    yield
    spell_module._agent_cache.clear()


class Summary(BaseModel):
    key_points: list[str]
    sentiment: str


class TestBuildUserPrompt:
    """Tests for user prompt construction from function arguments."""

    def test_single_arg(self):
        def fn(text: str) -> str:
            pass

        result = _build_user_prompt(fn, ("hello world",), {})
        assert result == "text: 'hello world'"

    def test_multiple_args(self):
        def fn(a: str, b: int) -> str:
            pass

        result = _build_user_prompt(fn, ("foo", 42), {})
        assert "a: 'foo'" in result
        assert "b: 42" in result

    def test_kwargs(self):
        def fn(text: str, count: int = 5) -> str:
            pass

        result = _build_user_prompt(fn, (), {"text": "hello", "count": 10})
        assert "text: 'hello'" in result
        assert "count: 10" in result

    def test_default_values(self):
        def fn(text: str, count: int = 5) -> str:
            pass

        result = _build_user_prompt(fn, ("hello",), {})
        assert "text: 'hello'" in result
        assert "count: 5" in result


class TestSpellDecoratorInspection:
    """Tests for function inspection by @spell decorator."""

    def test_extracts_docstring_as_system_prompt(self):
        @spell
        def summarize(text: str) -> str:
            """Summarize the given text."""
            ...

        assert summarize._system_prompt == "Summarize the given text."

    def test_extracts_return_type(self):
        @spell
        def analyze(text: str) -> Summary:
            """Analyze text."""
            ...

        assert analyze._output_type == Summary

    def test_default_return_type_is_str(self):
        @spell
        def process(text: str):
            """Process text."""
            ...

        assert process._output_type == str

    def test_preserves_function_metadata(self):
        @spell
        def my_spell(text: str) -> str:
            """My docstring."""
            ...

        assert my_spell.__name__ == "my_spell"
        assert my_spell.__doc__ == "My docstring."

    def test_stores_original_function(self):
        def original(text: str) -> str:
            """Original."""
            ...

        decorated = spell(original)
        assert decorated._original_func is original


class TestSpellDecoratorParams:
    """Tests for @spell decorator parameters."""

    def test_literal_model_param(self):
        @spell(model="openai:gpt-4o")
        def fn(text: str) -> str:
            """Test."""
            ...

        resolved_model, _, _ = fn._resolve_model_and_settings()
        assert resolved_model == "openai:gpt-4o"

    def test_retries_param(self):
        @spell(retries=3)
        def fn(text: str) -> str:
            """Test."""
            ...

        assert fn._retries == 3

    def test_decorator_without_parens(self):
        @spell
        def fn(text: str) -> str:
            """Test."""
            ...

        assert hasattr(fn, "_original_func")

    def test_decorator_with_empty_parens(self):
        @spell()
        def fn(text: str) -> str:
            """Test."""
            ...

        assert hasattr(fn, "_original_func")


class TestSpellExecution:
    """Tests for @spell execution with mocked LLM."""

    def test_calls_agent_run_sync(self):
        @spell
        def summarize(text: str) -> str:
            """Summarize."""
            ...

        mock_result = MagicMock()
        mock_result.output = "Mocked summary"
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("magically.spell.Agent", return_value=mock_agent):
            result = summarize("Hello world")

            mock_agent.run_sync.assert_called_once()
            assert "text: 'Hello world'" in mock_agent.run_sync.call_args[0][0]
            assert result == "Mocked summary"

    def test_returns_structured_output(self):
        @spell
        def analyze(text: str) -> Summary:
            """Analyze."""
            ...

        expected = Summary(key_points=["point1"], sentiment="positive")
        mock_result = MagicMock()
        mock_result.output = expected
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("magically.spell.Agent", return_value=mock_agent):
            result = analyze("Test text")
            assert result == expected
            assert isinstance(result, Summary)


class TestLiteralModelDetection:
    """Tests for _is_literal_model helper."""

    def test_literal_with_colon(self):
        assert _is_literal_model("openai:gpt-4o") is True
        assert _is_literal_model("anthropic:claude-sonnet") is True

    def test_alias_without_colon(self):
        assert _is_literal_model("fast") is False
        assert _is_literal_model("reasoning") is False


class TestSpellConfigIntegration:
    """Tests for @spell with Config integration."""

    def test_literal_model_bypasses_config(self):
        @spell(model="openai:gpt-4o")
        def fn(text: str) -> str:
            """Test."""
            ...

        resolved_model, _, _ = fn._resolve_model_and_settings()
        assert resolved_model == "openai:gpt-4o"

    def test_alias_resolves_via_config_context(self):
        config = Config(models={
            "fast": ModelConfig(model="anthropic:claude-haiku", temperature=0.5)
        })

        @spell(model="fast")
        def fn(text: str) -> str:
            """Test."""
            ...

        with config:
            resolved_model, resolved_settings, _ = fn._resolve_model_and_settings()
            assert resolved_model == "anthropic:claude-haiku"
            assert resolved_settings.get("temperature") == 0.5

    def test_alias_resolves_via_process_default(self):
        config = Config(models={
            "reasoning": ModelConfig(model="anthropic:claude-opus-4")
        })
        config.set_as_default()

        @spell(model="reasoning")
        def fn(text: str) -> str:
            """Test."""
            ...

        resolved_model, _, _ = fn._resolve_model_and_settings()
        assert resolved_model == "anthropic:claude-opus-4"

    def test_alias_resolves_via_file_config(self, tmp_path, monkeypatch):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("""
[tool.magically.models.fast]
model = "file:model"
max_tokens = 2048
""")
        monkeypatch.chdir(tmp_path)

        @spell(model="fast")
        def fn(text: str) -> str:
            """Test."""
            ...

        resolved_model, resolved_settings, _ = fn._resolve_model_and_settings()
        assert resolved_model == "file:model"
        assert resolved_settings.get("max_tokens") == 2048

    def test_unresolved_alias_raises_at_call_time(self):
        @spell(model="nonexistent")
        def fn(text: str) -> str:
            """Test."""
            ...

        with pytest.raises(MagicallyConfigError, match="nonexistent"):
            fn._resolve_model_and_settings()

    def test_definition_time_warning_for_missing_alias(self, tmp_path, monkeypatch):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("""
[project]
name = "test"
""")
        monkeypatch.chdir(tmp_path)

        with pytest.warns(UserWarning, match="Model alias 'missing'"):
            @spell(model="missing")
            def fn(text: str) -> str:
                """Test."""
                ...

    def test_no_warning_for_literal_model(self, tmp_path, monkeypatch):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("""
[project]
name = "test"
""")
        monkeypatch.chdir(tmp_path)

        # Should not warn for literal models
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            @spell(model="openai:gpt-4o")
            def fn(text: str) -> str:
                """Test."""
                ...

    def test_model_settings_from_config_merged_with_explicit(self):
        config = Config(models={
            "fast": ModelConfig(
                model="test:model",
                temperature=0.5,
                max_tokens=1000
            )
        })

        @spell(model="fast", model_settings={"temperature": 0.9})
        def fn(text: str) -> str:
            """Test."""
            ...

        with config:
            _, resolved_settings, _ = fn._resolve_model_and_settings()
            # Explicit temperature overrides config
            assert resolved_settings.get("temperature") == 0.9
            # Config max_tokens is preserved
            assert resolved_settings.get("max_tokens") == 1000

    def test_context_overrides_process_default_for_alias(self):
        default = Config(models={
            "fast": ModelConfig(model="default:model")
        })
        default.set_as_default()

        override = Config(models={
            "fast": ModelConfig(model="override:model")
        })

        @spell(model="fast")
        def fn(text: str) -> str:
            """Test."""
            ...

        # Without context, uses process default
        resolved_model, _, _ = fn._resolve_model_and_settings()
        assert resolved_model == "default:model"

        # With context, uses context config
        with override:
            resolved_model, _, _ = fn._resolve_model_and_settings()
            assert resolved_model == "override:model"


class TestAgentCaching:
    """Tests for agent caching behavior."""

    def test_same_config_reuses_agent(self):
        config = Config(models={
            "fast": ModelConfig(model="test:model")
        })

        @spell(model="fast")
        def fn(text: str) -> str:
            """Test."""
            ...

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with config:
            with patch("magically.spell.Agent", return_value=mock_agent) as mock_agent_class:
                # First call creates agent
                fn("hello")
                assert mock_agent_class.call_count == 1

                # Second call with same config reuses agent
                fn("world")
                assert mock_agent_class.call_count == 1

                # Verify agent was used twice
                assert mock_agent.run_sync.call_count == 2

    def test_different_config_creates_new_agent(self):
        config1 = Config(models={
            "fast": ModelConfig(model="test:model1")
        })
        config2 = Config(models={
            "fast": ModelConfig(model="test:model2")
        })

        @spell(model="fast")
        def fn(text: str) -> str:
            """Test."""
            ...

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("magically.spell.Agent", return_value=mock_agent) as mock_agent_class:
            # Call with first config
            with config1:
                fn("hello")
            assert mock_agent_class.call_count == 1

            # Call with different config creates new agent
            with config2:
                fn("world")
            assert mock_agent_class.call_count == 2

    def test_different_spells_get_different_agents(self):
        config = Config(models={
            "fast": ModelConfig(model="test:model")
        })

        @spell(model="fast")
        def fn1(text: str) -> str:
            """Test 1."""
            ...

        @spell(model="fast")
        def fn2(text: str) -> str:
            """Test 2."""
            ...

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with config:
            with patch("magically.spell.Agent", return_value=mock_agent) as mock_agent_class:
                fn1("hello")
                assert mock_agent_class.call_count == 1

                # Different spell creates new agent even with same config
                fn2("world")
                assert mock_agent_class.call_count == 2

    def test_literal_model_caching(self):
        @spell(model="openai:gpt-4o")
        def fn(text: str) -> str:
            """Test."""
            ...

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("magically.spell.Agent", return_value=mock_agent) as mock_agent_class:
            fn("hello")
            fn("world")
            # Same literal model reuses agent
            assert mock_agent_class.call_count == 1

    def test_cache_key_includes_model_settings(self):
        config = Config(models={
            "fast": ModelConfig(model="test:model", temperature=0.5)
        })

        @spell(model="fast")
        def fn(text: str) -> str:
            """Test."""
            ...

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("magically.spell.Agent", return_value=mock_agent) as mock_agent_class:
            with config:
                fn("hello")
            assert mock_agent_class.call_count == 1

            # Different temperature creates new agent
            config2 = Config(models={
                "fast": ModelConfig(model="test:model", temperature=0.9)
            })
            with config2:
                fn("world")
            assert mock_agent_class.call_count == 2

    def test_spell_id_stored_on_wrapper(self):
        @spell(model="openai:gpt-4o")
        def fn(text: str) -> str:
            """Test."""
            ...

        assert hasattr(fn, "_spell_id")
        assert fn._spell_id == id(fn)
