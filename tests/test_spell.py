"""Tests for the @spell decorator."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from mage import spell
from mage.spell import _build_user_prompt


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

        assert summarize._agent._system_prompts == ("Summarize the given text.",)

    def test_extracts_return_type(self):
        @spell
        def analyze(text: str) -> Summary:
            """Analyze text."""
            ...

        assert analyze._agent._output_type == Summary

    def test_default_return_type_is_str(self):
        @spell
        def process(text: str):
            """Process text."""
            ...

        assert process._agent._output_type == str

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

    def test_model_param(self):
        @spell(model="openai:gpt-4o")
        def fn(text: str) -> str:
            """Test."""
            ...

        assert fn._agent.model.model_name == "gpt-4o"

    def test_retries_param(self):
        @spell(retries=3)
        def fn(text: str) -> str:
            """Test."""
            ...

        assert fn._agent._max_result_retries == 3

    def test_decorator_without_parens(self):
        @spell
        def fn(text: str) -> str:
            """Test."""
            ...

        assert hasattr(fn, "_agent")

    def test_decorator_with_empty_parens(self):
        @spell()
        def fn(text: str) -> str:
            """Test."""
            ...

        assert hasattr(fn, "_agent")


class TestSpellExecution:
    """Tests for @spell execution with mocked LLM."""

    def test_calls_agent_run_sync(self):
        @spell
        def summarize(text: str) -> str:
            """Summarize."""
            ...

        mock_result = MagicMock()
        mock_result.output = "Mocked summary"

        with patch.object(summarize._agent, "run_sync", return_value=mock_result) as mock_run:
            result = summarize("Hello world")

            mock_run.assert_called_once()
            assert "text: 'Hello world'" in mock_run.call_args[0][0]
            assert result == "Mocked summary"

    def test_returns_structured_output(self):
        @spell
        def analyze(text: str) -> Summary:
            """Analyze."""
            ...

        expected = Summary(key_points=["point1"], sentiment="positive")
        mock_result = MagicMock()
        mock_result.output = expected

        with patch.object(analyze._agent, "run_sync", return_value=mock_result):
            result = analyze("Test text")
            assert result == expected
            assert isinstance(result, Summary)
