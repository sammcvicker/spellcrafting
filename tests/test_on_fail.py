"""Tests for on_fail strategies."""

import sys
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from magically import spell, OnFail, Config, ModelConfig, ValidationError
from magically._pydantic_ai import UnexpectedModelBehavior
from magically.on_fail import (
    RetryStrategy,
    EscalateStrategy,
    FallbackStrategy,
    CustomStrategy,
    RaiseStrategy,
)

import magically.config as config_module

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


class Analysis(BaseModel):
    summary: str
    confidence: float


class TestOnFailStrategyTypes:
    """Tests for OnFail strategy factory methods."""

    def test_retry_creates_retry_strategy(self):
        strategy = OnFail.retry()
        assert isinstance(strategy, RetryStrategy)

    def test_escalate_creates_escalate_strategy(self):
        strategy = OnFail.escalate("reasoning")
        assert isinstance(strategy, EscalateStrategy)
        assert strategy.model == "reasoning"
        assert strategy.retries == 1

    def test_escalate_with_custom_retries(self):
        strategy = OnFail.escalate("reasoning", retries=3)
        assert strategy.retries == 3

    def test_fallback_creates_fallback_strategy(self):
        default = Analysis(summary="default", confidence=0.0)
        strategy = OnFail.fallback(default=default)
        assert isinstance(strategy, FallbackStrategy)
        assert strategy.default == default

    def test_custom_creates_custom_strategy(self):
        def handler(error, attempt, ctx):
            return "fixed"

        strategy = OnFail.custom(handler)
        assert isinstance(strategy, CustomStrategy)
        assert strategy.handler is handler

    def test_raise_is_raise_strategy(self):
        assert isinstance(OnFail.RAISE, RaiseStrategy)


class TestOnFailStoresOnWrapper:
    """Tests that on_fail is stored on the wrapper for introspection."""

    def test_on_fail_stored_none_by_default(self):
        @spell
        def fn(text: str) -> str:
            """Test."""
            ...

        assert fn._on_fail is None

    def test_on_fail_stored_when_specified(self):
        strategy = OnFail.escalate("reasoning")

        @spell(on_fail=strategy)
        def fn(text: str) -> str:
            """Test."""
            ...

        assert fn._on_fail is strategy


class TestOnFailFallbackStrategy:
    """Tests for OnFail.fallback() strategy."""

    def test_fallback_returns_default_on_validation_error(self):
        default = Analysis(summary="fallback", confidence=0.0)

        @spell(on_fail=OnFail.fallback(default=default))
        def analyze(text: str) -> Analysis:
            """Analyze text."""
            ...

        mock_agent = MagicMock()
        mock_agent.run_sync.side_effect = UnexpectedModelBehavior("Validation failed")

        with patch("magically.spell.Agent", return_value=mock_agent):
            result = analyze("test input")

            assert result == default
            mock_agent.run_sync.assert_called_once()

    def test_fallback_passes_through_on_success(self):
        default = Analysis(summary="fallback", confidence=0.0)
        expected = Analysis(summary="real", confidence=0.9)

        @spell(on_fail=OnFail.fallback(default=default))
        def analyze(text: str) -> Analysis:
            """Analyze text."""
            ...

        mock_result = MagicMock()
        mock_result.output = expected
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("magically.spell.Agent", return_value=mock_agent):
            result = analyze("test input")

            assert result == expected


class TestOnFailCustomStrategy:
    """Tests for OnFail.custom() strategy."""

    def test_custom_handler_receives_error_and_context(self):
        received = {}

        def handler(error, attempt, ctx):
            received["error"] = error
            received["attempt"] = attempt
            received["ctx"] = ctx
            return Analysis(summary="fixed", confidence=0.5)

        @spell(model="openai:gpt-4o", on_fail=OnFail.custom(handler))
        def analyze(text: str) -> Analysis:
            """Analyze text."""
            ...

        mock_agent = MagicMock()
        validation_error = UnexpectedModelBehavior("Validation failed")
        mock_agent.run_sync.side_effect = validation_error

        with patch("magically.spell.Agent", return_value=mock_agent):
            analyze("test input")

            assert received["error"] is validation_error
            assert received["attempt"] == 1
            assert received["ctx"]["spell_name"] == "analyze"
            assert received["ctx"]["model"] == "openai:gpt-4o"
            assert received["ctx"]["input_args"] == {"text": "test input"}

    def test_custom_handler_can_fix_output(self):
        def handler(error, attempt, ctx):
            return Analysis(summary="fixed", confidence=0.5)

        @spell(on_fail=OnFail.custom(handler))
        def analyze(text: str) -> Analysis:
            """Analyze text."""
            ...

        mock_agent = MagicMock()
        mock_agent.run_sync.side_effect = UnexpectedModelBehavior("Validation failed")

        with patch("magically.spell.Agent", return_value=mock_agent):
            result = analyze("test input")

            assert result.summary == "fixed"
            assert result.confidence == 0.5

    def test_custom_handler_can_re_raise(self):
        def handler(error, attempt, ctx):
            raise error

        @spell(on_fail=OnFail.custom(handler))
        def analyze(text: str) -> Analysis:
            """Analyze text."""
            ...

        mock_agent = MagicMock()
        mock_agent.run_sync.side_effect = UnexpectedModelBehavior("Validation failed")

        with patch("magically.spell.Agent", return_value=mock_agent):
            # Custom handler re-raises the original error, so we get the original exception
            with pytest.raises(UnexpectedModelBehavior):
                analyze("test input")


class TestOnFailEscalateStrategy:
    """Tests for OnFail.escalate() strategy."""

    def test_escalate_creates_new_agent_with_escalated_model(self):
        config = Config(
            models={
                "fast": ModelConfig(model="openai:gpt-4o-mini"),
                "reasoning": ModelConfig(model="openai:gpt-4o"),
            }
        )

        @spell(model="fast", on_fail=OnFail.escalate("reasoning"))
        def analyze(text: str) -> Analysis:
            """Analyze text."""
            ...

        # First agent fails, second succeeds
        mock_result = MagicMock()
        mock_result.output = Analysis(summary="escalated", confidence=0.9)

        agents_created = []

        def create_agent(*args, **kwargs):
            agent = MagicMock()
            if len(agents_created) == 0:
                # First agent fails
                agent.run_sync.side_effect = UnexpectedModelBehavior("Validation failed")
            else:
                # Escalated agent succeeds
                agent.run_sync.return_value = mock_result
            agents_created.append(kwargs.get("model"))
            return agent

        with config:
            with patch("magically.spell.Agent", side_effect=create_agent):
                result = analyze("test input")

                assert result.summary == "escalated"
                # Original agent + escalated agent
                assert len(agents_created) == 2
                assert agents_created[0] == "openai:gpt-4o-mini"
                assert agents_created[1] == "openai:gpt-4o"

    def test_escalate_uses_literal_model(self):
        @spell(model="openai:gpt-4o-mini", on_fail=OnFail.escalate("anthropic:claude-sonnet"))
        def analyze(text: str) -> Analysis:
            """Analyze text."""
            ...

        mock_result = MagicMock()
        mock_result.output = Analysis(summary="escalated", confidence=0.9)

        agents_created = []

        def create_agent(*args, **kwargs):
            agent = MagicMock()
            if len(agents_created) == 0:
                agent.run_sync.side_effect = UnexpectedModelBehavior("Validation failed")
            else:
                agent.run_sync.return_value = mock_result
            agents_created.append(kwargs.get("model"))
            return agent

        with patch("magically.spell.Agent", side_effect=create_agent):
            result = analyze("test input")

            assert agents_created[1] == "anthropic:claude-sonnet"

    def test_escalate_respects_retries_parameter(self):
        @spell(on_fail=OnFail.escalate("openai:gpt-4o", retries=3))
        def analyze(text: str) -> Analysis:
            """Analyze text."""
            ...

        mock_result = MagicMock()
        mock_result.output = Analysis(summary="escalated", confidence=0.9)

        agents_created = []

        def create_agent(*args, **kwargs):
            agent = MagicMock()
            if len(agents_created) == 0:
                agent.run_sync.side_effect = UnexpectedModelBehavior("Validation failed")
            else:
                agent.run_sync.return_value = mock_result
            agents_created.append(kwargs)
            return agent

        with patch("magically.spell.Agent", side_effect=create_agent):
            analyze("test input")

            # Escalated agent should have retries=3
            assert agents_created[1]["retries"] == 3


class TestOnFailRetryStrategy:
    """Tests for OnFail.retry() strategy (default behavior)."""

    def test_retry_re_raises_validation_error(self):
        @spell(on_fail=OnFail.retry())
        def analyze(text: str) -> Analysis:
            """Analyze text."""
            ...

        mock_agent = MagicMock()
        mock_agent.run_sync.side_effect = UnexpectedModelBehavior("Validation failed")

        with patch("magically.spell.Agent", return_value=mock_agent):
            with pytest.raises(ValidationError):
                analyze("test input")


class TestOnFailAsync:
    """Tests for on_fail strategies with async spells."""

    @pytest.mark.asyncio
    async def test_async_fallback_returns_default(self):
        default = Analysis(summary="fallback", confidence=0.0)

        @spell(on_fail=OnFail.fallback(default=default))
        async def analyze(text: str) -> Analysis:
            """Analyze text."""
            ...

        mock_agent = MagicMock()

        async def mock_run(prompt):
            raise UnexpectedModelBehavior("Validation failed")

        mock_agent.run = mock_run

        with patch("magically.spell.Agent", return_value=mock_agent):
            result = await analyze("test input")

            assert result == default

    @pytest.mark.asyncio
    async def test_async_custom_handler(self):
        def handler(error, attempt, ctx):
            return Analysis(summary="fixed", confidence=0.5)

        @spell(on_fail=OnFail.custom(handler))
        async def analyze(text: str) -> Analysis:
            """Analyze text."""
            ...

        mock_agent = MagicMock()

        async def mock_run(prompt):
            raise UnexpectedModelBehavior("Validation failed")

        mock_agent.run = mock_run

        with patch("magically.spell.Agent", return_value=mock_agent):
            result = await analyze("test input")

            assert result.summary == "fixed"

    @pytest.mark.asyncio
    async def test_async_custom_handler_supports_async(self):
        async def async_handler(error, attempt, ctx):
            return Analysis(summary="async fixed", confidence=0.5)

        @spell(on_fail=OnFail.custom(async_handler))
        async def analyze(text: str) -> Analysis:
            """Analyze text."""
            ...

        mock_agent = MagicMock()

        async def mock_run(prompt):
            raise UnexpectedModelBehavior("Validation failed")

        mock_agent.run = mock_run

        with patch("magically.spell.Agent", return_value=mock_agent):
            result = await analyze("test input")

            assert result.summary == "async fixed"

    @pytest.mark.asyncio
    async def test_async_escalate_creates_new_agent(self):
        @spell(model="openai:gpt-4o-mini", on_fail=OnFail.escalate("openai:gpt-4o"))
        async def analyze(text: str) -> Analysis:
            """Analyze text."""
            ...

        mock_result = MagicMock()
        mock_result.output = Analysis(summary="escalated", confidence=0.9)

        agents_created = []

        def create_agent(*args, **kwargs):
            agent = MagicMock()

            async def mock_run(prompt):
                if len(agents_created) == 1:
                    raise UnexpectedModelBehavior("Validation failed")
                return mock_result

            agent.run = mock_run
            agents_created.append(kwargs.get("model"))
            return agent

        with patch("magically.spell.Agent", side_effect=create_agent):
            result = await analyze("test input")

            assert result.summary == "escalated"
            assert len(agents_created) == 2


class TestOnFailNoStrategyProvided:
    """Tests for behavior when no on_fail strategy is provided."""

    def test_no_strategy_raises_validation_error(self):
        @spell
        def analyze(text: str) -> Analysis:
            """Analyze text."""
            ...

        mock_agent = MagicMock()
        mock_agent.run_sync.side_effect = UnexpectedModelBehavior("Validation failed")

        with patch("magically.spell.Agent", return_value=mock_agent):
            with pytest.raises(ValidationError):
                analyze("test input")

    def test_validation_error_has_original_error(self):
        """Test that ValidationError wraps the original pydantic_ai exception."""
        @spell
        def analyze(text: str) -> Analysis:
            """Analyze text."""
            ...

        mock_agent = MagicMock()
        original_error = UnexpectedModelBehavior("Validation failed")
        mock_agent.run_sync.side_effect = original_error

        with patch("magically.spell.Agent", return_value=mock_agent):
            with pytest.raises(ValidationError) as exc_info:
                analyze("test input")

            # Verify the original error is preserved
            assert exc_info.value.original_error is original_error
            assert "Validation failed" in str(exc_info.value)

    def test_no_strategy_passes_on_success(self):
        expected = Analysis(summary="success", confidence=0.9)

        @spell
        def analyze(text: str) -> Analysis:
            """Analyze text."""
            ...

        mock_result = MagicMock()
        mock_result.output = expected
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("magically.spell.Agent", return_value=mock_agent):
            result = analyze("test input")

            assert result == expected
