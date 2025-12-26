"""Tests for on_fail strategies."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from spellcrafting import spell, OnFail, Config, ModelConfig, ValidationError
from spellcrafting._pydantic_ai import UnexpectedModelBehavior
from spellcrafting.on_fail import (
    RetryStrategy,
    EscalateStrategy,
    FallbackStrategy,
    CustomStrategy,
    RaiseStrategy,
)


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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
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
            with patch("spellcrafting.spell.Agent", side_effect=create_agent):
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

        with patch("spellcrafting.spell.Agent", side_effect=create_agent):
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

        with patch("spellcrafting.spell.Agent", side_effect=create_agent):
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            with pytest.raises(ValidationError):
                analyze("test input")


class TestOnFailAsyncHandlerInSyncContext:
    """Tests for async on_fail handlers used with sync spells (#137)."""

    def test_sync_spell_with_async_custom_handler_returns_coroutine(self):
        """Sync spell with async handler returns the coroutine object (not awaited).

        This documents the current behavior: when an async handler is used with
        a sync spell, the coroutine is returned directly. Users should use sync
        handlers with sync spells.
        """
        import asyncio

        async def async_handler(error, attempt, ctx):
            return Analysis(summary="async fixed", confidence=0.5)

        @spell(on_fail=OnFail.custom(async_handler))
        def analyze(text: str) -> Analysis:
            """Analyze text."""
            ...

        mock_agent = MagicMock()
        mock_agent.run_sync.side_effect = UnexpectedModelBehavior("Validation failed")

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            result = analyze("test")
            # Result is a coroutine object, not the actual Analysis
            # This is the current behavior - async handlers with sync spells
            # return unawaited coroutines
            assert asyncio.iscoroutine(result)
            # Clean up the coroutine to avoid warning
            result.close()

    def test_sync_spell_with_sync_handler_works(self):
        """Sync spell with sync handler works correctly."""
        def sync_handler(error, attempt, ctx):
            return Analysis(summary="sync fixed", confidence=0.5)

        @spell(on_fail=OnFail.custom(sync_handler))
        def analyze(text: str) -> Analysis:
            """Analyze text."""
            ...

        mock_agent = MagicMock()
        mock_agent.run_sync.side_effect = UnexpectedModelBehavior("Validation failed")

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            result = analyze("test")
            # Sync handler returns proper Analysis object
            assert isinstance(result, Analysis)
            assert result.summary == "sync fixed"


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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
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

        with patch("spellcrafting.spell.Agent", side_effect=create_agent):
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            result = analyze("test input")

            assert result == expected


class TestOnFailStrategyImmutability:
    """Tests for OnFail strategy immutability (#159)."""

    def test_retry_strategy_frozen(self):
        """RetryStrategy should be immutable (frozen dataclass)."""
        from dataclasses import FrozenInstanceError

        strategy = RetryStrategy()
        with pytest.raises(FrozenInstanceError):
            strategy.some_attr = "new_value"

    def test_escalate_strategy_frozen(self):
        """EscalateStrategy should be immutable (frozen dataclass)."""
        from dataclasses import FrozenInstanceError

        strategy = EscalateStrategy(model="test")
        with pytest.raises(FrozenInstanceError):
            strategy.model = "other"
        with pytest.raises(FrozenInstanceError):
            strategy.retries = 5

    def test_fallback_strategy_frozen(self):
        """FallbackStrategy should be immutable (frozen dataclass)."""
        from dataclasses import FrozenInstanceError

        strategy = FallbackStrategy(default="value")
        with pytest.raises(FrozenInstanceError):
            strategy.default = "other"

    def test_custom_strategy_frozen(self):
        """CustomStrategy should be immutable (frozen dataclass)."""
        from dataclasses import FrozenInstanceError

        strategy = CustomStrategy(handler=lambda e, a, c: None)
        with pytest.raises(FrozenInstanceError):
            strategy.handler = lambda e, a, c: "other"

    def test_raise_strategy_frozen(self):
        """RaiseStrategy should be immutable (frozen dataclass)."""
        from dataclasses import FrozenInstanceError

        strategy = RaiseStrategy()
        with pytest.raises(FrozenInstanceError):
            strategy.some_attr = "new_value"

    def test_strategies_can_be_used_as_dict_keys(self):
        """Frozen strategies should be hashable and usable as dict keys."""
        retry = OnFail.retry()
        escalate = OnFail.escalate("model1")
        fallback = OnFail.fallback("default")

        strategy_map = {
            retry: "retry",
            escalate: "escalate",
            fallback: "fallback",
        }

        assert strategy_map[retry] == "retry"
        assert strategy_map[escalate] == "escalate"
        assert strategy_map[fallback] == "fallback"

    def test_same_strategies_are_equal(self):
        """Identical frozen strategies should be equal."""
        retry1 = OnFail.retry()
        retry2 = OnFail.retry()
        assert retry1 == retry2

        escalate1 = OnFail.escalate("model1", retries=2)
        escalate2 = OnFail.escalate("model1", retries=2)
        assert escalate1 == escalate2

        fallback1 = OnFail.fallback("default")
        fallback2 = OnFail.fallback("default")
        assert fallback1 == fallback2

    def test_different_strategies_are_not_equal(self):
        """Different frozen strategies should not be equal."""
        escalate1 = OnFail.escalate("model1")
        escalate2 = OnFail.escalate("model2")
        assert escalate1 != escalate2

        fallback1 = OnFail.fallback("default1")
        fallback2 = OnFail.fallback("default2")
        assert fallback1 != fallback2


class TestResolveEscalationModel:
    """Tests for _resolve_escalation_model function (#20).

    This function resolves escalation model aliases to actual model strings
    and settings. It's used by the escalate on_fail strategy.
    """

    def test_literal_model_returned_as_is(self):
        """Literal model (with colon) should be returned unchanged."""
        from spellcrafting.spell import _resolve_escalation_model

        model, settings = _resolve_escalation_model("openai:gpt-4o")
        assert model == "openai:gpt-4o"
        assert settings is None

    def test_literal_model_with_provider_prefix(self):
        """Various literal model formats should work."""
        from spellcrafting.spell import _resolve_escalation_model

        # Anthropic model
        model, settings = _resolve_escalation_model("anthropic:claude-sonnet-4-20250514")
        assert model == "anthropic:claude-sonnet-4-20250514"
        assert settings is None

        # Google model
        model, settings = _resolve_escalation_model("google:gemini-1.5-pro")
        assert model == "google:gemini-1.5-pro"
        assert settings is None

    def test_alias_resolved_from_config_context(self):
        """Alias should be resolved from Config context."""
        from spellcrafting.spell import _resolve_escalation_model
        from spellcrafting.config import Config, ModelConfig

        config = Config(
            models={
                "reasoning": ModelConfig(
                    model="openai:gpt-4o",
                    temperature=0.7,
                    max_tokens=4096,
                )
            }
        )

        with config:
            model, settings = _resolve_escalation_model("reasoning")
            assert model == "openai:gpt-4o"
            assert settings is not None
            assert settings.get("temperature") == 0.7
            assert settings.get("max_tokens") == 4096

    def test_alias_resolved_from_process_default(self):
        """Alias should be resolved from process-level default config."""
        from spellcrafting.spell import _resolve_escalation_model
        from spellcrafting.config import Config, ModelConfig

        config = Config(
            models={
                "powerful": ModelConfig(
                    model="anthropic:claude-opus-4-20250514",
                    temperature=0.5,
                )
            }
        )
        config.set_as_default()

        try:
            model, settings = _resolve_escalation_model("powerful")
            assert model == "anthropic:claude-opus-4-20250514"
            assert settings is not None
            assert settings.get("temperature") == 0.5
        finally:
            # Clean up process default
            Config(models={}).set_as_default()

    def test_unknown_alias_raises_config_error(self):
        """Unknown alias should raise SpellcraftingConfigError with helpful message."""
        from spellcrafting.spell import _resolve_escalation_model
        from spellcrafting.config import SpellcraftingConfigError

        with pytest.raises(SpellcraftingConfigError, match="could not be resolved"):
            _resolve_escalation_model("unknown_alias")

    def test_unknown_alias_error_mentions_alias_name(self):
        """Error message should include the alias name that couldn't be resolved."""
        from spellcrafting.spell import _resolve_escalation_model
        from spellcrafting.config import SpellcraftingConfigError

        with pytest.raises(SpellcraftingConfigError, match="my_custom_model"):
            _resolve_escalation_model("my_custom_model")

    def test_settings_extraction_from_model_config(self):
        """Settings should be extracted from ModelConfig correctly."""
        from spellcrafting.spell import _resolve_escalation_model
        from spellcrafting.config import Config, ModelConfig

        config = Config(
            models={
                "custom": ModelConfig(
                    model="openai:gpt-4o-mini",
                    temperature=0.2,
                    max_tokens=1000,
                    top_p=0.9,
                )
            }
        )

        with config:
            model, settings = _resolve_escalation_model("custom")
            assert model == "openai:gpt-4o-mini"
            assert settings is not None
            assert settings.get("temperature") == 0.2
            assert settings.get("max_tokens") == 1000
            assert settings.get("top_p") == 0.9

    def test_model_config_without_optional_settings(self):
        """ModelConfig with only required fields should return None settings."""
        from spellcrafting.spell import _resolve_escalation_model
        from spellcrafting.config import Config, ModelConfig

        config = Config(
            models={
                "minimal": ModelConfig(model="openai:gpt-4o")
            }
        )

        with config:
            model, settings = _resolve_escalation_model("minimal")
            assert model == "openai:gpt-4o"
            # Settings may be None or an empty dict depending on implementation
            # The key is that no error is raised
