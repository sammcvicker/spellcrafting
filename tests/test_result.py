"""Tests for SpellResult and with_metadata functionality."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from magically import spell, SpellResult
from magically.config import Config, ModelConfig
from magically.logging import trace_context, with_trace_id


class Category(BaseModel):
    name: str
    confidence: float


class TestSpellResultType:
    """Tests for SpellResult dataclass."""

    def test_basic_creation(self):
        result = SpellResult(
            output="test output",
            input_tokens=50,
            output_tokens=100,
            model_used="openai:gpt-4o",
            attempt_count=1,
            duration_ms=150.5,
        )
        assert result.output == "test output"
        assert result.input_tokens == 50
        assert result.output_tokens == 100
        assert result.model_used == "openai:gpt-4o"
        assert result.attempt_count == 1
        assert result.duration_ms == 150.5

    def test_default_values(self):
        result = SpellResult(output="test")
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.model_used == ""
        assert result.attempt_count == 1
        assert result.duration_ms == 0.0
        assert result.cost_estimate is None
        assert result.trace_id is None

    def test_total_tokens_property(self):
        result = SpellResult(
            output="test",
            input_tokens=50,
            output_tokens=100,
        )
        assert result.total_tokens == 150

    def test_total_tokens_with_defaults(self):
        result = SpellResult(output="test")
        assert result.total_tokens == 0

    def test_cost_estimate_field(self):
        result = SpellResult(
            output="test",
            cost_estimate=0.00015,
        )
        assert result.cost_estimate == 0.00015

    def test_trace_id_field(self):
        result = SpellResult(
            output="test",
            trace_id="abc123def456",
        )
        assert result.trace_id == "abc123def456"

    def test_generic_type(self):
        result: SpellResult[Category] = SpellResult(
            output=Category(name="test", confidence=0.9),
            input_tokens=10,
            output_tokens=20,
        )
        assert isinstance(result.output, Category)
        assert result.output.name == "test"
        assert result.output.confidence == 0.9


class TestWithMetadataSync:
    """Tests for sync spell.with_metadata()."""

    def test_with_metadata_exists_on_sync_spell(self):
        @spell
        def classify(text: str) -> str:
            """Classify the text."""
            ...

        assert hasattr(classify, "with_metadata")
        assert callable(classify.with_metadata)

    def test_with_metadata_returns_spell_result(self):
        @spell
        def classify(text: str) -> str:
            """Classify the text."""
            ...

        mock_usage = MagicMock()
        mock_usage.request_tokens = 50
        mock_usage.response_tokens = 100

        mock_result = MagicMock()
        mock_result.output = "positive"
        mock_result.usage.return_value = mock_usage

        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("magically.spell.Agent", return_value=mock_agent):
            result = classify.with_metadata("some text")

            assert isinstance(result, SpellResult)
            assert result.output == "positive"
            assert result.input_tokens == 50
            assert result.output_tokens == 100
            assert result.attempt_count == 1
            assert result.duration_ms > 0

    def test_with_metadata_with_structured_output(self):
        @spell
        def classify(text: str) -> Category:
            """Classify the text."""
            ...

        expected = Category(name="positive", confidence=0.95)
        mock_usage = MagicMock()
        mock_usage.request_tokens = 50
        mock_usage.response_tokens = 100

        mock_result = MagicMock()
        mock_result.output = expected
        mock_result.usage.return_value = mock_usage

        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("magically.spell.Agent", return_value=mock_agent):
            result = classify.with_metadata("some text")

            assert isinstance(result, SpellResult)
            assert isinstance(result.output, Category)
            assert result.output.name == "positive"
            assert result.output.confidence == 0.95

    def test_with_metadata_includes_model_used(self):
        @spell(model="openai:gpt-4o")
        def classify(text: str) -> str:
            """Classify the text."""
            ...

        mock_usage = MagicMock()
        mock_usage.request_tokens = 50
        mock_usage.response_tokens = 100

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_result.usage.return_value = mock_usage

        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("magically.spell.Agent", return_value=mock_agent):
            result = classify.with_metadata("some text")

            assert result.model_used == "openai:gpt-4o"

    def test_with_metadata_resolves_model_alias(self):
        config = Config(models={
            "fast": ModelConfig(model="anthropic:claude-haiku")
        })

        @spell(model="fast")
        def classify(text: str) -> str:
            """Classify the text."""
            ...

        mock_usage = MagicMock()
        mock_usage.request_tokens = 50
        mock_usage.response_tokens = 100

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_result.usage.return_value = mock_usage

        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with config:
            with patch("magically.spell.Agent", return_value=mock_agent):
                result = classify.with_metadata("some text")
                # model_used should be the resolved model, not the alias
                assert result.model_used == "anthropic:claude-haiku"

    def test_with_metadata_handles_usage_extraction_error(self):
        @spell
        def classify(text: str) -> str:
            """Classify the text."""
            ...

        mock_result = MagicMock()
        mock_result.output = "result"
        # Use AttributeError to simulate usage() method not available
        mock_result.usage.side_effect = AttributeError("Usage not available")

        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("magically.spell.Agent", return_value=mock_agent):
            result = classify.with_metadata("some text")

            # Should not raise, just have 0 tokens
            assert result.output == "result"
            assert result.input_tokens == 0
            assert result.output_tokens == 0


class TestWithMetadataAsync:
    """Tests for async spell.with_metadata()."""

    def test_with_metadata_exists_on_async_spell(self):
        @spell
        async def classify(text: str) -> str:
            """Classify the text."""
            ...

        assert hasattr(classify, "with_metadata")
        assert callable(classify.with_metadata)

    @pytest.mark.asyncio
    async def test_async_with_metadata_returns_spell_result(self):
        @spell
        async def classify(text: str) -> str:
            """Classify the text."""
            ...

        mock_usage = MagicMock()
        mock_usage.request_tokens = 50
        mock_usage.response_tokens = 100

        mock_result = MagicMock()
        mock_result.output = "positive"
        mock_result.usage.return_value = mock_usage

        mock_agent = MagicMock()

        async def mock_run(prompt):
            return mock_result
        mock_agent.run = mock_run

        with patch("magically.spell.Agent", return_value=mock_agent):
            result = await classify.with_metadata("some text")

            assert isinstance(result, SpellResult)
            assert result.output == "positive"
            assert result.input_tokens == 50
            assert result.output_tokens == 100
            assert result.attempt_count == 1
            assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_async_with_metadata_with_structured_output(self):
        @spell
        async def classify(text: str) -> Category:
            """Classify the text."""
            ...

        expected = Category(name="negative", confidence=0.85)
        mock_usage = MagicMock()
        mock_usage.request_tokens = 50
        mock_usage.response_tokens = 100

        mock_result = MagicMock()
        mock_result.output = expected
        mock_result.usage.return_value = mock_usage

        mock_agent = MagicMock()

        async def mock_run(prompt):
            return mock_result
        mock_agent.run = mock_run

        with patch("magically.spell.Agent", return_value=mock_agent):
            result = await classify.with_metadata("some text")

            assert isinstance(result, SpellResult)
            assert isinstance(result.output, Category)
            assert result.output.name == "negative"
            assert result.output.confidence == 0.85

    @pytest.mark.asyncio
    async def test_async_with_metadata_includes_model_used(self):
        @spell(model="openai:gpt-4o-mini")
        async def classify(text: str) -> str:
            """Classify the text."""
            ...

        mock_usage = MagicMock()
        mock_usage.request_tokens = 50
        mock_usage.response_tokens = 100

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_result.usage.return_value = mock_usage

        mock_agent = MagicMock()

        async def mock_run(prompt):
            return mock_result
        mock_agent.run = mock_run

        with patch("magically.spell.Agent", return_value=mock_agent):
            result = await classify.with_metadata("some text")
            assert result.model_used == "openai:gpt-4o-mini"


class TestWithMetadataEdgeCases:
    """Tests for edge cases in with_metadata."""

    def test_normal_call_still_works(self):
        """Ensure adding with_metadata doesn't break normal calls."""
        @spell
        def classify(text: str) -> str:
            """Classify the text."""
            ...

        mock_result = MagicMock()
        mock_result.output = "positive"

        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("magically.spell.Agent", return_value=mock_agent):
            # Normal call should still return just the output
            result = classify("some text")
            assert result == "positive"
            assert not isinstance(result, SpellResult)

    def test_with_metadata_uses_same_agent_cache(self):
        """Ensure with_metadata uses the same cached agent as normal calls."""
        @spell(model="openai:gpt-4o")
        def classify(text: str) -> str:
            """Classify the text."""
            ...

        mock_usage = MagicMock()
        mock_usage.request_tokens = 50
        mock_usage.response_tokens = 100

        mock_result = MagicMock()
        mock_result.output = "positive"
        mock_result.usage.return_value = mock_usage

        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("magically.spell.Agent", return_value=mock_agent) as mock_agent_class:
            # Normal call
            classify("text 1")
            assert mock_agent_class.call_count == 1

            # with_metadata call should reuse the same agent
            classify.with_metadata("text 2")
            assert mock_agent_class.call_count == 1

            # Both calls should have used the same agent
            assert mock_agent.run_sync.call_count == 2


class TestSpellResultCostEstimate:
    """Tests for cost_estimate field in SpellResult."""

    def test_cost_estimate_with_known_model(self):
        """Cost estimate is populated for models with known pricing."""
        @spell(model="openai:gpt-4o")
        def classify(text: str) -> str:
            """Classify the text."""
            ...

        mock_usage = MagicMock()
        mock_usage.request_tokens = 1000
        mock_usage.response_tokens = 500

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_result.usage.return_value = mock_usage

        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("magically.spell.Agent", return_value=mock_agent):
            result = classify.with_metadata("some text")

            assert result.cost_estimate is not None
            # gpt-4o: $2.50/1M input, $10.00/1M output
            # 1000 input tokens = $0.0025, 500 output tokens = $0.005
            expected_cost = (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.00
            assert abs(result.cost_estimate - expected_cost) < 0.0001

    def test_cost_estimate_none_for_unknown_model(self):
        """Cost estimate is None for models without known pricing."""
        @spell(model="unknown:some-model")
        def classify(text: str) -> str:
            """Classify the text."""
            ...

        mock_usage = MagicMock()
        mock_usage.request_tokens = 1000
        mock_usage.response_tokens = 500

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_result.usage.return_value = mock_usage

        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("magically.spell.Agent", return_value=mock_agent):
            result = classify.with_metadata("some text")

            assert result.cost_estimate is None

    def test_cost_estimate_none_when_no_tokens(self):
        """Cost estimate is None when there are no tokens."""
        @spell(model="openai:gpt-4o")
        def classify(text: str) -> str:
            """Classify the text."""
            ...

        mock_result = MagicMock()
        mock_result.output = "result"
        # Use AttributeError to simulate usage() method not available
        mock_result.usage.side_effect = AttributeError("No usage")

        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("magically.spell.Agent", return_value=mock_agent):
            result = classify.with_metadata("some text")

            assert result.cost_estimate is None


class TestSpellResultTraceId:
    """Tests for trace_id field in SpellResult."""

    def test_trace_id_none_without_trace_context(self):
        """Trace ID is None when no trace context is active."""
        @spell
        def classify(text: str) -> str:
            """Classify the text."""
            ...

        mock_usage = MagicMock()
        mock_usage.request_tokens = 50
        mock_usage.response_tokens = 100

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_result.usage.return_value = mock_usage

        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("magically.spell.Agent", return_value=mock_agent):
            result = classify.with_metadata("some text")

            assert result.trace_id is None

    def test_trace_id_populated_with_trace_context(self):
        """Trace ID is populated when trace context is active."""
        @spell
        def classify(text: str) -> str:
            """Classify the text."""
            ...

        mock_usage = MagicMock()
        mock_usage.request_tokens = 50
        mock_usage.response_tokens = 100

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_result.usage.return_value = mock_usage

        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("magically.spell.Agent", return_value=mock_agent):
            with trace_context() as ctx:
                result = classify.with_metadata("some text")

                assert result.trace_id is not None
                assert result.trace_id == ctx.trace_id

    def test_trace_id_with_specific_trace_id(self):
        """Trace ID matches when using with_trace_id context manager."""
        @spell
        def classify(text: str) -> str:
            """Classify the text."""
            ...

        mock_usage = MagicMock()
        mock_usage.request_tokens = 50
        mock_usage.response_tokens = 100

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_result.usage.return_value = mock_usage

        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        expected_trace_id = "my-external-trace-id-12345"

        with patch("magically.spell.Agent", return_value=mock_agent):
            with with_trace_id(expected_trace_id):
                result = classify.with_metadata("some text")

                assert result.trace_id == expected_trace_id

    @pytest.mark.asyncio
    async def test_async_trace_id_with_trace_context(self):
        """Async: Trace ID is populated when trace context is active."""
        @spell
        async def classify(text: str) -> str:
            """Classify the text."""
            ...

        mock_usage = MagicMock()
        mock_usage.request_tokens = 50
        mock_usage.response_tokens = 100

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_result.usage.return_value = mock_usage

        mock_agent = MagicMock()

        async def mock_run(prompt):
            return mock_result
        mock_agent.run = mock_run

        with patch("magically.spell.Agent", return_value=mock_agent):
            with trace_context() as ctx:
                result = await classify.with_metadata("some text")

                assert result.trace_id is not None
                assert result.trace_id == ctx.trace_id


class TestSpellResultTotalTokens:
    """Tests for total_tokens property in SpellResult via with_metadata."""

    def test_total_tokens_computed_correctly(self):
        """Total tokens is correctly computed from input + output tokens."""
        @spell(model="openai:gpt-4o")
        def classify(text: str) -> str:
            """Classify the text."""
            ...

        mock_usage = MagicMock()
        mock_usage.request_tokens = 150
        mock_usage.response_tokens = 250

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_result.usage.return_value = mock_usage

        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("magically.spell.Agent", return_value=mock_agent):
            result = classify.with_metadata("some text")

            assert result.input_tokens == 150
            assert result.output_tokens == 250
            assert result.total_tokens == 400
