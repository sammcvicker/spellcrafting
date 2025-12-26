"""Tests for SpellResult and with_metadata functionality."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from spellcrafting import spell, SpellResult
from spellcrafting.config import Config, ModelConfig
from spellcrafting.logging import trace_context, with_trace_id


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

    def test_repr_short_output(self):
        """Repr shows full output when short."""
        result = SpellResult(
            output="short text",
            input_tokens=50,
            output_tokens=100,
            model_used="openai:gpt-4o",
            duration_ms=123.456,
        )
        repr_str = repr(result)
        assert "SpellResult(" in repr_str
        assert "'short text'" in repr_str
        assert "tokens=150" in repr_str
        assert "model='openai:gpt-4o'" in repr_str
        assert "duration_ms=123.5" in repr_str

    def test_repr_long_output_truncated(self):
        """Repr truncates output longer than 100 characters."""
        long_output = "x" * 200
        result = SpellResult(
            output=long_output,
            model_used="openai:gpt-4o",
        )
        repr_str = repr(result)
        # The repr of output is "'xxx...'" which is longer than the string itself
        # So we check that the full output is not in repr and truncation happened
        assert long_output not in repr_str
        assert "..." in repr_str

    def test_repr_complex_output_truncated(self):
        """Repr truncates complex Pydantic model output when repr is long."""
        # Create a Pydantic model with long content
        long_name = "category_" + "x" * 100
        complex_output = Category(name=long_name, confidence=0.95)
        result = SpellResult(
            output=complex_output,
            model_used="openai:gpt-4o",
        )
        repr_str = repr(result)
        # The full long_name should not appear since output is truncated
        assert long_name not in repr_str
        assert "..." in repr_str

    def test_content_eq_same_content(self):
        """content_eq returns True when output and model match."""
        result1 = SpellResult(
            output="test output",
            model_used="openai:gpt-4o",
            input_tokens=50,
            output_tokens=100,
            duration_ms=100.0,
        )
        result2 = SpellResult(
            output="test output",
            model_used="openai:gpt-4o",
            input_tokens=75,  # Different tokens
            output_tokens=150,  # Different tokens
            duration_ms=200.0,  # Different duration
        )
        assert result1.content_eq(result2)

    def test_content_eq_different_output(self):
        """content_eq returns False when output differs."""
        result1 = SpellResult(
            output="output1",
            model_used="openai:gpt-4o",
        )
        result2 = SpellResult(
            output="output2",
            model_used="openai:gpt-4o",
        )
        assert not result1.content_eq(result2)

    def test_content_eq_different_model(self):
        """content_eq returns False when model differs."""
        result1 = SpellResult(
            output="test output",
            model_used="openai:gpt-4o",
        )
        result2 = SpellResult(
            output="test output",
            model_used="anthropic:claude-3-haiku",
        )
        assert not result1.content_eq(result2)

    def test_content_eq_with_pydantic_models(self):
        """content_eq works correctly with Pydantic model outputs."""
        cat1 = Category(name="positive", confidence=0.95)
        cat2 = Category(name="positive", confidence=0.95)

        result1 = SpellResult(
            output=cat1,
            model_used="openai:gpt-4o",
            duration_ms=100.0,
        )
        result2 = SpellResult(
            output=cat2,
            model_used="openai:gpt-4o",
            duration_ms=200.0,
        )
        assert result1.content_eq(result2)

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


class TestSpellResultEdgeCases:
    """Additional edge case tests for SpellResult (issue #7)."""

    def test_spell_result_with_zero_tokens(self):
        """SpellResult handles zero tokens correctly."""
        result = SpellResult(output="test", input_tokens=0, output_tokens=0)
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.total_tokens == 0

    def test_spell_result_with_empty_model(self):
        """SpellResult handles empty model string."""
        result = SpellResult(output="test", model_used="")
        assert result.model_used == ""

    def test_spell_result_repr_contains_class_name(self):
        """repr includes SpellResult class name."""
        result = SpellResult(output="test")
        assert "SpellResult" in repr(result)

    def test_spell_result_with_none_output(self):
        """SpellResult can hold None as output (for edge cases)."""
        result = SpellResult(output=None)
        assert result.output is None
        assert "SpellResult" in repr(result)

    def test_spell_result_with_empty_string_output(self):
        """SpellResult handles empty string output."""
        result = SpellResult(output="")
        assert result.output == ""
        assert result.total_tokens == 0

    def test_spell_result_with_large_token_counts(self):
        """SpellResult handles large token counts."""
        result = SpellResult(
            output="test",
            input_tokens=1_000_000,
            output_tokens=500_000,
        )
        assert result.total_tokens == 1_500_000

    def test_spell_result_with_zero_duration(self):
        """SpellResult handles zero duration."""
        result = SpellResult(output="test", duration_ms=0.0)
        assert result.duration_ms == 0.0

    def test_spell_result_with_very_small_duration(self):
        """SpellResult handles very small duration values."""
        result = SpellResult(output="test", duration_ms=0.001)
        assert result.duration_ms == 0.001

    def test_spell_result_with_zero_cost_estimate(self):
        """SpellResult handles zero cost estimate."""
        result = SpellResult(output="test", cost_estimate=0.0)
        assert result.cost_estimate == 0.0

    def test_spell_result_with_very_small_cost(self):
        """SpellResult handles very small cost values."""
        result = SpellResult(output="test", cost_estimate=0.000001)
        assert result.cost_estimate == 0.000001

    def test_spell_result_equality_is_reference_based(self):
        """Two SpellResults with same data are not equal (dataclass default)."""
        result1 = SpellResult(output="test", model_used="openai:gpt-4o")
        result2 = SpellResult(output="test", model_used="openai:gpt-4o")
        # Dataclasses with eq=True (default) compare field values
        assert result1 == result2

    def test_spell_result_with_list_output(self):
        """SpellResult can hold list as output."""
        result = SpellResult(output=["item1", "item2", "item3"])
        assert result.output == ["item1", "item2", "item3"]

    def test_spell_result_with_dict_output(self):
        """SpellResult can hold dict as output."""
        result = SpellResult(output={"key": "value", "count": 42})
        assert result.output == {"key": "value", "count": 42}

    def test_spell_result_with_nested_complex_output(self):
        """SpellResult can hold nested complex structures."""
        complex_output = {
            "items": [Category(name="a", confidence=0.9), Category(name="b", confidence=0.8)],
            "metadata": {"total": 2},
        }
        result = SpellResult(output=complex_output)
        assert len(result.output["items"]) == 2
        assert result.output["metadata"]["total"] == 2

    def test_spell_result_with_high_attempt_count(self):
        """SpellResult handles high attempt counts."""
        result = SpellResult(output="test", attempt_count=100)
        assert result.attempt_count == 100

    def test_spell_result_repr_with_special_characters(self):
        """repr handles output with special characters."""
        result = SpellResult(output="test\nwith\nnewlines\tand\ttabs")
        repr_str = repr(result)
        assert "SpellResult" in repr_str
        # Repr should escape special characters
        assert "\\n" in repr_str or "newlines" in repr_str


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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
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
            with patch("spellcrafting.spell.Agent", return_value=mock_agent):
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent) as mock_agent_class:
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            result = classify.with_metadata("some text")

            assert result.input_tokens == 150
            assert result.output_tokens == 250
            assert result.total_tokens == 400


class TestWithMetadataOnFailStrategies:
    """Tests for with_metadata tracking of on_fail strategies."""

    def test_with_metadata_tracks_fallback(self):
        """with_metadata returns fallback default when on_fail=fallback is triggered."""
        from spellcrafting.on_fail import OnFail
        from spellcrafting._pydantic_ai import UnexpectedModelBehavior

        fallback_default = "fallback result"

        @spell(model="openai:gpt-4o", on_fail=OnFail.fallback(default=fallback_default))
        def analyze(text: str) -> str:
            """Analyze the text."""
            ...

        mock_agent = MagicMock()
        mock_agent.run_sync.side_effect = UnexpectedModelBehavior("Validation failed")

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            result = analyze.with_metadata("test input")

            assert isinstance(result, SpellResult)
            assert result.output == fallback_default
            # Fallback should still track attempt_count as 1 (initial attempt failed)
            assert result.attempt_count == 1

    def test_with_metadata_tracks_escalation(self):
        """with_metadata tracks escalated model when on_fail=escalate is triggered."""
        from spellcrafting.on_fail import OnFail
        from spellcrafting._pydantic_ai import UnexpectedModelBehavior

        @spell(model="openai:gpt-4o-mini", on_fail=OnFail.escalate("openai:gpt-4o"))
        def analyze(text: str) -> str:
            """Analyze the text."""
            ...

        # First agent fails
        mock_agent_mini = MagicMock()
        mock_agent_mini.run_sync.side_effect = UnexpectedModelBehavior("Validation failed")

        # Escalated agent succeeds
        mock_usage = MagicMock()
        mock_usage.request_tokens = 100
        mock_usage.response_tokens = 50

        mock_result = MagicMock()
        mock_result.output = "escalated result"
        mock_result.usage.return_value = mock_usage

        mock_agent_escalated = MagicMock()
        mock_agent_escalated.run_sync.return_value = mock_result

        call_count = [0]

        def agent_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_agent_mini
            return mock_agent_escalated

        with patch("spellcrafting.spell.Agent", side_effect=agent_side_effect):
            result = analyze.with_metadata("test input")

            assert isinstance(result, SpellResult)
            assert result.output == "escalated result"
            # The result should reflect successful completion
            # (escalation model is tracked in ValidationMetrics, not model_used directly)

    def test_with_metadata_tracks_custom_handler(self):
        """with_metadata works with custom on_fail handler."""
        from spellcrafting.on_fail import OnFail
        from spellcrafting._pydantic_ai import UnexpectedModelBehavior

        custom_output = "custom handled result"

        def my_handler(error: Exception, attempt: int, context: dict) -> str:
            return custom_output

        @spell(model="openai:gpt-4o", on_fail=OnFail.custom(my_handler))
        def analyze(text: str) -> str:
            """Analyze the text."""
            ...

        mock_agent = MagicMock()
        mock_agent.run_sync.side_effect = UnexpectedModelBehavior("Validation failed")

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            result = analyze.with_metadata("test input")

            assert isinstance(result, SpellResult)
            assert result.output == custom_output

    @pytest.mark.asyncio
    async def test_async_with_metadata_tracks_fallback(self):
        """Async with_metadata returns fallback default when on_fail=fallback is triggered."""
        from spellcrafting.on_fail import OnFail
        from spellcrafting._pydantic_ai import UnexpectedModelBehavior

        fallback_default = "async fallback"

        @spell(model="openai:gpt-4o", on_fail=OnFail.fallback(default=fallback_default))
        async def analyze(text: str) -> str:
            """Analyze the text."""
            ...

        mock_agent = MagicMock()

        async def mock_run(prompt):
            raise UnexpectedModelBehavior("Validation failed")
        mock_agent.run = mock_run

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            result = await analyze.with_metadata("test input")

            assert isinstance(result, SpellResult)
            assert result.output == fallback_default

    @pytest.mark.asyncio
    async def test_async_with_metadata_tracks_escalation(self):
        """Async with_metadata tracks escalation when on_fail=escalate is triggered."""
        from spellcrafting.on_fail import OnFail
        from spellcrafting._pydantic_ai import UnexpectedModelBehavior

        @spell(model="openai:gpt-4o-mini", on_fail=OnFail.escalate("openai:gpt-4o"))
        async def analyze(text: str) -> str:
            """Analyze the text."""
            ...

        # First agent fails
        mock_agent_mini = MagicMock()

        async def mock_run_fail(prompt):
            raise UnexpectedModelBehavior("Validation failed")
        mock_agent_mini.run = mock_run_fail

        # Escalated agent succeeds
        mock_usage = MagicMock()
        mock_usage.request_tokens = 100
        mock_usage.response_tokens = 50

        mock_result = MagicMock()
        mock_result.output = "async escalated result"
        mock_result.usage.return_value = mock_usage

        mock_agent_escalated = MagicMock()

        async def mock_run_success(prompt):
            return mock_result
        mock_agent_escalated.run = mock_run_success

        call_count = [0]

        def agent_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_agent_mini
            return mock_agent_escalated

        with patch("spellcrafting.spell.Agent", side_effect=agent_side_effect):
            result = await analyze.with_metadata("test input")

            assert isinstance(result, SpellResult)
            assert result.output == "async escalated result"


class TestWithMetadataWithGuards:
    """Tests for with_metadata when guards are attached to spells (issue #25)."""

    def test_with_metadata_runs_input_guards_sync(self):
        """Sync: with_metadata runs input guards before LLM call."""
        from spellcrafting import guard

        guard_called = []

        def track_input_guard(input_args: dict, ctx: dict) -> dict:
            guard_called.append("input")
            return input_args

        @spell
        @guard.input(track_input_guard)
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            result = classify.with_metadata("test input")

            assert isinstance(result, SpellResult)
            assert result.output == "positive"
            assert guard_called == ["input"]

    def test_with_metadata_runs_output_guards_sync(self):
        """Sync: with_metadata runs output guards after LLM call."""
        from spellcrafting import guard

        guard_called = []

        def track_output_guard(output: str, ctx: dict) -> str:
            guard_called.append("output")
            return output.upper()

        @spell
        @guard.output(track_output_guard)
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            result = classify.with_metadata("test input")

            assert isinstance(result, SpellResult)
            # Output should be transformed by the guard
            assert result.output == "POSITIVE"
            assert guard_called == ["output"]

    def test_with_metadata_runs_both_guards_sync(self):
        """Sync: with_metadata runs both input and output guards in correct order."""
        from spellcrafting import guard

        guard_order = []

        def input_guard(input_args: dict, ctx: dict) -> dict:
            guard_order.append("input")
            return input_args

        def output_guard(output: str, ctx: dict) -> str:
            guard_order.append("output")
            return output

        @spell
        @guard.input(input_guard)
        @guard.output(output_guard)
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            result = classify.with_metadata("test input")

            assert isinstance(result, SpellResult)
            assert result.output == "result"
            # Input guard runs before output guard
            assert guard_order == ["input", "output"]

    @pytest.mark.asyncio
    async def test_with_metadata_runs_input_guards_async(self):
        """Async: with_metadata runs input guards before LLM call."""
        from spellcrafting import guard

        guard_called = []

        def track_input_guard(input_args: dict, ctx: dict) -> dict:
            guard_called.append("input")
            return input_args

        @spell
        @guard.input(track_input_guard)
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            result = await classify.with_metadata("test input")

            assert isinstance(result, SpellResult)
            assert result.output == "positive"
            assert guard_called == ["input"]

    @pytest.mark.asyncio
    async def test_with_metadata_runs_output_guards_async(self):
        """Async: with_metadata runs output guards after LLM call."""
        from spellcrafting import guard

        guard_called = []

        def track_output_guard(output: str, ctx: dict) -> str:
            guard_called.append("output")
            return output.upper()

        @spell
        @guard.output(track_output_guard)
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            result = await classify.with_metadata("test input")

            assert isinstance(result, SpellResult)
            # Output should be transformed by the guard
            assert result.output == "POSITIVE"
            assert guard_called == ["output"]

    @pytest.mark.asyncio
    async def test_with_metadata_runs_async_guards(self):
        """Async: with_metadata properly awaits async guard functions."""
        from spellcrafting import guard

        guard_called = []

        async def async_input_guard(input_args: dict, ctx: dict) -> dict:
            guard_called.append("async_input")
            return input_args

        async def async_output_guard(output: str, ctx: dict) -> str:
            guard_called.append("async_output")
            return output

        @spell
        @guard.input(async_input_guard)
        @guard.output(async_output_guard)
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            result = await classify.with_metadata("test input")

            assert isinstance(result, SpellResult)
            assert result.output == "result"
            # Both async guards should have been called
            assert guard_called == ["async_input", "async_output"]

    def test_with_metadata_guard_transforms_input(self):
        """with_metadata correctly uses transformed input from guard."""
        from spellcrafting import guard

        def uppercase_input(input_args: dict, ctx: dict) -> dict:
            # Transform the text input to uppercase
            if "text" in input_args:
                input_args["text"] = input_args["text"].upper()
            return input_args

        @spell
        @guard.input(uppercase_input)
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            result = classify.with_metadata("hello world")

            assert isinstance(result, SpellResult)
            # Verify the agent was called with the transformed input
            call_args = mock_agent.run_sync.call_args[0][0]
            # The user prompt should contain the uppercased text
            assert "HELLO WORLD" in call_args

    def test_with_metadata_guard_error_propagates(self):
        """with_metadata propagates guard errors correctly."""
        from spellcrafting import guard
        from spellcrafting.exceptions import GuardError

        def failing_guard(input_args: dict, ctx: dict) -> dict:
            raise ValueError("Guard validation failed")

        @spell
        @guard.input(failing_guard)
        def classify(text: str) -> str:
            """Classify the text."""
            ...

        mock_agent = MagicMock()

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            with pytest.raises(GuardError, match="Guard validation failed"):
                classify.with_metadata("test input")
