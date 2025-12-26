"""Integration tests combining multiple features (#112).

These tests verify that complex interactions between features work correctly:
- @spell with model alias
- @guard.input and @guard.output
- on_fail strategy
- Logging enabled
- with_metadata call
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from spellcrafting import spell, guard, OnFail, Config, ModelConfig
from spellcrafting import configure_logging, LoggingConfig


class Result(BaseModel):
    value: str
    processed: bool


class TestFullPipelineIntegration:
    """Tests combining all features in a single spell execution."""

    def test_spell_with_guards_and_logging(self):
        """Test spell with input/output guards and logging enabled."""
        handler = MagicMock()
        configure_logging(LoggingConfig(enabled=True, handlers=[handler]))

        call_order = []

        def input_guard(args, ctx):
            call_order.append("input_guard")
            args["text"] = args["text"].upper()
            return args

        def output_guard(out, ctx):
            call_order.append("output_guard")
            return out + " PROCESSED"

        @spell
        @guard.input(input_guard)
        @guard.output(output_guard)
        def process(text: str) -> str:
            """Process the text."""
            ...

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_result.usage.return_value = MagicMock(request_tokens=100, response_tokens=50)
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            result = process("hello")

        # Verify execution order
        assert call_order == ["input_guard", "output_guard"]

        # Verify output transformation
        assert result == "result PROCESSED"

        # Verify logging captured everything
        handler.handle.assert_called_once()
        log = handler.handle.call_args[0][0]
        assert log.success is True
        assert log.validation is not None
        assert "input_guard" in log.validation.input_guards_passed
        assert "output_guard" in log.validation.output_guards_passed

    def test_spell_with_model_alias_guards_and_logging(self):
        """Test spell with model alias, guards, and logging."""
        handler = MagicMock()
        configure_logging(LoggingConfig(enabled=True, handlers=[handler]))

        config = Config(models={
            "fast": ModelConfig(model="test:fast-model")
        })

        def validate_input(args, ctx):
            if len(args.get("text", "")) < 3:
                raise ValueError("Input too short")
            return args

        @spell(model="fast")
        @guard.input(validate_input)
        def summarize(text: str) -> str:
            """Summarize."""
            ...

        mock_result = MagicMock()
        mock_result.output = "summary"
        mock_result.usage.return_value = MagicMock(request_tokens=50, response_tokens=25)
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with config:
            with patch("spellcrafting.spell.Agent", return_value=mock_agent):
                result = summarize("hello world")

        assert result == "summary"

        log = handler.handle.call_args[0][0]
        assert log.model == "test:fast-model"
        assert log.model_alias == "fast"
        assert "validate_input" in log.validation.input_guards_passed

    def test_spell_with_fallback_strategy_and_logging(self):
        """Test spell with on_fail=fallback and logging."""
        handler = MagicMock()
        configure_logging(LoggingConfig(enabled=True, handlers=[handler]))

        from spellcrafting._pydantic_ai import UnexpectedModelBehavior

        @spell(on_fail=OnFail.fallback("default result"))
        def risky_spell(text: str) -> str:
            """Risky operation."""
            ...

        mock_agent = MagicMock()
        mock_agent.run_sync.side_effect = UnexpectedModelBehavior("Model failed")

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            result = risky_spell("test")

        # Should return fallback value
        assert result == "default result"

        # Logging should capture the fallback
        log = handler.handle.call_args[0][0]
        assert log.validation is not None
        assert log.validation.on_fail_triggered == "fallback"

    def test_spell_with_structured_output_guards_logging(self):
        """Test spell with Pydantic output type, guards, and logging."""
        handler = MagicMock()
        configure_logging(LoggingConfig(enabled=True, handlers=[handler]))

        def check_confidence(output: Result, ctx) -> Result:
            if output.confidence < 0.5:
                raise ValueError("Confidence too low")
            return output

        # Can't use check_confidence since Result doesn't have confidence
        # Use a simpler guard
        def mark_processed(output: Result, ctx) -> Result:
            return Result(value=output.value, processed=True)

        @spell
        @guard.output(mark_processed)
        def analyze(text: str) -> Result:
            """Analyze text."""
            ...

        mock_result = MagicMock()
        mock_result.output = Result(value="analysis", processed=False)
        mock_result.usage.return_value = MagicMock(request_tokens=100, response_tokens=50)
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            result = analyze("test input")

        assert isinstance(result, Result)
        assert result.value == "analysis"
        assert result.processed is True

        log = handler.handle.call_args[0][0]
        assert "mark_processed" in log.validation.output_guards_passed

    def test_spell_with_metadata_and_guards(self):
        """Test with_metadata combined with guards."""
        def add_prefix(args, ctx):
            args["text"] = "PREFIX: " + args["text"]
            return args

        @spell(model="openai:gpt-4o")
        @guard.input(add_prefix)
        def process(text: str) -> str:
            """Process."""
            ...

        mock_result = MagicMock()
        mock_result.output = "processed"
        mock_result.usage.return_value = MagicMock(request_tokens=100, response_tokens=50)
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            result = process.with_metadata("test")

        # Verify SpellResult
        assert result.output == "processed"
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.model_used == "openai:gpt-4o"

        # Verify guard was applied - PREFIX should be in the prompt
        call_args = mock_agent.run_sync.call_args[0][0]
        assert "PREFIX: test" in call_args

    @pytest.mark.asyncio
    async def test_async_spell_full_pipeline(self):
        """Test async spell with all features combined."""
        handler = MagicMock()
        configure_logging(LoggingConfig(enabled=True, handlers=[handler]))

        config = Config(models={
            "async_model": ModelConfig(model="test:async")
        })

        async def async_input_guard(args, ctx):
            args["text"] = args["text"].lower()
            return args

        def sync_output_guard(out, ctx):
            return out.strip()

        @spell(model="async_model")
        @guard.input(async_input_guard)
        @guard.output(sync_output_guard)
        async def async_process(text: str) -> str:
            """Async process."""
            ...

        mock_result = MagicMock()
        mock_result.output = "  async result  "
        mock_result.usage.return_value = MagicMock(request_tokens=75, response_tokens=30)
        mock_agent = MagicMock()

        async def mock_run(prompt):
            return mock_result
        mock_agent.run = mock_run

        with config:
            with patch("spellcrafting.spell.Agent", return_value=mock_agent):
                result = await async_process("HELLO")

        # Output should be transformed by guards
        assert result == "async result"

        # Logging should capture async execution
        log = handler.handle.call_args[0][0]
        assert log.model == "test:async"
        assert log.model_alias == "async_model"
        assert "async_input_guard" in log.validation.input_guards_passed
        assert "sync_output_guard" in log.validation.output_guards_passed

    def test_multiple_input_output_guards_with_logging(self):
        """Test spell with multiple input and output guards."""
        handler = MagicMock()
        configure_logging(LoggingConfig(enabled=True, handlers=[handler]))

        transform_order = []

        def input_guard_1(args, ctx):
            transform_order.append("in1")
            args["text"] = f"[IN1]{args['text']}"
            return args

        def input_guard_2(args, ctx):
            transform_order.append("in2")
            args["text"] = f"[IN2]{args['text']}"
            return args

        def output_guard_1(out, ctx):
            transform_order.append("out1")
            return f"[OUT1]{out}"

        def output_guard_2(out, ctx):
            transform_order.append("out2")
            return f"[OUT2]{out}"

        @spell
        @guard.input(input_guard_1)
        @guard.input(input_guard_2)
        @guard.output(output_guard_1)
        @guard.output(output_guard_2)
        def multi_guard_spell(text: str) -> str:
            """Multi-guard spell."""
            ...

        mock_result = MagicMock()
        mock_result.output = "CORE"
        mock_result.usage.return_value = MagicMock(request_tokens=10, response_tokens=5)
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            result = multi_guard_spell("hello")

        # Verify transform order
        # Input guards: in1 runs first (outermost), then in2
        # Output guards: out2 runs first (innermost applied first in decorator order),
        # then out1
        assert transform_order == ["in1", "in2", "out2", "out1"]

        # Verify final output
        assert result == "[OUT1][OUT2]CORE"

        # Check logging captured all guards
        log = handler.handle.call_args[0][0]
        assert len(log.validation.input_guards_passed) == 2
        assert len(log.validation.output_guards_passed) == 2

    def test_guard_failure_raises_correctly(self):
        """Test that guard failure raises the appropriate exception."""

        def failing_guard(args, ctx):
            raise ValueError("Guard validation failed!")

        @spell
        @guard.input(failing_guard)
        def guarded_spell(text: str) -> str:
            """Guarded spell."""
            ...

        mock_agent = MagicMock()

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            with pytest.raises(Exception, match="Guard validation failed"):
                guarded_spell("test")

        # Agent should not be called since guard failed
        mock_agent.run_sync.assert_not_called()

    def test_cost_estimate_with_all_features(self):
        """Test cost estimation works with all features enabled."""
        handler = MagicMock()
        configure_logging(LoggingConfig(enabled=True, handlers=[handler]))

        def passthrough(args, ctx):
            return args

        @spell(model="openai:gpt-4o")
        @guard.input(passthrough)
        def priced_spell(text: str) -> str:
            """Priced spell."""
            ...

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_result.usage.return_value = MagicMock(request_tokens=1000, response_tokens=500)
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            result = priced_spell.with_metadata("test")

        # Cost estimate should be populated
        assert result.cost_estimate is not None
        # gpt-4o: $2.50/1M input, $10.00/1M output
        # 1000 input = $0.0025, 500 output = $0.005
        expected_cost = (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.00
        assert abs(result.cost_estimate - expected_cost) < 0.0001


class TestEdgeCaseIntegration:
    """Test edge cases when features are combined."""

    def test_empty_input_with_guards(self):
        """Test handling of empty input with guards enabled."""
        handler = MagicMock()
        configure_logging(LoggingConfig(enabled=True, handlers=[handler]))

        def check_not_empty(args, ctx):
            if not args.get("text"):
                raise ValueError("Input cannot be empty")
            return args

        @spell
        @guard.input(check_not_empty)
        def non_empty_spell(text: str) -> str:
            """Requires non-empty input."""
            ...

        mock_agent = MagicMock()

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            with pytest.raises(Exception, match="Input cannot be empty"):
                non_empty_spell("")

        # Agent should not be called
        mock_agent.run_sync.assert_not_called()

    def test_logging_disabled_with_all_features(self):
        """Test that everything works when logging is disabled."""
        configure_logging(LoggingConfig(enabled=False))

        config = Config(models={
            "test": ModelConfig(model="test:model")
        })

        def input_guard(args, ctx):
            return args

        def output_guard(out, ctx):
            return out

        @spell(model="test")
        @guard.input(input_guard)
        @guard.output(output_guard)
        def no_logging_spell(text: str) -> str:
            """No logging spell."""
            ...

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with config:
            with patch("spellcrafting.spell.Agent", return_value=mock_agent):
                result = no_logging_spell("test")

        assert result == "result"

    def test_trace_context_with_guards_and_metadata(self):
        """Test trace context propagation with guards and with_metadata."""
        from spellcrafting.logging import trace_context

        handler = MagicMock()
        configure_logging(LoggingConfig(enabled=True, handlers=[handler]))

        captured_ctx = {}

        def capture_context_guard(args, ctx):
            captured_ctx.update(ctx)
            return args

        @spell(model="openai:gpt-4o")
        @guard.input(capture_context_guard)
        def traced_spell(text: str) -> str:
            """Traced spell."""
            ...

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_result.usage.return_value = MagicMock(request_tokens=50, response_tokens=25)
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            with trace_context() as ctx:
                result = traced_spell.with_metadata("test")

        # with_metadata should capture trace_id
        assert result.trace_id == ctx.trace_id

        # Guard context should have spell metadata
        assert captured_ctx["spell_name"] == "traced_spell"
        assert captured_ctx["model"] == "openai:gpt-4o"
