"""Tests for the logging module."""

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import magically.logging as logging_module
from magically import (
    CostEstimate,
    JSONFileHandler,
    LoggingConfig,
    LogLevel,
    OpenTelemetryHandler,
    PythonLoggingHandler,
    SpellExecutionLog,
    TokenUsage,
    ToolCallLog,
    TraceContext,
    configure_logging,
    current_trace,
    get_logging_config,
    setup_logging,
    spell,
    trace_context,
    with_trace_id,
)
from magically.config import Config, ModelConfig


@pytest.fixture(autouse=True)
def reset_logging_config():
    """Reset logging config state between tests."""
    logging_module._process_logging_config = None
    logging_module._file_logging_config_cache = None
    logging_module._logging_config = logging_module.ContextVar("logging_config", default=None)
    yield
    logging_module._process_logging_config = None
    logging_module._file_logging_config_cache = None


@pytest.fixture(autouse=True)
def reset_trace_context():
    """Reset trace context between tests."""
    logging_module._trace_context = logging_module.ContextVar("trace", default=None)
    yield


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""

    def test_default_values(self):
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.cache_read_tokens == 0
        assert usage.cache_write_tokens == 0

    def test_total_tokens(self):
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.total_tokens == 150

    def test_to_dict(self):
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        d = usage.to_dict()
        assert d["input_tokens"] == 100
        assert d["output_tokens"] == 50
        assert d["cache_read_tokens"] == 0


class TestCostEstimate:
    """Tests for CostEstimate dataclass."""

    def test_default_values(self):
        cost = CostEstimate()
        assert cost.input_cost == 0.0
        assert cost.output_cost == 0.0
        assert cost.total_cost == 0.0
        assert cost.currency == "USD"

    def test_to_dict(self):
        cost = CostEstimate(input_cost=0.01, output_cost=0.02, total_cost=0.03)
        d = cost.to_dict()
        assert d["input_cost"] == 0.01
        assert d["output_cost"] == 0.02
        assert d["total_cost"] == 0.03
        assert d["currency"] == "USD"


class TestToolCallLog:
    """Tests for ToolCallLog dataclass."""

    def test_default_values(self):
        log = ToolCallLog(tool_name="test_tool")
        assert log.tool_name == "test_tool"
        assert log.success is True
        assert log.error is None
        assert log.duration_ms == 0.0

    def test_to_dict(self):
        log = ToolCallLog(
            tool_name="test_tool",
            duration_ms=150.5,
            success=True,
        )
        d = log.to_dict()
        assert d["tool_name"] == "test_tool"
        assert d["duration_ms"] == 150.5
        assert d["success"] is True


class TestSpellExecutionLog:
    """Tests for SpellExecutionLog dataclass."""

    def test_creation(self):
        log = SpellExecutionLog(
            spell_name="test_spell",
            spell_id=123,
            trace_id="abc123",
            span_id="def456",
        )
        assert log.spell_name == "test_spell"
        assert log.spell_id == 123
        assert log.trace_id == "abc123"
        assert log.success is True

    def test_finalize_success(self):
        log = SpellExecutionLog(
            spell_name="test",
            spell_id=1,
            trace_id="t",
            span_id="s",
        )
        log.finalize(success=True, output="result")
        assert log.success is True
        assert log.output == "result"
        assert log.output_type == "str"
        assert log.end_time is not None
        assert log.duration_ms is not None
        assert log.duration_ms >= 0

    def test_finalize_failure(self):
        log = SpellExecutionLog(
            spell_name="test",
            spell_id=1,
            trace_id="t",
            span_id="s",
        )
        error = ValueError("test error")
        log.finalize(success=False, error=error)
        assert log.success is False
        assert log.error == "test error"
        assert log.error_type == "ValueError"

    def test_to_dict(self):
        log = SpellExecutionLog(
            spell_name="test",
            spell_id=1,
            trace_id="abc",
            span_id="def",
            model="test:model",
            model_alias="fast",
        )
        log.finalize(success=True, output="result")
        d = log.to_dict()

        assert d["spell_name"] == "test"
        assert d["trace_id"] == "abc"
        assert d["model"] == "test:model"
        assert d["model_alias"] == "fast"
        assert d["success"] is True

    def test_to_json(self):
        log = SpellExecutionLog(
            spell_name="test",
            spell_id=1,
            trace_id="abc",
            span_id="def",
        )
        log.finalize(success=True)
        json_str = log.to_json()
        parsed = json.loads(json_str)
        assert parsed["spell_name"] == "test"


class TestTraceContext:
    """Tests for TraceContext."""

    def test_new_trace(self):
        ctx = TraceContext.new()
        assert len(ctx.trace_id) == 32
        assert len(ctx.span_id) == 16
        assert ctx.parent_span_id is None

    def test_child_trace(self):
        parent = TraceContext.new()
        child = TraceContext.new(parent)

        assert child.trace_id == parent.trace_id
        assert child.span_id != parent.span_id
        assert child.parent_span_id == parent.span_id

    def test_w3c_traceparent(self):
        ctx = TraceContext(
            trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
            span_id="00f067aa0ba902b7",
        )
        traceparent = ctx.to_w3c_traceparent()
        assert traceparent == "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


class TestTraceContextPropagation:
    """Tests for trace context propagation."""

    def test_no_current_trace_by_default(self):
        assert current_trace() is None

    def test_trace_context_manager(self):
        assert current_trace() is None

        with trace_context() as ctx:
            assert current_trace() is ctx
            assert len(ctx.trace_id) == 32
            assert len(ctx.span_id) == 16

        assert current_trace() is None

    def test_nested_trace_contexts(self):
        with trace_context() as outer:
            assert current_trace() == outer

            with trace_context() as inner:
                assert current_trace() == inner
                assert inner.trace_id == outer.trace_id
                assert inner.parent_span_id == outer.span_id

            assert current_trace() == outer

    def test_with_trace_id(self):
        external_trace_id = "external12345678901234567890ab"

        with with_trace_id(external_trace_id) as ctx:
            assert current_trace() == ctx
            assert ctx.trace_id == external_trace_id
            assert len(ctx.span_id) == 16

        assert current_trace() is None


class TestPythonLoggingHandler:
    """Tests for PythonLoggingHandler."""

    def test_handle_success(self):
        import logging

        handler = PythonLoggingHandler("test_logger")
        log = SpellExecutionLog(
            spell_name="test",
            spell_id=1,
            trace_id="abc",
            span_id="def",
        )
        log.finalize(success=True)

        with patch.object(handler.logger, "log") as mock_log:
            handler.handle(log)
            mock_log.assert_called_once()
            call_args = mock_log.call_args
            assert call_args[0][0] == logging.INFO

    def test_handle_failure(self):
        import logging

        handler = PythonLoggingHandler("test_logger")
        log = SpellExecutionLog(
            spell_name="test",
            spell_id=1,
            trace_id="abc",
            span_id="def",
        )
        log.finalize(success=False, error=ValueError("test"))

        with patch.object(handler.logger, "log") as mock_log:
            handler.handle(log)
            call_args = mock_log.call_args
            assert call_args[0][0] == logging.ERROR


class TestJSONFileHandler:
    """Tests for JSONFileHandler."""

    def test_handle_and_flush(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "logs" / "test.jsonl"
            handler = JSONFileHandler(path, buffer_size=10)

            log = SpellExecutionLog(
                spell_name="test",
                spell_id=1,
                trace_id="abc",
                span_id="def",
            )
            log.finalize(success=True)

            handler.handle(log)
            assert len(handler.buffer) == 1

            handler.flush()
            assert len(handler.buffer) == 0
            assert path.exists()

            content = path.read_text()
            parsed = json.loads(content.strip())
            assert parsed["spell_name"] == "test"

    def test_auto_flush_on_buffer_full(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            handler = JSONFileHandler(path, buffer_size=2)

            for i in range(3):
                log = SpellExecutionLog(
                    spell_name=f"test{i}",
                    spell_id=i,
                    trace_id="abc",
                    span_id="def",
                )
                log.finalize(success=True)
                handler.handle(log)

            # After 3 logs with buffer_size=2, should have flushed once
            # and have 1 log in buffer
            assert len(handler.buffer) == 1
            assert path.exists()


class TestLoggingConfig:
    """Tests for LoggingConfig."""

    def test_default_disabled(self):
        config = LoggingConfig()
        assert config.enabled is False
        assert config.level == LogLevel.INFO
        assert config.redact_content is False

    def test_configure_and_get(self):
        config = LoggingConfig(enabled=True, level=LogLevel.DEBUG)
        configure_logging(config)

        retrieved = get_logging_config()
        assert retrieved.enabled is True
        assert retrieved.level == LogLevel.DEBUG

    def test_default_config_when_not_configured(self):
        config = get_logging_config()
        assert config.enabled is False


class TestSetupLogging:
    """Tests for setup_logging helper."""

    def test_basic_setup(self):
        setup_logging(level=LogLevel.DEBUG)
        config = get_logging_config()

        assert config.enabled is True
        assert config.level == LogLevel.DEBUG
        assert len(config.handlers) >= 1
        assert isinstance(config.handlers[0], PythonLoggingHandler)

    def test_setup_with_json_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "test.jsonl"
            setup_logging(json_file=json_path)
            config = get_logging_config()

            assert config.enabled is True
            # Should have both Python and JSON handlers
            handler_types = [type(h) for h in config.handlers]
            assert PythonLoggingHandler in handler_types
            assert JSONFileHandler in handler_types

    def test_setup_with_redaction(self):
        setup_logging(redact_content=True)
        config = get_logging_config()

        assert config.enabled is True
        assert config.redact_content is True
        assert config.include_input is False
        assert config.include_output is False


class TestCostEstimation:
    """Tests for cost estimation."""

    def test_estimate_known_model(self):
        from magically.logging import estimate_cost

        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        cost = estimate_cost("claude-sonnet-4-20250514", usage)

        assert cost is not None
        assert cost.input_cost > 0
        assert cost.output_cost > 0
        assert cost.total_cost == cost.input_cost + cost.output_cost

    def test_estimate_with_provider_prefix(self):
        from magically.logging import estimate_cost

        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        cost = estimate_cost("anthropic:claude-sonnet-4-20250514", usage)

        assert cost is not None
        assert cost.model == "claude-sonnet-4-20250514"

    def test_estimate_unknown_model(self):
        from magically.logging import estimate_cost

        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        cost = estimate_cost("unknown:model", usage)

        assert cost is None


class TestLogEmission:
    """Tests for log emission."""

    def test_emit_log_when_disabled(self):
        from magically.logging import _emit_log

        log = SpellExecutionLog(
            spell_name="test",
            spell_id=1,
            trace_id="abc",
            span_id="def",
        )

        # Should not raise, even when disabled
        _emit_log(log)

    def test_emit_log_with_redaction(self):
        from magically.logging import _emit_log

        handler = MagicMock()
        configure_logging(LoggingConfig(
            enabled=True,
            handlers=[handler],
            redact_content=True,
        ))

        log = SpellExecutionLog(
            spell_name="test",
            spell_id=1,
            trace_id="abc",
            span_id="def",
            input_args={"secret": "password123"},
            output="sensitive data",
        )

        _emit_log(log)

        # Handler should have been called
        handler.handle.assert_called_once()
        emitted_log = handler.handle.call_args[0][0]
        assert emitted_log.input_args == "[REDACTED]"
        assert emitted_log.output == "[REDACTED]"

    def test_emit_log_with_default_tags(self):
        from magically.logging import _emit_log

        handler = MagicMock()
        configure_logging(LoggingConfig(
            enabled=True,
            handlers=[handler],
            default_tags={"env": "test", "version": "1.0"},
        ))

        log = SpellExecutionLog(
            spell_name="test",
            spell_id=1,
            trace_id="abc",
            span_id="def",
            tags={"custom": "tag"},
        )

        _emit_log(log)

        emitted_log = handler.handle.call_args[0][0]
        assert emitted_log.tags["env"] == "test"
        assert emitted_log.tags["version"] == "1.0"
        assert emitted_log.tags["custom"] == "tag"


class TestSpellLoggingIntegration:
    """Tests for logging integration with @spell decorator."""

    @pytest.fixture(autouse=True)
    def reset_spell_state(self):
        """Reset spell module state."""
        spell_module = sys.modules["magically.spell"]
        spell_module._agent_cache.clear()
        yield
        spell_module._agent_cache.clear()

    def test_no_logging_overhead_when_disabled(self):
        """When logging is disabled, spell should not create logs."""
        configure_logging(LoggingConfig(enabled=False))

        @spell
        def test_spell(text: str) -> str:
            """Test spell."""
            ...

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("magically.spell.Agent", return_value=mock_agent):
            with patch("magically.spell.trace_context") as mock_trace:
                result = test_spell("hello")
                # trace_context should not be called when logging disabled
                mock_trace.assert_not_called()

    def test_logging_creates_trace_context(self):
        """When logging is enabled, spell should create trace context."""
        handler = MagicMock()
        configure_logging(LoggingConfig(enabled=True, handlers=[handler]))

        @spell
        def test_spell(text: str) -> str:
            """Test spell."""
            ...

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_result.usage.return_value = MagicMock(
            request_tokens=100,
            response_tokens=50,
        )
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("magically.spell.Agent", return_value=mock_agent):
            result = test_spell("hello")

        # Handler should have been called with a log
        handler.handle.assert_called_once()
        log = handler.handle.call_args[0][0]
        assert log.spell_name == "test_spell"
        assert len(log.trace_id) == 32
        assert len(log.span_id) == 16
        assert log.success is True

    def test_logging_captures_token_usage(self):
        """Logging should capture token usage from result."""
        handler = MagicMock()
        configure_logging(LoggingConfig(enabled=True, handlers=[handler]))

        @spell
        def test_spell(text: str) -> str:
            """Test spell."""
            ...

        mock_usage = MagicMock()
        mock_usage.request_tokens = 150
        mock_usage.response_tokens = 75

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_result.usage.return_value = mock_usage

        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("magically.spell.Agent", return_value=mock_agent):
            test_spell("hello")

        log = handler.handle.call_args[0][0]
        assert log.token_usage.input_tokens == 150
        assert log.token_usage.output_tokens == 75

    def test_logging_captures_errors(self):
        """Logging should capture errors when spell fails."""
        handler = MagicMock()
        configure_logging(LoggingConfig(enabled=True, handlers=[handler]))

        @spell
        def test_spell(text: str) -> str:
            """Test spell."""
            ...

        mock_agent = MagicMock()
        mock_agent.run_sync.side_effect = ValueError("API error")

        with patch("magically.spell.Agent", return_value=mock_agent):
            with pytest.raises(ValueError, match="API error"):
                test_spell("hello")

        handler.handle.assert_called_once()
        log = handler.handle.call_args[0][0]
        assert log.success is False
        assert log.error == "API error"
        assert log.error_type == "ValueError"

    def test_logging_captures_input_args(self):
        """Logging should capture function input arguments."""
        handler = MagicMock()
        configure_logging(LoggingConfig(enabled=True, handlers=[handler]))

        @spell
        def test_spell(text: str, count: int = 5) -> str:
            """Test spell."""
            ...

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_result.usage.return_value = MagicMock(request_tokens=0, response_tokens=0)

        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("magically.spell.Agent", return_value=mock_agent):
            test_spell("hello", count=10)

        log = handler.handle.call_args[0][0]
        assert log.input_args["text"] == "hello"
        assert log.input_args["count"] == 10

    def test_logging_with_model_alias(self):
        """Logging should capture model and alias."""
        handler = MagicMock()
        configure_logging(LoggingConfig(enabled=True, handlers=[handler]))

        config = Config(models={
            "fast": ModelConfig(model="test:model")
        })

        @spell(model="fast")
        def test_spell(text: str) -> str:
            """Test spell."""
            ...

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_result.usage.return_value = MagicMock(request_tokens=0, response_tokens=0)

        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with config:
            with patch("magically.spell.Agent", return_value=mock_agent):
                test_spell("hello")

        log = handler.handle.call_args[0][0]
        assert log.model == "test:model"
        assert log.model_alias == "fast"

    @pytest.mark.asyncio
    async def test_async_spell_logging(self):
        """Async spells should also emit logs."""
        handler = MagicMock()
        configure_logging(LoggingConfig(enabled=True, handlers=[handler]))

        @spell
        async def test_async_spell(text: str) -> str:
            """Test async spell."""
            ...

        mock_result = MagicMock()
        mock_result.output = "async result"
        mock_result.usage.return_value = MagicMock(request_tokens=100, response_tokens=50)

        mock_agent = MagicMock()

        async def mock_run(prompt):
            return mock_result
        mock_agent.run = mock_run

        with patch("magically.spell.Agent", return_value=mock_agent):
            result = await test_async_spell("hello")

        assert result == "async result"
        handler.handle.assert_called_once()
        log = handler.handle.call_args[0][0]
        assert log.spell_name == "test_async_spell"
        assert log.success is True


class TestPyprojectLogging:
    """Tests for pyproject.toml logging configuration."""

    def test_load_from_pyproject(self, tmp_path, monkeypatch):
        """Test loading logging config from pyproject.toml."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("""
[tool.magically.logging]
enabled = true
level = "debug"
redact_content = true

[tool.magically.logging.default_tags]
env = "test"

[tool.magically.logging.handlers.python]
type = "python"
logger_name = "custom_logger"
""")
        monkeypatch.chdir(tmp_path)

        # Reset cache to force reload
        logging_module._file_logging_config_cache = None

        config = get_logging_config()
        assert config.enabled is True
        assert config.level == LogLevel.DEBUG
        assert config.redact_content is True
        assert config.default_tags.get("env") == "test"
        assert len(config.handlers) >= 1

    def test_pyproject_json_handler(self, tmp_path, monkeypatch):
        """Test loading JSON file handler from pyproject.toml."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("""
[tool.magically.logging]
enabled = true

[tool.magically.logging.handlers.json]
type = "json_file"
path = "logs/magically.jsonl"
""")
        monkeypatch.chdir(tmp_path)

        logging_module._file_logging_config_cache = None

        config = get_logging_config()
        assert config.enabled is True

        handler_types = [type(h).__name__ for h in config.handlers]
        assert "JSONFileHandler" in handler_types

    def test_pyproject_missing_section(self, tmp_path, monkeypatch):
        """Test handling when logging section is missing."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("""
[project]
name = "test"
""")
        monkeypatch.chdir(tmp_path)

        logging_module._file_logging_config_cache = None

        config = get_logging_config()
        assert config.enabled is False  # Default

    def test_pyproject_disabled_no_handlers(self, tmp_path, monkeypatch):
        """Test that disabled logging doesn't add handlers."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("""
[tool.magically.logging]
enabled = false
""")
        monkeypatch.chdir(tmp_path)

        logging_module._file_logging_config_cache = None

        config = get_logging_config()
        assert config.enabled is False
        assert len(config.handlers) == 0


class TestOpenTelemetryHandler:
    """Tests for OpenTelemetryHandler."""

    def test_graceful_degradation_without_otel(self):
        """Handler should not fail when opentelemetry is not installed."""
        handler = OpenTelemetryHandler()
        # Even if otel not available, handle should not raise
        log = SpellExecutionLog(
            spell_name="test",
            spell_id=1,
            trace_id="abc",
            span_id="def",
        )
        log.finalize(success=True)
        handler.handle(log)  # Should not raise
        handler.flush()  # Should not raise
