"""Tests for the logging module."""

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import spellcrafting.logging as logging_module
from spellcrafting import (
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
    ValidationMetrics,
    configure_logging,
    current_trace,
    get_logging_config,
    guard,
    setup_logging,
    spell,
    trace_context,
    with_trace_id,
)
from spellcrafting.config import Config, ModelConfig


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

    def test_flush_empty_buffer(self):
        """Flushing empty buffer should be safe and not create file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            handler = JSONFileHandler(path)

            handler.flush()  # Should not raise
            # File should not be created for empty flush
            assert not path.exists()

    def test_flush_empty_buffer_multiple_times(self):
        """Multiple flushes on empty buffer should be safe."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            handler = JSONFileHandler(path)

            # Multiple empty flushes should all succeed
            for _ in range(5):
                handler.flush()

            assert not path.exists()

    def test_handler_creates_parent_directories(self):
        """Handler should create parent directories on flush."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Nested path that doesn't exist
            path = Path(tmpdir) / "deep" / "nested" / "dir" / "test.jsonl"
            handler = JSONFileHandler(path)

            log = SpellExecutionLog(
                spell_name="test",
                spell_id=1,
                trace_id="abc",
                span_id="def",
            )
            log.finalize(success=True)
            handler.handle(log)
            handler.flush()

            assert path.exists()
            assert path.parent.exists()

    def test_handler_appends_to_existing_file(self):
        """Handler should append to existing file, not overwrite."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            handler1 = JSONFileHandler(path, buffer_size=10)
            handler2 = JSONFileHandler(path, buffer_size=10)

            # Write with first handler
            log1 = SpellExecutionLog(
                spell_name="first",
                spell_id=1,
                trace_id="abc",
                span_id="def",
            )
            log1.finalize(success=True)
            handler1.handle(log1)
            handler1.flush()

            # Write with second handler (same file)
            log2 = SpellExecutionLog(
                spell_name="second",
                spell_id=2,
                trace_id="xyz",
                span_id="uvw",
            )
            log2.finalize(success=True)
            handler2.handle(log2)
            handler2.flush()

            # Both logs should be in the file
            lines = path.read_text().strip().split("\n")
            assert len(lines) == 2
            assert "first" in lines[0]
            assert "second" in lines[1]

    def test_buffer_cleared_after_flush(self):
        """Buffer should be empty after successful flush."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            handler = JSONFileHandler(path, buffer_size=10)

            for i in range(5):
                log = SpellExecutionLog(
                    spell_name=f"test{i}",
                    spell_id=i,
                    trace_id="abc",
                    span_id="def",
                )
                log.finalize(success=True)
                handler.handle(log)

            assert len(handler.buffer) == 5
            handler.flush()
            assert len(handler.buffer) == 0


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
        from spellcrafting.logging import estimate_cost

        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        cost = estimate_cost("claude-sonnet-4-20250514", usage)

        assert cost is not None
        assert cost.input_cost > 0
        assert cost.output_cost > 0
        assert cost.total_cost == cost.input_cost + cost.output_cost

    def test_estimate_with_provider_prefix(self):
        from spellcrafting.logging import estimate_cost

        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        cost = estimate_cost("anthropic:claude-sonnet-4-20250514", usage)

        assert cost is not None
        assert cost.model == "claude-sonnet-4-20250514"

    def test_estimate_unknown_model(self):
        from spellcrafting.logging import estimate_cost

        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        cost = estimate_cost("unknown:model", usage)

        assert cost is None


class TestLogEmission:
    """Tests for log emission."""

    def test_emit_log_when_disabled(self):
        from spellcrafting.logging import _emit_log

        log = SpellExecutionLog(
            spell_name="test",
            spell_id=1,
            trace_id="abc",
            span_id="def",
        )

        # Should not raise, even when disabled
        _emit_log(log)

    def test_emit_log_with_redaction(self):
        from spellcrafting.logging import _emit_log

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
        from spellcrafting.logging import _emit_log

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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            with patch("spellcrafting.spell.trace_context") as mock_trace:
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
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
            with patch("spellcrafting.spell.Agent", return_value=mock_agent):
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
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
[tool.spellcrafting.logging]
enabled = true
level = "debug"
redact_content = true

[tool.spellcrafting.logging.default_tags]
env = "test"

[tool.spellcrafting.logging.handlers.python]
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
[tool.spellcrafting.logging]
enabled = true

[tool.spellcrafting.logging.handlers.json]
type = "json_file"
path = "logs/spellcrafting.jsonl"
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
[tool.spellcrafting.logging]
enabled = false
""")
        monkeypatch.chdir(tmp_path)

        logging_module._file_logging_config_cache = None

        config = get_logging_config()
        assert config.enabled is False
        assert len(config.handlers) == 0

    def test_pyproject_unknown_handler_type(self, tmp_path, monkeypatch):
        """Test that unknown handler types are silently skipped."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('''
[tool.spellcrafting.logging]
enabled = true

[tool.spellcrafting.logging.handlers.custom]
type = "unknown_type"
''')
        monkeypatch.chdir(tmp_path)

        logging_module._file_logging_config_cache = None

        config = get_logging_config()
        assert config.enabled is True
        # Unknown handler type is skipped, but default Python handler is added
        # since logging is enabled and no valid handlers were specified
        handler_types = [type(h).__name__ for h in config.handlers]
        assert "PythonLoggingHandler" in handler_types

    def test_pyproject_json_handler_missing_path(self, tmp_path, monkeypatch):
        """Test that json_file handler without path is skipped."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('''
[tool.spellcrafting.logging]
enabled = true

[tool.spellcrafting.logging.handlers.json]
type = "json_file"
# Missing path - should be skipped
''')
        monkeypatch.chdir(tmp_path)

        logging_module._file_logging_config_cache = None

        config = get_logging_config()
        assert config.enabled is True
        # JSON handler skipped due to missing path, default Python handler added
        handler_types = [type(h).__name__ for h in config.handlers]
        assert "JSONFileHandler" not in handler_types
        assert "PythonLoggingHandler" in handler_types

    def test_pyproject_multiple_python_handlers(self, tmp_path, monkeypatch):
        """Test multiple handlers of same type."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('''
[tool.spellcrafting.logging]
enabled = true

[tool.spellcrafting.logging.handlers.python1]
type = "python"
logger_name = "logger1"

[tool.spellcrafting.logging.handlers.python2]
type = "python"
logger_name = "logger2"
''')
        monkeypatch.chdir(tmp_path)

        logging_module._file_logging_config_cache = None

        config = get_logging_config()
        assert config.enabled is True
        # Both Python handlers should be created
        python_handlers = [h for h in config.handlers if isinstance(h, PythonLoggingHandler)]
        assert len(python_handlers) == 2
        # Verify different logger names
        logger_names = {h.logger.name for h in python_handlers}
        assert logger_names == {"logger1", "logger2"}

    def test_pyproject_handler_with_extra_fields(self, tmp_path, monkeypatch):
        """Test handler config with extra unknown fields is still processed."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('''
[tool.spellcrafting.logging]
enabled = true

[tool.spellcrafting.logging.handlers.python]
type = "python"
logger_name = "test_logger"
unknown_field = "ignored"
another_unknown = 123
''')
        monkeypatch.chdir(tmp_path)

        logging_module._file_logging_config_cache = None

        config = get_logging_config()
        assert config.enabled is True
        # Handler should still be created despite extra fields
        handler_types = [type(h).__name__ for h in config.handlers]
        assert "PythonLoggingHandler" in handler_types

    def test_pyproject_handler_not_dict(self, tmp_path, monkeypatch):
        """Test handler config that is not a dict is skipped."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('''
[tool.spellcrafting.logging]
enabled = true

[tool.spellcrafting.logging.handlers]
invalid = "not a dict"
''')
        monkeypatch.chdir(tmp_path)

        logging_module._file_logging_config_cache = None

        config = get_logging_config()
        assert config.enabled is True
        # Invalid handler skipped, default handler added
        handler_types = [type(h).__name__ for h in config.handlers]
        assert "PythonLoggingHandler" in handler_types

    def test_pyproject_otel_handler(self, tmp_path, monkeypatch):
        """Test OpenTelemetry handler can be configured via pyproject."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('''
[tool.spellcrafting.logging]
enabled = true

[tool.spellcrafting.logging.handlers.otel]
type = "otel"
''')
        monkeypatch.chdir(tmp_path)

        logging_module._file_logging_config_cache = None

        config = get_logging_config()
        assert config.enabled is True
        handler_types = [type(h).__name__ for h in config.handlers]
        assert "OpenTelemetryHandler" in handler_types

    def test_pyproject_opentelemetry_handler_alias(self, tmp_path, monkeypatch):
        """Test OpenTelemetry handler with 'opentelemetry' type alias."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('''
[tool.spellcrafting.logging]
enabled = true

[tool.spellcrafting.logging.handlers.tracing]
type = "opentelemetry"
''')
        monkeypatch.chdir(tmp_path)

        logging_module._file_logging_config_cache = None

        config = get_logging_config()
        assert config.enabled is True
        handler_types = [type(h).__name__ for h in config.handlers]
        assert "OpenTelemetryHandler" in handler_types


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

    def test_otel_handler_survives_tracer_error(self):
        """Handler should not fail if tracer raises."""
        handler = OpenTelemetryHandler()

        log = SpellExecutionLog(
            spell_name="test",
            spell_id=1,
            trace_id="abc",
            span_id="def",
        )
        log.finalize(success=True)

        # Mock tracer to raise
        with patch.object(handler, '_tracer', create=True) as mock_tracer:
            mock_tracer.start_as_current_span.side_effect = Exception("Tracer error")
            handler._available = True  # Force handler to try using tracer

            # Should not raise - handler catches all exceptions
            handler.handle(log)

    def test_otel_handler_with_none_tracer(self):
        """Handler should handle None tracer gracefully."""
        handler = OpenTelemetryHandler()
        handler._tracer = None
        handler._available = True  # Simulate partial initialization

        log = SpellExecutionLog(
            spell_name="test",
            spell_id=1,
            trace_id="abc",
            span_id="def",
        )
        log.finalize(success=True)

        # Should not raise - handler checks for None tracer
        handler.handle(log)

    def test_otel_handler_handles_import_error(self):
        """Handler should handle ImportError during span creation."""
        handler = OpenTelemetryHandler()

        log = SpellExecutionLog(
            spell_name="test",
            spell_id=1,
            trace_id="abc",
            span_id="def",
        )
        log.finalize(success=True)

        # Mock to simulate import error in handle path
        with patch.object(handler, '_available', True):
            with patch.object(handler, '_tracer') as mock_tracer:
                mock_tracer.start_as_current_span.side_effect = ImportError("otel not found")

                # Should not raise - handler catches all exceptions
                handler.handle(log)


class TestValidationMetrics:
    """Tests for ValidationMetrics dataclass."""

    def test_default_values(self):
        metrics = ValidationMetrics()
        assert metrics.attempt_count == 1
        assert metrics.retry_reasons == []
        assert metrics.input_guards_passed == []
        assert metrics.input_guards_failed == []
        assert metrics.output_guards_passed == []
        assert metrics.output_guards_failed == []
        assert metrics.pydantic_errors == []
        assert metrics.on_fail_triggered is None
        assert metrics.escalated_to_model is None

    def test_to_dict(self):
        metrics = ValidationMetrics(
            attempt_count=2,
            retry_reasons=["validation error"],
            input_guards_passed=["guard1"],
            output_guards_passed=["guard2"],
            pydantic_errors=["field error"],
            on_fail_triggered="escalate",
            escalated_to_model="openai:gpt-4o",
        )
        d = metrics.to_dict()

        assert d["attempt_count"] == 2
        assert d["retry_reasons"] == ["validation error"]
        assert d["input_guards_passed"] == ["guard1"]
        assert d["output_guards_passed"] == ["guard2"]
        assert d["pydantic_errors"] == ["field error"]
        assert d["on_fail_triggered"] == "escalate"
        assert d["escalated_to_model"] == "openai:gpt-4o"


class TestSpellExecutionLogWithValidation:
    """Tests for SpellExecutionLog with ValidationMetrics."""

    def test_log_with_validation_metrics(self):
        metrics = ValidationMetrics(
            input_guards_passed=["check_length"],
            on_fail_triggered="fallback",
        )
        log = SpellExecutionLog(
            spell_name="test",
            spell_id=1,
            trace_id="abc",
            span_id="def",
            validation=metrics,
        )
        log.finalize(success=True)

        d = log.to_dict()
        assert d["validation"] is not None
        assert d["validation"]["input_guards_passed"] == ["check_length"]
        assert d["validation"]["on_fail_triggered"] == "fallback"

    def test_log_without_validation_metrics(self):
        log = SpellExecutionLog(
            spell_name="test",
            spell_id=1,
            trace_id="abc",
            span_id="def",
        )
        log.finalize(success=True)

        d = log.to_dict()
        assert d["validation"] is None

    def test_validation_metrics_in_json(self):
        metrics = ValidationMetrics(
            attempt_count=3,
            pydantic_errors=["error1", "error2"],
        )
        log = SpellExecutionLog(
            spell_name="test",
            spell_id=1,
            trace_id="abc",
            span_id="def",
            validation=metrics,
        )
        log.finalize(success=True)

        json_str = log.to_json()
        parsed = json.loads(json_str)
        assert parsed["validation"]["attempt_count"] == 3
        assert parsed["validation"]["pydantic_errors"] == ["error1", "error2"]


class TestValidationMetricsIntegration:
    """Tests for ValidationMetrics integration with @spell decorator."""

    def test_logging_creates_validation_metrics(self):
        """When logging is enabled, spell should create validation metrics."""
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            test_spell("hello")

        handler.handle.assert_called_once()
        log = handler.handle.call_args[0][0]
        assert log.validation is not None
        assert isinstance(log.validation, ValidationMetrics)
        assert log.validation.attempt_count == 1

    def test_input_guards_tracked_in_validation_metrics(self):
        """Input guard results should be tracked in validation metrics."""
        handler = MagicMock()
        configure_logging(LoggingConfig(enabled=True, handlers=[handler]))

        def my_input_guard(input_args: dict, context: dict) -> dict:
            return input_args

        @spell
        @guard.input(my_input_guard)
        def test_spell(text: str) -> str:
            """Test spell."""
            ...

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_result.usage.return_value = MagicMock(request_tokens=100, response_tokens=50)
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            test_spell("hello")

        log = handler.handle.call_args[0][0]
        assert log.validation is not None
        assert "my_input_guard" in log.validation.input_guards_passed
        assert log.validation.input_guards_failed == []

    def test_output_guards_tracked_in_validation_metrics(self):
        """Output guard results should be tracked in validation metrics."""
        handler = MagicMock()
        configure_logging(LoggingConfig(enabled=True, handlers=[handler]))

        def my_output_guard(output: str, context: dict) -> str:
            return output

        @spell
        @guard.output(my_output_guard)
        def test_spell(text: str) -> str:
            """Test spell."""
            ...

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_result.usage.return_value = MagicMock(request_tokens=100, response_tokens=50)
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            test_spell("hello")

        log = handler.handle.call_args[0][0]
        assert log.validation is not None
        assert "my_output_guard" in log.validation.output_guards_passed
        assert log.validation.output_guards_failed == []

    def test_no_validation_metrics_when_logging_disabled(self):
        """When logging is disabled, no validation metrics overhead."""
        configure_logging(LoggingConfig(enabled=False))

        @spell
        def test_spell(text: str) -> str:
            """Test spell."""
            ...

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        # This should work without any issues (fast path)
        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            result = test_spell("hello")
            assert result == "result"

    @pytest.mark.asyncio
    async def test_async_spell_validation_metrics(self):
        """Async spells should also track validation metrics."""
        handler = MagicMock()
        configure_logging(LoggingConfig(enabled=True, handlers=[handler]))

        def my_guard(input_args: dict, context: dict) -> dict:
            return input_args

        @spell
        @guard.input(my_guard)
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

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            result = await test_async_spell("hello")

        assert result == "async result"
        log = handler.handle.call_args[0][0]
        assert log.validation is not None
        assert "my_guard" in log.validation.input_guards_passed


class TestToolCallLogging:
    """Tests for tool call logging integration with @spell decorator."""

    def test_tool_calls_logged_when_enabled(self):
        """Tool calls should be captured in the log when logging is enabled."""
        handler = MagicMock()
        configure_logging(LoggingConfig(enabled=True, handlers=[handler]))

        def my_tool(x: int) -> int:
            """Double a number."""
            return x * 2

        @spell(tools=[my_tool])
        def test_spell(text: str) -> str:
            """Test spell with tool."""
            ...

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_result.usage.return_value = MagicMock(request_tokens=100, response_tokens=50)

        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        # We need to capture the Agent instantiation to verify tools are wrapped
        captured_agents = []
        original_agent_class = sys.modules["spellcrafting.spell"].Agent

        def capture_agent(*args, **kwargs):
            agent = original_agent_class(*args, **kwargs)
            captured_agents.append(kwargs)
            return mock_agent

        with patch("spellcrafting.spell.Agent", side_effect=capture_agent):
            test_spell("hello")

        # Should have been called twice: once for cache (unwrapped), once for logging (wrapped)
        # Or just once if tools are present (wrapped version only when logging enabled)
        handler.handle.assert_called_once()
        log = handler.handle.call_args[0][0]
        # tool_calls list exists even if no tools were actually called by the LLM
        assert hasattr(log, "tool_calls")
        assert isinstance(log.tool_calls, list)

    def test_tool_call_success_logged(self):
        """Successful tool calls should be logged with arguments and result."""
        from spellcrafting.spell import _wrap_tool, SpellExecutionLog

        log = SpellExecutionLog(
            spell_name="test",
            spell_id=1,
            trace_id="abc",
            span_id="def",
        )

        def my_tool(x: int, y: int = 10) -> int:
            """Add two numbers."""
            return x + y

        wrapped = _wrap_tool(my_tool, log)
        result = wrapped(5, y=3)

        assert result == 8
        assert len(log.tool_calls) == 1
        tool_call = log.tool_calls[0]
        assert tool_call.tool_name == "my_tool"
        assert tool_call.arguments == {"x": 5, "y": 3}
        assert tool_call.result == 8
        assert tool_call.success is True
        assert tool_call.error is None
        assert tool_call.duration_ms >= 0

    def test_tool_call_failure_logged(self):
        """Failed tool calls should be logged with error."""
        from spellcrafting.spell import _wrap_tool, SpellExecutionLog

        log = SpellExecutionLog(
            spell_name="test",
            spell_id=1,
            trace_id="abc",
            span_id="def",
        )

        def failing_tool(x: int) -> int:
            """Always fails."""
            raise ValueError("intentional error")

        wrapped = _wrap_tool(failing_tool, log)

        with pytest.raises(ValueError, match="intentional error"):
            wrapped(42)

        assert len(log.tool_calls) == 1
        tool_call = log.tool_calls[0]
        assert tool_call.tool_name == "failing_tool"
        assert tool_call.arguments == {"x": 42}
        assert tool_call.success is False
        assert tool_call.error == "intentional error"
        assert tool_call.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_async_tool_call_logged(self):
        """Async tool calls should be logged correctly."""
        from spellcrafting.spell import _wrap_tool_async, SpellExecutionLog

        log = SpellExecutionLog(
            spell_name="test",
            spell_id=1,
            trace_id="abc",
            span_id="def",
        )

        async def async_tool(value: str) -> str:
            """Transform a string."""
            return value.upper()

        wrapped = _wrap_tool_async(async_tool, log)
        result = await wrapped("hello")

        assert result == "HELLO"
        assert len(log.tool_calls) == 1
        tool_call = log.tool_calls[0]
        assert tool_call.tool_name == "async_tool"
        assert tool_call.arguments == {"value": "hello"}
        assert tool_call.result == "HELLO"
        assert tool_call.success is True

    @pytest.mark.asyncio
    async def test_async_tool_call_failure_logged(self):
        """Failed async tool calls should be logged with error."""
        from spellcrafting.spell import _wrap_tool_async, SpellExecutionLog

        log = SpellExecutionLog(
            spell_name="test",
            spell_id=1,
            trace_id="abc",
            span_id="def",
        )

        async def failing_async_tool(x: int) -> int:
            """Always fails."""
            raise RuntimeError("async failure")

        wrapped = _wrap_tool_async(failing_async_tool, log)

        with pytest.raises(RuntimeError, match="async failure"):
            await wrapped(99)

        assert len(log.tool_calls) == 1
        tool_call = log.tool_calls[0]
        assert tool_call.tool_name == "failing_async_tool"
        assert tool_call.success is False
        assert tool_call.error == "async failure"

    def test_wrap_tools_for_logging(self):
        """_wrap_tools_for_logging should wrap all tools correctly."""
        from spellcrafting.spell import _wrap_tools_for_logging, SpellExecutionLog

        log = SpellExecutionLog(
            spell_name="test",
            spell_id=1,
            trace_id="abc",
            span_id="def",
        )

        def sync_tool(a: int) -> int:
            return a * 2

        async def async_tool(b: str) -> str:
            return b.lower()

        wrapped = _wrap_tools_for_logging([sync_tool, async_tool], log)

        assert len(wrapped) == 2
        # Verify sync tool works
        result1 = wrapped[0](5)
        assert result1 == 10
        assert len(log.tool_calls) == 1
        assert log.tool_calls[0].tool_name == "sync_tool"

    def test_multiple_tool_calls_logged(self):
        """Multiple tool calls in a single execution should all be logged."""
        from spellcrafting.spell import _wrap_tool, SpellExecutionLog

        log = SpellExecutionLog(
            spell_name="test",
            spell_id=1,
            trace_id="abc",
            span_id="def",
        )

        def tool_a(x: int) -> int:
            return x + 1

        def tool_b(x: int) -> int:
            return x * 2

        wrapped_a = _wrap_tool(tool_a, log)
        wrapped_b = _wrap_tool(tool_b, log)

        wrapped_a(1)
        wrapped_b(2)
        wrapped_a(3)

        assert len(log.tool_calls) == 3
        assert log.tool_calls[0].tool_name == "tool_a"
        assert log.tool_calls[0].result == 2
        assert log.tool_calls[1].tool_name == "tool_b"
        assert log.tool_calls[1].result == 4
        assert log.tool_calls[2].tool_name == "tool_a"
        assert log.tool_calls[2].result == 4

    def test_tool_logging_preserves_function_metadata(self):
        """Wrapped tools should preserve original function metadata."""
        from spellcrafting.spell import _wrap_tool, SpellExecutionLog

        log = SpellExecutionLog(
            spell_name="test",
            spell_id=1,
            trace_id="abc",
            span_id="def",
        )

        def documented_tool(x: int) -> int:
            """This tool has documentation."""
            return x

        wrapped = _wrap_tool(documented_tool, log)

        assert wrapped.__name__ == "documented_tool"
        assert wrapped.__doc__ == "This tool has documentation."

    def test_no_tools_no_overhead(self):
        """When there are no tools, regular agent should be used."""
        handler = MagicMock()
        configure_logging(LoggingConfig(enabled=True, handlers=[handler]))

        @spell
        def test_spell_no_tools(text: str) -> str:
            """Test spell without tools."""
            ...

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_result.usage.return_value = MagicMock(request_tokens=100, response_tokens=50)

        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            test_spell_no_tools("hello")

        handler.handle.assert_called_once()
        log = handler.handle.call_args[0][0]
        assert log.tool_calls == []

    def test_tool_call_to_dict(self):
        """ToolCallLog.to_dict should include key fields."""
        tool_log = ToolCallLog(
            tool_name="test_tool",
            arguments={"x": 1, "y": 2},
            result=3,
            duration_ms=10.5,
            success=True,
        )

        d = tool_log.to_dict()
        assert d["tool_name"] == "test_tool"
        assert d["duration_ms"] == 10.5
        assert d["success"] is True
        assert d["error"] is None
        assert d["redacted"] is False


class TestSetupHelpers:
    """Tests for setup_logfire and setup_datadog helpers (#164)."""

    def test_setup_logfire_enables_logging(self):
        """setup_logfire should enable logging with OTel handler."""
        from spellcrafting import setup_logfire

        setup_logfire()
        config = get_logging_config()

        assert config.enabled is True
        # Should have OTel handler
        handler_types = [type(h).__name__ for h in config.handlers]
        assert "OpenTelemetryHandler" in handler_types

    def test_setup_logfire_with_redaction(self):
        """setup_logfire should support redact_content parameter."""
        from spellcrafting import setup_logfire

        setup_logfire(redact_content=True)
        config = get_logging_config()

        assert config.enabled is True
        assert config.redact_content is True

    def test_setup_datadog_enables_logging(self):
        """setup_datadog should enable logging with OTel handler."""
        from spellcrafting import setup_datadog

        setup_datadog()
        config = get_logging_config()

        assert config.enabled is True
        # Should have OTel handler
        handler_types = [type(h).__name__ for h in config.handlers]
        assert "OpenTelemetryHandler" in handler_types

    def test_setup_datadog_with_redaction(self):
        """setup_datadog should support redact_content parameter."""
        from spellcrafting import setup_datadog

        setup_datadog(redact_content=True)
        config = get_logging_config()

        assert config.enabled is True
        assert config.redact_content is True

    def test_setup_helpers_have_docstrings(self):
        """Both setup helpers should have documentation."""
        from spellcrafting import setup_logfire, setup_datadog

        assert setup_logfire.__doc__ is not None
        assert setup_datadog.__doc__ is not None
        assert "logfire" in setup_logfire.__doc__.lower()
        assert "ddtrace" in setup_datadog.__doc__.lower()


class TestLogLevelEnum:
    """Tests for LogLevel enum usage (#155)."""

    def test_log_level_values(self):
        """All LogLevel values should be valid."""
        assert LogLevel.DEBUG.value == "debug"
        assert LogLevel.INFO.value == "info"
        assert LogLevel.WARNING.value == "warning"
        assert LogLevel.ERROR.value == "error"

    def test_log_level_is_string_enum(self):
        """LogLevel should be a string enum for easy serialization."""
        assert isinstance(LogLevel.DEBUG, str)
        assert isinstance(LogLevel.INFO, str)
        assert isinstance(LogLevel.WARNING, str)
        assert isinstance(LogLevel.ERROR, str)

    def test_log_level_comparison(self):
        """LogLevel values should be comparable as strings."""
        assert LogLevel.DEBUG == "debug"
        assert LogLevel.INFO == "info"
        assert LogLevel.WARNING == "warning"
        assert LogLevel.ERROR == "error"

    def test_logging_config_uses_log_level(self):
        """LoggingConfig should use LogLevel for level field."""
        config = LoggingConfig(level=LogLevel.DEBUG)
        assert config.level == LogLevel.DEBUG

        config = LoggingConfig(level=LogLevel.ERROR)
        assert config.level == LogLevel.ERROR

    def test_log_level_from_pyproject_string(self, tmp_path, monkeypatch):
        """LogLevel should be parsed correctly from pyproject.toml string values."""
        for level_str, expected_level in [
            ("debug", LogLevel.DEBUG),
            ("info", LogLevel.INFO),
            ("warning", LogLevel.WARNING),
            ("error", LogLevel.ERROR),
        ]:
            pyproject = tmp_path / "pyproject.toml"
            pyproject.write_text(f"""
[tool.spellcrafting.logging]
enabled = true
level = "{level_str}"
""")
            monkeypatch.chdir(tmp_path)
            logging_module._file_logging_config_cache = None

            config = get_logging_config()
            assert config.level == expected_level, f"Failed for level {level_str}"


class TestPricingDictCoverage:
    """Tests for PRICING dict coverage in cost estimation (#153)."""

    def test_all_pricing_models_have_required_keys(self):
        """All models in PRICING should have input and output keys."""
        from spellcrafting.logging import PRICING

        for model_name, prices in PRICING.items():
            assert "input" in prices, f"{model_name} missing input price"
            assert "output" in prices, f"{model_name} missing output price"
            assert isinstance(prices["input"], (int, float)), f"{model_name} input is not numeric"
            assert isinstance(prices["output"], (int, float)), f"{model_name} output is not numeric"
            # Allow 0.0 for free/experimental models (e.g., gemini-2.0-flash-exp)
            assert prices["input"] >= 0, f"{model_name} has negative input price"
            assert prices["output"] >= 0, f"{model_name} has negative output price"

    def test_cost_estimation_for_all_pricing_models(self):
        """Every model in PRICING should return valid cost estimate."""
        from spellcrafting.logging import PRICING, estimate_cost

        usage = TokenUsage(input_tokens=1000, output_tokens=500)

        for model_name in PRICING.keys():
            cost = estimate_cost(model_name, usage)

            assert cost is not None, f"{model_name} returned None cost"
            # Allow 0.0 for free/experimental models (e.g., gemini-2.0-flash-exp)
            assert cost.total_cost >= 0, f"{model_name} has negative total cost"
            assert cost.input_cost >= 0, f"{model_name} has negative input cost"
            assert cost.output_cost >= 0, f"{model_name} has negative output cost"
            assert cost.total_cost == cost.input_cost + cost.output_cost, f"{model_name} total != input + output"

    def test_anthropic_models_in_pricing(self):
        """Anthropic models should be in PRICING dict."""
        from spellcrafting.logging import PRICING

        anthropic_models = [m for m in PRICING.keys() if "claude" in m]
        assert len(anthropic_models) >= 4, "Expected at least 4 Claude models"

    def test_openai_models_in_pricing(self):
        """OpenAI models should be in PRICING dict."""
        from spellcrafting.logging import PRICING

        openai_models = [m for m in PRICING.keys() if "gpt" in m]
        assert len(openai_models) >= 4, "Expected at least 4 GPT models"

    def test_google_models_in_pricing(self):
        """Google models should be in PRICING dict."""
        from spellcrafting.logging import PRICING

        google_models = [m for m in PRICING.keys() if "gemini" in m]
        assert len(google_models) >= 2, "Expected at least 2 Gemini models"

    def test_pricing_values_are_reasonable(self):
        """Pricing values should be in reasonable ranges (per 1M tokens)."""
        from spellcrafting.logging import PRICING

        for model_name, prices in PRICING.items():
            # Input price should be between $0 (free/experimental) and $100 per 1M tokens
            assert 0 <= prices["input"] <= 100, f"{model_name} input price out of range"
            # Output price should be between $0 (free/experimental) and $150 per 1M tokens
            assert 0 <= prices["output"] <= 150, f"{model_name} output price out of range"
            # Output typically costs more than input (or equal)
            assert prices["output"] >= prices["input"], f"{model_name} output < input price"


class TestExtensiblePricing:
    """Tests for extensible pricing API (#149)."""

    def test_model_pricing_type(self):
        """ModelPricing TypedDict should have correct structure."""
        from spellcrafting import ModelPricing

        # Create a valid ModelPricing
        pricing: ModelPricing = {"input": 1.0, "output": 2.0}
        assert pricing["input"] == 1.0
        assert pricing["output"] == 2.0

    def test_register_model_pricing_new_model(self):
        """register_model_pricing should add new model pricing."""
        from spellcrafting import register_model_pricing, get_model_pricing
        from spellcrafting.logging import _custom_pricing

        # Register a new model
        register_model_pricing("my-custom-model-149", 5.0, 15.0)

        pricing = get_model_pricing("my-custom-model-149")
        assert pricing is not None
        assert pricing["input"] == 5.0
        assert pricing["output"] == 15.0

        # Clean up
        _custom_pricing.pop("my-custom-model-149", None)

    def test_register_model_pricing_override_default(self):
        """register_model_pricing should allow overriding default pricing."""
        from spellcrafting import register_model_pricing, get_model_pricing
        from spellcrafting.logging import _custom_pricing, _DEFAULT_PRICING

        # Get original pricing
        original = _DEFAULT_PRICING.get("gpt-4o")
        assert original is not None

        # Register override
        register_model_pricing("gpt-4o", 999.0, 1999.0)

        # Custom should take precedence
        pricing = get_model_pricing("gpt-4o")
        assert pricing is not None
        assert pricing["input"] == 999.0
        assert pricing["output"] == 1999.0

        # Clean up
        _custom_pricing.pop("gpt-4o", None)

        # Original should be restored
        pricing = get_model_pricing("gpt-4o")
        assert pricing is not None
        assert pricing["input"] == original["input"]
        assert pricing["output"] == original["output"]

    def test_get_model_pricing_with_provider_prefix(self):
        """get_model_pricing should strip provider prefix."""
        from spellcrafting import get_model_pricing

        pricing = get_model_pricing("anthropic:claude-sonnet-4-20250514")
        assert pricing is not None
        assert pricing["input"] == 3.0
        assert pricing["output"] == 15.0

    def test_get_model_pricing_unknown_model(self):
        """get_model_pricing should return None for unknown models."""
        from spellcrafting import get_model_pricing

        pricing = get_model_pricing("totally-unknown-model-xyz")
        assert pricing is None

    def test_get_model_pricing_default_models(self):
        """get_model_pricing should work for all default models."""
        from spellcrafting import get_model_pricing
        from spellcrafting.logging import PRICING

        for model_name in PRICING.keys():
            pricing = get_model_pricing(model_name)
            assert pricing is not None, f"{model_name} should have pricing"
            assert "input" in pricing
            assert "output" in pricing

    def test_estimate_cost_uses_custom_pricing(self):
        """estimate_cost should use custom pricing when registered."""
        from spellcrafting import register_model_pricing
        from spellcrafting.logging import estimate_cost, _custom_pricing

        # Register custom pricing for a new model
        register_model_pricing("my-test-model-149", 10.0, 20.0)

        usage = TokenUsage(input_tokens=1_000_000, output_tokens=500_000)
        cost = estimate_cost("my-test-model-149", usage)

        assert cost is not None
        assert cost.input_cost == 10.0  # 1M tokens * $10/1M
        assert cost.output_cost == 10.0  # 0.5M tokens * $20/1M
        assert cost.total_cost == 20.0

        # Clean up
        _custom_pricing.pop("my-test-model-149", None)

    def test_pricing_from_pyproject(self, tmp_path, monkeypatch):
        """Pricing should be loadable from pyproject.toml."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('''
[tool.spellcrafting.pricing."my-pyproject-model"]
input = 7.5
output = 22.5
''')
        monkeypatch.chdir(tmp_path)

        # Reset cache to force reload
        logging_module._file_pricing_cache = None

        from spellcrafting import get_model_pricing

        pricing = get_model_pricing("my-pyproject-model")
        assert pricing is not None
        assert pricing["input"] == 7.5
        assert pricing["output"] == 22.5

        # Reset cache after test
        logging_module._file_pricing_cache = None

    def test_pricing_priority_custom_over_file(self, tmp_path, monkeypatch):
        """Custom pricing should take precedence over file pricing."""
        from spellcrafting import register_model_pricing, get_model_pricing
        from spellcrafting.logging import _custom_pricing

        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('''
[tool.spellcrafting.pricing."priority-test-model"]
input = 1.0
output = 2.0
''')
        monkeypatch.chdir(tmp_path)
        logging_module._file_pricing_cache = None

        # Register custom pricing
        register_model_pricing("priority-test-model", 100.0, 200.0)

        pricing = get_model_pricing("priority-test-model")
        assert pricing is not None
        assert pricing["input"] == 100.0  # Custom takes precedence
        assert pricing["output"] == 200.0

        # Clean up
        _custom_pricing.pop("priority-test-model", None)
        logging_module._file_pricing_cache = None

    def test_pricing_priority_file_over_default(self, tmp_path, monkeypatch):
        """File pricing should take precedence over default pricing."""
        from spellcrafting import get_model_pricing

        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('''
[tool.spellcrafting.pricing."gpt-4o"]
input = 999.0
output = 1999.0
''')
        monkeypatch.chdir(tmp_path)
        logging_module._file_pricing_cache = None

        pricing = get_model_pricing("gpt-4o")
        assert pricing is not None
        assert pricing["input"] == 999.0  # File takes precedence over default
        assert pricing["output"] == 1999.0

        # Reset cache after test
        logging_module._file_pricing_cache = None

    def test_pricing_invalid_pyproject_values_skipped(self, tmp_path, monkeypatch):
        """Invalid pricing values in pyproject.toml should be skipped."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('''
[tool.spellcrafting.pricing."invalid-model"]
input = "not a number"
output = 2.0

[tool.spellcrafting.pricing."valid-model"]
input = 1.0
output = 2.0
''')
        monkeypatch.chdir(tmp_path)
        logging_module._file_pricing_cache = None

        from spellcrafting import get_model_pricing

        # Invalid model should be skipped
        pricing = get_model_pricing("invalid-model")
        assert pricing is None

        # Valid model should work
        pricing = get_model_pricing("valid-model")
        assert pricing is not None
        assert pricing["input"] == 1.0
        assert pricing["output"] == 2.0

        logging_module._file_pricing_cache = None

    def test_pricing_exported_from_package(self):
        """ModelPricing, register_model_pricing, get_model_pricing should be exported."""
        from spellcrafting import ModelPricing, register_model_pricing, get_model_pricing

        assert ModelPricing is not None
        assert callable(register_model_pricing)
        assert callable(get_model_pricing)

    def test_backwards_compatibility_pricing_dict(self):
        """PRICING dict should still be accessible for backwards compatibility."""
        from spellcrafting.logging import PRICING

        assert isinstance(PRICING, dict)
        assert "gpt-4o" in PRICING
        assert "claude-sonnet-4-20250514" in PRICING


class TestEmitLogHandlerErrors:
    """Tests for _emit_log handler error handling (#129).

    The _emit_log function silently catches all handler exceptions to prevent
    logging from breaking spell execution. These tests verify that behavior.
    """

    def test_emit_log_survives_handler_exception(self):
        """Handler exceptions should not break spell execution."""
        from spellcrafting.logging import _emit_log

        failing_handler = MagicMock()
        failing_handler.handle.side_effect = Exception("Handler error")

        working_handler = MagicMock()

        configure_logging(LoggingConfig(
            enabled=True,
            handlers=[failing_handler, working_handler],
        ))

        log = SpellExecutionLog(
            spell_name="test",
            spell_id=1,
            trace_id="abc",
            span_id="def",
        )

        # Should not raise
        _emit_log(log)

        # Failing handler was called
        failing_handler.handle.assert_called_once()
        # Working handler should still be called
        working_handler.handle.assert_called_once()

    def test_emit_log_all_handlers_fail(self):
        """Even if all handlers fail, no exception should propagate."""
        from spellcrafting.logging import _emit_log

        failing_handler1 = MagicMock()
        failing_handler1.handle.side_effect = Exception("Error 1")

        failing_handler2 = MagicMock()
        failing_handler2.handle.side_effect = Exception("Error 2")

        configure_logging(LoggingConfig(
            enabled=True,
            handlers=[failing_handler1, failing_handler2],
        ))

        log = SpellExecutionLog(
            spell_name="test",
            spell_id=1,
            trace_id="abc",
            span_id="def",
        )

        # Should not raise even when all handlers fail
        _emit_log(log)

        # Both handlers were called despite failures
        failing_handler1.handle.assert_called_once()
        failing_handler2.handle.assert_called_once()

    def test_emit_log_handler_exception_doesnt_affect_log_data(self):
        """Handler exception should not corrupt log data for subsequent handlers."""
        from spellcrafting.logging import _emit_log

        captured_logs = []

        failing_handler = MagicMock()
        failing_handler.handle.side_effect = Exception("Handler error")

        def capture_log(log):
            captured_logs.append(log)

        capturing_handler = MagicMock()
        capturing_handler.handle = capture_log

        configure_logging(LoggingConfig(
            enabled=True,
            handlers=[failing_handler, capturing_handler],
        ))

        log = SpellExecutionLog(
            spell_name="test_spell",
            spell_id=1,
            trace_id="abc123",
            span_id="def456",
        )

        _emit_log(log)

        # Log should be properly captured despite earlier handler failure
        assert len(captured_logs) == 1
        assert captured_logs[0].spell_name == "test_spell"
        assert captured_logs[0].trace_id == "abc123"

    def test_emit_log_handler_runtime_error(self):
        """RuntimeError from handler should be caught."""
        from spellcrafting.logging import _emit_log

        handler = MagicMock()
        handler.handle.side_effect = RuntimeError("Runtime error")

        configure_logging(LoggingConfig(enabled=True, handlers=[handler]))

        log = SpellExecutionLog(
            spell_name="test",
            spell_id=1,
            trace_id="abc",
            span_id="def",
        )

        # Should not raise
        _emit_log(log)
        handler.handle.assert_called_once()

    def test_emit_log_handler_type_error(self):
        """TypeError from handler should be caught."""
        from spellcrafting.logging import _emit_log

        handler = MagicMock()
        handler.handle.side_effect = TypeError("Type error")

        configure_logging(LoggingConfig(enabled=True, handlers=[handler]))

        log = SpellExecutionLog(
            spell_name="test",
            spell_id=1,
            trace_id="abc",
            span_id="def",
        )

        # Should not raise
        _emit_log(log)
        handler.handle.assert_called_once()


class TestExtractTokenUsageErrorHandling:
    """Tests for _extract_token_usage error handling in logging path (#124).

    The _extract_token_usage function catches AttributeError and TypeError.
    These tests verify that expected error types are handled gracefully.
    """

    def test_logging_handles_token_extraction_attribute_error(self):
        """When usage() raises AttributeError, logging works with zero tokens."""
        handler = MagicMock()
        configure_logging(LoggingConfig(enabled=True, handlers=[handler]))

        @spell
        def test_spell(text: str) -> str:
            """Test."""
            ...

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_result.usage.side_effect = AttributeError("No usage method")

        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            result = test_spell("hello")

        assert result == "result"
        log = handler.handle.call_args[0][0]
        assert log.token_usage.input_tokens == 0
        assert log.token_usage.output_tokens == 0

    def test_logging_handles_token_extraction_type_error(self):
        """When usage() raises TypeError, logging works with zero tokens."""
        handler = MagicMock()
        configure_logging(LoggingConfig(enabled=True, handlers=[handler]))

        @spell
        def test_spell(text: str) -> str:
            """Test."""
            ...

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_result.usage.side_effect = TypeError("Unexpected type")

        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            result = test_spell("hello")

        assert result == "result"
        log = handler.handle.call_args[0][0]
        assert log.token_usage.input_tokens == 0
        assert log.token_usage.output_tokens == 0

    def test_logging_handles_usage_returns_none(self):
        """When usage() returns unexpected None values, logging handles gracefully."""
        handler = MagicMock()
        configure_logging(LoggingConfig(enabled=True, handlers=[handler]))

        @spell
        def test_spell(text: str) -> str:
            """Test."""
            ...

        mock_usage = MagicMock()
        mock_usage.request_tokens = None
        mock_usage.response_tokens = None

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_result.usage.return_value = mock_usage

        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            result = test_spell("hello")

        assert result == "result"
        log = handler.handle.call_args[0][0]
        # None values should default to 0
        assert log.token_usage.input_tokens == 0
        assert log.token_usage.output_tokens == 0

    @pytest.mark.asyncio
    async def test_async_logging_handles_token_extraction_attribute_error(self):
        """Async: When usage() raises AttributeError, logging works with zero tokens."""
        handler = MagicMock()
        configure_logging(LoggingConfig(enabled=True, handlers=[handler]))

        @spell
        async def test_async_spell(text: str) -> str:
            """Test async."""
            ...

        mock_result = MagicMock()
        mock_result.output = "async result"
        mock_result.usage.side_effect = AttributeError("No usage method")

        mock_agent = MagicMock()

        async def mock_run(prompt):
            return mock_result
        mock_agent.run = mock_run

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            result = await test_async_spell("hello")

        assert result == "async result"
        log = handler.handle.call_args[0][0]
        assert log.token_usage.input_tokens == 0
        assert log.token_usage.output_tokens == 0

    @pytest.mark.asyncio
    async def test_async_logging_handles_token_extraction_type_error(self):
        """Async: When usage() raises TypeError, logging works with zero tokens."""
        handler = MagicMock()
        configure_logging(LoggingConfig(enabled=True, handlers=[handler]))

        @spell
        async def test_async_spell(text: str) -> str:
            """Test async."""
            ...

        mock_result = MagicMock()
        mock_result.output = "async result"
        mock_result.usage.side_effect = TypeError("Type error")

        mock_agent = MagicMock()

        async def mock_run(prompt):
            return mock_result
        mock_agent.run = mock_run

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            result = await test_async_spell("hello")

        assert result == "async result"
        log = handler.handle.call_args[0][0]
        assert log.token_usage.input_tokens == 0
        assert log.token_usage.output_tokens == 0


class TestNestedSpellTracing:
    """Tests for nested spell calls with trace propagation (#120).

    When a spell calls another spell, trace context should propagate:
    - Both spells should share the same trace_id
    - Each spell should have a unique span_id
    - Child spell should have parent_span_id pointing to outer spell's span_id
    """

    def test_nested_spells_share_trace_id(self):
        """Nested spell calls should share the same trace_id."""
        handler = MagicMock()
        configure_logging(LoggingConfig(enabled=True, handlers=[handler]))

        inner_called = []

        @spell
        def inner_spell(text: str) -> str:
            """Inner spell."""
            inner_called.append(True)
            ...

        @spell
        def outer_spell(text: str) -> str:
            """Outer spell that calls inner."""
            ...

        # Set up mocks
        inner_mock_result = MagicMock()
        inner_mock_result.output = "inner result"
        inner_mock_result.usage.return_value = MagicMock(request_tokens=10, response_tokens=5)

        outer_mock_result = MagicMock()
        outer_mock_result.output = "outer result"
        outer_mock_result.usage.return_value = MagicMock(request_tokens=20, response_tokens=10)

        call_count = [0]

        def create_mock_agent(*args, **kwargs):
            agent = MagicMock()
            # Outer spell calls first, then inner
            def run_sync(prompt):
                call_count[0] += 1
                if call_count[0] == 1:
                    # Outer spell - it will call inner
                    inner_spell("inner call")
                    return outer_mock_result
                else:
                    return inner_mock_result
            agent.run_sync = run_sync
            return agent

        with patch("spellcrafting.spell.Agent", side_effect=create_mock_agent):
            outer_spell("test")

        # Should have logged both spells
        assert handler.handle.call_count == 2
        logs = [call[0][0] for call in handler.handle.call_args_list]

        # Both should have the same trace_id
        trace_ids = {log.trace_id for log in logs}
        assert len(trace_ids) == 1, "Both spells should share the same trace_id"

        # Each should have unique span_id
        span_ids = {log.span_id for log in logs}
        assert len(span_ids) == 2, "Each spell should have unique span_id"

    def test_nested_spell_has_parent_span_id(self):
        """Inner spell should have parent_span_id pointing to outer spell's span_id."""
        handler = MagicMock()
        configure_logging(LoggingConfig(enabled=True, handlers=[handler]))

        @spell
        def inner_spell(text: str) -> str:
            """Inner spell."""
            ...

        @spell
        def outer_spell(text: str) -> str:
            """Outer spell."""
            ...

        call_count = [0]

        inner_mock_result = MagicMock()
        inner_mock_result.output = "inner"
        inner_mock_result.usage.return_value = MagicMock(request_tokens=10, response_tokens=5)

        outer_mock_result = MagicMock()
        outer_mock_result.output = "outer"
        outer_mock_result.usage.return_value = MagicMock(request_tokens=20, response_tokens=10)

        def create_mock_agent(*args, **kwargs):
            agent = MagicMock()
            def run_sync(prompt):
                call_count[0] += 1
                if call_count[0] == 1:
                    inner_spell("inner")
                    return outer_mock_result
                else:
                    return inner_mock_result
            agent.run_sync = run_sync
            return agent

        with patch("spellcrafting.spell.Agent", side_effect=create_mock_agent):
            outer_spell("test")

        assert handler.handle.call_count == 2
        logs = [call[0][0] for call in handler.handle.call_args_list]

        # Find inner and outer logs
        inner_log = next(log for log in logs if log.spell_name == "inner_spell")
        outer_log = next(log for log in logs if log.spell_name == "outer_spell")

        # Inner should have parent_span_id pointing to outer's span_id
        assert inner_log.parent_span_id == outer_log.span_id
        # Outer should have no parent (or None)
        # Note: The outer spell creates a new trace, so it won't have a parent

    def test_deeply_nested_spells_maintain_trace(self):
        """Deeply nested spell calls should all share the same trace_id."""
        handler = MagicMock()
        configure_logging(LoggingConfig(enabled=True, handlers=[handler]))

        @spell
        def level3_spell(text: str) -> str:
            """Level 3 (innermost)."""
            ...

        @spell
        def level2_spell(text: str) -> str:
            """Level 2."""
            ...

        @spell
        def level1_spell(text: str) -> str:
            """Level 1 (outermost)."""
            ...

        level_results = [
            MagicMock(output="level1", usage=MagicMock(return_value=MagicMock(request_tokens=10, response_tokens=5))),
            MagicMock(output="level2", usage=MagicMock(return_value=MagicMock(request_tokens=10, response_tokens=5))),
            MagicMock(output="level3", usage=MagicMock(return_value=MagicMock(request_tokens=10, response_tokens=5))),
        ]

        call_count = [0]

        def create_mock_agent(*args, **kwargs):
            agent = MagicMock()
            def run_sync(prompt):
                idx = call_count[0]
                call_count[0] += 1
                if idx == 0:
                    level2_spell("level2")
                    return level_results[0]
                elif idx == 1:
                    level3_spell("level3")
                    return level_results[1]
                else:
                    return level_results[2]
            agent.run_sync = run_sync
            return agent

        with patch("spellcrafting.spell.Agent", side_effect=create_mock_agent):
            level1_spell("test")

        assert handler.handle.call_count == 3
        logs = [call[0][0] for call in handler.handle.call_args_list]

        # All should share the same trace_id
        trace_ids = {log.trace_id for log in logs}
        assert len(trace_ids) == 1, "All nested spells should share trace_id"

        # All should have unique span_ids
        span_ids = {log.span_id for log in logs}
        assert len(span_ids) == 3, "Each spell should have unique span_id"
