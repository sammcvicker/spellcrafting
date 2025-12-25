"""Structured logging, distributed tracing, and cost tracking for magically.

This module provides enterprise-grade observability with zero overhead when disabled.
"""

from __future__ import annotations

import json
import logging as stdlib_logging
import time
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4


class LogLevel(str, Enum):
    """Log levels for magically logging."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Core Types
# ---------------------------------------------------------------------------


@dataclass
class TokenUsage:
    """Token usage metrics from a spell execution."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
        }


@dataclass
class CostEstimate:
    """Estimated cost of a spell execution in USD."""

    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    currency: str = "USD"
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "total_cost": self.total_cost,
            "currency": self.currency,
        }


@dataclass
class ToolCallLog:
    """Log of a single tool invocation during spell execution."""

    tool_name: str
    arguments: dict[str, Any] | str = field(default_factory=dict)
    result: Any | str = None
    duration_ms: float = 0.0
    success: bool = True
    error: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class SpellExecutionLog:
    """Complete log of a spell execution."""

    # Identity
    spell_name: str
    spell_id: int
    trace_id: str
    span_id: str
    parent_span_id: str | None = None

    # Timing
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: datetime | None = None
    duration_ms: float | None = None

    # Model
    model: str = ""
    model_alias: str | None = None

    # I/O (redactable)
    input_args: dict[str, Any] | str = field(default_factory=dict)
    output: Any | str = None
    output_type: str = ""

    # Metrics
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    cost_estimate: CostEstimate | None = None

    # Tools
    tool_calls: list[ToolCallLog] = field(default_factory=list)

    # Status
    success: bool = True
    error: str | None = None
    error_type: str | None = None
    retry_count: int = 0

    # Custom
    tags: dict[str, str] = field(default_factory=dict)

    def finalize(
        self,
        *,
        success: bool,
        error: Exception | None = None,
        output: Any = None,
    ) -> None:
        """Finalize the log with completion status."""
        self.end_time = datetime.now(timezone.utc)
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.success = success
        if error is not None:
            self.error = str(error)
            self.error_type = type(error).__name__
        if output is not None:
            self.output = output
            self.output_type = type(output).__name__

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "spell_name": self.spell_name,
            "spell_id": self.spell_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "model": self.model,
            "model_alias": self.model_alias,
            "input_args": self.input_args if isinstance(self.input_args, str) else dict(self.input_args),
            "output": self.output if isinstance(self.output, str) else repr(self.output),
            "output_type": self.output_type,
            "token_usage": self.token_usage.to_dict(),
            "cost_estimate": self.cost_estimate.to_dict() if self.cost_estimate else None,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "success": self.success,
            "error": self.error,
            "error_type": self.error_type,
            "retry_count": self.retry_count,
            "tags": self.tags,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())


# ---------------------------------------------------------------------------
# Trace Context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TraceContext:
    """W3C Trace Context compatible correlation IDs."""

    trace_id: str  # 32 hex chars, shared across spans
    span_id: str  # 16 hex chars, unique per execution
    parent_span_id: str | None = None

    @classmethod
    def new(cls, parent: TraceContext | None = None) -> TraceContext:
        """Create a new trace context, optionally inheriting from parent."""
        return cls(
            trace_id=parent.trace_id if parent else uuid4().hex,
            span_id=uuid4().hex[:16],
            parent_span_id=parent.span_id if parent else None,
        )

    def to_w3c_traceparent(self) -> str:
        """Format as W3C traceparent header: 00-{trace_id}-{span_id}-01"""
        return f"00-{self.trace_id}-{self.span_id}-01"


_trace_context: ContextVar[TraceContext | None] = ContextVar("trace", default=None)


def current_trace() -> TraceContext | None:
    """Get the current trace context, if any."""
    return _trace_context.get()


@contextmanager
def trace_context(parent: TraceContext | None = None):
    """Context manager for trace context propagation.

    Creates a new span within the current trace, or starts a new trace
    if no parent is provided and no current trace exists.
    """
    current = parent or _trace_context.get()
    ctx = TraceContext.new(current)
    token = _trace_context.set(ctx)
    try:
        yield ctx
    finally:
        _trace_context.reset(token)


@contextmanager
def with_trace_id(trace_id: str):
    """Context manager to set a specific trace ID (for external correlation)."""
    ctx = TraceContext(
        trace_id=trace_id,
        span_id=uuid4().hex[:16],
        parent_span_id=None,
    )
    token = _trace_context.set(ctx)
    try:
        yield ctx
    finally:
        _trace_context.reset(token)


# ---------------------------------------------------------------------------
# Handler Protocol and Implementations
# ---------------------------------------------------------------------------


class LogHandler(Protocol):
    """Protocol for log handlers."""

    def handle(self, log: SpellExecutionLog) -> None:
        """Handle a spell execution log."""
        ...

    def flush(self) -> None:
        """Flush any buffered logs."""
        ...


class PythonLoggingHandler:
    """Handler that emits logs to Python's stdlib logging."""

    def __init__(self, logger_name: str = "magically"):
        self.logger = stdlib_logging.getLogger(logger_name)

    def handle(self, log: SpellExecutionLog) -> None:
        level = stdlib_logging.INFO if log.success else stdlib_logging.ERROR
        self.logger.log(level, log.to_json())

    def flush(self) -> None:
        for handler in self.logger.handlers:
            handler.flush()


class JSONFileHandler:
    """Handler that writes JSON logs to a file."""

    def __init__(self, path: str | Path, buffer_size: int = 100):
        self.path = Path(path)
        self.buffer_size = buffer_size
        self.buffer: list[SpellExecutionLog] = []

    def handle(self, log: SpellExecutionLog) -> None:
        self.buffer.append(log)
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            for log in self.buffer:
                f.write(log.to_json() + "\n")
        self.buffer.clear()


class OpenTelemetryHandler:
    """Handler that exports to OpenTelemetry.

    Requires opentelemetry-api and opentelemetry-sdk packages.
    Gracefully degrades if packages not installed.
    """

    def __init__(self, service_name: str = "magically"):
        self._tracer = None
        self._meter = None
        self._available = False
        self._service_name = service_name
        self._initialize()

    def _initialize(self) -> None:
        try:
            from opentelemetry import metrics, trace
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.trace import TracerProvider

            # Use existing providers if already configured, otherwise create new ones
            tracer_provider = trace.get_tracer_provider()
            if not isinstance(tracer_provider, TracerProvider):
                tracer_provider = TracerProvider()
                trace.set_tracer_provider(tracer_provider)

            meter_provider = metrics.get_meter_provider()
            if not isinstance(meter_provider, MeterProvider):
                meter_provider = MeterProvider()
                metrics.set_meter_provider(meter_provider)

            self._tracer = trace.get_tracer(self._service_name)
            self._meter = metrics.get_meter(self._service_name)
            self._available = True
        except ImportError:
            pass

    def handle(self, log: SpellExecutionLog) -> None:
        if not self._available or self._tracer is None:
            return

        try:
            from opentelemetry import trace
            from opentelemetry.trace import StatusCode

            with self._tracer.start_as_current_span(
                log.spell_name,
                attributes={
                    "spell.id": log.spell_id,
                    "spell.model": log.model,
                    "spell.model_alias": log.model_alias or "",
                    "spell.duration_ms": log.duration_ms or 0,
                    "spell.success": log.success,
                    "spell.tokens.input": log.token_usage.input_tokens,
                    "spell.tokens.output": log.token_usage.output_tokens,
                    "spell.tokens.total": log.token_usage.total_tokens,
                },
            ) as span:
                if not log.success and log.error:
                    span.set_status(StatusCode.ERROR, log.error)
                if log.cost_estimate:
                    span.set_attribute("spell.cost.total", log.cost_estimate.total_cost)
                for key, value in log.tags.items():
                    span.set_attribute(f"spell.tag.{key}", value)
        except Exception:
            pass  # Never let logging break execution

    def flush(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------


@dataclass
class LoggingConfig:
    """Configuration for magically logging."""

    enabled: bool = False
    level: LogLevel = LogLevel.INFO
    handlers: list[LogHandler] = field(default_factory=list)
    redact_content: bool = False
    include_input: bool = True
    include_output: bool = True
    cost_tracking: bool = True
    default_tags: dict[str, str] = field(default_factory=dict)


_logging_config: ContextVar[LoggingConfig | None] = ContextVar("logging_config", default=None)
_process_logging_config: LoggingConfig | None = None
_file_logging_config_cache: LoggingConfig | None = None


def configure_logging(config: LoggingConfig) -> None:
    """Set the logging configuration for the current process."""
    global _process_logging_config
    _process_logging_config = config


def _load_logging_config_from_file() -> LoggingConfig | None:
    """Load logging configuration from pyproject.toml."""
    import tomllib

    # Find pyproject.toml
    cwd = Path.cwd()
    pyproject_path = None
    for parent in [cwd, *cwd.parents]:
        candidate = parent / "pyproject.toml"
        if candidate.exists():
            pyproject_path = candidate
            break

    if pyproject_path is None:
        return None

    try:
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
    except Exception:
        return None

    logging_config = data.get("tool", {}).get("magically", {}).get("logging", {})
    if not logging_config:
        return None

    # Parse handlers
    handlers: list[LogHandler] = []
    handlers_config = logging_config.get("handlers", {})

    for handler_name, handler_settings in handlers_config.items():
        if not isinstance(handler_settings, dict):
            continue

        handler_type = handler_settings.get("type", handler_name)

        if handler_type == "python":
            logger_name = handler_settings.get("logger_name", "magically")
            handlers.append(PythonLoggingHandler(logger_name))
        elif handler_type == "json_file":
            path = handler_settings.get("path")
            if path:
                handlers.append(JSONFileHandler(path))
        elif handler_type == "otel" or handler_type == "opentelemetry":
            handlers.append(OpenTelemetryHandler())

    # If no handlers specified but logging enabled, add default Python handler
    enabled = logging_config.get("enabled", False)
    if enabled and not handlers:
        handlers.append(PythonLoggingHandler())

    # Parse level
    level_str = logging_config.get("level", "info").lower()
    level_map = {
        "debug": LogLevel.DEBUG,
        "info": LogLevel.INFO,
        "warning": LogLevel.WARNING,
        "error": LogLevel.ERROR,
    }
    level = level_map.get(level_str, LogLevel.INFO)

    # Parse default tags
    default_tags = logging_config.get("default_tags", {})
    if not isinstance(default_tags, dict):
        default_tags = {}

    return LoggingConfig(
        enabled=enabled,
        level=level,
        handlers=handlers,
        redact_content=logging_config.get("redact_content", False),
        include_input=logging_config.get("include_input", True),
        include_output=logging_config.get("include_output", True),
        cost_tracking=logging_config.get("cost_tracking", True),
        default_tags=default_tags,
    )


def get_logging_config() -> LoggingConfig:
    """Get the current logging configuration.

    Resolution order:
    1. Active context (from `logging_context()`)
    2. Process config (from `configure_logging()`)
    3. File config (from pyproject.toml, cached)
    4. Default disabled config
    """
    global _file_logging_config_cache

    ctx_config = _logging_config.get()
    if ctx_config is not None:
        return ctx_config
    if _process_logging_config is not None:
        return _process_logging_config

    # Try file config (cached)
    if _file_logging_config_cache is None:
        _file_logging_config_cache = _load_logging_config_from_file()
    if _file_logging_config_cache is not None:
        return _file_logging_config_cache

    return LoggingConfig()  # Disabled by default


@contextmanager
def logging_context(config: LoggingConfig):
    """Context manager for temporary logging configuration."""
    token = _logging_config.set(config)
    try:
        yield
    finally:
        _logging_config.reset(token)


# ---------------------------------------------------------------------------
# Cost Estimation
# ---------------------------------------------------------------------------

# Pricing per 1M tokens (as of 2024-01)
PRICING: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    # Google
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
}


def estimate_cost(model: str, usage: TokenUsage) -> CostEstimate | None:
    """Estimate the cost of a spell execution based on token usage.

    Args:
        model: The model identifier (with or without provider prefix)
        usage: Token usage metrics

    Returns:
        CostEstimate if pricing is available, None otherwise
    """
    # Strip provider prefix if present
    model_name = model.split(":")[-1] if ":" in model else model

    pricing = PRICING.get(model_name)
    if pricing is None:
        return None

    input_cost = (usage.input_tokens / 1_000_000) * pricing["input"]
    output_cost = (usage.output_tokens / 1_000_000) * pricing["output"]

    return CostEstimate(
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=input_cost + output_cost,
        currency="USD",
        model=model_name,
    )


# ---------------------------------------------------------------------------
# Observability Helpers
# ---------------------------------------------------------------------------


def setup_logging(
    level: LogLevel = LogLevel.INFO,
    *,
    otel: bool = False,
    redact_content: bool = False,
    json_file: str | Path | None = None,
) -> None:
    """Quick setup for logging.

    Args:
        level: Minimum log level
        otel: Enable OpenTelemetry export
        redact_content: Redact input/output content
        json_file: Path to JSON log file (optional)
    """
    handlers: list[LogHandler] = [PythonLoggingHandler()]

    if json_file:
        handlers.append(JSONFileHandler(json_file))

    if otel:
        handlers.append(OpenTelemetryHandler())

    configure_logging(
        LoggingConfig(
            enabled=True,
            level=level,
            handlers=handlers,
            redact_content=redact_content,
            include_input=not redact_content,
            include_output=not redact_content,
            cost_tracking=True,
        )
    )


def setup_logfire(*, redact_content: bool = False) -> None:
    """Setup logging with Logfire (OpenTelemetry-based).

    Note: Requires logfire package to be installed and configured.
    """
    setup_logging(otel=True, redact_content=redact_content)


def setup_datadog(*, redact_content: bool = False) -> None:
    """Setup logging with Datadog (OpenTelemetry-based).

    Note: Requires ddtrace package to be installed and configured.
    """
    setup_logging(otel=True, redact_content=redact_content)


# ---------------------------------------------------------------------------
# Log Emission
# ---------------------------------------------------------------------------


def _emit_log(log: SpellExecutionLog) -> None:
    """Emit a log to all configured handlers."""
    config = get_logging_config()
    if not config.enabled:
        return

    # Apply redaction if configured
    if config.redact_content:
        log.input_args = "[REDACTED]"
        log.output = "[REDACTED]"

    # Apply default tags
    log.tags = {**config.default_tags, **log.tags}

    # Emit to all handlers
    for handler in config.handlers:
        try:
            handler.handle(log)
        except Exception:
            pass  # Never let logging break execution
