"""Structured logging, distributed tracing, and cost tracking for spellcrafting.

This module provides enterprise-grade observability with zero overhead when disabled.

Design Note: Pydantic vs Dataclass Usage
----------------------------------------
This module uses dataclasses (not Pydantic) for all types because:
- These are internal telemetry types, not user-provided configuration
- No external parsing or validation is needed - we create them directly
- Dataclasses have lower overhead (no validation on every instantiation)
- Simple to_dict() methods are sufficient for serialization

See config.py for contrast - it uses Pydantic for parsing pyproject.toml
where validation and error messages for user config are important.
"""

from __future__ import annotations

import json
import logging as stdlib_logging
import sys
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from collections.abc import Awaitable, Generator
from typing import Any, Literal, Protocol, TypedDict
from uuid import uuid4

# Python 3.11+ has tomllib in stdlib; use tomli for 3.10
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from spellcrafting._pyproject import find_pyproject

# Module-level logger for debug messages about internal operations
_logger = stdlib_logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# W3C Trace Context defines span_id as 16 hex chars
SPAN_ID_LENGTH = 16

# Default buffer size for JSONFileHandler before flushing to disk
DEFAULT_BUFFER_SIZE = 100

# Tokens per million for cost calculations (standard pricing unit)
TOKENS_PER_MILLION = 1_000_000


class LogLevel(str, Enum):
    """Log levels for spellcrafting logging."""

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


# Redacted content marker type
REDACTED = Literal["[REDACTED]"]


@dataclass
class ToolCallLog:
    """Log of a single tool invocation during spell execution.

    When content is redacted (via LoggingConfig.redact_content), the
    `arguments` and `result` fields will contain the literal string
    "[REDACTED]" instead of their actual values.
    """

    tool_name: str
    arguments: dict[str, Any] | REDACTED = field(default_factory=dict)
    result: Any = None
    duration_ms: float = 0.0
    success: bool = True
    error: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    redacted: bool = False  # Explicit flag indicating if content was redacted

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
            "redacted": self.redacted,
        }


@dataclass
class ValidationMetrics:
    """Metrics about validation during spell execution.

    Tracks retry attempts, guard results, Pydantic validation errors,
    and on_fail strategy usage for observability and debugging.
    """

    # Retry tracking
    attempt_count: int = 1  # Total attempts (1 = no retries)
    retry_reasons: list[str] = field(default_factory=list)

    # Guard results (if @guard decorators used)
    input_guards_passed: list[str] = field(default_factory=list)
    input_guards_failed: list[str] = field(default_factory=list)
    output_guards_passed: list[str] = field(default_factory=list)
    output_guards_failed: list[str] = field(default_factory=list)

    # Pydantic validation errors (errors before eventual success or final failure)
    pydantic_errors: list[str] = field(default_factory=list)

    # On-fail strategy tracking
    on_fail_triggered: str | None = None  # "escalate", "fallback", "custom", "retry"
    escalated_to_model: str | None = None  # Model used for escalation

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "attempt_count": self.attempt_count,
            "retry_reasons": self.retry_reasons,
            "input_guards_passed": self.input_guards_passed,
            "input_guards_failed": self.input_guards_failed,
            "output_guards_passed": self.output_guards_passed,
            "output_guards_failed": self.output_guards_failed,
            "pydantic_errors": self.pydantic_errors,
            "on_fail_triggered": self.on_fail_triggered,
            "escalated_to_model": self.escalated_to_model,
        }


@dataclass
class SpellExecutionLog:
    """Complete log of a spell execution.

    This is a mutable dataclass that tracks execution state. The finalize()
    method should be called exactly once when execution completes.

    Note: While this uses a dataclass for convenience, it is intentionally
    mutable. The _finalized flag prevents accidental double-finalization
    which would corrupt timing data.
    """

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

    # Validation metrics (guards, retries, on_fail)
    validation: ValidationMetrics | None = None

    # Internal state tracking
    _finalized: bool = field(default=False, init=False, repr=False)

    def finalize(
        self,
        *,
        success: bool,
        error: Exception | None = None,
        output: Any = None,
    ) -> None:
        """Finalize the log with completion status.

        This method should be called exactly once when spell execution completes.
        Calling it multiple times raises RuntimeError to prevent corrupted timing data.

        Args:
            success: Whether the spell execution succeeded
            error: The exception if execution failed (optional)
            output: The output value if execution succeeded (optional)

        Raises:
            RuntimeError: If finalize() has already been called on this log
        """
        if self._finalized:
            raise RuntimeError(
                f"SpellExecutionLog for '{self.spell_name}' already finalized. "
                "finalize() should only be called once per execution."
            )
        self._finalized = True
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
            "validation": self.validation.to_dict() if self.validation else None,
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
            span_id=uuid4().hex[:SPAN_ID_LENGTH],
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
def trace_context(
    parent: TraceContext | None = None,
) -> Generator[TraceContext, None, None]:
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
def with_trace_id(trace_id: str) -> Generator[TraceContext, None, None]:
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
    """Protocol for synchronous log handlers."""

    def handle(self, log: SpellExecutionLog) -> None:
        """Handle a spell execution log."""
        ...

    def flush(self) -> None:
        """Flush any buffered logs."""
        ...


class AsyncLogHandler(Protocol):
    """Protocol for asynchronous log handlers.

    Use this protocol when implementing handlers that need async I/O,
    such as async HTTP transports or async file operations.

    Example:
        class AsyncHTTPHandler:
            async def handle(self, log: SpellExecutionLog) -> None:
                async with aiohttp.ClientSession() as session:
                    await session.post(self.endpoint, json=log.to_dict())

            async def flush(self) -> None:
                pass
    """

    def handle(self, log: SpellExecutionLog) -> Awaitable[None]:
        """Handle a spell execution log asynchronously."""
        ...

    def flush(self) -> Awaitable[None]:
        """Flush any buffered logs asynchronously."""
        ...


class PythonLoggingHandler:
    """Handler that emits logs to Python's stdlib logging."""

    def __init__(self, logger_name: str = "spellcrafting"):
        self.logger = stdlib_logging.getLogger(logger_name)

    def handle(self, log: SpellExecutionLog) -> None:
        level = stdlib_logging.INFO if log.success else stdlib_logging.ERROR
        self.logger.log(level, log.to_json())

    def flush(self) -> None:
        for handler in self.logger.handlers:
            handler.flush()


class JSONFileHandler:
    """Handler that writes JSON logs to a file."""

    def __init__(self, path: str | Path, buffer_size: int = DEFAULT_BUFFER_SIZE):
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
    Install with: pip install spellcrafting[otel]
    Gracefully degrades if packages not installed.
    """

    def __init__(self, service_name: str = "spellcrafting"):
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
        except Exception as e:
            # Intentionally broad: OTEL failures should never break spell execution
            # Log at debug level for troubleshooting
            _logger.debug("OpenTelemetry export failed for spell '%s': %s", log.spell_name, e)

    def flush(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------


@dataclass
class LoggingConfig:
    """Configuration for spellcrafting logging."""

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

# Context variable for capturing logs instead of emitting them (used by with_metadata)
_capture_log: ContextVar[list[SpellExecutionLog] | None] = ContextVar("capture_log", default=None)

# Track handlers that have already warned about failures (issue #182)
# This prevents spamming warnings for handlers that consistently fail
_handler_failure_warned: set[int] = set()


def configure_logging(config: LoggingConfig) -> None:
    """Set the logging configuration for the current process.

    This sets the process-level default configuration. Use logging_context()
    for scoped configuration that applies only within a context manager.

    Args:
        config: LoggingConfig instance with desired settings.

    Note:
        Process-level config has lower priority than context config set via
        logging_context(). It has higher priority than file config from
        pyproject.toml.

    Example:
        config = LoggingConfig(
            enabled=True,
            level=LogLevel.INFO,
            handlers=[PythonLoggingHandler()],
        )
        configure_logging(config)
    """
    global _process_logging_config
    _process_logging_config = config


# ---------------------------------------------------------------------------
# Handler Factory for Config Parsing
# ---------------------------------------------------------------------------


def _create_handler(handler_type: str, config: dict[str, Any]) -> LogHandler | None:
    """Create a log handler from configuration.

    Args:
        handler_type: The type of handler to create.
        config: Handler configuration dict.

    Returns:
        LogHandler instance or None if handler cannot be created.
    """
    if handler_type == "python":
        logger_name = config.get("logger_name", "spellcrafting")
        return PythonLoggingHandler(logger_name)

    if handler_type == "json_file":
        path = config.get("path")
        if not path:
            return None
        return JSONFileHandler(path)

    if handler_type in ("otel", "opentelemetry"):
        return OpenTelemetryHandler()

    # Unknown handler type
    return None


def _parse_handlers(handlers_config: dict[str, Any]) -> list[LogHandler]:
    """Parse handler configurations into handler instances.

    Args:
        handlers_config: Dict mapping handler names to their configurations.

    Returns:
        List of successfully created LogHandler instances.
    """
    handlers: list[LogHandler] = []

    for handler_name, handler_settings in handlers_config.items():
        if not isinstance(handler_settings, dict):
            continue

        handler_type = handler_settings.get("type", handler_name)
        handler = _create_handler(handler_type, handler_settings)
        if handler is not None:
            handlers.append(handler)

    return handlers


def _parse_log_level(level_str: str) -> LogLevel:
    """Parse a log level string into a LogLevel enum.

    Args:
        level_str: Case-insensitive level string ("debug", "info", etc.)

    Returns:
        Corresponding LogLevel, defaults to INFO if unknown.
    """
    level_map = {
        "debug": LogLevel.DEBUG,
        "info": LogLevel.INFO,
        "warning": LogLevel.WARNING,
        "error": LogLevel.ERROR,
    }
    return level_map.get(level_str.lower(), LogLevel.INFO)


def _load_logging_config_from_file() -> LoggingConfig | None:
    """Load logging configuration from pyproject.toml.

    Returns:
        LoggingConfig if valid configuration found, None otherwise.
    """
    pyproject_path = find_pyproject()
    if pyproject_path is None:
        return None

    try:
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return None

    logging_config = data.get("tool", {}).get("spellcrafting", {}).get("logging", {})
    if not logging_config:
        return None

    enabled = logging_config.get("enabled", False)
    handlers = _parse_handlers(logging_config.get("handlers", {}))

    # Add default Python handler if logging enabled but no handlers specified
    if enabled and not handlers:
        handlers.append(PythonLoggingHandler())

    level = _parse_log_level(logging_config.get("level", "info"))

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
def logging_context(config: LoggingConfig) -> Generator[None, None, None]:
    """Context manager for scoped logging configuration.

    Temporarily overrides the logging configuration for the duration of the
    context. This has the highest priority in config resolution, overriding
    both process-level config and file config.

    Args:
        config: LoggingConfig instance to use within this context.

    Yields:
        None

    Example:
        # Temporarily enable verbose logging for debugging
        debug_config = LoggingConfig(
            enabled=True,
            level=LogLevel.DEBUG,
            handlers=[PythonLoggingHandler()],
        )

        with logging_context(debug_config):
            result = my_spell("test input")  # Uses debug logging
        # After context, reverts to previous config
    """
    token = _logging_config.set(config)
    try:
        yield
    finally:
        _logging_config.reset(token)


# ---------------------------------------------------------------------------
# Cost Estimation
# ---------------------------------------------------------------------------


class ModelPricing(TypedDict):
    """Pricing for a model per 1M tokens.

    Attributes:
        input: Cost per 1M input tokens in USD
        output: Cost per 1M output tokens in USD
    """

    input: float
    output: float


# Default pricing per 1M tokens (as of 2025-01)
# NOTE: Prices are estimates and may change. Check provider pricing pages for current rates.
_DEFAULT_PRICING: dict[str, ModelPricing] = {
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
    "o1": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 3.00, "output": 12.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    # Google
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-exp": {"input": 0.0, "output": 0.0},  # Free during experimental period
}

# User-registered custom pricing (merged with defaults)
_custom_pricing: dict[str, ModelPricing] = {}

# Cache for pricing loaded from pyproject.toml
_file_pricing_cache: dict[str, ModelPricing] | None = None


def _load_pricing_from_file() -> dict[str, ModelPricing]:
    """Load custom pricing configuration from pyproject.toml.

    Example pyproject.toml configuration:
        [tool.spellcrafting.pricing."my-custom-model"]
        input = 1.0
        output = 2.0
    """
    global _file_pricing_cache

    if _file_pricing_cache is not None:
        return _file_pricing_cache

    _file_pricing_cache = {}
    pyproject_path = find_pyproject()
    if pyproject_path is None:
        return _file_pricing_cache

    try:
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return _file_pricing_cache

    pricing_config = data.get("tool", {}).get("spellcrafting", {}).get("pricing", {})
    if not pricing_config or not isinstance(pricing_config, dict):
        return _file_pricing_cache

    for model_name, prices in pricing_config.items():
        if not isinstance(prices, dict):
            continue
        input_cost = prices.get("input")
        output_cost = prices.get("output")
        if isinstance(input_cost, (int, float)) and isinstance(output_cost, (int, float)):
            _file_pricing_cache[model_name] = {
                "input": float(input_cost),
                "output": float(output_cost),
            }

    return _file_pricing_cache


def register_model_pricing(
    model: str,
    input_cost: float,
    output_cost: float,
) -> None:
    """Register pricing for a custom or new model.

    This allows users to add pricing for custom models (e.g., fine-tuned models)
    or update pricing for existing models without modifying the source code.

    Args:
        model: The model identifier (e.g., "my-custom-model" or "gpt-4o")
        input_cost: Cost per 1M input tokens in USD
        output_cost: Cost per 1M output tokens in USD

    Example:
        >>> register_model_pricing("my-fine-tuned-gpt", 5.0, 15.0)
        >>> register_model_pricing("gpt-4o", 3.0, 12.0)  # Override default pricing
    """
    _custom_pricing[model] = {"input": input_cost, "output": output_cost}


def get_model_pricing(model: str) -> ModelPricing | None:
    """Get pricing for a model.

    Resolution order:
    1. User-registered custom pricing (via register_model_pricing)
    2. File-based pricing (from pyproject.toml)
    3. Default built-in pricing

    Args:
        model: The model identifier (with or without provider prefix).
               Provider prefixes like "anthropic:" are stripped.

    Returns:
        ModelPricing dict with input and output costs per 1M tokens,
        or None if no pricing is available for the model.

    Example:
        >>> pricing = get_model_pricing("gpt-4o")
        >>> if pricing:
        ...     print(f"Input: ${pricing['input']}/1M tokens")
    """
    # Strip provider prefix if present
    model_name = model.split(":")[-1] if ":" in model else model

    # Check custom pricing first (user-registered)
    if model_name in _custom_pricing:
        return _custom_pricing[model_name]

    # Check file-based pricing (pyproject.toml)
    file_pricing = _load_pricing_from_file()
    if model_name in file_pricing:
        return file_pricing[model_name]

    # Fall back to default pricing
    return _DEFAULT_PRICING.get(model_name)


# Backwards compatibility: PRICING dict now delegates to get_model_pricing
# This is kept for users who import PRICING directly
PRICING = _DEFAULT_PRICING


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

    pricing = get_model_pricing(model_name)
    if pricing is None:
        return None

    input_cost = (usage.input_tokens / TOKENS_PER_MILLION) * pricing["input"]
    output_cost = (usage.output_tokens / TOKENS_PER_MILLION) * pricing["output"]

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
    """Setup logging for Logfire (OpenTelemetry-based).

    This is a convenience alias for setup_logging(otel=True).

    Both setup_logfire() and setup_datadog() have identical implementations
    because they both use OpenTelemetry for export. The separate functions
    exist for:
    1. Discoverability - users searching for their provider find the right function
    2. Future extensibility - provider-specific options can be added later

    Prerequisites:
        pip install spellcrafting[logfire]
        Configure via LOGFIRE_TOKEN environment variable or logfire.configure()

    Args:
        redact_content: If True, redact input/output content from logs

    Example:
        import logfire
        from spellcrafting import setup_logfire

        logfire.configure()  # Uses LOGFIRE_TOKEN env var
        setup_logfire()
    """
    setup_logging(otel=True, redact_content=redact_content)


def setup_datadog(*, redact_content: bool = False) -> None:
    """Setup logging for Datadog (OpenTelemetry-based).

    This is a convenience alias for setup_logging(otel=True).

    Both setup_logfire() and setup_datadog() have identical implementations
    because they both use OpenTelemetry for export. The separate functions
    exist for:
    1. Discoverability - users searching for their provider find the right function
    2. Future extensibility - provider-specific options can be added later

    Prerequisites:
        pip install spellcrafting[datadog]  # installs ddtrace
        Configure via DD_* environment variables

    Args:
        redact_content: If True, redact input/output content from logs

    Example:
        from spellcrafting import setup_datadog

        # Ensure DD_AGENT_HOST, DD_TRACE_AGENT_PORT are set
        setup_datadog()
    """
    setup_logging(otel=True, redact_content=redact_content)


# ---------------------------------------------------------------------------
# Log Emission
# ---------------------------------------------------------------------------


def _emit_log(log: SpellExecutionLog) -> None:
    """Emit a log to all configured handlers.

    This is an internal function called by spell execution to emit logs.
    It applies redaction if configured and adds default tags.

    Args:
        log: The SpellExecutionLog to emit.

    Note:
        This function is designed to never raise exceptions - handler failures
        are logged at debug level but do not interrupt spell execution.
    """
    # Check if we're in capture mode (used by with_metadata)
    capture_list = _capture_log.get()
    if capture_list is not None:
        capture_list.append(log)
        return

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
        except Exception as e:
            # Intentionally broad: handler failures should never break spell execution
            # Warn once per handler, then log at debug level for troubleshooting
            handler_name = type(handler).__name__
            handler_id = id(handler)
            if handler_id not in _handler_failure_warned:
                _handler_failure_warned.add(handler_id)
                warnings.warn(
                    f"Log handler {handler_name} failed: {e}. "
                    f"Further errors from this handler will be suppressed.",
                    stacklevel=2,
                )
            _logger.debug("Log handler %s failed: %s", handler_name, e)


@dataclass
class CapturedLog:
    """Container for captured execution logs.

    This is returned by capture_execution_log() and provides access to
    the captured log after spell execution completes.
    """

    logs: list[SpellExecutionLog] = field(default_factory=list)

    @property
    def log(self) -> SpellExecutionLog | None:
        """Get the first (and typically only) captured log."""
        return self.logs[0] if self.logs else None


@contextmanager
def capture_execution_log() -> Generator[CapturedLog, None, None]:
    """Context manager to capture execution logs without emitting them.

    This is used by with_metadata() to reuse the core execution path
    while capturing the metadata for SpellResult construction.

    Yields:
        CapturedLog container that will hold the captured log(s) after
        spell execution completes.

    Example:
        with capture_execution_log() as captured:
            output = my_spell("input")

        if captured.log:
            result = SpellResult.from_execution_log(output, captured.log)
    """
    captured = CapturedLog()
    token = _capture_log.set(captured.logs)
    try:
        yield captured
    finally:
        _capture_log.reset(token)


def is_capture_mode() -> bool:
    """Check if we're in capture mode for execution logs.

    This is used by spell execution to determine if the logging path
    should be taken even when logging is disabled.

    Returns:
        True if capture_execution_log() context is active.
    """
    return _capture_log.get() is not None
