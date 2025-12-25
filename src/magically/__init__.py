from magically.config import Config, MagicallyConfigError, ModelConfig, current_config
from magically.logging import (
    # Configuration
    LoggingConfig,
    LogLevel,
    configure_logging,
    get_logging_config,
    # Quick setup
    setup_logging,
    setup_logfire,
    setup_datadog,
    # Tracing
    TraceContext,
    trace_context,
    current_trace,
    with_trace_id,
    # Handlers
    LogHandler,
    PythonLoggingHandler,
    JSONFileHandler,
    OpenTelemetryHandler,
    # Types (for custom handlers)
    SpellExecutionLog,
    TokenUsage,
    CostEstimate,
    ToolCallLog,
)
from magically.spell import spell

__all__ = [
    # Core
    "spell",
    "Config",
    "ModelConfig",
    "MagicallyConfigError",
    "current_config",
    # Logging Configuration
    "LoggingConfig",
    "LogLevel",
    "configure_logging",
    "get_logging_config",
    # Quick setup
    "setup_logging",
    "setup_logfire",
    "setup_datadog",
    # Tracing
    "TraceContext",
    "trace_context",
    "current_trace",
    "with_trace_id",
    # Handlers
    "LogHandler",
    "PythonLoggingHandler",
    "JSONFileHandler",
    "OpenTelemetryHandler",
    # Types
    "SpellExecutionLog",
    "TokenUsage",
    "CostEstimate",
    "ToolCallLog",
]
