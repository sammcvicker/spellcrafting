from magically.config import Config, MagicallyConfigError, ModelConfig, current_config
from magically.guard import (
    guard,
    GuardError,
    InputGuard,
    OutputGuard,
    GuardContext,
)
from magically.on_fail import OnFail
from magically.validator import llm_validator, ValidationResult
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
    ValidationMetrics,
)
from magically.spell import spell

__all__ = [
    # Core
    "spell",
    "Config",
    "ModelConfig",
    "MagicallyConfigError",
    "current_config",
    # Guards
    "guard",
    "GuardError",
    "OnFail",
    "InputGuard",
    "OutputGuard",
    "GuardContext",
    # Validators
    "llm_validator",
    "ValidationResult",
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
    "ValidationMetrics",
]
