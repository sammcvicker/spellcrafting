from magically.config import Config, ModelConfig, current_config
from magically.exceptions import (
    MagicallyError,
    MagicallyConfigError,
    GuardError,
    ValidationError,
)
from magically.guard import (
    guard,
    GuardConfig,
    get_guard_config,
    InputGuard,
    OutputGuard,
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
    AsyncLogHandler,
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
from magically.spell import (
    spell,
    clear_agent_cache,
    get_cache_stats,
    set_cache_max_size,
    CacheStats,
)
from magically.result import SpellResult, SyncSpell, AsyncSpell

__all__ = [
    # Core
    "spell",
    "SpellResult",
    "SyncSpell",
    "AsyncSpell",
    # Cache management
    "clear_agent_cache",
    "get_cache_stats",
    "set_cache_max_size",
    "CacheStats",
    # Config
    "Config",
    "ModelConfig",
    "current_config",
    # Exceptions (all inherit from MagicallyError)
    "MagicallyError",
    "MagicallyConfigError",
    "GuardError",
    "ValidationError",
    # Guards
    "guard",
    "GuardConfig",
    "get_guard_config",
    "OnFail",
    "InputGuard",
    "OutputGuard",
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
    "AsyncLogHandler",
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
