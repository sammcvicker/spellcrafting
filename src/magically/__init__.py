"""Magically - LLMs as a Python language feature.

This module provides @spell and @guard decorators to turn functions into
LLM-powered operations with input/output validation.

Import Guidelines:
    All public API is exported at the top level:
        from magically import spell, guard, Config, OnFail, ...

    For type hints, import directly from the module:
        from magically import SpellResult, ValidationResult

    Individual modules are NOT exported at package level.
    Use explicit imports if needed:
        from magically.logging import SpellExecutionLog
"""

from importlib.metadata import version, PackageNotFoundError

from magically.config import Config, ModelConfig, current_config, clear_config_cache
from magically.exceptions import (
    MagicallyError,
    MagicallyConfigError,
    GuardError,
    ValidationError,
)
from magically.guard import (
    guard,
    GuardConfig,
    GuardContext,
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
    # Pricing
    ModelPricing,
    register_model_pricing,
    get_model_pricing,
)
from magically.spell import (
    spell,
    clear_agent_cache,
    get_cache_stats,
    set_cache_max_size,
    CacheStats,
)
from magically.result import SpellResult, SyncSpell, AsyncSpell

# Package metadata
try:
    __version__ = version("magically")
except PackageNotFoundError:
    __version__ = "0.1.0"  # Fallback for development/editable installs

# Explicit public API - all items importable via `from magically import X`
# NOTE: Individual modules (logging, spell, config, etc.) are NOT exported.
# Use direct imports: `from magically.logging import SpellExecutionLog`
__all__ = [
    # Package metadata
    "__version__",
    # Core decorator and result type
    "spell",
    "SpellResult",
    "SyncSpell",
    "AsyncSpell",
    # Cache management (for testing/memory management)
    "clear_agent_cache",
    "get_cache_stats",
    "set_cache_max_size",
    "CacheStats",
    # Configuration
    "Config",
    "ModelConfig",
    "current_config",
    "clear_config_cache",
    # Exceptions (all inherit from MagicallyError)
    "MagicallyError",
    "MagicallyConfigError",
    "GuardError",
    "ValidationError",
    # Guards - validation decorators
    "guard",
    "GuardConfig",
    "GuardContext",
    "get_guard_config",
    "InputGuard",
    "OutputGuard",
    # Failure strategies
    "OnFail",
    # LLM-powered validators
    "llm_validator",
    "ValidationResult",
    # Logging configuration
    "LoggingConfig",
    "LogLevel",
    "configure_logging",
    "get_logging_config",
    # Quick setup helpers (aliases for setup_logging)
    "setup_logging",
    "setup_logfire",
    "setup_datadog",
    # Distributed tracing
    "TraceContext",
    "trace_context",
    "current_trace",
    "with_trace_id",
    # Log handlers (for custom integrations)
    "LogHandler",
    "AsyncLogHandler",
    "PythonLoggingHandler",
    "JSONFileHandler",
    "OpenTelemetryHandler",
    # Telemetry types (for custom handlers)
    "SpellExecutionLog",
    "TokenUsage",
    "CostEstimate",
    "ToolCallLog",
    "ValidationMetrics",
    # Cost tracking
    "ModelPricing",
    "register_model_pricing",
    "get_model_pricing",
]
