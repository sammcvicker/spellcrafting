"""Spellcrafting - LLMs as a Python language feature.

This module provides @spell and @guard decorators to turn functions into
LLM-powered operations with input/output validation.

Import Guidelines:
    All public API is exported at the top level:
        from spellcrafting import spell, guard, Config, OnFail, ...

    For type hints, import directly from the module:
        from spellcrafting import SpellResult, ValidationResult

    Individual modules are NOT exported at package level.
    Use explicit imports if needed:
        from spellcrafting.logging import SpellExecutionLog
"""

from importlib.metadata import version, PackageNotFoundError

from spellcrafting.config import (
    Config,
    ModelConfig,
    ModelConfigDict,
    RateLimitConfig,
    current_config,
    clear_config_cache,
    configure_rate_limits,
    get_rate_limit_config,
)
from spellcrafting.exceptions import (
    SpellcraftingError,
    SpellcraftingConfigError,
    GuardError,
    ValidationError,
)
from spellcrafting.guard import (
    guard,
    GuardConfig,
    GuardContext,
    get_guard_config,
    InputGuard,
    OutputGuard,
)
from spellcrafting.on_fail import (
    OnFail,
    OnFailStrategy,
    ValidatorOnFailStrategy,
    RaiseStrategy,
    FixStrategy,
    RetryStrategy,
    EscalateStrategy,
    FallbackStrategy,
    CustomStrategy,
)
from spellcrafting.validator import llm_validator, ValidationResult
from spellcrafting.logging import (
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
from spellcrafting.spell import (
    spell,
    clear_agent_cache,
    get_cache_stats,
    set_cache_max_size,
    CacheStats,
)
from spellcrafting.result import SpellResult, SyncSpell, AsyncSpell

# Package metadata
try:
    __version__ = version("spellcrafting")
except PackageNotFoundError:
    __version__ = "0.1.0"  # Fallback for development/editable installs

# Explicit public API - all items importable via `from spellcrafting import X`
# NOTE: Individual modules (logging, spell, config, etc.) are NOT exported.
# Use direct imports: `from spellcrafting.logging import SpellExecutionLog`
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
    "ModelConfigDict",
    "current_config",
    "clear_config_cache",
    # Rate limiting
    "RateLimitConfig",
    "configure_rate_limits",
    "get_rate_limit_config",
    # Exceptions (all inherit from SpellcraftingError)
    "SpellcraftingError",
    "SpellcraftingConfigError",
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
    "OnFailStrategy",
    "ValidatorOnFailStrategy",
    "RaiseStrategy",
    "FixStrategy",
    "RetryStrategy",
    "EscalateStrategy",
    "FallbackStrategy",
    "CustomStrategy",
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
